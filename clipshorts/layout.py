"""Pure layout classification (no OpenCV, no I/O).

Input: per-frame face detections (normalized boxes) plus a coarse frame-difference grid.
Output: fullcam | stacked | face | content with a confidence, rects and notes.

All thresholds are named module constants feeding `LayoutParams`, so they can be tuned in one
place or overridden through Config.layout_overrides. The values are PLAN starting points and
have NOT been tuned on real fixture clips yet.
"""

import statistics
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

FULLCAM = "fullcam"
STACKED = "stacked"
FACE = "face"
CONTENT = "content"
LAYOUT_TYPES = (FULLCAM, STACKED, FACE, CONTENT)

# -- tunables (PLAN section 4) ---------------------------------------------------
FACE_FRAMES_MIN_FRAC = 6 / 8  # face in at least this fraction of frames -> face-based layout
NO_FACE_MAX_FRAC = 3 / 8  # face in fewer than this fraction -> content
FULLCAM_MIN_AREA = 0.12  # median face area (fraction of frame) at/above -> fullcam
STACKED_MAX_AREA = 0.08  # below this, a stable corner box is a clean "stacked" signal
CORNER_MARGIN = 0.35  # face center within this distance of left/right AND top/bottom edge
STABLE_IOU = 0.5  # median IoU of per-frame boxes vs the median box
FOCUS_MAX_MOTION = 0.02  # content-region mean abs frame diff (0..1) below this = face-focus
FACECAM_MARGIN_X = 0.75  # face box expansion per side, fraction of face width
FACECAM_MARGIN_Y = 0.5  # ... fraction of face height
EDGE_SNAP = 0.05  # expanded facecam edge this close to the frame border snaps to it
CONF_W_AGREEMENT = 0.6
CONF_W_STABILITY = 0.4
CONF_FALLBACK_FACTOR = 0.7  # applied when no rule matched cleanly (gap zones, odd faces)
OUT_ASPECT = 9 / 16

SAMPLE_FRAMES = 8
SAMPLE_START = 0.05
SAMPLE_END = 0.95


@dataclass(frozen=True)
class LayoutParams:
    face_frames_min_frac: float = FACE_FRAMES_MIN_FRAC
    no_face_max_frac: float = NO_FACE_MAX_FRAC
    fullcam_min_area: float = FULLCAM_MIN_AREA
    stacked_max_area: float = STACKED_MAX_AREA
    corner_margin: float = CORNER_MARGIN
    stable_iou: float = STABLE_IOU
    focus_max_motion: float = FOCUS_MAX_MOTION
    facecam_margin_x: float = FACECAM_MARGIN_X
    facecam_margin_y: float = FACECAM_MARGIN_Y
    edge_snap: float = EDGE_SNAP
    conf_w_agreement: float = CONF_W_AGREEMENT
    conf_w_stability: float = CONF_W_STABILITY
    conf_fallback_factor: float = CONF_FALLBACK_FACTOR

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any] | None) -> "LayoutParams":
        if not overrides:
            return cls()
        known = {f.name for f in fields(cls)}
        bad = sorted(set(overrides) - known)
        if bad:
            raise ValueError(f"unknown layout threshold(s): {', '.join(bad)}")
        return replace(cls(), **{k: float(v) for k, v in overrides.items()})

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


# -- geometry -------------------------------------------------------------------
@dataclass(frozen=True)
class Box:
    """Axis-aligned box in normalized frame coordinates (0..1)."""

    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def iou(self, o: "Box") -> float:
        iw = min(self.right, o.right) - max(self.x, o.x)
        ih = min(self.bottom, o.bottom) - max(self.y, o.y)
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        union = self.area + o.area - inter
        return inter / union if union > 0 else 0.0

    def clamp(self) -> "Box":
        x0, y0 = min(max(self.x, 0.0), 1.0), min(max(self.y, 0.0), 1.0)
        x1, y1 = min(max(self.right, 0.0), 1.0), min(max(self.bottom, 0.0), 1.0)
        return Box(x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))

    def expand(self, mx: float, my: float) -> "Box":
        """Grow by mx*w on left and right, my*h on top and bottom (not clamped)."""
        return Box(self.x - mx * self.w, self.y - my * self.h,
                   self.w * (1 + 2 * mx), self.h * (1 + 2 * my))

    def to_list(self) -> list[float]:
        return [round(self.x, 4), round(self.y, 4), round(self.w, 4), round(self.h, 4)]

    @classmethod
    def from_list(cls, v: list[float]) -> "Box":
        return cls(float(v[0]), float(v[1]), float(v[2]), float(v[3]))

    def to_pixels(self, width: int, height: int) -> list[int]:
        return [round(self.x * width), round(self.y * height),
                round(self.w * width), round(self.h * height)]


FULL_FRAME = Box(0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True)
class Face:
    box: Box
    score: float = 1.0


# per-frame detections; a frame without faces is an empty list
Detections = list[list[Face]]
# rows x cols of mean absolute frame difference (0..1), averaged over consecutive frame pairs
MotionGrid = list[list[float]]


@dataclass
class FrameAnalysis:
    size: tuple[int, int]  # (width, height) of the detection frames
    detections: Detections
    motion: MotionGrid | None = None


# -- sampling ---------------------------------------------------------------------
def sample_times(duration: float, n: int = SAMPLE_FRAMES, start: float = SAMPLE_START,
                 end: float = SAMPLE_END) -> list[float]:
    """n timestamps evenly spaced between start*duration and end*duration (inclusive)."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if n < 1:
        raise ValueError("need at least one frame")
    lo, hi = duration * start, duration * end
    if n == 1:
        return [round((lo + hi) / 2, 3)]
    return [round(lo + (hi - lo) * i / (n - 1), 3) for i in range(n)]


# -- classifier ---------------------------------------------------------------------
@dataclass
class FaceSummary:
    face_frames: int
    frames: int
    box: Box  # per-coordinate median of the per-frame primary (largest) face
    area: float  # median primary-face area
    iou: float  # median IoU of per-frame boxes against `box` (rect stability)
    near_corner: bool


CONTENT_CAM_PAD = 0.75  # extra cam-sized margin kept out of the content rect (per side)
TRACK_MAX_DIST = 0.12  # max center distance (normalized) for a face to join a track


def primary_faces(detections: Detections, corner_margin: float = 0.35) -> list[Box]:
    """Boxes of the facecam track: faces are chained across frames by position and the track
    seen in the most frames wins (ties: the larger face). Faces shown inside the content
    (images, thumbnails, other streams) are not stable across frames and so lose against
    the real facecam, however large they are."""
    tracks: list[list[tuple[int, Box]]] = []
    for i, fs in enumerate(detections):
        used: set[int] = set()
        for f in sorted(fs, key=lambda f: -f.box.area):
            # a face may move by about its own size between sampled frames
            best, best_d = None, max(TRACK_MAX_DIST, 2 * f.box.w)
            for ti, tr in enumerate(tracks):
                if ti in used:
                    continue
                last = tr[-1][1]
                d = ((last.cx - f.box.cx) ** 2 + (last.cy - f.box.cy) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = ti, d
            if best is None:
                tracks.append([(i, f.box)])
                used.add(len(tracks) - 1)
            else:
                tracks[best].append((i, f.box))
                used.add(best)
    if not tracks:
        return []

    n = len(detections)

    def frames_of(tr: list[tuple[int, Box]]) -> int:
        return len({i for i, _ in tr})

    def area_of(tr: list[tuple[int, Box]]) -> float:
        return statistics.median(b.area for _, b in tr)

    def in_corner(tr: list[tuple[int, Box]]) -> bool:
        cx = statistics.median(b.cx for _, b in tr)
        cy = statistics.median(b.cy for _, b in tr)
        return (cx < corner_margin or cx > 1 - corner_margin) and \
               (cy < corner_margin or cy > 1 - corner_margin)

    seen = [t for t in tracks if frames_of(t) * 2 >= n]  # present in at least half the frames
    # 1) a big face is the main cam (fullcam); 2) else a corner overlay is the facecam;
    # 3) else whatever was seen most
    big = [t for t in seen if area_of(t) >= STACKED_MAX_AREA]
    corner = [t for t in seen if in_corner(t)]
    pool = big or corner or tracks
    best_track = max(pool, key=lambda t: (frames_of(t), area_of(t)))
    return [b for _, b in best_track]


def summarize_faces(detections: Detections, params: LayoutParams) -> FaceSummary | None:
    boxes = primary_faces(detections, params.corner_margin)
    if not boxes:
        return None
    med = Box(
        statistics.median(b.x for b in boxes), statistics.median(b.y for b in boxes),
        statistics.median(b.w for b in boxes), statistics.median(b.h for b in boxes),
    )
    m = params.corner_margin
    corner = (med.cx < m or med.cx > 1 - m) and (med.cy < m or med.cy > 1 - m)
    return FaceSummary(
        face_frames=len(boxes),
        frames=len(detections),
        box=med,
        area=statistics.median(b.area for b in boxes),
        # stability of the rect that is actually cropped (face + margin), not of the tiny
        # face box: a small face moving a little inside a fixed facecam is still stable
        iou=statistics.median(
            med.expand(params.facecam_margin_x, params.facecam_margin_y).iou(
                b.expand(params.facecam_margin_x, params.facecam_margin_y)) for b in boxes),
        near_corner=corner,
    )


def facecam_rect(face: Box, params: LayoutParams) -> Box:
    """Face box expanded by a margin, clamped to the frame, snapped to nearby borders."""
    r = face.expand(params.facecam_margin_x, params.facecam_margin_y).clamp()
    s = params.edge_snap
    x0, y0, x1, y1 = r.x, r.y, r.right, r.bottom
    if x0 < s:
        x0 = 0.0
    if y0 < s:
        y0 = 0.0
    if x1 > 1 - s:
        x1 = 1.0
    if y1 > 1 - s:
        y1 = 1.0
    return Box(x0, y0, x1 - x0, y1 - y0)


def content_rect(cam: Box) -> Box:
    """Largest axis-aligned rectangle of the frame that does not overlap `cam`."""
    cands = [
        Box(0, 0, cam.x, 1),  # left of cam
        Box(cam.right, 0, 1 - cam.right, 1),  # right of cam
        Box(0, 0, 1, cam.y),  # above cam
        Box(0, cam.bottom, 1, 1 - cam.bottom),  # below cam
    ]
    cands = [c for c in cands if c.w > 1e-6 and c.h > 1e-6]
    return max(cands, key=lambda c: c.area) if cands else FULL_FRAME


def fullcam_crop(face: Box, frame_size: tuple[int, int]) -> Box:
    """Full-height 9:16 window centered on the face x, clamped to the frame."""
    w, h = frame_size
    cw = min(1.0, OUT_ASPECT * h / w)
    x = min(max(face.cx - cw / 2, 0.0), 1.0 - cw)
    return Box(x, 0.0, cw, 1.0)


def content_motion(motion: MotionGrid | None, cam: Box | None) -> float | None:
    """Mean frame difference over grid cells whose center lies outside `cam`."""
    if not motion or not motion[0]:
        return None
    rows, cols = len(motion), len(motion[0])
    vals = []
    for r in range(rows):
        for c in range(cols):
            cx, cy = (c + 0.5) / cols, (r + 0.5) / rows
            inside = cam is not None and cam.x <= cx <= cam.right and cam.y <= cy <= cam.bottom
            if not inside:
                vals.append(motion[r][c])
    return sum(vals) / len(vals) if vals else None


def rects_for(layout_type: str, summary: FaceSummary | None, frame_size: tuple[int, int],
              params: LayoutParams) -> tuple[dict[str, Box], list[str]]:
    """Rects a renderer needs for `layout_type`. Returns (rects, notes)."""
    if layout_type == CONTENT:
        return {"content": FULL_FRAME}, []
    if summary is None:
        return {}, [f"no face detected: no rects available for {layout_type}"]
    face = summary.box
    if layout_type == FULLCAM:
        return {"face": face, "crop": fullcam_crop(face, frame_size)}, []
    cam = facecam_rect(face, params)
    if layout_type == STACKED:
        # the real facecam overlay (frame, chat column, ...) reaches past the face + margin
        # crop, so keep the content clear of a padded cam to avoid showing the cam twice
        padded = cam.expand(CONTENT_CAM_PAD, CONTENT_CAM_PAD).clamp()
        return {"facecam": cam, "content": content_rect(padded)}, []
    if layout_type == FACE:
        return {"face": face, "facecam": cam}, []
    raise ValueError(f"unknown layout type {layout_type!r}")


@dataclass
class LayoutResult:
    type: str
    confidence: float
    rects: dict[str, Box]
    notes: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _conf(agreement: float, stability: float, params: LayoutParams, penalty: float = 1.0) -> float:
    total = params.conf_w_agreement + params.conf_w_stability
    raw = (params.conf_w_agreement * agreement + params.conf_w_stability * stability) / total
    return round(max(0.0, min(1.0, raw * penalty)), 3)


def classify(detections: Detections, motion: MotionGrid | None, frame_size: tuple[int, int],
             params: LayoutParams | None = None) -> LayoutResult:
    params = params or LayoutParams()
    n = len(detections)
    if n == 0:
        return LayoutResult(CONTENT, 0.0, {"content": FULL_FRAME}, ["no frames to analyze"])
    summary = summarize_faces(detections, params)
    face_frames = summary.face_frames if summary else 0
    stats: dict[str, Any] = {"frames": n, "face_frames": face_frames}
    notes: list[str] = []

    # too few faces: gameplay / content
    if face_frames / n < params.no_face_max_frac or summary is None:
        agreement = (n - face_frames) / n
        rects, _ = rects_for(CONTENT, summary, frame_size, params)
        notes.append(f"face in {face_frames}/{n} frames")
        return LayoutResult(CONTENT, _conf(agreement, 1.0, params), rects, notes, stats)

    stats.update(median_area=round(summary.area, 4), median_iou=round(summary.iou, 3),
                 near_corner=summary.near_corner,
                 face_box=summary.box.to_list())

    # ambiguous: some faces, but not enough for a face-based layout
    if face_frames / n < params.face_frames_min_frac:
        notes.append(f"face in only {face_frames}/{n} frames: ambiguous, falling back to content")
        rects, _ = rects_for(CONTENT, summary, frame_size, params)
        return LayoutResult(CONTENT, _conf((n - face_frames) / n, 1.0, params), rects, notes,
                            stats)

    agreement = face_frames / n
    stable = summary.iou >= params.stable_iou
    notes.append(f"face in {face_frames}/{n} frames, median area {summary.area:.1%}, "
                 f"box IoU {summary.iou:.2f}")

    if summary.area >= params.fullcam_min_area:
        rects, _ = rects_for(FULLCAM, summary, frame_size, params)
        return LayoutResult(FULLCAM, _conf(agreement, summary.iou, params), rects, notes, stats)

    if summary.near_corner and stable:
        penalty = 1.0
        if summary.area >= params.stacked_max_area:
            penalty = params.conf_fallback_factor
            notes.append(f"face area between {params.stacked_max_area:.0%} and "
                         f"{params.fullcam_min_area:.0%}: stacked with reduced confidence")
        rects, _ = rects_for(STACKED, summary, frame_size, params)
        cam = rects["facecam"]
        cm = content_motion(motion, cam)
        stats["content_motion"] = None if cm is None else round(cm, 4)
        if cm is not None and cm < params.focus_max_motion:
            notes.append(f"content region nearly static (motion {cm:.3f}): face-focus")
            rects, _ = rects_for(FACE, summary, frame_size, params)
            return LayoutResult(FACE, _conf(agreement, summary.iou, params, penalty), rects,
                                notes, stats)
        if cm is None:
            notes.append("no motion data: face-focus not evaluated")
        return LayoutResult(STACKED, _conf(agreement, summary.iou, params, penalty), rects,
                            notes, stats)

    # small or unstable face that is not a corner facecam: fullcam is the safest fit
    why = "not in a corner" if not summary.near_corner else "box position unstable"
    notes.append(f"small face ({summary.area:.1%}), {why}: fullcam with reduced confidence")
    rects, _ = rects_for(FULLCAM, summary, frame_size, params)
    return LayoutResult(
        FULLCAM, _conf(agreement, summary.iou, params, params.conf_fallback_factor), rects,
        notes, stats)


def rects_to_json(rects: dict[str, Box]) -> dict[str, list[float]]:
    return {k: v.to_list() for k, v in rects.items()}


def rects_to_pixels(rects: dict[str, Box], width: int, height: int) -> dict[str, list[int]]:
    return {k: v.to_pixels(width, height) for k, v in rects.items()}


__all__ = [
    "CONTENT", "FACE", "FULLCAM", "STACKED", "LAYOUT_TYPES", "Box", "Face", "FrameAnalysis",
    "LayoutParams", "LayoutResult", "classify", "rects_for", "sample_times",
]
