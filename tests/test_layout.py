import pytest

from clipshorts.layout import (
    CONTENT,
    FACE,
    FULL_FRAME,
    FULLCAM,
    STACKED,
    Box,
    Face,
    LayoutParams,
    classify,
    content_motion,
    content_rect,
    facecam_rect,
    fullcam_crop,
    rects_for,
    sample_times,
    summarize_faces,
)

SIZE = (640, 360)
P = LayoutParams()


def faces(*boxes, n=8, present=None):
    """n frames; the same boxes in every frame (or only in `present` frame indexes)."""
    out = []
    for i in range(n):
        if present is None or i in present:
            out.append([Face(b, 0.9) for b in boxes])
        else:
            out.append([])
    return out


def jitter(base: Box, n=8, amount=0.01):
    """Per-frame boxes shifted by a small deterministic amount."""
    out = []
    for i in range(n):
        d = amount * ((i % 3) - 1)
        out.append([Face(Box(base.x + d, base.y - d, base.w, base.h), 0.9)])
    return out


def static_grid(v=0.0, rows=12, cols=12):
    return [[v] * cols for _ in range(rows)]


BIG = Box(0.35, 0.15, 0.3, 0.5)  # 15% area, centered
CORNER = Box(0.04, 0.66, 0.12, 0.2)  # 2.4% area, bottom-left
MID_CORNER = Box(0.04, 0.55, 0.3, 0.3)  # 9% area, in the 8-12% gap


# -- sampling -------------------------------------------------------------------
def test_sample_times_even_between_5_and_95_percent():
    t = sample_times(100.0)
    assert len(t) == 8
    assert t[0] == pytest.approx(5.0) and t[-1] == pytest.approx(95.0)
    steps = [b - a for a, b in zip(t, t[1:], strict=False)]
    assert max(steps) - min(steps) < 0.01


def test_sample_times_edge_cases():
    assert sample_times(10.0, n=1) == [5.0]
    assert len(sample_times(0.5)) == 8
    with pytest.raises(ValueError):
        sample_times(0)
    with pytest.raises(ValueError):
        sample_times(10, n=0)


# -- geometry -------------------------------------------------------------------
def test_iou():
    a = Box(0, 0, 0.5, 0.5)
    assert a.iou(a) == pytest.approx(1.0)
    assert a.iou(Box(0.5, 0.5, 0.2, 0.2)) == 0.0
    assert a.iou(Box(0.25, 0, 0.5, 0.5)) == pytest.approx(1 / 3)


def test_expand_and_clamp():
    b = Box(0.0, 0.0, 0.1, 0.1).expand(1.0, 1.0)
    assert (b.x, b.y) == (-0.1, -0.1)
    c = b.clamp()
    assert c.x == 0 and c.y == 0 and c.right == pytest.approx(0.2)
    assert Box(2, 2, 1, 1).clamp().area == 0


def test_facecam_rect_clamped_and_snapped_to_borders():
    cam = facecam_rect(CORNER, P)
    assert cam.x == 0.0  # expanded past/near left edge -> snapped
    assert cam.bottom == 1.0  # expanded past/near bottom -> snapped
    assert cam.right <= 1 and cam.y >= 0
    assert cam.w > CORNER.w and cam.h > CORNER.h


def test_facecam_rect_in_middle_keeps_margin_and_does_not_snap():
    face = Box(0.4, 0.4, 0.1, 0.1)
    cam = facecam_rect(face, P)
    assert cam.x == pytest.approx(0.4 - 0.75 * 0.1)
    assert cam.w == pytest.approx(0.1 * (1 + 2 * 0.75))


def test_content_rect_is_largest_remainder():
    cam = Box(0, 0.7, 0.3, 0.3)  # bottom-left corner
    rest = content_rect(cam)
    # above the cam: 1.0 x 0.7 = 0.70; right of cam: 0.7 x 1.0 = 0.70 -> tie, both valid
    assert rest.area == pytest.approx(0.7)
    assert rest.iou(cam) == 0.0
    cam2 = Box(0, 0.5, 0.2, 0.5)  # left half-height strip
    r2 = content_rect(cam2)
    assert r2 == Box(0.2, 0, 0.8, 1)


def test_content_rect_cam_covering_everything_falls_back():
    assert content_rect(Box(0, 0, 1, 1)) == FULL_FRAME


def test_fullcam_crop_is_9_16_clamped():
    crop = fullcam_crop(Box(0.5, 0.2, 0.1, 0.2), (1920, 1080))
    assert crop.w == pytest.approx(0.5625 * 1080 / 1920)
    assert crop.cx == pytest.approx(0.55)
    edge = fullcam_crop(Box(0.95, 0.2, 0.05, 0.2), (1920, 1080))
    assert edge.right == pytest.approx(1.0) and edge.x >= 0
    left = fullcam_crop(Box(0.0, 0.2, 0.05, 0.2), (1920, 1080))
    assert left.x == 0.0
    # frame already narrower than 9:16: crop is the whole frame width
    assert fullcam_crop(Box(0.4, 0.1, 0.2, 0.2), (360, 1080)).w == 1.0


def test_content_motion_ignores_cells_inside_cam():
    grid = static_grid(0.0)
    for r in range(6, 12):
        for c in range(0, 4):
            grid[r][c] = 1.0  # busy bottom-left block
    cam = Box(0, 0.5, 0.34, 0.5)  # covers exactly that block
    assert content_motion(grid, cam) == pytest.approx(0.0)
    assert content_motion(grid, None) > 0.1
    assert content_motion(None, cam) is None
    assert content_motion([], cam) is None


# -- classifier: every layout type ----------------------------------------------------
def test_fullcam_big_face_most_frames():
    r = classify(faces(BIG), None, SIZE)
    assert r.type == FULLCAM
    assert r.confidence > 0.95
    assert "crop" in r.rects and "face" in r.rects
    assert r.rects["crop"].h == 1.0


def test_fullcam_threshold_boundary_6_of_8():
    r = classify(faces(BIG, present={0, 1, 2, 3, 4, 5}), None, SIZE)
    assert r.type == FULLCAM
    assert 0.7 < r.confidence < 0.95  # 6/8 agreement lowers it


def test_stacked_corner_facecam_with_moving_content():
    r = classify(faces(CORNER), static_grid(0.2), SIZE)
    assert r.type == STACKED
    assert r.rects["facecam"].iou(r.rects["content"]) == 0.0
    assert r.rects["facecam"].area > CORNER.area
    assert r.confidence > 0.95


def test_face_focus_when_stacked_and_content_static():
    r = classify(faces(CORNER), static_grid(0.003), SIZE)
    assert r.type == FACE
    assert "facecam" in r.rects and "face" in r.rects
    assert r.stats["content_motion"] == pytest.approx(0.003)


def test_motion_inside_facecam_does_not_prevent_face_focus():
    cam = facecam_rect(CORNER, P)
    grid = static_grid(0.003)
    rows = cols = 12
    for r in range(rows):
        for c in range(cols):
            if cam.x <= (c + 0.5) / cols <= cam.right and cam.y <= (r + 0.5) / rows <= cam.bottom:
                grid[r][c] = 0.5  # lively streamer in the cam
    assert classify(faces(CORNER), grid, SIZE).type == FACE


def test_stacked_without_motion_data_is_stacked_with_note():
    r = classify(faces(CORNER), None, SIZE)
    assert r.type == STACKED
    assert any("no motion data" in n for n in r.notes)


def test_content_when_no_faces_at_all():
    r = classify(faces(present=set()), None, SIZE)
    assert r.type == CONTENT and r.rects == {"content": FULL_FRAME}
    assert r.confidence == 1.0


def test_content_when_two_of_eight_frames_have_faces():
    r = classify(faces(BIG, present={1, 5}), None, SIZE)
    assert r.type == CONTENT
    assert r.confidence == pytest.approx(0.6 * 0.75 + 0.4, abs=1e-3)


# -- edge cases -----------------------------------------------------------------------
def test_no_frames_at_all():
    r = classify([], None, SIZE)
    assert r.type == CONTENT and r.confidence == 0.0 and "no frames" in r.notes[0]


def test_ambiguous_3_to_5_faces_falls_back_to_content_with_low_confidence():
    r = classify(faces(BIG, present={0, 2, 4, 6}), None, SIZE)  # 4 of 8
    assert r.type == CONTENT
    assert r.confidence < 0.8
    assert any("ambiguous" in n for n in r.notes)
    three = classify(faces(BIG, present={0, 1, 2}), None, SIZE)
    assert three.type == CONTENT and any("ambiguous" in n for n in three.notes)


def test_jittery_boxes_still_stable_and_stacked():
    r = classify(jitter(CORNER, amount=0.004), static_grid(0.2), SIZE)
    assert r.type == STACKED
    assert r.stats["median_iou"] > 0.7


def test_iou_instability_small_face_not_stacked():
    # small face whose median sits in the left corner region but barely overlaps frame to frame
    boxes = [Box(0.02, 0.62, 0.1, 0.2), Box(0.2, 0.62, 0.1, 0.2)]
    dets = [[Face(boxes[i % 2], 0.9)] for i in range(8)]
    r = classify(dets, static_grid(0.2), SIZE)
    assert r.type == FULLCAM
    assert r.confidence < 0.6
    assert any("unstable" in n for n in r.notes)


def test_small_face_not_in_corner_is_low_confidence_fullcam():
    center_small = Box(0.45, 0.4, 0.1, 0.15)  # 1.5%, dead center
    r = classify(faces(center_small), None, SIZE)
    assert r.type == FULLCAM
    assert r.confidence < 0.75
    assert any("not in a corner" in n for n in r.notes)


def test_gap_zone_area_corner_is_stacked_with_reduced_confidence():
    r = classify(faces(MID_CORNER), static_grid(0.2), SIZE)
    assert r.type == STACKED
    assert r.confidence == pytest.approx(0.7, abs=0.01)
    assert any("reduced confidence" in n for n in r.notes)


def test_multiple_faces_uses_largest_per_frame():
    dets = [[Face(CORNER, 0.9), Face(BIG, 0.8)] for _ in range(8)]
    assert classify(dets, None, SIZE).type == FULLCAM


def test_thresholds_are_tunable():
    strict = LayoutParams.from_overrides({"fullcam_min_area": 0.5})
    # 15% face is no longer fullcam; it is centered (not a corner) -> fallback fullcam, low conf
    r = classify(faces(BIG), None, SIZE, strict)
    assert r.type == FULLCAM and r.confidence < 0.8
    with pytest.raises(ValueError):
        LayoutParams.from_overrides({"nope": 1})
    assert LayoutParams.from_overrides(None) == LayoutParams()


def test_confidence_is_clamped_0_1_and_drops_with_instability():
    stable = classify(faces(BIG), None, SIZE).confidence
    dets = [[Face(Box(0.05 + 0.1 * (i % 5), 0.15, 0.3, 0.5), 0.9)] for i in range(8)]
    shaky = classify(dets, None, SIZE)
    assert shaky.type == FULLCAM
    assert 0 <= shaky.confidence < stable <= 1


# -- rects_for / summary -------------------------------------------------------------
def test_rects_for_without_face_has_note():
    rects, notes = rects_for(STACKED, None, SIZE, P)
    assert rects == {} and "no face" in notes[0]
    assert rects_for(CONTENT, None, SIZE, P)[0] == {"content": FULL_FRAME}


def test_rects_for_override_type_uses_detected_face():
    s = summarize_faces(faces(CORNER), P)
    assert set(rects_for(FULLCAM, s, SIZE, P)[0]) == {"face", "crop"}
    assert set(rects_for(STACKED, s, SIZE, P)[0]) == {"facecam", "content"}
    assert set(rects_for(FACE, s, SIZE, P)[0]) == {"face", "facecam"}
    with pytest.raises(ValueError):
        rects_for("bogus", s, SIZE, P)


def test_summary_none_without_faces_and_corner_flag():
    assert summarize_faces(faces(present=set()), P) is None
    assert summarize_faces(faces(CORNER), P).near_corner is True
    assert summarize_faces(faces(BIG), P).near_corner is False
