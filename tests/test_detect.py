import shutil
import subprocess

import numpy as np
import pytest

from clipshorts.config import YUNET_FILE, Config
from clipshorts.detect import (
    DetectError,
    YuNetDetector,
    analyze_frames,
    motion_grid,
)
from clipshorts.layout import Box, Face
from clipshorts.media import MediaError, ffmpeg_extract_frames

cv2 = pytest.importorskip("cv2")


def write_jpg(path, color=(30, 30, 30), size=(64, 48), rect=None):
    img = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    if rect:
        x0, y0, x1, y1 = rect
        img[y0:y1, x0:x1] = (255, 255, 255)
    cv2.imwrite(str(path), img)
    return img


class StubBackend:
    """Mimics cv2.FaceDetectorYN: rows are x,y,w,h + 10 landmark values + score."""

    def __init__(self, rows):
        self.rows = rows
        self.input_size = None

    def setInputSize(self, size):  # noqa: N802 - OpenCV naming
        self.input_size = size

    def detect(self, image):
        if not self.rows:
            return 0, None
        return 1, np.array([[*r[:4], *([0.0] * 10), r[4]] for r in self.rows], dtype=np.float32)


def test_yunet_wrapper_normalizes_and_clamps(tmp_path):
    stub = StubBackend([(32, 12, 16, 12, 0.93), (-8, 40, 20, 20, 0.8)])
    det = YuNetDetector(tmp_path / "unused.onnx", backend=stub)
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    faces = det.detect(img)
    assert stub.input_size == (64, 48)
    assert len(faces) == 2
    assert faces[0].box == Box(0.5, 0.25, 0.25, 0.25)
    assert faces[0].score == pytest.approx(0.93, abs=1e-5)
    # second box starts left of the frame and runs off the bottom: clamped into 0..1
    b = faces[1].box
    assert b.x == 0.0 and b.right <= 1.0 and b.bottom <= 1.0 and b.area > 0


def test_yunet_wrapper_no_faces(tmp_path):
    det = YuNetDetector(tmp_path / "x", backend=StubBackend([]))
    assert det.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_yunet_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(DetectError) as e:
        YuNetDetector(tmp_path / "nope.onnx")
    assert "YuNet model not found" in str(e.value) and "YUNET_MODEL" in str(e.value)
    assert isinstance(e.value, MediaError)  # pipeline reports it as a normal stage failure


def test_analyze_frames_with_stub_detector(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"det_{i}.jpg"
        write_jpg(p, rect=(0, 0, 10 * (i + 1), 10))
        paths.append(p)

    class Det:
        def __init__(self):
            self.calls = 0

        def detect(self, image):
            self.calls += 1
            return [Face(Box(0.1, 0.1, 0.2, 0.2), 0.9)] if self.calls != 2 else []

    d = Det()
    res = analyze_frames(paths, d)
    assert res.size == (64, 48) and d.calls == 3
    assert [len(x) for x in res.detections] == [1, 0, 1]
    assert res.motion is not None and len(res.motion) == 12 and len(res.motion[0]) == 12


def test_analyze_frames_unreadable_frame(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(DetectError, match="cannot read frame"):
        analyze_frames([bad], StubBackend([]))
    with pytest.raises(DetectError, match="no frames"):
        analyze_frames([], StubBackend([]))


def test_motion_grid_static_vs_changed_region():
    same = np.full((48, 64, 3), 40, dtype=np.uint8)
    g = motion_grid([same, same.copy(), same.copy()])
    assert max(max(r) for r in g) == 0.0
    changed = same.copy()
    changed[:, 32:] = 255  # right half flips
    g2 = motion_grid([same, changed])
    left = sum(g2[r][c] for r in range(12) for c in range(0, 5)) / 60
    right = sum(g2[r][c] for r in range(12) for c in range(7, 12)) / 60
    assert left < 0.01 and right > 0.5
    assert motion_grid([same]) is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_extract_frames_real_ffmpeg(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=1280x720:rate=15:duration=4", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    seen = []
    frames = ffmpeg_extract_frames(src, [0.2, 2.0, 3.8], tmp_path / "frames", "ffmpeg", 640,
                                   seen.append)
    assert len(frames) == 3 and seen[-1] == 1.0
    orig = cv2.imread(str(frames[0].path))
    det = cv2.imread(str(frames[0].det_path))
    assert orig.shape[:2] == (720, 1280)  # original kept at full resolution
    assert det.shape[:2] == (360, 640)  # detection copy is 640 wide
    res = analyze_frames([f.det_path for f in frames], StubBackend([]))
    assert res.size == (640, 360) and res.motion is not None
    with pytest.raises(MediaError):
        ffmpeg_extract_frames(tmp_path / "missing.mp4", [1.0], tmp_path / "f2")


# Optional: the vendored model. Skipped when absent. Checks load + run only: a synthetic image
# has no real face, so this does not measure detection accuracy.
MODEL = Config(_env_file=None).yunet_model_path


@pytest.mark.skipif(not MODEL.is_file(), reason=f"{YUNET_FILE} not present")
def test_real_yunet_loads_and_runs_on_synthetic_frame():
    det = YuNetDetector(MODEL, 0.7)
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 180), 80, (200, 180, 160), -1)
    faces = det.detect(img)
    assert isinstance(faces, list)  # may legitimately be empty for a drawn circle
    assert all(0 <= f.box.x <= 1 and f.box.right <= 1 for f in faces)
