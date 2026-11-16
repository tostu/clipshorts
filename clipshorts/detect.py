"""OpenCV side of layout detection: YuNet wrapper, frame loading, frame-difference grid.

cv2/numpy are imported lazily so importing this module costs nothing in the unit-test path.
The detector is injectable (`FaceDetector` protocol), so tests use a stub instead of the model.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from .layout import Box, Detections, Face, FrameAnalysis, MotionGrid
from .media import MediaError

MOTION_RES = 96  # frames are downscaled to MOTION_RES x MOTION_RES gray before differencing
MOTION_GRID = 12  # grid cells per side


class DetectError(MediaError):
    """Face detection could not run (missing model, unreadable frame, ...)."""


class FaceDetector(Protocol):
    def detect(self, image: Any) -> list[Face]:
        """Faces in a BGR uint8 image, as normalized boxes."""


class YuNetDetector:
    """cv2.FaceDetectorYN wrapper returning normalized boxes.

    `backend` lets tests inject an object with setInputSize()/detect() instead of the model.
    """

    def __init__(self, model_path: Path, score_threshold: float = 0.7, nms_threshold: float = 0.3,
                 top_k: int = 50, backend: Any = None):
        if backend is not None:
            self._det = backend
            return
        if not Path(model_path).is_file():
            raise DetectError(
                f"YuNet model not found at {model_path}. Place face_detection_yunet_2023mar.onnx "
                "there or point the YUNET_MODEL setting at it."
            )
        import cv2

        if not hasattr(cv2, "FaceDetectorYN"):
            raise DetectError("this OpenCV build has no FaceDetectorYN (need opencv >= 4.5.4)")
        try:
            self._det = cv2.FaceDetectorYN.create(
                str(model_path), "", (320, 320), score_threshold, nms_threshold, top_k
            )
        except cv2.error as e:
            raise DetectError(f"cannot load YuNet model {model_path}: {e}") from e

    def detect(self, image: Any) -> list[Face]:
        h, w = image.shape[:2]
        self._det.setInputSize((w, h))
        _, rows = self._det.detect(image)
        faces: list[Face] = []
        if rows is None:
            return faces
        for row in rows:
            x, y, bw, bh = (float(v) for v in row[:4])
            box = Box(x / w, y / h, bw / w, bh / h).clamp()
            if box.area > 0:
                faces.append(Face(box, float(row[-1])))
        return faces


def motion_grid(images: list[Any], grid: int = MOTION_GRID, res: int = MOTION_RES
                ) -> MotionGrid | None:
    """Mean abs gray difference between consecutive frames, reduced to a grid x grid map."""
    if len(images) < 2:
        return None
    import cv2
    import numpy as np

    small = [
        cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (res, res),
                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        for im in images
    ]
    diff = np.mean([np.abs(a - b) for a, b in zip(small, small[1:], strict=False)], axis=0)
    k = res // grid
    cells = diff[: k * grid, : k * grid].reshape(grid, k, grid, k).mean(axis=(1, 3))
    return [[round(float(v), 5) for v in row] for row in cells]


def analyze_frames(paths: list[Path], detector: FaceDetector) -> FrameAnalysis:
    """Run `detector` on each frame image and compute the frame-difference grid."""
    import cv2

    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise DetectError(f"cannot read frame {Path(p).name}")
        images.append(img)
    if not images:
        raise DetectError("no frames to analyze")
    h, w = images[0].shape[:2]
    detections: Detections = [detector.detect(im) for im in images]
    return FrameAnalysis((w, h), detections, motion_grid(images))


def analyze_with_yunet(paths: list[Path], model_path: Path, score_threshold: float,
                       on_progress: Callable[[float], None] | None = None) -> FrameAnalysis:
    """Default `Tools.analyze_frames`: YuNet on the given detection frames."""
    detector = YuNetDetector(model_path, score_threshold)
    analysis = analyze_frames(paths, detector)
    if on_progress:
        on_progress(1.0)
    return analysis
