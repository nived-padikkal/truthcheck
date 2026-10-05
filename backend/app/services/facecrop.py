"""Shared face-cropping helper for MesoNet inputs.

MesoNet is trained on tight 256x256 face crops, so scoring a whole photo
(scenes, groups, small faces) pushes the prediction toward noise. Both the
image and video services crop to the largest detected face first and fall
back to the full frame when no face is found (or TC_FACE_CROP=0).

Detectors, in order of preference:

1. **YuNet** (DNN, ``cv2.FaceDetectorYN``) - accurate and the only option on
   OpenCV 5.x, which removed ``CascadeClassifier``. The ONNX model (~230 KB)
   is downloaded once to ``app/models/`` on first use.
2. **Haar cascade** - OpenCV 4.x installs where ``cv2.data.haarcascades`` ships
   the XML files.
3. No detector: the original frame is returned unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from app.config import (
    FACE_CROP,
    FACE_MODEL_PATH,
    FACE_MODEL_URL,
    FACE_SCORE_THRESHOLD,
)

logger = logging.getLogger("truthcheck")

_detector = None
_detector_tried = False
_detector_kind: Optional[str] = None  # "yunet" | "haar" | None


def detector_kind() -> Optional[str]:
    """Which backend is in use ("yunet", "haar", or None when unavailable)."""
    return _detector_kind


def _fetch_model() -> Optional[str]:
    """Return a local path to the YuNet ONNX, downloading it if needed."""
    if FACE_MODEL_PATH.exists() and FACE_MODEL_PATH.stat().st_size > 0:
        return str(FACE_MODEL_PATH)
    try:
        import requests

        logger.info("downloading YuNet face detector from %s", FACE_MODEL_URL)
        resp = requests.get(FACE_MODEL_URL, timeout=60)
        resp.raise_for_status()
        FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = FACE_MODEL_PATH.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(FACE_MODEL_PATH)
        return str(FACE_MODEL_PATH)
    except Exception as exc:  # network blocked / disk error -> fall through
        logger.warning("could not fetch YuNet face detector: %s", exc)
        return None


def _load_detector(cv2):
    """Create the best available detector once (None when nothing works)."""
    global _detector, _detector_tried, _detector_kind
    if _detector_tried:
        return _detector
    _detector_tried = True

    if hasattr(cv2, "FaceDetectorYN_create"):
        model = _fetch_model()
        if model is not None:
            try:
                _detector = cv2.FaceDetectorYN_create(
                    model, "", (320, 320), score_threshold=FACE_SCORE_THRESHOLD
                )
                _detector_kind = "yunet"
                return _detector
            except Exception as exc:
                logger.warning("YuNet init failed: %s", exc)

    try:  # OpenCV 4.x only; 5.x has no CascadeClassifier/data files
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            _detector = cascade
            _detector_kind = "haar"
            return _detector
    except Exception:
        pass

    logger.warning("no face detector available; scoring full frames")
    _detector_kind = None
    return None


def _largest_face_box(cv2, frame: np.ndarray, rgb: bool) -> Optional[Tuple[int, int, int, int]]:
    """Largest face as (x, y, w, h) in the frame's own coordinates, or None."""
    det = _load_detector(cv2)
    if det is None:
        return None

    if _detector_kind == "yunet":
        # YuNet expects BGR; flip a cheap view when the caller passed RGB.
        bgr = frame[:, :, ::-1] if rgb else frame
        det.setInputSize((bgr.shape[1], bgr.shape[0]))
        ok, faces = det.detect(bgr)
        if not ok or faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[-1])  # last value is the score
        x, y, w, h = (int(v) for v in best[:4])
        return x, y, max(w, 1), max(h, 1)

    # Haar cascade path
    code = cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY
    gray = cv2.cvtColor(frame, code)
    faces = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def crop_face(cv2, frame: np.ndarray, rgb: bool = False) -> Tuple[np.ndarray, bool]:
    """Crop to the largest detected face with 15% padding.

    Returns ``(cropped_frame, face_detected)``. When no face is found, cropping
    is disabled, or no detector is available, the original frame is returned
    with ``face_detected=False``. ``rgb=True`` means the caller's frame is RGB
    (single-image path); the default assumes BGR (video frames).
    """
    if not FACE_CROP:
        return frame, False
    box = _largest_face_box(cv2, frame, rgb)
    if box is None:
        return frame, False
    x, y, w, h = box
    pad_x, pad_y = int(w * 0.15), int(h * 0.15)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(frame.shape[1], x + w + pad_x)
    y1 = min(frame.shape[0], y + h + pad_y)
    cropped = frame[y0:y1, x0:x1]
    if not cropped.size:
        return frame, False
    return cropped, True
