from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.services import facecrop  # noqa: E402


def _synthetic_no_face(size: int = 320) -> np.ndarray:
    """Noise-free geometric image with no face-like features."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[: size // 2] = (40, 80, 160)
    img[size // 2 :] = (200, 120, 30)
    return img


def test_no_face_returns_original_frame():
    frame = _synthetic_no_face()
    out, detected = facecrop.crop_face(cv2, frame, rgb=True)
    assert detected is False
    assert out.shape == frame.shape


def test_crop_disabled_returns_original(monkeypatch):
    monkeypatch.setattr(facecrop, "FACE_CROP", False)
    frame = _synthetic_no_face()
    out, detected = facecrop.crop_face(cv2, frame, rgb=True)
    assert detected is False
    assert out is frame


def test_crop_returns_tuple_and_never_empty():
    frame = _synthetic_no_face()
    out, detected = facecrop.crop_face(cv2, frame, rgb=True)
    assert isinstance(detected, bool)
    assert out.ndim == 3
    assert out.size > 0


def test_real_face_crop_image_detects_face():
    """A cached DFDC face crop should (almost always) contain a detectable face."""
    from pathlib import Path

    cache = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "mesonet_cache"
        / "val"
        / "real"
    )
    files = sorted(cache.glob("*.jpg")) if cache.exists() else []
    if not files:
        pytest.skip("mesonet cache not present")
    frame = cv2.imread(str(files[0]))
    if frame is None:
        pytest.skip("cv2 could not read cache image")
    out, detected = facecrop.crop_face(cv2, frame, rgb=False)
    assert out.size > 0
    if detected:
        assert out.shape[0] <= frame.shape[0] and out.shape[1] <= frame.shape[1]
