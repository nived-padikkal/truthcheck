"""Video deepfake detection: extract frames, score them with MesoNet, aggregate.

Every ``every_n``-th frame is taken (up to ``max_frames``), each frame is
optionally cropped to the largest detected face (MesoNet is trained on face
crops), resized to 256x256, and scored. The video-level fake probability is
the mean of the frame scores; low-margin results are gated to UNKNOWN
(see services.gating).
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List

import numpy as np

from app.config import VIDEO_EVERY_N, VIDEO_MAX_FRAMES
from app.services import facecrop, gating, mesonet

INPUT_SIZE = (256, 256)


def _collect_frames(
    video_bytes: bytes,
    every_n: int,
    max_frames: int,
    meta: Dict[str, Any],
) -> List[np.ndarray]:
    import cv2

    frames: List[np.ndarray] = []
    faces_seen = 0
    # delete=False: on Windows a still-open NamedTemporaryFile cannot be
    # reopened by OpenCV, so we write first, then read by path, then clean up.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            meta["error"] = "could not open video stream"
            return frames
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps and fps > 0:
                meta["duration_sec"] = round(total / fps, 2)
                meta["fps"] = round(fps, 2)
            i = 0
            while cap.isOpened() and len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % every_n == 0:
                    frame, found = facecrop.crop_face(cv2, frame)
                    faces_seen += int(found)
                    frame = cv2.resize(frame, INPUT_SIZE)
                    frames.append(frame[:, :, ::-1])  # BGR -> RGB
                i += 1
            meta["source_frames"] = i
            meta["frames_with_face"] = faces_seen
            meta["face_detector"] = facecrop.detector_kind()
        finally:
            cap.release()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return frames


def predict(
    video_bytes: bytes,
    every_n: int = VIDEO_EVERY_N,
    max_frames: int = VIDEO_MAX_FRAMES,
) -> Dict[str, Any]:
    """Classify a video. Returns {verdict, confidence, details}."""
    if not video_bytes:
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "details": {"error": "empty upload", "p_fake": 0.5},
        }

    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "details": {"error": f"opencv not installed: {exc}", "p_fake": 0.5},
        }

    details: Dict[str, Any] = {"p_fake": 0.5, "every_n": every_n}
    frames_rgb = _collect_frames(video_bytes, every_n, max_frames, details)

    if not frames_rgb:
        details.setdefault("error", "no frames extracted")
        return {"verdict": "UNKNOWN", "confidence": 0.0, "details": details}

    details["frames_analyzed"] = len(frames_rgb)

    model = mesonet.load_meso_model()
    if model is None:
        details.update(
            {
                "model_loaded": False,
                "model": "meso4",
                "reason": mesonet.model_status().get("error") or "model unavailable",
            }
        )
        return {"verdict": "UNKNOWN", "confidence": 0.0, "details": details}

    batch = np.asarray(frames_rgb, dtype="float32") / 255.0
    scores = model.predict(batch, verbose=0).ravel()  # 1 = fake
    p_fake = float(np.mean(scores))
    verdict = "FAKE" if p_fake >= 0.5 else "REAL"
    confidence = p_fake if verdict == "FAKE" else 1.0 - p_fake
    details.update(
        {
            "model_loaded": True,
            "model": "meso4",
            "p_fake": round(p_fake, 4),
            "frame_scores": [round(float(s), 4) for s in scores],
            "frames_forged": int(np.sum(scores >= 0.5)),
        }
    )
    result = {"verdict": verdict, "confidence": round(confidence, 4), "details": details}
    return gating.apply_gate(result)
