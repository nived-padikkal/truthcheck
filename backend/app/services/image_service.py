"""Image analysis: AI-generated image check (CIFAKE) + face-swap check (Meso4).

Two independent engines feed the result aggregator (report §2/§4):

* **cifake** - CNN on the CIFAKE dataset, flags AI-generated images
  (services.cifake). Scores the whole frame at 32x32.
* **meso4** - MesoNet on a detected face crop, flags face-swap deepfakes
  (services.mesonet + services.facecrop). Scores at 256x256.

Merge rule (primary/advisory): the engines answer *different* questions -
meso4 asks "is this a face swap?", cifake asks "is this an AI-generated
picture?" - so a literal agreement rule (the first version) made every
genuine face swap fight cifake's correct-for-its-question REAL vote: measured
accuracy on DFDC fake frames collapsed from 0.61 to 0.15. Instead the
domain-primary engine decides (face detected -> meso4, else cifake) and the
other engine runs as an advisory check that may only veto a *REAL* verdict
when it is extremely confident on its native scale (>= 0.95 raw), which
costs ~2% of real frames (measured false-positive rate at that cutoff)
while catching AI portraits the face-swap model cannot judge. Confidence
gate (services.gating) applies afterwards. Missing models degrade to
UNKNOWN with the reason spelled out.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.services import cifake, facecrop, gating, mesonet

INPUT_SIZE = (256, 256)  # MesoNet input
CIFAKE_SIZE = (32, 32)  # CIFAKE input (CIFAR-10 sized)


def _decode(image_bytes: bytes) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    """Decode image bytes -> (RGB uint8 array, (width, height)) or None."""
    if not image_bytes:
        return None
    try:
        import cv2

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        return bgr[:, :, ::-1], (w, h)
    except ImportError:
        pass

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.asarray(img), img.size
    except Exception:
        return None


def _resize(rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(rgb, size)
    except ImportError:
        from PIL import Image

        img = Image.fromarray(rgb)
        return np.asarray(img.resize(size))


def _to_batch(rgb: np.ndarray) -> np.ndarray:
    return np.expand_dims(rgb.astype("float32") / 255.0, axis=0)


def _check_cifake(model, rgb: np.ndarray) -> Dict[str, Any]:
    small = _resize(rgb, CIFAKE_SIZE)
    raw = float(model.predict(_to_batch(small), verbose=0).ravel()[0])
    p_fake = cifake.prior_shift(raw)
    return {
        "model": "cifake",
        "task": "ai_generated_image",
        "p_fake": round(p_fake, 4),
        "p_raw": round(raw, 4),
    }


def _check_meso(model, rgb: np.ndarray) -> Dict[str, Any]:
    frame = rgb
    face_detected = False
    try:
        import cv2

        frame, face_detected = facecrop.crop_face(cv2, rgb, rgb=True)
    except ImportError:
        pass  # Pillow-only install: score the whole frame
    resized = _resize(frame, INPUT_SIZE)
    p_fake = float(model.predict(_to_batch(resized), verbose=0).ravel()[0])
    return {
        "model": "meso4",
        "task": "face_swap",
        "p_fake": round(p_fake, 4),
        "face_detected": face_detected,
        "face_detector": facecrop.detector_kind(),
    }


ADVISORY_VETO_THR = 0.95  # advisory engine must be *natively* this sure to overturn REAL


def _merge(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate engine votes under the primary/advisory rule (see module doc)."""
    by_model = {c["model"]: c for c in checks}
    meso = by_model.get("meso4")
    if meso is not None and meso.get("face_detected"):
        primary, advisory = meso, by_model.get("cifake")
    else:
        primary = by_model.get("cifake") or meso
        advisory = None if primary is meso else meso

    verdict = "FAKE" if primary["p_fake"] >= 0.5 else "REAL"
    confidence = primary["p_fake"] if verdict == "FAKE" else 1.0 - primary["p_fake"]
    details: Dict[str, Any] = {
        "p_fake": primary["p_fake"],
        "deciding_model": primary["model"],
    }

    if advisory is not None:
        a_vote = "FAKE" if advisory["p_fake"] >= 0.5 else "REAL"
        # Veto compares the engine's *native* probability (cifake.p_raw - the
        # calibrated p_fake would equate 0.95 with raw 0.27 and fire on ~5% of
        # real frames, measured).
        a_native = advisory.get("p_raw", advisory["p_fake"])
        details["advisory"] = {
            "model": advisory["model"],
            "vote": a_vote,
            "p_fake": advisory["p_fake"],
            **({"p_raw": advisory["p_raw"]} if "p_raw" in advisory else {}),
        }
        if a_vote == verdict:
            # engines agree: blend confidence as documented (aggregator layer)
            a_conf = advisory["p_fake"] if a_vote == "FAKE" else 1.0 - advisory["p_fake"]
            confidence = (confidence + a_conf) / 2.0
        elif verdict == "REAL" and a_native >= ADVISORY_VETO_THR:
            # advisory screams FAKE (natively sure): refuse to answer
            # confidently instead of claiming REAL. Measured: only ~2% of real
            # frames reach raw >= 0.95, so this rarely costs real material;
            # a FAKE verdict is never vetoed the other way.
            details["merged_p_fake"] = round(
                (primary["p_fake"] + advisory["p_fake"]) / 2.0, 4
            )
            return {
                "verdict": "UNKNOWN",
                "confidence": 0.0,
                "details": {
                    **details,
                    "reason": (
                        f"{primary['model']} says REAL but {advisory['model']} "
                        f"flags AI generation (score={a_native:.2f} >= "
                        f"{ADVISORY_VETO_THR:g})"
                    ),
                },
            }

    details["merged_p_fake"] = round(
        sum(c["p_fake"] for c in checks) / len(checks), 4
    ) if len(checks) > 1 else primary["p_fake"]
    return {"verdict": verdict, "confidence": round(confidence, 4), "details": details}


def predict(image_bytes: bytes) -> Dict[str, Any]:
    """Classify a single image. Returns {verdict, confidence, details}."""
    decoded = _decode(image_bytes)
    if decoded is None:
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "details": {"error": "could not decode image", "p_fake": 0.5},
        }

    rgb, (width, height) = decoded
    details: Dict[str, Any] = {
        "image_size": [width, height],
        "p_fake": 0.5,
    }

    checks: List[Dict[str, Any]] = []
    unavailable: List[str] = []

    cifake_model = cifake.load_cifake_model()
    if cifake_model is not None:
        checks.append(_check_cifake(cifake_model, rgb))
    else:
        unavailable.append(f"cifake: {cifake.model_status().get('error') or 'unavailable'}")

    meso_model = mesonet.load_meso_model()
    if meso_model is not None:
        checks.append(_check_meso(meso_model, rgb))
    else:
        unavailable.append(f"meso4: {mesonet.model_status().get('error') or 'unavailable'}")

    if not checks:
        details.update(
            {
                "model_loaded": False,
                "reason": "; ".join(unavailable) or "no image model available",
            }
        )
        return {"verdict": "UNKNOWN", "confidence": 0.0, "details": details}

    merged = _merge(checks)
    details.update(merged["details"])
    details["model_loaded"] = True
    details["checks"] = checks
    details["model"] = "+".join(c["model"] for c in checks)
    deciding = next(
        (c for c in checks if c["model"] == details.get("deciding_model")), checks[0]
    )
    details["task"] = deciding["task"]
    meso_check = next((c for c in checks if c["model"] == "meso4"), None)
    if meso_check is not None:
        details["face_detected"] = meso_check["face_detected"]
        details["face_detector"] = meso_check["face_detector"]

    if unavailable:
        details["models_unavailable"] = unavailable

    result = {
        "verdict": merged["verdict"],
        "confidence": merged["confidence"],
        "details": details,
    }
    return gating.apply_gate(result)
