"""CIFAKE model: AI-generated (synthetic) image detection.

Report §4.2: a CNN trained on the CIFAKE dataset (60k CIFAR-10 real photos +
60k Stable Diffusion images, Bird & Lotfi) that picks up pixel- and
frequency-domain artefacts left by generators. Distinct from MesoNet, which
looks for face-swap artefacts in portraits.

Training: ``training/train_cifake.py`` (downloads the HF parquet mirror).
Lazy loading mirrors ``mesonet.py`` so the API boots without TensorFlow or
without the weights file.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from app.config import CIFAKE_MODEL_PATH, CIFAKE_THRESHOLD

INPUT_SHAPE = (32, 32, 3)  # CIFAKE images are 32x32 (CIFAR-10 sized)

_model = None
_load_error: Optional[str] = None
_tf_available: Optional[bool] = None


def prior_shift(p: float, threshold: float = CIFAKE_THRESHOLD) -> float:
    """Re-centre a raw probability so ``threshold`` maps to 0.5.

    The CNN's ranking is good (AUC ~0.977) but its probabilities are biased
    low (best operating point ~0.015 on held-out data). Shifting the logit is
    a monotone map, so ordering - and therefore AUC - is unchanged while every
    downstream 0.5 rule, confidence value and the confidence gate keep working.
    """
    if not (0.0 < threshold < 1.0):
        return p
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    logit = math.log(p / (1.0 - p)) - math.log(threshold / (1.0 - threshold))
    return 1.0 / (1.0 + math.exp(-logit))


def build_cifake(input_shape=INPUT_SHAPE):
    """Compact 3-block CNN for 32x32 RGB inputs. Requires TensorFlow/Keras."""
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=input_shape)
    x = inp
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out)


def load_cifake_model() -> Optional[Any]:
    """Return the trained CIFAKE model, or None (reason via model_status)."""
    global _model, _load_error, _tf_available
    if _model is not None:
        return _model

    if _tf_available is None:
        try:
            import tensorflow  # noqa: F401

            _tf_available = True
        except ImportError as exc:
            _tf_available = False
            _load_error = f"tensorflow not installed: {exc}"
    if not _tf_available:
        return None

    if not CIFAKE_MODEL_PATH.exists():
        _load_error = (
            f"model file not found: {CIFAKE_MODEL_PATH} "
            "(train with training/train_cifake.py)"
        )
        return None

    try:
        from tensorflow.keras.models import load_model

        _model = load_model(str(CIFAKE_MODEL_PATH))
        _load_error = None
    except Exception as exc:  # pragma: no cover - depends on local artifacts
        _model = None
        _load_error = f"failed to load CIFAKE weights: {exc}"
    return _model


def model_status() -> Dict[str, Any]:
    return {
        "loaded": load_cifake_model() is not None,
        "path": str(CIFAKE_MODEL_PATH),
        "tensorflow": bool(_tf_available),
        "error": _load_error,
    }
