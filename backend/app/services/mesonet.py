"""MesoNet (Meso4) architecture and lazy model loading.

Shared by the image and video services, and imported by
``training/train_mesonet.py`` so training and inference use the exact same
architecture.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.config import MESO_MODEL_PATH

INPUT_SHAPE = (256, 256, 3)

_model = None
_load_error: Optional[str] = None
_tf_available: Optional[bool] = None


def build_meso4(input_shape=INPUT_SHAPE):
    """Build the Meso4 network (Damer et al., "MesoNet: A Compact Facial
    Video Forgery Detection Network"). Requires TensorFlow/Keras."""
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=input_shape)
    x = inp
    for filters, k, pool in [(8, 3, 2), (8, 5, 2), (16, 5, 2), (16, 5, 4)]:
        x = layers.Conv2D(filters, k, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(pool, pool), padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(16)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out)


def load_meso_model() -> Optional[Any]:
    """Return the loaded Meso4 model, or None when it cannot be used.

    Failure reasons (surfaced via ``model_status``):
      * TensorFlow not installed
      * app/models/meso4.h5 missing (not trained yet)
      * weights file corrupt / incompatible
    """
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

    if not MESO_MODEL_PATH.exists():
        _load_error = (
            f"model file not found: {MESO_MODEL_PATH} "
            "(train with training/train_mesonet.py)"
        )
        return None

    try:
        from tensorflow.keras.models import load_model

        _model = load_model(str(MESO_MODEL_PATH))
        _load_error = None
    except Exception as exc:  # pragma: no cover - depends on local artifacts
        _model = None
        _load_error = f"failed to load MesoNet weights: {exc}"
    return _model


def model_status() -> Dict[str, Any]:
    return {
        "loaded": load_meso_model() is not None,
        "path": str(MESO_MODEL_PATH),
        "tensorflow": bool(_tf_available),
        "error": _load_error,
    }
