"""Runtime configuration for the TruthCheck API."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent

MODELS_DIR = BASE_DIR / "models"
TEXT_MODEL_PATH = MODELS_DIR / "text_model.joblib"
MESO_MODEL_PATH = MODELS_DIR / "meso4.h5"
CIFAKE_MODEL_PATH = MODELS_DIR / "cifake_model.h5"
# Operating point for the CIFAKE head. The trained CNN ranks well
# (test AUC 0.977) but its raw probabilities sit low: accuracy on 8k
# held-out test images is 0.754 at 0.50 vs 0.925 at 0.015 (real recall
# 0.924 / fake recall 0.925). cifake.py shifts the logit so this threshold
# lands on 0.5 for every downstream consumer. Override with TC_CIFAKE_THR.
CIFAKE_THRESHOLD = float(os.getenv("TC_CIFAKE_THR", "0.015"))

# # Optional: Google Fact Check Tools API key. When absent, verification falls
# # back to offline claim markers (see verify_service).
# FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY", "").strip()
# FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# # Optional: Google Programmable Search (CSE) - report §1's "Google Search
# # API" hook. Needs an API key plus the search engine id (cx). When both are
# # set, verification searches "<claim> fact check" and scores the snippets.
# GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY", "").strip()
# GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "").strip()
# GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

# Hybrid verification blend weights (must sum to 1.0 for the plain weighted
# average; they are normalized anyway inside verify_service.combine).
VERIFY_W_ML = float(os.getenv("TC_W_ML", "0.6"))
VERIFY_W_VER = float(os.getenv("TC_W_VER", "0.4"))

# Input guards.
MAX_TEXT_LENGTH = int(os.getenv("TC_MAX_TEXT", "20000"))
MAX_UPLOAD_BYTES = int(os.getenv("TC_MAX_UPLOAD", str(64 * 1024 * 1024)))  # 64 MB

# Video inference defaults.
VIDEO_EVERY_N = int(os.getenv("TC_VIDEO_EVERY_N", "15"))
VIDEO_MAX_FRAMES = int(os.getenv("TC_VIDEO_MAX_FRAMES", "30"))

# Crop to the detected face before MesoNet inference. MesoNet is trained on
# tight face crops, so feeding it full frames collapses accuracy.
FACE_CROP = os.getenv("TC_FACE_CROP", "1") not in ("0", "false", "False")
VIDEO_FACE_CROP = FACE_CROP  # backwards-compatible alias

# YuNet face detector (OpenCV DNN). Auto-downloaded once on first use; Haar
# is the fallback on OpenCV 4.x installs and full-frame when neither works.
FACE_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
FACE_MODEL_URL = os.getenv(
    "TC_FACE_MODEL_URL",
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
FACE_SCORE_THRESHOLD = float(os.getenv("TC_FACE_SCORE", "0.6"))

# Confidence gate: REAL/FAKE verdicts below this become UNKNOWN (the raw
# score stays visible in details.p_fake). Measured on the DFDC val cache:
#   0.50 -> 100% coverage, 0.65 acc | 0.55 -> 51%, 0.70 | 0.65 -> 12%, 0.73
# 0.55 keeps half the answers while cutting the least certain (and most
# often wrong) predictions; raise it for precision, lower to answer more.
MIN_CONFIDENCE = float(os.getenv("TC_MIN_CONF", "0.55"))
