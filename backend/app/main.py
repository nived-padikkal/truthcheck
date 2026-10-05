"""TruthCheck API - FastAPI entry point.

Run:  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import (
    CIFAKE_MODEL_PATH,
    MAX_TEXT_LENGTH,
    MAX_UPLOAD_BYTES,
    MESO_MODEL_PATH,
)
from app.schemas import AnalysisResult, HealthOut, TextIn
from app.services import image_service, text_service, verify_service, video_service

logger = logging.getLogger("truthcheck")

app = FastAPI(
    title="TruthCheck API",
    version=__version__,
    description="AI-based multimedia misinformation detection: text, image and video.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your extension origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error_result(exc: Exception, stage: str) -> dict:
    logger.exception("error during %s", stage)
    return {
        "verdict": "ERROR",
        "confidence": 0.0,
        "details": {"error": f"{type(exc).__name__}: {exc}", "stage": stage},
    }


def _with_latency(result: Dict[str, Any], started: float) -> Dict[str, Any]:
    """Stamp wall-clock latency so the <5 s target (report §7) is measurable."""
    details = dict(result.get("details") or {})
    details["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    out = dict(result)
    out["details"] = details
    return out


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    """Liveness probe. ``models`` reports which artifacts are in place
    (the MesoNet/CIFAKE checks are file presence only, so it stays fast)."""
    return HealthOut(
        status="ok",
        version=__version__,
        models={
            "text": text_service.model_status()["loaded"],
            "meso4": MESO_MODEL_PATH.exists(),
            "cifake": CIFAKE_MODEL_PATH.exists(),
        },
    )


@app.post("/analyze/text", response_model=AnalysisResult)
def analyze_text(body: TextIn) -> AnalysisResult:
    """Classify text as REAL/FAKE. Set ``verify=true`` to blend in the
    hybrid external verification signal."""
    started = time.perf_counter()
    try:
        if len(body.text) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=413, detail=f"text longer than {MAX_TEXT_LENGTH} characters")
        result = text_service.predict(body.text)
        if body.verify:
            result = verify_service.apply(result, body.text)
        return _with_latency(result, started)
    except HTTPException:
        raise
    except Exception as exc:
        return _with_latency(_error_result(exc, "text"), started)


@app.post("/analyze/image", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)) -> AnalysisResult:
    """Classify a single image (multipart field name: ``file``)."""
    started = time.perf_counter()
    try:
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="file too large")
        return _with_latency(image_service.predict(data), started)
    except HTTPException:
        raise
    except Exception as exc:
        return _with_latency(_error_result(exc, "image"), started)


@app.post("/analyze/video", response_model=AnalysisResult)
async def analyze_video(
    file: UploadFile = File(...),
    every_n: int = Form(15),
    max_frames: int = Form(30),
) -> AnalysisResult:
    """Classify a video by sampling frames through MesoNet.

    Multipart field name: ``file``. Optional form fields: ``every_n``
    (sample stride, default 15) and ``max_frames`` (default 30)."""
    started = time.perf_counter()
    try:
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="file too large")
        every_n = max(1, min(every_n, 600))
        max_frames = max(1, min(max_frames, 240))
        return _with_latency(
            video_service.predict(data, every_n=every_n, max_frames=max_frames),
            started,
        )
    except HTTPException:
        raise
    except Exception as exc:
        return _with_latency(_error_result(exc, "video"), started)
