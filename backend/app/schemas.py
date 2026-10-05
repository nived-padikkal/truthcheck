"""Request/response models for the TruthCheck API."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

Verdict = Literal["REAL", "FAKE", "UNKNOWN", "ERROR"]


class TextIn(BaseModel):
    """Body for POST /analyze/text."""

    text: str = Field(..., min_length=1, max_length=20000, description="Claim or article text to classify.")
    verify: bool = Field(
        default=False,
        description="Blend the ML score with external fact-check/context verification.",
    )


class AnalysisResult(BaseModel):
    """Shape returned by every /analyze/* endpoint."""

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthOut(BaseModel):
    status: str = "ok"
    models: Dict[str, bool] = Field(default_factory=dict)
    version: Optional[str] = None
