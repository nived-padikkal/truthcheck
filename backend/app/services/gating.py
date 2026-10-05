"""Confidence gate: refuse to guess when the model is not sure enough.

A near-0.5 score rendered as a confident-looking FAKE/REAL is worse than
admitting ignorance, so any REAL/FAKE verdict whose confidence falls below
``MIN_CONFIDENCE`` is downgraded to UNKNOWN. The raw score stays visible in
``details`` (``p_fake`` plus ``ungated_verdict``) for debugging and demos.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app import config


def apply_gate(result: Dict[str, Any], min_confidence: Optional[float] = None) -> Dict[str, Any]:
    """Downgrade REAL/FAKE verdicts whose confidence is below the threshold.

    ``min_confidence`` defaults to ``config.MIN_CONFIDENCE`` (env TC_MIN_CONF),
    read at call time so runtime overrides/tests take effect.
    """
    if result.get("verdict") not in ("REAL", "FAKE"):
        return result
    threshold = config.MIN_CONFIDENCE if min_confidence is None else min_confidence
    confidence = float(result.get("confidence", 0.0))
    if confidence >= threshold:
        return result
    details = dict(result.get("details") or {})
    details["ungated_verdict"] = result["verdict"]
    details["reason"] = (
        f"low confidence: {confidence:.2f} < {threshold:g} required; "
        f"model unsure (raw score kept in p_fake)"
    )
    return {"verdict": "UNKNOWN", "confidence": 0.0, "details": details}
