from __future__ import annotations

from app.services.gating import apply_gate


def test_gate_passes_high_confidence():
    out = apply_gate({"verdict": "FAKE", "confidence": 0.91, "details": {"p_fake": 0.91}})
    assert out["verdict"] == "FAKE"
    assert out["confidence"] == 0.91


def test_gate_downgrades_low_confidence():
    out = apply_gate({"verdict": "REAL", "confidence": 0.52, "details": {"p_fake": 0.48}})
    assert out["verdict"] == "UNKNOWN"
    assert out["confidence"] == 0.0
    assert out["details"]["ungated_verdict"] == "REAL"
    assert out["details"]["p_fake"] == 0.48
    assert "low confidence" in out["details"]["reason"]


def test_gate_ignores_non_real_fake_verdicts():
    unknown = {"verdict": "UNKNOWN", "confidence": 0.0, "details": {}}
    error = {"verdict": "ERROR", "confidence": 0.0, "details": {}}
    assert apply_gate(unknown) == unknown
    assert apply_gate(error) == error


def test_gate_threshold_is_configurable():
    out = apply_gate(
        {"verdict": "FAKE", "confidence": 0.70, "details": {}},
        min_confidence=0.80,
    )
    assert out["verdict"] == "UNKNOWN"

    out = apply_gate(
        {"verdict": "FAKE", "confidence": 0.70, "details": {}},
        min_confidence=0.65,
    )
    assert out["verdict"] == "FAKE"


def test_gate_uses_config_default(monkeypatch):
    from app import config

    result = {"verdict": "REAL", "confidence": 0.60, "details": {"p_fake": 0.40}}
    monkeypatch.setattr(config, "MIN_CONFIDENCE", 0.90)
    assert apply_gate(result)["verdict"] == "UNKNOWN"
    monkeypatch.setattr(config, "MIN_CONFIDENCE", 0.55)
    assert apply_gate(result)["verdict"] == "REAL"
