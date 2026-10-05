from __future__ import annotations

from app.services import verify_service


def test_combine_weights():
    assert verify_service.combine(1.0, 0.0) == 0.6
    assert verify_service.combine(0.0, 1.0) == 0.4
    assert verify_service.combine(0.5, 0.5) == 0.5
    assert verify_service.combine(1.0, 1.0, w_ml=0.5, w_ver=0.5) == 1.0


def test_extract_keywords():
    keywords = verify_service.extract_keywords(
        'The "moon landing" was faked according to a report about moon landing evidence'
    )
    assert keywords
    assert "moon landing" in keywords


def test_offline_score_contradiction():
    result = verify_service.score_claim(
        "Fact check: the viral claim that vaccines cause autism is false and misleading."
    )
    assert result["found"] is True
    assert result["score"] > 0.5
    assert result["contradicting"]


def test_offline_score_corroboration():
    result = verify_service.score_claim(
        "The peer-reviewed study published in the Journal of Medicine was "
        "verified by an independent audit using official data."
    )
    assert result["found"] is True
    assert result["score"] < 0.5
    assert result["corroborating"]


def test_offline_score_no_signal():
    result = verify_service.score_claim("Cats enjoy sleeping in boxes.")
    assert result["found"] is False
    assert result["method"] == "offline-markers"


def test_apply_keeps_ml_verdict_without_signal():
    ml = {"verdict": "FAKE", "confidence": 0.9, "details": {"p_fake": 0.9}}
    out = verify_service.apply(ml, "Cats enjoy sleeping in boxes.")
    assert out["verdict"] == "FAKE"
    assert out["confidence"] == 0.9
    assert out["details"]["verification"]["found"] is False


def test_apply_blends_when_signal_exists():
    ml = {"verdict": "FAKE", "confidence": 0.55, "details": {"p_fake": 0.55}}
    out = verify_service.apply(ml, "Fact check: this claim is false and debunked.")
    assert out["details"]["blended"] is True
    assert out["details"]["p_fake"] > 0.55
    assert out["verdict"] == "FAKE"


def test_apply_ignores_unknown_results():
    ml = {"verdict": "UNKNOWN", "confidence": 0.0, "details": {"p_fake": 0.5}}
    out = verify_service.apply(ml, "Fact check: false.")
    assert out["verdict"] == "UNKNOWN"
    assert out["confidence"] == 0.0
