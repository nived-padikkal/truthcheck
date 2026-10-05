from __future__ import annotations

from app.services.image_service import _merge
from app.services import cifake


def _check(model: str, p_fake: float, face_detected: bool | None = None) -> dict:
    c = {"model": model, "task": "t", "p_fake": p_fake}
    if face_detected is not None:
        c["face_detected"] = face_detected
    return c


def test_merge_agreement_takes_mean_confidence():
    out = _merge([_check("meso4", 0.8, face_detected=True), _check("cifake", 0.9)])
    assert out["verdict"] == "FAKE"
    assert abs(out["confidence"] - 0.85) < 1e-9
    assert out["details"]["merged_p_fake"] == 0.85


def test_merge_agreement_real():
    out = _merge([_check("meso4", 0.3, face_detected=True), _check("cifake", 0.1)])
    assert out["verdict"] == "REAL"
    assert abs(out["confidence"] - 0.8) < 1e-9  # mean of 0.7 and 0.9


def test_merge_face_present_meso_decides_despite_cifake():
    # DFDC fake frame: meso sees the swap, cifake (a diffusion detector) says REAL
    out = _merge([_check("meso4", 0.7, face_detected=True), _check("cifake", 0.1)])
    assert out["verdict"] == "FAKE"
    assert out["confidence"] == 0.7  # disagreeing advisory does not dilute
    assert out["details"]["deciding_model"] == "meso4"
    assert out["details"]["advisory"]["vote"] == "REAL"


def test_merge_faceless_cifake_decides():
    out = _merge([_check("meso4", 0.2, face_detected=False), _check("cifake", 0.9)])
    assert out["verdict"] == "FAKE"
    assert out["confidence"] == 0.9
    assert out["details"]["deciding_model"] == "cifake"


def test_merge_advisory_vetoes_confident_real():
    out = _merge([
        _check("meso4", 0.2, face_detected=True),
        {"model": "cifake", "task": "t", "p_fake": 0.995, "p_raw": 0.99},
    ])
    assert out["verdict"] == "UNKNOWN"
    assert out["confidence"] == 0.0
    assert "flags AI generation" in out["details"]["reason"]
    assert "cifake" in out["details"]["reason"]


def test_merge_veto_uses_native_score_not_calibrated():
    # calibrated 0.97 == raw 0.27: cifake is not *natively* sure, no veto
    out = _merge([
        _check("meso4", 0.2, face_detected=True),
        {"model": "cifake", "task": "t", "p_fake": 0.97, "p_raw": 0.27},
    ])
    assert out["verdict"] == "REAL"
    assert out["confidence"] == 0.8
    assert out["details"]["advisory"]["vote"] == "FAKE"
    assert out["details"]["advisory"]["p_raw"] == 0.27


def test_merge_advisory_below_veto_does_not_block_real():
    out = _merge([_check("meso4", 0.1, face_detected=True), _check("cifake", 0.7)])
    assert out["verdict"] == "REAL"  # 0.7 < 0.95 veto threshold
    assert out["details"]["advisory"]["vote"] == "FAKE"


def test_merge_fake_is_never_vetoed():
    out = _merge([_check("meso4", 0.7, face_detected=True), _check("cifake", 0.02)])
    assert out["verdict"] == "FAKE"
    assert out["confidence"] == 0.7


def test_merge_single_check_is_that_check():
    out = _merge([_check("meso4", 0.7, face_detected=True)])
    assert out["verdict"] == "FAKE"
    assert out["confidence"] == 0.7
    assert out["details"]["p_fake"] == 0.7
    assert out["details"]["deciding_model"] == "meso4"
    assert "advisory" not in out["details"]


def test_cifake_status_shape_without_model(tmp_path, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "CIFAKE_MODEL_PATH", tmp_path / "missing.h5")
    monkeypatch.setattr(cifake, "CIFAKE_MODEL_PATH", tmp_path / "missing.h5")
    cifake._model = None
    try:
        status = cifake.model_status()
        assert status["loaded"] is False
        assert "not found" in (status["error"] or "")
    finally:
        cifake._model = None


def test_prior_shift_maps_threshold_to_half():
    assert abs(cifake.prior_shift(0.015, 0.015) - 0.5) < 1e-9
    assert abs(cifake.prior_shift(0.5, 0.5) - 0.5) < 1e-9


def test_prior_shift_is_monotone_and_keeps_extremes():
    lo, mid, hi = cifake.prior_shift(0.001), cifake.prior_shift(0.02), cifake.prior_shift(0.9)
    assert lo < mid < hi
    assert lo < 0.5 < hi
    assert cifake.prior_shift(0.0) < 0.001  # clipped, still clearly REAL
    assert cifake.prior_shift(1.0) > 0.999


def test_prior_shift_disabled_out_of_range():
    assert cifake.prior_shift(0.3, threshold=1.0) == 0.3
    assert cifake.prior_shift(0.3, threshold=0.0) == 0.3
