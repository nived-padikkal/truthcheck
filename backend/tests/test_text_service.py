from __future__ import annotations

from app.services import text_service


def test_empty_text_is_unknown():
    result = text_service.predict("   ")
    assert result["verdict"] == "UNKNOWN"
    assert result["confidence"] == 0.0


def test_heuristic_flags_sensational_text():
    p_fake, signals = text_service._heuristic_p_fake(
        "SHOCKING!!! You won't believe this miracle cure. Share before it is deleted!"
    )
    assert p_fake > 0.5
    assert signals


def test_heuristic_accepts_attributed_text():
    p_fake, _ = text_service._heuristic_p_fake(
        "According to a study published in the Journal of Medicine, researchers "
        "reported that peer-reviewed data from 4,200 patients confirmed the "
        "result on March 4, 2024."
    )
    assert p_fake < 0.5


def test_heuristic_stays_in_bounds():
    for sample in ("a", "!!!", "BREAKING!!! " * 50, "word " * 500):
        p_fake, _ = text_service._heuristic_p_fake(sample)
        assert 0.0 <= p_fake <= 1.0


def test_model_status_shape():
    status = text_service.model_status()
    assert set(status) >= {"loaded", "path", "error"}


def test_model_name_reflects_loaded_classifier():
    model = text_service._get_model()
    if model is None:
        return  # offline-heuristic mode: nothing to label
    name = text_service._model_name(model)
    assert name.startswith("tfidf+")
    assert name == "tfidf+passive_aggressive" or name == "tfidf+logistic_regression"


def test_model_name_falls_back_for_unknown_objects():
    class Weird:
        pass

    assert text_service._model_name(Weird()) == "tfidf+classifier"
