from __future__ import annotations

from app.services import verify_service as vs


def test_search_backend_disabled_without_keys(monkeypatch):
    monkeypatch.setattr(vs, "GOOGLE_SEARCH_KEY", "")
    monkeypatch.setattr(vs, "GOOGLE_CSE_ID", "")
    assert vs._search_score("any claim") is None


def test_score_claim_falls_back_to_offline(monkeypatch):
    monkeypatch.setattr(vs, "FACT_CHECK_API_KEY", "")
    monkeypatch.setattr(vs, "GOOGLE_SEARCH_KEY", "")
    monkeypatch.setattr(vs, "GOOGLE_CSE_ID", "")
    result = vs.score_claim("the claim was debunked as misleading")
    assert result["method"] == "offline-markers"
    assert result["found"] is True
    assert result["score"] > 0.5  # contradiction cues push toward fake


def test_search_scores_snippet_language(monkeypatch):
    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "items": [
                    {"title": "Fact check: claim debunked as misleading",
                     "snippet": "The claim is false and misleading, experts said."},
                    {"title": "Reuters fact check", "snippet": "debunked"},
                ]
            }

    def fake_get(url, params=None, timeout=None):
        assert params["q"].endswith("fact check")
        return FakeResp()

    monkeypatch.setattr(vs, "GOOGLE_SEARCH_KEY", "k")
    monkeypatch.setattr(vs, "GOOGLE_CSE_ID", "cx")
    monkeypatch.setattr("requests.get", fake_get)

    result = vs._search_score("the moon is made of cheese")
    assert result is not None
    assert result["found"] is True
    assert result["method"] == "google-search"
    assert result["score"] > 0.5
    assert result["results_used"] == 2


def test_blend_weights_still_sum(monkeypatch):
    p = vs.combine(0.8, 0.2, w_ml=0.6, w_ver=0.4)
    assert abs(p - 0.56) < 1e-9  # 0.6*0.8 + 0.4*0.2
