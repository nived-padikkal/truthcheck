from __future__ import annotations

REAL_SAMPLE = (
    "According to a study published in the Journal of Medicine on May 3, 2023, "
    "researchers at Oxford University reported that a trial of 4,200 patients "
    "found the treatment reduced symptoms by 18 percent. Officials said the "
    "results were peer-reviewed and independent auditors confirmed the data."
)

FAKE_SAMPLE = (
    "SHOCKING!!! You won't believe this miracle cure that big pharma is hiding. "
    "Doctors hate it! The deep state doesn't want you to know - share before "
    "this is deleted!!! 100% guaranteed, act now!"
)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "text" in body["models"] and "meso4" in body["models"]


def test_analyze_text_shape(client):
    resp = client.post("/analyze/text", json={"text": REAL_SAMPLE})
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] in ("REAL", "FAKE")
    assert 0.0 <= body["confidence"] <= 1.0
    assert "p_fake" in body["details"]
    # report §7 target: verdict in well under 5 s
    assert body["details"]["latency_ms"] < 5000


def test_analyze_text_sensational_is_fake(client):
    body = client.post("/analyze/text", json={"text": FAKE_SAMPLE}).json()
    assert body["verdict"] == "FAKE"
    assert body["confidence"] >= 0.5
    assert body["details"]["signals"] or body["details"]["model_loaded"]


def test_analyze_text_attribution_is_real(client):
    body = client.post("/analyze/text", json={"text": REAL_SAMPLE}).json()
    assert body["verdict"] == "REAL"


def test_analyze_text_verify_flag_adds_details(client):
    claim = (
        "Fact check: the viral claim that the election was stolen is false and "
        "misleading - debunked by independent audits."
    )
    resp = client.post("/analyze/text", json={"text": claim, "verify": True})
    assert resp.status_code == 200
    details = resp.json()["details"]
    verification = details["verification"]
    assert verification["found"] is True
    assert 0.0 <= verification["score"] <= 1.0
    assert verification["method"]
    assert details["blended"] is True
    assert details["p_fake_ml"] is not None


def test_analyze_text_empty_rejected(client):
    resp = client.post("/analyze/text", json={"text": ""})
    assert resp.status_code == 422


def test_analyze_image_garbage_bytes(client):
    resp = client.post(
        "/analyze/image",
        files={"file": ("junk.png", b"this is not an image", "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "UNKNOWN"
    assert "error" in body["details"]


def test_analyze_image_decodes_png(client):
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, format="PNG")
    resp = client.post(
        "/analyze/image",
        files={"file": ("red.png", buf.getvalue(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["details"]["image_size"] == [64, 64]
    if body["details"].get("model_loaded"):
        assert body["verdict"] in ("REAL", "FAKE", "UNKNOWN")
        if body["verdict"] == "UNKNOWN":
            # confidence gate: low-margin scores must not look decisive
            assert "low confidence" in body["details"]["reason"]
            assert body["details"]["ungated_verdict"] in ("REAL", "FAKE")
            assert 0.0 <= body["details"]["p_fake"] <= 1.0
    else:
        assert body["verdict"] == "UNKNOWN"


def test_analyze_image_gated_when_uncertain(client, monkeypatch):
    from io import BytesIO

    from PIL import Image

    # force the gate open at 1.0 so any model output must be downgraded
    from app import config

    monkeypatch.setattr(config, "MIN_CONFIDENCE", 1.0)

    buf = BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, format="PNG")
    body = client.post(
        "/analyze/image",
        files={"file": ("red.png", buf.getvalue(), "image/png")},
    ).json()
    if body["details"].get("model_loaded"):
        assert body["verdict"] == "UNKNOWN"
        assert body["confidence"] == 0.0
        assert "reason" in body["details"]


def test_analyze_video_garbage_bytes(client):
    resp = client.post(
        "/analyze/video",
        files={"file": ("junk.mp4", b"not a video at all", "video/mp4")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "UNKNOWN"
    assert "error" in body["details"]


def test_missing_file_rejected(client):
    assert client.post("/analyze/image").status_code == 422
    assert client.post("/analyze/video").status_code == 422
