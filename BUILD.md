# TruthCheck — Build Guide

**AI-based multimedia misinformation detection** as a Chrome extension (Manifest V3)
with a FastAPI backend.

> Team SyntaxX · Sreepathy Institute of Management and Technology
> **TRUTHCHECK: VERIFY BEFORE YOU TRUST.**

Select suspicious **text, an image, or a video** on any page → right-click →
**TruthCheck** → an on-page overlay reports:

```
Verdict: FAKE
Confidence: 93%
```

---

## 1. Architecture

```
User selects content
        |
Chrome Extension (content.js + background.js + popup)
        |  POST /analyze/{text|image|video}
        v
FastAPI backend (backend/app)
   |- text_service      TF-IDF + logistic regression (scikit-learn)
   |- image_service     MesoNet Meso4 (TensorFlow) - single frame
   |- video_service     frame sampling -> MesoNet -> mean score
   '- verify_service    hybrid verification (fact-check API / offline markers)
        |
JSON {verdict, confidence, details}
        |
Extension shows Real/Fake + confidence + evidence
```

## 2. Project layout

```
hackathon/
|- extension/
|  |- manifest.json          MV3 manifest (permissions, popup, content script)
|  |- background.js          context menus + backend calls + badge
|  |- content.js             on-page result overlay
|  |- popup.html/css/js      paste text, cross-check toggle, settings
|  `- icons/                 icon16/48/128.png
|- verify_meso.py            end-to-end image/video verification (§8)
|- backend/
|  |- app/
|  |  |- main.py             FastAPI routes (+ latency_ms per request)
|  |  |- schemas.py          request/response models
|  |  |- config.py           paths, limits, tunable weights (env vars)
|  |  |- services/
|  |  |  |- text_service.py      text classifier + offline fallback
|  |  |  |- image_service.py     ensemble: CIFAKE + Meso4 per image
|  |  |  |- video_service.py     frame extraction + face crop + inference
|  |  |  |- verify_service.py    hybrid verification / score blending
|  |  |  |- mesonet.py           Meso4 architecture + lazy loader
|  |  |  |- cifake.py            CIFAKE CNN architecture + lazy loader
|  |  |  |- facecrop.py          YuNet/Haar face detector (shared)
|  |  |  `- gating.py            confidence gate (low-margin -> UNKNOWN)
|  |  `- models/             text_model.joblib, meso4.h5, cifake_model.h5,
|  |                         face_detection_yunet_2023mar.onnx (all gitignored)
|  |- training/
|  |  |- train_text.py       ISOT -> TF-IDF + Passive-Aggressive (or logreg)
|  |  |- train_mesonet.py    videos or JPEG cache -> Meso4 weights
|  |  |- train_cifake.py     CIFAKE (HF mirror) -> AI-image CNN
|  |  `- prepare_dfdc_crops.py  download DFDC face crops + build JPEG cache
|  |- data/                  datasets (gitignored)
|  |- tests/                 pytest suite
|  |- requirements.txt
|  `- requirements-dev.txt
`- BUILD.md
```

## 3. Setup — backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest + httpx + pyarrow, for tests/training
uvicorn app.main:app --reload --port 8000
```

Check it:

```bash
curl http://localhost:8000/health
# {"status":"ok","models":{"text":true,"meso4":true,"cifake":true},"version":"0.1.0"}
```

**Python version note:** TensorFlow ships wheels for Python 3.10–3.14 (verify
yours with `pip install tensorflow`). `requirements.txt` marks TensorFlow so it
is skipped automatically on interpreters without wheels — the API still boots
and every endpoint still answers; image/video simply return `UNKNOWN` until the
models are available.

**OpenCV 5 note:** OpenCV 5 removed `CascadeClassifier` and the Haar XML data.
Face detection therefore uses YuNet (`cv2.FaceDetectorYN`), whose ~230 KB ONNX
model auto-downloads once to `app/models/` — Haar remains only as a fallback
on OpenCV 4.x installs.

## 4. Training

### 4.1 Text model (ISOT)

1. Get the [ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
   (Kaggle) — download and place `Fake.csv` and `True.csv` in `backend/data/`.
   Mirrors work too: the trainer accepts `title`/`text` columns (title optional)
   and either a file-of-origin or a `label`/`type` column (1 = fake, 0 = real).
2. Train:

```bash
cd backend
python training/train_text.py --data-dir data            # Passive-Aggressive (default, report §4.3)
python training/train_text.py --data-dir data --clf logreg
```

Expected: ~**0.99 accuracy** on the held-out split (measured: **0.9886** for
`pac`, 0.9853 for `logreg`); model saved to
`backend/app/models/text_model.joblib`. The API reloads the file on the next
request when its mtime changes (no restart needed).

The head is `PassiveAggressiveClassifier`, wrapped in `CalibratedClassifierCV`
because PA exposes only a decision function while the API needs
`predict_proba` for its confidence. `scikit-learn` is pinned `<1.10` because
the PA class is scheduled for removal there (retrain with
`SGDClassifier(loss='hinge', learning_rate='pa1')` before moving past it).

**Source-bias caveat:** almost every REAL ISOT article is Reuters copy
(`WASHINGTON (Reuters) - ...`) while FAKE articles are not. `train_text.py`
strips agency leads/tokens (`strip_source_bias`) so the model learns content
instead of publisher style — otherwise it collapses on real-world text. This is
also why hybrid verification (§6) exists.

### 4.2 MesoNet (deepfakes, images + video)

**Quick path (what produced the demo weights) — DFDC face crops, no access
request needed:**

```bash
cd backend
python training/prepare_dfdc_crops.py --per-class 166   # ~1.5 GB download, resumable
python training/train_mesonet.py --from-cache --epochs 40 --patience 10
```

`prepare_dfdc_crops.py` pulls labelled `.npz` face-crop files from the Hugging
Face mirror `Shironx/DFDC_Opencv_Face_Crops` (Facebook Deepfake Detection
Challenge crops, MIT licence), balances real/fake, and writes 256×256 JPEGs
into `data/mesonet_cache/{train,val}/{real,fake}/`. The split is **by source
file (= video)**, never by frame. `train_mesonet.py --from-cache` then trains
Meso4 straight from that cache.

**Reference path — FaceForensics++ (higher accuracy possible):**

1. Request dataset access: <https://github.com/ondyari/FaceForensics>
   (accept the terms, then use their download script — e.g. 720p `c23`).
2. Train:

```bash
cd backend
python training/train_mesonet.py --data-dir data/faceforensicsplusplus \
    --every-n 5 --epochs 40 --patience 10
```

The script auto-labels clips from folder names (`original/`/`real/` vs
`deepfakes/`/`fake/`/...), **splits by video id** (frame-level splits leak
near-duplicates and inflate accuracy), extracts every Nth frame, crops the
largest face with OpenCV's Haar cascade, caches 256×256 JPEGs, trains Meso4,
prints accuracy/precision/recall/AUC, and saves
`backend/app/models/meso4.h5`.

> **Honest demo metrics:** the shipped `meso4.h5` was trained on the small
> DFDC subset above (166 videos/side, 10 crops/video ≈ 3.3k frames, split by
> video) and reaches **val accuracy 0.666, AUC 0.707** on 500 held-out frames
> (REAL recall 0.62 / FAKE recall 0.72); a live API spot-check scored 80/120
> correct on unseen frames. That is genuinely better than chance but far below
> published MesoNet numbers on FaceForensics++ — use §4.2's reference path for
> a stronger model, and say so in the demo. FaceForensics++ weights are a
> drop-in replacement (same architecture, same file path).

If the weights file is absent (fresh clone), `/analyze/image` and
`/analyze/video` return `verdict: UNKNOWN` with `details.reason` telling you
exactly what is missing — frame extraction, metadata, and sampling still run,
so you can validate the pipeline before spending GPU hours.

### 4.3 Image model (CIFAKE, AI-generated images)

Detects **AI-generated** images (Stable Diffusion etc.) rather than face
swaps — report §4.2. The Kaggle upload needs an account, so the trainer pulls
the identical Hugging Face mirror (`dragonintelligence/CIFAKE-image-dataset`,
100k train / 20k test, no credentials, ~50 MB):

```bash
cd backend
python training/train_cifake.py --download --epochs 15
```

Saves `backend/app/models/cifake_model.h5` (32×32 inputs, the dataset's native
size; `1 = AI-generated`, like every other model here). The API picks it up on
the next `/analyze/image` request and reports it in `/health` as `cifake`.

Held-out test metrics: **AUC 0.977**, 92.5% accuracy at the shipped operating
point (real recall 0.924 / AI recall 0.925) — but only 75.4% at a naive 0.50
threshold, because the CNN ranks well while emitting probabilities biased low.
`cifake.prior_shift()` therefore logit-shifts the raw score so `TC_CIFAKE_THR`
(default `0.015`, found by sweeping the held-out set) becomes the 0.5 decision
point for every downstream consumer. `/analyze/image` reports both sides:
`checks[].p_raw` is the raw CNN output, `checks[].p_fake` the calibrated one.
Re-run the sweep after retraining (the trainer prints the recommendation) and
pass it as `TC_CIFAKE_THR` if it differs.

CIFAKE images are CIFAR-10 sized, so natural photos are out of its training
domain — measured on 150 real video frames in RGB (the service's feed), the
median raw score is 0.0005, but 7% exceed 0.5 and 2% exceed 0.95. Treat its
vote as one signal in the ensemble (§7), not gospel.

## 5. Setup — extension

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. **Load unpacked** → select the `extension/` folder
4. Start the backend (§3), then reload the extension

Right-click anything:

| You select | Menu item | Backend call |
|---|---|---|
| Selected text | TruthCheck: analyze text | `POST /analyze/text` (with verification) |
| An image | TruthCheck: analyze image | downloads the file → `POST /analyze/image` |
| A `<video>` | TruthCheck: analyze video | downloads the file → `POST /analyze/video` |

Results appear as an overlay on the page (12 s auto-dismiss, `×` to close) and
the toolbar badge shows `FAKE` / `REAL` / `?` / `ERR`.

The popup accepts pasted text, has a **Use selection** button (reads the page's
selection), a **cross-check with fact-check sources** toggle, and an API URL
field (stored in `chrome.storage`) for pointing at a remote backend.

## 6. Hybrid verification

`POST /analyze/text` with `{"verify": true}` blends the ML probability with an
external evidence score:

```
p_fake = 0.6 * p_fake(MR) + 0.4 * verification_score
verification_score: 0 = corroborated, 1 = contradicted / unsupported
```

Backends, picked automatically (first configured wins):

1. **Google Fact Check Tools API** — set `FACT_CHECK_API_KEY` to enable real
   claim lookups (ratings are mapped to a 0–1 truth scale, then inverted).
2. **Google Search (Programmable Search)** — set `GOOGLE_SEARCH_KEY` +
   `GOOGLE_CSE_ID` to search `<claim> fact check` and score the language in
   the returned titles/snippets (report §1's "Google Search API" hook).
3. **Offline markers** (default, no network) — contradiction cues
   ("fact check", "debunked", "misleading", ...) push toward FAKE, corroboration
   cues ("peer-reviewed", "published in", ...) push toward REAL. Transparent and
   always reported in `details.verification`.

If verification finds no signal it changes nothing and says so. Weights are
tunable via `TC_W_ML` / `TC_W_VER` — validate any change on a held-out set.

## 7. API

| Method | Route | Input | Output |
|---|---|---|---|
| GET | `/health` | — | `{"status":"ok","models":{...}}` |
| POST | `/analyze/text` | JSON `{"text":"...","verify":bool}` | verdict + confidence |
| POST | `/analyze/image` | multipart `file` | verdict + confidence |
| POST | `/analyze/video` | multipart `file` + form `every_n`, `max_frames` | verdict + confidence |

All analyze routes return:

```json
{ "verdict": "FAKE", "confidence": 0.93,
  "details": { "p_fake": 0.93, "model": "tfidf+passive_aggressive",
               "latency_ms": 41.2 } }
```

`verdict` ∈ `REAL | FAKE | UNKNOWN | ERROR` (UNKNOWN = not enough evidence /
model missing / gated; ERROR = request failed — `details.error` says why).
`details.latency_ms` makes the <5 s target (report §7) measurable per request.

**Image ensemble (§2's result aggregator).** `/analyze/image` runs every
installed engine and reports each vote in `details.checks`:

| Engine | Detects | Input |
|---|---|---|
| `cifake` | AI-generated images | whole frame @ 32×32 |
| `meso4` | face-swap deepfakes | largest face @ 256×256 |

The engines answer *different* questions — `meso4` "is this a face swap?",
`cifake` "is this an AI-generated picture?" — so instead of demanding
literal agreement (CIFAKE correctly calls face swaps REAL for *its* question;
that rule collapsed DFDC fake-frame accuracy from 0.61 to 0.15) the merge
picks the **domain-primary** engine: face detected → `meso4`, else `cifake`;
one engine installed → it decides alone. The other engine runs as an
**advisory** check (`details.advisory`) and may only *veto* a `REAL` verdict
when it is natively sure (`p_raw >= 0.95`, true for ~2% of real frames) →
`UNKNOWN` with the reason in `details.reason`. On agreement the confidence
of the deciding engine is blended with the advisory's (mean).

**Confidence gate.** Any `REAL`/`FAKE` whose confidence is below
`TC_MIN_CONF` (default `0.55`) comes back `UNKNOWN` with the raw score kept
in `details.p_fake` and `details.ungated_verdict` — measured trade-off on the
DFDC val cache: `0.50 → 100% coverage / 0.65 acc`, `0.55 → 51% / 0.70`,
`0.65 → 12% / 0.73`. Set `TC_MIN_CONF=0.50` to always answer.

**Environment variables:** `TC_MAX_TEXT`, `TC_MAX_UPLOAD`, `TC_VIDEO_EVERY_N`,
`TC_VIDEO_MAX_FRAMES`, `TC_FACE_CROP`, `TC_FACE_SCORE`, `TC_FACE_MODEL_URL`,
`TC_MIN_CONF`, `TC_CLF`, `TC_CIFAKE_THR`, `TC_W_ML`, `TC_W_VER`, `FACT_CHECK_API_KEY`,
`GOOGLE_SEARCH_KEY`, `GOOGLE_CSE_ID`.

```bash
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text":"SHOCKING!!! You won't believe this...", "verify": true}'

curl -X POST http://localhost:8000/analyze/image -F "file=@photo.jpg"
curl -X POST http://localhost:8000/analyze/video -F "file=@clip.mp4" -F "every_n=15"
```

## 8. Tests and verification

```bash
cd backend
python -m pytest tests -q
```

Covers: health shape, text verdicts (sensational → FAKE, attributed → REAL,
valid with or without a trained model), verification blending and its three
backends, image ensemble merge (primary/advisory veto, native-score cutoff,
gate), face cropping, image and
video error handling, decode paths, and `verify_service` blend math.

End-to-end numbers (report §8) come from the repo-root script — 60 real + 60
fake held-out frames through `/analyze/image`, then a synthesized MP4 through
`/analyze/video`:

```bash
python verify_meso.py                 # human-readable
python verify_meso.py --json          # machine-readable
```

It reports **raw model accuracy** (ungated), **gate coverage** and **gated
accuracy** separately, so the confidence gate is never mistaken for model
quality.

## 9. Demo checklist

- [ ] Backend running: `uvicorn app.main:app --reload --port 8000`
- [ ] `/health` reports `"text": true`, `"meso4": true` and `"cifake": true`
- [ ] `python verify_meso.py` prints accuracy/coverage + video verdicts
- [ ] Extension loaded unpacked; popup status dot is green
- [ ] Select a claim on a news site → right-click → **TruthCheck: analyze text**
- [ ] Overlay shows **Verdict + Confidence + evidence lines**
- [ ] Popup: paste text, toggle cross-check, see blended score + signals
- [ ] Right-click an image/video → REAL/FAKE + confidence (+ frame counts)
- [ ] `details.latency_ms` under 5000 on the demo machine

## 10. Known limitations (be upfront in the demo)

- Confidence is the model's probability, **not** a guarantee of truth.
- ISOT-trained text models can overfit to publisher style — we strip the worst
  cues, but evaluate on source-diverse data before trusting it.
- MesoNet is lightweight and works best on face-swap deepfakes like its
  training data; newer generative video may evade it. It judges faces, not
  whether a *story* is true. Measured: val acc 0.666 / AUC 0.707 on the small
  DFDC cache; `verify_meso.py` prints today's raw/gated numbers.
- CIFAKE is trained on 32×32 CIFAR-sized images — full-resolution photos are
  out of its domain, so read its vote together with `meso4` in `details.checks`.
- The confidence gate returns `UNKNOWN` on purpose (see §7); it is a honesty
  feature, not a bug, and is tunable via `TC_MIN_CONF`.
- Cross-origin media fetch can fail; the extension reports `ERROR` with the
  cause instead of hanging (`<all_urls>` host permission exists exactly for this).
- Unknown/short inputs return tempered or `UNKNOWN` answers — say so in the demo.
- Never send private user content to a remote server without clear consent.

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| Popup dot red | Start the backend; check the API URL in the popup footer |
| `verdict: UNKNOWN` on images/video | Install TensorFlow + train MesoNet (§4.2) |
| `model_loaded: false` on text | Run `training/train_text.py` (§4.1) |
| `cifake: false` in `/health` | Run `training/train_cifake.py --download` (§4.3) |
| `face_detected: false` everywhere | YuNet ONNX missing (no network) — set `TC_FACE_MODEL_URL` or drop the file into `app/models/`; check `details.face_detector` |
| `UNKNOWN` + `"low confidence"` | Gate says the model is unsure — raise answers with `TC_MIN_CONF=0.50` |
| `UNKNOWN` + `"flags AI generation"` | Advisory CIFAKE is natively sure (`p_raw >= 0.95`) while the primary says REAL — read `details.advisory`; fires on ~2% of real frames |
| Text model changes not picked up | mtime-based reload is automatic; restart the server if the venv changed |
| Image/video `ERROR` | Media blocked by CORS/site — open the media directly or check `uvicorn.log` |
| Video analysis slow | Raise `every_n`, lower `max_frames` (form fields) or env `TC_VIDEO_EVERY_N` |
| Extension changes not applied | Hit reload on `chrome://extensions`, refresh the tab |
