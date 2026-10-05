"""Text misinformation detection.

Primary model: TF-IDF + logistic regression trained on the ISOT Fake News
dataset (see training/train_text.py), persisted as a joblib pipeline.

If the model file is missing (fresh clone, no training run yet) the service
falls back to a transparent keyword/surface heuristic so the API and the
extension keep working end to end. Results from the fallback are always
labelled in ``details`` so nobody mistakes them for model output.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from app.config import TEXT_MODEL_PATH

_model = None
_model_checked = False
_model_mtime: float | None = None
_load_error: str | None = None

# --- offline fallback heuristics -------------------------------------------

FAKE_PHRASES = [
    "you won't believe",
    "you wont believe",
    "shocking",
    "unbelievable",
    "secret they don't want",
    "doctors hate",
    "miracle cure",
    "cures cancer",
    "big pharma",
    "deep state",
    "chemtrail",
    "microchip",
    "5g causes",
    "flat earth",
    "stolen election",
    "wake up sheeple",
    "share before its deleted",
    "share before it's deleted",
    "going viral",
    "what happened next",
    "Exposed!",
    "hoax",
    "conspiracy",
    "mainstream media won't",
    "banned by the government",
    "they don't want you to know",
    "100% guaranteed",
    "act now",
]

CONSPIRACY_TERMS = [
    "hoax",
    "sheeple",
    "chemtrail",
    "microchip",
    "new world order",
    "false flag",
    "cover-up",
    "coverup",
    " rigged",
    "mind control",
]

ATTRIBUTION_PHRASES = [
    "according to",
    "reported by",
    "study published",
    "peer-reviewed",
    "researchers said",
    "in a statement",
    "the study found",
    "data from",
    "officials said",
    "the authors",
    "journal of",
    "university",
    "ministry of health",
    "world health organization",
    "cdc",
    "nasa",
]

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "with", "as", "at", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "it", "its", "this", "that", "these", "those", "has",
    "have", "had", "will", "would", "can", "could", "should", "may", "might",
    "do", "does", "did", "not", "no", "yes", "he", "she", "they", "we", "you",
    "i", "his", "her", "their", "our", "your", "my", "who", "whom", "what",
    "when", "where", "how", "why", "which", "than", "there", "here", "about",
    "into", "over", "after", "before", "up", "down", "out", "so", "such",
    "some", "any", "all", "more", "most", "other", "also", "said", "say",
    "says", "new", "one", "two",
}


def _get_model():
    """Lazily load the joblib pipeline; returns None when unavailable.

    The file's mtime is checked so a retrained model is picked up on the next
    request (as BUILD.md promises) without restarting the server.
    """
    global _model, _model_checked, _model_mtime, _load_error
    try:
        mtime = TEXT_MODEL_PATH.stat().st_mtime if TEXT_MODEL_PATH.exists() else None
    except OSError:
        mtime = None

    if _model is not None:
        if mtime == _model_mtime:
            return _model
        _model = None  # retrained on disk -> reload below
        _model_checked = False
    if _model_checked and _load_error and mtime is None:
        return None
    _model_checked = True
    if mtime is None:
        _load_error = f"model file not found: {TEXT_MODEL_PATH}"
        return None
    try:
        import joblib

        _model = joblib.load(TEXT_MODEL_PATH)
        _model_mtime = mtime
        _load_error = None
    except Exception as exc:  # pragma: no cover - depends on local artifacts
        _model = None
        _load_error = f"failed to load model: {exc}"
    return _model


def model_status() -> Dict[str, Any]:
    return {
        "loaded": _get_model() is not None,
        "path": str(TEXT_MODEL_PATH),
        "error": _load_error,
    }


def _heuristic_p_fake(text: str) -> Tuple[float, List[str]]:
    """Surface-feature fallback used when no trained model is present."""
    lowered = text.lower()
    signals: List[str] = []
    score = 0.5

    hits = [p for p in FAKE_PHRASES if p in lowered]
    if hits:
        step = min(0.05 * len(hits), 0.30)
        score += step
        signals.append(f"sensational phrases: {', '.join(hits[:4])} (+{step:.2f})")

    conspiracy = [t for t in CONSPIRACY_TERMS if t in lowered]
    if conspiracy:
        score += 0.10
        signals.append(f"conspiracy vocabulary: {', '.join(conspiracy[:4])} (+0.10)")

    words = re.findall(r"[A-Za-z']+", text)
    if len(words) >= 8:
        caps = [w for w in words if len(w) > 2 and w.isupper()]
        ratio = len(caps) / max(len(words), 1)
        if ratio > 0.3:
            score += 0.08
            signals.append(f"excessive capitalisation ({ratio:.0%}) (+0.08)")

    if re.search(r"!{2,}|\?{3,}|!\?", text):
        score += 0.07
        signals.append("excessive punctuation (+0.07)")

    if re.search(r"\b(click here|act now|limited time|repost|forward this)\b", lowered):
        score += 0.06
        signals.append("call-to-action / virality bait (+0.06)")

    attrs = [p for p in ATTRIBUTION_PHRASES if p in lowered]
    if attrs:
        step = min(0.04 * len(attrs), 0.16)
        score -= step
        signals.append(f"attribution cues: {', '.join(attrs[:4])} (-{step:.2f})")

    if re.search(r"\b\d{1,3}(\.\d+)?%", text):
        score -= 0.04
        signals.append("specific statistic cited (-0.04)")

    if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b", lowered):
        score -= 0.04
        signals.append("concrete date cited (-0.04)")

    if len(words) < 12:
        score += 0.05
        signals.append("too little context to verify (+0.05)")

    score = max(0.02, min(0.98, score))
    if not signals:
        signals.append("no decisive surface signals; neutral prior (0.50)")
    return score, signals


_CLF_NAMES = {
    "PassiveAggressiveClassifier": "passive_aggressive",
    "LogisticRegression": "logistic_regression",
    "SGDClassifier": "sgd",
}


def _model_name(model: Any) -> str:
    """Label the loaded pipeline (e.g. ``tfidf+passive_aggressive``).

    Derived from the artefact itself so the API never claims the wrong
    classifier when ``--clf`` flips between PAC and logreg.
    """
    try:
        est = model.steps[-1][1]
        if hasattr(est, "estimator") and not hasattr(est, "coef_"):
            est = est.estimator  # CalibratedClassifierCV wraps the base classifier
        name = type(est).__name__
        return f"tfidf+{_CLF_NAMES.get(name, name.lower())}"
    except Exception:
        return "tfidf+classifier"


def predict(text: str) -> Dict[str, Any]:
    """Classify a piece of text. Returns {verdict, confidence, details}."""
    text = (text or "").strip()
    if not text:
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "details": {"error": "empty text", "p_fake": 0.5},
        }

    model = _get_model()
    signals: List[str] = []
    if model is not None:
        p_fake = float(model.predict_proba([text])[0][1])
        model_name = _model_name(model)
        model_loaded = True
    else:
        p_fake, signals = _heuristic_p_fake(text)
        model_name = "offline-heuristic"
        model_loaded = False

    verdict = "FAKE" if p_fake >= 0.5 else "REAL"
    confidence = p_fake if verdict == "FAKE" else 1.0 - p_fake
    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "details": {
            "p_fake": round(p_fake, 4),
            "model": model_name,
            "model_loaded": model_loaded,
            "signals": signals,
            **({"model_error": _load_error} if not model_loaded and _load_error else {}),
        },
    }
