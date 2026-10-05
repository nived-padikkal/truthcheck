"""Hybrid verification: cross-check a claim against external evidence and
blend that signal with the ML probability.

Three backends, picked automatically (first configured wins):

1. **Google Fact Check Tools API** - used when the ``FACT_CHECK_API_KEY``
   environment variable is set. Real claims from fact-check publishers,
   reduced to a 0..1 "contradiction" score.
2. **Google Search (Programmable Search / CSE)** - used when
   ``GOOGLE_SEARCH_KEY`` + ``GOOGLE_CSE_ID`` are set (report §1's "Google
   Search API" hook): searches ``<claim> fact check`` and scores the
   language found in the returned titles/snippets.
3. **Offline claim markers** - no network, no key. Looks for debunk language
   and corroboration cues in the text itself. Crude, but it keeps the hybrid
   pipeline alive in a demo without secrets.

Blend rule (see BUILD.md): ``verification_score`` is 0 when the claim is well
corroborated and 1 when it is contradicted / unsupported:

    p_fake = w_ml * ml_p_fake + w_ver * verification_score
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.config import (
    FACT_CHECK_API_KEY,
    FACT_CHECK_URL,
    GOOGLE_CSE_ID,
    GOOGLE_SEARCH_KEY,
    GOOGLE_SEARCH_URL,
    VERIFY_W_ML,
    VERIFY_W_VER,
)

CONTRADICTION_MARKERS = [
    "fact check",
    "fact-check",
    "false claim",
    "the claim is false",
    "claims is false",
    "misleading",
    "misleadingly",
    "debunked",
    "debunk",
    "misattributed",
    "out of context",
    "false context",
    "satire",
    "satirical",
    "conspiracy theory",
    "disinformation",
    "doctored",
    "photoshopped",
    "no evidence",
    "unsupported claim",
    "actually false",
    "is not true",
]

CORROBORATION_MARKERS = [
    "peer-reviewed",
    "peer reviewed",
    "study published",
    "published in the",
    "systematic review",
    "meta-analysis",
    "official data",
    "government data",
    "according to the world health organization",
    "according to nasa",
    "reuters reported",
    "associated press",
    "verified by",
    "independent audit",
    "preprint",
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


def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    """Top content words by frequency, quoted phrases kept intact."""
    phrases = re.findall(r'"([^"]{3,80})"', text) or re.findall(r"'([^']{3,80})'", text)
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
    freq: Dict[str, int] = {}
    for w in words:
        if w in _STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq, key=lambda w: (-freq[w], w))
    return (phrases + ranked)[:top_n]


def _offline_score(text: str) -> Dict[str, Any]:
    lowered = text.lower()
    contradicting = [m for m in CONTRADICTION_MARKERS if m in lowered]
    corroborating = [m for m in CORROBORATION_MARKERS if m in lowered]
    found = bool(contradicting or corroborating)

    score = 0.5
    if contradicting:
        score += min(0.15 * len(contradicting), 0.45)
    if corroborating:
        score -= min(0.15 * len(corroborating), 0.45)
    score = max(0.0, min(1.0, score))

    return {
        "found": found,
        "score": round(score, 4),
        "method": "offline-markers",
        "contradicting": contradicting,
        "corroborating": corroborating,
        "keywords": extract_keywords(text),
        "explanation": (
            "Offline marker check: contradiction/corroboration phrases found in "
            "the text. Set FACT_CHECK_API_KEY (or GOOGLE_SEARCH_KEY + "
            "GOOGLE_CSE_ID) for live lookups."
            if found
            else "No external or in-text evidence found; verification skipped."
        ),
    }


def _parse_rating(rating: Dict[str, Any]) -> Optional[float]:
    """Map a schema.org ClaimReview rating to truth in [0, 1]."""
    if not rating:
        return None

    def as_float(value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    value = as_float(rating.get("ratingValue"))
    if value is not None:
        best = as_float(rating.get("bestRating"))
        worst = as_float(rating.get("worstRating"))
        if best is not None and worst is not None and best != worst:
            return max(0.0, min(1.0, (value - worst) / (best - worst)))
        if 0.0 <= value <= 1.0:
            return value
        if 0.0 <= value <= 5.0:
            return value / 5.0

    blob = " ".join(
        str(rating.get(k, "")) for k in ("ratingValue", "alternateName", "ratingModality")
    ).lower()
    if not blob.strip():
        return None
    if any(k in blob for k in ("true", "verified", "correct", "accurate", "yes")):
        return 1.0
    if any(k in blob for k in ("false", "wrong", "incorrect", "pants on fire")):
        return 0.0
    if any(k in blob for k in ("half", "mixed", "misleading", "mostly", "partly")):
        return 0.5
    if "satire" in blob:
        return 0.3
    return None


def _factcheck_score(text: str) -> Optional[Dict[str, Any]]:
    """Query the Google Fact Check Tools API. Returns None when unusable."""
    if not FACT_CHECK_API_KEY:
        return None
    try:
        import requests
    except ImportError:
        return None

    query = " ".join(extract_keywords(text, top_n=6))
    try:
        resp = requests.get(
            FACT_CHECK_URL,
            params={"query": query, "key": FACT_CHECK_API_KEY, "languageCode": "en"},
            timeout=5,
        )
    except Exception as exc:
        return {
            "found": False,
            "score": 0.5,
            "method": "google-factcheck",
            "error": f"request failed: {exc}",
            "keywords": extract_keywords(text),
        }
    if resp.status_code != 200:
        return {
            "found": False,
            "score": 0.5,
            "method": "google-factcheck",
            "error": f"HTTP {resp.status_code}",
            "keywords": extract_keywords(text),
        }

    claims = resp.json().get("claims", []) or []
    truths: List[float] = []
    reviews: List[Dict[str, Any]] = []
    for claim in claims:
        for review in claim.get("claimReview", []) or []:
            truth = _parse_rating(review.get("rating", {}) or {})
            if truth is None:
                continue
            truths.append(truth)
            reviews.append(
                {
                    "publisher": (review.get("publisher") or {}).get("name"),
                    "url": review.get("url"),
                    "truth": truth,
                }
            )

    if not truths:
        return {
            "found": False,
            "score": 0.5,
            "method": "google-factcheck",
            "claims_seen": len(claims),
            "keywords": extract_keywords(text),
            "explanation": "Fact-check API returned no rated claims for this query.",
        }

    mean_truth = sum(truths) / len(truths)
    return {
        "found": True,
        "score": round(1.0 - mean_truth, 4),  # 0 = corroborated, 1 = contradicted
        "method": "google-factcheck",
        "claims_seen": len(claims),
        "reviews_used": reviews,
        "keywords": extract_keywords(text),
        "explanation": f"{len(truths)} fact-check rating(s); mean truth={mean_truth:.2f}.",
    }


def _search_score(text: str) -> Optional[Dict[str, Any]]:
    """Query Google Programmable Search for ``<claim> fact check``.

    Returns None when the hook is not configured (falls through to the next
    backend). Snippet language is scored with the same contradiction /
    corroboration markers the offline backend uses.
    """
    if not (GOOGLE_SEARCH_KEY and GOOGLE_CSE_ID):
        return None
    try:
        import requests
    except ImportError:
        return None

    query = " ".join(extract_keywords(text, top_n=6)) + " fact check"
    try:
        resp = requests.get(
            GOOGLE_SEARCH_URL,
            params={
                "key": GOOGLE_SEARCH_KEY,
                "cx": GOOGLE_CSE_ID,
                "q": query,
                "num": 8,
            },
            timeout=5,
        )
    except Exception as exc:
        return {
            "found": False,
            "score": 0.5,
            "method": "google-search",
            "error": f"request failed: {exc}",
            "keywords": extract_keywords(text),
        }
    if resp.status_code != 200:
        return {
            "found": False,
            "score": 0.5,
            "method": "google-search",
            "error": f"HTTP {resp.status_code}",
            "keywords": extract_keywords(text),
        }

    items = resp.json().get("items", []) or []
    if not items:
        return {
            "found": False,
            "score": 0.5,
            "method": "google-search",
            "results_used": 0,
            "keywords": extract_keywords(text),
            "explanation": "Search returned no results for this claim.",
        }

    titles = [str(i.get("title", "")) for i in items]
    snippets = [str(i.get("snippet", "")) for i in items]
    blob = " ".join(titles + snippets).lower()

    contradicting = [m for m in CONTRADICTION_MARKERS if m in blob]
    corroborating = [m for m in CORROBORATION_MARKERS if m in blob]
    score = 0.5
    if contradicting:
        score += min(0.15 * len(contradicting), 0.45)
    if corroborating:
        score -= min(0.15 * len(corroborating), 0.45)
    score = max(0.0, min(1.0, score))

    return {
        "found": True,
        "score": round(score, 4),
        "method": "google-search",
        "results_used": len(items),
        "top_titles": titles[:3],
        "contradicting": contradicting,
        "corroborating": corroborating,
        "keywords": extract_keywords(text),
        "explanation": (
            f"{len(items)} search result(s) for '{query}': "
            f"{len(contradicting)} contradiction cue(s), "
            f"{len(corroborating)} corroboration cue(s)."
        ),
    }


def score_claim(text: str) -> Dict[str, Any]:
    """Return the verification signal for a claim (0 = corroborated,
    1 = contradicted / unsupported, found=False means 'no signal')."""
    result = _factcheck_score(text)
    if result is None:
        result = _search_score(text)
    if result is None:
        result = _offline_score(text)
    return result


def combine(ml_p_fake: float, verification_score: float,
            w_ml: float = VERIFY_W_ML, w_ver: float = VERIFY_W_VER) -> float:
    """Weighted blend of the ML probability and the verification score."""
    total = w_ml + w_ver
    if total <= 0:
        return ml_p_fake
    return (w_ml * ml_p_fake + w_ver * verification_score) / total


def apply(result: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Blend verification into an /analyze/text result (mutates a copy)."""
    result = dict(result)
    details = dict(result.get("details") or {})

    p_ml = details.get("p_fake")
    if result.get("verdict") in ("UNKNOWN", "ERROR") or p_ml is None:
        details["verification"] = {"found": False, "explanation": "no ML score to blend"}
        result["details"] = details
        return result

    verification = score_claim(text)
    details["verification"] = verification
    details["p_fake_ml"] = p_ml

    if not verification.get("found"):
        details["verification"]["explanation"] = verification.get(
            "explanation", "no evidence found; ML verdict kept unchanged"
        )
        result["details"] = details
        return result

    p_final = combine(float(p_ml), float(verification["score"]))
    verdict = "FAKE" if p_final >= 0.5 else "REAL"
    confidence = p_final if verdict == "FAKE" else 1.0 - p_final
    details["p_fake"] = round(p_final, 4)
    details["blended"] = True
    result.update({"verdict": verdict, "confidence": round(confidence, 4)})
    result["details"] = details
    return result
