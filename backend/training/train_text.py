"""Train the text model: TF-IDF + linear classifier on the ISOT Fake News
dataset, saved as a scikit-learn pipeline (joblib).

Usage (from backend/):

    python training/train_text.py --data-dir data --clf pac
    python training/train_text.py --data-dir data --clf logreg

Classifiers (report §4.3 uses the Passive-Aggressive one):

* ``pac``  - PassiveAggressiveClassifier (online, linear, hinge loss). It has
  no native ``predict_proba``, so it is wrapped in CalibratedClassifierCV
  (sigmoid, cv=3) to keep the confidence numbers the API returns.
* ``logreg`` - LogisticRegression (C=4.0), the earlier default.

Dataset: ISOT Fake News - put Fake.csv and True.csv in backend/data/.
Download: https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

Source-bias caveat (why strip_source_bias exists): almost every REAL article
in ISOT is Reuters copy starting with "WASHINGTON (Reuters) -" and sprinkled
with the token "Reuters", while FAKE articles are not. If you train raw, the
model learns publisher style instead of truthfulness and collapses on real
world text. We strip agency leads/tokens before training, and the hybrid
verification step (verify_service) covers what style cues remain.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import TEXT_MODEL_PATH

# "WASHINGTON (Reuters) - The U.S. ..." style lead.
AGENCY_LEAD = re.compile(
    r"^\s*[A-Z][A-Z \.,'&-]{2,60}\s*\((?:Reuters|AP|AFP|BBC)\)\s*[-–—:]\s*"
)
AGENCY_TOKEN = re.compile(r"\b(Reuters|Associated Press|AP News|AFP|BBC News)\b", re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")


def strip_source_bias(text: str) -> str:
    """Remove publisher fingerprints so the model learns content, not style."""
    if not isinstance(text, str):
        return ""
    text = AGENCY_LEAD.sub("", text.strip())
    text = AGENCY_TOKEN.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def _label_series(df: pd.DataFrame, default: int) -> pd.Series:
    """Use an in-file label column when present (mirrors ship 'label'/'type'
    as 1=fake/0=real), otherwise fall back to the file of origin."""
    for col in ("label", "type"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.isin([0, 1]).mean() > 0.99:
                return values.fillna(default).astype(int)
    return pd.Series(default, index=df.index, dtype=int)


def load_isot(data_dir: Path) -> pd.DataFrame:
    fake_path = data_dir / "Fake.csv"
    true_path = data_dir / "True.csv"
    for path in (fake_path, true_path):
        if not path.exists():
            raise SystemExit(
                f"missing {path}\n"
                "Download the ISOT Fake News dataset and put Fake.csv + True.csv "
                f"in {data_dir}"
            )
    frames = []
    for path, default_label in ((fake_path, 1), (true_path, 0)):
        df = pd.read_csv(path)
        df["label"] = _label_series(df, default_label)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if "title" not in df.columns:
        df["title"] = ""
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def build_classifier(kind: str):
    """The linear head on top of TF-IDF.

    ``pac`` (report §4.3) is calibrated because PassiveAggressiveClassifier
    exposes only a decision function, while the API needs predict_proba for
    its confidence value.
    """
    if kind == "pac":
        from sklearn.calibration import CalibratedClassifierCV

        base = PassiveAggressiveClassifier(C=1.0, max_iter=1000, random_state=42)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return LogisticRegression(max_iter=1000, C=4.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BACKEND_DIR / "data")
    parser.add_argument("--out", type=Path, default=TEXT_MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument(
        "--clf",
        choices=("pac", "logreg"),
        default=os.getenv("TC_CLF", "pac"),
        help="linear head: pac = Passive-Aggressive (report default), "
             "logreg = logistic regression (TC_CLF overrides)",
    )
    args = parser.parse_args()

    df = load_isot(args.data_dir)
    print(f"loaded {len(df):,} rows "
          f"({int((df['label'] == 1).sum()):,} fake / {int((df['label'] == 0).sum()):,} real)")

    title = df["title"].fillna("").map(strip_source_bias)
    body = df["text"].fillna("").map(strip_source_bias)
    X = title + " " + body
    y = df["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                max_features=args.max_features,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            )),
            ("clf", build_classifier(args.clf)),
        ]
    )
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    print(f"classifier: {args.clf}")
    print(classification_report(y_te, y_pred, target_names=["REAL", "FAKE"]))
    print(f"accuracy: {accuracy_score(y_te, y_pred):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f"saved -> {args.out}")

    sample_fake = "SHOCKING!!! You won't believe this miracle cure doctors hate"
    sample_real = (
        "According to a study published in the Journal of Medicine on May 3, 2023, "
        "researchers at Oxford University reported that the trial of 4,200 patients "
        "found no statistically significant effect."
    )
    print("sanity check:",
          {"fake_sample_p": round(float(pipe.predict_proba([sample_fake])[0][1]), 3),
           "real_sample_p": round(float(pipe.predict_proba([sample_real])[0][1]), 3)})


if __name__ == "__main__":
    main()
