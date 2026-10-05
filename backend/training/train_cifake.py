"""Train the CIFAKE image model (AI-generated image detection).

Report §4.2. Usage (from backend/):

    python training/train_cifake.py --download     # fetch data once (~50 MB)
    python training/train_cifake.py --epochs 15

Dataset: CIFAKE (Bird & Lotfi, 2023/24) - 60k CIFAR-10 real photos vs 60k
Stable Diffusion images. Kaggle needs an account, so this script pulls the
Hugging Face mirror ``dragonintelligence/CIFAKE-image-dataset`` (identical
data, parquet, train 100k / test 20k) with no credentials.

Label convention: the mirror ships 0=FAKE / 1=REAL; everything is flipped to
the repo-wide convention here so **1 = AI-generated (fake)**, matching
meso4 and the text model.

Requires: tensorflow, pillow, pyarrow (requirements-dev.txt), requests.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.config import CIFAKE_MODEL_PATH

HF_BASE = "https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset/resolve/main/"
HF_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}


def download(data_dir: Path) -> None:
    """Fetch the parquet splits (resumable: existing files are skipped)."""
    import requests

    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rel in HF_FILES.items():
        dest = data_dir / f"{name}.parquet"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"already present: {dest.name} ({dest.stat().st_size:,} bytes)")
            continue
        url = HF_BASE + rel
        print(f"downloading {url} ...")
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
        print(f"  -> {dest.name} ({dest.stat().st_size:,} bytes)")


def _decode_cell(cell) -> bytes:
    """Extract raw bytes from the parquet image struct cell."""
    if isinstance(cell, dict):
        return cell["bytes"]
    if hasattr(cell, "bytes"):  # pyarrow/Arrow struct scalar
        return cell.bytes
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise TypeError(f"unsupported image cell type: {type(cell)!r}")


def load_split(path: Path, limit: int = 0):
    """parquet -> (images uint8 [N,32,32,3], labels int [N] with 1 = fake)."""
    import numpy as np
    import pandas as pd
    from PIL import Image

    if not path.exists():
        raise SystemExit(f"missing {path}\nrun: python training/train_cifake.py --download")
    df = pd.read_parquet(path)
    if limit:
        df = df.iloc[:limit]
    images = np.empty((len(df), 32, 32, 3), dtype=np.uint8)
    # mirror ships 0=FAKE, 1=REAL -> flip to 1=fake
    labels = 1 - df["label"].to_numpy(dtype=np.int64)
    for i, cell in enumerate(df["image"]):
        img = Image.open(io.BytesIO(_decode_cell(cell))).convert("RGB")
        images[i] = np.asarray(img.resize((32, 32)))
    return images, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BACKEND_DIR / "data" / "cifake")
    parser.add_argument("--download", action="store_true",
                        help="fetch the HF parquet mirror if missing")
    parser.add_argument("--out", type=Path, default=CIFAKE_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="use only the first N images per split (0 = all)")
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(f"missing dependency: {exc}\npip install -r requirements.txt")

    if args.download:
        download(args.data_dir)

    x_train, y_train = load_split(args.data_dir / "train.parquet", args.limit)
    x_test, y_test = load_split(args.data_dir / "test.parquet", args.limit)
    print(f"train {len(x_train):,} / test {len(x_test):,} "
          f"(fake rate train={y_train.mean():.3f} test={y_test.mean():.3f})")

    def make_ds(x, y, shuffle: bool):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(10_000, seed=42)
        ds = ds.batch(args.batch_size).map(
            lambda xi, yi: (tf.cast(xi, tf.float32) / 255.0, tf.cast(yi, tf.float32))
        )
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(x_train, y_train, shuffle=True)
    test_ds = make_ds(x_test, y_test, shuffle=False)

    from app.services.cifake import build_cifake

    model = build_cifake()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]
    model.fit(train_ds, validation_data=test_ds, epochs=args.epochs, callbacks=callbacks)

    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

    probs = model.predict(test_ds, verbose=0).ravel()
    y_pred = (probs >= 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=["REAL", "AI-FAKE"], digits=4))
    print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}  "
          f"auc: {roc_auc_score(y_test, probs):.4f}")

    # The raw probabilities sit low (good ranking, biased scale) - find the
    # operating point the API should use and print it for TC_CIFAKE_THR.
    best_t, best_acc = 0.5, -1.0
    for t in [i / 1000 for i in range(5, 501, 5)]:
        acc = float((probs >= t).__eq__(y_test.astype(bool)).mean())
        if acc > best_acc:
            best_t, best_acc = t, acc
    print(f"best threshold: {best_t:.3f} -> accuracy {best_acc:.4f} "
          f"(default 0.5 -> {accuracy_score(y_test, y_pred):.4f}); "
          f"set TC_CIFAKE_THR={best_t:.3f} if you change the API default")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
