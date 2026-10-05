"""Train MesoNet (Meso4) on FaceForensics++ and save app/models/meso4.h5.

Usage (from backend/):

    python training/train_mesonet.py --data-dir data/faceforensicsplusplus

1. Request the dataset: https://github.com/ondyari/FaceForensics
   (accept the terms, then use their download script - e.g. 720p c23).
2. Point --data-dir at the directory tree that contains the videos. Folder
   names decide the label, so layouts like

       <root>/original/*.mp4            <root>/manipulated/deepfakes/*.mp4
       <root>/c23/real/*.mp4            <root>/c23/fake/*.mp4
       <root>/train/real/*.mp4          <root>/train/fake/*.mp4

   all work; unknown folders are skipped with a warning.
3. The script splits by VIDEO id (never by frame - frame-level splits leak
   near-duplicates across train/test and inflate accuracy), extracts every
   Nth frame, crops the largest face (shared YuNet/Haar helper in
   app.services.facecrop - OpenCV 5 removed CascadeClassifier, so a Haar-only
   path would silently stop cropping), caches 256x256 JPEGs, trains Meso4,
   reports accuracy/precision/recall/AUC, saves the weights.

Requires: tensorflow, opencv-python (both in requirements.txt).
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.config import MESO_MODEL_PATH
from app.services import facecrop

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

REAL_DIR_NAMES = {"real", "reals", "original", "originals", "source", "sources",
                  "gt", "true", "bonafide", "bonafides"}
FAKE_DIR_NAMES = {"fake", "fakes", "manipulated", "manipulated_videos",
                  "deepfake", "deepfakes", "face2face", "face_swap", "faceswap",
                  "neural_textures", "faceswap-hq", "df", "ff", "tampered",
                  "forged", "forgeries", "synthetic"}

SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def label_from_path(path: Path, root: Path) -> str | None:
    """Infer real/fake from the folder names above the file."""
    parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    for part in reversed(parts):
        if part in REAL_DIR_NAMES:
            return "real"
        if part in FAKE_DIR_NAMES:
            return "fake"
        if "deepfake" in part or "manipulat" in part:
            return "fake"
        if part in {"original", "orig"} or part.startswith("original"):
            return "real"
    return None


def collect_clips(root: Path) -> List[Tuple[Path, str]]:
    clips: List[Tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTS or not path.is_file():
            continue
        label = label_from_path(path, root)
        if label is None:
            print(f"  skipping (unknown label): {path.relative_to(root)}")
            continue
        clips.append((path, label))
    return clips


def split_by_video(
    clips: List[Tuple[Path, str]], val_fraction: float, seed: int
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """video_id -> 'train'|'val', video_id -> label (split happens on ids)."""
    by_id: Dict[str, str] = {}
    for path, label in clips:
        by_id.setdefault(str(path), label)

    ids = sorted(by_id)
    real_ids = [i for i in ids if by_id[i] == "real"]
    fake_ids = [i for i in ids if by_id[i] == "fake"]
    rng = random.Random(seed)
    rng.shuffle(real_ids)
    rng.shuffle(fake_ids)

    def holdout(group: List[str]) -> Tuple[List[str], List[str]]:
        n_val = max(1, round(len(group) * val_fraction)) if len(group) > 1 else 0
        return group[n_val:], group[:n_val]

    train_real, val_real = holdout(real_ids)
    train_fake, val_fake = holdout(fake_ids)
    assignment = {i: "train" for i in train_real + train_fake}
    assignment.update({i: "val" for i in val_real + val_fake})
    return assignment, by_id


def slug(rel_path: str) -> str:
    return SLUG_RE.sub("_", rel_path.strip("/\\").replace("/", "_").replace("\\", "_"))


def extract_frames(
    clip: Path,
    label: str,
    split: str,
    cache_dir: Path,
    every_n: int,
    max_frames: int,
    face_crop: bool,
) -> int:
    import cv2

    out_dir = cache_dir / split / label
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        print(f"  ! could not open {clip}")
        return 0

    name = slug(clip.stem)
    written = 0
    index = 0
    try:
        while written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % every_n == 0:
                if face_crop:
                    frame, _found = facecrop.crop_face(cv2, frame)
                if frame is None or frame.size == 0:
                    index += 1
                    continue
                frame = cv2.resize(frame, (256, 256))
                cv2.imwrite(str(out_dir / f"{name}_{written:04d}.jpg"), frame)
                written += 1
            index += 1
    finally:
        cap.release()
    return written


def make_datasets(cache_dir: Path, batch_size: int, seed: int, augment: bool = True):
    import tensorflow as tf

    common = dict(
        labels="inferred",
        label_mode="int",
        image_size=(256, 256),
        batch_size=batch_size,
        seed=seed,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        cache_dir / "train", **common, shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        cache_dir / "val", **common, shuffle=False
    )

    class_names = train_ds.class_names  # alphabetical: ['fake', 'real']
    fake_index = class_names.index("fake")

    def convert(x, y):
        # Our convention everywhere else: 1 = fake, 0 = real; and inputs are
        # already divided by 255.0 before they reach the network.
        is_fake = tf.cast(tf.equal(y, fake_index), tf.float32)
        return tf.cast(x, tf.float32) / 255.0, is_fake

    train_ds = train_ds.map(convert)
    if augment:
        # Horizontal mirroring is always safe for faces; keep other jitter
        # OFF - photometric jitter on [0,1] face crops was measured to stall
        # training on small caches (val AUC pinned at ~0.50).
        train_ds = train_ds.map(
            lambda x, y: (tf.image.random_flip_left_right(x, seed=seed), y)
        )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(convert).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="root folder containing the FaceForensics++ videos")
    parser.add_argument("--cache-dir", type=Path, default=BACKEND_DIR / "data" / "mesonet_cache")
    parser.add_argument("--from-cache", action="store_true",
                        help="train from an existing JPEG cache; skips video scanning "
                             "and frame extraction (data-dir not needed)")
    parser.add_argument("--no-augment", action="store_true",
                        help="disable horizontal-flip augmentation on the train split")
    parser.add_argument("--out", type=Path, default=MESO_MODEL_PATH)
    parser.add_argument("--every-n", type=int, default=5, help="sample every Nth frame")
    parser.add_argument("--max-frames-per-video", type=int, default=40)
    parser.add_argument("--limit-videos", type=int, default=0, help="0 = all")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8,
                        help="early-stopping patience (epochs without val_auc gain)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-face-crop", action="store_true")
    args = parser.parse_args()

    try:
        import cv2  # noqa: F401
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"missing dependency: {exc}\npip install -r requirements.txt")

    if not args.no_face_crop:
        import numpy as np

        facecrop.crop_face(cv2, np.zeros((64, 64, 3), dtype=np.uint8))
        print(f"face detector: {facecrop.detector_kind() or 'NONE - would cache full frames'}")

    if args.from_cache:
        for split in ("train", "val"):
            for label in ("real", "fake"):
                if not (args.cache_dir / split / label).is_dir():
                    raise SystemExit(
                        f"cache missing: {args.cache_dir / split / label}\n"
                        "build it with training/prepare_dfdc_crops.py, or run "
                        "without --from-cache to extract frames from videos"
                    )
        print(f"using existing cache -> {args.cache_dir}")
    else:
        if args.data_dir is None:
            raise SystemExit("--data-dir is required unless --from-cache is set")
        if not args.data_dir.is_dir():
            raise SystemExit(f"not a directory: {args.data_dir}")

        print(f"scanning {args.data_dir} ...")
        clips = collect_clips(args.data_dir)
        if not clips:
            raise SystemExit("no videos with a recognisable real/fake label found")
        n_real = sum(1 for _, l in clips if l == "real")
        print(f"found {len(clips)} clips ({n_real} real / {len(clips) - n_real} fake)")

        if args.limit_videos:
            random.Random(args.seed).shuffle(clips)
            clips = clips[: args.limit_videos]

        assignment, labels = split_by_video(clips, args.val_fraction, args.seed)
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total_written = 0
        for path, label in clips:
            split = assignment[str(path)]
            written = extract_frames(
                path, label, split, args.cache_dir,
                args.every_n, args.max_frames_per_video, not args.no_face_crop,
            )
            counts[split][label] += written
            total_written += written
        print(f"cached {total_written} frames -> {args.cache_dir}")
        for split, per_label in counts.items():
            print(f"  {split}: {dict(per_label)}")
        if total_written == 0:
            raise SystemExit("no frames extracted - check the videos / OpenCV build")

    train_ds, val_ds, class_names = make_datasets(
        args.cache_dir, args.batch_size, args.seed, augment=not args.no_augment
    )
    print(f"class order (alphabetical): {class_names}  ->  1 = fake")

    from app.services.mesonet import build_meso4

    model = build_meso4()
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
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Per-video style evaluation on the val set: accuracy / precision / recall / AUC.
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

    y_true: List[int] = []
    y_prob: List[float] = []
    for x, y in val_ds.unbatch():
        y_true.append(int(y.numpy()))
    probs = model.predict(val_ds, verbose=0).ravel()
    y_prob = [float(p) for p in probs]
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"], digits=4))
    print(f"accuracy: {accuracy_score(y_true, y_pred):.4f}  "
          f"auc: {roc_auc_score(y_true, y_prob):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
