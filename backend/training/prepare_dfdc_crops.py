"""Download DFDC face crops from an open Hugging Face mirror and build the
JPEG cache that training/train_mesonet.py consumes.

Usage (from backend/):

    python training/prepare_dfdc_crops.py --per-class 166
    python training/prepare_dfdc_crops.py --build-only     # reuse downloaded npz

Source: Shironx/DFDC_Opencv_Face_Crops (face crops from the Facebook Deepfake
Detection Challenge dataset, MIT licence). Each ``.npz`` holds a ``frames``
array of shape (10, 224, 224, 3) - float32 in [0, 1] - i.e. face crops from a
single video, so the train/val split happens at file (= video) level, never at
frame level.

Download is resumable: finished files are skipped on re-run. With fewer REAL
than FAKE clips available we use every REAL file and an equally sized random
FAKE sample (``--per-class`` caps each side).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

REPO = "Shironx/DFDC_Opencv_Face_Crops"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main"
NPZ_DIR = BACKEND_DIR / "data" / "dfdc_npz"
DEFAULT_CACHE = BACKEND_DIR / "data" / "mesonet_cache"


def _get(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "truthcheck/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.headers, r.read()


def list_repo_files() -> list[dict]:
    """All .npz files at the repo root and under face_crops/ (paginated)."""
    entries: list[dict] = []
    url = f"{API}?limit=1000"
    while url:
        headers, body = _get(url)
        entries.extend(json.loads(body))
        link = headers.get("Link", "")
        url = link.split(";")[0].strip(" <>") if 'rel="next"' in link else None

    files = []
    for e in entries:
        path = e.get("path", "")
        if e.get("type") == "file" and path.endswith(".npz"):
            files.append({"path": path, "size": e.get("size", 0)})
    # also walk the face_crops/ subtree explicitly
    url = f"{API}/face_crops?limit=1000"
    while url:
        try:
            headers, body = _get(url)
        except Exception:
            break
        batch = json.loads(body)
        for e in batch:
            path = e.get("path", "")
            if e.get("type") == "file" and path.endswith(".npz"):
                files.append({"path": path, "size": e.get("size", 0)})
        link = headers.get("Link", "")
        url = link.split(";")[0].strip(" <>") if 'rel="next"' in link else None

    seen: set[str] = set()
    unique = []
    for f in sorted(files, key=lambda f: f["path"]):
        name = Path(f["path"]).name
        if name in seen:
            continue
        seen.add(name)
        unique.append(f)
    return unique


def label_of(path: str) -> str | None:
    name = Path(path).name.upper()
    if "FAKE" in name:
        return "fake"
    if "REAL" in name:
        return "real"
    return None


def select(files: list[dict], per_class: int, seed: int) -> list[dict]:
    real = [f for f in files if label_of(f["path"]) == "real"]
    fake = [f for f in files if label_of(f["path"]) == "fake"]
    rng = random.Random(seed)
    rng.shuffle(real)
    rng.shuffle(fake)
    real = real[:per_class]
    fake = fake[: min(per_class, len(fake))]
    chosen = real + fake
    print(f"selected {len(real)} real + {len(fake)} fake files "
          f"({sum(f['size'] for f in chosen) / 1e6:.0f} MB)")
    return chosen


def download_one(file: dict) -> tuple[str, bool, str]:
    dest = NPZ_DIR / Path(file["path"]).name
    if dest.exists() and dest.stat().st_size == file["size"] and file["size"] > 0:
        return dest.name, True, "cached"
    url = f"{RAW}/{urllib.parse.quote(file['path'])}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "truthcheck/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    fh.write(chunk)
            if file["size"] and dest.stat().st_size != file["size"]:
                raise IOError(f"size mismatch {dest.stat().st_size} != {file['size']}")
            return dest.name, True, "downloaded"
        except Exception as exc:
            last = str(exc)
            time.sleep(1.5 * (attempt + 1))
    return dest.name, False, last


def download(files: list[dict], workers: int) -> list[Path]:
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    ok: list[Path] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, f): f for f in files}
        for fut in as_completed(futures):
            name, success, note = fut.result()
            done += 1
            if success:
                ok.append(NPZ_DIR / name)
            else:
                print(f"  ! failed {name}: {note}")
            if done % 20 == 0 or done == len(files):
                print(f"  {done}/{len(files)} files ({len(ok)} ok)")
    return ok


def build_cache(npz_files: list[Path], cache_dir: Path, val_fraction: float, seed: int,
                limit_frames: int) -> dict:
    import cv2
    import numpy as np

    rng = random.Random(seed)
    counts: dict[str, int] = {}

    for label in ("real", "fake"):
        group = sorted([p for p in npz_files if label_of(p.name) == label])
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction)) if len(group) > 1 else 0
        assignment = {"val": group[:n_val], "train": group[n_val:]}
        for split, files in assignment.items():
            out_dir = cache_dir / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            written = 0
            for path in files:
                try:
                    with np.load(path, allow_pickle=True) as data:
                        frames = data["frames"]
                except Exception as exc:
                    print(f"  ! unreadable {path.name}: {exc}")
                    continue
                for i, frame in enumerate(frames[:limit_frames]):
                    img = (np.clip(frame, 0.0, 1.0) * 255).astype("uint8")
                    img = cv2.resize(img, (256, 256))
                    cv2.imwrite(str(out_dir / f"{path.stem}_{i:02d}.jpg"), img)
                    written += 1
            counts[f"{split}/{label}"] = written
            print(f"  {split}/{label}: {written} frames from {len(files)} videos")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=166)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--limit-frames", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--build-only", action="store_true", help="skip download, use existing npz")
    parser.add_argument("--purge", action="store_true", help="delete npz files after caching")
    args = parser.parse_args()

    if args.build_only:
        npz_files = sorted(NPZ_DIR.glob("*.npz"))
        if not npz_files:
            raise SystemExit(f"no npz files in {NPZ_DIR} - run without --build-only first")
    else:
        print("listing repo files ...")
        files = list_repo_files()
        total_real = sum(1 for f in files if label_of(f["path"]) == "real")
        total_fake = sum(1 for f in files if label_of(f["path"]) == "fake")
        print(f"repo has {total_real} real / {total_fake} fake npz files")
        chosen = select(files, args.per_class, args.seed)
        print(f"downloading {len(chosen)} files with {args.workers} workers ...")
        npz_files = download(chosen, args.workers)
        if not npz_files:
            raise SystemExit("nothing downloaded")

    print(f"building JPEG cache in {args.cache_dir} ...")
    counts = build_cache(npz_files, args.cache_dir, args.val_fraction, args.seed,
                         args.limit_frames)
    print("cache counts:", counts)

    if args.purge:
        removed = 0
        for p in npz_files:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
        print(f"purged {removed} npz files")

    print("next: python training/train_mesonet.py --from-cache --epochs 12")


if __name__ == "__main__":
    main()
