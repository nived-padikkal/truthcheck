"""End-to-end verification of the MesoNet image/video pipeline.

This is the script described in the report (§8 "Verification Flow"):

1. Wait for ``/health`` and confirm ``models.meso4`` is loaded.
2. Send 60 real + 60 fake held-out validation frames to ``/analyze/image``.
3. Report per-class accuracy, average confidence and overall accuracy.
4. Build a short MP4 from validation frames and send it to
   ``/analyze/video``.
5. Print the verdict, confidence, frames analyzed and ``p_fake``.

Usage (backend server must already be running):

    python verify_meso.py
    python verify_meso.py --api http://localhost:8000 --frames 60
    python verify_meso.py --json            # machine-readable summary

Exit code 0 = the API answered and every sample was scored;
exit code 1 = server unreachable / no frames / request errors.
Accuracy is *reported*, never asserted - the point is honest numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = BACKEND_DIR / "backend" / "data" / "mesonet_cache" / "val"


def wait_for_health(api: str, timeout_sec: int = 60) -> Dict:
    """Poll /health until the server answers; returns the JSON body."""
    deadline = time.time() + timeout_sec
    last_error = "no attempt yet"
    while time.time() < deadline:
        try:
            resp = requests.get(f"{api}/health", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            last_error = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2)
    raise SystemExit(f"server not reachable at {api}/health ({last_error})")


def collect_frames(cache: Path, per_class: int) -> List[Tuple[Path, int]]:
    """Up to ``per_class`` held-out frames per class, label 1 = fake.

    Frames are stride-sampled across the whole sorted list rather than taken
    from the front: filenames are grouped by source video, so taking the
    first N would evaluate only a handful of videos.
    """
    samples: List[Tuple[Path, int]] = []
    for label_dir, label in (("fake", 1), ("real", 0)):
        files = sorted((cache / label_dir).glob("*.jpg"))
        if not files:
            print(f"  ! no frames in {cache / label_dir}")
            continue
        step = max(1, len(files) // per_class)
        picked = files[::step][:per_class]
        samples.extend((p, label) for p in picked)
    return samples


def eval_images(api: str, samples: List[Tuple[Path, int]]) -> Dict:
    """POST every frame to /analyze/image and score the answers.

    Two views of the same run:
      * **raw** - the model's ungated verdict (details.ungated_verdict, or the
        verdict itself when the confidence gate passed) = model quality;
      * **gated** - what the API actually returned = system behaviour
        (UNKNOWN counts as "not correct" here).
    """
    per_class = {0: {"n": 0, "correct": 0, "raw_correct": 0, "unknown": 0, "conf": 0.0},
                 1: {"n": 0, "correct": 0, "raw_correct": 0, "unknown": 0, "conf": 0.0}}
    errors: List[str] = []

    for path, label in samples:
        try:
            with open(path, "rb") as fh:
                resp = requests.post(
                    f"{api}/analyze/image",
                    files={"file": (path.name, fh, "image/jpeg")},
                    timeout=120,
                )
            if resp.status_code != 200:
                errors.append(f"{path.name}: HTTP {resp.status_code}")
                continue
            body = resp.json()
        except requests.RequestException as exc:
            errors.append(f"{path.name}: {exc}")
            continue

        details = body.get("details") or {}
        cls = per_class[label]
        cls["n"] += 1
        verdict = body.get("verdict")
        if verdict == "UNKNOWN":
            cls["unknown"] += 1
        elif (verdict == "FAKE") == bool(label):
            cls["correct"] += 1
        raw = details.get("ungated_verdict") or verdict
        if raw and (raw == "FAKE") == bool(label):
            cls["raw_correct"] += 1
        cls["conf"] += float(body.get("confidence") or 0.0)

    total = sum(c["n"] for c in per_class.values())
    correct = sum(c["correct"] for c in per_class.values())
    raw_correct = sum(c["raw_correct"] for c in per_class.values())
    unknown = sum(c["unknown"] for c in per_class.values())
    conf = sum(c["conf"] for c in per_class.values())
    return {
        "total": total,
        "correct": correct,
        "errors": errors,
        "unknown": unknown,
        "accuracy": correct / total if total else 0.0,
        "raw_accuracy": raw_correct / total if total else 0.0,
        "coverage": (total - unknown) / total if total else 0.0,
        "mean_confidence": conf / total if total else 0.0,
        "per_class": {
            "REAL": _class_stats(per_class[0]),
            "FAKE": _class_stats(per_class[1]),
        },
    }


def _class_stats(c: Dict) -> Dict:
    decided = c["n"] - c["unknown"]
    return {
        "support": c["n"],
        "correct": c["correct"],
        "accuracy": c["correct"] / c["n"] if c["n"] else 0.0,
        "raw_accuracy": c["raw_correct"] / c["n"] if c["n"] else 0.0,
        "gated_unknown": c["unknown"],
        "decided": decided,
        "mean_confidence": c["conf"] / c["n"] if c["n"] else 0.0,
    }


def build_mp4(paths: List[Path], out_path: Path, fps: int = 10) -> bool:
    """Assemble 256x256 frames into a short MP4 (the model's input size)."""
    import cv2

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (256, 256)
    )
    if not writer.isOpened():
        return False
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        writer.write(cv2.resize(frame, (256, 256)))
    writer.release()
    return out_path.exists() and out_path.stat().st_size > 0


def eval_video(api: str, clip: Path, expected: str) -> Optional[Dict]:
    try:
        with open(clip, "rb") as fh:
            resp = requests.post(
                f"{api}/analyze/video",
                files={"file": (clip.name, fh, "video/mp4")},
                data={"every_n": "5", "max_frames": "30"},
                timeout=300,
            )
    except requests.RequestException as exc:
        return {"clip": clip.name, "error": str(exc)}
    if resp.status_code != 200:
        return {"clip": clip.name, "error": f"HTTP {resp.status_code}"}
    body = resp.json()
    details = body.get("details") or {}
    return {
        "clip": clip.name,
        "expected": expected,
        "verdict": body.get("verdict"),
        "confidence": body.get("confidence"),
        "frames_analyzed": details.get("frames_analyzed"),
        "p_fake": details.get("p_fake"),
        "face_detector": details.get("face_detector"),
        "frames_with_face": details.get("frames_with_face"),
        "error": details.get("error"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--frames", type=int, default=60,
                        help="frames per class for the image sweep (report uses 60)")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help="directory with val/real and val/fake JPEGs")
    parser.add_argument("--video-frames", type=int, default=30)
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    args = parser.parse_args()
    api = args.api.rstrip("/")

    health = wait_for_health(api)
    models = health.get("models") or {}
    if not models.get("meso4"):
        print("meso4 model NOT loaded - train it first "
              "(python training/train_mesonet.py --from-cache ...)")
        if not args.json:
            print(json.dumps(health, indent=2))
        return 1

    samples = collect_frames(args.cache, args.frames)
    if not samples:
        print(f"no validation frames under {args.cache}")
        return 1

    if not args.json:
        print(f"health ok: models={models}")
        print(f"scoring {len(samples)} frames ({args.frames}/class) ...")

    image_stats = eval_images(api, samples)

    videos: List[Dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for label_dir, expected in (("fake", "FAKE"), ("real", "REAL")):
            all_files = sorted((args.cache / label_dir).glob("*.jpg"))
            if not all_files:
                continue
            step = max(1, len(all_files) // args.video_frames)
            paths = all_files[::step][: args.video_frames]
            clip = Path(tmp) / f"val_{label_dir}.mp4"
            if build_mp4(paths, clip):
                videos.append(eval_video(api, clip, expected) or {})

    summary = {
        "api": api,
        "models": models,
        "image": image_stats,
        "video": videos,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0 if not image_stats["errors"] else 1

    print("\n== image sweep (/analyze/image) ==")
    print(f"raw model accuracy   : {image_stats['raw_accuracy']:.4f} "
          f"(ungated verdicts, {image_stats['total']} frames)")
    print(f"gate coverage        : {image_stats['coverage']:.4f} "
          f"({image_stats['total'] - image_stats['unknown']}/{image_stats['total']} decided, "
          f"{image_stats['unknown']} -> UNKNOWN)")
    print(f"gated accuracy       : {image_stats['accuracy']:.4f} "
          f"(UNKNOWN counts as incorrect)")
    print(f"mean confidence      : {image_stats['mean_confidence']:.4f}")
    for name, c in image_stats["per_class"].items():
        print(f"  {name:4} raw_acc={c['raw_accuracy']:.4f} "
              f"gated_acc={c['accuracy']:.4f} n={c['support']} "
              f"conf={c['mean_confidence']:.4f} unknown={c['gated_unknown']}")
    if image_stats["errors"]:
        print(f"  request errors: {len(image_stats['errors'])} "
              f"(first: {image_stats['errors'][0]})")

    print("\n== video sweep (/analyze/video) ==")
    if not videos:
        print("  no clips built")
    for v in videos:
        if "error" in v and v["error"]:
            print(f"  {v['clip']}: ERROR {v['error']}")
            continue
        print(f"  {v['clip']} (expected {v['expected']}): "
              f"verdict={v['verdict']} confidence={v['confidence']} "
              f"frames={v['frames_analyzed']} p_fake={v['p_fake']}")

    return 0 if not image_stats["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
