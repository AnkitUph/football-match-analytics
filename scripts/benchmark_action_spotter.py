"""Benchmark action spotter against SN-PCBAS validation ground truth."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TDEED_DIR = ROOT / "ai_engine/stage6_event_detection/tdeed"
if str(TDEED_DIR) not in sys.path:
    sys.path.insert(0, str(TDEED_DIR))

from action_spotter import TDEEDActionSpotter  # noqa: E402


TARGET_CLASSES = [
    "DRIVE",
    "PASS",
    "CROSS",
    "THROW IN",
    "SHOT",
    "HEADER",
    "PLAYER SUCCESSFUL TACKLE",
    "BALL PLAYER BLOCK",
]


def evaluate_spotter(
    weights_path: Path,
    config_path: Path,
    val_dir: Path,
    device: str = "cpu",
    threshold: float = 0.25,
    tolerance_sec: float = 2.0,
    fps: float = 25.0,
    game_id: str = None,
    limit: int = None,
):
    tolerance_frames = round(tolerance_sec * fps)
    print("=" * 75)
    print("🎯 ACTION SPOTTER BENCHMARK EVALUATION")
    print(f"Weights: {weights_path}")
    print(f"Tolerance window: ±{tolerance_sec:.1f}s (±{tolerance_frames} frames @ {fps} fps)")
    print(f"Confidence threshold: {threshold}")
    print("=" * 75)

    spotter = TDEEDActionSpotter(weights_path=weights_path, config_path=config_path, device=device)

    videos = sorted((val_dir / "videos").glob("game_*.mp4"))
    if game_id:
        videos = [v for v in videos if v.stem == game_id or v.name == game_id]
    if limit and limit > 0:
        videos = videos[:limit]
    if not videos:
        print(f"No validation videos found in {val_dir / 'videos'}")
        return

    # Per-class counters
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    gt_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    for video_file in videos:
        game_id = video_file.stem
        ann_file = val_dir / "annotations" / game_id / "events.json"
        if not ann_file.is_file():
            print(f"Skipping {game_id}: no annotation file found at {ann_file}")
            continue

        ann_data = json.loads(ann_file.read_text())
        gt_events = ann_data.get("events", [])
        print(f"\nProcessing {video_file.name} ({len(gt_events)} ground-truth events)...", flush=True)

        for e in gt_events:
            label = e["label"]
            if label in TARGET_CLASSES:
                gt_counts[label] += 1

        # Run inference
        raw_predictions = spotter.spot_events(video_file, threshold=threshold)
        filtered_preds = [p for p in raw_predictions if p["label"] in TARGET_CLASSES]
        for p in filtered_preds:
            pred_counts[p["label"]] += 1

        print(f"  Detected {len(filtered_preds)} candidate events across target classes", flush=True)

        # Match predictions to GT per class
        for cls in TARGET_CLASSES:
            cls_gt = sorted([e["frame"] for e in gt_events if e["label"] == cls])
            cls_preds = sorted([p["frame"] for p in filtered_preds if p["label"] == cls])

            matched_gt = set()
            for pred_f in cls_preds:
                # Find closest unmatched GT within tolerance
                candidates = [
                    (abs(pred_f - gt_f), idx)
                    for idx, gt_f in enumerate(cls_gt)
                    if abs(pred_f - gt_f) <= tolerance_frames and idx not in matched_gt
                ]
                if candidates:
                    candidates.sort()
                    best_diff, best_idx = candidates[0]
                    matched_gt.add(best_idx)
                    tp[cls] += 1
                else:
                    fp[cls] += 1

            unmatched_gt = len(cls_gt) - len(matched_gt)
            fn[cls] += unmatched_gt

    # Print results table
    print("\n" + "=" * 75)
    print(f"{'CLASS':<26} | {'GT':<6} | {'PRED':<6} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'PREC':<6} | {'REC':<6} | {'F1':<6}")
    print("-" * 75)

    tot_gt = 0
    tot_pred = 0
    tot_tp = 0
    tot_fp = 0
    tot_fn = 0

    for cls in TARGET_CLASSES:
        g = gt_counts[cls]
        p = pred_counts[cls]
        t = tp[cls]
        f_p = fp[cls]
        f_n = fn[cls]

        prec = (t / p) if p > 0 else 0.0
        rec = (t / g) if g > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        tot_gt += g
        tot_pred += p
        tot_tp += t
        tot_fp += f_p
        tot_fn += f_n

        print(f"{cls:<26} | {g:<6} | {p:<6} | {t:<6} | {f_p:<6} | {f_n:<6} | {prec*100:5.1f}% | {rec*100:5.1f}% | {f1*100:5.1f}%")

    print("-" * 75)
    macro_prec = (tot_tp / tot_pred) if tot_pred > 0 else 0.0
    macro_rec = (tot_tp / tot_gt) if tot_gt > 0 else 0.0
    macro_f1 = (2 * macro_prec * macro_rec / (macro_prec + macro_rec)) if (macro_prec + macro_rec) > 0 else 0.0
    print(f"{'OVERALL':<26} | {tot_gt:<6} | {tot_pred:<6} | {tot_tp:<6} | {tot_fp:<6} | {tot_fn:<6} | {macro_prec*100:5.1f}% | {macro_rec*100:5.1f}% | {macro_f1*100:5.1f}%")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    FINETUNED = ROOT / "ai_engine/models/finetuned/tdeed_ball_finetuned_epoch3.pt"
    default_w = FINETUNED if FINETUNED.is_file() else ROOT / "ai_engine/models/tdeed_ball_action_spotter.pt"
    parser.add_argument("--weights", type=Path, default=default_w)
    parser.add_argument("--config", type=Path, default=ROOT / "ai_engine/models/tdeed_ball_action_spotter.json")
    parser.add_argument("--val-dir", type=Path, default=ROOT / "scratch/pcbas2026/prepared/val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--tolerance-sec", type=float, default=2.0)
    parser.add_argument("--game-id", default=None, help="Evaluate only a specific game, e.g. game_18")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of validation games")
    args = parser.parse_args()

    evaluate_spotter(
        weights_path=args.weights,
        config_path=args.config,
        val_dir=args.val_dir,
        device=args.device,
        threshold=args.threshold,
        tolerance_sec=args.tolerance_sec,
        game_id=args.game_id,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
