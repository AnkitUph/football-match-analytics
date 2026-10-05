#!/usr/bin/env python3
"""
eval_soccernet_tracking.py
==========================
Run your YOLO model (best.pt) + BoT-SORT on SoccerNet-tracking sequences and
write the results in MOT format so TrackEval can score them (HOTA/DetA/AssA,
MOTA, IDF1).

What it does
------------
1. Reads every sequence folder in --split-dir (SNMOT-xxx/img1/*.jpg).
2. Runs the detector on each frame.
3. Persons (player + goalkeeper + referee) -> ultralytics BoT-SORT tracker.
4. Ball -> best ball detection per frame, one fixed track ID (like your
   pipeline), optional gap interpolation.
5. Writes  <out-dir>/<name>/data/<SEQ>.txt   (TrackEval layout)
   and       <out-dir>/seqmap.txt
6. Optionally runs TrackEval for you (--trackeval-dir).

SoccerNet ignores object classes, so all classes go into one file.

Expected folder layout (SoccerNet, after unzipping):
    <split-dir>/SNMOT-116/img1/000001.jpg ...
    <split-dir>/SNMOT-116/gt/gt.txt
    <split-dir>/SNMOT-116/seqinfo.ini

Example:
    python eval_soccernet_tracking.py \
        --model models/best.pt \
        --split-dir ~/SoccerNet/tracking/test \
        --out-dir runs_eval --name yolo11s_640 --limit 3
"""
from __future__ import annotations

import argparse
import configparser
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BALL_TRACK_ID = 99999          # any ID that tracker IDs will never reach
IMG_EXTS = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/best.pt", help="YOLO weights (.pt)")
    p.add_argument("--split-dir", required=True,
                   help="Folder that contains the SNMOT-* sequence folders")
    p.add_argument("--out-dir", default="runs_eval", help="Where results are written")
    p.add_argument("--name", default="yolo11s_botsort",
                   help="Tracker name = results sub-folder. Use a new name per experiment.")

    # detection
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference size. best.pt was trained at 640; try 1280 as an experiment")
    p.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold")
    p.add_argument("--ball-conf", type=float, default=0.1, help="Min confidence for ball")
    p.add_argument("--device", default="cpu", help="'cpu', '0' (CUDA), 'xpu', ...")
    p.add_argument("--batch", type=int, default=8, help="Frames per inference batch")
    p.add_argument("--no-agnostic-nms", action="store_true",
                   help="Disable class-agnostic NMS (default: ON, avoids the same person "
                        "being output as both player and goalkeeper)")

    # tracker (only given values override the ultralytics botsort.yaml defaults)
    p.add_argument("--tracker-cfg", default=None, help="Custom botsort.yaml path")
    p.add_argument("--track-high-thresh", type=float, default=None)
    p.add_argument("--track-low-thresh", type=float, default=None)
    p.add_argument("--new-track-thresh", type=float, default=None)
    p.add_argument("--track-buffer", type=int, default=None)
    p.add_argument("--match-thresh", type=float, default=None,
                   help="Higher = more lenient matching")
    p.add_argument("--gmc-method", default=None,
                   help="Camera motion compensation: sparseOptFlow | orb | ecc | none")

    # ball
    p.add_argument("--no-ball", action="store_true", help="Leave the ball out of the output")
    p.add_argument("--ball-interp-gap", type=int, default=25,
                   help="Linearly fill ball gaps up to this many frames (0 = off)")

    # selection
    p.add_argument("--limit", type=int, default=None, help="Only the first N sequences")
    p.add_argument("--seq", nargs="+", default=None, help="Only these sequence names")

    # evaluation
    p.add_argument("--trackeval-dir", default=None,
                   help="Path to a TrackEval clone. If given, evaluation runs after tracking.")
    return p.parse_args()


# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------
def find_sequences(split_dir: Path, only, limit) -> list[Path]:
    seqs = sorted(p for p in split_dir.iterdir() if (p / "img1").is_dir())
    if only:
        wanted = set(only)
        seqs = [s for s in seqs if s.name in wanted]
    if limit:
        seqs = seqs[:limit]
    if not seqs:
        sys.exit(f"No sequences (folders with img1/) found in {split_dir}")
    return seqs


def read_frame_rate(seq_dir: Path) -> float:
    ini = seq_dir / "seqinfo.ini"
    if ini.exists():
        cp = configparser.ConfigParser()
        cp.read(ini)
        if cp.has_section("Sequence"):
            return float(cp["Sequence"].get("frameRate", 25))
    return 25.0


def list_frames(seq_dir: Path) -> list[tuple[int, Path]]:
    """Return [(frame_number, path)] with 1-based frame numbers like the GT."""
    paths = sorted(p for p in (seq_dir / "img1").iterdir() if p.suffix.lower() in IMG_EXTS)
    out = []
    for i, p in enumerate(paths, start=1):
        try:
            out.append((int(p.stem), p))
        except ValueError:
            out.append((i, p))
    return out


# --------------------------------------------------------------------------
# Model / tracker setup (ultralytics imported lazily)
# --------------------------------------------------------------------------
def resolve_class_ids(names: dict) -> tuple[int, list[int]]:
    inv = {str(v).lower(): int(k) for k, v in names.items()}
    needed = ["ball", "player", "goalkeeper", "referee"]
    missing = [n for n in needed if n not in inv]
    if missing:
        sys.exit(f"Model is missing classes {missing}. Model classes: {names}")
    return inv["ball"], [inv["player"], inv["goalkeeper"], inv["referee"]]


def build_tracker(args: argparse.Namespace, frame_rate: float):
    from ultralytics.trackers import BOTSORT
    from ultralytics.utils import ROOT, IterableSimpleNamespace
    try:
        from ultralytics.utils import YAML
        load_yaml = YAML.load
    except ImportError:                       # older ultralytics
        from ultralytics.utils import yaml_load as load_yaml

    cfg_path = Path(args.tracker_cfg) if args.tracker_cfg else ROOT / "cfg/trackers/botsort.yaml"
    cfg = dict(load_yaml(cfg_path))
    overrides = {
        "track_high_thresh": args.track_high_thresh,
        "track_low_thresh": args.track_low_thresh,
        "new_track_thresh": args.new_track_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
        "gmc_method": args.gmc_method,
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return BOTSORT(args=IterableSimpleNamespace(**cfg)), cfg


# --------------------------------------------------------------------------
# Ball helpers
# --------------------------------------------------------------------------
def interpolate_ball(ball: dict[int, tuple], max_gap: int) -> dict[int, tuple]:
    """ball: {frame: (x1, y1, x2, y2, conf)}. Fill gaps of <= max_gap missing frames."""
    if max_gap <= 0 or len(ball) < 2:
        return ball
    out = dict(ball)
    frames = sorted(ball)
    for f0, f1 in zip(frames[:-1], frames[1:]):
        gap = f1 - f0 - 1
        if 0 < gap <= max_gap:
            a, b = np.array(ball[f0]), np.array(ball[f1])
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                out[f0 + k] = tuple((a + (b - a) * t).tolist())
    return out


# --------------------------------------------------------------------------
# Core: track one sequence
# --------------------------------------------------------------------------
def track_sequence(model, seq_dir: Path, args, ball_cls: int, person_cls: list[int]) -> list[str]:
    frames = list_frames(seq_dir)
    tracker, _ = build_tracker(args, read_frame_rate(seq_dir))

    rows: list[tuple] = []            # (frame, id, x, y, w, h, conf)
    ball: dict[int, tuple] = {}
    img_w = img_h = None

    for i in range(0, len(frames), args.batch):
        chunk = frames[i:i + args.batch]
        results = model.predict(
            [str(p) for _, p in chunk],
            imgsz=args.imgsz, conf=args.conf, device=args.device,
            agnostic_nms=not args.no_agnostic_nms, batch=args.batch, verbose=False,
        )
        # tracker must see frames strictly in order
        for (fnum, _), res in zip(chunk, results):
            if img_h is None:
                img_h, img_w = res.orig_shape
            boxes = res.boxes.cpu().numpy()
            cls = boxes.cls.astype(int)

            # ----- persons -> BoT-SORT
            det = boxes[np.isin(cls, person_cls)]
            tracks = tracker.update(det, res.orig_img)
            for t in tracks:
                x1, y1, x2, y2 = (float(v) for v in t[:4])
                x1, x2 = np.clip([x1, x2], 0, img_w)
                y1, y2 = np.clip([y1, y2], 0, img_h)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                rows.append((fnum, int(t[4]), x1, y1, x2 - x1, y2 - y1, float(t[5])))

            # ----- ball -> best detection, fixed ID
            if not args.no_ball:
                m = (cls == ball_cls) & (boxes.conf >= args.ball_conf)
                if m.any():
                    j = int(np.argmax(np.where(m, boxes.conf, -1)))
                    x1, y1, x2, y2 = boxes.xyxy[j].tolist()
                    ball[fnum] = (x1, y1, x2, y2, float(boxes.conf[j]))

    if not args.no_ball:
        for fnum, (x1, y1, x2, y2, c) in interpolate_ball(ball, args.ball_interp_gap).items():
            rows.append((fnum, BALL_TRACK_ID, x1, y1, x2 - x1, y2 - y1, c))

    rows.sort(key=lambda r: (r[0], r[1]))
    return [f"{f},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{c:.4f},-1,-1,-1"
            for f, tid, x, y, w, h, c in rows]


# --------------------------------------------------------------------------
# TrackEval
# --------------------------------------------------------------------------
def run_trackeval(args, split_dir: Path, out_root: Path, seqmap: Path) -> None:
    script = Path(args.trackeval_dir) / "scripts" / "run_mot_challenge.py"
    if not script.exists():
        sys.exit(f"TrackEval script not found: {script}")
    cmd = [
        sys.executable, str(script),
        "--BENCHMARK", "SNMOT",
        "--SPLIT_TO_EVAL", split_dir.name,
        "--GT_FOLDER", str(split_dir.resolve()),
        "--TRACKERS_FOLDER", str(out_root.resolve()),
        "--TRACKERS_TO_EVAL", args.name,
        "--SEQ_INFO", *[p.name for p in find_sequences(split_dir, args.seq, args.limit)],
        "--SKIP_SPLIT_FOL", "True",
        "--TRACKER_SUB_FOLDER", "data",
        "--DO_PREPROC", "False",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
        "--PRINT_CONFIG", "False",
    ]
    print("\n$ " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir).expanduser()
    out_root = Path(args.out_dir).expanduser()
    data_dir = out_root / args.name / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    seqs = find_sequences(split_dir, args.seq, args.limit)
    print(f"{len(seqs)} sequence(s) from {split_dir}")

    from ultralytics import YOLO
    model = YOLO(args.model)
    ball_cls, person_cls = resolve_class_ids(model.names)

    (out_root / args.name / "run_config.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())) + "\n")

    t_all = time.time()
    n_frames = 0
    for k, seq in enumerate(seqs, start=1):
        t0 = time.time()
        lines = track_sequence(model, seq, args, ball_cls, person_cls)
        (data_dir / f"{seq.name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        nf = len(list_frames(seq))
        n_frames += nf
        dt = time.time() - t0
        print(f"[{k}/{len(seqs)}] {seq.name}: {nf} frames, {len(lines)} boxes, "
              f"{nf / dt:.1f} fps")

    seqmap = out_root / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(s.name for s in seqs) + "\n")
    print(f"\nDone: {n_frames} frames in {(time.time() - t_all) / 60:.1f} min")
    print(f"Tracker files: {data_dir}")

    if args.trackeval_dir:
        run_trackeval(args, split_dir, out_root, seqmap)
    else:
        print("\nTo score (from your TrackEval clone):\n"
              f"  python scripts/run_mot_challenge.py --BENCHMARK SNMOT "
              f"--SPLIT_TO_EVAL {split_dir.name} --GT_FOLDER {split_dir.resolve()} "
              f"--TRACKERS_FOLDER {out_root.resolve()} --TRACKERS_TO_EVAL {args.name} "
              f"--SEQMAP_FILE {seqmap.resolve()} --SKIP_SPLIT_FOL True "
              f"--TRACKER_SUB_FOLDER data --DO_PREPROC False "
              f"--METRICS HOTA CLEAR Identity --USE_PARALLEL False")


if __name__ == "__main__":
    main()
