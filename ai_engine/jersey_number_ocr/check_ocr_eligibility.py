#!/usr/bin/env python3
"""
check_ocr_eligibility.py
------------------------
Answers one question: "Are our player crops big enough for jersey-number OCR
to have a real chance?"

It does NOT run OCR. It only measures box sizes. Four input modes:

  1) Django Match ID:
       python check_ocr_eligibility.py --match-id 75

  2) Pickle stub from the tracker (tracks["players"][frame][track_id]["bbox"]):
       python check_ocr_eligibility.py --pickle stubs/track_stubs.pkl

  3) CSV with one row per player per frame:
       python check_ocr_eligibility.py --csv player_tracking.csv

  4) A folder of crop images:
       python check_ocr_eligibility.py --images-dir path/to/sn-jersey/train/images
"""
import argparse
import csv
import io
import os
import pickle
import statistics as st
import sys
from collections import defaultdict

# Rule of thumb from the failure analysis: digits of 12-22 px inside 80-130 px
# boxes, so the digit is about 15-17% of the box height.
DIGIT_FRACTION = 0.16

DEFAULT_MIN_HEIGHT = 150      # box height (px) before we even try OCR
DEFAULT_MIN_FRAMES = 50       # tall-enough frames a track needs
ELIGIBLE_SHARE = 0.40         # share of tracks that pass -> worth doing fully
PARTIAL_SHARE = 0.15          # below this -> skip OCR


def pct(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = min(len(values) - 1, max(0, int(round((p / 100.0) * (len(values) - 1)))))
    return float(values[k])


def load_pickle(path):
    with open(path, "rb") as f:
        tracks = pickle.load(f)
    per_track = defaultdict(list)
    for frame_dets in tracks.get("players", []):
        for tid, info in frame_dets.items():
            x1, y1, x2, y2 = info["bbox"]
            per_track[int(tid)].append(float(y2) - float(y1))
    return per_track


def load_csv(path_or_content):
    if os.path.exists(path_or_content):
        f = open(path_or_content, newline="", encoding="utf-8")
        should_close = True
    else:
        f = io.StringIO(path_or_content)
        should_close = False

    try:
        reader = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in reader.fieldnames}

        def pick(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        tid_c = pick("track_id", "stitched_id", "player_id", "id", "tracker_id")
        cls_c = pick("class", "label", "category")
        h_c = pick("h", "height", "bbox_h", "box_h")
        y1_c, y2_c = pick("y1", "ymin", "top"), pick("y2", "ymax", "bottom")
        if tid_c is None or (h_c is None and not (y1_c and y2_c)):
            raise SystemExit(
                "Could not find columns. Need a track id column and either a "
                "height column or y1/y2 columns. Found: %s" % list(cols)
            )
        per_track = defaultdict(list)
        for row in reader:
            if cls_c and row.get(cls_c) in ("ball", "referee"):
                continue
            try:
                h = float(row[h_c]) if h_c else float(row[y2_c]) - float(row[y1_c])
                if h > 5.0:
                    per_track[row[tid_c]].append(h)
            except (ValueError, TypeError):
                continue
    finally:
        if should_close:
            f.close()
    return per_track


def load_images(folder):
    try:
        import cv2
    except ImportError:
        raise SystemExit("Install opencv-python for --images-dir mode.")
    heights = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                img = cv2.imread(os.path.join(root, name))
                if img is not None:
                    heights.append(float(img.shape[0]))
    return heights


def summarize_heights(heights, label):
    print("\n[%s] crop/box height in pixels" % label)
    print("  count      : %d" % len(heights))
    for p in (10, 25, 50, 75, 90, 95, 99):
        print("  p%-2d        : %.0f px  (digit ~ %.0f px)"
              % (p, pct(heights, p), pct(heights, p) * DIGIT_FRACTION))


def check_eligibility_dict(per_track, min_height=DEFAULT_MIN_HEIGHT, min_frames=DEFAULT_MIN_FRAMES):
    all_heights = [h for hs in per_track.values() for h in hs]
    if not all_heights:
        return {"status": "NO_DETECTIONS", "verdict": "NOT ELIGIBLE", "share": 0.0}

    summarize_heights(all_heights, "all player boxes")

    tall_frames = {t: sum(1 for h in hs if h >= min_height)
                   for t, hs in per_track.items()}
    max_h = {t: max(hs) for t, hs in per_track.items() if hs}
    passing = [t for t, n in tall_frames.items() if n >= min_frames]
    share = len(passing) / max(1, len(per_track))
    frame_share = sum(tall_frames.values()) / len(all_heights)

    print("\n[gate] box height >= %d px (digit ~ %.0f px), >= %d such frames per track"
          % (min_height, min_height * DIGIT_FRACTION, min_frames))
    print("  tracks total                 : %d" % len(per_track))
    print("  tracks passing the gate      : %d (%.1f%%)" % (len(passing), share * 100))
    print("  share of all boxes tall enough: %.1f%%" % (frame_share * 100))
    if max_h:
        print("  median of each track's tallest box: %.0f px"
              % st.median(max_h.values()))

    print("\n[verdict]")
    if share >= ELIGIBLE_SHARE:
        verdict = "ELIGIBLE"
        print("  ELIGIBLE: enough tracks have large views. Do the full OCR plan.")
    elif share >= PARTIAL_SHARE:
        verdict = "PARTIAL"
        print("  PARTIAL: only some tracks are large enough. Build OCR as a bonus\n"
              "  signal for those tracks only. Lineup matching must work without it.")
    else:
        verdict = "NOT ELIGIBLE"
        print("  NOT ELIGIBLE: crops are too small (<15% tracks pass gate). Skip OCR.\n"
              "  Use color + position + human review. More training will not add missing pixels.")
    print("\nNote: this checks size only. Also look at ~30 tall crops by eye:\n"
          "if you cannot read the number yourself, the model will not either.")

    return {
        "verdict": verdict,
        "share": share,
        "total_tracks": len(per_track),
        "passing_tracks": len(passing),
        "all_heights_median": pct(all_heights, 50),
    }


def check_ocr_eligibility_for_match(match_id: int):
    """Programmatic API to evaluate OCR eligibility for a Django match."""
    import django
    if not os.environ.get("DJANGO_SETTINGS_MODULE"):
        sys.path.insert(0, "/app")
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
        django.setup()
    from apps.matches.models import Match

    match = Match.objects.get(id=match_id)
    if not match.files or not match.files.player_tracking_csv:
        raise ValueError(f"Match #{match_id} does not have a player_tracking_csv.")

    with match.files.player_tracking_csv.open("r") as f:
        content = f.read()

    print(f"\nEvaluating Match #{match_id}: {match.home_team.name} vs {match.away_team.name}")
    per_track = load_csv(content)
    return check_eligibility_dict(per_track)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", type=int, help="Django Match ID to evaluate")
    ap.add_argument("--pickle", help="Path to tracking pickle file")
    ap.add_argument("--csv", help="Path to player tracking CSV")
    ap.add_argument("--images-dir", help="Path to directory of crop images")
    ap.add_argument("--min-height", type=int, default=DEFAULT_MIN_HEIGHT)
    ap.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    args = ap.parse_args()

    if args.match_id:
        check_ocr_eligibility_for_match(args.match_id)
        return

    if args.images_dir:
        heights = load_images(args.images_dir)
        if not heights:
            raise SystemExit("No images found.")
        summarize_heights(heights, "images")
        print("\nCompare these numbers with your own footage (--pickle / --csv / --match-id).")
        return

    if args.pickle:
        per_track = load_pickle(args.pickle)
    elif args.csv:
        per_track = load_csv(args.csv)
    else:
        raise SystemExit("Give --match-id, --pickle, --csv or --images-dir.")

    check_eligibility_dict(per_track, min_height=args.min_height, min_frames=args.min_frames)


if __name__ == "__main__":
    main()
