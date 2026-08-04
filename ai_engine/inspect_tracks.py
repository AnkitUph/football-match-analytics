"""
Solves the manual-pixel-picking problem at its root: instead of guessing
coordinates on one static frame, this uses the ACTUAL detected bounding
boxes for every player across the whole video (from Stage 2's
players.csv) and computes each track's real median jersey color - the
exact same sampling logic team_classification.py uses internally.

Workflow:
    1. Run this against your Stage 2 output
    2. It prints every track_id with its computed color and sample count
    3. Watch tracked.mp4 (from Stage 2) and note which track_id number
       belongs to which team (e.g. "#7 is the dark-kit team")
    4. Use THAT track's computed hex color directly as --home/--away in
       team_classification.py - no manual coordinate-finding needed

Usage:
    python -m ai_engine.inspect_tracks <video_path> <players_csv>
"""

import argparse
import colorsys
import csv
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _sample_jersey_hsv(frame_bgr, x1, y1, x2, y2):
    """Identical crop logic to team_classification.py, kept in sync deliberately."""
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    cx1 = int(x1 + w * 0.25)
    cx2 = int(x1 + w * 0.75)
    cy1 = int(y1 + h * 0.15)
    cy2 = int(y1 + h * 0.55)

    frame_h, frame_w = frame_bgr.shape[:2]
    cx1, cx2 = max(0, cx1), min(frame_w, max(cx1 + 1, cx2))
    cy1, cy2 = max(0, cy1), min(frame_h, max(cy1 + 1, cy2))

    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
    med_r, med_g, med_b = np.median(crop_rgb, axis=0)
    h_val, s_val, v_val = colorsys.rgb_to_hsv(med_r, med_g, med_b)
    return (h_val * 360, s_val, v_val)


def _hsv_to_hex(hsv):
    h, s, v = hsv
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


def inspect_tracks(video_path, player_tracking_csv_path):
    video_path = Path(video_path)
    player_tracking_csv_path = Path(player_tracking_csv_path)

    detections_by_frame = defaultdict(list)
    with open(player_tracking_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_index"])
            track_id = int(row["track_id"])
            x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
            detections_by_frame[frame_idx].append((track_id, x1, y1, x2, y2))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    track_samples = defaultdict(list)
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for track_id, x1, y1, x2, y2 in detections_by_frame.get(frame_index, []):
            hsv = _sample_jersey_hsv(frame, x1, y1, x2, y2)
            if hsv is not None:
                track_samples[track_id].append(hsv)

        frame_index += 1

    cap.release()

    results = []
    for track_id, samples in track_samples.items():
        median_hsv = tuple(np.median(np.array(samples), axis=0))
        hex_color = _hsv_to_hex(median_hsv)
        results.append((track_id, hex_color, len(samples), median_hsv))

    results.sort(key=lambda r: -r[2])  # most-sampled tracks first (most reliable)

    print(f"{'track_id':>8}  {'hex_color':>9}  {'samples':>7}  swatch")
    print("-" * 50)
    for track_id, hex_color, sample_count, hsv in results:
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        brightness_note = " (dark)" if hsv[2] < 0.25 else (" (bright)" if hsv[2] > 0.7 else "")
        print(f"{track_id:>8}  {hex_color:>9}  {sample_count:>7}  RGB({r},{g},{b}){brightness_note}")

    print()
    print(f"{len(results)} tracks found. Watch tracked.mp4, note the track_id for each")
    print("team's players (and the referee), then use their hex_color values directly")
    print("as --home/--away/--home-gk/--away-gk in ai_engine.team_classification.")
    print()
    print("Prefer tracks with a HIGH sample count - a track with only a handful of")
    print("samples (a brief/unstable ID) is less trustworthy than one tracked for")
    print("hundreds of frames.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print each track's real computed jersey color.")
    parser.add_argument("video_path")
    parser.add_argument("players_csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    inspect_tracks(args.video_path, args.players_csv)