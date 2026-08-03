"""
Stage 4 of the CV pipeline: team classification.

Consumes Stage 2's output (players.csv) rather than re-running YOLO -
detection/tracking is expensive, no need to pay for it twice. For every
tracked player box, crops the torso region and samples its color, builds
up a per-track_id color history across every frame that player appears
in, then classifies each track against the match's four kit colors
(home outfield, home GK, away outfield, away GK - captured back in
Phase 3's upload form).

Anyone whose sampled color doesn't closely match ANY of the four - most
likely the referee, but could also be a linesman or ball boy that slipped
through Stage 2's pitch filtering - gets labeled "unclassified" rather
than force-matched to a team. That's deliberate: a wrong team assignment
is worse than an honest "don't know," since it would silently corrupt
team-level stats later.

Still fully standalone - no Django/Celery/database. Run it by hand:

    python -m ai_engine.team_classification \\
        <video_path> <players_csv_from_stage2> <output_dir> \\
        --home "#3D8B5F" --away "#274690" \\
        --home-gk "#F4C542" --away-gk "#4CAF50"
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

# How much of each axis matters when comparing colors. Hue (the actual
# color, e.g. "red" vs "blue") dominates; saturation and value (which
# shift a lot with lighting/shadow/motion blur) matter much less.
HUE_WEIGHT = 0.7
SATURATION_WEIGHT = 0.2
VALUE_WEIGHT = 0.1

# If the closest kit color match is still farther than this, the player
# is labeled unclassified rather than force-assigned. Tune this if real
# players are landing in "unclassified" (raise it) or the referee is
# getting matched to a team (lower it).
UNCLASSIFIED_DISTANCE_THRESHOLD = 0.35

DRAW_COLORS = {
    "home": (255, 100, 40),        # blue-ish (BGR)
    "home_gk": (255, 200, 40),
    "away": (40, 40, 220),          # red-ish (BGR)
    "away_gk": (40, 140, 220),
    "unclassified": (180, 180, 180),  # gray
}


def hex_to_hsv(hex_color):
    """'#3D8B5F' -> (hue_degrees, saturation, value), all in intuitive ranges."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (h * 360, s, v)


def _sample_jersey_hsv(frame_bgr, x1, y1, x2, y2):
    """
    Crops the torso-ish region of a bounding box (avoiding head/hair at
    the top and shorts/legs at the bottom) and returns its median color
    in HSV. Returns None if the crop is degenerate (box too small/at
    frame edge).
    """
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


def _hsv_distance(hsv1, hsv2):
    """
    Weighted HSV distance, with one important correction: hue is only
    meaningful when a color is actually saturated. A near-black or
    near-gray referee kit has an almost arbitrary hue value (there's
    barely any "color" to have a direction), so if we weighted hue the
    same as for a vivid team kit, the referee could coincidentally land
    close to a team by pure hue-noise. Instead, hue's weight scales down
    with how UNSATURATED either color is, and that weight gets shifted
    onto saturation/value instead - which are the features that actually
    separate "vivid team color" from "black/gray officiating kit."
    """
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2

    dh = min(abs(h1 - h2), 360 - abs(h1 - h2)) / 180  # circular, normalized 0-1
    ds = abs(s1 - s2)
    dv = abs(v1 - v2)

    hue_reliability = min(s1, s2)  # low if either color is near-grayscale
    effective_hue_weight = HUE_WEIGHT * hue_reliability
    remaining_weight = 1 - effective_hue_weight
    other_total = SATURATION_WEIGHT + VALUE_WEIGHT
    effective_sat_weight = remaining_weight * (SATURATION_WEIGHT / other_total)
    effective_val_weight = remaining_weight * (VALUE_WEIGHT / other_total)

    return dh * effective_hue_weight + ds * effective_sat_weight + dv * effective_val_weight


def classify_teams(
    video_path,
    player_tracking_csv_path,
    annotated_output_path,
    classification_csv_path,
    home_color,
    away_color,
    home_gk_color=None,
    away_gk_color=None,
    ball_tracking_csv_path=None,
    progress_callback=None,
):
    """
    home_color/away_color/home_gk_color/away_gk_color: hex strings like
    "#3D8B5F". GK colors are optional - if not given, that reference is
    simply skipped (players won't be able to match it).

    Returns a summary dict, and writes:
      - annotated_output_path: video with boxes colored/labeled by team
      - classification_csv_path: one row per track_id with its assigned
        team and match confidence (1 - distance, roughly 0 to 1)
    """
    video_path = Path(video_path)
    player_tracking_csv_path = Path(player_tracking_csv_path)
    annotated_output_path = Path(annotated_output_path)
    classification_csv_path = Path(classification_csv_path)

    references = {}
    if home_color:
        references["home"] = hex_to_hsv(home_color)
    if away_color:
        references["away"] = hex_to_hsv(away_color)
    if home_gk_color:
        references["home_gk"] = hex_to_hsv(home_gk_color)
    if away_gk_color:
        references["away_gk"] = hex_to_hsv(away_gk_color)

    if not references:
        raise ValueError("At least one kit color (home or away) must be provided.")

    # ---- Load Stage 2's tracking output, grouped by frame for a single video pass ----
    detections_by_frame = defaultdict(list)  # frame_index -> [(track_id, x1,y1,x2,y2)]
    with open(player_tracking_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_index"])
            track_id = int(row["track_id"])
            x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
            detections_by_frame[frame_idx].append((track_id, x1, y1, x2, y2))

    ball_by_frame = {}  # frame_index -> (x, y, is_interpolated)
    if ball_tracking_csv_path is not None:
        with open(ball_tracking_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_idx = int(row["frame_index"])
                x, y = float(row["x"]), float(row["y"])
                is_interp = row["is_interpolated"] in ("True", "true", "1")
                ball_by_frame[frame_idx] = (x, y, is_interp)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---- Pass 1: sample jersey colors for every track, every frame it appears ----
    track_color_samples = defaultdict(list)
    frame_index = 0
    last_reported_pct = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for track_id, x1, y1, x2, y2 in detections_by_frame.get(frame_index, []):
            hsv = _sample_jersey_hsv(frame, x1, y1, x2, y2)
            if hsv is not None:
                track_color_samples[track_id].append(hsv)

        frame_index += 1
        if progress_callback and frame_count:
            pct = int((frame_index / frame_count) * 60)  # reserve 40% for pass 2
            if pct != last_reported_pct:
                progress_callback(pct)
                last_reported_pct = pct

    cap.release()

    # ---- Classify each track from its median color across all its samples ----
    track_assignments = {}  # track_id -> (label, confidence)
    for track_id, samples in track_color_samples.items():
        samples_arr = np.array(samples)
        median_hsv = tuple(np.median(samples_arr, axis=0))

        best_label, best_distance = None, float("inf")
        for label, ref_hsv in references.items():
            d = _hsv_distance(median_hsv, ref_hsv)
            if d < best_distance:
                best_label, best_distance = label, d

        if best_distance > UNCLASSIFIED_DISTANCE_THRESHOLD:
            track_assignments[track_id] = ("unclassified", round(1 - best_distance, 3))
        else:
            track_assignments[track_id] = (best_label, round(1 - best_distance, 3))

    with open(classification_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "team", "confidence", "sample_count"])
        for track_id, (label, confidence) in sorted(track_assignments.items()):
            writer.writerow([track_id, label, confidence, len(track_color_samples[track_id])])

    # ---- Pass 2: re-read video, draw team-colored/labeled boxes ----
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_out = cv2.VideoWriter(str(annotated_output_path), fourcc, fps, (width, height))

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for track_id, x1, y1, x2, y2 in detections_by_frame.get(frame_index, []):
            label, confidence = track_assignments.get(track_id, ("unclassified", 0))
            color = DRAW_COLORS.get(label, DRAW_COLORS["unclassified"])
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, color, 2)
            cv2.putText(frame, f"#{track_id} {label}", (p1[0], p1[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if frame_index in ball_by_frame:
            bx, by, is_interp = ball_by_frame[frame_index]
            ball_color = (0, 140, 255) if is_interp else (0, 215, 255)  # dimmer if estimated
            cv2.circle(frame, (int(bx), int(by)), 6, ball_color, -1 if not is_interp else 2)

        writer_out.write(frame)
        frame_index += 1

        if progress_callback and frame_count:
            pct = 60 + int((frame_index / frame_count) * 40)
            if pct != last_reported_pct:
                progress_callback(min(pct, 100))
                last_reported_pct = pct

    cap.release()
    writer_out.release()

    label_counts = defaultdict(int)
    for label, _ in track_assignments.values():
        label_counts[label] += 1

    summary = {
        "total_tracks_classified": len(track_assignments),
        **{f"count_{label}": count for label, count in label_counts.items()},
        "annotated_output_path": str(annotated_output_path),
        "classification_csv_path": str(classification_csv_path),
    }
    logger.info("Team classification summary: %s", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: classify tracked players by team via kit color.")
    parser.add_argument("video_path")
    parser.add_argument("players_csv", help="players.csv produced by Stage 2 (ai_engine.tracking)")
    parser.add_argument("output_dir")
    parser.add_argument("--home", required=True, help="Home team outfield kit color, e.g. #3D8B5F")
    parser.add_argument("--away", required=True, help="Away team outfield kit color, e.g. #274690")
    parser.add_argument("--home-gk", default=None, help="Home goalkeeper kit color (optional)")
    parser.add_argument("--away-gk", default=None, help="Away goalkeeper kit color (optional)")
    parser.add_argument("--ball-csv", default=None, help="ball.csv from Stage 2, to draw the ball too (optional)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    result = classify_teams(
        video_path=args.video_path,
        player_tracking_csv_path=args.players_csv,
        annotated_output_path=out_dir / "classified.mp4",
        classification_csv_path=out_dir / "classification.csv",
        home_color=args.home,
        away_color=args.away,
        home_gk_color=args.home_gk,
        away_gk_color=args.away_gk,
        ball_tracking_csv_path=args.ball_csv,
        progress_callback=lambda pct: print(f"\r{pct}%", end="", flush=True),
    )
    print()
    print("--- Team Classification Summary ---")
    for key, value in result.items():
        print(f"{key}: {value}")