"""
Stage 4 of the CV pipeline: team classification.

Hybrid design, combining two approaches for two different problems:

  1. OUTFIELD PLAYERS (the bulk of the roster): K-means (k=2) discovers
     the two dominant jersey colors that actually exist in this specific
     video, rather than requiring an exact pre-specified hex color to
     match against. This is far more robust to real-world variation -
     lighting, camera quality, slightly-off reference colors - than
     matching every player against one fixed target color. The two
     discovered cluster centers are then labeled "home"/"away" by
     comparing them to the match's stored kit colors (used as LABELS,
     not as strict matching targets).

  2. GOALKEEPERS: there are only ever 2 GK tracks in a whole match - far
     too few for K-means to reliably treat as their own cluster (they'd
     just get absorbed into whichever outfield cluster is nearest, which
     is usually wrong, since GK kits are deliberately different from
     everyone else). Instead, GK tracks are identified FIRST via direct
     distance matching against the stored GK colors, and excluded from
     the outfield clustering pool entirely.

  3. SAFETY NET: after clustering, any track whose color is still far
     from ITS OWN assigned cluster's center (not just far from the DB
     color) is relabeled "unclassified" rather than force-kept. K-means
     with k=2 always assigns every point to one of two groups regardless
     of fit quality - this catches the referee (a genuine third color
     group) who would otherwise get silently folded into whichever team
     cluster happens to be nearest.

  4. Tracks with too few color samples (a short-lived/unstable track_id,
     which can happen if Stage 2 loses and reassigns a player mid-clip)
     are labeled "insufficient_data" rather than classified on weak
     evidence.

Consumes Stage 2's output (players.csv, ball.csv) - detection/tracking
already happened, no need to re-run YOLO.

Standalone - no Django/Celery/database:

    python -m ai_engine.team_classification \\
        <video_path> <players_csv> <output_dir> \\
        --home "#RRGGBB" --away "#RRGGBB" \\
        --home-gk "#RRGGBB" --away-gk "#RRGGBB" \\
        --ball-csv <ball_csv>

home/away/home-gk/away-gk should be the match's actual stored kit colors
(e.g. from the upload form) - they're used to LABEL what K-means
discovers, not as the primary matching mechanism, so they don't need to
be pixel-perfect.
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

HUE_WEIGHT = 0.7
SATURATION_WEIGHT = 0.2
VALUE_WEIGHT = 0.1

# Tracks with fewer color samples than this aren't classified - too
# little evidence to trust, regardless of what the color says.
MIN_SAMPLES_FOR_CLASSIFICATION = 5

# GK colors are deliberately distinctive, so this can be strict - if a
# track isn't clearly close to a GK color, it goes into the outfield
# clustering pool instead rather than being force-matched as a GK.
GK_MATCH_THRESHOLD = 0.20

# After clustering, if a track is farther than this from its OWN
# cluster's center, it's relabeled unclassified - this is what catches
# the referee (a real third color group that k=2 clustering would
# otherwise silently force into a team).
OUTLIER_THRESHOLD = 0.28

DRAW_COLORS = {
    "home": (255, 100, 40),
    "home_gk": (255, 200, 40),
    "away": (40, 40, 220),
    "away_gk": (40, 140, 220),
    "team_0": (200, 200, 40),
    "team_1": (40, 200, 200),
    "unclassified": (180, 180, 180),
    "insufficient_data": (100, 100, 100),
}


def hex_to_hsv(hex_color):
    if not hex_color:
        return None
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (h * 360, s, v)


def _rgb_to_hsv_deg(rgb):
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return (h * 360, s, v)


def _sample_jersey_rgb(frame_bgr, x1, y1, x2, y2):
    """Crops the torso region and returns its median color as normalized (r, g, b)."""
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
    return tuple(np.median(crop_rgb, axis=0))


def _hsv_distance(hsv1, hsv2):
    """
    Weighted HSV distance, hue down-weighted for low-saturation colors
    (near-black/gray/white) since hue is unreliable/near-meaningless for
    such colors - without this, a black referee kit could coincidentally
    match a team by hue-noise alone.
    """
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2

    dh = min(abs(h1 - h2), 360 - abs(h1 - h2)) / 180
    ds = abs(s1 - s2)
    dv = abs(v1 - v2)

    hue_reliability = min(s1, s2)
    effective_hue_weight = HUE_WEIGHT * hue_reliability
    remaining = 1 - effective_hue_weight
    other_total = SATURATION_WEIGHT + VALUE_WEIGHT
    eff_sat = remaining * (SATURATION_WEIGHT / other_total)
    eff_val = remaining * (VALUE_WEIGHT / other_total)

    return dh * effective_hue_weight + ds * eff_sat + dv * eff_val


def _assign_cluster_labels(centers_rgb, home_hsv, away_hsv):
    """
    Labels the 2 K-means cluster centers as "home"/"away" by finding
    whichever pairing (0->home,1->away vs 0->away,1->home) has the
    lower total distance - guarantees the two clusters get DIFFERENT
    labels, rather than both greedily matching the same closer color.
    """
    if home_hsv is None or away_hsv is None:
        return {0: "team_0", 1: "team_1"}

    center_hsvs = [_rgb_to_hsv_deg(c) for c in centers_rgb]

    cost_a = _hsv_distance(center_hsvs[0], home_hsv) + _hsv_distance(center_hsvs[1], away_hsv)
    cost_b = _hsv_distance(center_hsvs[0], away_hsv) + _hsv_distance(center_hsvs[1], home_hsv)

    if cost_a <= cost_b:
        return {0: "home", 1: "away"}
    return {0: "away", 1: "home"}


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
    video_path = Path(video_path)
    player_tracking_csv_path = Path(player_tracking_csv_path)
    annotated_output_path = Path(annotated_output_path)
    classification_csv_path = Path(classification_csv_path)

    home_hsv = hex_to_hsv(home_color)
    away_hsv = hex_to_hsv(away_color)
    gk_references = {}
    if home_gk_color:
        gk_references["home_gk"] = hex_to_hsv(home_gk_color)
    if away_gk_color:
        gk_references["away_gk"] = hex_to_hsv(away_gk_color)

    # ---- Load Stage 2's output ----
    detections_by_frame = defaultdict(list)
    with open(player_tracking_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_index"])
            track_id = int(row["track_id"])
            x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
            detections_by_frame[frame_idx].append((track_id, x1, y1, x2, y2))

    ball_by_frame = {}
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

    # ---- Pass 1: sample jersey color for every track, every frame ----
    track_color_samples = defaultdict(list)
    frame_index = 0
    last_reported_pct = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for track_id, x1, y1, x2, y2 in detections_by_frame.get(frame_index, []):
            rgb = _sample_jersey_rgb(frame, x1, y1, x2, y2)
            if rgb is not None:
                track_color_samples[track_id].append(rgb)

        frame_index += 1
        if progress_callback and frame_count:
            pct = int((frame_index / frame_count) * 60)
            if pct != last_reported_pct:
                progress_callback(pct)
                last_reported_pct = pct

    cap.release()

    # ---- Per-track median color (RGB for clustering, HSV for distance checks) ----
    track_median_rgb = {}
    track_median_hsv = {}
    track_assignments = {}

    for track_id, samples in track_color_samples.items():
        if len(samples) < MIN_SAMPLES_FOR_CLASSIFICATION:
            track_assignments[track_id] = ("insufficient_data", None)
            continue
        med_rgb = tuple(np.median(np.array(samples), axis=0))
        track_median_rgb[track_id] = med_rgb
        track_median_hsv[track_id] = _rgb_to_hsv_deg(med_rgb)

    # ---- Step 1: direct GK matching, excluded from the outfield clustering pool ----
    outfield_candidates = []
    for track_id, hsv in track_median_hsv.items():
        best_label, best_distance = None, float("inf")
        for label, ref_hsv in gk_references.items():
            d = _hsv_distance(hsv, ref_hsv)
            if d < best_distance:
                best_label, best_distance = label, d

        if best_label is not None and best_distance <= GK_MATCH_THRESHOLD:
            track_assignments[track_id] = (best_label, round(1 - best_distance, 3))
        else:
            outfield_candidates.append(track_id)

    # ---- Step 2: K-means discovers the two outfield team colors ----
    if len(outfield_candidates) >= 2:
        rgb_data = np.float32([track_median_rgb[tid] for tid in outfield_candidates])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(rgb_data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()

        cluster_label_map = _assign_cluster_labels(centers, home_hsv, away_hsv)

        for i, track_id in enumerate(outfield_candidates):
            cluster_idx = int(labels[i])
            centroid_hsv = _rgb_to_hsv_deg(centers[cluster_idx])
            own_hsv = track_median_hsv[track_id]
            centroid_distance = _hsv_distance(own_hsv, centroid_hsv)

            if centroid_distance > OUTLIER_THRESHOLD:
                track_assignments[track_id] = ("unclassified", round(1 - centroid_distance, 3))
            else:
                track_assignments[track_id] = (cluster_label_map[cluster_idx], round(1 - centroid_distance, 3))
    else:
        for track_id in outfield_candidates:
            track_assignments[track_id] = ("unclassified", None)

    with open(classification_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "team", "confidence", "sample_count"])
        for track_id, (label, confidence) in sorted(track_assignments.items()):
            sample_count = len(track_color_samples.get(track_id, []))
            writer.writerow([track_id, label, confidence, sample_count])

    # ---- Pass 2: draw annotated video ----
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
            ball_color = (0, 140, 255) if is_interp else (0, 215, 255)
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
        "total_tracks": len(track_assignments),
        **{f"count_{label}": count for label, count in label_counts.items()},
        "annotated_output_path": str(annotated_output_path),
        "classification_csv_path": str(classification_csv_path),
    }
    logger.info("Team classification summary: %s", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: classify tracked players by team.")
    parser.add_argument("video_path")
    parser.add_argument("players_csv")
    parser.add_argument("output_dir")
    parser.add_argument("--home", required=True, help="Home team outfield kit color, e.g. #3D8B5F")
    parser.add_argument("--away", required=True, help="Away team outfield kit color, e.g. #274690")
    parser.add_argument("--home-gk", default=None)
    parser.add_argument("--away-gk", default=None)
    parser.add_argument("--ball-csv", default=None)
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
    for key, value in result.items():
        print(f"{key}: {value}")