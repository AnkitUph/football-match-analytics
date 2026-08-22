"""
Full-clip test: Stages 1-4 (detection, tracking, team classification,
ball interpolation) across the ENTIRE video, not just a segment.

These stages don't depend on any camera calibration, so unlike Stage
5/6, this genuinely validates the whole clip end to end regardless of
how many camera cuts/angles it contains.

Usage (inside your Docker container):
    docker compose exec web python -m ai_engine.test_full_clip_stages1_4 \
        media/test_clips/test_11.mp4
"""

import argparse
from collections import Counter, defaultdict

import cv2

from ai_engine.config import DEFAULT_CONFIG
from ai_engine.stage2_tracking.tracker import Tracker
from ai_engine.stage3_team_reid.team_classifier import (
    classify_team,
    fit_team_color_clusters,
    sample_torso_color,
)
from ai_engine.stage4_ball_tracking.ball_tracker import extract_ball_detections, interpolate_gaps


def main(args):
    print("Stage 1+2: detection + tracking across the FULL clip...")
    tracker = Tracker(DEFAULT_CONFIG.detection, DEFAULT_CONFIG.tracking)
    tracklets = tracker.track_video(args.input_video)
    print(f"  {len(tracklets)} raw tracklets")

    by_class = Counter(t.cls.value if t.cls else "unknown" for t in tracklets.values())
    print(f"  by class: {dict(by_class)}")

    min_duration_frames = int(DEFAULT_CONFIG.pitch_mapping.min_tracklet_duration_sec * 25)
    survivors = {tid: t for tid, t in tracklets.items() if t.duration_frames >= min_duration_frames}
    print(f"  {len(survivors)} survive duration filter (>= {DEFAULT_CONFIG.pitch_mapping.min_tracklet_duration_sec}s)")

    durations = sorted((t.duration_frames for t in survivors.values()), reverse=True)
    print(f"  longest tracklet: {durations[0] if durations else 0} frames")
    print(f"  shortest surviving: {durations[-1] if durations else 0} frames")

    print()
    print("Stage 3: team classification across all survivors...")
    cap = cv2.VideoCapture(args.input_video)
    colors, valid = {}, []
    for t in survivors.values():
        det = t.detections[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, det.frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = map(int, det.bbox)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        color = sample_torso_color(crop)
        if color is not None:
            colors[t.track_id] = color
            valid.append(t)
    clusters = fit_team_color_clusters(list(colors.values()), DEFAULT_CONFIG.team_reid)
    for t in valid:
        t.team = classify_team(t.cls.value if t.cls else "player", colors[t.track_id], clusters)
    team_counts = Counter(t.team.value for t in valid)
    print(f"  team split: {dict(team_counts)}")

    print()
    print("Stage 4: ball extraction + interpolation across the FULL clip...")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_detections_by_frame = defaultdict(list)
    for t in tracklets.values():
        for det in t.detections:
            all_detections_by_frame[det.frame_idx].append(det)
    ball_by_frame = extract_ball_detections(dict(all_detections_by_frame))
    ball_trajectory = interpolate_gaps(ball_by_frame, DEFAULT_CONFIG.ball_tracking, frame_w, frame_h)

    real = sum(1 for p in ball_trajectory if p.x_m is not None and not p.interpolated)
    interp = sum(1 for p in ball_trajectory if p.interpolated)
    missing = sum(1 for p in ball_trajectory if p.x_m is None)
    off_bounds = sum(
        1 for p in ball_trajectory
        if p.x_m is not None and not (0 <= p.x_m <= frame_w and 0 <= p.y_m <= frame_h)
    )
    print(f"  real: {real}, interpolated: {interp}, missing: {missing}")
    print(f"  points outside frame bounds (should be 0): {off_bounds}")

    print()
    print("FULL CLIP TEST COMPLETE — Stages 1-4, no errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    args = parser.parse_args()
    main(args)
