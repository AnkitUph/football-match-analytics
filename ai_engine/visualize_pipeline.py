"""
Visualization script: runs Stages 1-4 on a real clip and renders an
annotated output video, so you can visually sanity-check detection,
tracking, team classification, and ball interpolation together before
trusting the numbers alone.

Usage (run inside your Docker container):
    docker compose exec web python -m ai_engine.visualize_pipeline \
        media/test_clips/test_11.mp4 \
        media/test_clips/annotated_output.mp4 \
        --start 0 --end 250

Then pull the result out to view on your host:
    docker cp football_web:/app/media/test_clips/annotated_output.mp4 ./annotated_output.mp4

Note: this does NOT run Stage 5's stitching/gallery step — track IDs
shown are Stage 2's raw per-shot IDs (may briefly appear/disappear on
short fragments), not final Master IDs. This is intentional: the point
here is to visually check Stages 1-4 individually. Add stitching to this
script yourself once you want to preview Stage 5's consolidated output
instead.
"""

import argparse
from collections import defaultdict

import cv2

from ai_engine.config import DEFAULT_CONFIG
from ai_engine.stage2_tracking.tracker import Tracker
from ai_engine.stage3_team_reid.team_classifier import (
    classify_team,
    fit_team_color_clusters,
    sample_torso_color,
)
from ai_engine.stage4_ball_tracking.ball_tracker import extract_ball_detections, interpolate_gaps

TEAM_COLORS = {
    "team_a": (255, 255, 255),
    "team_b": (0, 255, 100),
    "referee": (0, 255, 255),
    "unknown": (128, 128, 128),
}


def build_frame_annotations(tracklets, min_duration_frames, input_video_path):
    """Groups detections by frame, keeping only tracklets that survive
    the same duration filter Stage 5 uses — cuts obvious noise from the
    preview without needing the full stitching pipeline."""
    survivors = {tid: t for tid, t in tracklets.items() if t.duration_frames >= min_duration_frames}

    cap = cv2.VideoCapture(input_video_path)
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
    cap.release()

    if colors:
        clusters = fit_team_color_clusters(list(colors.values()), DEFAULT_CONFIG.team_reid)
        for t in valid:
            t.team = classify_team(t.cls.value if t.cls else "player", colors[t.track_id], clusters)

    frame_annotations = defaultdict(list)
    for t in valid:
        for det in t.detections:
            frame_annotations[det.frame_idx].append((det.bbox, t.track_id, t.team.value))
    return frame_annotations


def main(args):
    tracker = Tracker(DEFAULT_CONFIG.detection, DEFAULT_CONFIG.tracking)
    print("Running detection + tracking (Stage 1+2)...")
    tracklets = tracker.track_video(args.input_video)
    print(f"  {len(tracklets)} raw tracklets")

    print("Classifying teams (Stage 3)...")
    min_duration_frames = int(DEFAULT_CONFIG.pitch_mapping.min_tracklet_duration_sec * 25)
    frame_annotations = build_frame_annotations(tracklets, min_duration_frames, args.input_video)

    print("Extracting + interpolating ball trajectory (Stage 4)...")
    all_detections_by_frame = defaultdict(list)
    for t in tracklets.values():
        for det in t.detections:
            all_detections_by_frame[det.frame_idx].append(det)
    ball_by_frame = extract_ball_detections(dict(all_detections_by_frame))
    ball_trajectory = interpolate_gaps(ball_by_frame, DEFAULT_CONFIG.ball_tracking)
    ball_lookup = {p.frame_idx: (p.x_m, p.y_m, p.interpolated) for p in ball_trajectory if p.x_m is not None}

    print("Rendering annotated video...")
    cap = cv2.VideoCapture(args.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = args.start
    end = args.end if args.end is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (w, h))

    ball_trail = []
    for frame_idx in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break

        for bbox, track_id, team in frame_annotations.get(frame_idx, []):
            x1, y1, x2, y2 = map(int, bbox)
            color = TEAM_COLORS.get(team, (128, 128, 128))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, str(track_id), (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        nearest = min(ball_lookup.keys(), key=lambda k: abs(k - frame_idx)) if ball_lookup else None
        if nearest is not None and abs(nearest - frame_idx) <= 2:
            bx, by, interp = ball_lookup[nearest]
            ball_trail.append((int(bx), int(by)))
            color = (0, 140, 255) if interp else (0, 0, 255)
            cv2.circle(frame, (int(bx), int(by)), 8, color, -1)

        trail = ball_trail[-15:]
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Done: {args.output_video}")
    print("White = Team A, Green = Team B, Yellow = Referee")
    print("Red dot = real ball detection, Orange dot = interpolated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("output_video")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    main(args)
