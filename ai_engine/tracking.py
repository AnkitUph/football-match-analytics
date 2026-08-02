"""
Stage 2 of the CV pipeline: tracking.

Builds directly on Stage 1 (ai_engine.detection) - reuses the same pitch-
area filtering and split person/ball confidence thresholds, so a track
never gets created for a coach or ball boy in the first place.

Two different problems get solved here, deliberately differently:

  - PLAYERS: a real multi-object tracking problem. ByteTrack assigns a
    consistent track_id to each detected person across frames, surviving
    brief occlusion (players crossing paths, etc.).

  - BALL: not really a tracking problem - there's only one ball, so
    "identity" isn't in question. What actually matters is filling the
    gaps where it wasn't detected at all (small/fast/occluded object).
    This does that with straightforward linear interpolation, capped at
    a maximum gap length so a genuinely long absence (ball out of frame,
    stoppage) doesn't get faked into a fictional trajectory.

Still fully standalone - no Django/Celery/database involved. Run it by
hand against a clip and inspect the output before this gets wired into
the actual app pipeline.

    python -m ai_engine.tracking <input_video> <output_dir>
"""

import csv
import logging
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from ai_engine.detection import (
    COCO_BALL_CLASS,
    COCO_PERSON_CLASS,
    DEFAULT_MODEL,
    _compute_pitch_mask,
    _point_in_mask,
    _polygon_pitch_mask,
)

logger = logging.getLogger(__name__)

TRACK_BOX_COLOR = (60, 179, 113)      # green (BGR) - matches detection.py
BALL_COLOR_REAL = (0, 215, 255)        # solid gold - actual detection
BALL_COLOR_INTERPOLATED = (0, 140, 255)  # dimmer orange - estimated position


def _interpolate_ball_positions(raw_ball_points, max_gap_frames):
    """
    raw_ball_points: dict of {frame_index: (x, y, confidence)} for frames
    where the ball was actually detected.

    Returns a dict covering every frame from the first to last detection,
    filling gaps <= max_gap_frames with linearly interpolated (x, y) and
    confidence=None, is_interpolated=True. Gaps longer than that are left
    out entirely (position genuinely unknown for that stretch).
    """
    if not raw_ball_points:
        return {}

    known_frames = sorted(raw_ball_points.keys())
    filled = {}

    for i in range(len(known_frames) - 1):
        f1, f2 = known_frames[i], known_frames[i + 1]
        x1, y1, _ = raw_ball_points[f1]
        x2, y2, _ = raw_ball_points[f2]

        filled[f1] = (x1, y1, raw_ball_points[f1][2], False)

        gap = f2 - f1
        if 1 < gap <= max_gap_frames:
            for step in range(1, gap):
                t = step / gap
                x = x1 + (x2 - x1) * t
                y = y1 + (y2 - y1) * t
                filled[f1 + step] = (x, y, None, True)
        # else: gap too long (or zero) - leave those frames genuinely empty

    last_frame = known_frames[-1]
    x, y, conf = raw_ball_points[last_frame]
    filled[last_frame] = (x, y, conf, False)

    return filled


def track_video(
    video_path,
    annotated_output_path,
    player_tracking_csv_path,
    ball_tracking_csv_path,
    model_name=DEFAULT_MODEL,
    person_confidence=0.35,
    ball_confidence=0.10,
    filter_to_pitch=True,
    pitch_polygon=None,
    max_ball_gap_seconds=1.5,
    progress_callback=None,
):
    """
    Runs detection + player tracking + ball gap-filling over the full
    video. Returns a summary dict.
    """
    video_path = Path(video_path)
    annotated_output_path = Path(annotated_output_path)
    player_tracking_csv_path = Path(player_tracking_csv_path)
    ball_tracking_csv_path = Path(ball_tracking_csv_path)

    logger.info("Loading YOLO model: %s", model_name)
    model = YOLO(model_name)
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_ball_gap_frames = int(max_ball_gap_seconds * fps)

    pitch_mask = None
    if filter_to_pitch:
        if pitch_polygon is not None:
            pitch_mask = _polygon_pitch_mask((height, width), pitch_polygon)
            logger.info("Using manually specified pitch boundary.")
        else:
            ok, first_frame = cap.read()
            if ok:
                pitch_mask = _compute_pitch_mask(first_frame)
                if pitch_mask is None:
                    logger.warning(
                        "Could not confidently detect a pitch area - proceeding "
                        "WITHOUT sideline filtering. Consider passing "
                        "pitch_polygon= for a reliable manual boundary."
                    )
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ---- Pass 1: run detection + player tracking, collect raw ball points ----
    all_frames = []  # list of (frame, player_detections) per frame, kept in memory for pass 2 annotation
    player_rows = []  # (frame_index, timestamp, track_id, x1, y1, x2, y2, confidence)
    raw_ball_points = {}  # frame_index -> (x, y, confidence)

    frame_index = 0
    last_reported_pct = -1
    seen_track_ids = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(
            frame,
            classes=[COCO_PERSON_CLASS, COCO_BALL_CLASS],
            conf=ball_confidence,
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # Split person vs ball, apply the stricter person-confidence
        # threshold and the pitch-area filter, BEFORE handing anything
        # to the tracker - so sideline staff never get a track_id.
        person_mask = detections.class_id == COCO_PERSON_CLASS
        person_detections = detections[person_mask]
        person_detections = person_detections[person_detections.confidence >= person_confidence]

        if pitch_mask is not None and len(person_detections) > 0:
            keep = []
            for i in range(len(person_detections)):
                x1, y1, x2, y2 = person_detections.xyxy[i]
                foot_x, foot_y = (x1 + x2) / 2, y2
                keep.append(_point_in_mask(pitch_mask, foot_x, foot_y))
            person_detections = person_detections[np.array(keep, dtype=bool)]

        person_detections = tracker.update_with_detections(person_detections)

        timestamp = round(frame_index / fps, 2)
        for i in range(len(person_detections)):
            track_id = person_detections.tracker_id[i]
            x1, y1, x2, y2 = person_detections.xyxy[i]
            conf = float(person_detections.confidence[i])
            player_rows.append([
                frame_index, timestamp, int(track_id),
                round(float(x1), 1), round(float(y1), 1),
                round(float(x2), 1), round(float(y2), 1),
                round(conf, 3),
            ])
            seen_track_ids.add(int(track_id))

        # Ball: keep the single highest-confidence detection this frame, if any.
        ball_mask = detections.class_id == COCO_BALL_CLASS
        ball_detections = detections[ball_mask]
        if len(ball_detections) > 0:
            best_idx = int(np.argmax(ball_detections.confidence))
            x1, y1, x2, y2 = ball_detections.xyxy[best_idx]
            conf = float(ball_detections.confidence[best_idx])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            raw_ball_points[frame_index] = (cx, cy, conf)

        all_frames.append((frame_index, person_detections.xyxy.tolist() if len(person_detections) else [],
                            person_detections.tracker_id.tolist() if len(person_detections) else []))

        frame_index += 1

        if progress_callback and frame_count:
            # Reserve the last 20% of progress for the annotation pass below.
            pct = int((frame_index / frame_count) * 80)
            if pct != last_reported_pct:
                progress_callback(pct)
                last_reported_pct = pct

    total_frames = frame_index
    cap.release()

    # ---- Interpolate ball gaps ----
    ball_positions = _interpolate_ball_positions(raw_ball_points, max_ball_gap_frames)

    ball_rows = []
    for f in sorted(ball_positions.keys()):
        x, y, conf, is_interp = ball_positions[f]
        ball_rows.append([
            f, round(f / fps, 2), round(x, 1), round(y, 1),
            "" if conf is None else round(conf, 3), is_interp,
        ])

    with open(player_tracking_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "timestamp_seconds", "track_id", "x1", "y1", "x2", "y2", "confidence"])
        writer.writerows(player_rows)

    with open(ball_tracking_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "timestamp_seconds", "x", "y", "confidence", "is_interpolated"])
        writer.writerows(ball_rows)

    # ---- Pass 2: re-read the video to draw annotations (keeps memory low - we didn't hold frames) ----
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_out = cv2.VideoWriter(str(annotated_output_path), fourcc, fps, (width, height))

    for frame_index, boxes, track_ids in all_frames:
        ok, frame = cap.read()
        if not ok:
            break

        for (x1, y1, x2, y2), track_id in zip(boxes, track_ids):
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, TRACK_BOX_COLOR, 2)
            cv2.putText(frame, f"#{track_id}", (p1[0], p1[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TRACK_BOX_COLOR, 1, cv2.LINE_AA)

        if frame_index in ball_positions:
            x, y, _, is_interp = ball_positions[frame_index]
            color = BALL_COLOR_INTERPOLATED if is_interp else BALL_COLOR_REAL
            cv2.circle(frame, (int(x), int(y)), 6, color, -1 if not is_interp else 2)

        writer_out.write(frame)

        if progress_callback and total_frames:
            pct = 80 + int(((frame_index + 1) / total_frames) * 20)
            if pct != last_reported_pct:
                progress_callback(min(pct, 100))
                last_reported_pct = pct

    cap.release()
    writer_out.release()

    real_ball_frames = sum(1 for v in ball_positions.values() if not v[3])
    interpolated_ball_frames = sum(1 for v in ball_positions.values() if v[3])

    summary = {
        "frame_count": total_frames,
        "fps": fps,
        "duration_seconds": round(total_frames / fps, 1) if fps else None,
        "unique_player_tracks": len(seen_track_ids),
        "ball_frames_detected": real_ball_frames,
        "ball_frames_interpolated": interpolated_ball_frames,
        "ball_frames_unknown": total_frames - real_ball_frames - interpolated_ball_frames,
        "pct_ball_position_known": round(
            ((real_ball_frames + interpolated_ball_frames) / total_frames) * 100, 1
        ) if total_frames else 0,
        "annotated_output_path": str(annotated_output_path),
        "player_tracking_csv_path": str(player_tracking_csv_path),
        "ball_tracking_csv_path": str(ball_tracking_csv_path),
    }
    logger.info("Tracking summary: %s", summary)
    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (3, 4):
        print("Usage: python -m ai_engine.tracking <video_path> <output_dir> [polygon]")
        print()
        print('  polygon is optional: a space-separated list of "x,y" pixel')
        print('  points tracing the pitch boundary, e.g.:')
        print('    python -m ai_engine.tracking clip.mp4 out/ "120,80 1150,75 1200,640 60,650"')
        print()
        print("  Get these coordinates by running ai_engine.extract_frame first")
        print("  and reading pixel positions off the saved image.")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    polygon = None
    if len(sys.argv) == 4:
        polygon = [
            tuple(int(v) for v in pair.split(","))
            for pair in sys.argv[3].split()
        ]
        print(f"Using manual pitch boundary: {polygon}")

    logging.basicConfig(level=logging.INFO)

    result = track_video(
        video_path=in_path,
        annotated_output_path=out_dir / "tracked.mp4",
        player_tracking_csv_path=out_dir / "players.csv",
        ball_tracking_csv_path=out_dir / "ball.csv",
        pitch_polygon=polygon,
        progress_callback=lambda pct: print(f"\r{pct}%", end="", flush=True),
    )
    print()
    print("--- Tracking Summary ---")
    for key, value in result.items():
        print(f"{key}: {value}")
    print()
    print(f"Watch {out_dir / 'tracked.mp4'} to judge quality before moving to Stage 3+ (team classification).")