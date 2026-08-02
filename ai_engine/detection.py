"""
Stage 1 of the CV pipeline, and ONLY Stage 1: raw player/ball detection.

Deliberately does nothing else. No tracking, no IDs across frames, no
Django/Celery integration. The point of this script is to answer one
question before anything else gets built: is YOLO actually detecting
players and the ball reliably in real match footage?

Run it standalone against a short clip and watch the output video:

    python -m ai_engine.detection <input_video> <output_dir>

If detection quality looks good (players consistently boxed, ball
detected at least some of the time), Stage 2 (tracking) gets built next.
If it doesn't, the fix belongs here - trying a bigger model, adjusting
confidence, fine-tuning - not in whatever gets built on top of it.
"""

import csv
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class IDs used by the pretrained YOLO model.
COCO_PERSON_CLASS = 0
COCO_BALL_CLASS = 32  # "sports ball"

# Start with the small model - good balance of speed/accuracy. If
# detection quality is poor, try yolov8m.pt (slower, more accurate)
# before assuming the whole approach doesn't work.
DEFAULT_MODEL = "yolov8s.pt"

BOX_COLOR_PERSON = (60, 179, 113)   # green (BGR)
BOX_COLOR_BALL = (0, 215, 255)      # gold (BGR)


def _compute_pitch_mask(frame):
    """
    Approximates "where the grass is" using a green color threshold, then
    keeps only the single largest connected green region (the pitch
    itself, not stray green in the crowd/ads).

    LIMITATION: this is a color-based heuristic. If the technical area
    (where coaches/staff stand) is the same turf color as the playing
    field - common at local/non-professional venues - this CANNOT
    distinguish between them, because there's no color difference to
    detect. In that case, use _polygon_pitch_mask() instead with a
    manually defined boundary.

    Returns a binary mask the same size as the frame, or None if no
    plausible pitch region was found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    frame_area = frame.shape[0] * frame.shape[1]
    if cv2.contourArea(largest) < frame_area * 0.15:
        return None

    pitch_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(pitch_mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Small buffer only - just enough that a player's feet right at the
    # touchline (throw-ins, corners) aren't clipped. A large buffer here
    # is exactly what lets nearby technical-area staff slip back in.
    pitch_mask = cv2.dilate(pitch_mask, np.ones((7, 7), np.uint8))

    return pitch_mask


def _polygon_pitch_mask(frame_shape, polygon_points):
    """
    Builds a pitch mask from manually specified corner points instead of
    color detection. Use this when the technical area shares the pitch's
    turf color, since no color threshold can separate them in that case.

    polygon_points: list of (x, y) pixel coordinates tracing the
    playing-area boundary, in order (e.g. the four corners of the pitch
    as seen from the camera angle). Get these by extracting a still frame
    (see ai_engine/extract_frame.py) and reading off pixel coordinates in
    any image viewer.
    """
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    points = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def _point_in_mask(mask, x, y):
    h, w = mask.shape
    xi, yi = int(x), int(y)
    if 0 <= xi < w and 0 <= yi < h:
        return mask[yi, xi] > 0
    return False


def detect_video(
    video_path,
    annotated_output_path,
    detections_csv_path,
    model_name=DEFAULT_MODEL,
    person_confidence=0.35,
    ball_confidence=0.10,
    filter_to_pitch=True,
    pitch_polygon=None,
    progress_callback=None,
):
    """
    Runs detection only (no tracking) over every frame of the video.

    Draws boxes on an output video and writes every detection to a CSV
    (frame_index, timestamp, class_name, x1, y1, x2, y2, confidence).

    progress_callback, if given, is called with an int 0-100 as frames
    are processed.

    Returns a summary dict - frame count, average detections per frame
    for each class, etc. - useful for judging quality at a glance before
    even watching the video.
    """
    video_path = Path(video_path)
    annotated_output_path = Path(annotated_output_path)
    detections_csv_path = Path(detections_csv_path)

    logger.info("Loading YOLO model: %s", model_name)
    model = YOLO(model_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_output_path), fourcc, fps, (width, height))

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
                        "WITHOUT sideline filtering. Coaches/ball boys/staff "
                        "near the pitch may get detected as players. Consider "
                        "passing pitch_polygon= for a reliable manual boundary."
                    )
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_index = 0
    last_reported_pct = -1
    person_detection_count = 0
    ball_detection_count = 0
    frames_with_ball = 0
    sideline_filtered_count = 0

    with open(detections_csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["frame_index", "timestamp_seconds", "class_name", "x1", "y1", "x2", "y2", "confidence"]
        )

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model(
                frame,
                classes=[COCO_PERSON_CLASS, COCO_BALL_CLASS],
                conf=ball_confidence,  # loosest threshold; person gets filtered stricter below
                verbose=False,
            )[0]

            timestamp = round(frame_index / fps, 2)
            ball_seen_this_frame = False

            for box in results.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

                is_ball = class_id == COCO_BALL_CLASS
                class_name = "ball" if is_ball else "person"

                if not is_ball and conf < person_confidence:
                    continue  # below the stricter person threshold - inference ran at ball_confidence

                if not is_ball and pitch_mask is not None:
                    # Use the bottom-center of the box (feet position) to
                    # decide if this person is standing on the pitch.
                    foot_x = (x1 + x2) / 2
                    foot_y = y2
                    if not _point_in_mask(pitch_mask, foot_x, foot_y):
                        sideline_filtered_count += 1
                        continue  # skip: likely a coach/ball boy/staff member

                if is_ball:
                    ball_detection_count += 1
                    ball_seen_this_frame = True
                    color = BOX_COLOR_BALL
                else:
                    person_detection_count += 1
                    color = BOX_COLOR_PERSON

                csv_writer.writerow([
                    frame_index, timestamp, class_name,
                    round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                    round(conf, 3),
                ])

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

            if ball_seen_this_frame:
                frames_with_ball += 1

            writer.write(frame)
            frame_index += 1

            if progress_callback and frame_count:
                pct = int((frame_index / frame_count) * 100)
                if pct != last_reported_pct:
                    progress_callback(pct)
                    last_reported_pct = pct

    cap.release()
    writer.release()

    summary = {
        "frame_count": frame_index,
        "fps": fps,
        "duration_seconds": round(frame_index / fps, 1) if fps else None,
        "total_person_detections": person_detection_count,
        "total_ball_detections": ball_detection_count,
        "sideline_detections_filtered": sideline_filtered_count,
        "pitch_filtering_active": pitch_mask is not None,
        "avg_persons_per_frame": round(person_detection_count / frame_index, 2) if frame_index else 0,
        "pct_frames_with_ball_detected": round((frames_with_ball / frame_index) * 100, 1) if frame_index else 0,
        "annotated_output_path": str(annotated_output_path),
        "detections_csv_path": str(detections_csv_path),
    }
    logger.info("Detection summary: %s", summary)
    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m ai_engine.detection <video_path> <output_dir>")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    result = detect_video(
        video_path=in_path,
        annotated_output_path=out_dir / "detected.mp4",
        detections_csv_path=out_dir / "detections.csv",
        progress_callback=lambda pct: print(f"\r{pct}%", end="", flush=True),
    )
    print()
    print("--- Detection Summary ---")
    for key, value in result.items():
        print(f"{key}: {value}")
    print()
    print(f"Watch {out_dir / 'detected.mp4'} to judge quality before moving to Stage 2 (tracking).")