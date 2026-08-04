"""
Stage 1 of the CV pipeline: player/ball detection.

Standalone - no Django/Celery/database. Run it by hand against a clip
before trusting it in the full pipeline:

    python -m ai_engine.detection <video_path> <output_dir>

Design decisions baked in from earlier testing (kept here so the reasons
aren't lost):

  - imgsz=1280 (not YOLO's 640 default): broadcast/wide-angle football
    footage has small/distant players that get lost entirely at 640px -
    there's no detection to threshold-tune if the resolution already
    destroyed the pixel detail. This was the actual fix for missed
    small/edge players, not a confidence-threshold change.

  - Split person/ball confidence: the ball needs a much looser threshold
    than a person (small, blurry, low model confidence even when
    genuinely detected). Inference runs at the loose ball threshold,
    then person detections are filtered stricter afterward.

  - Pitch-area filtering, two modes: automatic (green color detection)
    or manual polygon. Automatic FAILS when the technical area shares
    the pitch's turf color (common at local venues) - color segmentation
    literally cannot separate two regions of the same color. Manual
    polygon is the reliable fallback for that case.
"""

import csv
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

COCO_PERSON_CLASS = 0
COCO_BALL_CLASS = 32  # "sports ball"

DEFAULT_MODEL = "yolov8s.pt"
DEFAULT_IMGSZ = 1280

BOX_COLOR_PERSON = (60, 179, 113)
BOX_COLOR_BALL = (0, 215, 255)


def _compute_pitch_mask(frame):
    """
    Automatic pitch detection via green color threshold + largest
    connected region. FAILS (returns None) if the technical area shares
    the pitch's turf color - no color threshold can separate two regions
    of identical color. Use _polygon_pitch_mask() in that case.
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
    pitch_mask = cv2.dilate(pitch_mask, np.ones((7, 7), np.uint8))
    return pitch_mask


def _polygon_pitch_mask(frame_shape, polygon_points, buffer_pixels=25):
    """
    Manual pitch boundary from corner points. Use ai_engine.extract_frame
    + ai_engine.zoom_frame to find accurate coordinates.

    buffer_pixels expands the mask outward so players legitimately near
    the touchline (throw-ins, corners) aren't clipped just for being a
    few pixels outside the traced boundary.
    """
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    points = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)

    if buffer_pixels > 0:
        kernel_size = buffer_pixels * 2 + 1
        mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8))

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
    imgsz=DEFAULT_IMGSZ,
    progress_callback=None,
):
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
                        "WITHOUT sideline filtering. Consider pitch_polygon= "
                        "for a reliable manual boundary."
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
                conf=ball_confidence,
                imgsz=imgsz,
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

                if not is_ball:
                    if conf < person_confidence:
                        continue
                    if pitch_mask is not None:
                        foot_x, foot_y = (x1 + x2) / 2, y2
                        if not _point_in_mask(pitch_mask, foot_x, foot_y):
                            sideline_filtered_count += 1
                            continue

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

    if len(sys.argv) not in (3, 4):
        print("Usage: python -m ai_engine.detection <video_path> <output_dir> [polygon]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    polygon = None
    if len(sys.argv) == 4:
        polygon = [tuple(int(v) for v in pair.split(",")) for pair in sys.argv[3].split()]

    logging.basicConfig(level=logging.INFO)

    result = detect_video(
        video_path=in_path,
        annotated_output_path=out_dir / "detected.mp4",
        detections_csv_path=out_dir / "detections.csv",
        pitch_polygon=polygon,
        progress_callback=lambda pct: print(f"\r{pct}%", end="", flush=True),
    )
    print()
    for key, value in result.items():
        print(f"{key}: {value}")