"""
Pulls a single frame out of a video and saves it as a PNG, so you can
find pitch-boundary corner coordinates for ai_engine.detection's/
ai_engine.tracking's pitch_polygon parameter.

Usage:
    python -m ai_engine.extract_frame <video_path> <output_png> [frame_number]
"""

import sys
from pathlib import Path

import cv2


def extract_frame(video_path, output_path, frame_number=0):
    video_path = Path(video_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number} from {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    height, width = frame.shape[:2]
    print(f"Saved frame {frame_number} ({width}x{height}) to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python -m ai_engine.extract_frame <video_path> <output_png> [frame_number]")
        sys.exit(1)

    video_arg = sys.argv[1]
    output_arg = sys.argv[2]
    frame_arg = int(sys.argv[3]) if len(sys.argv) == 4 else 0

    extract_frame(video_arg, output_arg, frame_arg)