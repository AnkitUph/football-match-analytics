"""
One-off utility: pulls a single frame out of a video and saves it as a
PNG, so you can open it in any image viewer and read off the pixel
coordinates of the pitch's corners (or any boundary you want to define).

Usage:
    python -m ai_engine.extract_frame <video_path> <output_png> [frame_number]

frame_number defaults to 0 (the first frame). Pick a frame where the
camera angle is representative of the whole video (i.e. not mid-zoom or
mid-pan, if the camera moves).

Once you have the PNG, open it in an image viewer that shows pixel
coordinates on hover (GIMP, or most OS "preview" tools, or even just
loading it into a Jupyter cell with matplotlib) and note the (x, y) pixel
position of each corner of the playing area, in order around the
boundary. Pass that list to detect_video()/track_video() as
pitch_polygon=[(x1,y1), (x2,y2), (x3,y3), (x4,y4)].
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
    print("Open it and note the pixel (x, y) coordinates of the pitch's corners.")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python -m ai_engine.extract_frame <video_path> <output_png> [frame_number]")
        sys.exit(1)

    video_arg = sys.argv[1]
    output_arg = sys.argv[2]
    frame_arg = int(sys.argv[3]) if len(sys.argv) == 4 else 0

    extract_frame(video_arg, output_arg, frame_arg)