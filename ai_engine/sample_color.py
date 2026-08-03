"""
One-off utility: reads the color at a specific pixel (or a small
averaged patch around it) in a saved frame, and prints it as a hex
string ready to paste into team_classification.py's --home/--away/etc.
arguments.

Usage:
    python -m ai_engine.sample_color <image_path> <x> <y> [patch_size]

patch_size (default 15) averages a small square around that point
instead of a single pixel, to avoid picking up noise/compression
artifacts from one unlucky pixel.

Workflow:
    1. Run ai_engine.extract_frame to get a still frame (frame.png)
    2. Open it in any image viewer, find a player's jersey, note the
       approximate (x, y) pixel location
    3. Run this script with that coordinate to get the real hex color
    4. Repeat for home outfield, away outfield, home GK, away GK
    5. Use those real values with ai_engine.team_classification
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def sample_color(image_path, x, y, patch_size=15):
    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    h, w = frame.shape[:2]
    half = patch_size // 2
    x1, x2 = max(0, x - half), min(w, x + half + 1)
    y1, y2 = max(0, y - half), min(h, y + half + 1)

    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        raise ValueError(f"Coordinates ({x}, {y}) are outside the image ({w}x{h})")

    # BGR (OpenCV's order) -> average -> hex
    b, g, r = np.median(patch.reshape(-1, 3), axis=0)
    hex_color = f"#{int(r):02X}{int(g):02X}{int(b):02X}"

    print(f"Sampled at ({x}, {y}), {patch_size}x{patch_size} patch: {hex_color}")
    print(f"  RGB: ({int(r)}, {int(g)}, {int(b)})")
    return hex_color


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print("Usage: python -m ai_engine.sample_color <image_path> <x> <y> [patch_size]")
        sys.exit(1)

    img_path = sys.argv[1]
    x_coord = int(sys.argv[2])
    y_coord = int(sys.argv[3])
    patch = int(sys.argv[4]) if len(sys.argv) == 5 else 15

    sample_color(img_path, x_coord, y_coord, patch)