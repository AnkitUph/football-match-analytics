"""
One-off utility: crops a small region around a rough (x, y) guess, blows
it up to a large size, and overlays a coordinate grid labeled with the
ORIGINAL image's pixel coordinates. Solves the problem of trying to
precisely click a player who's only ~20px tall at normal zoom - you get
a big, sharp, labeled view instead.

Usage:
    python -m ai_engine.zoom_frame <image_path> <output_path> <center_x> <center_y> [region_size] [scale]

    region_size: size (in ORIGINAL pixels) of the square region to crop
                 around (center_x, center_y). Default 120 - enough to
                 comfortably contain one player and some surrounding
                 grass for context.
    scale:       how much to enlarge that crop by. Default 8 (so a
                 120x120 crop becomes a 960x960 image).

Workflow:
    1. Take your best rough guess at a player's (x, y) in the full frame
    2. Run this to get a big, gridded, zoomed view centered there
    3. Read the ACTUAL coordinates directly off the grid labels
    4. Feed those into ai_engine.sample_color with a small patch_size (3-5)
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def zoom_region(image_path, output_path, center_x, center_y, region_size=120, scale=8, grid_step=10):
    image_path = Path(image_path)
    output_path = Path(output_path)

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    h, w = frame.shape[:2]
    half = region_size // 2
    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(w, center_x + half)
    y2 = min(h, center_y + half)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Region around ({center_x}, {center_y}) is outside the image ({w}x{h})")

    zoomed = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # Draw grid lines + coordinate labels, mapped back to ORIGINAL pixel space.
    grid_color = (0, 255, 255)  # cyan, visible against most kit colors
    crop_w, crop_h = x2 - x1, y2 - y1

    for gx in range(0, crop_w + 1, grid_step):
        orig_x = x1 + gx
        px = gx * scale
        cv2.line(zoomed, (px, 0), (px, zoomed.shape[0]), grid_color, 1)
        cv2.putText(zoomed, str(orig_x), (px + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1, cv2.LINE_AA)

    for gy in range(0, crop_h + 1, grid_step):
        orig_y = y1 + gy
        py = gy * scale
        cv2.line(zoomed, (0, py), (zoomed.shape[1], py), grid_color, 1)
        cv2.putText(zoomed, str(orig_y), (2, py + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), zoomed)

    print(f"Saved zoomed region to {output_path}")
    print(f"Original region: x=[{x1},{x2}] y=[{y1},{y2}], zoomed {scale}x")
    print("Read the cyan grid labels to find the exact original-image (x, y) for a jersey pixel.")


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6, 7):
        print("Usage: python -m ai_engine.zoom_frame <image_path> <output_path> <center_x> <center_y> [region_size] [scale]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_path = sys.argv[2]
    cx = int(sys.argv[3])
    cy = int(sys.argv[4])
    region = int(sys.argv[5]) if len(sys.argv) >= 6 else 120
    scale_factor = int(sys.argv[6]) if len(sys.argv) == 7 else 8

    zoom_region(img_path, out_path, cx, cy, region, scale_factor)