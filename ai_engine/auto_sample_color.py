"""
Solves the "manual pixel-picking keeps landing on grass" problem
directly: instead of guessing one exact (x, y) on a small/blurry player,
give a rough bounding box around them, and this automatically excludes
anything grass-colored within that box, then returns the median color of
whatever's left (the actual player - jersey, skin, hair, whatever isn't
green).

Usage:
    python -m ai_engine.auto_sample_color <image_path> <x1> <y1> <x2> <y2>

x1,y1,x2,y2: a rough box around the player - doesn't need to be tight,
just needs to contain them. Get these from ai_engine.zoom_frame's grid
labels, eyeballing the general area rather than one exact pixel.
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def auto_sample_color(image_path, x1, y1, x2, y2):
    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    h, w = frame.shape[:2]
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))

    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        raise ValueError(f"Region ({x1},{y1})-({x2},{y2}) is outside the image ({w}x{h})")

    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Grass and a dark green/black jersey can share the same HUE - the
    # real difference is brightness. Real turf under stadium lighting is
    # fairly bright; a dark kit is much darker even at the same hue. So
    # only exclude a pixel as "grass" if it's ALSO bright enough - this
    # keeps dark-colored jerseys correctly classified as "the player,"
    # not accidentally swept into "background."
    lower_green = np.array([30, 20, 70])
    upper_green = np.array([90, 255, 255])
    grass_mask = cv2.inRange(hsv_region, lower_green, upper_green)
    non_grass_mask = cv2.bitwise_not(grass_mask)

    # Anti-aliased edges between the player and the grass blend the two
    # colors together (neither pure jersey nor pure grass) - those edge
    # pixels survive the grass filter (they're not green enough to
    # exclude) but still corrupt the median toward a muddy in-between
    # color. Eroding shrinks the non-grass mask inward, discarding those
    # blended edge pixels and keeping only the solid interior.
    erosion_kernel = np.ones((3, 3), np.uint8)
    non_grass_mask_eroded = cv2.erode(non_grass_mask, erosion_kernel, iterations=1)

    # If erosion wiped out almost everything (the non-grass region was
    # already thin/small), fall back to the un-eroded mask rather than
    # returning nothing.
    if np.count_nonzero(non_grass_mask_eroded) >= 10:
        non_grass_mask = non_grass_mask_eroded
    else:
        print("Region too thin to erode safely - using un-eroded mask (results may be less clean).")

    non_grass_pixel_count = int(np.count_nonzero(non_grass_mask))
    total_pixels = region.shape[0] * region.shape[1]

    if non_grass_pixel_count < total_pixels * 0.02:
        print(f"WARNING: only {non_grass_pixel_count}/{total_pixels} pixels in this box "
              f"aren't grass-colored. The box likely doesn't actually contain the player - "
              f"widen it or move it.")

    region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    pixels = region_rgb.reshape(-1, 3)
    mask_flat = non_grass_mask.reshape(-1) > 0

    non_grass_pixels = pixels[mask_flat]
    if len(non_grass_pixels) == 0:
        print("No non-grass pixels found at all in this box - it's entirely grass. "
              "Try a different region.")
        return None

    med_r, med_g, med_b = np.median(non_grass_pixels, axis=0)
    hex_color = f"#{int(med_r):02X}{int(med_g):02X}{int(med_b):02X}"

    print(f"Region ({x1},{y1})-({x2},{y2}): {non_grass_pixel_count}/{total_pixels} "
          f"pixels were non-grass ({100*non_grass_pixel_count/total_pixels:.0f}%)")
    print(f"Median non-grass color: {hex_color}")
    print(f"  RGB: ({int(med_r)}, {int(med_g)}, {int(med_b)})")

    # Also save a visual: the box with grass masked out, so you can SEE
    # what was actually being averaged - useful for sanity-checking.
    debug_path = image_path.parent / f"{image_path.stem}_debug_mask.png"
    debug_region = region.copy()
    debug_region[grass_mask > 0] = (255, 0, 255)  # magenta = "excluded as grass"
    cv2.imwrite(str(debug_path), debug_region)
    print(f"Saved debug visualization (magenta = excluded as grass) to {debug_path}")

    return hex_color


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python -m ai_engine.auto_sample_color <image_path> <x1> <y1> <x2> <y2>")
        sys.exit(1)

    img_path = sys.argv[1]
    x1_arg, y1_arg, x2_arg, y2_arg = (int(v) for v in sys.argv[2:6])

    auto_sample_color(img_path, x1_arg, y1_arg, x2_arg, y2_arg)