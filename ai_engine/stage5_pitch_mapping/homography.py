"""
Homography utilities for mapping image points to pitch coordinates.

Calibration currently requires known image-to-pitch point pairs. The line
helpers are experimental; they do not provide automatic calibration or
drift correction. Accuracy has not been benchmarked in the current worktree.

1. BOOTSTRAP (once per shot): a manual 4-point calibration — you
   supply 4 pixel coordinates matched to known real-world pitch points
   for a frame in a continuous shot.

2. PROPAGATE (every frame): optical-flow tracked background features
   (same masking approach as your existing camera_movement_estimator.py)
   estimate the frame-to-frame image transform, composed with the
   current homography. This propagation can drift or fail and must be
   reviewed against calibration points (see homography_tracker.py).

3. Line detection helpers below are experimental aids and are not
   connected to automatic calibration or drift correction.

This file holds the line-mask and point-mapping helpers.
homography_tracker.py holds the stateful propagation logic.
"""

import cv2
import numpy as np

from ai_engine.config import PitchMappingConfig
from ai_engine.utils.types import PitchPoint


def detect_pitch_line_mask(frame: np.ndarray) -> np.ndarray:
    """
    Returns a binary mask of likely pitch-line pixels, restricted to the
    actual pitch surface, approximated by the largest detected green
    region.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 30, 30])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    pitch_roi_mask = np.zeros_like(green_mask)
    cv2.drawContours(pitch_roi_mask, [largest], -1, 255, -1)
    pitch_roi_mask = cv2.erode(pitch_roi_mask, np.ones((7, 7), np.uint8))

    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    return cv2.bitwise_and(white_mask, pitch_roi_mask)


def detect_halfway_line(frame: np.ndarray) -> tuple[float, float, float, float] | None:
    """
    Detects a candidate halfway line. This heuristic has not been
    validated across camera orientations or broadcast footage.

    Returns (x1, y1, x2, y2) in pixel space for the longest, most
    vertical line found, or None if nothing confident enough is found.
    Used for drift correction, not bootstrap — a single line alone is
    under-constrained for a full homography.
    """
    line_mask = detect_pitch_line_mask(frame)
    edges = cv2.Canny(line_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=80, maxLineGap=15)

    if lines is None:
        return None

    best_line = None
    best_length = 0
    for line in lines:
        pts = np.asarray(line).reshape(-1)
        if len(pts) < 4:
            continue
        x1, y1, x2, y2 = pts[:4]
        length = np.hypot(x2 - x1, y2 - y1)
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        # Halfway line is roughly vertical in this broadcast angle —
        # adjust this angle window if your footage uses a different
        # camera orientation.
        is_roughly_vertical = 60 < angle < 120
        if is_roughly_vertical and length > best_length:
            best_length = length
            best_line = (float(x1), float(y1), float(x2), float(y2))

    return best_line


def compute_homography_from_points(
    image_points: list[tuple[float, float]],
    pitch_points: list[tuple[float, float]],
    config: PitchMappingConfig,
) -> np.ndarray | None:
    """
    Solves for the homography given >=4 matched point pairs. Used for
    the manual bootstrap calibration — see homography_tracker.py's
    bootstrap() for the entry point you'll actually call.
    """
    if len(image_points) < 4 or len(image_points) != len(pitch_points):
        return None
    src = np.array(image_points, dtype=np.float32)
    dst = np.array(pitch_points, dtype=np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, config.homography_ransac_thresh)
    return H


def image_point_to_pitch(px: float, py: float, homography_matrix: np.ndarray) -> PitchPoint:
    """Applies a homography to a single image-space point (pixels) to get
    pitch-space meters. Use the player's FOOT position (bottom-center of
    their bbox), not the box center, for accurate ground-plane mapping."""
    point = np.array([[[px, py]]], dtype=np.float32)
    pitch_point = cv2.perspectiveTransform(point, homography_matrix)
    x_m, y_m = pitch_point[0][0]
    return PitchPoint(x_m=float(x_m), y_m=float(y_m))
