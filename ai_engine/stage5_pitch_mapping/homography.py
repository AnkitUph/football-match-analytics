"""
Stage 5a: Pitch Perspective Mapping.

Only run this on frames belonging to MAIN_WIDE shot segments (Stage 2.5) —
close-ups and replays usually don't show enough of the pitch to calibrate
against, and forcing homography on them will produce garbage coordinates.
"""

import numpy as np

from ai_engine.config import PitchMappingConfig
from ai_engine.utils.types import PitchPoint


# Standard pitch keypoints in meters (origin at center circle), the
# targets your detected image-space keypoints get mapped onto.
# TODO: fill in the specific keypoints you detect (corner flags, penalty
# box corners, center circle, halfway line intersections, etc.) matched
# to their real-world (x_m, y_m) equivalents on a 105x68m pitch.
PITCH_KEYPOINTS_M: dict[str, tuple[float, float]] = {
    # "top_left_corner": (-52.5, -34.0),
    # "top_right_corner": (52.5, -34.0),
    # "center_spot": (0.0, 0.0),
    # ... etc, fill in based on your keypoint detection method
}


def detect_pitch_keypoints(frame: np.ndarray) -> dict[str, tuple[float, float]]:
    """
    Detects field line intersections / corner markers in image space for
    one frame. Returns {keypoint_name: (px_x, px_y)}.

    TODO: this is the least turnkey part of Stage 5. Two common approaches:
        1. Classical CV: line detection (Hough transform) + intersection
           finding, matched against expected pitch line geometry.
        2. A small keypoint-detection model (e.g. fine-tuned on pitch
           corner/line-intersection annotations) — more robust to lighting
           and partial occlusion, but needs its own training data.
    Start with (1) to get something working; consider (2) later if
    homography accuracy is a bottleneck.
    """
    raise NotImplementedError("detect_pitch_keypoints is stubbed.")


def compute_homography(
    image_keypoints: dict[str, tuple[float, float]],
    config: PitchMappingConfig,
) -> np.ndarray | None:
    """
    Solves for the homography matrix mapping image-space pixels to
    pitch-space meters, using matched keypoint pairs. Returns None if too
    few keypoints were matched (need at least 4) to compute reliably.

    TODO:
        matched_names = set(image_keypoints) & set(PITCH_KEYPOINTS_M)
        if len(matched_names) < 4:
            return None
        src = np.array([image_keypoints[n] for n in matched_names], dtype=np.float32)
        dst = np.array([PITCH_KEYPOINTS_M[n] for n in matched_names], dtype=np.float32)
        H, mask = cv2.findHomography(
            src, dst, cv2.RANSAC, config.homography_ransac_thresh
        )
        return H
    """
    raise NotImplementedError("compute_homography is stubbed.")


def image_point_to_pitch(
    px: float, py: float, homography_matrix: np.ndarray
) -> PitchPoint:
    """
    Applies the homography to a single image-space point (e.g. a player's
    foot position — bottom-center of their bbox, not the box center) to
    get pitch-space meters.

    TODO:
        point = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
        pitch_point = cv2.perspectiveTransform(point, homography_matrix)
        x_m, y_m = pitch_point[0][0]
        return PitchPoint(x_m=float(x_m), y_m=float(y_m))
    """
    raise NotImplementedError("image_point_to_pitch is stubbed.")
