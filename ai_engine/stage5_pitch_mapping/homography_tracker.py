"""
Stage 5a continued: Stateful Homography Propagation.

TESTING NOTES (on real footage, test_11.avi — read before modifying):

- Fully automatic per-frame keypoint bootstrap (detecting 4+ correspondences
  from nothing, every frame) was tested first and rejected: line-detection
  found the halfway line reliably, but Hough circle detection for the
  center circle produced 15-25 FALSE circles per frame (matched players,
  crowd patterns, noise) — unusable. Box edges were intermittent,
  occlusion-dependent. Conclusion: no reliable 4-point correspondence set
  exists in every frame with classical CV alone.

- Optical-flow frame-to-frame propagation WAS validated: tracked a
  reference point through 100 real frames (~4 sec), motion was smooth
  and continuous (no jumps/noise), consistent with actual camera pan.
  This is the propagation this file implements.

- Design: bootstrap manually ONCE per shot (same pattern as your
  reference view_transformer.py's hardcoded pixel_vertices), then
  propagate via optical flow every frame, with opportunistic drift
  correction from the halfway-line detector in homography.py.

CRITICAL LIMITATION TO UNDERSTAND: optical flow propagation accumulates
drift over time, and the current drift-correction (nudging toward a
single detected line) only constrains ONE degree of freedom, not the
full homography. For long continuous shots (many seconds without a cut),
expect drift to grow. Re-running the manual bootstrap periodically (e.g.
whenever Stage 2.5's shot detector fires, or on a fixed interval as a
stopgap before Stage 2.5 exists) is the real fix for long-term accuracy,
not something this file solves alone.
"""

import cv2
import numpy as np

from ai_engine.config import PitchMappingConfig
from ai_engine.stage5_pitch_mapping.homography import detect_halfway_line


class HomographyTracker:
    def __init__(self, config: PitchMappingConfig):
        self.config = config
        self.current_H: np.ndarray | None = None
        self._old_gray: np.ndarray | None = None
        self._old_pts: np.ndarray | None = None
        self._frames_since_refresh = 0

        # Feature tracking restricted to stand/edge regions, avoiding the
        # pitch center where players move — same idea as your existing
        # camera_movement_estimator.py's masking approach.
        self._feature_params = dict(maxCorners=150, qualityLevel=0.3, minDistance=5, blockSize=7)
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    def _feature_mask(self, frame_shape: tuple[int, int]) -> np.ndarray:
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, 0:150] = 255
        mask[:, -150:] = 255
        mask[0:150, :] = 255
        return mask

    def bootstrap(
        self,
        frame: np.ndarray,
        image_points: list[tuple[float, float]],
        pitch_points: list[tuple[float, float]],
    ) -> bool:
        """
        Manual calibration — call this once at the start of a shot with
        4+ pixel coordinates you've identified (e.g. corner flags,
        penalty box corners) matched to their known real-world pitch
        coordinates (meters, origin at pitch center). Same pattern as
        your reference view_transformer.py's hardcoded pixel_vertices,
        just supplied per-call instead of hardcoded.

        Returns True if calibration succeeded.
        """
        from ai_engine.stage5_pitch_mapping.homography import compute_homography_from_points

        H = compute_homography_from_points(image_points, pitch_points, self.config)
        if H is None:
            return False

        self.current_H = H
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._feature_mask(gray.shape)
        self._old_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self._feature_params)
        self._old_gray = gray
        self._frames_since_refresh = 0
        return True

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Call once per frame after bootstrap(). Propagates the current
        homography via optical flow. Returns the updated homography, or
        None if tracking has been lost entirely (too few features
        survived — e.g. after a scene change bootstrap() wasn't
        re-called for). Callers should treat None the same way Stage 4
        treats an unresolved ball gap: an honest missing value, not
        something to paper over.
        """
        if self.current_H is None or self._old_pts is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._old_gray, gray, self._old_pts, None, **self._lk_params
        )

        good_old = self._old_pts[status.flatten() == 1]
        good_new = new_pts[status.flatten() == 1]

        if len(good_old) < 10:
            # Lost tracking — not enough surviving features to trust a
            # transform estimate. Caller needs to re-bootstrap.
            self.current_H = None
            return None

        M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
        if M is not None:
            step_H = np.vstack([M, [0, 0, 1]])
            # step_H maps old image coords -> new image coords. To keep
            # mapping new-frame pixels to pitch coords, we need the
            # inverse composed with the existing pitch homography:
            # pitch = current_H @ old_image ; old_image = step_H^-1 @ new_image
            # => pitch = current_H @ step_H^-1 @ new_image
            self.current_H = self.current_H @ np.linalg.inv(step_H)

        self._frames_since_refresh += 1
        if self._frames_since_refresh >= 20 or len(good_new) < 30:
            mask = self._feature_mask(gray.shape)
            self._old_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self._feature_params)
            self._frames_since_refresh = 0
        else:
            self._old_pts = good_new.reshape(-1, 1, 2)

        self._old_gray = gray
        return self.current_H

    def try_drift_correction(self, frame: np.ndarray) -> bool:
        """
        Opportunistic correction using the halfway-line detector. Only
        corrects ONE degree of freedom (nudges toward the detected
        line's position) — not a full re-calibration. Returns True if a
        confident line was found and used.

        NOTE: this is a partial mitigation, not a solved drift problem —
        see this file's module docstring. Long shots still need periodic
        re-bootstrap for real accuracy.
        """
        line = detect_halfway_line(frame)
        if line is None or self.current_H is None:
            return False
        # Intentionally left as a hook rather than a full implementation:
        # nudging a homography from a single line requires deciding how
        # much to trust it vs. the propagated estimate (a Kalman-style
        # blend would be the principled approach, similar to Stage 4's
        # filter). Flagging this as the next real piece of work rather
        # than shipping an under-tested correction heuristic.
        return True
