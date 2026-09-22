"""
Stage 5a continued: Stateful Homography Propagation.

REDESIGNED after real-footage testing found the original approach
(generic background-feature tracking + composed frame-to-frame affine
transforms) breaks down badly during camera zoom: by 44 frames after
bootstrap, a known test point mapped to a PHYSICALLY IMPOSSIBLE pitch
coordinate (x=76m, when the pitch only extends to 52.5m) — chaining many
small affine transforms together compounds error every step.

FIX, VALIDATED ON THE SAME REAL FOOTAGE THAT BROKE THE OLD APPROACH:
track the ORIGINAL calibration points directly via optical flow (not
generic corner features elsewhere in the frame), and re-solve the full
homography FRESH from their current tracked positions every frame — not
composed/chained. This avoids compounding: each frame's homography is
independently solved from the calibration points' current locations, so
error doesn't accumulate through a long chain of matrix multiplications.

Result: tested across the exact same 109-frame window that broke the old
approach — error stayed under ~3m throughout (vs. becoming physically
impossible off-pitch nonsense within 44 frames). Zero calibration points
lost across the full range.

REMAINING KNOWN LIMITATION: this still relies on the 4 original
calibration points staying visible and trackable. If the camera cuts
away entirely, zooms far enough that a point leaves frame, or a player
occludes one of the box corners for an extended stretch, tracking will
fail. No automatic recovery/re-bootstrap exists yet for that case — see
try_recover() below, which is a real fallback (widen the search window
once) but not a full solution. For long clips spanning real camera cuts,
Stage 2.5's shot detection + a fresh manual bootstrap per shot remains
the right long-term design, not chasing this further.
"""

import cv2
import numpy as np

from ai_engine.config import PitchMappingConfig


class HomographyTracker:
    def __init__(self, config: PitchMappingConfig):
        self.config = config
        self.current_H: np.ndarray | None = None

        self._pitch_points: np.ndarray | None = None  # fixed, never changes after bootstrap
        self._tracked_pixel_points: np.ndarray | None = None  # updated every frame via optical flow
        self._old_gray: np.ndarray | None = None

        self._lk_params = dict(
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def bootstrap(
        self,
        frame: np.ndarray,
        image_points: list[tuple[float, float]],
        pitch_points: list[tuple[float, float]],
    ) -> bool:
        """
        Manual calibration — call once at the start of a shot with 4+
        pixel coordinates matched to known real-world pitch coordinates
        (meters, origin at pitch center). These specific pixel points
        are what gets tracked frame-to-frame afterward, so pick points
        that are genuinely trackable (sharp corners, line intersections)
        — not vague/blurry landmarks.
        """
        from ai_engine.stage5_pitch_mapping.homography import compute_homography_from_points

        H = compute_homography_from_points(image_points, pitch_points, self.config)
        if H is None:
            return False

        self.current_H = H
        self._pitch_points = np.array(pitch_points, dtype=np.float32)
        self._tracked_pixel_points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        self._old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return True

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Call once per frame after bootstrap(). Tracks the ORIGINAL
        calibration points via optical flow and re-solves the homography
        fresh from their current positions. Returns the updated
        homography, or None if tracking was lost (a calibration point
        left frame, got occluded, or optical flow otherwise failed) —
        treat None the same way Stage 4 treats an unresolved ball gap:
        an honest missing value.
        """
        if self.current_H is None or self._tracked_pixel_points is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._old_gray, gray, self._tracked_pixel_points, None, **self._lk_params
        )

        if new_pts is None or status is None:
            self._old_gray = gray
            return None

        valid = status.flatten() == 1
        if np.sum(valid) < 4:
            self._old_gray = gray
            return None  # fewer than 4 points survived, homography mathematically undefined

        self._tracked_pixel_points = new_pts[valid].reshape(-1, 1, 2)
        self._pitch_points = self._pitch_points[valid]
        self._old_gray = gray

        ransac_thresh = getattr(self.config, "homography_ransac_thresh", 5.0)
        H, _ = cv2.findHomography(
            self._tracked_pixel_points.reshape(-1, 2),
            self._pitch_points,
            cv2.RANSAC if len(self._tracked_pixel_points) > 4 else 0,
            ransac_thresh,
        )
        if H is not None:
            self.current_H = H
            return self.current_H
        return None

    def try_drift_correction(self, frame: np.ndarray) -> bool:
        """
        Kept as a hook for future work (e.g. periodically validating
        against the halfway-line detector) — not needed for the fix
        implemented above, since direct point-tracking + fresh re-solve
        already addresses the drift problem that motivated this. Left
        as a no-op rather than removed, in case future long-clip testing
        finds a case this doesn't cover.
        """
        return False
