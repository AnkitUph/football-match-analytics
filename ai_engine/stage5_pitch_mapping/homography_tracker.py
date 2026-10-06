"""Stage 5: propagate a calibrated image-to-pitch homography when possible.

The tracker follows calibration points and background features between
frames. It can lose track during cuts, close-ups, occlusion, or camera
motion; callers must treat missing or unvalidated homographies as
unavailable. This implementation has not been benchmarked in the current
worktree.
"""

import cv2
import numpy as np

from ai_engine.config import PitchMappingConfig


class HomographyTracker:
    """
    Robust Homography Tracker for football broadcast footage.
    Combines:
    1. Direct re-solve from tracked ground-truth calibration points when visible.
    2. Continuous camera-motion propagation (optical flow on background features
       with RANSAC affine estimation) when calibration points leave the frame,
       are occluded, or during long camera pans.
    Returns a homography estimate when tracking succeeds. It does not
    guarantee coverage or calibrated accuracy for every frame.
    """
    def __init__(self, config: PitchMappingConfig):
        self.config = config
        self.current_H: np.ndarray | None = None

        self._pitch_points: np.ndarray | None = None
        self._calib_px_points: np.ndarray | None = None
        self._feat_pts: np.ndarray | None = None
        self._old_gray: np.ndarray | None = None
        self._frames_since_refresh = 0

        self._lk_calib = dict(
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self._lk_feat = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self._feature_params = dict(
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=5,
        )

    def _feature_mask(self, frame_shape: tuple[int, int]) -> np.ndarray:
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0:int(h * 0.45), :] = 255  # Stands, advertising boards, upper pitch
        mask[:, 0:int(w * 0.2)] = 255   # Left boundary
        mask[:, int(w * 0.8):] = 255   # Right boundary
        return mask

    def bootstrap(
        self,
        frame: np.ndarray,
        image_points: list[tuple[float, float]],
        pitch_points: list[tuple[float, float]],
    ) -> bool:
        """
        Initializes homography from manual or automatic calibration point pairs.
        """
        from ai_engine.stage5_pitch_mapping.homography import compute_homography_from_points

        H = compute_homography_from_points(image_points, pitch_points, self.config)
        if H is None:
            return False

        self.current_H = H / H[2, 2]
        self._pitch_points = np.array(pitch_points, dtype=np.float32)
        self._calib_px_points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._old_gray = gray
        mask = self._feature_mask(gray.shape)
        self._feat_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self._feature_params)
        self._frames_since_refresh = 0
        return True

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Propagates homography to the next frame.
        Uses direct point re-solve when 4+ calibration points survive optical flow,
        otherwise falls back to robust frame-to-frame camera motion estimation.
        Pauses tracking during scene cuts or non-tactical closeups to prevent drift.
        """
        if self.current_H is None or self._old_gray is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 0. Camera cut / non-tactical shot guard
        diff = float(np.mean(cv2.absdiff(gray, self._old_gray)))
        if diff > 35.0:
            # Hard camera cut: do not track optical flow across disjoint scenes
            self._old_gray = gray
            self._feat_pts = None
            self._calib_px_points = None
            return None

        # Check green grass ratio to ignore dugouts, closeups, and replays
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv, (35, 38, 38), (85, 255, 255))
        if (np.count_nonzero(grass_mask) / (frame.shape[0] * frame.shape[1])) < 0.35:
            # Non-tactical view: pause homography propagation
            self._old_gray = gray
            self._feat_pts = None
            return None

        mask = self._feature_mask(gray.shape)

        # 1. Background feature tracking for camera motion
        if self._feat_pts is None or len(self._feat_pts) < 50:
            self._feat_pts = cv2.goodFeaturesToTrack(self._old_gray, mask=mask, **self._feature_params)

        step_H = None
        new_feat = f_status = None
        if self._feat_pts is not None and len(self._feat_pts) >= 8:
            new_feat, f_status, _ = cv2.calcOpticalFlowPyrLK(
                self._old_gray, gray, self._feat_pts, None, **self._lk_feat
            )
            if new_feat is not None and f_status is not None:
                good_old = self._feat_pts[f_status.flatten() == 1]
                good_new = new_feat[f_status.flatten() == 1]
                if len(good_new) >= 8:
                    M, _ = cv2.estimateAffinePartial2D(
                        good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0
                    )
                    if M is not None:
                        step_H = np.vstack([M, [0, 0, 1]])

        # 2. Try direct calibration point tracking
        direct_success = False
        homography_updated = False
        if self._calib_px_points is not None and len(self._calib_px_points) >= 4:
            new_calib, c_status, _ = cv2.calcOpticalFlowPyrLK(
                self._old_gray, gray, self._calib_px_points, None, **self._lk_calib
            )
            if new_calib is not None and c_status is not None:
                c_valid = c_status.flatten() == 1
                if np.sum(c_valid) >= 4:
                    valid_px = new_calib[c_valid].reshape(-1, 2)
                    valid_pitch = self._pitch_points[c_valid]
                    ransac_thresh = getattr(self.config, "homography_ransac_thresh", 5.0)
                    H_direct, _ = cv2.findHomography(
                        valid_px,
                        valid_pitch,
                        cv2.RANSAC if len(valid_px) > 4 else 0,
                        ransac_thresh,
                    )
                    if H_direct is not None and abs(H_direct[2, 2]) > 1e-4:
                        H_norm = H_direct / H_direct[2, 2]
                        det = abs(np.linalg.det(H_norm))
                        x_span = np.ptp(valid_pitch[:, 0])
                        y_span = np.ptp(valid_pitch[:, 1])
                        # Only accept direct re-solve if points have true 2D pitch spread and non-zero determinant
                        if det > 1e-12 and x_span >= 10.0 and y_span >= 10.0:
                            self.current_H = H_norm
                            self._calib_px_points = new_calib[c_valid].reshape(-1, 1, 2).astype(np.float32)
                            self._pitch_points = valid_pitch
                            direct_success = True
                            homography_updated = True

        # 3. If direct re-solve was not possible, propagate via camera motion
        if not direct_success:
            if step_H is not None:
                try:
                    H_prop = self.current_H @ np.linalg.inv(step_H)
                    if abs(H_prop[2, 2]) > 1e-4:
                        H_prop_norm = H_prop / H_prop[2, 2]
                        det_prop = abs(np.linalg.det(H_prop_norm))
                        cond_prop = np.linalg.cond(H_prop_norm)
                        if det_prop > 1e-12 and cond_prop < 50000000:
                            self.current_H = H_prop_norm
                            homography_updated = True
                except np.linalg.LinAlgError:
                    pass

                # Propagate calib pixel points forward with camera motion
                if self._calib_px_points is not None:
                    ones = np.ones((len(self._calib_px_points), 1, 1), dtype=np.float32)
                    homog = np.concatenate([self._calib_px_points, ones], axis=2).reshape(-1, 3)
                    trans = (step_H @ homog.T).T
                    self._calib_px_points = (trans[:, :2] / trans[:, 2:3]).astype(np.float32).reshape(-1, 1, 2)

        # 4. Prepare for next frame
        self._old_gray = gray
        self._frames_since_refresh += 1
        if new_feat is not None and f_status is not None:
            survived = new_feat[f_status.flatten() == 1].reshape(-1, 1, 2).astype(np.float32)
            if self._frames_since_refresh >= 30 or len(survived) < 50:
                self._feat_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self._feature_params)
                self._frames_since_refresh = 0
            else:
                self._feat_pts = survived
        else:
            self._feat_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self._feature_params)
            self._frames_since_refresh = 0

        return self.current_H if homography_updated else None

    def resume_after_cut(
        self,
        frame: np.ndarray,
        H_target: np.ndarray | None = None,
        image_points: list[tuple[float, float]] | None = None,
        pitch_points: list[tuple[float, float]] | None = None,
    ) -> bool:
        """
        Resumes tracking when camera returns to a tactical view after a cutscene.
        Re-initializes background optical flow features and calibration points on the new frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._old_gray = gray
        mask = self._feature_mask(gray.shape)
        self._feat_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self._feature_params)
        self._frames_since_refresh = 0

        if H_target is not None:
            self.current_H = H_target.copy()
        if image_points is not None and pitch_points is not None:
            self._pitch_points = np.array(pitch_points, dtype=np.float32)
            self._calib_px_points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        return True

    def try_drift_correction(self, frame: np.ndarray) -> bool:
        return False
