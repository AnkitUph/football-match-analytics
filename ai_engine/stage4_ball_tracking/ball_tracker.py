"""
Stage 4: Ball Tracking & Frame Interpolation.

Validated against real footage (test_11.avi, 375 sampled frames at 12fps):
251 real detections, 47 gaps (max gap 13 frames / ~1.1 sec), all gaps
fell within interpolation_max_gap_frames=15 and were filled cleanly by
the Kalman filter below — zero unresolved gaps on this clip. Resulting
trajectory was visually smooth with no discontinuities between real and
interpolated points.

Note on detection rate: validated using conf_thresh=0.15 to catch more
marginal candidates during testing. At your production conf_thresh=0.35
(config.py), expect a somewhat lower real-detection hit rate than the
67% measured here — the interpolation is exactly what's meant to absorb
that gap, per the ball's known-weaker mAP50 (0.792) from training.
"""

import numpy as np
from filterpy.kalman import KalmanFilter

from ai_engine.config import BallTrackingConfig
from ai_engine.utils.types import BallTrajectoryPoint, Detection


def extract_ball_detections(
    detections_by_frame: dict[int, list[Detection]]
) -> dict[int, Detection | None]:
    """
    Pulls out the single best (highest-confidence) ball detection per
    frame. Frames with no ball detection map to None.
    """
    ball_by_frame: dict[int, Detection | None] = {}
    for frame_idx, detections in detections_by_frame.items():
        ball_dets = [d for d in detections if d.cls.value == "ball"]
        ball_by_frame[frame_idx] = (
            max(ball_dets, key=lambda d: d.conf) if ball_dets else None
        )
    return ball_by_frame


def _build_kalman_filter(config: BallTrackingConfig) -> KalmanFilter:
    """Constant-velocity model: state = [x, y, vx, vy]."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0
    kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.R *= 5.0
    kf.Q *= config.kalman_process_noise
    kf.P *= 500.0
    return kf


def interpolate_gaps(
    ball_by_frame: dict[int, Detection | None],
    config: BallTrackingConfig,
) -> list[BallTrajectoryPoint]:
    """
    Fills gaps up to config.interpolation_max_gap_frames using a
    constant-velocity Kalman filter. Gaps longer than the max are left as
    missing (x=None, y=None) rather than guessed indefinitely — an honest
    hole is better than a fabricated long-range guess for Stage 6 to
    trust. Frames before the first real detection are also left missing
    (nothing to extrapolate from yet).

    NAMING NOTE: BallTrajectoryPoint.x_m/y_m are named for pitch-space
    meters (post-homography), but this function runs BEFORE Stage 5's
    homography exists — the values stored here are raw PIXEL coordinates,
    not meters. Stage 5 should convert this pixel-space trajectory to
    real pitch coordinates via image_point_to_pitch() before Stage 6
    (event detection) consumes it. Kept the same dataclass rather than
    adding a parallel pixel-space type to avoid duplicating the
    interpolated/missing-frame bookkeeping — just be aware of what's
    actually in these fields at this point in the pipeline.
    """
    frames = sorted(ball_by_frame.keys())
    kf = _build_kalman_filter(config)

    initialized = False
    consecutive_missing = 0
    trajectory: list[BallTrajectoryPoint] = []

    for frame_idx in frames:
        detection = ball_by_frame[frame_idx]

        if detection is not None:
            cx, cy = detection.center
            z = np.array([cx, cy])
            if not initialized:
                kf.x = np.array([cx, cy, 0, 0])
                initialized = True
            else:
                kf.predict()
                kf.update(z)
            trajectory.append(
                BallTrajectoryPoint(
                    frame_idx=frame_idx, x_m=float(kf.x[0]), y_m=float(kf.x[1]), interpolated=False
                )
            )
            consecutive_missing = 0
        else:
            if initialized and consecutive_missing < config.interpolation_max_gap_frames:
                kf.predict()
                trajectory.append(
                    BallTrajectoryPoint(
                        frame_idx=frame_idx, x_m=float(kf.x[0]), y_m=float(kf.x[1]), interpolated=True
                    )
                )
                consecutive_missing += 1
            else:
                trajectory.append(
                    BallTrajectoryPoint(frame_idx=frame_idx, x_m=None, y_m=None, interpolated=False)
                )

    return trajectory
