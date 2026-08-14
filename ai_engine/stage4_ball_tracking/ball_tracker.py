"""
Stage 4: Ball Tracking & Frame Interpolation.

Ball detection (mAP50 0.792, recall 0.71 in your Kaggle checkpoint) is
the weakest link — expected, given the ball's tiny pixel footprint and
motion blur. This stage's job is to smooth over that: interpolate through
gaps rather than expect the detector to catch every frame.

If ball recall proves insufficient in practice, the fix belongs here (a
dedicated high-res crop / small-object detection pass for the ball class
specifically), NOT in more generic Stage 1 training epochs.
"""

import numpy as np

from ai_engine.config import BallTrackingConfig
from ai_engine.utils.types import BallTrajectoryPoint, Detection


def extract_ball_detections(
    detections_by_frame: dict[int, list[Detection]]
) -> dict[int, Detection | None]:
    """
    Pulls out the single best ball detection per frame (highest confidence,
    if the detector returns multiple ball candidates). Frames with no ball
    detection map to None — that's expected and handled by interpolation.
    """
    raise NotImplementedError("extract_ball_detections is stubbed.")


def interpolate_gaps(
    ball_by_frame: dict[int, Detection | None],
    config: BallTrackingConfig,
) -> list[BallTrajectoryPoint]:
    """
    Fills gaps of up to config.interpolation_max_gap_frames using a Kalman
    filter (preferred, handles velocity) or fallback linear/cubic
    interpolation for shorter gaps. Gaps longer than the max are left as
    missing (interpolated=False, x_m/y_m=None) rather than guessed —
    better to have an honest hole in the trajectory than a fabricated one
    for the event-detection stage to trust.

    TODO: a reasonable starting approach —
        from filterpy.kalman import KalmanFilter
        # constant-velocity model: state = [x, y, vx, vy]
        kf = KalmanFilter(dim_x=4, dim_z=2)
        ... set kf.F, kf.H, kf.Q (config.kalman_process_noise),
            kf.R (config.kalman_measurement_noise) ...
        # walk frames in order: kf.predict() every frame,
        # kf.update(z) when a real detection exists,
        # use kf.x[:2] as the position for frames with no detection,
        # but only up to interpolation_max_gap_frames consecutive misses.
    """
    raise NotImplementedError("interpolate_gaps is stubbed.")
