"""Stage 4: ball trajectory extraction and bounded Kalman interpolation.

The tracker produces one candidate ball detection per sampled video frame.
This module selects the best candidate and interpolates short gaps in pixel
coordinates. Long or out-of-frame gaps remain unknown; no accuracy claim is
made here without a current, reproducible evaluation.
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
    kf.R *= config.kalman_measurement_noise
    kf.Q *= config.kalman_process_noise
    kf.P *= 500.0
    return kf


def interpolate_gaps(
    ball_by_frame: dict[int, Detection | None],
    config: BallTrackingConfig,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> list[BallTrajectoryPoint]:
    """
    Fills gaps up to config.interpolation_max_gap_frames using a
    constant-velocity Kalman filter. Gaps longer than the max are left as
    missing (x=None, y=None) rather than guessed indefinitely.

    FRAME-BOUNDS GUARD (added after real-footage testing): the gap-length
    cutoff alone isn't sufficient — found that a high-velocity state
    right before a gap (e.g. a hard shot) can cause constant-velocity
    extrapolation to predict a pixel position OUTSIDE THE ACTUAL VIDEO
    FRAME within just a few interpolated steps, well before hitting the
    max_gap_frames limit. A ball genuinely cannot be at pixel x=2336 in a
    1920-wide frame — that's not a plausible position to hand downstream
    stages, regardless of how few frames into the gap it occurred.
    Pass frame_width/frame_height (from the source video) to enable this
    check; once a predicted position leaves frame bounds, remaining
    frames in that gap are marked missing rather than continuing to
    extrapolate into impossible territory.
    """
    if not ball_by_frame:
        return []

    # Advance one Kalman step per video frame, even if callers omit frames
    # with no detections from the mapping.
    detections_by_frame = ball_by_frame
    frames = range(min(detections_by_frame), max(detections_by_frame) + 1)
    kf = _build_kalman_filter(config)

    initialized = False
    consecutive_missing = 0
    out_of_bounds = False
    reinitialize_on_detection = False
    trajectory: list[BallTrajectoryPoint] = []

    for frame_idx in frames:
        detection = detections_by_frame.get(frame_idx)

        if detection is not None:
            cx, cy = detection.center
            z = np.array([cx, cy])
            if not initialized or reinitialize_on_detection:
                if reinitialize_on_detection:
                    # The filter stopped advancing during an untrusted gap.
                    # Treat this observation as a new segment instead of
                    # applying it to stale position/velocity covariance.
                    kf = _build_kalman_filter(config)
                kf.x = np.array([cx, cy, 0, 0])
                initialized = True
                reinitialize_on_detection = False
            else:
                kf.predict()
                kf.update(z)
            trajectory.append(
                BallTrajectoryPoint(
                    frame_idx=frame_idx, x_m=float(kf.x[0]), y_m=float(kf.x[1]), interpolated=False
                )
            )
            consecutive_missing = 0
            out_of_bounds = False
        else:
            can_extrapolate = (
                initialized
                and consecutive_missing < config.interpolation_max_gap_frames
                and not out_of_bounds
            )
            if can_extrapolate:
                kf.predict()
                px, py = float(kf.x[0]), float(kf.x[1])

                within_bounds = True
                if frame_width is not None and not (0 <= px <= frame_width):
                    within_bounds = False
                if frame_height is not None and not (0 <= py <= frame_height):
                    within_bounds = False

                if within_bounds:
                    trajectory.append(
                        BallTrajectoryPoint(frame_idx=frame_idx, x_m=px, y_m=py, interpolated=True)
                    )
                    consecutive_missing += 1
                else:
                    # Predicted position left the frame — stop trusting
                    # this gap's extrapolation from here on, same
                    # treatment as exceeding max_gap_frames.
                    out_of_bounds = True
                    trajectory.append(
                        BallTrajectoryPoint(frame_idx=frame_idx, x_m=None, y_m=None, interpolated=False)
                    )
            else:
                if initialized:
                    reinitialize_on_detection = True
                trajectory.append(
                    BallTrajectoryPoint(frame_idx=frame_idx, x_m=None, y_m=None, interpolated=False)
                )

    return trajectory


from pathlib import Path
import logging

_logger = logging.getLogger(__name__)
_TRACKNET_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "tracknetv3_ball_best.pt"
_TRACKNET_MODEL = None


def get_tracknet_model():
    """
    Lazy loader for trained TrackNetV2 ball model from ai_engine/models/tracknetv3_ball_best.pt.
    """
    global _TRACKNET_MODEL
    if _TRACKNET_MODEL is None and _TRACKNET_WEIGHTS_PATH.exists():
        try:
            import torch
            import torch.nn as nn

            class TrackNetV2(nn.Module):
                def __init__(self, in_channels=9):
                    super().__init__()
                    self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64))
                    self.pool1 = nn.MaxPool2d(2, 2)
                    self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128))
                    self.pool2 = nn.MaxPool2d(2, 2)
                    self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256))
                    self.pool3 = nn.MaxPool2d(2, 2)
                    self.bottleneck = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512))
                    self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
                    self.conv_up3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256))
                    self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
                    self.conv_up2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128))
                    self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
                    self.conv_up1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64))
                    self.out_conv = nn.Conv2d(64, 1, 1)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    c1 = self.conv1(x)
                    c2 = self.conv2(self.pool1(c1))
                    c3 = self.conv3(self.pool2(c2))
                    b = self.bottleneck(self.pool3(c3))
                    u3 = self.conv_up3(self.up3(b))
                    u2 = self.conv_up2(self.up2(u3))
                    u1 = self.conv_up1(self.up1(u2))
                    return self.sigmoid(self.out_conv(u1))

            model = TrackNetV2()
            model.load_state_dict(torch.load(str(_TRACKNET_WEIGHTS_PATH), map_location="cpu"))
            model.eval()
            _TRACKNET_MODEL = model
            _logger.info("Loaded trained TrackNet ball model from %s", _TRACKNET_WEIGHTS_PATH)
        except Exception as e:
            _logger.warning("Could not load TrackNet ball model: %s", e)
    return _TRACKNET_MODEL
