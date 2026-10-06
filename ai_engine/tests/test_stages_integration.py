"""
Integration tests for Stages 1 through 6 verifying fixes and module stability.
"""

import numpy as np
import pytest

from ai_engine.config import PipelineConfig, MODELS_DIR
from ai_engine.stage1_detection.model_loader import get_yolo_model
from ai_engine.stage2_5_shot_detection.shot_detector import classify_shot
from ai_engine.stage4_ball_tracking.ball_tracker import interpolate_gaps
from ai_engine.stage5_pitch_mapping.homography_tracker import HomographyTracker
from ai_engine.stage5_pitch_mapping.smart_assist import has_local_pitch_model, _get_local_pitch_model
from ai_engine.stage6_event_detection.events import detect_possession, detect_passes
from ai_engine.utils.types import BallTrajectoryPoint, Detection, MasterIdentity, ObjectClass, PitchPoint, ShotType, Team


def test_stage1_model_loading_and_class_names():
    cfg = PipelineConfig().detection
    if (MODELS_DIR / "best.pt").exists():
        cfg.model_path = MODELS_DIR / "best.pt"
        model = get_yolo_model(cfg)
        assert model is not None
        assert "ball" in [str(n).lower() for n in model.names.values()]
        assert "player" in [str(n).lower() for n in model.names.values()]


def test_stage2_5_shot_classifier_opencv_shape_resilience():
    cfg = PipelineConfig().shot_detection
    # Blank frame -> OTHER
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    assert classify_shot(blank, cfg) == ShotType.OTHER

    # Green turf dummy frame
    green_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # BGR green: high green channel, moderate red and blue
    green_frame[:, :] = (35, 120, 35)
    # Draw white pitch lines
    for y in range(100, 600, 50):
        green_frame[y:y+3, :] = (255, 255, 255)
    shot_type = classify_shot(green_frame, cfg)
    assert shot_type in (ShotType.MAIN_WIDE, ShotType.CLOSE_UP)


def test_stage4_ball_gap_interpolation():
    cfg = PipelineConfig().ball_tracking
    # 10 frames with ball at x=100+i*5, gap between frames 3 and 7
    balls = {}
    for f in [0, 1, 2, 3, 7, 8, 9]:
        balls[f] = Detection(frame_idx=f, cls=ObjectClass.BALL, conf=0.8, x1=100 + f * 5, y1=200, x2=110 + f * 5, y2=210)
    for f in [4, 5, 6]:
        balls[f] = None

    interpolated = interpolate_gaps(balls, cfg, frame_width=1920, frame_height=1080)
    assert len(interpolated) == 10
    # Gap frames should be flagged as interpolated
    for pt in interpolated:
        if pt.frame_idx in (4, 5, 6):
            assert pt.interpolated is True
            assert pt.x_m is not None
            assert pt.y_m is not None
        else:
            assert pt.interpolated is False


def test_stage5_pitch_keypoint_model_exists():
    assert has_local_pitch_model() is True
    model = _get_local_pitch_model()
    assert model is not None


def test_stage5_homography_condition_number_threshold():
    cfg = PipelineConfig().pitch_mapping
    tracker = HomographyTracker(cfg)

    # Realistic 1080p pixel to pitch meter coordinates
    img_pts = [(960.0, 540.0), (1200.0, 300.0), (1200.0, 800.0), (700.0, 300.0), (700.0, 800.0)]
    pitch_pts = [(0.0, 0.0), (20.0, 15.0), (20.0, -15.0), (-20.0, 15.0), (-20.0, -15.0)]
    dummy_frame = np.full((1080, 1920, 3), 60, dtype=np.uint8)

    ok = tracker.bootstrap(dummy_frame, img_pts, pitch_pts)
    assert ok is True
    assert tracker.current_H is not None
    cond = np.linalg.cond(tracker.current_H)
    # Ensure tolerance accommodates broadcast 1080p scale (< 5,000,000)
    assert cond < 5000000


def test_stage6_events_detection():
    cfg = PipelineConfig().event_detection
    # Setup dummy identities
    id1 = MasterIdentity(master_id=1, team=Team.TEAM_A, reid_embedding=[0.0]*384)
    id2 = MasterIdentity(master_id=2, team=Team.TEAM_A, reid_embedding=[0.0]*384)
    id3 = MasterIdentity(master_id=3, team=Team.TEAM_B, reid_embedding=[0.0]*384)

    # Player 1 near (0, 0), Player 2 near (20, 0), Player 3 near (-20, 0)
    for f in range(50):
        id1.trajectory[f] = PitchPoint(x_m=0.0, y_m=0.0)
        id2.trajectory[f] = PitchPoint(x_m=20.0, y_m=0.0)
        id3.trajectory[f] = PitchPoint(x_m=-20.0, y_m=0.0)

    # Ball starts at Player 1, moves to Player 2
    ball_traj = []
    for f in range(25):
        ball_traj.append(BallTrajectoryPoint(frame_idx=f, x_m=0.0, y_m=0.0, interpolated=False))
    for f in range(25, 50):
        ball_traj.append(BallTrajectoryPoint(frame_idx=f, x_m=20.0, y_m=0.0, interpolated=False))

    possession = detect_possession(ball_traj, [id1, id2, id3], cfg)
    assert len(possession) > 0
    passes = detect_passes(possession, [id1, id2, id3])
    # Expect pass from player 1 to player 2
    assert len(passes) >= 1
    assert passes[0].player_master_id == 1
    assert passes[0].target_master_id == 2
