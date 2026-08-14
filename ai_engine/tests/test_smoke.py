"""
Smoke tests — verify the skeleton imports cleanly and config/types work,
before any real stage logic exists. Run with: pytest ai_engine/tests/

Add real tests here per-stage as you implement each one. Useful first real
test once Stage 1 is wired up: run detector.detect_frame() on one known
frame from your Kaggle test set and assert boxes roughly match.
"""

from ai_engine.config import DEFAULT_CONFIG, PipelineConfig
from ai_engine.stage5_pitch_mapping.identity_association import IdentityGallery
from ai_engine.utils.types import Detection, ObjectClass, Team, Tracklet


def test_config_loads():
    assert DEFAULT_CONFIG.pitch_mapping.max_gallery_size == 22
    assert DEFAULT_CONFIG.pitch_mapping.max_per_team == 11


def test_detection_dataclass():
    d = Detection(frame_idx=0, cls=ObjectClass.PLAYER, conf=0.9, x1=0, y1=0, x2=10, y2=20)
    assert d.center == (5.0, 10.0)


def test_gallery_enforces_cap():
    config = PipelineConfig().pitch_mapping
    gallery = IdentityGallery(config)

    for i in range(11):
        tracklet = Tracklet(
            track_id=i, shot_id=0, team=Team.TEAM_A, reid_embedding=[0.1] * 8
        )
        assert gallery.can_add(Team.TEAM_A)
        gallery.add(tracklet)

    # 12th player on the same team should be rejected by can_add().
    assert not gallery.can_add(Team.TEAM_A)
    assert gallery.count_for_team(Team.TEAM_A) == 11


def test_gallery_total_cap():
    config = PipelineConfig().pitch_mapping
    gallery = IdentityGallery(config)

    for i in range(11):
        gallery.add(Tracklet(track_id=i, shot_id=0, team=Team.TEAM_A, reid_embedding=[0.1] * 8))
    for i in range(11, 22):
        gallery.add(Tracklet(track_id=i, shot_id=0, team=Team.TEAM_B, reid_embedding=[0.1] * 8))

    assert len(gallery.identities) == 22
    assert not gallery.can_add(Team.TEAM_A)
    assert not gallery.can_add(Team.TEAM_B)
