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


def test_reid_embedder():
    import numpy as np
    from ai_engine.stage3_team_reid.reid import ReidEmbedder, cosine_similarity

    embedder = ReidEmbedder()
    # Dummy valid crop (height 60, width 30, 3 channels)
    crop = np.full((60, 30, 3), 120, dtype=np.uint8)
    emb = embedder.embed(crop)
    assert emb is not None
    assert len(emb) == 384
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)

    # Invalid / empty crop returns None
    assert embedder.embed(np.zeros((0, 0, 3), dtype=np.uint8)) is None

    # Cosine similarity
    sim = cosine_similarity(emb, emb)
    assert np.isclose(sim, 1.0, atol=1e-4)


def test_tracklet_stitcher():
    import numpy as np
    from ai_engine.stage2_tracking.stitcher import stitch_tracklets
    from ai_engine.utils.types import Detection, ObjectClass, Tracklet

    # Tracklet 1: frames 0 to 20
    t1 = Tracklet(track_id=1, shot_id=0, cls=ObjectClass.PLAYER)
    for f in range(0, 20):
        t1.detections.append(Detection(frame_idx=f, cls=ObjectClass.PLAYER, conf=0.9, x1=100, y1=100, x2=130, y2=160))

    # Tracklet 2: frames 25 to 45 (same player returning 5 frames later)
    t2 = Tracklet(track_id=2, shot_id=0, cls=ObjectClass.PLAYER)
    for f in range(25, 45):
        t2.detections.append(Detection(frame_idx=f, cls=ObjectClass.PLAYER, conf=0.9, x1=110, y1=105, x2=140, y2=165))

    # Identical visual embeddings
    emb = np.ones(384, dtype=np.float32) / np.sqrt(384)
    embs = {1: emb, 2: emb}

    tracklets = {1: t1, 2: t2}
    stitched, mapping = stitch_tracklets(tracklets, embs, max_gap_frames=30, similarity_thresh=0.85)

    # Should be merged into 1 single tracklet
    assert len(stitched) == 1
    root_id = list(stitched.keys())[0]
    merged_track = stitched[root_id]
    assert len(merged_track.detections) == 40  # 20 + 20
    assert mapping[2] == mapping[1]


def test_classify_teams_with_dinov2():
    import numpy as np
    from ai_engine.config import TeamReidConfig
    from ai_engine.stage3_team_reid.team_classifier import classify_teams_with_dinov2
    from ai_engine.utils.types import Team

    # 3 players with feature vector A, 3 players with feature vector B
    v_a = np.zeros(384, dtype=np.float32)
    v_a[:192] = 1.0 / np.sqrt(192)

    v_b = np.zeros(384, dtype=np.float32)
    v_b[192:] = 1.0 / np.sqrt(192)

    track_embs = {
        1: v_a, 2: v_a, 3: v_a,
        4: v_b, 5: v_b, 6: v_b,
    }
    track_cols = {
        1: np.array([255, 0, 0]), 2: np.array([255, 0, 0]), 3: np.array([255, 0, 0]),
        4: np.array([0, 255, 0]), 5: np.array([0, 255, 0]), 6: np.array([0, 255, 0]),
    }

    config = TeamReidConfig()
    teams = classify_teams_with_dinov2(track_embs, track_cols, home_bgr=np.array([255, 0, 0]), away_bgr=np.array([0, 255, 0]), config=config)

    # Assert players 1, 2, 3 have same team and 4, 5, 6 have the other team
    assert teams[1] == teams[2] == teams[3]
    assert teams[4] == teams[5] == teams[6]
    assert teams[1] != teams[4]
    assert teams[1] in (Team.TEAM_A, Team.TEAM_B)
    assert teams[4] in (Team.TEAM_A, Team.TEAM_B)

