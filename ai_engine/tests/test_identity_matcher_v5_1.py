"""
Tests for Global Identity Matcher V5.1.

Uses synthetic embeddings only.
No OSNet / Torch / YOLO / ByteTrack dependency is required.

Run:

pytest -q ai_engine/tests/test_identity_matcher_v5_1.py

Or:

python -m pytest -q ai_engine/tests/test_identity_matcher_v5_1.py
"""

from math import sqrt

import pytest

from ai_engine.identity.identity_matcher import IdentityMatcher
from ai_engine.schemas.track import (
    BoundingBox,
    Identity,
    Track,
    TrackObservation,
)

# ============================================================
# SYNTHETIC EMBEDDINGS
# ============================================================

EMBEDDING_A = [
    1.0,
    0.0,
    0.0,
    0.0,
]

EMBEDDING_B = [
    0.0,
    1.0,
    0.0,
    0.0,
]

EMBEDDING_C = [
    0.0,
    0.0,
    1.0,
    0.0,
]

EMBEDDING_D = [
    0.0,
    0.0,
    0.0,
    1.0,
]


def normalized_embedding(values) -> list[float]:
    """Return a normalized synthetic embedding."""
    norm = sqrt(sum(value * value for value in values))

    if norm == 0:
        return [0.0] * len(values)

    return [value / norm for value in values]


# ============================================================
# HELPERS
# ============================================================


def bbox_from_position(
    x: float,
    y: float,
    width: float = 20.0,
    height: float = 40.0,
) -> BoundingBox:
    """Create a bounding box from a position."""
    return BoundingBox(
        x1=x - width / 2.0,
        y1=y - height,
        x2=x + width / 2.0,
        y2=y,
    )


def observation(
    frame: int,
    x: float,
    y: float,
    embedding=None,
    team: str | None = None,
    jersey: int | None = None,
    confidence: float = 0.95,
    class_name: str = "player",
) -> TrackObservation:
    """Create a track observation."""
    return TrackObservation(
        frame_index=frame,
        bbox=bbox_from_position(x, y),
        confidence=confidence,
        class_name=class_name,
        team=team,
        jersey_number=jersey,
        appearance_embedding=embedding,
    )


def make_track(
    local_id: int = 1,
    frames=None,
    positions=None,
    embeddings=None,
    team: str | None = None,
    jersey: int | None = None,
    class_name: str = "player",
) -> Track:
    """Create a track."""
    if frames is None:
        frames = [1, 2]

    if positions is None:
        positions = [
            (100.0, 100.0),
            (102.0, 100.0),
        ]

    if embeddings is None:
        embeddings = [
            EMBEDDING_A,
            EMBEDDING_A,
        ]

    track = Track(
        local_id=local_id,
        class_name=class_name,
    )

    for frame, position, embedding in zip(
        frames,
        positions,
        embeddings,
    ):
        track.add_observation(
            observation(
                frame=frame,
                x=position[0],
                y=position[1],
                embedding=embedding,
                team=team,
                jersey=jersey,
                class_name=class_name,
            )
        )

    return track


def make_identity(
    identity_id: str = "identity-1",
    frames=None,
    positions=None,
    embeddings=None,
    team: str | None = None,
    jersey: int | None = None,
    class_name: str = "player",
) -> Identity:
    """Create an identity."""
    if frames is None:
        frames = [1, 2]

    if positions is None:
        positions = [
            (100.0, 100.0),
            (102.0, 100.0),
        ]

    if embeddings is None:
        embeddings = [
            EMBEDDING_A,
            EMBEDDING_A,
        ]

    identity = Identity(
        identity_id=identity_id,
        class_name=class_name,
        team=team,
        jersey_number=jersey,
    )

    for frame, position, embedding in zip(
        frames,
        positions,
        embeddings,
    ):
        identity.add_observation(
            observation(
                frame=frame,
                x=position[0],
                y=position[1],
                embedding=embedding,
                team=team,
                jersey=jersey,
                class_name=class_name,
            )
        )

    return identity


def assert_match(
    result,
    reason_contains: str | None = None,
):
    assert result.matched is True
    assert result.score >= 0.0
    assert result.score <= 1.0

    if reason_contains is not None:
        assert reason_contains in result.reason


def assert_rejected(
    result,
    reason: str,
):
    assert result.matched is False
    assert result.reason == reason


# ============================================================
# FIXTURE
# ============================================================


@pytest.fixture
def matcher():
    return IdentityMatcher()


# ============================================================
# TEST 01: Exact continuity + strong appearance
# ============================================================


def test_01_exact_continuity_with_strong_appearance(matcher):
    identity = make_identity(
        identity_id="identity-1",
        frames=[1, 2],
        positions=[
            (100, 100),
            (102, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
    )

    track = make_track(
        local_id=10,
        frames=[3, 4],
        positions=[
            (104, 100),
            (106, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert_match(
        result,
        "strong_appearance",
    )

    assert result.temporal_gap == 0
    assert result.appearance_evaluated is True
    assert result.appearance_similarity >= 0.85


# ============================================================
# TEST 02: Class mismatch
# ============================================================


def test_02_class_mismatch_is_hard_rejection(matcher):
    identity = make_identity(
        class_name="player",
    )

    track = make_track(
        class_name="goalkeeper",
    )

    result = matcher.match(
        track,
        identity,
    )

    assert_rejected(
        result,
        "class_mismatch",
    )

    assert result.appearance_evaluated is False
    assert result.appearance_similarity == -1.0


# ============================================================
# TEST 03: Temporal gap too large
# ============================================================


def test_03_temporal_gap_too_large(matcher):
    identity = make_identity(
        frames=[1, 2],
        positions=[
            (100, 100),
            (102, 100),
        ],
    )

    track = make_track(
        frames=[100, 101],
        positions=[
            (104, 100),
            (106, 100),
        ],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert_rejected(
        result,
        "temporal_gap_too_large",
    )

    assert result.temporal_gap == 97
    assert result.appearance_evaluated is False


# ============================================================
# TEST 04: Overlapping track
# ============================================================


def test_04_overlapping_track_is_rejected(matcher):
    identity = make_identity(
        frames=[10, 20],
    )

    track = make_track(
        frames=[15, 16],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert_rejected(
        result,
        "track_overlaps_identity",
    )

    assert result.temporal_gap < 0
    assert result.appearance_evaluated is False


# ============================================================
# TEST 05: Spatial distance hard gate
# ============================================================


def test_05_spatial_distance_hard_gate(matcher):
    identity = make_identity(
        frames=[1, 2],
        positions=[
            (100, 100),
            (100, 100),
        ],
    )

    track = make_track(
        frames=[3, 4],
        positions=[
            (500, 100),
            (502, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert_rejected(
        result,
        "spatial_distance_too_large",
    )

    assert result.spatial_distance > 300
    assert result.appearance_evaluated is False


# ============================================================
# TEST 06: Motion difference hard gate
# ============================================================


def test_06_motion_difference_hard_gate(matcher):
    identity = make_identity(
        frames=[1, 2],
        positions=[
            (100, 100),
            (110, 100),
        ],
    )

    track = make_track(
        frames=[3, 4],
        positions=[
            (120, 100),
            (220, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert_rejected(
        result,
        "motion_difference_too_large",
    )

    assert result.motion_difference > 80
    assert result.appearance_evaluated is False


# ============================================================
# TEST 07: Appearance unavailable + strong geometry
# ============================================================


def test_07_appearance_unavailable_but_geometry_strong(matcher):
    identity = make_identity(
        frames=[1, 2],
        positions=[
            (100, 100),
            (102, 100),
        ],
        embeddings=[
            None,
            None,
        ],
    )

    track = make_track(
        frames=[3, 4],
        positions=[
            (104, 100),
            (106, 100),
        ],
        embeddings=[
            None,
            None,
        ],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert result.matched is True

    assert result.appearance_similarity == -1.0
    assert result.appearance_evaluated is True

    assert result.reason in {
        "weak_appearance_strong_geometry",
        "normal_appearance_geometry",
        "strong_appearance_continuity",
        "strong_appearance_temporal_continuity",
    }


# ============================================================
# TEST 08: Appearance unavailable + geometry NOT strong
# ============================================================


def test_08_appearance_unavailable_geometry_not_strong(matcher):
    identity = make_identity(
        frames=[1, 2],
        positions=[
            (100, 100),
            (100, 100),
        ],
        embeddings=[
            None,
            None,
        ],
    )

    track = make_track(
        frames=[3, 4],
        positions=[
            (250, 100),
            (250, 100),
        ],
        embeddings=[
            None,
            None,
        ],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert_rejected(
        result,
        "appearance_unavailable_geometry_not_strong",
    )

    assert result.spatial_distance == 150.0
    assert result.appearance_similarity == -1.0
    assert result.appearance_evaluated is True


# ============================================================
# TEST 09: Strong appearance can relax geometry
# ============================================================


def test_09_strong_appearance_can_relax_geometry(matcher):
    identity = make_identity(
        frames=[1, 2],
        positions=[
            (100, 100),
            (100, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
    )

    track = make_track(
        frames=[3, 4],
        positions=[
            (200, 100),
            (200, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert result.appearance_similarity >= 0.85
    assert result.spatial_distance <= 300

    assert_match(
        result,
        "strong_appearance",
    )


# ============================================================
# TEST 10: Weak appearance + very strong geometry
# ============================================================


def test_10_weak_appearance_requires_very_strong_geometry(matcher):
    weak_embedding = normalized_embedding(
        [
            0.44,
            sqrt(1.0 - 0.44**2),
            0.0,
            0.0,
        ]
    )

    identity = make_identity(
        frames=[1, 2],
        positions=[
            (100, 100),
            (102, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
    )

    track = make_track(
        frames=[3, 4],
        positions=[
            (105, 100),
            (107, 100),
        ],
        embeddings=[
            weak_embedding,
            weak_embedding,
        ],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert result.appearance_evaluated is True

    assert (
        result.appearance_similarity >= matcher.minimum_appearance_similarity
    )

    assert result.spatial_distance <= matcher.max_spatial_distance

    assert_match(
        result,
        "weak_appearance_strong_geometry",
    )


# ============================================================
# TEST 11: Weak appearance cannot rescue weak geometry
# ============================================================


def test_11_weak_appearance_cannot_rescue_weak_geometry(matcher):
    weak_embedding = normalized_embedding(
        [
            0.44,
            sqrt(1.0 - 0.44**2),
            0.0,
            0.0,
        ]
    )

    identity = make_identity(
        frames=[1, 2],
        positions=[
            (100, 100),
            (100, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
    )

    track = make_track(
        frames=[3, 4],
        positions=[
            (180, 100),
            (180, 100),
        ],
        embeddings=[
            weak_embedding,
            weak_embedding,
        ],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert result.appearance_evaluated is True

    assert (
        result.appearance_similarity >= matcher.minimum_appearance_similarity
    )

    assert_rejected(
        result,
        "weak_appearance_geometry",
    )


# ============================================================
# TEST 12: Normal appearance + reasonable geometry
# ============================================================


def test_12_normal_appearance_with_reasonable_geometry(matcher):
    normal_embedding = normalized_embedding(
        [
            0.60,
            sqrt(1.0 - 0.60**2),
            0.0,
            0.0,
        ]
    )

    identity = make_identity(
        frames=[1, 2],
        positions=[
            (100, 100),
            (102, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
    )

    track = make_track(
        frames=[3, 4],
        positions=[
            (150, 100),
            (152, 100),
        ],
        embeddings=[
            normal_embedding,
            normal_embedding,
        ],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert result.appearance_evaluated is True

    assert (
        result.appearance_similarity >= matcher.appearance_normal_similarity
    )

    assert_match(
        result,
        "normal_appearance_geometry",
    )


# ============================================================
# TEST 13: Normal appearance cannot rescue bad geometry
# ============================================================


def test_13_normal_appearance_cannot_rescue_bad_geometry(matcher):
    normal_embedding = normalized_embedding(
        [
            0.60,
            sqrt(1.0 - 0.60**2),
            0.0,
            0.0,
        ]
    )

    identity = make_identity(
        frames=[1, 2],
        positions=[
            (100, 100),
            (100, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
    )

    track = make_track(
        frames=[3, 4],
        positions=[
            (320, 100),
            (320, 100),
        ],
        embeddings=[
            normal_embedding,
            normal_embedding,
        ],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert result.appearance_evaluated is True

    assert (
        result.appearance_similarity >= matcher.appearance_normal_similarity
    )

    assert result.spatial_distance == 220.0

    assert_rejected(
        result,
        "normal_appearance_geometry_too_weak",
    )


# ============================================================
# TEST 14: Long gap + weak appearance
# ============================================================


def test_14_long_gap_requires_stronger_appearance(matcher):
    identity = make_identity(
        frames=[1, 2],
        positions=[
            (100, 100),
            (102, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
    )

    weak_embedding = normalized_embedding(
        [
            0.44,
            sqrt(1.0 - 0.44**2),
            0.0,
            0.0,
        ]
    )

    track = make_track(
        frames=[41, 42],
        positions=[
            (105, 100),
            (107, 100),
        ],
        embeddings=[
            weak_embedding,
            weak_embedding,
        ],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert result.temporal_gap == 38

    assert result.appearance_evaluated is True

    assert result.appearance_similarity < matcher.long_gap_min_appearance

    assert_rejected(
        result,
        "long_gap_weak_appearance",
    )


# ============================================================
# TEST 15: Long gap + strong appearance
# ============================================================


def test_15_long_gap_with_strong_appearance_can_pass(matcher):
    identity = make_identity(
        frames=[1, 2],
        positions=[
            (100, 100),
            (102, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
    )

    track = make_track(
        frames=[41, 42],
        positions=[
            (110, 100),
            (112, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert result.temporal_gap == 38

    assert result.appearance_similarity >= 0.85

    assert_match(
        result,
        "strong_appearance",
    )


# ============================================================
# TEST 16: Team mismatch
# ============================================================


def test_16_team_mismatch_is_hard_rejection(matcher):
    identity = make_identity(
        frames=[1, 2],
        positions=[
            (100, 100),
            (102, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
        team="team_a",
    )

    track = make_track(
        frames=[3, 4],
        positions=[
            (104, 100),
            (106, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
        team="team_b",
    )

    result = matcher.match(
        track,
        identity,
    )

    assert_rejected(
        result,
        "team_mismatch",
    )

    assert result.appearance_evaluated is True
    assert result.team_match is False


# ============================================================
# TEST 17: Jersey mismatch
# ============================================================


def test_17_jersey_mismatch_is_hard_rejection(matcher):
    identity = make_identity(
        frames=[1, 2],
        positions=[
            (100, 100),
            (102, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
        team="team_a",
        jersey=10,
    )

    track = make_track(
        frames=[3, 4],
        positions=[
            (104, 100),
            (106, 100),
        ],
        embeddings=[
            EMBEDDING_A,
            EMBEDDING_A,
        ],
        team="team_a",
        jersey=7,
    )

    result = matcher.match(
        track,
        identity,
    )

    assert_rejected(
        result,
        "jersey_mismatch",
    )

    assert result.appearance_evaluated is True
    assert result.team_match is True
    assert result.jersey_match is False


# ============================================================
# TEST 18: Empty track
# ============================================================


def test_18_empty_track_rejected(matcher):
    track = Track(
        local_id=1,
        class_name="player",
    )

    identity = make_identity()

    result = matcher.match(
        track,
        identity,
    )

    assert_rejected(
        result,
        "track_has_no_observations",
    )

    assert result.appearance_evaluated is False


# ============================================================
# TEST 19: Empty identity
# ============================================================


def test_19_empty_identity_rejected(matcher):
    track = make_track()

    identity = Identity(
        identity_id="identity-empty",
        class_name="player",
    )

    result = matcher.match(
        track,
        identity,
    )

    assert_rejected(
        result,
        "identity_has_no_observations",
    )

    assert result.appearance_evaluated is False


# ============================================================
# TEST 20: Spatial information unavailable guard
# ============================================================


def test_20_spatial_information_unavailable_guard_is_blocked_by_temporal_gate(
    matcher,
):
    identity = Identity(
        identity_id="identity-1",
        class_name="player",
    )

    identity.add_observation(
        observation(
            frame=1,
            x=100,
            y=100,
        )
    )

    track = make_track(
        frames=[1, 2],
    )

    result = matcher.match(
        track,
        identity,
    )

    assert_rejected(
        result,
        "track_overlaps_identity",
    )

    assert result.temporal_gap == -1
    assert result.spatial_distance == float("inf")
    assert result.appearance_evaluated is False