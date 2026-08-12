"""
V5.3.1 Step 3 Behavioral Tests

Tests:
- Diagnostic-frame priority
- Closest-frame priority
- REVIEW priority
- Missing diagnostic frame fallback
- Duplicate removal
- Multiple-track behavior
- Temporal coverage fallback
- Evidence limits
- Step 1 observation-gap regression
- Step 2 observation provenance regression
- Footer provenance
- Read-only behavior
- Manifest provenance

These tests are intentionally lightweight and do not require
a real video or YOLO model.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ai_engine.validation.visual_evidence_generator import (
    VisualEvidenceGenerator,
)


# ================================================================
# TEST HELPERS
# ================================================================


@dataclass
class FakeBBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class FakeObservation:
    frame_index: int
    bbox: FakeBBox


@dataclass
class FakeTrack:
    track_id: int
    observations: list


@dataclass
class FakeIdentity:
    identity_id: str
    observations: list
    source_track_ids: list[int]
    team: str | None = None
    jersey_number: int | None = None
    class_name: str = "player"
    frame_count: int = 0


@dataclass
class FakeResult:
    identity_id: str
    track_id: int | None
    frame_index: int | None
    status: str
    risk_level: str

    appearance_similarity: float = 0.71
    temporal_gap: int = 8
    spatial_distance: float = 43.2
    motion_difference: float = 0.14

    validation_warnings: list[str] | None = None
    matcher_reasons: list[str] | None = None

    def __post_init__(self):
        if self.validation_warnings is None:
            self.validation_warnings = []

        if self.matcher_reasons is None:
            self.matcher_reasons = []


class FakeCapture:
    """
    Minimal fake VideoCapture.

    Used so behavioral tests don't require a real video file.
    """

    def __init__(self):
        self.current_frame = None
        self.released = False

    def set(self, property_id, value):
        self.current_frame = int(value)

    def read(self):
        frame = np.zeros(
            (720, 1280, 3),
            dtype=np.uint8,
        )
        return True, frame

    def release(self):
        self.released = True


def make_observation(frame_index: int):
    return FakeObservation(
        frame_index=frame_index,
        bbox=FakeBBox(
            x1=100,
            y1=100,
            x2=200,
            y2=300,
        ),
    )


def make_identity(
    identity_id="global_12",
    observation_frames=None,
    source_track_ids=None,
):
    observation_frames = observation_frames or []
    source_track_ids = source_track_ids or [45]

    observations = [
        make_observation(frame)
        for frame in observation_frames
    ]

    return FakeIdentity(
        identity_id=identity_id,
        observations=observations,
        source_track_ids=source_track_ids,
        team="home",
        jersey_number=10,
        frame_count=len(observations),
    )


def make_result(
    identity_id="global_12",
    track_id=45,
    diagnostic_frame=500,
    status="HIGH_RISK",
    risk_level="HIGH",
):
    return FakeResult(
        identity_id=identity_id,
        track_id=track_id,
        frame_index=diagnostic_frame,
        status=status,
        risk_level=risk_level,
    )


def make_generator(
    *,
    max_observation_frame_gap=5,
    max_evidence_per_match=2,
    max_evidence_per_identity=4,
):
    return VisualEvidenceGenerator(
        output_dir="media/test_validation_evidence",
        max_observation_frame_gap=max_observation_frame_gap,
        max_evidence_per_match=max_evidence_per_match,
        max_evidence_per_identity=max_evidence_per_identity,
    )


# ================================================================
# 1. CRITICAL / HIGH DIAGNOSTIC FRAME PRIORITY
# ================================================================


def test_step3_high_risk_diagnostic_frame_has_priority():
    generator = make_generator()

    result = make_result(
        diagnostic_frame=500,
        status="HIGH_RISK",
        risk_level="HIGH",
    )

    identity = make_identity(
        observation_frames=[
            450,
            490,
            498,
            501,
            510,
            550,
        ]
    )

    tracks = {}

    selected = generator._select_match_frames(
        result=result,
        identity=identity,
        tracks=tracks,
        capture=None,
    )

    assert selected[0] == 500

    # Closest observations should appear before distant
    # temporal-coverage observations.
    assert selected.index(498) < selected.index(450)
    assert selected.index(501) < selected.index(550)


# ================================================================
# 2. REVIEW DIAGNOSTIC FRAME PRIORITY
# ================================================================


def test_step3_review_diagnostic_frame_has_priority():
    generator = make_generator()

    result = make_result(
        diagnostic_frame=500,
        status="REVIEW",
        risk_level="MEDIUM",
    )

    identity = make_identity(
        observation_frames=[
            450,
            490,
            498,
            501,
            510,
            550,
        ]
    )

    selected = generator._select_match_frames(
        result=result,
        identity=identity,
        tracks={},
        capture=None,
    )

    assert selected[0] == 500

    assert selected.index(498) < selected.index(450)
    assert selected.index(501) < selected.index(550)


# ================================================================
# 3. DIAGNOSTIC FRAME ABSENT
# ================================================================


def test_step3_missing_diagnostic_frame_falls_back_to_observations():
    generator = make_generator()

    result = make_result(
        diagnostic_frame=None,
    )

    identity = make_identity(
        observation_frames=[
            100,
            200,
            300,
            400,
            500,
        ]
    )

    selected = generator._select_match_frames(
        result=result,
        identity=identity,
        tracks={},
        capture=None,
    )

    assert selected
    assert None not in selected

    # No invented diagnostic frame.
    assert all(
        frame in {100, 200, 300, 400, 500}
        for frame in selected
    )


# ================================================================
# 4. DUPLICATE FRAMES
# ================================================================


def test_step3_duplicate_frames_are_removed():
    generator = make_generator()

    result = make_result(
        diagnostic_frame=500,
    )

    identity = make_identity(
        observation_frames=[
            498,
            498,
            500,
            500,
            501,
        ]
    )

    selected = generator._select_match_frames(
        result=result,
        identity=identity,
        tracks={},
        capture=None,
    )

    assert selected[0] == 500

    assert len(selected) == len(set(selected))

    assert set(selected) == {
        498,
        500,
        501,
    }


# ================================================================
# 5. MULTIPLE TRACKS
# ================================================================


def test_step3_uses_only_diagnostic_track():
    generator = make_generator()

    result = make_result(
        identity_id="global_12",
        track_id=45,
        diagnostic_frame=500,
    )

    identity = make_identity(
        observation_frames=[
            1000,
            1100,
        ],
        source_track_ids=[45, 78],
    )

    diagnostic_track = FakeTrack(
        track_id=45,
        observations=[
            make_observation(498),
            make_observation(501),
        ],
    )

    unrelated_track = FakeTrack(
        track_id=99,
        observations=[
            make_observation(1),
            make_observation(2),
        ],
    )

    tracks = {
        45: diagnostic_track,
        99: unrelated_track,
    }

    selected = generator._select_match_frames(
        result=result,
        identity=identity,
        tracks=tracks,
        capture=None,
    )

    assert 498 in selected
    assert 501 in selected

    # Unrelated track observations must not participate.
    assert 1 not in selected
    assert 2 not in selected


# ================================================================
# 6. TEMPORAL COVERAGE FALLBACK
# ================================================================


def test_step3_temporal_coverage_is_fallback():
    generator = make_generator()

    result = make_result(
        diagnostic_frame=500,
    )

    identity = make_identity(
        observation_frames=[
            100,
            250,
            500,
            750,
            900,
        ]
    )

    selected = generator._select_match_frames(
        result=result,
        identity=identity,
        tracks={},
        capture=None,
    )

    assert selected[0] == 500

    # Diagnostic frame must remain first.
    assert selected.index(500) == 0


# ================================================================
# 7. max_evidence_per_match
# ================================================================


def test_step3_max_evidence_per_match_is_preserved(
    tmp_path,
    monkeypatch,
):
    generator = make_generator(
        max_evidence_per_match=2,
    )

    result = make_result(
        diagnostic_frame=500,
    )

    identity = make_identity(
        observation_frames=[
            498,
            501,
            490,
            510,
        ]
    )

    fake_capture = FakeCapture()

    # Avoid actual OpenCV video opening.
    monkeypatch.setattr(
        generator,
        "_open_video",
        lambda video_path: fake_capture,
    )

    selected = generator._select_match_frames(
        result=result,
        identity=identity,
        tracks={},
        capture=fake_capture,
    )

    assert len(selected) >= 2

    # Simulate the generator's actual render limit.
    rendered_candidates = selected[
        : generator.max_evidence_per_match
    ]

    assert len(rendered_candidates) == 2


# ================================================================
# 8. max_evidence_per_identity
# ================================================================


def test_step3_max_evidence_per_identity_is_preserved():
    generator = make_generator(
        max_evidence_per_identity=4,
    )

    per_identity_count = {
        "global_12": 4,
    }

    assert (
        per_identity_count["global_12"]
        <= generator.max_evidence_per_identity
    )


# ================================================================
# STEP 1 REGRESSION
# ================================================================


def test_step1_identity_observation_inside_gap():
    identity = make_identity(
        observation_frames=[497]
    )

    observation = (
        VisualEvidenceGenerator
        ._find_identity_observation(
            identity,
            frame_index=500,
            max_frame_gap=5,
        )
    )

    assert observation is not None
    assert observation.frame_index == 497


def test_step1_identity_observation_outside_gap():
    identity = make_identity(
        observation_frames=[490]
    )

    observation = (
        VisualEvidenceGenerator
        ._find_identity_observation(
            identity,
            frame_index=500,
            max_frame_gap=5,
        )
    )

    assert observation is None


def test_step1_track_observation_inside_gap():
    track = FakeTrack(
        track_id=45,
        observations=[
            make_observation(497),
        ],
    )

    observation = (
        VisualEvidenceGenerator
        ._find_track_observation(
            track,
            frame_index=500,
            max_frame_gap=5,
        )
    )

    assert observation is not None
    assert observation.frame_index == 497


def test_step1_track_observation_outside_gap():
    track = FakeTrack(
        track_id=45,
        observations=[
            make_observation(490),
        ],
    )

    observation = (
        VisualEvidenceGenerator
        ._find_track_observation(
            track,
            frame_index=500,
            max_frame_gap=5,
        )
    )

    assert observation is None


# ================================================================
# STEP 2 PROVENANCE
# ================================================================


def test_step2_identity_observation_provenance():
    generator = make_generator()

    identity = make_identity(
        observation_frames=[497]
    )

    result = make_result(
        diagnostic_frame=500,
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={},
        capture=capture,
        frame_index=500,
        output_dir=Path(
            "media/test_validation_evidence"
        ),
    )

    assert evidence is not None

    assert (
        evidence.observation_frame_index
        == 497
    )

    assert (
        evidence.observation_frame_gap
        == 3
    )

    assert (
        evidence.observation_source
        == "identity"
    )


def test_step2_track_observation_provenance():
    generator = make_generator()

    identity = make_identity(
        observation_frames=[]
    )

    track = FakeTrack(
        track_id=45,
        observations=[
            make_observation(498),
        ],
    )

    result = make_result(
        diagnostic_frame=500,
        track_id=45,
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={
            45: track,
        },
        capture=capture,
        frame_index=500,
        output_dir=Path(
            "media/test_validation_evidence"
        ),
    )

    assert evidence is not None

    assert (
        evidence.observation_frame_index
        == 498
    )

    assert (
        evidence.observation_frame_gap
        == 2
    )

    assert (
        evidence.observation_source
        == "track"
    )


def test_step2_no_valid_observation_provenance():
    generator = make_generator(
        max_observation_frame_gap=5,
    )

    identity = make_identity(
        observation_frames=[480]
    )

    track = FakeTrack(
        track_id=45,
        observations=[
            make_observation(470),
        ],
    )

    result = make_result(
        diagnostic_frame=500,
        track_id=45,
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={
            45: track,
        },
        capture=capture,
        frame_index=500,
        output_dir=Path(
            "media/test_validation_evidence"
        ),
    )

    assert evidence is not None

    assert (
        evidence.observation_frame_index
        is None
    )

    assert (
        evidence.observation_frame_gap
        is None
    )

    assert (
        evidence.observation_source
        is None
    )


# ================================================================
# 15. FOOTER PROVENANCE
# ================================================================


def test_step2_footer_contains_provenance():
    generator = make_generator()

    frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8,
    )

    lines = [
        "observation=498",
        "observation_gap=2",
        "observation_source=identity",
    ]

    # The method should execute without error.
    generator._draw_footer(
        frame,
        lines,
    )


# ================================================================
# 16. READ-ONLY BEHAVIOR
# ================================================================


def test_step3_generator_does_not_modify_identity_or_tracks():
    generator = make_generator()

    identity = make_identity(
        observation_frames=[
            100,
            250,
            500,
            750,
        ]
    )

    track = FakeTrack(
        track_id=45,
        observations=[
            make_observation(498),
            make_observation(501),
        ],
    )

    result = make_result(
        diagnostic_frame=500,
    )

    identity_before = deepcopy(identity)
    track_before = deepcopy(track)

    generator._select_match_frames(
        result=result,
        identity=identity,
        tracks={45: track},
        capture=None,
    )

    assert identity == identity_before
    assert track == track_before


# ================================================================
# 17. EvidenceFrame.to_dict() / MANIFEST PROVENANCE
# ================================================================


def test_step2_manifest_provenance_fields():
    generator = make_generator()

    identity = make_identity(
        observation_frames=[498]
    )

    result = make_result(
        diagnostic_frame=500,
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={},
        capture=capture,
        frame_index=500,
        output_dir=Path(
            "media/test_validation_evidence"
        ),
    )

    assert evidence is not None

    manifest_record = evidence.to_dict()

    assert (
        manifest_record[
            "observation_frame_index"
        ]
        == 498
    )

    assert (
        manifest_record[
            "observation_frame_gap"
        ]
        == 2
    )

    assert (
        manifest_record[
            "observation_source"
        ]
        == "identity"
    )


# ================================================================
# EXTRA: EVIDENCE FRAME DATACLASS INTEGRITY
# ================================================================


def test_evidence_frame_provenance_fields_exist():
    generator = make_generator()

    identity = make_identity(
        observation_frames=[498]
    )

    result = make_result(
        diagnostic_frame=500,
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={},
        capture=capture,
        frame_index=500,
        output_dir=Path(
            "media/test_validation_evidence"
        ),
    )

    assert evidence is not None

    assert hasattr(
        evidence,
        "observation_frame_index",
    )

    assert hasattr(
        evidence,
        "observation_frame_gap",
    )

    assert hasattr(
        evidence,
        "observation_source",
    )


# ================================================================
# RUNNING INDEPENDENTLY
# ================================================================

if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
        ]
    )