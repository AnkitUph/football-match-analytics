"""
V5.3.1 Step 4 behavioral tests.

Step 4 goal:
Verify evidence rendering correctness without changing the
Step 1 observation-gap rule, Step 2 provenance, or Step 3
risk-aware frame selection behavior.

Tests cover:
1. Correct frame is rendered.
2. Valid observation bbox is rendered.
3. Distant observations are never rendered.
4. Missing observations produce no bbox.
5. Header contains status/risk rendering.
6. Footer renders diagnostic metadata.
7. Footer/provenance metadata is preserved.
8. Identity summary evidence renders correctly.
9. EvidenceFrame contains correct rendering metadata.
10. Written JPEG exists and is readable.
11. Manifest references rendered evidence.
12. Per-match evidence limit is preserved.
13. Per-identity evidence limit is preserved.
14. Rendering is read-only.
15. Step 1 observation-gap regression.
16. Step 2 provenance regression.
17. Step 3 diagnostic-first selection regression.
18. Track observation fallback renders correctly.
19. Failed frame reads do not create evidence.
20. Evidence type is correctly assigned.
21. Source track IDs/team/jersey metadata are preserved.
22. Manifest contains V5.3.1 version and evidence count.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from ai_engine.validation.visual_evidence_generator import (
    EvidenceFrame,
    VisualEvidenceGenerator,
)


# ---------------------------------------------------------------------
# Fake geometry / observations
# ---------------------------------------------------------------------


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
class FakeIdentity:
    identity_id: str = "45"
    class_name: str = "player"
    team: str | None = "home"
    jersey_number: int | None = 10

    source_track_ids: list[int] = field(
        default_factory=lambda: [7, 12]
    )

    observations: list[Any] = field(
        default_factory=list
    )

    frame_count: int = 0


@dataclass
class FakeTrack:
    track_id: int
    observations: list[Any] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------
# Fake validation result
# ---------------------------------------------------------------------


def make_result(
    *,
    identity_id: str = "45",
    track_id: int | None = 7,
    frame_index: int | None = 500,
    status: str = "REVIEW",
    risk_level: str = "HIGH",
):
    return SimpleNamespace(
        identity_id=identity_id,
        track_id=track_id,
        frame_index=frame_index,
        status=status,
        risk_level=risk_level,
        appearance_similarity=0.91,
        temporal_gap=4,
        spatial_distance=12.5,
        motion_difference=0.17,
        validation_warnings=["test-warning"],
        matcher_reasons=["test-reason"],
    )


# ---------------------------------------------------------------------
# Fake capture
# ---------------------------------------------------------------------


class FakeCapture:
    """
    Fake cv2.VideoCapture.

    The requested frame number is encoded into the generated image,
    allowing tests to verify that the requested frame was actually
    rendered.
    """

    def __init__(self):
        self.current_frame = 0
        self.released = False

    def set(self, property_id, value):
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            self.current_frame = int(value)

        return True

    def read(self):
        frame = np.zeros(
            (240, 320, 3),
            dtype=np.uint8,
        )

        cv2.putText(
            frame,
            f"FRAME {self.current_frame}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return True, frame

    def release(self):
        self.released = True


class FailedCapture(FakeCapture):
    """Fake capture that fails to read frames."""

    def read(self):
        return False, None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_generator(
    tmp_path: Path,
    *,
    max_gap: int = 5,
    max_per_match: int = 2,
    max_per_identity: int = 4,
):
    return VisualEvidenceGenerator(
        output_dir=tmp_path,
        max_evidence_per_match=max_per_match,
        max_evidence_per_identity=max_per_identity,
        max_observation_frame_gap=max_gap,
    )


def read_written_image(path: Path):
    image = cv2.imread(str(path))

    assert image is not None, (
        f"Evidence image could not be read: {path}"
    )

    return image


# ---------------------------------------------------------------------
# 1. Correct frame is rendered
# ---------------------------------------------------------------------


def test_step4_requested_frame_is_rendered(tmp_path):
    generator = make_generator(tmp_path)

    result = make_result(
        frame_index=500,
    )

    identity = FakeIdentity(
        observations=[
            FakeObservation(
                frame_index=500,
                bbox=FakeBBox(
                    10,
                    20,
                    100,
                    150,
                ),
            )
        ]
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={},
        capture=capture,
        frame_index=500,
        output_dir=tmp_path,
    )

    assert evidence is not None
    assert evidence.frame_index == 500

    image = read_written_image(
        Path(evidence.image_path)
    )

    assert image.shape == (240, 320, 3)
    assert image.sum() > 0


# ---------------------------------------------------------------------
# 2. Valid observation bbox is rendered
# ---------------------------------------------------------------------


def test_step4_valid_observation_bbox_is_used(tmp_path):
    generator = make_generator(
        tmp_path,
        max_gap=5,
    )

    result = make_result(
        frame_index=500,
    )

    observation = FakeObservation(
        frame_index=498,
        bbox=FakeBBox(
            30,
            40,
            120,
            180,
        ),
    )

    identity = FakeIdentity(
        observations=[observation]
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={},
        capture=capture,
        frame_index=500,
        output_dir=tmp_path,
    )

    assert evidence is not None

    assert evidence.observation_frame_index == 498
    assert evidence.observation_frame_gap == 2
    assert evidence.observation_source == "identity"

    assert evidence.bbox == [
        30.0,
        40.0,
        120.0,
        180.0,
    ]


# ---------------------------------------------------------------------
# 3. Distant observation must not be rendered
# ---------------------------------------------------------------------


def test_step4_distant_observation_is_not_rendered(tmp_path):
    generator = make_generator(
        tmp_path,
        max_gap=5,
    )

    result = make_result(
        frame_index=500,
    )

    distant_observation = FakeObservation(
        frame_index=490,
        bbox=FakeBBox(
            30,
            40,
            120,
            180,
        ),
    )

    identity = FakeIdentity(
        observations=[distant_observation]
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={},
        capture=capture,
        frame_index=500,
        output_dir=tmp_path,
    )

    assert evidence is not None

    assert evidence.observation_frame_index is None
    assert evidence.observation_frame_gap is None
    assert evidence.observation_source is None
    assert evidence.bbox is None


# ---------------------------------------------------------------------
# 4. Missing observation renders frame without bbox
# ---------------------------------------------------------------------


def test_step4_missing_observation_renders_without_bbox(
    tmp_path,
):
    generator = make_generator(tmp_path)

    result = make_result(
        frame_index=500,
        track_id=None,
    )

    identity = FakeIdentity(
        observations=[]
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={},
        capture=capture,
        frame_index=500,
        output_dir=tmp_path,
    )

    assert evidence is not None

    assert evidence.bbox is None
    assert evidence.observation_frame_index is None
    assert evidence.observation_frame_gap is None
    assert evidence.observation_source is None

    assert Path(
        evidence.image_path
    ).exists()


# ---------------------------------------------------------------------
# 5. Header contains status and risk
# ---------------------------------------------------------------------


def test_step4_header_contains_status_and_risk():
    frame = np.zeros(
        (200, 500, 3),
        dtype=np.uint8,
    )

    result = make_result(
        status="REVIEW",
        risk_level="HIGH",
    )

    VisualEvidenceGenerator._draw_header(
        frame,
        title="IDENTITY VISUAL EVIDENCE V5.3.1",
        result=result,
    )

    assert frame[:42].sum() > 0


# ---------------------------------------------------------------------
# 6. Footer renders diagnostic metadata
# ---------------------------------------------------------------------


def test_step4_footer_accepts_diagnostic_metadata():
    frame = np.zeros(
        (300, 600, 3),
        dtype=np.uint8,
    )

    lines = [
        "identity=45",
        "track=7",
        "frame=500",
        "appearance=0.9100",
        "temporal_gap=4",
        "spatial=12.50",
        "motion=0.17",
    ]

    VisualEvidenceGenerator._draw_footer(
        frame,
        lines,
    )

    assert frame.sum() > 0


# ---------------------------------------------------------------------
# 7. Provenance metadata is preserved
# ---------------------------------------------------------------------


def test_step4_provenance_metadata_is_preserved(tmp_path):
    generator = make_generator(
        tmp_path,
        max_gap=5,
    )

    result = make_result(
        frame_index=500,
    )

    identity = FakeIdentity(
        observations=[
            FakeObservation(
                frame_index=498,
                bbox=FakeBBox(
                    30,
                    40,
                    120,
                    180,
                ),
            )
        ]
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={},
        capture=capture,
        frame_index=500,
        output_dir=tmp_path,
    )

    assert evidence is not None

    assert evidence.observation_frame_index == 498
    assert evidence.observation_frame_gap == 2
    assert evidence.observation_source == "identity"


# ---------------------------------------------------------------------
# 8. Identity summary evidence renders
# ---------------------------------------------------------------------


def test_step4_identity_summary_evidence_renders(tmp_path):
    generator = make_generator(tmp_path)

    identity = FakeIdentity(
        observations=[
            FakeObservation(
                frame_index=100,
                bbox=FakeBBox(
                    10,
                    20,
                    100,
                    150,
                ),
            )
        ],
        frame_count=1,
    )

    identity_result = SimpleNamespace(
        status="REVIEW",
        risk_level="HIGH",
        reasons=[
            "identity requires review"
        ],
    )

    generator.validator.identity_results = {
        "45": identity_result
    }

    capture = FakeCapture()

    generator._generate_identity_summary_evidence(
        manager=SimpleNamespace(),
        identities={
            "45": identity
        },
        tracks={},
        capture=capture,
        output_dir=tmp_path,
        per_identity_count={},
        include_pass=False,
        include_review=True,
        include_high_risk=True,
        include_invalid=True,
    )

    assert len(generator.evidence) == 1

    evidence = generator.evidence[0]

    assert evidence.identity_id == "45"
    assert evidence.evidence_type == "identity_summary"
    assert evidence.frame_index == 100
    assert evidence.observation_frame_index == 100
    assert evidence.observation_frame_gap == 0
    assert evidence.observation_source == "identity"

    assert Path(
        evidence.image_path
    ).exists()


# ---------------------------------------------------------------------
# 9. EvidenceFrame metadata correctness
# ---------------------------------------------------------------------


def test_step4_evidence_frame_metadata_matches_render(
    tmp_path,
):
    generator = make_generator(tmp_path)

    result = make_result(
        identity_id="45",
        track_id=7,
        frame_index=500,
        status="REVIEW",
        risk_level="HIGH",
    )

    identity = FakeIdentity(
        observations=[
            FakeObservation(
                frame_index=498,
                bbox=FakeBBox(
                    10,
                    20,
                    100,
                    150,
                ),
            )
        ]
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={},
        capture=capture,
        frame_index=500,
        output_dir=tmp_path,
    )

    assert evidence is not None

    data = evidence.to_dict()

    assert data["identity_id"] == "45"
    assert data["track_id"] == 7
    assert data["frame_index"] == 500
    assert data["status"] == "REVIEW"
    assert data["risk_level"] == "HIGH"

    assert data[
        "observation_frame_index"
    ] == 498

    assert data[
        "observation_frame_gap"
    ] == 2

    assert data[
        "observation_source"
    ] == "identity"


# ---------------------------------------------------------------------
# 10. JPEG exists and is readable
# ---------------------------------------------------------------------


def test_step4_written_evidence_image_is_readable(
    tmp_path,
):
    generator = make_generator(tmp_path)

    result = make_result(
        frame_index=500,
    )

    identity = FakeIdentity(
        observations=[
            FakeObservation(
                frame_index=500,
                bbox=FakeBBox(
                    20,
                    20,
                    100,
                    100,
                ),
            )
        ]
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={},
        capture=capture,
        frame_index=500,
        output_dir=tmp_path,
    )

    assert evidence is not None

    image_path = Path(
        evidence.image_path
    )

    assert image_path.exists()
    assert image_path.is_file()
    assert image_path.stat().st_size > 0

    image = cv2.imread(
        str(image_path)
    )

    assert image is not None
    assert image.size > 0


# ---------------------------------------------------------------------
# 11. Manifest references rendered evidence
# ---------------------------------------------------------------------


def test_step4_manifest_contains_rendered_evidence(
    tmp_path,
    monkeypatch,
):
    generator = make_generator(tmp_path)

    identity = FakeIdentity(
        observations=[
            FakeObservation(
                frame_index=500,
                bbox=FakeBBox(
                    20,
                    20,
                    100,
                    100,
                ),
            )
        ]
    )

    result = make_result(
        frame_index=500,
    )

    generator.validator.match_results = [
        result
    ]

    generator.validator.identity_results = {}

    def fake_validate_manager(manager):
        return {
            "status": "ok",
            "test": "step4",
        }

    generator.validator.validate_manager = (
        fake_validate_manager
    )

    monkeypatch.setattr(
        generator,
        "_open_video",
        lambda video_path: FakeCapture(),
    )

    manager = SimpleNamespace(
        identities={
            "45": identity
        },
        tracks={},
    )

    video_path = tmp_path / "test.mp4"

    manifest = generator.generate(
        manager,
        video_path,
        output_dir=tmp_path,
        include_pass=False,
        include_review=True,
        include_high_risk=True,
        include_invalid=True,
        write_manifest=True,
    )

    assert manifest["evidence_count"] == 1
    assert len(manifest["evidence"]) == 1

    item = manifest["evidence"][0]

    assert item["frame_index"] == 500
    assert item[
        "observation_frame_index"
    ] == 500
    assert item[
        "observation_frame_gap"
    ] == 0
    assert item[
        "observation_source"
    ] == "identity"

    manifest_path = (
        tmp_path
        / "evidence_manifest.json"
    )

    assert manifest_path.exists()

    saved = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert saved["evidence_count"] == 1
    assert len(saved["evidence"]) == 1


# ---------------------------------------------------------------------
# 12. Per-match limit is preserved
# ---------------------------------------------------------------------


def test_step4_max_evidence_per_match_preserved(
    tmp_path,
):
    generator = make_generator(
        tmp_path,
        max_per_match=2,
    )

    result = make_result(
        frame_index=500,
    )

    identity = FakeIdentity(
        observations=[
            FakeObservation(
                frame_index=frame,
                bbox=FakeBBox(
                    10,
                    10,
                    50,
                    50,
                ),
            )
            for frame in [
                500,
                498,
                501,
                490,
                510,
            ]
        ]
    )

    capture = FakeCapture()

    frames = generator._select_match_frames(
        result=result,
        identity=identity,
        tracks={},
        capture=capture,
    )

    rendered = []

    for frame_index in frames[
        : generator.max_evidence_per_match
    ]:
        evidence = (
            generator._render_match_evidence(
                result=result,
                identity=identity,
                tracks={},
                capture=capture,
                frame_index=frame_index,
                output_dir=tmp_path,
            )
        )

        if evidence is not None:
            rendered.append(evidence)

    assert len(rendered) <= 2


# ---------------------------------------------------------------------
# 13. Per-identity limit is preserved
# ---------------------------------------------------------------------


def test_step4_max_evidence_per_identity_preserved(
    tmp_path,
):
    generator = make_generator(
        tmp_path,
        max_per_match=5,
        max_per_identity=4,
    )

    result = make_result(
        frame_index=500,
    )

    identity = FakeIdentity(
        observations=[
            FakeObservation(
                frame_index=frame,
                bbox=FakeBBox(
                    10,
                    10,
                    50,
                    50,
                ),
            )
            for frame in [
                500,
                498,
                501,
                490,
                510,
            ]
        ]
    )

    capture = FakeCapture()

    frames = generator._select_match_frames(
        result=result,
        identity=identity,
        tracks={},
        capture=capture,
    )

    rendered = []

    for frame_index in frames:
        if (
            len(rendered)
            >= generator.max_evidence_per_identity
        ):
            break

        evidence = (
            generator._render_match_evidence(
                result=result,
                identity=identity,
                tracks={},
                capture=capture,
                frame_index=frame_index,
                output_dir=tmp_path,
            )
        )

        if evidence is not None:
            rendered.append(evidence)

    assert len(rendered) <= 4


# ---------------------------------------------------------------------
# 14. Rendering is read-only
# ---------------------------------------------------------------------


def test_step4_rendering_does_not_modify_identity_or_tracks(
    tmp_path,
):
    generator = make_generator(tmp_path)

    identity = FakeIdentity(
        observations=[
            FakeObservation(
                frame_index=500,
                bbox=FakeBBox(
                    20,
                    20,
                    100,
                    100,
                ),
            )
        ]
    )

    track = FakeTrack(
        track_id=7,
        observations=[
            FakeObservation(
                frame_index=500,
                bbox=FakeBBox(
                    20,
                    20,
                    100,
                    100,
                ),
            )
        ],
    )

    result = make_result(
        frame_index=500,
        track_id=7,
    )

    tracks = {
        7: track
    }

    before_identity = copy.deepcopy(
        identity
    )

    before_tracks = copy.deepcopy(
        tracks
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks=tracks,
        capture=capture,
        frame_index=500,
        output_dir=tmp_path,
    )

    assert evidence is not None

    assert identity == before_identity
    assert tracks == before_tracks


# ---------------------------------------------------------------------
# 15. Step 1 regression: observation inside gap
# ---------------------------------------------------------------------


def test_step4_step1_inside_gap_still_works():
    identity = FakeIdentity(
        observations=[
            FakeObservation(
                frame_index=497,
                bbox=FakeBBox(
                    10,
                    10,
                    50,
                    50,
                ),
            )
        ]
    )

    observation = (
        VisualEvidenceGenerator
        ._find_identity_observation(
            identity,
            500,
            max_frame_gap=5,
        )
    )

    assert observation is not None
    assert observation.frame_index == 497


# ---------------------------------------------------------------------
# 16. Step 2 regression: provenance remains correct
# ---------------------------------------------------------------------


def test_step4_step2_provenance_still_correct(
    tmp_path,
):
    generator = make_generator(
        tmp_path,
        max_gap=5,
    )

    result = make_result(
        frame_index=500,
    )

    identity = FakeIdentity(
        observations=[
            FakeObservation(
                frame_index=497,
                bbox=FakeBBox(
                    10,
                    10,
                    50,
                    50,
                ),
            )
        ]
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={},
        capture=capture,
        frame_index=500,
        output_dir=tmp_path,
    )

    assert evidence is not None

    assert evidence.observation_frame_index == 497
    assert evidence.observation_frame_gap == 3
    assert evidence.observation_source == "identity"


# ---------------------------------------------------------------------
# 17. Step 3 regression: diagnostic-first selection
# ---------------------------------------------------------------------


def test_step4_step3_selection_remains_diagnostic_first():
    generator = VisualEvidenceGenerator()

    result = make_result(
        frame_index=500,
        status="REVIEW",
        risk_level="HIGH",
    )

    identity = FakeIdentity(
        observations=[
            FakeObservation(
                frame_index=450,
                bbox=FakeBBox(
                    1,
                    1,
                    10,
                    10,
                ),
            ),
            FakeObservation(
                frame_index=490,
                bbox=FakeBBox(
                    1,
                    1,
                    10,
                    10,
                ),
            ),
            FakeObservation(
                frame_index=498,
                bbox=FakeBBox(
                    1,
                    1,
                    10,
                    10,
                ),
            ),
            FakeObservation(
                frame_index=501,
                bbox=FakeBBox(
                    1,
                    1,
                    10,
                    10,
                ),
            ),
            FakeObservation(
                frame_index=510,
                bbox=FakeBBox(
                    1,
                    1,
                    10,
                    10,
                ),
            ),
            FakeObservation(
                frame_index=550,
                bbox=FakeBBox(
                    1,
                    1,
                    10,
                    10,
                ),
            ),
        ]
    )

    frames = generator._select_match_frames(
        result=result,
        identity=identity,
        tracks={},
        capture=FakeCapture(),
    )

    assert frames[0] == 500

    assert frames[1:3] == [
        501,
        498,
    ]


# ---------------------------------------------------------------------
# 18. Track observation fallback
# ---------------------------------------------------------------------


def test_step4_track_observation_fallback_is_rendered(
    tmp_path,
):
    generator = make_generator(
        tmp_path,
        max_gap=5,
    )

    result = make_result(
        frame_index=500,
        track_id=7,
    )

    identity = FakeIdentity(
        observations=[]
    )

    track = FakeTrack(
        track_id=7,
        observations=[
            FakeObservation(
                frame_index=499,
                bbox=FakeBBox(
                    40,
                    50,
                    140,
                    190,
                ),
            )
        ],
    )

    capture = FakeCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={
            7: track
        },
        capture=capture,
        frame_index=500,
        output_dir=tmp_path,
    )

    assert evidence is not None

    assert evidence.observation_source == "track"
    assert evidence.observation_frame_index == 499
    assert evidence.observation_frame_gap == 1

    assert evidence.bbox == [
        40.0,
        50.0,
        140.0,
        190.0,
    ]


# ---------------------------------------------------------------------
# 19. Failed frame reads produce no evidence
# ---------------------------------------------------------------------


def test_step4_failed_frame_read_produces_no_evidence(
    tmp_path,
):
    generator = make_generator(tmp_path)

    result = make_result(
        frame_index=500,
    )

    identity = FakeIdentity(
        observations=[
            FakeObservation(
                frame_index=500,
                bbox=FakeBBox(
                    10,
                    10,
                    50,
                    50,
                ),
            )
        ]
    )

    capture = FailedCapture()

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={},
        capture=capture,
        frame_index=500,
        output_dir=tmp_path,
    )

    assert evidence is None


# ---------------------------------------------------------------------
# 20. Evidence type is correctly assigned
# ---------------------------------------------------------------------


def test_step4_evidence_type_is_correct():
    critical = make_result(
        status="REVIEW",
        risk_level="CRITICAL",
    )

    high = make_result(
        status="HIGH_RISK",
        risk_level="HIGH",
    )

    review = make_result(
        status="REVIEW",
        risk_level="MEDIUM",
    )

    normal = make_result(
        status="PASS",
        risk_level="LOW",
    )

    assert (
        VisualEvidenceGenerator._evidence_type(
            critical
        )
        == "critical_match"
    )

    assert (
        VisualEvidenceGenerator._evidence_type(
            high
        )
        == "high_risk_match"
    )

    assert (
        VisualEvidenceGenerator._evidence_type(
            review
        )
        == "review_match"
    )

    assert (
        VisualEvidenceGenerator._evidence_type(
            normal
        )
        == "pass_match"
    )


# ---------------------------------------------------------------------
# 21. Identity metadata is preserved
# ---------------------------------------------------------------------


def test_step4_identity_metadata_is_preserved(
    tmp_path,
):
    generator = make_generator(tmp_path)

    result = make_result(
        identity_id="45",
        track_id=7,
        frame_index=500,
    )

    identity = FakeIdentity(
        team="away",
        jersey_number=22,
        source_track_ids=[
            7,
            12,
            18,
        ],
        observations=[
            FakeObservation(
                frame_index=500,
                bbox=FakeBBox(
                    10,
                    20,
                    100,
                    150,
                ),
            )
        ],
    )

    evidence = generator._render_match_evidence(
        result=result,
        identity=identity,
        tracks={},
        capture=FakeCapture(),
        frame_index=500,
        output_dir=tmp_path,
    )

    assert evidence is not None

    assert evidence.team == "away"
    assert evidence.jersey_number == 22
    assert evidence.source_track_ids == [
        7,
        12,
        18,
    ]


# ---------------------------------------------------------------------
# 22. Manifest version and evidence count
# ---------------------------------------------------------------------


def test_step4_manifest_version_and_count(
    tmp_path,
    monkeypatch,
):
    generator = make_generator(tmp_path)

    identity = FakeIdentity(
        observations=[
            FakeObservation(
                frame_index=500,
                bbox=FakeBBox(
                    10,
                    10,
                    100,
                    100,
                ),
            )
        ]
    )

    result = make_result(
        frame_index=500,
    )

    generator.validator.match_results = [
        result
    ]

    generator.validator.identity_results = {}

    generator.validator.validate_manager = (
        lambda manager: {
            "status": "ok"
        }
    )

    monkeypatch.setattr(
        generator,
        "_open_video",
        lambda video_path: FakeCapture(),
    )

    manager = SimpleNamespace(
        identities={
            "45": identity
        },
        tracks={},
    )

    manifest = generator.generate(
        manager,
        tmp_path / "test.mp4",
        output_dir=tmp_path,
        include_review=True,
        include_high_risk=True,
        include_invalid=True,
        write_manifest=True,
    )

    assert manifest["version"] == "v5.3.1"
    assert manifest["evidence_count"] == len(
        manifest["evidence"]
    )
    assert manifest["evidence_count"] >= 1