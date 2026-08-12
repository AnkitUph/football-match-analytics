from types import SimpleNamespace

import pytest

from ai_engine.validation.identity_visual.identity_visual_validator import (
    IdentityVisualValidator,
)

def test_validate_manager():
    validator = IdentityVisualValidator()

    identity = fake_identity(
        identity_id="identity-1"
    )

    manager = SimpleNamespace(
        match_diagnostics=[
            diagnostic(
                identity_id="identity-1",
                appearance_similarity=0.90,
            ),
            diagnostic(
                identity_id="identity-1",
                track_id=20,
                appearance_similarity=0.75,
            ),
        ],
        identities={
            "identity-1": identity
        },
    )

    report = validator.validate_manager(manager)

    assert report["summary"]["candidates"] == 2
    assert report["summary"]["accepted_matches"] == 2

    assert (
        report["summary"]["appearance"]["strong"]
        == 1
    )

    assert (
        report["summary"]["appearance"]["weak"]
        == 1
    )

    assert (
        report["summary"]["identities"]["review"]
        == 1
    )

@pytest.fixture
def validator():
    return IdentityVisualValidator()


def diagnostic(
    *,
    identity_id="identity-1",
    track_id=10,
    matched=True,
    appearance_similarity=0.90,
    appearance_evaluated=True,
    temporal_gap=0,
    spatial_distance=0.0,
    motion_difference=0.0,
    team_match=True,
    jersey_match=True,
    score=0.90,
    reason="matched",
):
    return {
        "identity_id": identity_id,
        "track_id": track_id,
        "matched": matched,
        "appearance_similarity": appearance_similarity,
        "appearance_evaluated": appearance_evaluated,
        "temporal_gap": temporal_gap,
        "spatial_distance": spatial_distance,
        "motion_difference": motion_difference,
        "team_match": team_match,
        "jersey_match": jersey_match,
        "score": score,
        "reason": reason,
    }


# ============================================================================
# Configuration tests
# ============================================================================


def test_default_configuration():
    validator = IdentityVisualValidator()

    assert validator.minimum_appearance_similarity == 0.72
    assert validator.normal_appearance_similarity == 0.78
    assert validator.strong_appearance_similarity == 0.85


def test_invalid_appearance_threshold_range():
    with pytest.raises(ValueError):
        IdentityVisualValidator(
            minimum_appearance_similarity=1.5
        )


def test_invalid_appearance_threshold_order():
    with pytest.raises(ValueError):
        IdentityVisualValidator(
            minimum_appearance_similarity=0.85,
            normal_appearance_similarity=0.80,
            strong_appearance_similarity=0.90,
        )


def test_invalid_ratio():
    with pytest.raises(ValueError):
        IdentityVisualValidator(
            weak_spatial_ratio=1.5
        )


def test_invalid_temporal_gap():
    with pytest.raises(ValueError):
        IdentityVisualValidator(
            max_temporal_gap=-1
        )


def test_invalid_spatial_distance():
    with pytest.raises(ValueError):
        IdentityVisualValidator(
            max_spatial_distance=0
        )


def test_invalid_motion_difference():
    with pytest.raises(ValueError):
        IdentityVisualValidator(
            max_motion_difference=0
        )


# ============================================================================
# Appearance classification
# ============================================================================


@pytest.mark.parametrize(
    "similarity,expected",
    [
        (-1.0, IdentityVisualValidator.APPEARANCE_MISSING),
        (0.50, IdentityVisualValidator.APPEARANCE_MISMATCH),
        (0.72, IdentityVisualValidator.APPEARANCE_WEAK),
        (0.75, IdentityVisualValidator.APPEARANCE_WEAK),
        (0.78, IdentityVisualValidator.APPEARANCE_NORMAL),
        (0.82, IdentityVisualValidator.APPEARANCE_NORMAL),
        (0.85, IdentityVisualValidator.APPEARANCE_STRONG),
        (0.95, IdentityVisualValidator.APPEARANCE_STRONG),
    ],
)
def test_appearance_categories(
    validator,
    similarity,
    expected,
):
    assert (
        validator._appearance_category(
            similarity,
            True,
        )
        == expected
    )


def test_not_evaluated_is_distinct_from_missing(validator):
    assert (
        validator._appearance_category(
            -1.0,
            False,
        )
        == validator.APPEARANCE_NOT_EVALUATED
    )


# ============================================================================
# Spatial / motion normalization
# ============================================================================


def test_spatial_ratio():
    assert validator_value(
        IdentityVisualValidator()._spatial_ratio(0)
    ) == 0.0

    assert (
        IdentityVisualValidator()
        ._spatial_ratio(150)
        == pytest.approx(0.5)
    )

    assert (
        IdentityVisualValidator()
        ._spatial_ratio(600)
        == 1.0
    )


def test_negative_spatial_distance_is_max_risk_ratio():
    validator = IdentityVisualValidator()

    assert validator._spatial_ratio(-1) == 1.0


def test_motion_ratio():
    validator = IdentityVisualValidator()

    assert validator._motion_ratio(0) == 0.0

    assert validator._motion_ratio(40) == pytest.approx(
        0.5
    )

    assert validator._motion_ratio(100) == 1.0


def test_negative_motion_difference_is_max_risk_ratio():
    validator = IdentityVisualValidator()

    assert validator._motion_ratio(-1) == 1.0


# ============================================================================
# Basic accepted matches
# ============================================================================


def test_strong_appearance_good_geometry_is_low_risk(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                appearance_similarity=0.90,
                spatial_distance=30,
                motion_difference=10,
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_LOW
    assert result.status == validator.STATUS_PASS
    assert result.matched is True
    assert result.validation_warnings == []


def test_normal_appearance_good_geometry_is_low_risk(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                appearance_similarity=0.80,
                spatial_distance=30,
                motion_difference=10,
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_LOW
    assert result.status == validator.STATUS_PASS


def test_weak_appearance_is_review(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                appearance_similarity=0.75,
                spatial_distance=30,
                motion_difference=10,
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_MEDIUM
    assert result.status == validator.STATUS_REVIEW

    assert (
        "accepted_weak_appearance"
        in result.validation_warnings
    )


def test_weak_appearance_with_poor_spatial_geometry_is_high_risk(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                appearance_similarity=0.75,
                spatial_distance=100,
                motion_difference=10,
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_HIGH
    assert result.status == validator.STATUS_HIGH_RISK

    assert (
        "weak_appearance_spatial_uncertainty"
        in result.validation_warnings
    )


def test_weak_appearance_with_poor_motion_is_high_risk(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                appearance_similarity=0.75,
                spatial_distance=10,
                motion_difference=30,
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_HIGH
    assert result.status == validator.STATUS_HIGH_RISK

    assert (
        "weak_appearance_motion_uncertainty"
        in result.validation_warnings
    )


# ============================================================================
# Appearance mismatch
# ============================================================================


def test_appearance_mismatch_is_critical(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                appearance_similarity=0.60
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_CRITICAL
    assert result.status == validator.STATUS_INVALID


# ============================================================================
# Missing / unavailable appearance
# ============================================================================


def test_accepted_without_appearance_is_review(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                appearance_evaluated=False,
                appearance_similarity=-1.0,
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_MEDIUM
    assert result.status == validator.STATUS_REVIEW

    assert (
        "accepted_without_appearance"
        in result.validation_warnings
    )


def test_missing_embedding_is_review(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                appearance_evaluated=True,
                appearance_similarity=-1.0,
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_MEDIUM
    assert result.status == validator.STATUS_REVIEW

    assert (
        "appearance_evaluated_but_unavailable"
        in result.validation_warnings
    )


def test_appearance_similarity_without_evaluation_is_critical(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                appearance_evaluated=False,
                appearance_similarity=0.85,
                matched=False,
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_CRITICAL
    assert result.status == validator.STATUS_INVALID

    assert (
        "appearance_similarity_without_evaluation"
        in result.validation_warnings
    )


# ============================================================================
# Team / jersey mismatch
# ============================================================================


def test_accepted_team_mismatch_is_critical(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                team_match=False
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_CRITICAL
    assert result.status == validator.STATUS_INVALID

    assert (
        "accepted_team_mismatch"
        in result.validation_warnings
    )


def test_accepted_jersey_mismatch_is_critical(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                jersey_match=False
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_CRITICAL
    assert result.status == validator.STATUS_INVALID

    assert (
        "accepted_jersey_mismatch"
        in result.validation_warnings
    )


def test_accepted_team_and_jersey_mismatch_is_critical(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                team_match=False,
                jersey_match=False,
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_CRITICAL

    assert (
        "accepted_team_mismatch"
        in result.validation_warnings
    )

    assert (
        "accepted_jersey_mismatch"
        in result.validation_warnings
    )


# ============================================================================
# Temporal gap
# ============================================================================


def test_long_gap_with_strong_appearance_is_review(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                appearance_similarity=0.90,
                temporal_gap=40,
                spatial_distance=20,
                motion_difference=10,
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_MEDIUM
    assert result.status == validator.STATUS_REVIEW

    assert (
        "accepted_long_temporal_gap"
        in result.validation_warnings
    )


def test_long_gap_with_weak_appearance_is_high_risk(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                appearance_similarity=0.75,
                temporal_gap=40,
                spatial_distance=20,
                motion_difference=10,
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_HIGH
    assert result.status == validator.STATUS_HIGH_RISK


def test_long_gap_with_normal_appearance_and_bad_geometry_is_high_risk(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                appearance_similarity=0.80,
                temporal_gap=40,
                spatial_distance=250,
                motion_difference=70,
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_HIGH
    assert result.status == validator.STATUS_HIGH_RISK


# ============================================================================
# Rejected matches
# ============================================================================


def test_rejected_match_is_low_risk(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                matched=False,
                appearance_similarity=0.60,
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_LOW
    assert result.status == validator.STATUS_PASS


def test_rejected_team_mismatch_is_not_critical(
    validator,
):
    result = validator.validate_diagnostics(
        [
            diagnostic(
                matched=False,
                team_match=False,
            )
        ]
    )[0]

    assert result.risk_level == validator.RISK_LOW


# ============================================================================
# Identity-level validation
# ============================================================================


def fake_identity(
    identity_id="identity-1",
    source_track_ids=None,
    observations=None,
    first_frame=0,
    last_frame=100,
    class_name="player",
    team="home",
    jersey_number=10,
):
    return SimpleNamespace(
        identity_id=identity_id,
        source_track_ids=source_track_ids or [1, 2],
        observations=observations or [object()],
        first_frame=first_frame,
        last_frame=last_frame,
        class_name=class_name,
        team=team,
        jersey_number=jersey_number,
    )


def test_identity_with_valid_structure_is_low_risk(
    validator,
):
    identity = fake_identity()

    result = validator.validate_identity(identity)

    assert result.risk_level == validator.RISK_LOW
    assert result.status == validator.STATUS_PASS


def test_identity_without_source_tracks_is_critical(
    validator,
):
    identity = fake_identity(
        source_track_ids=[],
    )

    result = validator.validate_identity(identity)

    assert result.risk_level == validator.RISK_CRITICAL
    assert result.status == validator.STATUS_INVALID

    assert (
        "identity_has_no_source_tracks"
        in result.reasons
    )


def test_identity_without_observations_is_critical(
    validator,
):
    identity = fake_identity(
        observations=[],
    )

    result = validator.validate_identity(identity)

    assert result.risk_level == validator.RISK_CRITICAL
    assert result.status == validator.STATUS_INVALID

    assert (
        "identity_has_no_observations"
        in result.reasons
    )


def test_identity_with_invalid_frame_range_is_critical(
    validator,
):
    identity = fake_identity(
        first_frame=100,
        last_frame=50,
    )

    result = validator.validate_identity(identity)

    assert result.risk_level == validator.RISK_CRITICAL
    assert result.status == validator.STATUS_INVALID

    assert (
        "identity_frame_range_invalid"
        in result.reasons
    )


def test_identity_counts_weak_matches(
    validator,
):
    validator.validate_diagnostics(
        [
            diagnostic(
                identity_id="identity-1",
                appearance_similarity=0.75,
            ),
            diagnostic(
                identity_id="identity-1",
                track_id=11,
                appearance_similarity=0.90,
            ),
        ]
    )

    identity = fake_identity()

    result = validator.validate_identity(identity)

    assert result.accepted_matches == 2
    assert result.weak_appearance_matches == 1
    assert result.strong_appearance_matches == 1


def test_identity_counts_missing_embeddings(
    validator,
):
    validator.validate_diagnostics(
        [
            diagnostic(
                identity_id="identity-1",
                appearance_evaluated=True,
                appearance_similarity=-1.0,
            )
        ]
    )

    identity = fake_identity()

    result = validator.validate_identity(identity)

    assert result.missing_embeddings == 1
    assert (
        "contains_missing_embeddings"
        in result.reasons
    )


def test_identity_with_high_risk_match_becomes_high_risk(
    validator,
):
    validator.validate_diagnostics(
        [
            diagnostic(
                identity_id="identity-1",
                appearance_similarity=0.75,
                spatial_distance=100,
            )
        ]
    )

    identity = fake_identity()

    result = validator.validate_identity(identity)

    assert result.risky_matches == 1
    assert result.risk_level == validator.RISK_HIGH
    assert result.status == validator.STATUS_HIGH_RISK


# ============================================================================
# Summary / report
# ============================================================================


def test_summary_counts_matches(
    validator,
):
    validator.validate_diagnostics(
        [
            diagnostic(
                identity_id="identity-1",
                appearance_similarity=0.90,
            ),
            diagnostic(
                identity_id="identity-2",
                matched=False,
                appearance_similarity=0.60,
            ),
            diagnostic(
                identity_id="identity-3",
                appearance_similarity=0.75,
            ),
        ]
    )

    summary = validator.get_summary()

    assert summary["candidates"] == 3
    assert summary["accepted_matches"] == 2
    assert summary["rejected_matches"] == 1

    assert summary["appearance"]["strong"] == 1
    assert summary["appearance"]["weak"] == 1


def test_report_contains_expected_sections(
    validator,
):
    validator.validate_diagnostics(
        [
            diagnostic()
        ]
    )

    report = validator.get_report()

    assert report["version"] == "v5.2"
    assert "status" in report
    assert "summary" in report
    assert "recommendation" in report
    assert "recommendations" in report
    assert "matches" in report
    assert "identities" in report


def test_empty_validator_report(
    validator,
):
    report = validator.get_report()

    assert report["status"] == validator.STATUS_PASS
    assert report["summary"]["candidates"] == 0

    assert "NO MATCHES" in report["recommendation"]


# ============================================================================
# Read-only behavior
# ============================================================================


def test_validate_manager_does_not_mutate_manager(
    validator,
):
    identity = fake_identity()

    manager = SimpleNamespace(
        match_diagnostics=[
            diagnostic(
                identity_id="identity-1"
            )
        ],
        identities={
            "identity-1": identity
        },
    )

    original_diagnostics = list(
        manager.match_diagnostics
    )

    original_identities = dict(
        manager.identities
    )

    validator.validate_manager(manager)

    assert (
        manager.match_diagnostics
        == original_diagnostics
    )

    assert (
        manager.identities
        == original_identities
    )


# ============================================================================
# Helpers
# ============================================================================

def test_spatial_ratio():
    validator = IdentityVisualValidator()

    assert validator._spatial_ratio(0) == 0.0

    assert validator._spatial_ratio(150) == pytest.approx(0.5)

    assert validator._spatial_ratio(600) == 1.0