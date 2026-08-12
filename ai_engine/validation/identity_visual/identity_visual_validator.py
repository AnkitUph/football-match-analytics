"""
Identity Visual Validation V5.4.

Read-only validation layer for GlobalIdentityManager V5.x.

Design principles
-----------------

1. Matcher diagnostic reasons are informational only.
2. Only validator-generated warnings on ACCEPTED matches affect
   identity risk.
3. Rejected matcher candidates do not become identity failures.
4. Appearance-not-evaluated is distinct from missing embeddings.
5. Geometry-only accepted matches are REVIEW, not automatically HIGH_RISK.
6. Weak appearance + poor geometry/motion can become HIGH_RISK.
7. Appearance mismatch on an accepted match is CRITICAL.
8. Team mismatch on an accepted match is CRITICAL.
9. Jersey mismatch on an accepted match is CRITICAL.
10. Normal matcher rejection such as:
       - track_overlaps_identity
       - temporal_gap_too_large
       - spatial_distance_too_large
       - motion_difference_too_large
    does not make the identity validation INVALID.
11. The validator never mutates manager, Identity, Track,
    or TrackObservation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


from ai_engine.schemas.track import Identity, Track


logger = logging.getLogger(__name__)


# ============================================================================
# Result objects
# ============================================================================


@dataclass
class VisualValidationResult:
    """Validation result for one identity matching diagnostic."""

    identity_id: str
    track_id: Optional[int] = None

    status: str = "PASS"
    risk_level: str = "LOW"
    reason: str = ""

    appearance_similarity: float = -1.0
    appearance_evaluated: bool = False

    temporal_gap: int = 0
    spatial_distance: float = 0.0
    motion_difference: float = 0.0

    team_match: Optional[bool] = None
    jersey_match: Optional[bool] = None

    score: float = 0.0
    matched: bool = False

    reasons: list[str] = field(default_factory=list)

    # Matcher reasons are deliberately separated from validation warnings.
    matcher_reasons: list[str] = field(default_factory=list)

    # Only these warnings affect accepted-match risk.
    validation_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "track_id": self.track_id,
            "status": self.status,
            "risk_level": self.risk_level,
            "reason": self.reason,
            "appearance_similarity": float(
                self.appearance_similarity
            ),
            "appearance_evaluated": bool(
                self.appearance_evaluated
            ),
            "temporal_gap": int(self.temporal_gap),
            "spatial_distance": float(
                self.spatial_distance
            ),
            "motion_difference": float(
                self.motion_difference
            ),
            "team_match": self.team_match,
            "jersey_match": self.jersey_match,
            "score": float(self.score),
            "matched": bool(self.matched),
            "reasons": list(self.reasons),
            "matcher_reasons": list(
                self.matcher_reasons
            ),
            "validation_warnings": list(
                self.validation_warnings
            ),
        }


@dataclass
class IdentityValidationResult:
    """Validation result for one global identity."""

    identity_id: str
    class_name: Optional[str] = None

    source_track_count: int = 0
    source_track_ids: list[Any] = field(default_factory=list)

    observation_count: int = 0

    first_frame: Optional[int] = None
    last_frame: Optional[int] = None

    team: Any = None
    jersey_number: Any = None

    status: str = "PASS"
    risk_level: str = "LOW"

    reasons: list[str] = field(default_factory=list)

    accepted_matches: int = 0
    rejected_matches: int = 0

    appearance_matches: int = 0
    strong_appearance_matches: int = 0
    weak_appearance_matches: int = 0
    mismatch_appearance_matches: int = 0

    missing_embeddings: int = 0

    risky_matches: int = 0
    suspicious_matches: int = 0

    validation_warning_count: int = 0
    matcher_diagnostic_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "class_name": self.class_name,
            "source_track_count": self.source_track_count,
            "source_track_ids": list(
                self.source_track_ids
            ),
            "observation_count": self.observation_count,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "team": self.team,
            "jersey_number": self.jersey_number,
            "status": self.status,
            "risk_level": self.risk_level,
            "reasons": list(self.reasons),
            "accepted_matches": self.accepted_matches,
            "rejected_matches": self.rejected_matches,
            "appearance_matches": self.appearance_matches,
            "strong_appearance_matches": (
                self.strong_appearance_matches
            ),
            "weak_appearance_matches": (
                self.weak_appearance_matches
            ),
            "mismatch_appearance_matches": (
                self.mismatch_appearance_matches
            ),
            "missing_embeddings": (
                self.missing_embeddings
            ),
            "risky_matches": self.risky_matches,
            "suspicious_matches": (
                self.suspicious_matches
            ),
            "validation_warning_count": (
                self.validation_warning_count
            ),
            "matcher_diagnostic_count": (
                self.matcher_diagnostic_count
            ),
        }


# ============================================================================
# Validator
# ============================================================================


class IdentityVisualValidator:
    """
    Read-only visual validator for GlobalIdentityManager V5.x.

    Important:
        Rejected matcher candidates are not identity assignments.

    Therefore:
        rejected candidate != invalid identity
    """

    VERSION = "v5.4"

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    STATUS_PASS = "PASS"
    STATUS_REVIEW = "REVIEW"
    STATUS_HIGH_RISK = "HIGH_RISK"
    STATUS_INVALID = "INVALID"

    # ------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------

    RISK_LOW = "LOW"
    RISK_MEDIUM = "MEDIUM"
    RISK_HIGH = "HIGH"
    RISK_CRITICAL = "CRITICAL"

    # ------------------------------------------------------------------
    # Appearance
    # ------------------------------------------------------------------

    APPEARANCE_NOT_EVALUATED = "not_evaluated"
    APPEARANCE_MISSING = "missing"
    APPEARANCE_MISMATCH = "mismatch"
    APPEARANCE_WEAK = "weak"
    APPEARANCE_NORMAL = "normal"
    APPEARANCE_STRONG = "strong"

    # ------------------------------------------------------------------
    # Matcher rejection reasons
    # ------------------------------------------------------------------

    NON_FAILURE_REJECTION_REASONS = {
        "track_overlaps_identity",
        "temporal_gap_too_large",
        "spatial_distance_too_large",
        "motion_difference_too_large",
        "score_below_threshold",
        "strong_appearance_geometry_too_weak",
        "normal_appearance_geometry_too_weak",
        "weak_appearance_geometry",
        "long_gap_motion_uncertainty",
        "long_gap_spatial_uncertainty",
        "long_gap_appearance_uncertainty",
    }

    # ==================================================================
    # Initialization
    # ==================================================================

    def __init__(
        self,
        minimum_appearance_similarity: float = 0.72,
        normal_appearance_similarity: float = 0.78,
        strong_appearance_similarity: float = 0.85,
        weak_spatial_ratio: float = 0.20,
        weak_motion_ratio: float = 0.25,
        strong_spatial_ratio: float = 0.35,
        strong_motion_ratio: float = 0.40,
        long_gap_ratio: float = 0.50,
        max_temporal_gap: int = 75,
        max_spatial_distance: float = 300.0,
        max_motion_difference: float = 80.0,
    ) -> None:

        self._validate_configuration(
            minimum_appearance_similarity,
            normal_appearance_similarity,
            strong_appearance_similarity,
            weak_spatial_ratio,
            weak_motion_ratio,
            strong_spatial_ratio,
            strong_motion_ratio,
            long_gap_ratio,
            max_temporal_gap,
            max_spatial_distance,
            max_motion_difference,
        )

        self.minimum_appearance_similarity = float(
            minimum_appearance_similarity
        )

        self.normal_appearance_similarity = float(
            normal_appearance_similarity
        )

        self.strong_appearance_similarity = float(
            strong_appearance_similarity
        )

        self.weak_spatial_ratio = float(
            weak_spatial_ratio
        )

        self.weak_motion_ratio = float(
            weak_motion_ratio
        )

        self.strong_spatial_ratio = float(
            strong_spatial_ratio
        )

        self.strong_motion_ratio = float(
            strong_motion_ratio
        )

        self.long_gap_ratio = float(
            long_gap_ratio
        )

        self.max_temporal_gap = int(
            max_temporal_gap
        )

        self.max_spatial_distance = float(
            max_spatial_distance
        )

        self.max_motion_difference = float(
            max_motion_difference
        )

        self.match_results: list[
            VisualValidationResult
        ] = []

        self.identity_results: dict[
            str,
            IdentityValidationResult,
        ] = {}

        self.reset()

    # ==================================================================
    # Configuration
    # ==================================================================

    @staticmethod
    def _validate_configuration(
        minimum_appearance_similarity: float,
        normal_appearance_similarity: float,
        strong_appearance_similarity: float,
        weak_spatial_ratio: float,
        weak_motion_ratio: float,
        strong_spatial_ratio: float,
        strong_motion_ratio: float,
        long_gap_ratio: float,
        max_temporal_gap: int,
        max_spatial_distance: float,
        max_motion_difference: float,
    ) -> None:

        appearance = (
            minimum_appearance_similarity,
            normal_appearance_similarity,
            strong_appearance_similarity,
        )

        if any(
            not 0.0 <= value <= 1.0
            for value in appearance
        ):
            raise ValueError(
                "Appearance thresholds must be between 0 and 1."
            )

        if not (
            minimum_appearance_similarity
            <= normal_appearance_similarity
            <= strong_appearance_similarity
        ):
            raise ValueError(
                "Appearance thresholds must satisfy "
                "minimum <= normal <= strong."
            )

        ratios = (
            weak_spatial_ratio,
            weak_motion_ratio,
            strong_spatial_ratio,
            strong_motion_ratio,
            long_gap_ratio,
        )

        if any(
            not 0.0 <= value <= 1.0
            for value in ratios
        ):
            raise ValueError(
                "Validation ratios must be between 0 and 1."
            )

        if max_temporal_gap < 0:
            raise ValueError(
                "max_temporal_gap must be >= 0."
            )

        if max_spatial_distance <= 0:
            raise ValueError(
                "max_spatial_distance must be > 0."
            )

        if max_motion_difference <= 0:
            raise ValueError(
                "max_motion_difference must be > 0."
            )

    # ==================================================================
    # Public API
    # ==================================================================

    def reset(self) -> None:
        """Clear validation results."""

        self.match_results = []
        self.identity_results = {}

    def validate_manager(
        self,
        manager: Any,
    ) -> dict[str, Any]:
        """
        Validate a GlobalIdentityManager.

        This function never modifies the manager.
        """

        self.reset()

        diagnostics = list(
            getattr(
                manager,
                "match_diagnostics",
                [],
            )
            or []
        )

        identities = (
            getattr(
                manager,
                "identities",
                {},
            )
            or {}
        )

        self.validate_diagnostics(
            diagnostics
        )

        for identity_id, identity in identities.items():
            self._validate_identity(
                str(identity_id),
                identity,
            )

        return self.get_report()

    def validate_diagnostics(
        self,
        diagnostics: Iterable[
            dict[str, Any]
        ],
    ) -> list[
        VisualValidationResult
    ]:

        results = []

        for diagnostic in diagnostics:

            result = self._validate_diagnostic(
                diagnostic
            )

            self.match_results.append(
                result
            )

            results.append(result)

        return results

    def validate_identity(
        self,
        identity: Identity,
    ) -> IdentityValidationResult:

        return self._validate_identity(
            str(identity.identity_id),
            identity,
        )

    # ==================================================================
    # Diagnostic validation
    # ==================================================================

    def _validate_diagnostic(
        self,
        diagnostic: dict[str, Any],
    ) -> VisualValidationResult:

        identity_id = str(
            diagnostic.get(
                "identity_id",
                "UNKNOWN",
            )
        )

        track_id = diagnostic.get(
            "track_id"
        )

        matched = bool(
            diagnostic.get(
                "matched",
                False,
            )
        )

        score = self._safe_float(
            diagnostic.get(
                "score",
                0.0,
            )
        )

        temporal_gap = self._safe_int(
            diagnostic.get(
                "temporal_gap",
                0,
            )
        )

        spatial_distance = self._safe_float(
            diagnostic.get(
                "spatial_distance",
                0.0,
            )
        )

        motion_difference = self._safe_float(
            diagnostic.get(
                "motion_difference",
                0.0,
            )
        )

        appearance_similarity = self._safe_float(
            diagnostic.get(
                "appearance_similarity",
                -1.0,
            ),
            default=-1.0,
        )

        appearance_evaluated = bool(
            diagnostic.get(
                "appearance_evaluated",
                False,
            )
        )

        team_match = diagnostic.get(
            "team_match"
        )

        jersey_match = diagnostic.get(
            "jersey_match"
        )

        matcher_reason = str(
            diagnostic.get(
                "reason",
                "",
            )
            or ""
        )

        matcher_reasons = (
            [matcher_reason]
            if matcher_reason
            else []
        )

        # ==============================================================
        # IMPORTANT:
        #
        # Validation warnings are generated differently for accepted
        # and rejected candidates.
        #
        # Rejected candidates do NOT become identity failures.
        # ==============================================================

        validation_warnings = (
            self._collect_validation_warnings(
                matched=matched,
                appearance_similarity=(
                    appearance_similarity
                ),
                appearance_evaluated=(
                    appearance_evaluated
                ),
                temporal_gap=temporal_gap,
                spatial_distance=(
                    spatial_distance
                ),
                motion_difference=(
                    motion_difference
                ),
                team_match=team_match,
                jersey_match=jersey_match,
            )
        )

        appearance_category = (
            self._appearance_category(
                appearance_similarity,
                appearance_evaluated,
            )
        )

        spatial_ratio = (
            self._spatial_ratio(
                spatial_distance
            )
        )

        motion_ratio = (
            self._motion_ratio(
                motion_difference
            )
        )

        risk_level = self._calculate_risk(
            matched=matched,
            appearance_category=(
                appearance_category
            ),
            appearance_similarity=(
                appearance_similarity
            ),
            temporal_gap=temporal_gap,
            spatial_ratio=spatial_ratio,
            motion_ratio=motion_ratio,
            team_match=team_match,
            jersey_match=jersey_match,
            reasons=validation_warnings,
        )

        reasons = list(
            validation_warnings
        )

        reasons.extend(
            [
                f"matcher:{reason}"
                for reason in matcher_reasons
            ]
        )

        return VisualValidationResult(
            identity_id=identity_id,
            track_id=track_id,
            status=self._status_from_risk(
                risk_level
            ),
            risk_level=risk_level,
            reason=self._primary_reason(
                validation_warnings,
                matched,
                matcher_reasons,
            ),
            appearance_similarity=(
                appearance_similarity
            ),
            appearance_evaluated=(
                appearance_evaluated
            ),
            temporal_gap=temporal_gap,
            spatial_distance=(
                spatial_distance
            ),
            motion_difference=(
                motion_difference
            ),
            team_match=team_match,
            jersey_match=jersey_match,
            score=score,
            matched=matched,
            reasons=reasons,
            matcher_reasons=matcher_reasons,
            validation_warnings=(
                validation_warnings
            ),
        )

    # ==================================================================
    # Validation warnings
    # ==================================================================

    def _collect_validation_warnings(
        self,
        matched: bool,
        appearance_similarity: float,
        appearance_evaluated: bool,
        temporal_gap: int,
        spatial_distance: float,
        motion_difference: float,
        team_match: Optional[bool],
        jersey_match: Optional[bool],
    ) -> list[str]:

        warnings: list[str] = []

        # ==============================================================
        # REJECTED CANDIDATE
        #
        # Do NOT generate identity-risk warnings here.
        #
        # A rejected candidate is not an identity assignment.
        # ==============================================================

        if not matched:

            # If appearance metadata is internally contradictory,
            # record it only as a diagnostic warning.
            #
            # It does NOT make the candidate critical.
            if (
                appearance_evaluated
                and appearance_similarity < 0.0
            ):
                warnings.append(
                    "appearance_evaluated_but_unavailable"
                )

            if (
                not appearance_evaluated
                and appearance_similarity >= 0.0
            ):
                warnings.append(
                    "appearance_similarity_without_evaluation"
                )

            return warnings

        # ==============================================================
        # ACCEPTED MATCH
        # ==============================================================

        category = self._appearance_category(
            appearance_similarity,
            appearance_evaluated,
        )

        # --------------------------------------------------------------
        # Appearance consistency
        # --------------------------------------------------------------

        if (
            appearance_evaluated
            and appearance_similarity < 0.0
        ):
            warnings.append(
                "appearance_evaluated_but_unavailable"
            )

        if (
            not appearance_evaluated
            and appearance_similarity >= 0.0
        ):
            warnings.append(
                "appearance_similarity_without_evaluation"
            )

        # --------------------------------------------------------------
        # Weak appearance
        # --------------------------------------------------------------

        if category == self.APPEARANCE_WEAK:
            warnings.append(
                "accepted_weak_appearance"
            )

        # --------------------------------------------------------------
        # Missing appearance
        # --------------------------------------------------------------

        if category == self.APPEARANCE_MISSING:
            warnings.append(
                "accepted_without_appearance"
            )

        # --------------------------------------------------------------
        # Appearance mismatch
        # --------------------------------------------------------------

        if category == self.APPEARANCE_MISMATCH:
            warnings.append(
                "accepted_appearance_mismatch"
            )

        spatial_ratio = (
            self._spatial_ratio(
                spatial_distance
            )
        )

        motion_ratio = (
            self._motion_ratio(
                motion_difference
            )
        )

        # --------------------------------------------------------------
        # Weak appearance + geometry
        # --------------------------------------------------------------

        if (
            category == self.APPEARANCE_WEAK
            and spatial_ratio
            > self.weak_spatial_ratio
        ):
            warnings.append(
                "weak_appearance_spatial_uncertainty"
            )

        if (
            category == self.APPEARANCE_WEAK
            and motion_ratio
            > self.weak_motion_ratio
        ):
            warnings.append(
                "weak_appearance_motion_uncertainty"
            )

        # --------------------------------------------------------------
        # Normal appearance + poor geometry
        # --------------------------------------------------------------

        if (
            category == self.APPEARANCE_NORMAL
            and spatial_ratio > 0.70
        ):
            warnings.append(
                "normal_appearance_high_spatial_ratio"
            )

        if (
            category == self.APPEARANCE_NORMAL
            and motion_ratio > 0.70
        ):
            warnings.append(
                "normal_appearance_high_motion_ratio"
            )

        # --------------------------------------------------------------
        # Strong appearance + extreme geometry
        # --------------------------------------------------------------

        if (
            category == self.APPEARANCE_STRONG
            and spatial_ratio
            > 0.85
        ):
            warnings.append(
                "strong_appearance_extreme_spatial_ratio"
            )

        if (
            category == self.APPEARANCE_STRONG
            and motion_ratio
            > 0.90
        ):
            warnings.append(
                "strong_appearance_extreme_motion_ratio"
            )

        # --------------------------------------------------------------
        # Temporal gap
        # --------------------------------------------------------------

        if self._is_long_gap(
            temporal_gap
        ):
            warnings.append(
                "accepted_long_temporal_gap"
            )

        # --------------------------------------------------------------
        # Team mismatch
        # --------------------------------------------------------------

        if team_match is False:
            warnings.append(
                "accepted_team_mismatch"
            )

        # --------------------------------------------------------------
        # Jersey mismatch
        # --------------------------------------------------------------

        if jersey_match is False:
            warnings.append(
                "accepted_jersey_mismatch"
            )

        return warnings

    # ==================================================================
    # Identity validation
    # ==================================================================

    def _validate_identity(
        self,
        identity_id: str,
        identity: Identity,
    ) -> IdentityValidationResult:

        source_track_ids = list(
            getattr(
                identity,
                "source_track_ids",
                [],
            )
            or []
        )

        observations = list(
            getattr(
                identity,
                "observations",
                [],
            )
            or []
        )

        result = IdentityValidationResult(
            identity_id=str(
                identity_id
            ),
            class_name=getattr(
                identity,
                "class_name",
                None,
            ),
            source_track_count=len(
                source_track_ids
            ),
            source_track_ids=(
                source_track_ids
            ),
            observation_count=len(
                observations
            ),
            first_frame=getattr(
                identity,
                "first_frame",
                None,
            ),
            last_frame=getattr(
                identity,
                "last_frame",
                None,
            ),
            team=getattr(
                identity,
                "team",
                None,
            ),
            jersey_number=getattr(
                identity,
                "jersey_number",
                None,
            ),
        )

        # --------------------------------------------------------------
        # Structural validation
        # --------------------------------------------------------------

        if not source_track_ids:
            result.reasons.append(
                "identity_has_no_source_tracks"
            )

        if not observations:
            result.reasons.append(
                "identity_has_no_observations"
            )

        if (
            result.first_frame is not None
            and result.last_frame is not None
            and result.last_frame
            < result.first_frame
        ):
            result.reasons.append(
                "identity_frame_range_invalid"
            )

        # --------------------------------------------------------------
        # Only diagnostics belonging to this identity
        # --------------------------------------------------------------

        identity_matches = [
            match
            for match in self.match_results
            if match.identity_id
            == str(identity_id)
        ]

        result.accepted_matches = sum(
            1
            for match in identity_matches
            if match.matched
        )

        result.rejected_matches = sum(
            1
            for match in identity_matches
            if not match.matched
        )

        result.matcher_diagnostic_count = (
            len(identity_matches)
        )

        # --------------------------------------------------------------
        # Appearance statistics
        # --------------------------------------------------------------

        result.appearance_matches = sum(
            1
            for match in identity_matches
            if (
                match.matched
                and match.appearance_evaluated
                and match.appearance_similarity
                >= self.minimum_appearance_similarity
            )
        )

        result.strong_appearance_matches = sum(
            1
            for match in identity_matches
            if (
                match.matched
                and match.appearance_evaluated
                and match.appearance_similarity
                >= self.strong_appearance_similarity
            )
        )

        result.weak_appearance_matches = sum(
            1
            for match in identity_matches
            if (
                match.matched
                and match.appearance_evaluated
                and self.minimum_appearance_similarity
                <= match.appearance_similarity
                < self.normal_appearance_similarity
            )
        )

        result.mismatch_appearance_matches = sum(
            1
            for match in identity_matches
            if (
                match.matched
                and match.appearance_evaluated
                and match.appearance_similarity
                >= 0.0
                and match.appearance_similarity
                < self.minimum_appearance_similarity
            )
        )

        result.missing_embeddings = sum(
            1
            for match in identity_matches
            if (
                match.matched
                and match.appearance_evaluated
                and match.appearance_similarity
                < 0.0
            )
        )

        # --------------------------------------------------------------
        # Warnings
        # --------------------------------------------------------------

        result.validation_warning_count = sum(
            len(
                match.validation_warnings
            )
            for match in identity_matches
        )

        # IMPORTANT:
        # suspicious only counts ACCEPTED matches.
        result.suspicious_matches = sum(
            1
            for match in identity_matches
            if (
                match.matched
                and match.validation_warnings
            )
        )

        result.risky_matches = sum(
            1
            for match in identity_matches
            if (
                match.matched
                and match.risk_level
                in {
                    self.RISK_HIGH,
                    self.RISK_CRITICAL,
                }
            )
        )

        # --------------------------------------------------------------
        # Identity-level reasons
        # --------------------------------------------------------------

        if result.risky_matches:
            result.reasons.append(
                "contains_risky_matches"
            )

        if result.suspicious_matches:
            result.reasons.append(
                "contains_suspicious_matches"
            )

        if result.weak_appearance_matches:
            result.reasons.append(
                "contains_weak_appearance_matches"
            )

        if result.mismatch_appearance_matches:
            result.reasons.append(
                "contains_appearance_mismatch"
            )

        if result.missing_embeddings:
            result.reasons.append(
                "contains_missing_embeddings"
            )

        # --------------------------------------------------------------
        # Identity-level structural failures
        # --------------------------------------------------------------

        structural_failure = any(
            reason in result.reasons
            for reason in (
                "identity_has_no_source_tracks",
                "identity_has_no_observations",
                "identity_frame_range_invalid",
            )
        )

        # --------------------------------------------------------------
        # Identity-level risk
        # --------------------------------------------------------------

        if structural_failure:
            result.risk_level = (
                self.RISK_CRITICAL
            )

        elif any(
            match.risk_level
            == self.RISK_CRITICAL
            for match in identity_matches
            if match.matched
        ):
            result.risk_level = (
                self.RISK_CRITICAL
            )

        elif result.risky_matches:
            result.risk_level = (
                self.RISK_HIGH
            )

        elif (
            result.suspicious_matches
            or result.weak_appearance_matches
            or result.missing_embeddings
        ):
            result.risk_level = (
                self.RISK_MEDIUM
            )

        else:
            result.risk_level = (
                self.RISK_LOW
            )

        result.status = (
            self._status_from_risk(
                result.risk_level
            )
        )

        self.identity_results[
            str(identity_id)
        ] = result

        return result

    # ==================================================================
    # Appearance
    # ==================================================================

    def _appearance_category(
        self,
        similarity: float,
        evaluated: bool,
    ) -> str:

        if not evaluated:
            return (
                self.APPEARANCE_NOT_EVALUATED
            )

        if similarity < 0.0:
            return self.APPEARANCE_MISSING

        if (
            similarity
            < self.minimum_appearance_similarity
        ):
            return (
                self.APPEARANCE_MISMATCH
            )

        if (
            similarity
            < self.normal_appearance_similarity
        ):
            return self.APPEARANCE_WEAK

        if (
            similarity
            < self.strong_appearance_similarity
        ):
            return self.APPEARANCE_NORMAL

        return self.APPEARANCE_STRONG

    # ==================================================================
    # Geometry / motion
    # ==================================================================

    def _spatial_ratio(
        self,
        distance: float,
    ) -> float:

        if (
            distance < 0
            or distance == float("inf")
        ):
            return 1.0

        return min(
            1.0,
            max(
                0.0,
                distance
                / self.max_spatial_distance,
            ),
        )

    def _motion_ratio(
        self,
        difference: float,
    ) -> float:

        if (
            difference < 0
            or difference == float("inf")
        ):
            return 1.0

        return min(
            1.0,
            max(
                0.0,
                difference
                / self.max_motion_difference,
            ),
        )

    def _is_long_gap(
        self,
        temporal_gap: int,
    ) -> bool:

        return (
            temporal_gap
            >= self.max_temporal_gap
            * self.long_gap_ratio
        )

    # ==================================================================
    # Risk calculation
    # ==================================================================

    def _calculate_risk(
        self,
        matched: bool,
        appearance_category: str,
        appearance_similarity: float,
        temporal_gap: int,
        spatial_ratio: float,
        motion_ratio: float,
        team_match: Optional[bool],
        jersey_match: Optional[bool],
        reasons: list[str],
    ) -> str:

        # ==============================================================
        # REJECTED CANDIDATE
        #
        # This is the most important change from V5.3.
        #
        # A rejected candidate is NOT an invalid identity.
        # ==============================================================

        if not matched:
            return self.RISK_LOW

        # ==============================================================
        # Accepted match without appearance
        # ==============================================================

        if appearance_category in {
            self.APPEARANCE_NOT_EVALUATED,
            self.APPEARANCE_MISSING,
        }:
            return self.RISK_MEDIUM

        # ==============================================================
        # Appearance mismatch
        # ==============================================================

        if (
            appearance_category
            == self.APPEARANCE_MISMATCH
        ):
            return self.RISK_CRITICAL

        # ==============================================================
        # Team / jersey mismatch
        # ==============================================================

        if (
            team_match is False
            or jersey_match is False
        ):
            return self.RISK_CRITICAL

        # ==============================================================
        # Long temporal gap
        # ==============================================================

        if self._is_long_gap(
            temporal_gap
        ):

            if (
                appearance_similarity
                < self.normal_appearance_similarity
            ):
                return self.RISK_HIGH

            if (
                spatial_ratio > 0.70
                or motion_ratio > 0.70
            ):
                return self.RISK_HIGH

            return self.RISK_MEDIUM

        # ==============================================================
        # Weak appearance
        # ==============================================================

        if (
            appearance_category
            == self.APPEARANCE_WEAK
        ):

            if (
                spatial_ratio
                > self.weak_spatial_ratio
                or motion_ratio
                > self.weak_motion_ratio
            ):
                return self.RISK_HIGH

            return self.RISK_MEDIUM

        # ==============================================================
        # Normal appearance
        # ==============================================================

        if (
            appearance_category
            == self.APPEARANCE_NORMAL
        ):

            if (
                spatial_ratio > 0.70
                or motion_ratio > 0.70
            ):
                return self.RISK_HIGH

            return self.RISK_LOW

        # ==============================================================
        # Strong appearance
        # ==============================================================

        if (
            appearance_category
            == self.APPEARANCE_STRONG
        ):

            if (
                spatial_ratio
                > self.strong_spatial_ratio
                or motion_ratio
                > self.strong_motion_ratio
            ):
                return self.RISK_HIGH

            return self.RISK_LOW

        return self.RISK_MEDIUM

    # ==================================================================
    # Status
    # ==================================================================

    def _status_from_risk(
        self,
        risk_level: str,
    ) -> str:

        mapping = {
            self.RISK_LOW: self.STATUS_PASS,
            self.RISK_MEDIUM: self.STATUS_REVIEW,
            self.RISK_HIGH: self.STATUS_HIGH_RISK,
            self.RISK_CRITICAL: self.STATUS_INVALID,
        }

        return mapping.get(
            risk_level,
            self.STATUS_INVALID,
        )

    @staticmethod
    def _primary_reason(
        warnings: list[str],
        matched: bool,
        matcher_reasons: list[str],
    ) -> str:

        if warnings:
            return warnings[0]

        if matched:
            return "accepted_match"

        if matcher_reasons:
            return (
                f"matcher_rejected:"
                f"{matcher_reasons[0]}"
            )

        return "rejected_match"

    # ==================================================================
    # Reporting
    # ==================================================================

    def get_summary(
        self,
    ) -> dict[str, Any]:

        results = self.match_results

        accepted = [
            result
            for result in results
            if result.matched
        ]

        rejected = [
            result
            for result in results
            if not result.matched
        ]

        evaluated = [
            result
            for result in results
            if result.appearance_evaluated
        ]

        valid_appearance = [
            result
            for result in evaluated
            if result.appearance_similarity >= 0.0
        ]

        strong = [
            result
            for result in valid_appearance
            if result.appearance_similarity
            >= self.strong_appearance_similarity
        ]

        normal = [
            result
            for result in valid_appearance
            if (
                self.normal_appearance_similarity
                <= result.appearance_similarity
                < self.strong_appearance_similarity
            )
        ]

        weak = [
            result
            for result in valid_appearance
            if (
                self.minimum_appearance_similarity
                <= result.appearance_similarity
                < self.normal_appearance_similarity
            )
        ]

        mismatch = [
            result
            for result in valid_appearance
            if (
                result.appearance_similarity
                < self.minimum_appearance_similarity
            )
        ]

        missing = [
            result
            for result in evaluated
            if result.appearance_similarity < 0.0
        ]

        not_evaluated = [
            result
            for result in results
            if not result.appearance_evaluated
        ]

        similarities = [
            result.appearance_similarity
            for result in valid_appearance
        ]

        accepted_similarities = [
            result.appearance_similarity
            for result in accepted
            if (
                result.appearance_evaluated
                and result.appearance_similarity
                >= 0.0
            )
        ]

        risky = [
            result
            for result in accepted
            if result.risk_level
            in {
                self.RISK_HIGH,
                self.RISK_CRITICAL,
            }
        ]

        suspicious = [
            result
            for result in accepted
            if result.validation_warnings
        ]

        validation_warning_count = sum(
            len(
                result.validation_warnings
            )
            for result in results
            if result.matched
        )

        matcher_diagnostic_count = sum(
            len(
                result.matcher_reasons
            )
            for result in results
        )

        accepted_strong = [
            result
            for result in accepted
            if (
                result.appearance_evaluated
                and result.appearance_similarity
                >= self.strong_appearance_similarity
            )
        ]

        accepted_normal = [
            result
            for result in accepted
            if (
                result.appearance_evaluated
                and self.normal_appearance_similarity
                <= result.appearance_similarity
                < self.strong_appearance_similarity
            )
        ]

        accepted_weak = [
            result
            for result in accepted
            if (
                result.appearance_evaluated
                and self.minimum_appearance_similarity
                <= result.appearance_similarity
                < self.normal_appearance_similarity
            )
        ]

        accepted_mismatch = [
            result
            for result in accepted
            if (
                result.appearance_evaluated
                and result.appearance_similarity
                >= 0.0
                and result.appearance_similarity
                < self.minimum_appearance_similarity
            )
        ]

        return {
            "version": self.VERSION,

            "candidates": len(results),

            "accepted_matches": len(
                accepted
            ),

            "rejected_matches": len(
                rejected
            ),

            "appearance": {
                "evaluated": len(
                    evaluated
                ),

                "not_evaluated": len(
                    not_evaluated
                ),

                "valid_comparisons": len(
                    valid_appearance
                ),

                "missing_embeddings": len(
                    missing
                ),

                "mismatch": len(
                    mismatch
                ),

                "strong": len(
                    strong
                ),

                "normal": len(
                    normal
                ),

                "weak": len(
                    weak
                ),

                "average_similarity": (
                    self._average(
                        similarities
                    )
                ),

                "accepted_average_similarity": (
                    self._average(
                        accepted_similarities
                    )
                ),

                "accepted_strong": len(
                    accepted_strong
                ),

                "accepted_normal": len(
                    accepted_normal
                ),

                "accepted_weak": len(
                    accepted_weak
                ),

                "accepted_mismatch": len(
                    accepted_mismatch
                ),
            },

            "risk": {
                "risky_matches": len(
                    risky
                ),

                "suspicious_matches": len(
                    suspicious
                ),

                "validation_warning_count": (
                    validation_warning_count
                ),

                "matcher_diagnostic_count": (
                    matcher_diagnostic_count
                ),

                "high_risk": sum(
                    result.risk_level
                    == self.RISK_HIGH
                    for result in results
                ),

                "critical": sum(
                    result.risk_level
                    == self.RISK_CRITICAL
                    for result in results
                ),
            },

            "identities": {
                "total": len(
                    self.identity_results
                ),

                "pass": sum(
                    result.status
                    == self.STATUS_PASS
                    for result
                    in self.identity_results.values()
                ),

                "review": sum(
                    result.status
                    == self.STATUS_REVIEW
                    for result
                    in self.identity_results.values()
                ),

                "high_risk": sum(
                    result.status
                    == self.STATUS_HIGH_RISK
                    for result
                    in self.identity_results.values()
                ),

                "invalid": sum(
                    result.status
                    == self.STATUS_INVALID
                    for result
                    in self.identity_results.values()
                ),

                "risky_identities": sum(
                    result.risk_level
                    in {
                        self.RISK_HIGH,
                        self.RISK_CRITICAL,
                    }
                    for result
                    in self.identity_results.values()
                ),

                "suspicious_identities": sum(
                    result.suspicious_matches > 0
                    for result
                    in self.identity_results.values()
                ),
            },

            "thresholds": {
                "minimum_appearance_similarity":
                    self.minimum_appearance_similarity,

                "normal_appearance_similarity":
                    self.normal_appearance_similarity,

                "strong_appearance_similarity":
                    self.strong_appearance_similarity,

                "weak_spatial_ratio":
                    self.weak_spatial_ratio,

                "weak_motion_ratio":
                    self.weak_motion_ratio,

                "strong_spatial_ratio":
                    self.strong_spatial_ratio,

                "strong_motion_ratio":
                    self.strong_motion_ratio,

                "long_gap_ratio":
                    self.long_gap_ratio,

                "max_temporal_gap":
                    self.max_temporal_gap,

                "max_spatial_distance":
                    self.max_spatial_distance,

                "max_motion_difference":
                    self.max_motion_difference,
            },
        }

    def get_report(
        self,
    ) -> dict[str, Any]:

        return {
            "version": self.VERSION,

            "status": self._overall_status(),

            "summary": self.get_summary(),

            "recommendation": (
                self._overall_recommendation()
            ),

            "recommendations": (
                self._build_recommendations()
            ),

            "matches": [
                result.to_dict()
                for result in self.match_results
            ],

            "identities": [
                result.to_dict()
                for result
                in self.identity_results.values()
            ],
        }

    # ==================================================================
    # Overall status
    # ==================================================================

    def _overall_status(
        self,
    ) -> str:

        # Only accepted match risks should determine the
        # global validation status.
        accepted_statuses = [
            result.status
            for result in self.match_results
            if result.matched
        ]

        identity_statuses = [
            result.status
            for result
            in self.identity_results.values()
        ]

        statuses = (
            accepted_statuses
            + identity_statuses
        )

        if self.STATUS_INVALID in statuses:
            return self.STATUS_INVALID

        if self.STATUS_HIGH_RISK in statuses:
            return self.STATUS_HIGH_RISK

        if self.STATUS_REVIEW in statuses:
            return self.STATUS_REVIEW

        return self.STATUS_PASS

    # ==================================================================
    # Recommendation
    # ==================================================================

    def _overall_recommendation(
        self,
    ) -> str:

        summary = self.get_summary()

        candidates = summary[
            "candidates"
        ]

        if candidates == 0:
            return (
                "NO MATCHES: no identity matching "
                "diagnostics were available for "
                "visual validation."
            )

        critical = summary[
            "risk"
        ]["critical"]

        high_risk = summary[
            "risk"
        ]["high_risk"]

        suspicious = summary[
            "risk"
        ]["suspicious_matches"]

        review = summary[
            "identities"
        ]["review"]

        if critical:
            return (
                "CRITICAL: one or more accepted identity "
                "assignments contain contradictory visual "
                "evidence and require immediate inspection."
            )

        if high_risk:
            return (
                "HIGH RISK: one or more accepted identity "
                "matches should be visually inspected."
            )

        if suspicious:
            return (
                "REVIEW: identity matching is generally usable, "
                "but some accepted matches contain validation "
                "warnings."
            )

        if review:
            return (
                "REVIEW: identity matching is generally usable, "
                "but some identities require additional inspection."
            )

        return (
            "EXCELLENT: accepted identity assignments show "
            "strong visual consistency and no significant "
            "validation risks were detected."
        )

    # ==================================================================
    # Recommendations
    # ==================================================================

    def _build_recommendations(
        self,
    ) -> list[str]:

        recommendations = []

        summary = self.get_summary()

        appearance = summary[
            "appearance"
        ]

        risk = summary[
            "risk"
        ]

        identities = summary[
            "identities"
        ]

        if appearance[
            "missing_embeddings"
        ] > 0:

            recommendations.append(
                "Some accepted candidates reached appearance "
                "evaluation but had no valid embeddings. "
                "Inspect the embedding extraction pipeline."
            )

        if appearance[
            "mismatch"
        ] > 0:

            recommendations.append(
                "Appearance mismatches were detected. "
                "Verify the corresponding accepted identity "
                "assignments visually."
            )

        if appearance[
            "weak"
        ] > 0:

            recommendations.append(
                "Weak appearance comparisons were detected. "
                "Review the corresponding accepted identity "
                "assignments before using them for final analytics."
            )

        if risk[
            "risky_matches"
        ] > 0:

            recommendations.append(
                "Risky accepted identity matches were detected. "
                "Inspect the corresponding track-to-identity "
                "assignments in the source video."
            )

        if risk[
            "suspicious_matches"
        ] > 0:

            recommendations.append(
                "Some accepted identity matches contain "
                "validation warnings. Review those assignments."
            )

        if risk[
            "critical"
        ] > 0:

            recommendations.append(
                "Critical accepted-match inconsistencies were "
                "detected. Do not treat affected identities as "
                "reliable until visually verified."
            )

        if summary[
            "appearance"
        ]["not_evaluated"] > 0:

            recommendations.append(
                "Many candidates were rejected before appearance "
                "evaluation. This is expected because the matcher "
                "uses temporal, spatial, motion, and overlap gates."
            )

        if identities[
            "pass"
        ] == identities[
            "total"
        ]:

            recommendations.append(
                "All global identities passed identity-level "
                "validation."
            )

        return (
            recommendations
            or [
                "No major visual validation issues were detected."
            ]
        )

    # ==================================================================
    # Convenience methods
    # ==================================================================

    def get_match_results(
        self,
    ) -> list[dict[str, Any]]:

        return [
            result.to_dict()
            for result in self.match_results
        ]

    def get_risky_matches(
        self,
    ) -> list[dict[str, Any]]:

        return [
            result.to_dict()
            for result in self.match_results
            if (
                result.matched
                and result.risk_level
                in {
                    self.RISK_HIGH,
                    self.RISK_CRITICAL,
                }
            )
        ]

    def get_suspicious_matches(
        self,
    ) -> list[dict[str, Any]]:

        return [
            result.to_dict()
            for result in self.match_results
            if (
                result.matched
                and result.validation_warnings
            )
        ]

    def get_identity_results(
        self,
    ) -> list[dict[str, Any]]:

        return [
            result.to_dict()
            for result
            in self.identity_results.values()
        ]

    def get_high_risk_identities(
        self,
    ) -> list[dict[str, Any]]:

        return [
            result.to_dict()
            for result
            in self.identity_results.values()
            if result.risk_level
            in {
                self.RISK_HIGH,
                self.RISK_CRITICAL,
            }
        ]

    # ==================================================================
    # Logging
    # ==================================================================

    def log_report(
        self,
    ) -> None:

        summary = self.get_summary()

        appearance = summary[
            "appearance"
        ]

        risk = summary[
            "risk"
        ]

        identities = summary[
            "identities"
        ]

        logger.info("=" * 64)

        logger.info(
            "IDENTITY VISUAL VALIDATION %s",
            self.VERSION.upper(),
        )

        logger.info(
            "Status: %s",
            self._overall_status(),
        )

        logger.info(
            "Candidates: %d",
            summary["candidates"],
        )

        logger.info(
            "Accepted matches: %d",
            summary["accepted_matches"],
        )

        logger.info(
            "Rejected candidates: %d",
            summary["rejected_matches"],
        )

        logger.info(
            "Appearance evaluated: %d",
            appearance["evaluated"],
        )

        logger.info(
            "Appearance not evaluated: %d",
            appearance["not_evaluated"],
        )

        logger.info(
            "Valid appearance comparisons: %d",
            appearance["valid_comparisons"],
        )

        logger.info(
            "Missing embeddings: %d",
            appearance["missing_embeddings"],
        )

        logger.info(
            "Appearance mismatches: %d",
            appearance["mismatch"],
        )

        logger.info(
            "Strong appearance comparisons: %d",
            appearance["strong"],
        )

        logger.info(
            "Normal appearance comparisons: %d",
            appearance["normal"],
        )

        logger.info(
            "Weak appearance comparisons: %d",
            appearance["weak"],
        )

        logger.info(
            "Average appearance similarity: %.4f",
            appearance["average_similarity"],
        )

        logger.info(
            "Accepted average appearance similarity: %.4f",
            appearance[
                "accepted_average_similarity"
            ],
        )

        logger.info(
            "Risky accepted matches: %d",
            risk["risky_matches"],
        )

        logger.info(
            "Suspicious accepted matches: %d",
            risk["suspicious_matches"],
        )

        logger.info(
            "Accepted-match validation warnings: %d",
            risk["validation_warning_count"],
        )

        logger.info(
            "Matcher diagnostics: %d",
            risk["matcher_diagnostic_count"],
        )

        logger.info(
            "High-risk accepted matches: %d",
            risk["high_risk"],
        )

        logger.info(
            "Critical accepted matches: %d",
            risk["critical"],
        )

        logger.info(
            "Total identities: %d",
            identities["total"],
        )

        logger.info(
            "Passing identities: %d",
            identities["pass"],
        )

        logger.info(
            "Review identities: %d",
            identities["review"],
        )

        logger.info(
            "High-risk identities: %d",
            identities["high_risk"],
        )

        logger.info(
            "Invalid identities: %d",
            identities["invalid"],
        )

        logger.info(
            "Recommendation: %s",
            self._overall_recommendation(),
        )

        logger.info("=" * 64)

    def log_risky_matches(
        self,
    ) -> None:

        risky = self.get_risky_matches()

        if not risky:
            logger.info(
                "No risky accepted identity matches detected."
            )
            return

        logger.warning(
            "Found %d risky accepted identity matches.",
            len(risky),
        )

        for result in risky:

            logger.warning(
                "RISKY MATCH | "
                "identity=%s "
                "track=%s "
                "score=%.4f "
                "appearance=%.4f "
                "temporal_gap=%d "
                "spatial=%.2f "
                "motion=%.2f "
                "risk=%s "
                "reason=%s "
                "warnings=%s",

                result["identity_id"],
                result["track_id"],
                result["score"],
                result["appearance_similarity"],
                result["temporal_gap"],
                result["spatial_distance"],
                result["motion_difference"],
                result["risk_level"],
                result["reason"],
                result["validation_warnings"],
            )

    # ==================================================================
    # Utilities
    # ==================================================================

    @staticmethod
    def _average(
        values: list[float],
    ) -> float:

        return (
            sum(values) / len(values)
            if values
            else 0.0
        )

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _safe_int(
        value: Any,
        default: int = 0,
    ) -> int:

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return default


# ============================================================================
# Convenience functions
# ============================================================================


def validate_identity_manager(
    manager: Any,
) -> dict[str, Any]:

    validator = IdentityVisualValidator()

    return validator.validate_manager(
        manager
    )


def validate_identity_diagnostics(
    diagnostics: Iterable[
        dict[str, Any]
    ],
) -> dict[str, Any]:

    validator = IdentityVisualValidator()

    validator.validate_diagnostics(
        diagnostics
    )

    return validator.get_report()


def get_risky_identity_matches(
    manager: Any,
) -> list[dict[str, Any]]:

    validator = IdentityVisualValidator()

    validator.validate_manager(
        manager
    )

    return validator.get_risky_matches()


def get_identity_validation_report(
    manager: Any,
) -> dict[str, Any]:

    validator = IdentityVisualValidator()

    return validator.validate_manager(
        manager
    )