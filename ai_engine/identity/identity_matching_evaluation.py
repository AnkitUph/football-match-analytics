"""
Global Identity Matching Evaluation V5.1.

This module provides TWO compatible evaluation APIs.

============================================================
1. IdentityMatchingEvaluation
============================================================

Live diagnostics evaluator used by GlobalIdentityManager.

It consumes diagnostic dictionaries produced by:

    GlobalIdentityManager._record_match_diagnostic()

It provides:

    - diagnostics
    - get_report()
    - get_summary()
    - log_report()

It does NOT perform matching.

============================================================
2. IdentityMatchingEvaluator
============================================================

Offline / labelled evaluation API.

It delegates every matching decision to:

    IdentityMatcher.match()

It provides:

    - EvaluationCase
    - EvaluationRecord
    - EvaluationMetrics
    - EvaluationReport
    - IdentityMatchingEvaluator
    - evaluate_identity_matching()
    - evaluate_candidate_pairs()
    - print_evaluation_report()
    - print_failed_cases()
    - print_false_positive_cases()
    - validate_matcher_interface()

============================================================
IMPORTANT
============================================================

This module never changes IdentityMatcher decisions.

It does not modify Track or Identity objects.

All production matching decisions remain inside:

    ai_engine.identity.identity_matcher.IdentityMatcher
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from ai_engine.identity.identity_matcher import (
    IdentityMatcher,
    MatchResult,
)

from ai_engine.schemas.track import (
    Identity,
    Track,
)


logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

VERSION = "v5.1"


# ============================================================
# LIVE DIAGNOSTICS EVALUATION
# ============================================================


class IdentityMatchingEvaluation:
    """
    Live evaluation of GlobalIdentityManager diagnostics.

    This class is intentionally separate from
    IdentityMatchingEvaluator.

    IdentityMatchingEvaluation:
        Consumes existing diagnostics.

    IdentityMatchingEvaluator:
        Calls IdentityMatcher.match() for explicit evaluation
        cases.

    This compatibility class exists because GlobalIdentityManager
    uses this API directly.
    """

    VERSION = VERSION

    def __init__(
        self,
        diagnostics: Optional[
            Iterable[dict[str, Any]]
        ] = None,
        minimum_appearance_similarity: float = 0.68,
        strong_appearance_similarity: float = 0.85,
    ) -> None:

        self.diagnostics = list(
            diagnostics or []
        )

        self.minimum_appearance_similarity = float(
            minimum_appearance_similarity
        )

        self.strong_appearance_similarity = float(
            strong_appearance_similarity
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def get_report(self) -> dict[str, Any]:
        """
        Build a complete live diagnostic report.

        This method does not call IdentityMatcher.
        """

        appearance = (
            self._evaluate_appearance()
        )

        similarity = (
            self._evaluate_similarity_distribution()
        )

        accepted = (
            self._evaluate_accepted_matches()
        )

        rejected = (
            self._evaluate_rejected_matches()
        )

        risky = (
            self._evaluate_risky_matches()
        )

        identities = (
            self._evaluate_identity_fragmentation()
        )

        return {
            "version": self.VERSION,

            "total_candidates": len(
                self.diagnostics
            ),

            "appearance": appearance,

            "similarity_distribution": similarity,

            "accepted_matches": accepted,

            "rejected_matches": rejected,

            "risky_matches": risky,

            "identity_fragmentation": identities,

            "recommendation": (
                self._generate_recommendation(
                    appearance,
                    similarity,
                    risky,
                )
            ),
        }

    def get_summary(self) -> dict[str, Any]:
        """
        Return the compact summary expected by
        GlobalIdentityManager.
        """

        report = self.get_report()

        appearance = (
            report["appearance"]
        )

        similarity = (
            report["similarity_distribution"]
        )

        risky = (
            report["risky_matches"]
        )

        return {
            "candidates": (
                report["total_candidates"]
            ),

            "appearance_comparisons": (
                appearance[
                    "valid_comparisons"
                ]
            ),

            "appearance_accepted": (
                appearance[
                    "accepted_matches"
                ]
            ),

            "appearance_rejected": (
                appearance[
                    "rejected_matches"
                ]
            ),

            "strong_matches": (
                appearance[
                    "strong_matches"
                ]
            ),

            "average_similarity": (
                similarity[
                    "overall"
                ]["avg"]
            ),

            "accepted_average_similarity": (
                similarity[
                    "accepted"
                ]["avg"]
            ),

            "rejected_average_similarity": (
                similarity[
                    "rejected"
                ]["avg"]
            ),

            "risky_matches": (
                risky["count"]
            ),

            "recommendation": (
                report["recommendation"]
            ),
        }

    def log_report(self) -> None:
        """
        Log the live evaluation report.
        """

        report = self.get_report()

        logger.info(
            "============================================================"
        )

        logger.info(
            "IDENTITY MATCHING EVALUATION %s",
            self.VERSION.upper(),
        )

        logger.info(
            "============================================================"
        )

        logger.info(
            "Candidates: %d",
            report["total_candidates"],
        )

        appearance = (
            report["appearance"]
        )

        logger.info(
            "Appearance comparisons: %d",
            appearance[
                "valid_comparisons"
            ],
        )

        logger.info(
            "Appearance accepted: %d",
            appearance[
                "accepted_matches"
            ],
        )

        logger.info(
            "Appearance rejected: %d",
            appearance[
                "rejected_matches"
            ],
        )

        logger.info(
            "Strong appearance matches: %d",
            appearance[
                "strong_matches"
            ],
        )

        similarity = (
            report[
                "similarity_distribution"
            ]
        )

        logger.info(
            "Overall similarity: %s",
            similarity["overall"],
        )

        logger.info(
            "Accepted similarity: %s",
            similarity["accepted"],
        )

        logger.info(
            "Rejected similarity: %s",
            similarity["rejected"],
        )

        risky = (
            report["risky_matches"]
        )

        logger.info(
            "Potentially risky matches: %d",
            risky["count"],
        )

        logger.info(
            "Recommendation: %s",
            report["recommendation"],
        )

    # ========================================================
    # APPEARANCE
    # ========================================================

    def _evaluate_appearance(
        self,
    ) -> dict[str, Any]:

        appearance_entries = [
            item
            for item in self.diagnostics
            if self._has_valid_appearance(item)
        ]

        accepted = [
            item
            for item in appearance_entries
            if self._is_accepted(item)
        ]

        rejected = [
            item
            for item in appearance_entries
            if not self._is_accepted(item)
        ]

        strong = [
            item
            for item in appearance_entries
            if self._similarity(item)
            >= self.strong_appearance_similarity
        ]

        weak = [
            item
            for item in appearance_entries
            if self._similarity(item)
            < self.strong_appearance_similarity
        ]

        mismatches = [
            item
            for item in appearance_entries
            if item.get("reason")
            in {
                "appearance_mismatch",
                "appearance_rejected",
                "appearance_rejected_by_threshold",
            }
        ]

        return {
            "valid_comparisons": len(
                appearance_entries
            ),

            "accepted_matches": len(
                accepted
            ),

            "rejected_matches": len(
                rejected
            ),

            "strong_matches": len(
                strong
            ),

            "weak_matches": len(
                weak
            ),

            "appearance_mismatches": len(
                mismatches
            ),

            "acceptance_rate": self._rate(
                len(accepted),
                len(appearance_entries),
            ),

            "strong_match_rate": self._rate(
                len(strong),
                len(appearance_entries),
            ),

            "weak_match_rate": self._rate(
                len(weak),
                len(appearance_entries),
            ),
        }

    # ========================================================
    # SIMILARITY DISTRIBUTION
    # ========================================================

    def _evaluate_similarity_distribution(
        self,
    ) -> dict[str, Any]:

        appearance_entries = [
            item
            for item in self.diagnostics
            if self._has_valid_appearance(item)
        ]

        accepted = [
            item
            for item in appearance_entries
            if self._is_accepted(item)
        ]

        rejected = [
            item
            for item in appearance_entries
            if not self._is_accepted(item)
        ]

        return {
            "overall": self._statistics(
                [
                    self._similarity(item)
                    for item in appearance_entries
                ]
            ),

            "accepted": self._statistics(
                [
                    self._similarity(item)
                    for item in accepted
                ]
            ),

            "rejected": self._statistics(
                [
                    self._similarity(item)
                    for item in rejected
                ]
            ),

            "bands": self._similarity_bands(
                appearance_entries
            ),
        }

    # ========================================================
    # ACCEPTED MATCHES
    # ========================================================

    def _evaluate_accepted_matches(
        self,
    ) -> dict[str, Any]:

        accepted = [
            item
            for item in self.diagnostics
            if self._is_accepted(item)
            and self._has_valid_appearance(item)
        ]

        low_confidence = [
            item
            for item in accepted
            if self._similarity(item) < 0.75
        ]

        medium_confidence = [
            item
            for item in accepted
            if (
                0.75
                <= self._similarity(item)
                < self.strong_appearance_similarity
            )
        ]

        strong = [
            item
            for item in accepted
            if self._similarity(item)
            >= self.strong_appearance_similarity
        ]

        return {
            "count": len(accepted),

            "low_confidence": len(
                low_confidence
            ),

            "medium_confidence": len(
                medium_confidence
            ),

            "strong": len(strong),

            "low_confidence_rate": self._rate(
                len(low_confidence),
                len(accepted),
            ),

            "medium_confidence_rate": self._rate(
                len(medium_confidence),
                len(accepted),
            ),

            "strong_rate": self._rate(
                len(strong),
                len(accepted),
            ),

            "low_confidence_examples": (
                self._compact_entries(
                    low_confidence,
                    limit=20,
                )
            ),
        }

    # ========================================================
    # REJECTED MATCHES
    # ========================================================

    def _evaluate_rejected_matches(
        self,
    ) -> dict[str, Any]:

        rejected = [
            item
            for item in self.diagnostics
            if self._has_valid_appearance(item)
            and not self._is_accepted(item)
        ]

        appearance_mismatches = [
            item
            for item in rejected
            if item.get("reason")
            in {
                "appearance_mismatch",
                "appearance_rejected",
                "appearance_rejected_by_threshold",
            }
        ]

        high_similarity_rejections = [
            item
            for item in rejected
            if self._similarity(item)
            >= self.strong_appearance_similarity
        ]

        score_rejections = [
            item
            for item in rejected
            if item.get("reason")
            in {
                "compatible",
                "score_too_low",
                "below_threshold",
            }
        ]

        return {
            "count": len(rejected),

            "appearance_mismatches": len(
                appearance_mismatches
            ),

            "score_rejections": len(
                score_rejections
            ),

            "high_similarity_rejections": len(
                high_similarity_rejections
            ),

            "high_similarity_examples": (
                self._compact_entries(
                    high_similarity_rejections,
                    limit=20,
                )
            ),

            "appearance_mismatch_examples": (
                self._compact_entries(
                    appearance_mismatches,
                    limit=20,
                )
            ),
        }

    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    def _evaluate_risky_matches(
        self,
    ) -> dict[str, Any]:

        accepted = [
            item
            for item in self.diagnostics
            if self._is_accepted(item)
            and self._has_valid_appearance(item)
        ]

        risky = []

        for item in accepted:

            similarity = self._similarity(
                item
            )

            if (
                similarity
                < self.minimum_appearance_similarity
            ):

                risky.append(
                    (
                        item,
                        "low_appearance_similarity",
                    )
                )

                continue

            if (
                similarity < 0.80
                and self._weak_motion_or_spatial(
                    item
                )
            ):

                risky.append(
                    (
                        item,
                        "moderate_appearance_weak_motion_spatial",
                    )
                )

        return {
            "count": len(risky),

            "rate": self._rate(
                len(risky),
                len(accepted),
            ),

            "low_similarity": sum(
                1
                for _, reason in risky
                if reason
                == "low_appearance_similarity"
            ),

            "weak_motion_spatial": sum(
                1
                for _, reason in risky
                if reason
                == "moderate_appearance_weak_motion_spatial"
            ),

            "examples": [
                {
                    **self._compact_entry(
                        item
                    ),
                    "risk_reason": reason,
                }
                for item, reason in risky[:20]
            ],
        }

    # ========================================================
    # IDENTITY FRAGMENTATION
    # ========================================================

    def _evaluate_identity_fragmentation(
        self,
    ) -> dict[str, Any]:

        identity_counts: dict[Any, int] = {}

        for item in self.diagnostics:

            if not self._is_accepted(item):
                continue

            identity_id = item.get(
                "identity_id"
            )

            if identity_id is None:
                continue

            identity_counts[
                identity_id
            ] = (
                identity_counts.get(
                    identity_id,
                    0,
                )
                + 1
            )

        if not identity_counts:

            return {
                "identities_with_matches": 0,
                "maximum_accepted_merges": 0,
                "average_accepted_merges": 0.0,
                "distribution": {},
            }

        distribution: dict[int, int] = {}

        for count in identity_counts.values():

            distribution[count] = (
                distribution.get(
                    count,
                    0,
                )
                + 1
            )

        return {
            "identities_with_matches": len(
                identity_counts
            ),

            "maximum_accepted_merges": max(
                identity_counts.values()
            ),

            "average_accepted_merges": (
                sum(
                    identity_counts.values()
                )
                / len(identity_counts)
            ),

            "distribution": dict(
                sorted(
                    distribution.items()
                )
            ),
        }

    # ========================================================
    # SIMILARITY BANDS
    # ========================================================

    def _similarity_bands(
        self,
        entries: Iterable[dict[str, Any]],
    ) -> dict[str, int]:

        bands = {
            "0.00-0.60": 0,
            "0.60-0.70": 0,
            "0.70-0.75": 0,
            "0.75-0.80": 0,
            "0.80-0.85": 0,
            "0.85-0.90": 0,
            "0.90-1.00": 0,
        }

        for item in entries:

            similarity = self._similarity(
                item
            )

            if similarity < 0.60:
                bands["0.00-0.60"] += 1

            elif similarity < 0.70:
                bands["0.60-0.70"] += 1

            elif similarity < 0.75:
                bands["0.70-0.75"] += 1

            elif similarity < 0.80:
                bands["0.75-0.80"] += 1

            elif similarity < 0.85:
                bands["0.80-0.85"] += 1

            elif similarity < 0.90:
                bands["0.85-0.90"] += 1

            else:
                bands["0.90-1.00"] += 1

        return bands

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _is_accepted(
        item: dict[str, Any],
    ) -> bool:
        """
        Support both diagnostic naming conventions:

            matched
            accepted

        GlobalIdentityManager V5.1 uses `matched`.
        """

        if "matched" in item:
            return bool(
                item.get("matched")
            )

        return bool(
            item.get("accepted")
        )

    @staticmethod
    def _has_valid_appearance(
        item: dict[str, Any],
    ) -> bool:

        similarity = item.get(
            "appearance_similarity"
        )

        try:
            similarity = float(
                similarity
            )
        except (
            TypeError,
            ValueError,
        ):
            return False

        return similarity >= 0.0

    @staticmethod
    def _similarity(
        item: dict[str, Any],
    ) -> float:

        try:
            return float(
                item.get(
                    "appearance_similarity",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    @staticmethod
    def _rate(
        numerator: int,
        denominator: int,
    ) -> float:

        if denominator <= 0:
            return 0.0

        return (
            numerator
            / denominator
            * 100.0
        )

    @staticmethod
    def _statistics(
        values: list[float],
    ) -> dict[str, Any]:

        if not values:

            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
            }

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": (
                sum(values)
                / len(values)
            ),
        }

    @staticmethod
    def _weak_motion_or_spatial(
        item: dict[str, Any],
    ) -> bool:

        try:

            spatial = (
                float(
                    item["spatial_distance"]
                )
                if item.get(
                    "spatial_distance"
                )
                is not None
                else None
            )

            motion = (
                float(
                    item["motion_difference"]
                )
                if item.get(
                    "motion_difference"
                )
                is not None
                else None
            )

        except (
            TypeError,
            ValueError,
        ):
            return False

        if (
            spatial is not None
            and spatial > 150.0
        ):
            return True

        if (
            motion is not None
            and motion > 100.0
        ):
            return True

        return False

    def _compact_entries(
        self,
        entries: Sequence[dict[str, Any]],
        limit: int = 20,
    ) -> list[dict[str, Any]]:

        return [
            self._compact_entry(item)
            for item in list(entries)[:limit]
        ]

    @staticmethod
    def _compact_entry(
        item: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "track_id": item.get(
                "track_id"
            ),

            "identity_id": item.get(
                "identity_id"
            ),

            "matched": (
                item.get("matched")
                if "matched" in item
                else item.get("accepted")
            ),

            "score": item.get(
                "score"
            ),

            "appearance_similarity": (
                item.get(
                    "appearance_similarity"
                )
            ),

            "temporal_gap": item.get(
                "temporal_gap"
            ),

            "spatial_distance": item.get(
                "spatial_distance"
            ),

            "motion_difference": item.get(
                "motion_difference"
            ),

            "team_match": item.get(
                "team_match"
            ),

            "jersey_match": item.get(
                "jersey_match"
            ),

            "reason": item.get(
                "reason"
            ),
        }

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    def _generate_recommendation(
        self,
        appearance: dict[str, Any],
        similarity: dict[str, Any],
        risky: dict[str, Any],
    ) -> str:

        valid = appearance[
            "valid_comparisons"
        ]

        if valid == 0:

            return (
                "INSUFFICIENT_DATA: "
                "No valid appearance comparisons."
            )

        accepted_avg = similarity[
            "accepted"
        ]["avg"]

        strong_rate = appearance[
            "strong_match_rate"
        ]

        risky_rate = risky[
            "rate"
        ]

        if (
            accepted_avg >= 0.85
            and strong_rate >= 50.0
            and risky_rate <= 10.0
        ):

            return (
                "EXCELLENT: appearance matching "
                "quality is strong; keep the current "
                "threshold for further validation."
            )

        if (
            accepted_avg >= 0.80
            and risky_rate <= 20.0
        ):

            return (
                "GOOD: appearance matching is useful; "
                "continue validation before changing "
                "the threshold."
            )

        if risky_rate > 20.0:

            return (
                "REVIEW: a significant portion of "
                "accepted appearance matches are "
                "potentially risky."
            )

        if (
            accepted_avg
            < self.minimum_appearance_similarity
        ):

            return (
                "WEAK: accepted appearance similarity "
                "is low; review the appearance threshold "
                "and embedding quality."
            )

        return (
            "MONITOR: appearance matching is "
            "functional but requires more validation."
        )


# ============================================================
# EVALUATION CASE
# ============================================================


@dataclass(frozen=True)
class EvaluationCase:
    """
    One Track/Identity candidate pair.

    expected_match:
        True:
            Track and Identity should represent the same
            global identity.

        False:
            Track and Identity should represent different
            identities.

        None:
            No ground truth is available.
    """

    track: Track

    identity: Identity

    expected_match: Optional[bool] = None

    name: str = ""


# ============================================================
# EVALUATION RECORD
# ============================================================


@dataclass(frozen=True)
class EvaluationRecord:
    """
    Immutable evaluation result for one candidate pair.
    """

    case_name: str

    expected_match: Optional[bool]

    predicted_match: bool

    correct: Optional[bool]

    score: float

    temporal_gap: int

    spatial_distance: float

    motion_difference: float

    appearance_similarity: float

    appearance_evaluated: bool

    team_match: Optional[bool]

    jersey_match: Optional[bool]

    reason: str


# ============================================================
# EVALUATION METRICS
# ============================================================


@dataclass(frozen=True)
class EvaluationMetrics:
    """
    Classification metrics calculated from labelled cases.
    """

    total_cases: int

    labelled_cases: int

    unlabelled_cases: int

    true_positive: int

    true_negative: int

    false_positive: int

    false_negative: int

    precision: float

    recall: float

    f1: float

    accuracy: float

    predicted_matches: int

    predicted_non_matches: int

    actual_matches: int

    actual_non_matches: int

    appearance_evaluated: int

    appearance_not_evaluated: int

    appearance_available: int

    appearance_unavailable: int


# ============================================================
# EVALUATION REPORT
# ============================================================


@dataclass
class EvaluationReport:
    """
    Complete offline matcher evaluation report.
    """

    matcher_version: str

    records: list[EvaluationRecord] = field(
        default_factory=list
    )

    metrics: Optional[
        EvaluationMetrics
    ] = None

    reason_counts: Counter[str] = field(
        default_factory=Counter
    )

    appearance_reason_counts: Counter[str] = field(
        default_factory=Counter
    )

    team_mismatch_count: int = 0

    jersey_mismatch_count: int = 0

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the report into a JSON-serializable
        dictionary.
        """

        metrics = self.metrics

        return {
            "matcher_version": (
                self.matcher_version
            ),

            "metrics": (
                None
                if metrics is None
                else {
                    "total_cases": (
                        metrics.total_cases
                    ),

                    "labelled_cases": (
                        metrics.labelled_cases
                    ),

                    "unlabelled_cases": (
                        metrics.unlabelled_cases
                    ),

                    "true_positive": (
                        metrics.true_positive
                    ),

                    "true_negative": (
                        metrics.true_negative
                    ),

                    "false_positive": (
                        metrics.false_positive
                    ),

                    "false_negative": (
                        metrics.false_negative
                    ),

                    "precision": (
                        metrics.precision
                    ),

                    "recall": (
                        metrics.recall
                    ),

                    "f1": (
                        metrics.f1
                    ),

                    "accuracy": (
                        metrics.accuracy
                    ),

                    "predicted_matches": (
                        metrics.predicted_matches
                    ),

                    "predicted_non_matches": (
                        metrics.predicted_non_matches
                    ),

                    "actual_matches": (
                        metrics.actual_matches
                    ),

                    "actual_non_matches": (
                        metrics.actual_non_matches
                    ),

                    "appearance_evaluated": (
                        metrics.appearance_evaluated
                    ),

                    "appearance_not_evaluated": (
                        metrics.appearance_not_evaluated
                    ),

                    "appearance_available": (
                        metrics.appearance_available
                    ),

                    "appearance_unavailable": (
                        metrics.appearance_unavailable
                    ),
                }
            ),

            "reason_counts": dict(
                self.reason_counts
            ),

            "appearance_reason_counts": dict(
                self.appearance_reason_counts
            ),

            "team_mismatch_count": (
                self.team_mismatch_count
            ),

            "jersey_mismatch_count": (
                self.jersey_mismatch_count
            ),

            "records": [
                {
                    "case_name": (
                        record.case_name
                    ),

                    "expected_match": (
                        record.expected_match
                    ),

                    "predicted_match": (
                        record.predicted_match
                    ),

                    "correct": (
                        record.correct
                    ),

                    "score": (
                        record.score
                    ),

                    "temporal_gap": (
                        record.temporal_gap
                    ),

                    "spatial_distance": (
                        record.spatial_distance
                    ),

                    "motion_difference": (
                        record.motion_difference
                    ),

                    "appearance_similarity": (
                        record.appearance_similarity
                    ),

                    "appearance_evaluated": (
                        record.appearance_evaluated
                    ),

                    "team_match": (
                        record.team_match
                    ),

                    "jersey_match": (
                        record.jersey_match
                    ),

                    "reason": (
                        record.reason
                    ),
                }
                for record in self.records
            ],
        }

    # ========================================================
    # HUMAN SUMMARY
    # ========================================================

    def summary(self) -> str:
        """
        Generate a human-readable offline evaluation summary.
        """

        lines = [
            "============================================================",
            "IDENTITY MATCHING EVALUATION",
            "============================================================",
            (
                f"Matcher version: "
                f"{self.matcher_version}"
            ),
            "",
        ]

        if self.metrics is None:

            lines.extend(
                [
                    "No ground-truth metrics available.",
                    "",
                ]
            )

        else:

            metrics = self.metrics

            lines.extend(
                [
                    "CLASSIFICATION METRICS",
                    "------------------------------------------------------------",
                    (
                        f"Total cases       : "
                        f"{metrics.total_cases}"
                    ),
                    (
                        f"Labelled cases    : "
                        f"{metrics.labelled_cases}"
                    ),
                    (
                        f"Unlabelled cases  : "
                        f"{metrics.unlabelled_cases}"
                    ),
                    "",
                    (
                        f"True positives    : "
                        f"{metrics.true_positive}"
                    ),
                    (
                        f"True negatives    : "
                        f"{metrics.true_negative}"
                    ),
                    (
                        f"False positives   : "
                        f"{metrics.false_positive}"
                    ),
                    (
                        f"False negatives   : "
                        f"{metrics.false_negative}"
                    ),
                    "",
                    (
                        f"Precision         : "
                        f"{metrics.precision:.4f}"
                    ),
                    (
                        f"Recall            : "
                        f"{metrics.recall:.4f}"
                    ),
                    (
                        f"F1 score          : "
                        f"{metrics.f1:.4f}"
                    ),
                    (
                        f"Accuracy          : "
                        f"{metrics.accuracy:.4f}"
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "MATCHING DECISIONS",
                "------------------------------------------------------------",
                (
                    f"Predicted matches     : "
                    f"{self._predicted_matches()}"
                ),
                (
                    f"Predicted non-matches : "
                    f"{self._predicted_non_matches()}"
                ),
                "",
                "REASON COUNTS",
                "------------------------------------------------------------",
            ]
        )

        if self.reason_counts:

            for reason, count in (
                self.reason_counts.most_common()
            ):

                lines.append(
                    f"{reason:<45} {count}"
                )

        else:

            lines.append(
                "No decisions recorded."
            )

        lines.extend(
            [
                "",
                "APPEARANCE",
                "------------------------------------------------------------",
                (
                    f"Appearance evaluated   : "
                    f"{self._appearance_evaluated()}"
                ),
                (
                    f"Appearance not reached : "
                    f"{self._appearance_not_evaluated()}"
                ),
                (
                    f"Appearance available   : "
                    f"{self._appearance_available()}"
                ),
                (
                    f"Appearance unavailable : "
                    f"{self._appearance_unavailable()}"
                ),
                "",
                "TEAM / JERSEY",
                "------------------------------------------------------------",
                (
                    f"Team mismatches        : "
                    f"{self.team_mismatch_count}"
                ),
                (
                    f"Jersey mismatches      : "
                    f"{self.jersey_mismatch_count}"
                ),
                "",
                "============================================================",
            ]
        )

        return "\n".join(lines)

    def _predicted_matches(self) -> int:

        return sum(
            record.predicted_match
            for record in self.records
        )

    def _predicted_non_matches(self) -> int:

        return sum(
            not record.predicted_match
            for record in self.records
        )

    def _appearance_evaluated(self) -> int:

        return sum(
            record.appearance_evaluated
            for record in self.records
        )

    def _appearance_not_evaluated(
        self,
    ) -> int:

        return sum(
            not record.appearance_evaluated
            for record in self.records
        )

    def _appearance_available(self) -> int:

        return sum(
            record.appearance_evaluated
            and record.appearance_similarity >= 0.0
            for record in self.records
        )

    def _appearance_unavailable(self) -> int:

        return sum(
            record.appearance_evaluated
            and record.appearance_similarity < 0.0
            for record in self.records
        )


# ============================================================
# OFFLINE V5.1 EVALUATOR
# ============================================================


class IdentityMatchingEvaluator:
    """
    Production evaluator for IdentityMatcher V5.1.

    This evaluator delegates matching decisions directly to
    IdentityMatcher.

    It does not duplicate matching rules or thresholds.
    """

    VERSION = VERSION

    def __init__(
        self,
        matcher: Optional[
            IdentityMatcher
        ] = None,
    ) -> None:

        self.matcher = (
            matcher
            if matcher is not None
            else IdentityMatcher()
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def evaluate(
        self,
        cases: Iterable[
            EvaluationCase
        ],
    ) -> EvaluationReport:
        """
        Evaluate multiple candidate pairs.
        """

        report = EvaluationReport(
            matcher_version=getattr(
                self.matcher,
                "VERSION",
                self.VERSION,
            )
        )

        for index, case in enumerate(
            cases
        ):

            record = self._evaluate_case(
                case=case,
                index=index,
            )

            report.records.append(
                record
            )

            report.reason_counts[
                record.reason
            ] += 1

            if (
                record.reason.startswith(
                    "appearance_"
                )
                or record.reason
                in {
                    "long_gap_weak_appearance",
                    "long_gap_spatial_uncertainty",
                    "long_gap_motion_uncertainty",
                    "strong_appearance_geometry_too_weak",
                    "normal_appearance_geometry_too_weak",
                    "weak_appearance_geometry",
                }
            ):

                report.appearance_reason_counts[
                    record.reason
                ] += 1

            if (
                record.team_match
                is False
            ):

                report.team_mismatch_count += 1

            if (
                record.jersey_match
                is False
            ):

                report.jersey_mismatch_count += 1

        report.metrics = (
            self._calculate_metrics(
                report.records
            )
        )

        return report

    def evaluate_case(
        self,
        case: EvaluationCase,
    ) -> EvaluationRecord:
        """
        Evaluate exactly one candidate pair.
        """

        return self._evaluate_case(
            case=case,
            index=0,
        )

    def match(
        self,
        track: Track,
        identity: Identity,
    ) -> MatchResult:
        """
        Direct convenience wrapper around
        IdentityMatcher.match().
        """

        return self.matcher.match(
            track,
            identity,
        )

    # ========================================================
    # CASE EVALUATION
    # ========================================================

    def _evaluate_case(
        self,
        case: EvaluationCase,
        index: int,
    ) -> EvaluationRecord:

        result = self.matcher.match(
            case.track,
            case.identity,
        )

        case_name = (
            case.name.strip()
            if case.name
            else f"case_{index + 1}"
        )

        if case.expected_match is None:

            correct = None

        else:

            correct = (
                result.matched
                == case.expected_match
            )

        return EvaluationRecord(
            case_name=case_name,

            expected_match=(
                case.expected_match
            ),

            predicted_match=(
                result.matched
            ),

            correct=correct,

            score=float(
                result.score
            ),

            temporal_gap=int(
                result.temporal_gap
            ),

            spatial_distance=float(
                result.spatial_distance
            ),

            motion_difference=float(
                result.motion_difference
            ),

            appearance_similarity=float(
                result.appearance_similarity
            ),

            appearance_evaluated=bool(
                result.appearance_evaluated
            ),

            team_match=result.team_match,

            jersey_match=result.jersey_match,

            reason=str(
                result.reason
            ),
        )

    # ========================================================
    # METRICS
    # ========================================================

    @staticmethod
    def _calculate_metrics(
        records: Sequence[
            EvaluationRecord
        ],
    ) -> EvaluationMetrics:

        total_cases = len(
            records
        )

        labelled_records = [
            record
            for record in records
            if record.expected_match
            is not None
        ]

        unlabelled_cases = (
            total_cases
            - len(labelled_records)
        )

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for record in labelled_records:

            expected = (
                record.expected_match
            )

            predicted = (
                record.predicted_match
            )

            if (
                expected is True
                and predicted
            ):

                true_positive += 1

            elif (
                expected is False
                and not predicted
            ):

                true_negative += 1

            elif (
                expected is False
                and predicted
            ):

                false_positive += 1

            elif (
                expected is True
                and not predicted
            ):

                false_negative += 1

        precision = (
            IdentityMatchingEvaluator
            ._safe_divide(
                true_positive,
                (
                    true_positive
                    + false_positive
                ),
            )
        )

        recall = (
            IdentityMatchingEvaluator
            ._safe_divide(
                true_positive,
                (
                    true_positive
                    + false_negative
                ),
            )
        )

        f1 = (
            IdentityMatchingEvaluator
            ._safe_f1(
                precision,
                recall,
            )
        )

        accuracy = (
            IdentityMatchingEvaluator
            ._safe_divide(
                (
                    true_positive
                    + true_negative
                ),
                len(labelled_records),
            )
        )

        predicted_matches = sum(
            record.predicted_match
            for record in records
        )

        predicted_non_matches = (
            total_cases
            - predicted_matches
        )

        actual_matches = sum(
            record.expected_match is True
            for record in labelled_records
        )

        actual_non_matches = sum(
            record.expected_match is False
            for record in labelled_records
        )

        appearance_evaluated = sum(
            record.appearance_evaluated
            for record in records
        )

        appearance_not_evaluated = (
            total_cases
            - appearance_evaluated
        )

        appearance_available = sum(
            record.appearance_evaluated
            and record.appearance_similarity
            >= 0.0
            for record in records
        )

        appearance_unavailable = sum(
            record.appearance_evaluated
            and record.appearance_similarity
            < 0.0
            for record in records
        )

        return EvaluationMetrics(
            total_cases=total_cases,

            labelled_cases=len(
                labelled_records
            ),

            unlabelled_cases=(
                unlabelled_cases
            ),

            true_positive=(
                true_positive
            ),

            true_negative=(
                true_negative
            ),

            false_positive=(
                false_positive
            ),

            false_negative=(
                false_negative
            ),

            precision=precision,

            recall=recall,

            f1=f1,

            accuracy=accuracy,

            predicted_matches=(
                predicted_matches
            ),

            predicted_non_matches=(
                predicted_non_matches
            ),

            actual_matches=(
                actual_matches
            ),

            actual_non_matches=(
                actual_non_matches
            ),

            appearance_evaluated=(
                appearance_evaluated
            ),

            appearance_not_evaluated=(
                appearance_not_evaluated
            ),

            appearance_available=(
                appearance_available
            ),

            appearance_unavailable=(
                appearance_unavailable
            ),
        )

    # ========================================================
    # MATH HELPERS
    # ========================================================

    @staticmethod
    def _safe_divide(
        numerator: int,
        denominator: int,
    ) -> float:

        if denominator <= 0:
            return 0.0

        return (
            numerator
            / denominator
        )

    @staticmethod
    def _safe_f1(
        precision: float,
        recall: float,
    ) -> float:

        denominator = (
            precision
            + recall
        )

        if denominator <= 0:
            return 0.0

        return (
            2.0
            * precision
            * recall
            / denominator
        )


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def evaluate_identity_matching(
    cases: Iterable[
        EvaluationCase
    ],
    matcher: Optional[
        IdentityMatcher
    ] = None,
) -> EvaluationReport:
    """
    Convenience API for evaluating matcher cases.
    """

    evaluator = (
        IdentityMatchingEvaluator(
            matcher=matcher
        )
    )

    return evaluator.evaluate(
        cases
    )


def evaluate_candidate_pairs(
    pairs: Iterable[
        tuple[Track, Identity]
    ],
    expected_matches: Optional[
        Sequence[Optional[bool]]
    ] = None,
    matcher: Optional[
        IdentityMatcher
    ] = None,
) -> EvaluationReport:
    """
    Evaluate raw Track/Identity pairs.

    expected_matches is optional.

    If supplied, its length must match the number
    of candidate pairs.
    """

    pair_list = list(
        pairs
    )

    if expected_matches is None:

        labels = [
            None
            for _ in pair_list
        ]

    else:

        labels = list(
            expected_matches
        )

        if (
            len(labels)
            != len(pair_list)
        ):

            raise ValueError(
                "expected_matches length "
                "must match the number of "
                "candidate pairs."
            )

    cases = [
        EvaluationCase(
            track=track,

            identity=identity,

            expected_match=(
                expected_match
            ),

            name=(
                f"pair_{index + 1}"
            ),
        )

        for index, (
            (track, identity),
            expected_match,
        ) in enumerate(
            zip(
                pair_list,
                labels,
            )
        )
    ]

    return evaluate_identity_matching(
        cases=cases,
        matcher=matcher,
    )


# ============================================================
# REPORT UTILITIES
# ============================================================


def print_evaluation_report(
    report: EvaluationReport,
) -> None:
    """
    Print an offline evaluation report.
    """

    print(
        report.summary()
    )


def print_failed_cases(
    report: EvaluationReport,
) -> None:
    """
    Print false-negative cases.

    These are cases where:

        expected_match == True

    but the matcher rejected the candidate.
    """

    print(
        "============================================================"
    )

    print(
        "FALSE NEGATIVE CANDIDATES"
    )

    print(
        "============================================================"
    )

    found = False

    for record in report.records:

        if (
            record.expected_match is True
            and not record.predicted_match
        ):

            found = True

            print(
                f"\nCase: "
                f"{record.case_name}"
            )

            print(
                f"Reason: "
                f"{record.reason}"
            )

            print(
                f"Score: "
                f"{record.score:.4f}"
            )

            print(
                f"Temporal gap: "
                f"{record.temporal_gap}"
            )

            print(
                f"Spatial distance: "
                f"{record.spatial_distance:.3f}"
            )

            print(
                f"Motion difference: "
                f"{record.motion_difference:.3f}"
            )

            print(
                f"Appearance similarity: "
                f"{record.appearance_similarity:.4f}"
            )

            print(
                f"Appearance evaluated: "
                f"{record.appearance_evaluated}"
            )

            print(
                f"Team match: "
                f"{record.team_match}"
            )

            print(
                f"Jersey match: "
                f"{record.jersey_match}"
            )

    if not found:

        print(
            "No false-negative candidates."
        )


def print_false_positive_cases(
    report: EvaluationReport,
) -> None:
    """
    Print false-positive cases.

    These are cases where:

        expected_match == False

    but the matcher accepted the candidate.
    """

    print(
        "============================================================"
    )

    print(
        "FALSE POSITIVE CANDIDATES"
    )

    print(
        "============================================================"
    )

    found = False

    for record in report.records:

        if (
            record.expected_match is False
            and record.predicted_match
        ):

            found = True

            print(
                f"\nCase: "
                f"{record.case_name}"
            )

            print(
                f"Reason: "
                f"{record.reason}"
            )

            print(
                f"Score: "
                f"{record.score:.4f}"
            )

            print(
                f"Temporal gap: "
                f"{record.temporal_gap}"
            )

            print(
                f"Spatial distance: "
                f"{record.spatial_distance:.3f}"
            )

            print(
                f"Motion difference: "
                f"{record.motion_difference:.3f}"
            )

            print(
                f"Appearance similarity: "
                f"{record.appearance_similarity:.4f}"
            )

            print(
                f"Appearance evaluated: "
                f"{record.appearance_evaluated}"
            )

            print(
                f"Team match: "
                f"{record.team_match}"
            )

            print(
                f"Jersey match: "
                f"{record.jersey_match}"
            )

    if not found:

        print(
            "No false-positive candidates."
        )


# ============================================================
# SELF-CHECK
# ============================================================


def validate_matcher_interface(
    matcher: Optional[
        IdentityMatcher
    ] = None,
) -> None:
    """
    Validate the IdentityMatcher V5.1 interface.

    This does not create fake Track or Identity objects.
    """

    matcher = (
        matcher
        if matcher is not None
        else IdentityMatcher()
    )

    required_matcher_attributes = (
        "match",
        "VERSION",
    )

    for attribute in (
        required_matcher_attributes
    ):

        if not hasattr(
            matcher,
            attribute,
        ):

            raise TypeError(
                "IdentityMatcher is missing "
                "required V5.1 attribute: "
                f"{attribute}"
            )

    matcher_version = getattr(
        matcher,
        "VERSION",
        None,
    )

    if matcher_version != "v5.1":

        raise ValueError(
            "Expected IdentityMatcher V5.1, "
            f"got {matcher_version!r}."
        )


# ============================================================
# MODULE SELF-TEST
# ============================================================


if __name__ == "__main__":

    matcher = IdentityMatcher()

    validate_matcher_interface(
        matcher
    )

    # Validate the live diagnostics API
    live_evaluation = (
        IdentityMatchingEvaluation()
    )

    live_summary = (
        live_evaluation.get_summary()
    )

    assert (
        live_summary["candidates"]
        == 0
    )

    # Validate the offline evaluator API
    evaluator = (
        IdentityMatchingEvaluator(
            matcher=matcher
        )
    )

    print(
        "Identity Matching Evaluation V5.1"
    )

    print(
        "Matcher version:",
        matcher.VERSION,
    )

    print(
        "Live evaluation API: PASS"
    )

    print(
        "Offline evaluator API: PASS"
    )

    print(
        "Production matcher interface: PASS"
    )

    print(
        "No synthetic Track/Identity objects "
        "were created."
    )