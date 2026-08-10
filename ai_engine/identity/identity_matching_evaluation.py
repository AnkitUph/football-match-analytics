"""
Identity Matching Evaluation V1.

Evaluates the quality of global identity matching without
changing actual matching decisions.

This module consumes diagnostics produced by
GlobalIdentityManager.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


class IdentityMatchingEvaluation:
    """
    Evaluate identity matching diagnostics.

    This evaluator NEVER modifies identities or matching
    decisions.
    """

    def __init__(
        self,
        diagnostics: Optional[
            Iterable[dict[str, Any]]
        ] = None,
        minimum_appearance_similarity: float = 0.70,
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
            "version": "v1",

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

        report = self.get_report()

        appearance = report["appearance"]
        similarity = report["similarity_distribution"]
        risky = report["risky_matches"]

        return {
            "candidates": report[
                "total_candidates"
            ],

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

            "risky_matches": risky[
                "count"
            ],

            "recommendation": report[
                "recommendation"
            ],
        }

    def log_report(self) -> None:

        report = self.get_report()

        logger.info(
            "=================================================="
        )

        logger.info(
            "IDENTITY MATCHING EVALUATION"
        )

        logger.info(
            "=================================================="
        )

        logger.info(
            "Candidates: %d",
            report["total_candidates"],
        )

        appearance = report["appearance"]

        logger.info(
            "Appearance comparisons: %d",
            appearance["valid_comparisons"],
        )

        logger.info(
            "Appearance accepted: %d",
            appearance["accepted_matches"],
        )

        logger.info(
            "Appearance rejected: %d",
            appearance["rejected_matches"],
        )

        logger.info(
            "Strong appearance matches: %d",
            appearance["strong_matches"],
        )

        similarity = report[
            "similarity_distribution"
        ]

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

        risky = report["risky_matches"]

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
            if item.get("accepted") is True
        ]

        rejected = [
            item
            for item in appearance_entries
            if item.get("accepted") is not True
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
            == "appearance_mismatch"
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
    # SIMILARITY
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
            if item.get("accepted") is True
        ]

        rejected = [
            item
            for item in appearance_entries
            if item.get("accepted") is not True
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
    # ACCEPTED
    # ========================================================

    def _evaluate_accepted_matches(
        self,
    ) -> dict[str, Any]:

        accepted = [
            item
            for item in self.diagnostics
            if item.get("accepted") is True
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
                < 0.85
            )
        ]

        strong = [
            item
            for item in accepted
            if self._similarity(item) >= 0.85
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
    # REJECTED
    # ========================================================

    def _evaluate_rejected_matches(
        self,
    ) -> dict[str, Any]:

        rejected = [
            item
            for item in self.diagnostics
            if self._has_valid_appearance(item)
            and item.get("accepted") is not True
        ]

        appearance_mismatches = [
            item
            for item in rejected
            if item.get("reason")
            == "appearance_mismatch"
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
            == "compatible"
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
    # RISK
    # ========================================================

    def _evaluate_risky_matches(
        self,
    ) -> dict[str, Any]:

        accepted = [
            item
            for item in self.diagnostics
            if item.get("accepted") is True
            and self._has_valid_appearance(item)
        ]

        risky = []

        for item in accepted:

            similarity = self._similarity(item)

            if similarity < 0.75:

                risky.append(
                    (
                        item,
                        "low_appearance_similarity",
                    )
                )

                continue

            if (
                similarity < 0.80
                and self._weak_motion_or_spatial(item)
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
                    **self._compact_entry(item),
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

        identity_counts = {}

        for item in self.diagnostics:

            if not item.get("accepted"):
                continue

            identity_id = item.get(
                "identity_id"
            )

            if identity_id is None:
                continue

            identity_counts[identity_id] = (
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

        distribution = {}

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
    # BANDS
    # ========================================================

    def _similarity_bands(
        self,
        entries,
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

            similarity = self._similarity(item)

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
    def _has_valid_appearance(
        item: dict[str, Any],
    ) -> bool:

        similarity = item.get(
            "appearance_similarity"
        )

        try:
            similarity = float(similarity)
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
                float(item["spatial_distance"])
                if item.get("spatial_distance") is not None
                else None
            )

            motion = (
                float(item["motion_difference"])
                if item.get("motion_difference") is not None
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
        entries,
        limit: int = 20,
    ) -> list[dict[str, Any]]:

        return [
            self._compact_entry(item)
            for item in entries[:limit]
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

        if accepted_avg < 0.75:

            return (
                "WEAK: accepted appearance similarity "
                "is low; review the appearance threshold "
                "and embedding quality."
            )

        return (
            "MONITOR: appearance matching is "
            "functional but requires more validation."
        )