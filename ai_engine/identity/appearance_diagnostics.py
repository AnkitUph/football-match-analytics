"""
Appearance diagnostics and validation.

Analyzes appearance-aware global identity matching
without changing the matching algorithm.

Provides:

- accepted appearance matches
- rejected appearance matches
- similarity distributions
- per-identity appearance statistics
- low-confidence appearance matches
- summary statistics
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Iterable, Optional


class AppearanceDiagnostics:
    """
    Collect and analyze appearance-based identity matches.
    """

    def __init__(
        self,
        low_similarity_threshold: float = 0.70,
        strong_similarity_threshold: float = 0.85,
    ) -> None:

        self.low_similarity_threshold = float(
            low_similarity_threshold
        )

        self.strong_similarity_threshold = float(
            strong_similarity_threshold
        )

        self.accepted_matches = []
        self.rejected_matches = []

    # ========================================================
    # RECORD
    # ========================================================

    def record(
        self,
        track_id: int,
        identity_id: str,
        appearance_similarity: float,
        score: float,
        accepted: bool,
        reason: str,
        temporal_gap: int = 0,
        spatial_distance: float = 0.0,
        motion_difference: float = 0.0,
    ) -> None:
        """
        Record one identity matching decision.

        Appearance similarity < 0 means appearance
        information was unavailable.
        """

        if appearance_similarity < 0:
            return

        entry = {
            "track_id": track_id,
            "identity_id": identity_id,
            "appearance_similarity": float(
                appearance_similarity
            ),
            "score": float(score),
            "accepted": bool(accepted),
            "reason": reason,
            "temporal_gap": int(
                temporal_gap
            ),
            "spatial_distance": float(
                spatial_distance
            ),
            "motion_difference": float(
                motion_difference
            ),
        }

        if accepted:
            self.accepted_matches.append(
                entry
            )
        else:
            self.rejected_matches.append(
                entry
            )

    # ========================================================
    # ALL MATCHES
    # ========================================================

    def get_all_matches(self) -> list[dict]:
        return (
            self.accepted_matches
            + self.rejected_matches
        )

    # ========================================================
    # ACCEPTED
    # ========================================================

    def get_accepted_matches(self) -> list[dict]:
        return list(
            self.accepted_matches
        )

    # ========================================================
    # REJECTED
    # ========================================================

    def get_rejected_matches(self) -> list[dict]:
        return list(
            self.rejected_matches
        )

    # ========================================================
    # LOW SIMILARITY
    # ========================================================

    def get_low_similarity_matches(
        self,
        threshold: Optional[float] = None,
    ) -> list[dict]:

        if threshold is None:
            threshold = (
                self.low_similarity_threshold
            )

        return [
            match
            for match in self.get_all_matches()
            if match[
                "appearance_similarity"
            ] < threshold
        ]

    # ========================================================
    # STRONG MATCHES
    # ========================================================

    def get_strong_matches(
        self,
        threshold: Optional[float] = None,
    ) -> list[dict]:

        if threshold is None:
            threshold = (
                self.strong_similarity_threshold
            )

        return [
            match
            for match in self.get_accepted_matches()
            if match[
                "appearance_similarity"
            ] >= threshold
        ]

    # ========================================================
    # PER IDENTITY
    # ========================================================

    def get_identity_statistics(
        self,
    ) -> dict:

        grouped = defaultdict(list)

        for match in self.get_accepted_matches():

            grouped[
                match["identity_id"]
            ].append(match)

        statistics = {}

        for identity_id, matches in grouped.items():

            similarities = [
                match[
                    "appearance_similarity"
                ]
                for match in matches
            ]

            statistics[identity_id] = {
                "matches": len(matches),

                "min_similarity": min(
                    similarities
                ),

                "max_similarity": max(
                    similarities
                ),

                "average_similarity": mean(
                    similarities
                ),

                "median_similarity": median(
                    similarities
                ),

                "strong_matches": sum(
                    similarity
                    >= self.strong_similarity_threshold
                    for similarity in similarities
                ),

                "weak_matches": sum(
                    similarity
                    < self.low_similarity_threshold
                    for similarity in similarities
                ),
            }

        return statistics

    # ========================================================
    # SUMMARY
    # ========================================================

    def get_summary(self) -> dict:

        accepted = (
            self.get_accepted_matches()
        )

        rejected = (
            self.get_rejected_matches()
        )

        all_matches = (
            accepted + rejected
        )

        similarities = [
            match[
                "appearance_similarity"
            ]
            for match in all_matches
        ]

        accepted_similarities = [
            match[
                "appearance_similarity"
            ]
            for match in accepted
        ]

        rejected_similarities = [
            match[
                "appearance_similarity"
            ]
            for match in rejected
        ]

        if similarities:

            similarity_min = min(
                similarities
            )

            similarity_max = max(
                similarities
            )

            similarity_avg = mean(
                similarities
            )

            similarity_median = median(
                similarities
            )

        else:

            similarity_min = 0.0
            similarity_max = 0.0
            similarity_avg = 0.0
            similarity_median = 0.0

        if accepted_similarities:

            accepted_avg = mean(
                accepted_similarities
            )

        else:

            accepted_avg = 0.0

        if rejected_similarities:

            rejected_avg = mean(
                rejected_similarities
            )

        else:

            rejected_avg = 0.0

        strong_matches = sum(
            similarity
            >= self.strong_similarity_threshold
            for similarity in accepted_similarities
        )

        weak_matches = sum(
            similarity
            < self.low_similarity_threshold
            for similarity in accepted_similarities
        )

        return {
            "total_appearance_matches": len(
                all_matches
            ),

            "accepted_matches": len(
                accepted
            ),

            "rejected_matches": len(
                rejected
            ),

            "similarity": {
                "min": similarity_min,
                "max": similarity_max,
                "average": similarity_avg,
                "median": similarity_median,
            },

            "accepted_similarity": {
                "average": accepted_avg,
            },

            "rejected_similarity": {
                "average": rejected_avg,
            },

            "strong_matches": strong_matches,

            "weak_matches": weak_matches,

            "low_similarity_threshold": (
                self.low_similarity_threshold
            ),

            "strong_similarity_threshold": (
                self.strong_similarity_threshold
            ),
        }

    # ========================================================
    # PRINT REPORT
    # ========================================================

    def print_report(self) -> None:

        summary = self.get_summary()

        print()
        print("=" * 70)
        print("APPEARANCE DIAGNOSTICS")
        print("=" * 70)

        print(
            "Total appearance matches:",
            summary[
                "total_appearance_matches"
            ],
        )

        print(
            "Accepted:",
            summary[
                "accepted_matches"
            ],
        )

        print(
            "Rejected:",
            summary[
                "rejected_matches"
            ],
        )

        print()

        similarity = summary[
            "similarity"
        ]

        print(
            "Similarity minimum:",
            f"{similarity['min']:.4f}",
        )

        print(
            "Similarity maximum:",
            f"{similarity['max']:.4f}",
        )

        print(
            "Similarity average:",
            f"{similarity['average']:.4f}",
        )

        print(
            "Similarity median:",
            f"{similarity['median']:.4f}",
        )

        print()

        accepted_similarity = (
            summary[
                "accepted_similarity"
            ]
        )

        rejected_similarity = (
            summary[
                "rejected_similarity"
            ]
        )

        print(
            "Accepted average:",
            f"{accepted_similarity['average']:.4f}",
        )

        print(
            "Rejected average:",
            f"{rejected_similarity['average']:.4f}",
        )

        print()

        print(
            "Strong matches:",
            summary[
                "strong_matches"
            ],
        )

        print(
            "Weak matches:",
            summary[
                "weak_matches"
            ],
        )

        print("=" * 70)

    # ========================================================
    # PRINT REJECTED
    # ========================================================

    def print_rejected_matches(
        self,
    ) -> None:

        rejected = sorted(
            self.get_rejected_matches(),
            key=lambda item: item[
                "appearance_similarity"
            ],
        )

        print()
        print("=" * 70)
        print("APPEARANCE REJECTIONS")
        print("=" * 70)

        if not rejected:

            print(
                "No appearance-based rejections."
            )

            print("=" * 70)

            return

        for match in rejected:

            print(
                f"Track {match['track_id']:3d} "
                f"-> {match['identity_id']} | "
                f"appearance="
                f"{match['appearance_similarity']:.4f} | "
                f"score="
                f"{match['score']:.4f} | "
                f"reason="
                f"{match['reason']}"
            )

        print("=" * 70)

    # ========================================================
    # PRINT IDENTITY STATISTICS
    # ========================================================

    def print_identity_statistics(
        self,
    ) -> None:

        statistics = (
            self.get_identity_statistics()
        )

        print()
        print("=" * 70)
        print("PER-IDENTITY APPEARANCE STATISTICS")
        print("=" * 70)

        if not statistics:

            print(
                "No accepted appearance matches."
            )

            print("=" * 70)

            return

        ordered = sorted(
            statistics.items(),
            key=lambda item: (
                item[1][
                    "average_similarity"
                ]
            ),
        )

        for identity_id, stats in ordered:

            print(
                f"{identity_id} | "
                f"matches={stats['matches']:3d} | "
                f"min="
                f"{stats['min_similarity']:.4f} | "
                f"avg="
                f"{stats['average_similarity']:.4f} | "
                f"max="
                f"{stats['max_similarity']:.4f} | "
                f"strong="
                f"{stats['strong_matches']:3d}"
            )

        print("=" * 70)