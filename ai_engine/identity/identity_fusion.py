
"""
Global identity evidence fusion.

Combines multiple signals for deciding whether a ByteTrack
fragment should belong to an existing Global Identity.

Signals:

    - appearance similarity
    - temporal consistency
    - spatial consistency
    - class consistency

This module only calculates evidence.

It does NOT:
    - modify GlobalIdentityManager
    - create identities
    - merge tracks
    - assign teams
    - perform jersey recognition

The purpose is to keep the scoring logic independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ============================================================
# RESULT
# ============================================================


@dataclass
class IdentityFusionResult:
    """
    Combined identity evidence.
    """

    score: float

    appearance_score: float
    temporal_score: float
    spatial_score: float
    class_score: float

    accepted: bool


# ============================================================
# FUSION
# ============================================================


class IdentityFusion:
    """
    Combine identity evidence into a single score.

    Default weights:

        appearance = 0.50
        temporal   = 0.20
        spatial    = 0.20
        class      = 0.10

    Appearance receives the largest weight because OSNet is the
    new identity signal.

    However, appearance alone cannot produce a match.
    """

    def __init__(
        self,
        appearance_weight: float = 0.50,
        temporal_weight: float = 0.20,
        spatial_weight: float = 0.20,
        class_weight: float = 0.10,
        acceptance_threshold: float = 0.70,
    ) -> None:

        weights = [
            appearance_weight,
            temporal_weight,
            spatial_weight,
            class_weight,
        ]

        if any(
            weight < 0
            for weight in weights
        ):
            raise ValueError(
                "Fusion weights cannot be negative."
            )

        total = sum(weights)

        if total <= 0:
            raise ValueError(
                "At least one fusion weight must be positive."
            )

        if not 0.0 <= acceptance_threshold <= 1.0:
            raise ValueError(
                "acceptance_threshold must be between 0 and 1."
            )

        # ----------------------------------------------------
        # Normalize weights.
        #
        # This allows callers to provide weights that don't
        # necessarily sum to exactly 1.
        # ----------------------------------------------------

        self.appearance_weight = (
            appearance_weight / total
        )

        self.temporal_weight = (
            temporal_weight / total
        )

        self.spatial_weight = (
            spatial_weight / total
        )

        self.class_weight = (
            class_weight / total
        )

        self.acceptance_threshold = float(
            acceptance_threshold
        )

    # ========================================================
    # SCORE
    # ========================================================

    def score(
        self,
        appearance_score: float,
        temporal_score: float,
        spatial_score: float,
        class_score: float,
    ) -> IdentityFusionResult:
        """
        Calculate the combined identity score.

        All input signals are expected to be normalized to:

            0.0 -> no evidence
            1.0 -> strong evidence
        """

        appearance_score = self._clamp(
            appearance_score
        )

        temporal_score = self._clamp(
            temporal_score
        )

        spatial_score = self._clamp(
            spatial_score
        )

        class_score = self._clamp(
            class_score
        )

        combined = (
            appearance_score
            * self.appearance_weight

            + temporal_score
            * self.temporal_weight

            + spatial_score
            * self.spatial_weight

            + class_score
            * self.class_weight
        )

        accepted = (
            combined >= self.acceptance_threshold
        )

        return IdentityFusionResult(
            score=float(combined),
            appearance_score=float(
                appearance_score
            ),
            temporal_score=float(
                temporal_score
            ),
            spatial_score=float(
                spatial_score
            ),
            class_score=float(
                class_score
            ),
            accepted=accepted,
        )

    # ========================================================
    # APPEARANCE ONLY
    # ========================================================

    def score_appearance_only(
        self,
        appearance_score: float,
    ) -> IdentityFusionResult:
        """
        Evaluate appearance when no additional evidence is
        available.

        This intentionally uses the normal fusion weights.

        Missing evidence is represented by zero rather than
        allowing appearance to automatically dominate.
        """

        return self.score(
            appearance_score=appearance_score,
            temporal_score=0.0,
            spatial_score=0.0,
            class_score=0.0,
        )

    # ========================================================
    # CLASS
    # ========================================================

    @staticmethod
    def class_consistency(
        first_class: Optional[str],
        second_class: Optional[str],
    ) -> float:
        """
        Return class consistency.

        Same class:
            1.0

        Different known classes:
            0.0

        Missing class information:
            0.5
        """

        if (
            first_class is None
            or second_class is None
        ):
            return 0.5

        return (
            1.0
            if first_class == second_class
            else 0.0
        )

    # ========================================================
    # TEMPORAL
    # ========================================================

    @staticmethod
    def temporal_consistency(
        previous_last_frame: Optional[int],
        current_first_frame: Optional[int],
        max_gap: int = 50,
    ) -> float:
        """
        Calculate temporal consistency.

        A small gap receives a high score.

        Example:

            gap = 0
                -> 1.0

            gap = max_gap
                -> 0.0
        """

        if (
            previous_last_frame is None
            or current_first_frame is None
        ):
            return 0.5

        gap = (
            current_first_frame
            - previous_last_frame
        )

        # Negative gaps indicate overlapping fragments.
        if gap <= 0:
            return 1.0

        if max_gap <= 0:
            return 0.0

        score = 1.0 - (
            gap / float(max_gap)
        )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    # ========================================================
    # SPATIAL
    # ========================================================

    @staticmethod
    def spatial_consistency(
        distance: Optional[float],
        max_distance: float = 250.0,
    ) -> float:
        """
        Calculate spatial consistency.

        Smaller spatial distance produces a higher score.
        """

        if distance is None:
            return 0.5

        if max_distance <= 0:
            return 0.0

        if distance <= 0:
            return 1.0

        if distance >= max_distance:
            return 0.0

        score = 1.0 - (
            distance / float(max_distance)
        )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    # ========================================================
    # CLAMP
    # ========================================================

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        """
        Clamp a score to [0, 1].
        """

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    def get_weights(self) -> dict[str, float]:
        """
        Return the normalized fusion weights.
        """

        return {
            "appearance": self.appearance_weight,
            "temporal": self.temporal_weight,
            "spatial": self.spatial_weight,
            "class": self.class_weight,
        }

