
"""
Fused Identity Matcher.

Identity V2 foundation.

Combines multiple identity signals:

    1. Appearance / ReID similarity
    2. Temporal consistency
    3. Spatial consistency
    4. Class consistency

The matcher is intentionally independent from:

    - GlobalIdentityManager
    - ByteTrack
    - video processing
    - jersey recognition
    - team assignment

It can therefore be tested independently before being integrated
into the production identity pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from ai_engine.identity.appearance_identity_matcher import (
    AppearanceIdentityMatcher,
)
from ai_engine.identity.identity_fusion import (
    IdentityFusion,
)


# ============================================================
# RESULT
# ============================================================


@dataclass
class FusedIdentityMatchResult:
    """
    Result of a fused identity comparison.
    """

    matched: bool

    identity_id: Optional[str]

    score: float

    appearance_score: float

    temporal_score: float

    spatial_score: float

    class_score: float


# ============================================================
# FUSED IDENTITY MATCHER
# ============================================================


class FusedIdentityMatcher:
    """
    Combine appearance and geometric/temporal identity signals.

    The matcher performs:

        candidate retrieval
                ↓
        appearance similarity
                ↓
        temporal consistency
                ↓
        spatial consistency
                ↓
        class consistency
                ↓
        weighted fusion
                ↓
        final identity decision
    """

    def __init__(
        self,
        match_threshold: float = 0.75,
        top_k: int = 5,
        fusion: Optional[IdentityFusion] = None,
    ) -> None:

        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError(
                "match_threshold must be between 0 and 1."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        self.match_threshold = float(
            match_threshold
        )

        self.top_k = int(top_k)

        # ----------------------------------------------------
        # Appearance candidate matcher
        # ----------------------------------------------------

        self.appearance_matcher = (
            AppearanceIdentityMatcher(
                match_threshold=match_threshold,
                top_k=top_k,
            )
        )

        # ----------------------------------------------------
        # Identity fusion
        # ----------------------------------------------------

        self.fusion = (
            fusion
            if fusion is not None
            else IdentityFusion()
        )

    # ========================================================
    # IDENTITY MANAGEMENT
    # ========================================================

    def add_identity_embedding(
        self,
        identity_id: str,
        embedding: Iterable[float] | np.ndarray,
    ) -> None:
        """
        Add an appearance embedding for an identity.
        """

        self.appearance_matcher.add_identity_embedding(
            identity_id,
            embedding,
        )

    def add_identity_embeddings(
        self,
        identity_id: str,
        embeddings: Iterable[
            Iterable[float] | np.ndarray
        ],
    ) -> None:
        """
        Add multiple embeddings for an identity.
        """

        for embedding in embeddings:

            self.add_identity_embedding(
                identity_id,
                embedding,
            )

    def remove_identity(
        self,
        identity_id: str,
    ) -> None:
        """
        Remove an identity from the matcher.
        """

        self.appearance_matcher.remove_identity(
            identity_id
        )

    def clear(self) -> None:
        """
        Remove all identities.
        """

        self.appearance_matcher.clear()

    def identity_count(self) -> int:
        """
        Return number of identities.
        """

        return self.appearance_matcher.identity_count()

    # ========================================================
    # MATCHING
    # ========================================================

    def match(
        self,
        embedding: Iterable[float] | np.ndarray,
        *,
        previous_last_frame: Optional[int] = None,
        current_first_frame: Optional[int] = None,
        distance: Optional[float] = None,
        max_gap: int = 50,
        max_distance: float = 250.0,
        current_class: Optional[str] = None,
        identity_classes: Optional[
            dict[str, str]
        ] = None,
    ) -> FusedIdentityMatchResult:
        """
        Match a new track fragment against known identities.

        Args:
            embedding:
                Appearance/ReID embedding of the new fragment.

            previous_last_frame:
                Last frame of the candidate identity.

            current_first_frame:
                First frame of the new fragment.

            distance:
                Spatial distance between the previous identity
                position and the new fragment position.

            max_gap:
                Maximum temporal gap used for normalization.

            max_distance:
                Maximum spatial distance used for normalization.

            current_class:
                Class of the current fragment.

            identity_classes:
                Optional mapping:

                    identity_id -> class_name

                Used for class consistency.

        Returns:
            FusedIdentityMatchResult.
        """

        # ----------------------------------------------------
        # Find appearance candidates first.
        # ----------------------------------------------------

        candidates = self.appearance_matcher.candidates(
            embedding
        )

        if not candidates:
            return FusedIdentityMatchResult(
                matched=False,
                identity_id=None,
                score=0.0,
                appearance_score=0.0,
                temporal_score=0.0,
                spatial_score=0.0,
                class_score=0.0,
            )

        best_result: Optional[
            FusedIdentityMatchResult
        ] = None

        # ----------------------------------------------------
        # Evaluate top appearance candidates.
        # ----------------------------------------------------

        for identity_id, appearance_score in candidates[
            : self.top_k
        ]:

            # -----------------------------------------------
            # Temporal consistency
            # -----------------------------------------------

            if (
                previous_last_frame is not None
                and current_first_frame is not None
            ):

                temporal_score = (
                    IdentityFusion.temporal_consistency(
                        previous_last_frame=(
                            previous_last_frame
                        ),
                        current_first_frame=(
                            current_first_frame
                        ),
                        max_gap=max_gap,
                    )
                )

            else:

                # No temporal information.
                #
                # Neutral value rather than rejecting the
                # candidate.
                temporal_score = 1.0

            # -----------------------------------------------
            # Spatial consistency
            # -----------------------------------------------

            if distance is not None:

                spatial_score = (
                    IdentityFusion.spatial_consistency(
                        distance=distance,
                        max_distance=max_distance,
                    )
                )

            else:

                spatial_score = 1.0

            # -----------------------------------------------
            # Class consistency
            # -----------------------------------------------

            if (
                current_class is not None
                and identity_classes is not None
            ):

                candidate_class = (
                    identity_classes.get(
                        identity_id
                    )
                )

                if candidate_class is None:

                    class_score = 0.0

                else:

                    class_score = (
                        IdentityFusion.class_consistency(
                            current_class,
                            candidate_class,
                        )
                    )

            else:

                # No class information.
                neutral_class_score = 1.0
                class_score = neutral_class_score

            # -----------------------------------------------
            # Fusion
            # -----------------------------------------------

            fusion_result = self.fusion.score(
                appearance_score=appearance_score,
                temporal_score=temporal_score,
                spatial_score=spatial_score,
                class_score=class_score,
            )

            result = FusedIdentityMatchResult(
                matched=fusion_result.accepted,
                identity_id=identity_id,
                score=fusion_result.score,
                appearance_score=appearance_score,
                temporal_score=temporal_score,
                spatial_score=spatial_score,
                class_score=class_score,
            )

            # -----------------------------------------------
            # Keep highest fused score.
            # -----------------------------------------------

            if (
                best_result is None
                or result.score > best_result.score
            ):

                best_result = result

        # ----------------------------------------------------
        # No candidate result should be impossible, but keep
        # the method defensive.
        # ----------------------------------------------------

        if best_result is None:

            return FusedIdentityMatchResult(
                matched=False,
                identity_id=None,
                score=0.0,
                appearance_score=0.0,
                temporal_score=0.0,
                spatial_score=0.0,
                class_score=0.0,
            )

        return best_result

    # ========================================================
    # CANDIDATES
    # ========================================================

    def candidates(
        self,
        embedding: Iterable[float] | np.ndarray,
    ) -> list[tuple[str, float]]:
        """
        Return appearance candidates.

        Results are sorted from highest appearance similarity
        to lowest.
        """

        return self.appearance_matcher.candidates(
            embedding
        )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def get_weights(self) -> dict[str, float]:
        """
        Return current fusion weights.
        """

        return self.fusion.get_weights()

    def get_match_threshold(self) -> float:
        """
        Return configured fusion match threshold.
        """

        return self.match_threshold

