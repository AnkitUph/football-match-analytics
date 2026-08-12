"""
Appearance matcher for football player re-identification.

Responsibilities:

- Compare appearance embeddings.
- Calculate cosine similarity.
- Match TrackObservation objects using appearance.
- Match Track objects using representative embeddings.
- Handle missing embeddings safely.
- Provide configurable similarity thresholds.

This module intentionally does NOT perform:

- global identity creation
- ByteTrack
- team assignment
- jersey-number recognition
- spatial matching
- temporal matching
- homography
- metrics

Pipeline:

    TrackObservation
          ↓
    appearance_embedding
          ↓
    AppearanceMatcher
          ↓
    similarity score
          ↓
    GlobalIdentityManagerV2
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from ai_engine.schemas.track import (
    Track,
    TrackObservation,
)

logger = logging.getLogger(__name__)


class AppearanceMatcher:
    """
    Appearance-based re-identification matcher.

    Uses cosine similarity between OSNet embeddings.

    Embeddings produced by AppearanceExtractor are already
    L2-normalized, but this class normalizes them again so that
    it remains safe if a different extractor is used later.
    """

    def __init__(
        self,
        minimum_similarity: float = 0.70,
        strong_similarity: float = 0.85,
        minimum_embedding_dimension: int = 128,
    ):
        """
        Args:
            minimum_similarity:
                Minimum cosine similarity considered a possible
                appearance match.

            strong_similarity:
                Similarity considered a strong appearance match.

            minimum_embedding_dimension:
                Reject embeddings shorter than this dimension.
        """

        if not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError(
                "minimum_similarity must be between 0 and 1"
            )

        if not 0.0 <= strong_similarity <= 1.0:
            raise ValueError(
                "strong_similarity must be between 0 and 1"
            )

        if strong_similarity < minimum_similarity:
            raise ValueError(
                "strong_similarity must be >= minimum_similarity"
            )

        self.minimum_similarity = float(
            minimum_similarity
        )

        self.strong_similarity = float(
            strong_similarity
        )

        self.minimum_embedding_dimension = int(
            minimum_embedding_dimension
        )

        # -----------------------------------------------------
        # Diagnostics
        # -----------------------------------------------------

        self.comparisons = 0
        self.valid_comparisons = 0

        self.accepted_matches = 0
        self.rejected_matches = 0

        # Number of comparisons where one or both embeddings
        # were explicitly missing (None).
        self.missing_embeddings = 0

        # Number of comparisons where embeddings existed but
        # were malformed, incompatible, non-finite, too small,
        # zero-norm, etc.
        self.invalid_embeddings = 0

        self.similarity_scores: list[float] = []

        logger.info(
            "AppearanceMatcher initialized: "
            "minimum_similarity=%.3f, "
            "strong_similarity=%.3f",
            self.minimum_similarity,
            self.strong_similarity,
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def compare_embeddings(
        self,
        embedding_a: Optional[Sequence[float]],
        embedding_b: Optional[Sequence[float]],
    ) -> Optional[float]:
        """
        Calculate cosine similarity between two embeddings.

        Returns:
            Float in [-1, 1], normally [0, 1] for OSNet
            embeddings.

            None if either embedding is missing or invalid.
        """

        self.comparisons += 1

        # -----------------------------------------------------
        # Explicitly distinguish missing embeddings from
        # malformed embeddings.
        # -----------------------------------------------------

        if embedding_a is None or embedding_b is None:
            self.missing_embeddings += 1
            return None

        vector_a = self._prepare_embedding(
            embedding_a
        )

        vector_b = self._prepare_embedding(
            embedding_b
        )

        if vector_a is None or vector_b is None:
            self.invalid_embeddings += 1

            return None

        # -----------------------------------------------------
        # Shape safety
        # -----------------------------------------------------

        if vector_a.shape != vector_b.shape:
            self.invalid_embeddings += 1

            logger.debug(
                "Embedding dimension mismatch: %s vs %s",
                vector_a.shape,
                vector_b.shape,
            )

            return None

        # -----------------------------------------------------
        # Norm safety
        #
        # _prepare_embedding() already normalizes vectors,
        # but retain this defensive check.
        # -----------------------------------------------------

        norm_a = np.linalg.norm(
            vector_a
        )

        norm_b = np.linalg.norm(
            vector_b
        )

        if norm_a <= 1e-12 or norm_b <= 1e-12:
            self.invalid_embeddings += 1

            return None

        # -----------------------------------------------------
        # Cosine similarity
        # -----------------------------------------------------

        similarity = float(
            np.dot(vector_a, vector_b)
            / (norm_a * norm_b)
        )

        # -----------------------------------------------------
        # Numerical safety
        # -----------------------------------------------------

        similarity = float(
            np.clip(
                similarity,
                -1.0,
                1.0,
            )
        )

        self.valid_comparisons += 1

        self.similarity_scores.append(
            similarity
        )

        return similarity

    def is_match(
        self,
        embedding_a: Optional[Sequence[float]],
        embedding_b: Optional[Sequence[float]],
    ) -> bool:
        """
        Determine whether two embeddings satisfy the
        minimum appearance similarity threshold.
        """

        similarity = self.compare_embeddings(
            embedding_a,
            embedding_b,
        )

        if similarity is None:
            return False

        if similarity >= self.minimum_similarity:
            self.accepted_matches += 1

            return True

        self.rejected_matches += 1

        return False

    def is_strong_match(
        self,
        embedding_a: Optional[Sequence[float]],
        embedding_b: Optional[Sequence[float]],
    ) -> bool:
        """
        Determine whether two embeddings are a strong
        appearance match.
        """

        similarity = self.compare_embeddings(
            embedding_a,
            embedding_b,
        )

        if similarity is None:
            return False

        return similarity >= self.strong_similarity

    # =========================================================
    # OBSERVATION MATCHING
    # =========================================================

    def compare_observations(
        self,
        observation_a: TrackObservation,
        observation_b: TrackObservation,
    ) -> Optional[float]:
        """
        Compare the appearance embeddings of two observations.
        """

        return self.compare_embeddings(
            observation_a.appearance_embedding,
            observation_b.appearance_embedding,
        )

    def observations_match(
        self,
        observation_a: TrackObservation,
        observation_b: TrackObservation,
    ) -> bool:
        """
        Determine whether two observations are an appearance match.
        """

        return self.is_match(
            observation_a.appearance_embedding,
            observation_b.appearance_embedding,
        )

    # =========================================================
    # TRACK MATCHING
    # =========================================================

    def get_track_embedding(
        self,
        track: Track,
    ) -> Optional[np.ndarray]:
        """
        Build a representative appearance embedding for a Track.

        Strategy:

            1. Collect all valid observation embeddings.
            2. Normalize each embedding.
            3. Calculate their mean.
            4. Normalize the resulting mean vector.

        This creates a stable representation of the track
        instead of relying on one frame.
        """

        embeddings = []

        for observation in track.observations:

            embedding = self._prepare_embedding(
                observation.appearance_embedding
            )

            if embedding is not None:
                embeddings.append(
                    embedding
                )

        if not embeddings:
            return None

        matrix = np.vstack(
            embeddings
        )

        mean_embedding = np.mean(
            matrix,
            axis=0,
        )

        norm = np.linalg.norm(
            mean_embedding
        )

        if norm <= 1e-12:
            return None

        mean_embedding = (
            mean_embedding / norm
        )

        return mean_embedding.astype(
            np.float32
        )

    def compare_tracks(
        self,
        track_a: Track,
        track_b: Track,
    ) -> Optional[float]:
        """
        Compare two tracks using representative
        appearance embeddings.
        """

        embedding_a = self.get_track_embedding(
            track_a
        )

        embedding_b = self.get_track_embedding(
            track_b
        )

        return self.compare_embeddings(
            embedding_a,
            embedding_b,
        )

    def tracks_match(
        self,
        track_a: Track,
        track_b: Track,
    ) -> bool:
        """
        Determine whether two tracks are an appearance match.
        """

        embedding_a = self.get_track_embedding(
            track_a
        )

        embedding_b = self.get_track_embedding(
            track_b
        )

        return self.is_match(
            embedding_a,
            embedding_b,
        )

    # =========================================================
    # BEST MATCH
    # =========================================================

    def find_best_match(
        self,
        query_embedding: Optional[Sequence[float]],
        candidate_embeddings: dict[
            str,
            Sequence[float],
        ],
        minimum_similarity: Optional[float] = None,
    ) -> Optional[tuple[str, float]]:
        """
        Find the most visually similar candidate.

        Args:
            query_embedding:
                Embedding being searched.

            candidate_embeddings:
                Mapping:

                    identity_id -> embedding

            minimum_similarity:
                Optional override of the default threshold.

        Returns:

            (identity_id, similarity)

        or:

            None

        if no candidate satisfies the threshold.
        """

        if query_embedding is None:
            return None

        threshold = (
            self.minimum_similarity
            if minimum_similarity is None
            else float(minimum_similarity)
        )

        best_identity_id = None
        best_similarity = -1.0

        for identity_id, candidate in (
            candidate_embeddings.items()
        ):

            similarity = self.compare_embeddings(
                query_embedding,
                candidate,
            )

            if similarity is None:
                continue

            if similarity > best_similarity:
                best_similarity = similarity
                best_identity_id = identity_id

        if (
            best_identity_id is None
            or best_similarity < threshold
        ):
            return None

        return (
            best_identity_id,
            best_similarity,
        )

    # =========================================================
    # EMBEDDING VALIDATION
    # =========================================================

    def _prepare_embedding(
        self,
        embedding: Optional[Sequence[float]],
    ) -> Optional[np.ndarray]:
        """
        Validate and normalize an embedding.

        Returns:
            Normalized float32 vector or None if invalid.
        """

        if embedding is None:
            return None

        try:
            vector = np.asarray(
                embedding,
                dtype=np.float32,
            )

        except (TypeError, ValueError):
            return None

        # -----------------------------------------------------
        # Must be a single vector.
        # -----------------------------------------------------

        if vector.ndim != 1:
            return None

        # -----------------------------------------------------
        # Minimum dimension safety.
        # -----------------------------------------------------

        if (
            len(vector)
            < self.minimum_embedding_dimension
        ):
            return None

        # -----------------------------------------------------
        # Reject NaN / Inf.
        # -----------------------------------------------------

        if not np.all(
            np.isfinite(vector)
        ):
            return None

        # -----------------------------------------------------
        # Reject zero / near-zero vectors.
        # -----------------------------------------------------

        norm = np.linalg.norm(
            vector
        )

        if norm <= 1e-12:
            return None

        # -----------------------------------------------------
        # L2 normalization.
        # -----------------------------------------------------

        vector = vector / norm

        return vector.astype(
            np.float32
        )

    # =========================================================
    # DIAGNOSTICS
    # =========================================================

    def get_statistics(
        self,
    ) -> dict:
        """
        Return appearance matching statistics.
        """

        if self.similarity_scores:

            minimum = min(
                self.similarity_scores
            )

            maximum = max(
                self.similarity_scores
            )

            average = (
                sum(self.similarity_scores)
                / len(self.similarity_scores)
            )

        else:

            minimum = 0.0
            maximum = 0.0
            average = 0.0

        return {
            "minimum_similarity": (
                self.minimum_similarity
            ),
            "strong_similarity": (
                self.strong_similarity
            ),
            "comparisons": (
                self.comparisons
            ),
            "valid_comparisons": (
                self.valid_comparisons
            ),
            "accepted_matches": (
                self.accepted_matches
            ),
            "rejected_matches": (
                self.rejected_matches
            ),
            "missing_embeddings": (
                self.missing_embeddings
            ),
            "invalid_embeddings": (
                self.invalid_embeddings
            ),
            "similarity": {
                "min": minimum,
                "max": maximum,
                "avg": average,
            },
        }

    def reset_statistics(
        self,
    ) -> None:
        """
        Reset matcher diagnostics without changing
        configuration.
        """

        self.comparisons = 0
        self.valid_comparisons = 0

        self.accepted_matches = 0
        self.rejected_matches = 0

        self.missing_embeddings = 0
        self.invalid_embeddings = 0

        self.similarity_scores.clear()