"""
Appearance-aware identity matcher.

This module uses OSNet ReID embeddings to find the most likely
existing Global Identity for a ByteTrack fragment.

Important:
Appearance is treated as supporting evidence only.

It does NOT:
- modify GlobalIdentityManager
- perform team assignment
- perform jersey recognition
- perform homography
- make final identity decisions by itself

Pipeline:

    ByteTrack fragment
          ↓
    OSNet embedding
          ↓
    AppearanceGallery
          ↓
    Candidate GIDs
          ↓
    similarity score
          ↓
    conservative match / no match
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from ai_engine.appearance.gallery import AppearanceGallery


# ============================================================
# RESULT
# ============================================================


@dataclass
class AppearanceMatch:
    """
    Result of an appearance-based identity search.
    """

    identity_id: Optional[str]
    similarity: float
    matched: bool


# ============================================================
# MATCHER
# ============================================================


class AppearanceIdentityMatcher:
    """
    Match a ReID embedding against known global identities.

    Appearance is intentionally conservative.

    A candidate is accepted only when:

        similarity >= match_threshold

    Otherwise:

        identity_id = None
        matched = False
    """

    def __init__(
        self,
        gallery: Optional[AppearanceGallery] = None,
        match_threshold: float = 0.90,
        top_k: int = 5,
    ) -> None:

        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError(
                "match_threshold must be between 0 and 1."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        self.gallery = (
            gallery
            if gallery is not None
            else AppearanceGallery()
        )

        self.match_threshold = float(
            match_threshold
        )

        self.top_k = int(top_k)

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self) -> None:
        """
        Remove all registered identity embeddings.
        """

        self.gallery.clear()

    # ========================================================
    # REMOVE IDENTITY
    # ========================================================

    def remove_identity(
        self,
        identity_id: str,
    ) -> None:
        """
        Remove all appearance embeddings belonging to
        one global identity.
        """

        self.gallery.remove(
            identity_id
        )

    # ========================================================
    # ADD IDENTITY
    # ========================================================

    def add_identity_embedding(
        self,
        identity_id: str,
        embedding: Iterable[float] | np.ndarray,
    ) -> None:
        """
        Add one OSNet embedding to an identity.
        """

        self.gallery.add(
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
        Add multiple embeddings to an identity.
        """

        self.gallery.add_many(
            identity_id,
            embeddings,
        )

    # ========================================================
    # MATCH
    # ========================================================

    def match(
        self,
        embedding: Iterable[float] | np.ndarray,
    ) -> AppearanceMatch:
        """
        Find the best matching global identity.

        A match is accepted only when the best similarity
        is greater than or equal to match_threshold.
        """

        nearest = self.gallery.nearest(
            embedding,
            top_k=self.top_k,
        )

        if not nearest:

            return AppearanceMatch(
                identity_id=None,
                similarity=0.0,
                matched=False,
            )

        identity_id, similarity = nearest[0]

        matched = (
            similarity >= self.match_threshold
        )

        if not matched:
            identity_id = None

        return AppearanceMatch(
            identity_id=identity_id,
            similarity=float(similarity),
            matched=matched,
        )

    # ========================================================
    # CANDIDATES
    # ========================================================

    def candidates(
        self,
        embedding: Iterable[float] | np.ndarray,
    ) -> list[tuple[str, float]]:
        """
        Return the nearest identity candidates.

        Unlike match(), this returns candidates even when
        they do not cross the acceptance threshold.
        """

        return self.gallery.nearest(
            embedding,
            top_k=self.top_k,
        )

    # ========================================================
    # BEST SCORE
    # ========================================================

    def best_similarity(
        self,
        embedding: Iterable[float] | np.ndarray,
    ) -> float:
        """
        Return the best appearance similarity.

        Returns 0.0 when the gallery is empty.
        """

        nearest = self.gallery.nearest(
            embedding,
            top_k=1,
        )

        if not nearest:
            return 0.0

        return float(
            nearest[0][1]
        )

    # ========================================================
    # DIRECT IDENTITY SIMILARITY
    # ========================================================

    def similarity(
        self,
        identity_id: str,
        embedding: Iterable[float] | np.ndarray,
    ) -> float:
        """
        Return similarity between an embedding and a
        specific global identity.
        """

        return float(
            self.gallery.similarity(
                identity_id,
                embedding,
            )
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    def identity_count(self) -> int:
        """
        Number of identities currently represented
        in the gallery.
        """

        return self.gallery.get_identity_count()

    def embedding_count(
        self,
        identity_id: str,
    ) -> int:
        """
        Number of appearance embeddings stored for
        one identity.
        """

        return self.gallery.get_embedding_count(
            identity_id
        )

    def get_gallery(self) -> AppearanceGallery:
        """
        Return the underlying appearance gallery.
        """

        return self.gallery

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def get_match_threshold(self) -> float:
        """
        Return the current appearance match threshold.
        """

        return self.match_threshold

    def get_top_k(self) -> int:
        """
        Return the configured candidate count.
        """

        return self.top_k

    def set_match_threshold(
        self,
        threshold: float,
    ) -> None:
        """
        Update the appearance match threshold.
        """

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        self.match_threshold = float(
            threshold
        )