"""
Appearance embedding gallery.

Appearance/ReID V1
------------------

Stores representative appearance embeddings and provides
cosine-similarity matching.

This module is intentionally independent from:

- ByteTrack
- IdentityMatcher
- GlobalIdentityManager
- Team assignment
- Jersey recognition

It will first be tested independently before being integrated
into global identity matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np


@dataclass
class GalleryEntry:
    """
    Appearance representation for one identity.
    """

    identity_id: str

    embeddings: list[np.ndarray] = field(
        default_factory=list
    )

    max_embeddings: int = 10

    def add(
        self,
        embedding: Iterable[float] | np.ndarray,
    ) -> None:
        """
        Add an embedding to the identity gallery.

        Only the most recent max_embeddings vectors are retained.
        """

        vector = _normalize(
            embedding
        )

        self.embeddings.append(
            vector
        )

        if len(self.embeddings) > self.max_embeddings:
            self.embeddings = self.embeddings[
                -self.max_embeddings:
            ]

    @property
    def representative(self) -> Optional[np.ndarray]:
        """
        Return the representative embedding.

        The representative is the normalized mean of all stored
        embeddings.
        """

        if not self.embeddings:
            return None

        matrix = np.stack(
            self.embeddings,
            axis=0,
        )

        mean_vector = np.mean(
            matrix,
            axis=0,
        )

        return _normalize(
            mean_vector
        )


class AppearanceGallery:
    """
    Collection of appearance embeddings indexed by identity ID.
    """

    def __init__(
        self,
        max_embeddings_per_identity: int = 10,
    ) -> None:

        if max_embeddings_per_identity <= 0:
            raise ValueError(
                "max_embeddings_per_identity must be positive."
            )

        self.max_embeddings_per_identity = int(
            max_embeddings_per_identity
        )

        self.entries: dict[
            str,
            GalleryEntry,
        ] = {}

    # ======================================================
    # ADD / UPDATE
    # ======================================================

    def add(
        self,
        identity_id: str,
        embedding: Iterable[float] | np.ndarray,
    ) -> None:
        """
        Add an embedding to an identity.
        """

        if identity_id not in self.entries:
            self.entries[identity_id] = GalleryEntry(
                identity_id=identity_id,
                max_embeddings=(
                    self.max_embeddings_per_identity
                ),
            )

        self.entries[
            identity_id
        ].add(
            embedding
        )

    def add_many(
        self,
        identity_id: str,
        embeddings: Iterable[
            Iterable[float] | np.ndarray
        ],
    ) -> None:
        """
        Add multiple embeddings to an identity.
        """

        for embedding in embeddings:
            self.add(
                identity_id,
                embedding,
            )

    # ======================================================
    # REMOVE
    # ======================================================

    def remove(
        self,
        identity_id: str,
    ) -> None:
        """
        Remove an identity from the gallery.
        """

        self.entries.pop(
            identity_id,
            None,
        )

    def clear(self) -> None:
        """
        Remove all gallery entries.
        """

        self.entries.clear()

    # ======================================================
    # SIMILARITY
    # ======================================================

    def similarity(
        self,
        identity_id: str,
        embedding: Iterable[float] | np.ndarray,
    ) -> float:
        """
        Compare an embedding against one identity.

        Returns:
            Cosine similarity in approximately [-1, 1].
        """

        entry = self.entries.get(
            identity_id
        )

        if entry is None:
            return 0.0

        representative = entry.representative

        if representative is None:
            return 0.0

        query = _normalize(
            embedding
        )

        return float(
            np.dot(
                representative,
                query,
            )
        )

    def nearest(
        self,
        embedding: Iterable[float] | np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Return the most similar identities.

        Results are sorted from highest similarity to lowest.
        """

        if top_k <= 0:
            return []

        query = _normalize(
            embedding
        )

        results = []

        for identity_id, entry in self.entries.items():

            representative = entry.representative

            if representative is None:
                continue

            score = float(
                np.dot(
                    representative,
                    query,
                )
            )

            results.append(
                (
                    identity_id,
                    score,
                )
            )

        results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return results[:top_k]

    # ======================================================
    # INFORMATION
    # ======================================================

    def __len__(self) -> int:
        return len(
            self.entries
        )

    def get_identity_count(self) -> int:
        return len(
            self.entries
        )

    def get_embedding_count(
        self,
        identity_id: str,
    ) -> int:
        entry = self.entries.get(
            identity_id
        )

        if entry is None:
            return 0

        return len(
            entry.embeddings
        )


# ==========================================================
# UTILITIES
# ==========================================================

def _normalize(
    embedding: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """
    Convert an embedding into a float32 L2-normalized vector.
    """

    vector = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if vector.ndim != 1:
        vector = vector.reshape(-1)

    norm = np.linalg.norm(
        vector
    )

    if norm <= 0:
        raise ValueError(
            "Cannot normalize a zero-length embedding."
        )

    return vector / norm