"""
Appearance embedding evaluation.

This module evaluates whether appearance embeddings provide
useful similarity information between ByteTrack fragments.

It does NOT modify global identity assignments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from ai_engine.appearance.embedder import AppearanceEmbedder
from ai_engine.schemas.track import BoundingBox, Track


@dataclass
class TrackAppearance:
    """
    Appearance representation for one ByteTrack fragment.
    """

    track_id: int
    class_name: str
    embedding: np.ndarray
    frame_index: int


class AppearanceEvaluator:
    """
    Extract representative appearance embeddings from tracks.
    """

    def __init__(
        self,
        embedder: AppearanceEmbedder | None = None,
        samples_per_track: int = 3,
    ) -> None:

        if samples_per_track <= 0:
            raise ValueError(
                "samples_per_track must be positive."
            )

        self.embedder = (
            embedder
            if embedder is not None
            else AppearanceEmbedder()
        )

        self.samples_per_track = int(
            samples_per_track
        )

    # ======================================================
    # TRACK EMBEDDING
    # ======================================================

    def embed_track(
        self,
        video: cv2.VideoCapture,
        track: Track,
    ) -> TrackAppearance | None:
        """
        Extract a representative appearance embedding
        from a ByteTrack fragment.

        Several observations are sampled and their embeddings
        are averaged.
        """

        observations = self._select_observations(
            track
        )

        embeddings = []

        for observation in observations:

            frame = self._read_frame(
                video,
                observation.frame_index,
            )

            if frame is None:
                continue

            embedding = self.embedder.embed_crop(
                frame,
                observation.bbox,
            )

            if embedding is None:
                continue

            embeddings.append(
                np.asarray(
                    embedding,
                    dtype=np.float32,
                )
            )

        if not embeddings:
            return None

        # --------------------------------------------------
        # Average multiple observations.
        # --------------------------------------------------

        matrix = np.stack(
            embeddings,
            axis=0,
        )

        representative = np.mean(
            matrix,
            axis=0,
        )

        norm = np.linalg.norm(
            representative
        )

        if norm <= 0:
            return None

        representative /= norm

        return TrackAppearance(
            track_id=track.local_id,
            class_name=track.class_name,
            embedding=representative,
            frame_index=observations[0].frame_index,
        )

    # ======================================================
    # OBSERVATION SAMPLING
    # ======================================================

    def _select_observations(
        self,
        track: Track,
    ):
        """
        Select evenly distributed observations.

        This avoids extracting embeddings from every frame.
        """

        observations = track.observations

        if not observations:
            return []

        if len(observations) <= self.samples_per_track:
            return observations

        indices = np.linspace(
            0,
            len(observations) - 1,
            self.samples_per_track,
            dtype=int,
        )

        return [
            observations[index]
            for index in indices
        ]

    # ======================================================
    # VIDEO ACCESS
    # ======================================================

    @staticmethod
    def _read_frame(
        video: cv2.VideoCapture,
        frame_index: int,
    ) -> np.ndarray | None:
        """
        Read a specific frame.
        """

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        ok, frame = video.read()

        if not ok:
            return None

        return frame


# ==========================================================
# SIMILARITY
# ==========================================================

def cosine_similarity(
    a: Iterable[float],
    b: Iterable[float],
) -> float:
    """
    Calculate cosine similarity between two embeddings.
    """

    vector_a = np.asarray(
        a,
        dtype=np.float32,
    )

    vector_b = np.asarray(
        b,
        dtype=np.float32,
    )

    norm_a = np.linalg.norm(
        vector_a
    )

    norm_b = np.linalg.norm(
        vector_b
    )

    if norm_a <= 0 or norm_b <= 0:
        return 0.0

    return float(
        np.dot(
            vector_a,
            vector_b,
        )
        / (
            norm_a
            * norm_b
        )
    )