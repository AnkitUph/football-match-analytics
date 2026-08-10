"""
Appearance / ReID V1 evaluator.

This module evaluates whether appearance embeddings provide a
useful signal for reconnecting fragmented ByteTrack tracks.

It does NOT modify GlobalIdentityManager or IdentityMatcher.

Evaluation flow:

    Video
      ↓
    ByteTrack
      ↓
    Track history
      ↓
    Global Identity V1
      ↓
    Representative crops
      ↓
    Appearance embeddings
      ↓
    Track-level appearance representations
      ↓
    Same-identity vs different-identity similarity
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import cv2
import numpy as np

from ai_engine.appearance.embedder import AppearanceEmbedder
from ai_engine.identity.identity_manager import GlobalIdentityManager
from ai_engine.schemas.track import Track
from ai_engine.tracking.tracking_runner import run_tracking


# ============================================================
# RESULT STRUCTURES
# ============================================================


@dataclass
class TrackAppearance:
    """
    Appearance representation for one ByteTrack fragment.
    """

    track_id: int
    identity_id: str
    class_name: str
    embedding: np.ndarray
    sample_count: int


# ============================================================
# EVALUATOR
# ============================================================


class AppearanceEvaluator:
    """
    Evaluate appearance similarity between ByteTrack fragments.

    The evaluator:

        1. runs ByteTrack
        2. builds Global Identity V1
        3. samples observations from tracks
        4. extracts appearance embeddings
        5. creates one representative vector per track
        6. compares fragments belonging to the same GID
        7. compares fragments belonging to different GIDs
    """

    ALLOWED_CLASSES = {
        "player",
        "goalkeeper",
    }

    def __init__(
        self,
        video_path: str,
        model_path: str,
        imgsz: int = 640,
        confidence: float = 0.25,
        max_samples_per_track: int = 3,
        min_confidence: float = 0.0,
        minimum_match_score: float = 0.55,
        max_temporal_gap: int = 50,
        max_spatial_distance: float = 250.0,
        max_motion_difference: float = 80.0,
    ) -> None:

        if max_samples_per_track <= 0:
            raise ValueError(
                "max_samples_per_track must be positive."
            )

        self.video_path = video_path
        self.model_path = model_path

        self.imgsz = int(imgsz)
        self.confidence = float(confidence)

        self.max_samples_per_track = int(
            max_samples_per_track
        )

        self.min_confidence = float(
            min_confidence
        )

        self.minimum_match_score = float(
            minimum_match_score
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

        self.embedder = AppearanceEmbedder()

        self.tracker = None
        self.identity_manager = None

        self.identities = {}
        self.track_to_identity: Dict[int, str] = {}

        self.track_appearances: Dict[
            int,
            TrackAppearance,
        ] = {}

    # ========================================================
    # PUBLIC API
    # ========================================================

    def run(self) -> dict:
        """
        Run the complete appearance evaluation.
        """

        print()
        print("=" * 70)
        print("APPEARANCE / REID V1 EVALUATION")
        print("=" * 70)

        # ----------------------------------------------------
        # Step 1: Existing ByteTrack pipeline
        # ----------------------------------------------------

        self.tracker = run_tracking(
            video_path=self.video_path,
            model_path=self.model_path,
            imgsz=self.imgsz,
            confidence=self.confidence,
        )

        track_history = (
            self.tracker.get_track_history()
        )

        print()
        print(
            f"Track history: {len(track_history)} "
            "ByteTrack IDs"
        )

        # ----------------------------------------------------
        # Step 2: Rebuild the frozen Global Identity V1
        # ----------------------------------------------------

        self.identity_manager = GlobalIdentityManager(
            minimum_match_score=(
                self.minimum_match_score
            ),
        )

        self.identities = (
            self.identity_manager.build_identities(
                track_history.values()
            )
        )

        self._build_track_identity_map()

        print(
            f"Global identities: "
            f"{len(self.identities)}"
        )

        # ----------------------------------------------------
        # Step 3: Select representative observations
        # ----------------------------------------------------

        samples = self._build_sample_index(
            track_history
        )

        print(
            f"Selected observations: "
            f"{len(samples)}"
        )

        # ----------------------------------------------------
        # Step 4: Extract embeddings
        # ----------------------------------------------------

        self._extract_embeddings(
            samples
        )

        # ----------------------------------------------------
        # Step 5: Evaluate similarity
        # ----------------------------------------------------

        results = self._evaluate_similarity()

        self._print_summary(
            results
        )

        return results

    # ========================================================
    # IDENTITY MAPPING
    # ========================================================

    def _build_track_identity_map(self) -> None:
        """
        Build:

            ByteTrack local ID -> Global Identity ID
        """

        self.track_to_identity.clear()

        for identity_id, identity in self.identities.items():

            for track_id in identity.source_track_ids:

                self.track_to_identity[
                    track_id
                ] = identity_id

    # ========================================================
    # SAMPLING
    # ========================================================

    def _build_sample_index(
        self,
        track_history: Dict[int, Track],
    ) -> Dict[int, list]:
        """
        Build:

            frame_index -> observations to embed

        Only player and goalkeeper tracks are included.

        Observations are sampled approximately evenly through
        each track.
        """

        samples = defaultdict(list)

        for track_id, track in track_history.items():

            if track.class_name not in self.ALLOWED_CLASSES:
                continue

            observations = [
                observation
                for observation in track.observations
                if observation.confidence
                >= self.min_confidence
            ]

            if not observations:
                continue

            selected = self._sample_observations(
                observations
            )

            for observation in selected:

                samples[
                    observation.frame_index
                ].append(
                    (
                        track_id,
                        observation,
                    )
                )

        return dict(samples)

    def _sample_observations(
        self,
        observations: list,
    ) -> list:
        """
        Select representative observations.

        For three samples:

            first
            middle
            last

        Duplicate indices are removed for short tracks.
        """

        count = len(observations)

        if count <= self.max_samples_per_track:
            return list(observations)

        if self.max_samples_per_track == 1:
            indices = [count // 2]

        elif self.max_samples_per_track == 2:
            indices = [
                0,
                count - 1,
            ]

        else:
            indices = [
                0,
                count // 2,
                count - 1,
            ]

            # Support values > 3 in a simple evenly spaced way.
            if self.max_samples_per_track > 3:
                raw = np.linspace(
                    0,
                    count - 1,
                    self.max_samples_per_track,
                )

                indices = [
                    int(round(index))
                    for index in raw
                ]

        selected = []

        seen = set()

        for index in indices:

            if index in seen:
                continue

            seen.add(index)

            selected.append(
                observations[index]
            )

        return selected

    # ========================================================
    # EMBEDDING EXTRACTION
    # ========================================================

    def _extract_embeddings(
        self,
        samples: Dict[int, list],
    ) -> None:
        """
        Read the video sequentially and extract embeddings only
        for requested frame indices.

        This avoids repeatedly seeking through the video.
        """

        requested_frames = set(
            samples.keys()
        )

        if not requested_frames:
            print(
                "No appearance samples selected."
            )
            return

        print()
        print(
            f"Embedding {len(requested_frames)} "
            "video frames..."
        )

        cap = cv2.VideoCapture(
            self.video_path
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open video: "
                f"{self.video_path}"
            )

        frame_index = 0
        processed_frames = 0
        embeddings_created = 0

        try:

            while True:

                ok, frame = cap.read()

                if not ok:
                    break

                if frame_index in requested_frames:

                    processed_frames += 1

                    frame_samples = samples[
                        frame_index
                    ]

                    for track_id, observation in frame_samples:

                        embedding = (
                            self.embedder.embed_crop(
                                frame,
                                observation.bbox,
                            )
                        )

                        if embedding is None:
                            continue

                        vector = np.asarray(
                            embedding,
                            dtype=np.float32,
                        )

                        identity_id = (
                            self.track_to_identity.get(
                                track_id
                            )
                        )

                        if identity_id is None:
                            continue

                        self._store_embedding(
                            track_id=track_id,
                            identity_id=identity_id,
                            class_name=(
                                observation.class_name
                            ),
                            embedding=vector,
                        )

                        embeddings_created += 1

                frame_index += 1

        finally:
            cap.release()

        print(
            f"Frames processed for appearance: "
            f"{processed_frames}"
        )

        print(
            f"Embeddings created: "
            f"{embeddings_created}"
        )

    def _store_embedding(
        self,
        track_id: int,
        identity_id: str,
        class_name: str,
        embedding: np.ndarray,
    ) -> None:
        """
        Store one embedding.

        Multiple observations from the same track are averaged
        into one representative track embedding later.
        """

        existing = self.track_appearances.get(
            track_id
        )

        if existing is None:

            self.track_appearances[
                track_id
            ] = TrackAppearance(
                track_id=track_id,
                identity_id=identity_id,
                class_name=class_name,
                embedding=embedding,
                sample_count=1,
            )

            return

        # Running average.
        count = existing.sample_count

        averaged = (
            existing.embedding * count
            + embedding
        ) / float(count + 1)

        norm = np.linalg.norm(
            averaged
        )

        if norm > 0:
            averaged = averaged / norm

        existing.embedding = (
            averaged.astype(np.float32)
        )

        existing.sample_count += 1

    # ========================================================
    # SIMILARITY
    # ========================================================

    @staticmethod
    def _cosine_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Cosine similarity between normalized vectors.
        """

        a = np.asarray(
            a,
            dtype=np.float32,
        )

        b = np.asarray(
            b,
            dtype=np.float32,
        )

        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        if a_norm <= 0 or b_norm <= 0:
            return 0.0

        return float(
            np.dot(
                a / a_norm,
                b / b_norm,
            )
        )

    def _evaluate_similarity(self) -> dict:
        """
        Compare track-level appearance representations.

        Same-identity comparisons:

            two different ByteTrack fragments
            belonging to the same GID.

        Different-identity comparisons:

            fragments belonging to different GIDs.
        """

        appearances = list(
            self.track_appearances.values()
        )

        same_identity_scores = []
        different_identity_scores = []

        same_identity_pairs = 0
        different_identity_pairs = 0

        for index, first in enumerate(appearances):

            for second in appearances[index + 1:]:

                # Only compare same class.
                if first.class_name != second.class_name:
                    continue

                score = self._cosine_similarity(
                    first.embedding,
                    second.embedding,
                )

                if first.identity_id == second.identity_id:

                    # A track compared with itself is impossible
                    # because we iterate over unique track IDs.
                    same_identity_scores.append(
                        score
                    )

                    same_identity_pairs += 1

                else:

                    different_identity_scores.append(
                        score
                    )

                    different_identity_pairs += 1

        same_stats = self._score_statistics(
            same_identity_scores
        )

        different_stats = self._score_statistics(
            different_identity_scores
        )

        # ----------------------------------------------------
        # Track-level nearest different identity.
        # ----------------------------------------------------

        nearest_different_scores = []

        for first in appearances:

            best_score = None

            for second in appearances:

                if first.track_id == second.track_id:
                    continue

                if first.class_name != second.class_name:
                    continue

                if first.identity_id == second.identity_id:
                    continue

                score = self._cosine_similarity(
                    first.embedding,
                    second.embedding,
                )

                if (
                    best_score is None
                    or score > best_score
                ):
                    best_score = score

            if best_score is not None:
                nearest_different_scores.append(
                    best_score
                )

        nearest_different_stats = (
            self._score_statistics(
                nearest_different_scores
            )
        )

        # ----------------------------------------------------
        # Simple separation estimate.
        # ----------------------------------------------------

        separation = (
            same_stats["avg"]
            - different_stats["avg"]
            if same_identity_scores
            and different_identity_scores
            else 0.0
        )

        return {
            "tracks_with_embeddings": len(
                appearances
            ),
            "same_identity_pairs": (
                same_identity_pairs
            ),
            "different_identity_pairs": (
                different_identity_pairs
            ),
            "same_identity_similarity": same_stats,
            "different_identity_similarity": (
                different_stats
            ),
            "nearest_different_similarity": (
                nearest_different_stats
            ),
            "average_separation": separation,
        }

    # ========================================================
    # STATISTICS
    # ========================================================

    @staticmethod
    def _score_statistics(
        scores: Iterable[float],
    ) -> dict:
        scores = list(scores)

        if not scores:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
                "median": 0.0,
            }

        values = np.asarray(
            scores,
            dtype=np.float32,
        )

        return {
            "count": len(scores),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "avg": float(np.mean(values)),
            "median": float(np.median(values)),
        }

    # ========================================================
    # OUTPUT
    # ========================================================

    def _print_summary(
        self,
        results: dict,
    ) -> None:

        same = results[
            "same_identity_similarity"
        ]

        different = results[
            "different_identity_similarity"
        ]

        nearest = results[
            "nearest_different_similarity"
        ]

        print()
        print("=" * 70)
        print("APPEARANCE / REID V1 BASELINE")
        print("=" * 70)

        print(
            "Embedding model: ResNet18"
        )

        print(
            f"Embedding dimension: "
            f"{self.embedder.embedding_dimension}"
        )

        print(
            f"Device: "
            f"{self.embedder.device}"
        )

        print()

        print(
            f"Tracks with embeddings: "
            f"{results['tracks_with_embeddings']}"
        )

        print()

        print(
            "Same global identity:"
        )

        print(
            f"  Pairs:   {same['count']}"
        )

        print(
            f"  Min:     {same['min']:.4f}"
        )

        print(
            f"  Max:     {same['max']:.4f}"
        )

        print(
            f"  Average: {same['avg']:.4f}"
        )

        print(
            f"  Median:  {same['median']:.4f}"
        )

        print()

        print(
            "Different global identities:"
        )

        print(
            f"  Pairs:   {different['count']}"
        )

        print(
            f"  Min:     {different['min']:.4f}"
        )

        print(
            f"  Max:     {different['max']:.4f}"
        )

        print(
            f"  Average: {different['avg']:.4f}"
        )

        print(
            f"  Median:  {different['median']:.4f}"
        )

        print()

        print(
            "Nearest different-identity similarity:"
        )

        print(
            f"  Min:     {nearest['min']:.4f}"
        )

        print(
            f"  Max:     {nearest['max']:.4f}"
        )

        print(
            f"  Average: {nearest['avg']:.4f}"
        )

        print(
            f"  Median:  {nearest['median']:.4f}"
        )

        print()

        print(
            f"Average separation: "
            f"{results['average_separation']:.4f}"
        )

        print(
            "=" * 70
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================


def run_appearance_evaluation(
    video_path: str,
    model_path: str,
    imgsz: int = 640,
    confidence: float = 0.25,
    max_samples_per_track: int = 3,
    min_confidence: float = 0.0,
) -> dict:
    """
    Convenience wrapper for AppearanceEvaluator.
    """

    evaluator = AppearanceEvaluator(
        video_path=video_path,
        model_path=model_path,
        imgsz=imgsz,
        confidence=confidence,
        max_samples_per_track=(
            max_samples_per_track
        ),
        min_confidence=min_confidence,
    )

    return evaluator.run()
