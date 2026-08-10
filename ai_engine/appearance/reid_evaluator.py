
"""
OSNet ReID V2 evaluator.

This module evaluates whether OSNet appearance embeddings provide
a useful signal for reconnecting fragmented ByteTrack tracks.

It does NOT modify GlobalIdentityManager or IdentityMatcher.

Evaluation flow:

    Video
      ↓
    ByteTrack
      ↓
    Global Identity V1
      ↓
    Representative observations
      ↓
    OSNet ReID embeddings
      ↓
    Track-level appearance representations
      ↓
    Same-identity vs different-identity similarity
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable

import cv2
import numpy as np

from ai_engine.appearance.reid_embedder import (
    OSNetReIDEmbedder,
)
from ai_engine.identity.identity_manager import (
    GlobalIdentityManager,
)
from ai_engine.schemas.track import Track
from ai_engine.tracking.tracking_runner import (
    run_tracking,
)


# ============================================================
# RESULT STRUCTURES
# ============================================================


@dataclass
class TrackReIDAppearance:
    """
    Appearance representation for one ByteTrack fragment.

    The embedding is the normalized representative vector
    obtained by averaging the sampled OSNet embeddings from
    this track.
    """

    track_id: int
    identity_id: str
    class_name: str
    embedding: np.ndarray
    sample_count: int


# ============================================================
# EVALUATOR
# ============================================================


class OSNetReIDEvaluator:
    """
    Evaluate OSNet ReID similarity between ByteTrack fragments.

    The evaluator:

        1. runs ByteTrack
        2. builds Global Identity V1
        3. samples observations from tracks
        4. extracts OSNet embeddings
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

        # ----------------------------------------------------
        # OSNet ReID embedder
        # ----------------------------------------------------

        self.embedder = OSNetReIDEmbedder()

        # ----------------------------------------------------
        # Runtime state
        # ----------------------------------------------------

        self.tracker = None
        self.identity_manager = None

        self.identities = {}

        # ByteTrack ID -> Global Identity ID
        self.track_to_identity: Dict[
            int,
            str,
        ] = {}

        # ByteTrack ID -> TrackReIDAppearance
        self.track_appearances: Dict[
            int,
            TrackReIDAppearance,
        ] = {}

    # ========================================================
    # PUBLIC API
    # ========================================================

    def run(self) -> dict:
        """
        Run the complete OSNet ReID evaluation.
        """

        print()
        print("=" * 70)
        print("OSNET ReID V2 EVALUATION")
        print("=" * 70)

        # ----------------------------------------------------
        # Step 1: Existing ByteTrack pipeline
        # ----------------------------------------------------

        print()
        print("STEP 1: Running ByteTrack...")

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
            f"Track history: "
            f"{len(track_history)} ByteTrack IDs"
        )

        # ----------------------------------------------------
        # Step 2: Build frozen Global Identity V1
        # ----------------------------------------------------

        print()
        print("STEP 2: Building Global Identity V1...")

        self.identity_manager = (
            GlobalIdentityManager(
                minimum_match_score=(
                    self.minimum_match_score
                ),
            )
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

        print()
        print(
            "STEP 3: Selecting appearance samples..."
        )

        samples = self._build_sample_index(
            track_history
        )

        print(
            f"Selected observations: "
            f"{sum(len(v) for v in samples.values())}"
        )

        # ----------------------------------------------------
        # Step 4: Extract OSNet embeddings
        # ----------------------------------------------------

        print()
        print(
            "STEP 4: Extracting OSNet embeddings..."
        )

        self._extract_embeddings(
            samples
        )

        # ----------------------------------------------------
        # Step 5: Evaluate similarity
        # ----------------------------------------------------

        print()
        print(
            "STEP 5: Evaluating appearance similarity..."
        )

        results = self._evaluate_similarity()

        # ----------------------------------------------------
        # Step 6: Print results
        # ----------------------------------------------------

        self._print_summary(
            results
        )

        return results

    # ========================================================
    # IDENTITY MAPPING
    # ========================================================

    def _build_track_identity_map(
        self,
    ) -> None:
        """
        Build:

            ByteTrack local ID
                ->
            Global Identity ID
        """

        self.track_to_identity.clear()

        for identity_id, identity in (
            self.identities.items()
        ):

            for track_id in (
                identity.source_track_ids
            ):

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

            frame_index ->
                observations to embed

        Only player and goalkeeper tracks are included.

        Observations are sampled approximately evenly
        through each track.
        """

        samples = defaultdict(list)

        for track_id, track in (
            track_history.items()
        ):

            # ------------------------------------------------
            # Only person-like football entities
            # ------------------------------------------------

            if (
                track.class_name
                not in self.ALLOWED_CLASSES
            ):
                continue

            observations = [
                observation
                for observation in track.observations
                if observation.confidence
                >= self.min_confidence
            ]

            if not observations:
                continue

            selected = (
                self._sample_observations(
                    observations
                )
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

        For values > 3, evenly spaced samples are selected.
        """

        count = len(observations)

        if count <= self.max_samples_per_track:
            return list(observations)

        if self.max_samples_per_track == 1:

            indices = [
                count // 2
            ]

        elif self.max_samples_per_track == 2:

            indices = [
                0,
                count - 1,
            ]

        else:

            if self.max_samples_per_track == 3:

                indices = [
                    0,
                    count // 2,
                    count - 1,
                ]

            else:

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
        Read the video sequentially and extract OSNet
        embeddings only for requested frame indices.

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
                "Unable to open video: "
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

                    for (
                        track_id,
                        observation,
                    ) in frame_samples:

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
            f"Frames processed for OSNet: "
            f"{processed_frames}"
        )

        print(
            f"OSNet embeddings created: "
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
        Store one OSNet embedding.

        Multiple observations from the same ByteTrack
        fragment are averaged into one representative
        embedding.
        """

        existing = (
            self.track_appearances.get(
                track_id
            )
        )

        if existing is None:

            # Ensure normalized representation.
            norm = np.linalg.norm(
                embedding
            )

            if norm > 0:

                embedding = (
                    embedding / norm
                )

            self.track_appearances[
                track_id
            ] = TrackReIDAppearance(
                track_id=track_id,
                identity_id=identity_id,
                class_name=class_name,
                embedding=embedding.astype(
                    np.float32
                ),
                sample_count=1,
            )

            return

        # ----------------------------------------------------
        # Running average
        # ----------------------------------------------------

        count = existing.sample_count

        averaged = (
            existing.embedding * count
            + embedding
        ) / float(count + 1)

        norm = np.linalg.norm(
            averaged
        )

        if norm > 0:

            averaged = (
                averaged / norm
            )

        existing.embedding = (
            averaged.astype(
                np.float32
            )
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
        Calculate cosine similarity.

        Embeddings are normalized before comparison,
        making the result approximately [-1, 1].
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

    # ========================================================
    # SIMILARITY EVALUATION
    # ========================================================

    def _evaluate_similarity(
        self,
    ) -> dict:
        """
        Compare track-level OSNet representations.

        Same-identity comparisons:

            Different ByteTrack fragments
            belonging to the same Global Identity.

        Different-identity comparisons:

            Fragments belonging to different Global
            Identities.
        """

        appearances = list(
            self.track_appearances.values()
        )

        same_identity_scores = []
        different_identity_scores = []

        same_identity_pairs = 0
        different_identity_pairs = 0

        # ----------------------------------------------------
        # Pairwise comparison
        # ----------------------------------------------------

        for index, first in enumerate(
            appearances
        ):

            for second in appearances[
                index + 1:
            ]:

                # Only compare the same object class.
                if (
                    first.class_name
                    != second.class_name
                ):
                    continue

                score = (
                    self._cosine_similarity(
                        first.embedding,
                        second.embedding,
                    )
                )

                if (
                    first.identity_id
                    == second.identity_id
                ):

                    same_identity_scores.append(
                        score
                    )

                    same_identity_pairs += 1

                else:

                    different_identity_scores.append(
                        score
                    )

                    different_identity_pairs += 1

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        same_stats = (
            self._score_statistics(
                same_identity_scores
            )
        )

        different_stats = (
            self._score_statistics(
                different_identity_scores
            )
        )

        # ----------------------------------------------------
        # Nearest different identity
        #
        # For every track, find the most similar track
        # belonging to another Global Identity.
        # ----------------------------------------------------

        nearest_different_scores = []

        for first in appearances:

            best_score = None

            for second in appearances:

                if (
                    first.track_id
                    == second.track_id
                ):
                    continue

                if (
                    first.class_name
                    != second.class_name
                ):
                    continue

                if (
                    first.identity_id
                    == second.identity_id
                ):
                    continue

                score = (
                    self._cosine_similarity(
                        first.embedding,
                        second.embedding,
                    )
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
        # Separation
        # ----------------------------------------------------

        if (
            same_identity_scores
            and different_identity_scores
        ):

            separation = (
                same_stats["avg"]
                - different_stats["avg"]
            )

        else:

            separation = 0.0

        # ----------------------------------------------------
        # Threshold diagnostics
        # ----------------------------------------------------

        threshold_results = (
            self._evaluate_thresholds(
                same_identity_scores,
                different_identity_scores,
            )
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
            "same_identity_similarity": (
                same_stats
            ),
            "different_identity_similarity": (
                different_stats
            ),
            "nearest_different_similarity": (
                nearest_different_stats
            ),
            "average_separation": (
                separation
            ),
            "threshold_analysis": (
                threshold_results
            ),
        }

    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    @staticmethod
    def _evaluate_thresholds(
        same_scores: Iterable[float],
        different_scores: Iterable[float],
    ) -> dict:
        """
        Evaluate several candidate cosine-similarity
        thresholds.

        This is diagnostic only.

        It does NOT change GlobalIdentityManager.
        """

        same_scores = list(
            same_scores
        )

        different_scores = list(
            different_scores
        )

        thresholds = [
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.92,
            0.94,
            0.95,
            0.96,
            0.97,
            0.98,
        ]

        results = []

        for threshold in thresholds:

            true_positive = sum(
                score >= threshold
                for score in same_scores
            )

            false_negative = (
                len(same_scores)
                - true_positive
            )

            false_positive = sum(
                score >= threshold
                for score in different_scores
            )

            true_negative = (
                len(different_scores)
                - false_positive
            )

            same_recall = (
                true_positive
                / len(same_scores)
                if same_scores
                else 0.0
            )

            different_rejection = (
                true_negative
                / len(different_scores)
                if different_scores
                else 0.0
            )

            precision_denominator = (
                true_positive
                + false_positive
            )

            precision = (
                true_positive
                / precision_denominator
                if precision_denominator
                else 0.0
            )

            results.append(
                {
                    "threshold": threshold,
                    "true_positive": (
                        true_positive
                    ),
                    "false_negative": (
                        false_negative
                    ),
                    "false_positive": (
                        false_positive
                    ),
                    "true_negative": (
                        true_negative
                    ),
                    "same_identity_recall": (
                        same_recall
                    ),
                    "different_identity_rejection": (
                        different_rejection
                    ),
                    "precision": precision,
                }
            )

        return results

    # ========================================================
    # STATISTICS
    # ========================================================

    @staticmethod
    def _score_statistics(
        scores: Iterable[float],
    ) -> dict:
        """
        Calculate descriptive statistics for similarity
        scores.
        """

        scores = list(
            scores
        )

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
            "min": float(
                np.min(values)
            ),
            "max": float(
                np.max(values)
            ),
            "avg": float(
                np.mean(values)
            ),
            "median": float(
                np.median(values)
            ),
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

        thresholds = results[
            "threshold_analysis"
        ]

        print()
        print("=" * 70)
        print("OSNET ReID V2 BASELINE")
        print("=" * 70)

        print(
            "Embedding model: OSNet x1.0"
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

        # ----------------------------------------------------
        # Same identity
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Different identity
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Nearest different identity
        # ----------------------------------------------------

        print(
            "Nearest different-identity similarity:"
        )

        print(
            f"  Count:   {nearest['count']}"
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

        # ----------------------------------------------------
        # Separation
        # ----------------------------------------------------

        print(
            f"Average separation: "
            f"{results['average_separation']:.4f}"
        )

        print()

        # ----------------------------------------------------
        # Threshold analysis
        # ----------------------------------------------------

        print(
            "Threshold analysis:"
        )

        print(
            "  Threshold | "
            "Same Recall | "
            "Different Rejection | "
            "Precision"
        )

        print(
            "  " + "-" * 60
        )

        for result in thresholds:

            print(
                f"  {result['threshold']:9.2f} | "
                f"{result['same_identity_recall']:11.3f} | "
                f"{result['different_identity_rejection']:20.3f} | "
                f"{result['precision']:9.3f}"
            )

        print()

        print("=" * 70)
        print("OSNET ReID V2 EVALUATION COMPLETE")
        print("=" * 70)


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================


def run_reid_evaluation(
    video_path: str,
    model_path: str,
    imgsz: int = 640,
    confidence: float = 0.25,
    max_samples_per_track: int = 3,
    min_confidence: float = 0.0,
    minimum_match_score: float = 0.55,
) -> dict:
    """
    Convenience wrapper for OSNetReIDEvaluator.
    """

    evaluator = OSNetReIDEvaluator(
        video_path=video_path,
        model_path=model_path,
        imgsz=imgsz,
        confidence=confidence,
        max_samples_per_track=(
            max_samples_per_track
        ),
        min_confidence=min_confidence,
        minimum_match_score=(
            minimum_match_score
        ),
    )

    return evaluator.run()

