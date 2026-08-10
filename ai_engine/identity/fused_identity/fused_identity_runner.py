"""
Fused Identity Pipeline Runner.

Pipeline:

    Video
      ↓
    YOLOv11 + ByteTrack
      ↓
    Track history
      ↓
    OSNet ReID embeddings
      ↓
    FusedIdentityManager
      ↓
    Global identities

This module is an integration layer only.

It does not modify:

- ByteTrack
- GlobalIdentityManager
- AppearanceIdentityMatcher
- IdentityFusion
- FusedIdentityMatcher
- FusedIdentityManager
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import cv2
import numpy as np

from ai_engine.appearance.reid_embedder import (
    OSNetReIDEmbedder,
)

from ai_engine.identity.fused_identity.fused_identity_manager import (
    FusedIdentityManager,
)

from ai_engine.schemas.track import Track

from ai_engine.tracking.tracking_runner import (
    run_tracking,
)


# ============================================================
# ALLOWED CLASSES
# ============================================================

ALLOWED_CLASSES = {
    "player",
    "goalkeeper",
}


# ============================================================
# SAMPLE SELECTION
# ============================================================

def _sample_observations(
    observations: list,
    max_samples: int,
) -> list:
    """
    Select representative observations from one track.

    For three samples:

        first
        middle
        last

    For larger values, evenly spaced observations are used.
    """

    if not observations:
        return []

    if max_samples <= 0:
        return []

    count = len(observations)

    if count <= max_samples:
        return list(observations)

    if max_samples == 1:

        indices = [
            count // 2
        ]

    elif max_samples == 2:

        indices = [
            0,
            count - 1,
        ]

    elif max_samples == 3:

        indices = [
            0,
            count // 2,
            count - 1,
        ]

    else:

        raw_indices = np.linspace(
            0,
            count - 1,
            max_samples,
        )

        indices = [
            int(round(index))
            for index in raw_indices
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


# ============================================================
# SAMPLE INDEX
# ============================================================

def _build_sample_index(
    track_history: Dict[int, Track],
    max_samples_per_track: int,
    min_confidence: float,
) -> dict:
    """
    Build:

        frame_index -> [
            (track_id, observation)
        ]

    Only player and goalkeeper tracks are included.
    """

    samples = defaultdict(list)

    for track_id, track in track_history.items():

        if track.class_name not in ALLOWED_CLASSES:
            continue

        observations = [
            observation
            for observation in track.observations
            if observation.confidence >= min_confidence
        ]

        if not observations:
            continue

        selected = _sample_observations(
            observations,
            max_samples_per_track,
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


# ============================================================
# REPRESENTATIVE EMBEDDING
# ============================================================

def _build_representative_embedding(
    embeddings: list[np.ndarray],
) -> np.ndarray | None:
    """
    Build one normalized representative embedding
    from multiple OSNet embeddings.

    The representative is the normalized mean vector.
    """

    if not embeddings:
        return None

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

    representative = (
        representative / norm
    )

    return representative.astype(
        np.float32
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_fused_identity_pipeline(
    video_path: str,
    model_path: str,
    imgsz: int = 640,
    confidence: float = 0.25,
    max_samples_per_track: int = 3,
    min_confidence: float = 0.0,
    match_threshold: float = 0.75,
    top_k: int = 5,
    max_gap: int = 50,
    max_distance: float = 250.0,
) -> FusedIdentityManager:
    """
    Run the complete fused identity pipeline.

    Returns:
        FusedIdentityManager containing:

        - global identities
        - track -> identity mapping
        - fused appearance/temporal/spatial/class matching
    """

    print()
    print("=" * 70)
    print("FUSED GLOBAL IDENTITY PIPELINE")
    print("=" * 70)

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print("STEP 1: Running ByteTrack...")

    tracker = run_tracking(
        video_path=video_path,
        model_path=model_path,
        imgsz=imgsz,
        confidence=confidence,
    )

    track_history = (
        tracker.get_track_history()
    )

    print()
    print(
        f"Track history: "
        f"{len(track_history)} ByteTrack IDs"
    )

    # ========================================================
    # STEP 2
    # ========================================================

    print()
    print("STEP 2: Selecting appearance samples...")

    samples = _build_sample_index(
        track_history=track_history,
        max_samples_per_track=max_samples_per_track,
        min_confidence=min_confidence,
    )

    selected_observations = sum(
        len(items)
        for items in samples.values()
    )

    print(
        f"Selected observations: "
        f"{selected_observations}"
    )

    # ========================================================
    # STEP 3
    # ========================================================

    print()
    print("STEP 3: Loading OSNet ReID...")

    embedder = OSNetReIDEmbedder()

    print(
        f"Embedding dimension: "
        f"{embedder.embedding_dimension}"
    )

    print(
        f"Device: "
        f"{embedder.device}"
    )

    # ========================================================
    # STEP 4
    # ========================================================

    print()
    print("STEP 4: Extracting OSNet embeddings...")

    requested_frames = set(
        samples.keys()
    )

    track_embeddings_raw: Dict[
        int,
        list[np.ndarray],
    ] = defaultdict(list)

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_path}"
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

                    embedding = embedder.embed_crop(
                        frame,
                        observation.bbox,
                    )

                    if embedding is None:
                        continue

                    vector = np.asarray(
                        embedding,
                        dtype=np.float32,
                    )

                    track_embeddings_raw[
                        track_id
                    ].append(
                        vector
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

    # ========================================================
    # STEP 5
    # BUILD REPRESENTATIVE EMBEDDING PER TRACK
    # ========================================================

    print()
    print(
        "STEP 5: Building track-level "
        "OSNet representations..."
    )

    track_embeddings: Dict[
        int,
        np.ndarray,
    ] = {}

    for track_id, embeddings in (
        track_embeddings_raw.items()
    ):

        representative = (
            _build_representative_embedding(
                embeddings
            )
        )

        if representative is None:
            continue

        track_embeddings[
            track_id
        ] = representative

    print(
        f"Tracks with valid OSNet embeddings: "
        f"{len(track_embeddings)}"
    )

    # ========================================================
    # STEP 6
    # BUILD FUSED IDENTITIES
    # ========================================================

    print()
    print(
        "STEP 6: Building fused global identities..."
    )

    manager = FusedIdentityManager(
        match_threshold=match_threshold,
        top_k=top_k,
        max_gap=max_gap,
        max_distance=max_distance,
    )

    identities = manager.build_identities(
        tracks=track_history.values(),
        track_embeddings=track_embeddings,
    )
    manager.track_history = tracker.track_history
    manager.track_classes = tracker.track_classes
    manager.last_seen = tracker.last_seen

    # ========================================================
    # STEP 7
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FUSED IDENTITY SUMMARY")
    print("=" * 70)

    print(
        f"ByteTrack IDs: "
        f"{len(track_history)}"
    )

    print(
        f"Tracks with OSNet embeddings: "
        f"{len(track_embeddings)}"
    )

    print(
        f"Global identities: "
        f"{len(identities)}"
    )

    print()

    # --------------------------------------------------------
    # Track -> Global Identity
    # --------------------------------------------------------

    print(
        "Track → Global Identity:"
    )

    track_identity_map = (
        manager.get_track_identity_map()
    )

    for (
        track_id,
        identity_id,
    ) in sorted(
        track_identity_map.items()
    ):

        print(
            f"  Track {track_id:3d} "
            f"→ {identity_id}"
        )

    # --------------------------------------------------------
    # Global Identity Composition
    # --------------------------------------------------------

    print()

    print(
        "Global Identity composition:"
    )

    for identity_id, identity in sorted(
        identities.items()
    ):

        print(
            f"  {identity_id}: "
            f"{len(identity.source_track_ids)} tracks"
        )

    # --------------------------------------------------------
    # Match configuration
    # --------------------------------------------------------

    print()

    print(
        "Fusion configuration:"
    )

    print(
        f"  Match threshold: "
        f"{match_threshold}"
    )

    print(
        f"  Top-K candidates: "
        f"{top_k}"
    )

    print(
        f"  Max temporal gap: "
        f"{max_gap}"
    )

    print(
        f"  Max spatial distance: "
        f"{max_distance}"
    )

    print(
        f"  Fusion weights: "
        f"{manager.get_matcher().get_weights()}"
    )

    print()

    print("=" * 70)
    print("FUSED IDENTITY PIPELINE COMPLETE")
    print("=" * 70)

    return manager


# ============================================================
# ALIASES
# ============================================================

run_fused_identity = (
    run_fused_identity_pipeline
)
