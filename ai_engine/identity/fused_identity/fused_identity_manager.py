
"""
Fused Global Identity Manager.

Identity V2.

Builds global identities from ByteTrack trajectories using:

    - Appearance / OSNet ReID
    - Temporal consistency
    - Spatial consistency
    - Class consistency
    - IdentityFusion

This module is intentionally separate from the original
GlobalIdentityManager so Identity V1 remains available as a
baseline for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np

from ai_engine.identity.fused_identity.fused_identity_matcher import (
    FusedIdentityMatchResult,
    FusedIdentityMatcher,
)
from ai_engine.schemas.track import Track


# ============================================================
# GLOBAL IDENTITY
# ============================================================


@dataclass
class FusedGlobalIdentity:
    """
    One global identity composed of one or more ByteTrack
    fragments.
    """

    identity_id: str

    class_name: str

    source_track_ids: list[int] = field(
        default_factory=list
    )

    embeddings: list[np.ndarray] = field(
        default_factory=list
    )

    first_frame: Optional[int] = None

    last_frame: Optional[int] = None

    last_x: Optional[float] = None

    last_y: Optional[float] = None

    observation_count: int = 0

    # --------------------------------------------------------
    # Representative embedding
    # --------------------------------------------------------

    @property
    def representative_embedding(
        self,
    ) -> Optional[np.ndarray]:

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

        norm = np.linalg.norm(
            mean_vector
        )

        if norm <= 0:
            return None

        return (
            mean_vector / norm
        ).astype(np.float32)

    # --------------------------------------------------------
    # Add track
    # --------------------------------------------------------

    def add_track(
        self,
        track: Track,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a ByteTrack trajectory to this identity.
        """

        if track.local_id not in self.source_track_ids:

            self.source_track_ids.append(
                track.local_id
            )

        self.observation_count += len(
            track.observations
        )

        if track.observations:

            first = track.observations[0]

            last = track.observations[-1]

            if self.first_frame is None:

                self.first_frame = (
                    first.frame_index
                )

            else:

                self.first_frame = min(
                    self.first_frame,
                    first.frame_index,
                )

            if self.last_frame is None:

                self.last_frame = (
                    last.frame_index
                )

            else:

                self.last_frame = max(
                    self.last_frame,
                    last.frame_index,
                )

            self.last_x = (
                last.bbox.center_x
                if hasattr(
                    last.bbox,
                    "center_x",
                )
                else (
                    last.bbox.x1
                    + last.bbox.x2
                )
                / 2.0
            )

            self.last_y = (
                last.bbox.center_y
                if hasattr(
                    last.bbox,
                    "center_y",
                )
                else (
                    last.bbox.y1
                    + last.bbox.y2
                )
                / 2.0
            )

        if embedding is not None:

            vector = np.asarray(
                embedding,
                dtype=np.float32,
            )

            norm = np.linalg.norm(
                vector
            )

            if norm > 0:

                vector = vector / norm

                self.embeddings.append(
                    vector
                )


# ============================================================
# MANAGER
# ============================================================


class FusedIdentityManager:
    """
    Identity V2 manager.

    Processes ByteTrack trajectories and attempts to reconnect
    fragmented tracks into global identities.
    """

    ALLOWED_CLASSES = {
        "player",
        "goalkeeper",
    }

    def __init__(
        self,
        match_threshold: float = 0.75,
        top_k: int = 5,
        max_gap: int = 50,
        max_distance: float = 250.0,
        max_identity_embeddings: int = 10,
    ) -> None:

        if max_gap < 0:

            raise ValueError(
                "max_gap must be >= 0."
            )

        if max_distance <= 0:

            raise ValueError(
                "max_distance must be positive."
            )

        if max_identity_embeddings <= 0:

            raise ValueError(
                "max_identity_embeddings must be positive."
            )

        self.match_threshold = float(
            match_threshold
        )

        self.top_k = int(top_k)

        self.max_gap = int(
            max_gap
        )

        self.max_distance = float(
            max_distance
        )

        self.max_identity_embeddings = int(
            max_identity_embeddings
        )

        self.matcher = FusedIdentityMatcher(
            match_threshold=(
                self.match_threshold
            ),
            top_k=self.top_k,
        )

        self.identities: Dict[
            str,
            FusedGlobalIdentity,
        ] = {}

        self.track_to_identity: Dict[
            int,
            str,
        ] = {}

        self.next_identity_number = 1

    # ========================================================
    # BUILD IDENTITIES
    # ========================================================

    def build_identities(
        self,
        tracks: Iterable[Track],
        track_embeddings: Optional[
            Dict[int, np.ndarray]
        ] = None,
    ) -> Dict[
        str,
        FusedGlobalIdentity,
    ]:
        """
        Build global identities from ByteTrack tracks.

        Tracks should normally be ordered by first frame.
        """

        self.identities.clear()

        self.track_to_identity.clear()

        self.matcher.clear()

        self.next_identity_number = 1

        track_embeddings = (
            track_embeddings
            if track_embeddings is not None
            else {}
        )

        sorted_tracks = sorted(
            tracks,
            key=self._track_start_frame,
        )

        for track in sorted_tracks:

            if not track.observations:

                continue

            if (
                track.class_name
                not in self.ALLOWED_CLASSES
            ):

                continue

            embedding = (
                track_embeddings.get(
                    track.local_id
                )
            )

            self._assign_track(
                track,
                embedding,
            )

        return self.identities

    # ========================================================
    # ASSIGN TRACK
    # ========================================================

    def _assign_track(
        self,
        track: Track,
        embedding: Optional[np.ndarray],
    ) -> None:
        """
        Assign one ByteTrack fragment to an existing identity
        or create a new identity.
        """

        if embedding is None:

            self._create_identity(
                track,
                embedding=None,
            )

            return

        candidate = self._find_best_identity(
            track,
            embedding,
        )

        if candidate is None:

            self._create_identity(
                track,
                embedding=embedding,
            )

            return

        identity_id, result = candidate

        if not result.matched:

            self._create_identity(
                track,
                embedding=embedding,
            )

            return

        self._attach_track(
            identity_id=identity_id,
            track=track,
            embedding=embedding,
        )

    # ========================================================
    # FIND BEST IDENTITY
    # ========================================================

    def _find_best_identity(
        self,
        track: Track,
        embedding: np.ndarray,
    ) -> Optional[
        tuple[
            str,
            FusedIdentityMatchResult,
        ]
    ]:
        """
        Compare a track against existing identities.
        """

        if not self.identities:

            return None

        first_observation = (
            track.observations[0]
        )

        current_frame = (
            first_observation.frame_index
        )

        current_x, current_y = (
            self._observation_center(
                first_observation
            )
        )

        identity_classes = {
            identity_id: identity.class_name
            for identity_id, identity
            in self.identities.items()
        }

        best_identity = None

        best_result = None

        # ----------------------------------------------------
        # Appearance candidates
        # ----------------------------------------------------

        candidates = self.matcher.candidates(
            embedding
        )

        # ----------------------------------------------------
        # If no appearance candidates exist, no match.
        # ----------------------------------------------------

        if not candidates:

            return None

        for identity_id, _ in candidates[
            : self.top_k
        ]:

            identity = self.identities.get(
                identity_id
            )

            if identity is None:

                continue

            if identity.last_frame is None:

                continue

            # ------------------------------------------------
            # Temporal gating
            # ------------------------------------------------

            gap = (
                current_frame
                - identity.last_frame
            )

            if gap < 0:

                continue

            if gap > self.max_gap:

                continue

            # ------------------------------------------------
            # Spatial distance
            # ------------------------------------------------

            if (
                identity.last_x is None
                or identity.last_y is None
            ):

                distance = None

            else:

                distance = float(
                    np.sqrt(
                        (
                            current_x
                            - identity.last_x
                        )
                        ** 2
                        +
                        (
                            current_y
                            - identity.last_y
                        )
                        ** 2
                    )
                )

                # Hard spatial gate.
                if distance > self.max_distance:

                    continue

            # ------------------------------------------------
            # Fused score
            # ------------------------------------------------

            result = self.matcher.match(
                embedding,
                previous_last_frame=(
                    identity.last_frame
                ),
                current_first_frame=(
                    current_frame
                ),
                distance=distance,
                max_gap=self.max_gap,
                max_distance=self.max_distance,
                current_class=track.class_name,
                identity_classes=identity_classes,
            )

            if (
                result.identity_id
                != identity_id
            ):

                continue

            if best_result is None:

                best_result = result
                best_identity = identity_id

            elif result.score > best_result.score:

                best_result = result
                best_identity = identity_id

        if (
            best_identity is None
            or best_result is None
        ):

            return None

        return (
            best_identity,
            best_result,
        )

    # ========================================================
    # CREATE
    # ========================================================

    def _create_identity(
        self,
        track: Track,
        embedding: Optional[np.ndarray],
    ) -> str:
        """
        Create a new global identity.
        """

        identity_id = (
            f"GID_{self.next_identity_number:04d}"
        )

        self.next_identity_number += 1

        identity = FusedGlobalIdentity(
            identity_id=identity_id,
            class_name=track.class_name,
        )

        identity.add_track(
            track,
            embedding,
        )

        self.identities[
            identity_id
        ] = identity

        self.track_to_identity[
            track.local_id
        ] = identity_id

        # ----------------------------------------------------
        # Add appearance representation to matcher.
        # ----------------------------------------------------

        if embedding is not None:

            self.matcher.add_identity_embedding(
                identity_id,
                embedding,
            )

        return identity_id

    # ========================================================
    # ATTACH
    # ========================================================

    def _attach_track(
        self,
        identity_id: str,
        track: Track,
        embedding: np.ndarray,
    ) -> None:
        """
        Attach a ByteTrack fragment to an existing identity.
        """

        identity = self.identities[
            identity_id
        ]

        identity.add_track(
            track,
            embedding,
        )

        self.track_to_identity[
            track.local_id
        ] = identity_id

        # ----------------------------------------------------
        # Keep only recent identity embeddings.
        # ----------------------------------------------------

        if (
            len(identity.embeddings)
            > self.max_identity_embeddings
        ):

            identity.embeddings = (
                identity.embeddings[
                    -self.max_identity_embeddings:
                ]
            )

        # ----------------------------------------------------
        # Rebuild matcher representation.
        # ----------------------------------------------------

        self.matcher.remove_identity(
            identity_id
        )

        for vector in identity.embeddings:

            self.matcher.add_identity_embedding(
                identity_id,
                vector,
            )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _track_start_frame(
        track: Track,
    ) -> int:

        if not track.observations:

            return 0

        return track.observations[
            0
        ].frame_index

    @staticmethod
    def _observation_center(
        observation,
    ) -> tuple[float, float]:

        bbox = observation.bbox

        if hasattr(
            bbox,
            "center_x",
        ):

            x = float(
                bbox.center_x
            )

        else:

            x = (
                float(bbox.x1)
                + float(bbox.x2)
            ) / 2.0

        if hasattr(
            bbox,
            "center_y",
        ):

            y = float(
                bbox.center_y
            )

        else:

            y = (
                float(bbox.y1)
                + float(bbox.y2)
            ) / 2.0

        return x, y

    # ========================================================
    # QUERY API
    # ========================================================

    def get_identity(
        self,
        identity_id: str,
    ) -> Optional[FusedGlobalIdentity]:

        return self.identities.get(
            identity_id
        )

    def get_identity_count(self) -> int:

        return len(
            self.identities
        )

    def get_track_identity(
        self,
        track_id: int,
    ) -> Optional[str]:

        return self.track_to_identity.get(
            track_id
        )

    def get_track_identity_map(
        self,
    ) -> Dict[int, str]:

        return dict(
            self.track_to_identity
        )

    def get_identities(
        self,
    ) -> Dict[
        str,
        FusedGlobalIdentity,
    ]:

        return self.identities

    def get_matcher(
        self,
    ) -> FusedIdentityMatcher:

        return self.matcher

