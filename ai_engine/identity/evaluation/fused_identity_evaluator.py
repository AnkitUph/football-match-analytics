"""
Fused Identity Evaluator.

Evaluates the output produced by FusedIdentityManager.

This module does NOT modify:

- ByteTrack
- GlobalIdentityManager
- AppearanceIdentityMatcher
- FusedIdentityMatcher
- FusedIdentityManager

It only analyzes the final identity assignments.

Pipeline:

    FusedIdentityManager
            |
            v
    FusedIdentityEvaluator
            |
            +--> identity statistics
            +--> merge statistics
            +--> singleton statistics
            +--> temporal consistency
            +--> suspicious identity detection
            +--> evaluation report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from ai_engine.identity.fused_identity.fused_identity_manager import (
    FusedGlobalIdentity,
    FusedIdentityManager,
)
from ai_engine.schemas.track import Track


# ============================================================
# EVALUATION RESULT
# ============================================================


@dataclass
class IdentityEvaluationResult:
    """
    Complete evaluation result for fused identities.
    """

    # --------------------------------------------------------
    # Basic counts
    # --------------------------------------------------------

    total_tracks: int = 0

    tracks_with_embeddings: int = 0

    total_global_identities: int = 0

    # --------------------------------------------------------
    # Identity composition
    # --------------------------------------------------------

    singleton_identities: int = 0

    merged_identities: int = 0

    max_tracks_per_identity: int = 0

    average_tracks_per_identity: float = 0.0

    # --------------------------------------------------------
    # Merge statistics
    # --------------------------------------------------------

    total_merged_tracks: int = 0

    merge_ratio: float = 0.0

    # --------------------------------------------------------
    # Temporal consistency
    # --------------------------------------------------------

    identities_with_temporal_overlap: int = 0

    temporal_overlap_pairs: int = 0

    # --------------------------------------------------------
    # Suspicious identities
    # --------------------------------------------------------

    suspicious_identity_count: int = 0

    suspicious_identity_ids: List[str] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Per identity statistics
    # --------------------------------------------------------

    identity_track_counts: Dict[str, int] = field(
        default_factory=dict
    )

    identity_track_ids: Dict[str, List[int]] = field(
        default_factory=dict
    )


# ============================================================
# EVALUATOR
# ============================================================


class FusedIdentityEvaluator:
    """
    Evaluate a FusedIdentityManager.

    The evaluator is intentionally read-only.

    It does not change the manager or identity assignments.
    """

    def __init__(
        self,
        manager: FusedIdentityManager,
        track_history: Optional[Dict[int, Track]] = None,
    ) -> None:

        if manager is None:
            raise ValueError(
                "manager must not be None."
            )

        self.manager = manager

        self.track_history = (
            track_history
            if track_history is not None
            else {}
        )

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    def get_total_tracks(self) -> int:
        """
        Return the total number of ByteTrack tracks.
        """

        if self.track_history:
            return len(self.track_history)

        return len(
            self.manager.get_track_identity_map()
        )

    # ========================================================

    def get_tracks_with_embeddings(
        self,
    ) -> int:
        """
        Return the number of tracks that received
        an identity assignment.

        This is generally the number of tracks that
        successfully reached the fused identity stage.
        """

        return len(
            self.manager.get_track_identity_map()
        )

    # ========================================================

    def get_identity_count(self) -> int:
        """
        Return the number of final global identities.
        """

        return self.manager.get_identity_count()

    # ========================================================
    # IDENTITY COMPOSITION
    # ========================================================

    def get_identity_track_counts(
        self,
    ) -> Dict[str, int]:
        """
        Return:

            GID -> number of ByteTrack fragments
        """

        result: Dict[str, int] = {}

        identities = self.manager.get_identities()

        for identity_id, identity in identities.items():

            result[identity_id] = len(
                identity.source_track_ids
            )

        return result

    # ========================================================

    def get_identity_track_ids(
        self,
    ) -> Dict[str, List[int]]:
        """
        Return:

            GID -> ByteTrack track IDs
        """

        result: Dict[str, List[int]] = {}

        identities = self.manager.get_identities()

        for identity_id, identity in identities.items():

            result[identity_id] = sorted(
                identity.source_track_ids
            )

        return result

    # ========================================================

    def get_singleton_identities(
        self,
    ) -> Dict[str, List[int]]:
        """
        Return identities containing exactly one track.
        """

        identity_tracks = (
            self.get_identity_track_ids()
        )

        return {
            identity_id: track_ids
            for identity_id, track_ids
            in identity_tracks.items()
            if len(track_ids) == 1
        }

    # ========================================================

    def get_merged_identities(
        self,
    ) -> Dict[str, List[int]]:
        """
        Return identities containing multiple
        ByteTrack fragments.
        """

        identity_tracks = (
            self.get_identity_track_ids()
        )

        return {
            identity_id: track_ids
            for identity_id, track_ids
            in identity_tracks.items()
            if len(track_ids) > 1
        }

    # ========================================================
    # MERGE STATISTICS
    # ========================================================

    def get_merge_ratio(self) -> float:
        """
        Calculate the percentage of assigned tracks
        that belong to merged identities.

        Example:

            100 assigned tracks
            20 belong to merged identities

            merge ratio = 0.20
        """

        identity_tracks = (
            self.get_identity_track_ids()
        )

        total_tracks = sum(
            len(track_ids)
            for track_ids in identity_tracks.values()
        )

        if total_tracks == 0:
            return 0.0

        merged_tracks = sum(
            len(track_ids)
            for track_ids in identity_tracks.values()
            if len(track_ids) > 1
        )

        return (
            merged_tracks
            / total_tracks
        )

    # ========================================================

    def get_max_tracks_per_identity(
        self,
    ) -> int:
        """
        Return the largest number of ByteTrack
        fragments assigned to one GID.
        """

        counts = (
            self.get_identity_track_counts()
        )

        if not counts:
            return 0

        return max(
            counts.values()
        )

    # ========================================================

    def get_average_tracks_per_identity(
        self,
    ) -> float:
        """
        Return average ByteTrack fragments per GID.
        """

        identity_count = (
            self.manager.get_identity_count()
        )

        if identity_count == 0:
            return 0.0

        total_tracks = sum(
            self.get_identity_track_counts().values()
        )

        return (
            total_tracks
            / identity_count
        )

    # ========================================================
    # TEMPORAL ANALYSIS
    # ========================================================

    @staticmethod
    def _tracks_temporally_overlap(
        track_a: Track,
        track_b: Track,
    ) -> bool:
        """
        Return True when two tracks have overlapping
        frame ranges.

        Example:

            Track A: 100 -> 200
            Track B: 150 -> 250

            overlap = True
        """

        if track_a.first_frame is None:
            return False

        if track_a.last_frame is None:
            return False

        if track_b.first_frame is None:
            return False

        if track_b.last_frame is None:
            return False

        return (
            track_a.first_frame
            <= track_b.last_frame
            and
            track_b.first_frame
            <= track_a.last_frame
        )

    # ========================================================

    def find_temporal_overlaps(
        self,
    ) -> Dict[str, List[tuple[int, int]]]:
        """
        Find identities containing ByteTrack fragments
        that overlap temporally.

        A strong identity merge should generally not contain
        two different physical tracks active at the same time.

        This is therefore useful for detecting suspicious
        fusion decisions.
        """

        result: Dict[
            str,
            List[tuple[int, int]],
        ] = {}

        identity_tracks = (
            self.get_identity_track_ids()
        )

        for identity_id, track_ids in identity_tracks.items():

            if len(track_ids) < 2:
                continue

            overlaps: List[
                tuple[int, int]
            ] = []

            for index, track_id_a in enumerate(
                track_ids
            ):

                track_a = self.track_history.get(
                    track_id_a
                )

                if track_a is None:
                    continue

                for track_id_b in track_ids[
                    index + 1:
                ]:

                    track_b = (
                        self.track_history.get(
                            track_id_b
                        )
                    )

                    if track_b is None:
                        continue

                    if self._tracks_temporally_overlap(
                        track_a,
                        track_b,
                    ):
                        overlaps.append(
                            (
                                track_id_a,
                                track_id_b,
                            )
                        )

            if overlaps:
                result[
                    identity_id
                ] = overlaps

        return result

    # ========================================================

    def get_temporal_overlap_count(
        self,
    ) -> int:
        """
        Return the number of global identities
        containing temporal overlap.
        """

        return len(
            self.find_temporal_overlaps()
        )

    # ========================================================

    def get_temporal_overlap_pair_count(
        self,
    ) -> int:
        """
        Return the total number of overlapping
        track pairs.
        """

        overlaps = (
            self.find_temporal_overlaps()
        )

        return sum(
            len(pairs)
            for pairs in overlaps.values()
        )

    # ========================================================
    # SUSPICIOUS IDENTITY DETECTION
    # ========================================================

    def find_suspicious_identities(
        self,
    ) -> Dict[
        str,
        List[tuple[int, int]],
    ]:
        """
        Identify potentially incorrect global identities.

        Currently, an identity is suspicious when two of
        its source tracks are simultaneously active.

        Example:

            GID_0010:
                Track 25: 100 -> 200
                Track 31: 150 -> 250

        These tracks overlap and therefore deserve inspection.
        """

        return self.find_temporal_overlaps()

    # ========================================================

    def get_suspicious_identity_ids(
        self,
    ) -> List[str]:
        """
        Return IDs of suspicious global identities.
        """

        return sorted(
            self.find_suspicious_identities().keys()
        )

    # ========================================================
    # FULL EVALUATION
    # ========================================================

    def evaluate(
        self,
    ) -> IdentityEvaluationResult:
        """
        Run the complete evaluation.
        """

        identity_track_counts = (
            self.get_identity_track_counts()
        )

        identity_track_ids = (
            self.get_identity_track_ids()
        )

        singleton_identities = sum(
            1
            for count
            in identity_track_counts.values()
            if count == 1
        )

        merged_identities = sum(
            1
            for count
            in identity_track_counts.values()
            if count > 1
        )

        assigned_tracks = sum(
            identity_track_counts.values()
        )

        merged_tracks = sum(
            count
            for count
            in identity_track_counts.values()
            if count > 1
        )

        suspicious = (
            self.find_suspicious_identities()
        )

        return IdentityEvaluationResult(

            # Basic
            total_tracks=self.get_total_tracks(),

            tracks_with_embeddings=(
                assigned_tracks
            ),

            total_global_identities=(
                self.get_identity_count()
            ),

            # Identity composition
            singleton_identities=(
                singleton_identities
            ),

            merged_identities=(
                merged_identities
            ),

            max_tracks_per_identity=(
                self.get_max_tracks_per_identity()
            ),

            average_tracks_per_identity=(
                self.get_average_tracks_per_identity()
            ),

            # Merge
            total_merged_tracks=(
                merged_tracks
            ),

            merge_ratio=(
                merged_tracks / assigned_tracks
                if assigned_tracks
                else 0.0
            ),

            # Temporal
            identities_with_temporal_overlap=(
                self.get_temporal_overlap_count()
            ),

            temporal_overlap_pairs=(
                self.get_temporal_overlap_pair_count()
            ),

            # Suspicious
            suspicious_identity_count=(
                len(suspicious)
            ),

            suspicious_identity_ids=(
                sorted(suspicious.keys())
            ),

            # Per identity
            identity_track_counts=(
                identity_track_counts
            ),

            identity_track_ids=(
                identity_track_ids
            ),
        )

    # ========================================================
    # PRINT REPORT
    # ========================================================

    def print_report(
        self,
        result: Optional[
            IdentityEvaluationResult
        ] = None,
    ) -> None:
        """
        Print a human-readable evaluation report.
        """

        if result is None:
            result = self.evaluate()

        print()
        print("=" * 70)
        print("FUSED IDENTITY EVALUATION")
        print("=" * 70)

        print()
        print("TRACK STATISTICS")
        print("-" * 70)

        print(
            f"ByteTrack tracks: "
            f"{result.total_tracks}"
        )

        print(
            f"Tracks assigned to GIDs: "
            f"{result.tracks_with_embeddings}"
        )

        print(
            f"Unassigned tracks: "
            f"{result.total_tracks - result.tracks_with_embeddings}"
        )

        print()
        print("GLOBAL IDENTITY STATISTICS")
        print("-" * 70)

        print(
            f"Global identities: "
            f"{result.total_global_identities}"
        )

        print(
            f"Singleton identities: "
            f"{result.singleton_identities}"
        )

        print(
            f"Merged identities: "
            f"{result.merged_identities}"
        )

        print(
            f"Average tracks / identity: "
            f"{result.average_tracks_per_identity:.2f}"
        )

        print(
            f"Maximum tracks / identity: "
            f"{result.max_tracks_per_identity}"
        )

        print()
        print("MERGE STATISTICS")
        print("-" * 70)

        print(
            f"Tracks belonging to merged identities: "
            f"{result.total_merged_tracks}"
        )

        print(
            f"Merge ratio: "
            f"{result.merge_ratio:.2%}"
        )

        print()
        print("TEMPORAL CONSISTENCY")
        print("-" * 70)

        print(
            f"Identities with temporal overlap: "
            f"{result.identities_with_temporal_overlap}"
        )

        print(
            f"Overlapping track pairs: "
            f"{result.temporal_overlap_pairs}"
        )

        print()
        print("SUSPICIOUS IDENTITIES")
        print("-" * 70)

        if not result.suspicious_identity_ids:

            print(
                "No temporally overlapping identities detected."
            )

        else:

            print(
                f"Suspicious identities: "
                f"{result.suspicious_identity_count}"
            )

            for identity_id in (
                result.suspicious_identity_ids
            ):

                track_ids = (
                    result.identity_track_ids[
                        identity_id
                    ]
                )

                print(
                    f"  {identity_id}: "
                    f"tracks {track_ids}"
                )

        print()
        print("=" * 70)
        print("FUSED IDENTITY EVALUATION COMPLETE")
        print("=" * 70)

    # ========================================================
    # PRINT MERGED IDENTITIES
    # ========================================================

    def print_merged_identities(
        self,
    ) -> None:
        """
        Print all identities containing multiple
        ByteTrack fragments.
        """

        merged = (
            self.get_merged_identities()
        )

        print()
        print("=" * 70)
        print("MERGED GLOBAL IDENTITIES")
        print("=" * 70)

        if not merged:

            print(
                "No merged identities found."
            )

            return

        for identity_id, track_ids in sorted(
            merged.items()
        ):

            print(
                f"{identity_id}: "
                f"{len(track_ids)} tracks "
                f"-> {track_ids}"
            )

        print()
        print(
            f"Total merged identities: "
            f"{len(merged)}"
        )

    # ========================================================
    # PRINT SUSPICIOUS IDENTITIES
    # ========================================================

    def print_suspicious_identities(
        self,
    ) -> None:
        """
        Print identities containing temporally
        overlapping tracks.
        """

        suspicious = (
            self.find_suspicious_identities()
        )

        print()
        print("=" * 70)
        print("SUSPICIOUS FUSED IDENTITIES")
        print("=" * 70)

        if not suspicious:

            print(
                "No suspicious identities found."
            )

            return

        for identity_id, pairs in sorted(
            suspicious.items()
        ):

            print()
            print(identity_id)

            for track_a, track_b in pairs:

                print(
                    f"  overlapping tracks: "
                    f"{track_a} <-> {track_b}"
                )

        print()
        print(
            f"Suspicious identities: "
            f"{len(suspicious)}"
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================


def evaluate_fused_identities(
    manager: FusedIdentityManager,
    track_history: Optional[
        Dict[int, Track]
    ] = None,
) -> IdentityEvaluationResult:
    """
    Convenience function for evaluating an existing
    FusedIdentityManager.
    """

    evaluator = FusedIdentityEvaluator(
        manager=manager,
        track_history=track_history,
    )

    return evaluator.evaluate()