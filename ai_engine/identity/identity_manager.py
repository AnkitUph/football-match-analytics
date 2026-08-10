"""
Global Identity Manager V3.

Converts short-term ByteTrack fragments into persistent
global identities.

Matching signals:

- class compatibility
- temporal continuity
- spatial continuity
- motion similarity
- appearance similarity
- team consistency
- jersey-number consistency

The manager does not perform detection or tracking.

It consumes ByteTrack Track objects and produces
persistent Identity objects.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

from ai_engine.identity.identity_matcher import (
    IdentityMatcher,
    MatchResult,
)

from ai_engine.identity.identity_matching_evaluation import (
    IdentityMatchingEvaluation,
)

from ai_engine.schemas.track import (
    Identity,
    MatchTracks,
    Track,
)

logger = logging.getLogger(__name__)


# ============================================================
# APPEARANCE DIAGNOSTICS
# ============================================================


class AppearanceDiagnostics:
    """
    Appearance matching diagnostics.

    Important:

    A candidate rejected before appearance evaluation is NOT
    counted as a missing embedding.
    """

    def __init__(
        self,
        minimum_similarity: float = 0.68,
        strong_similarity: float = 0.85,
    ) -> None:

        self.minimum_similarity = float(
            minimum_similarity
        )

        self.strong_similarity = float(
            strong_similarity
        )

        self.reset()

    def reset(self) -> None:

        self.candidates_requiring_appearance = 0

        self.comparisons = 0

        self.valid_comparisons = 0

        self.missing_embeddings = 0

        self.accepted_matches = 0

        self.rejected_matches = 0

        self.strong_matches = 0

        self.weak_matches = 0

        self.similarities: list[float] = []

        self.accepted_similarities: list[float] = []

        self.rejected_similarities: list[float] = []

        self.rejection_reasons = defaultdict(int)

    def record(
        self,
        similarity: float,
        accepted: bool,
        reason: str = "",
    ) -> None:

        self.comparisons += 1

        # Appearance unavailable.
        if similarity < 0.0:

            self.missing_embeddings += 1

            return

        self.valid_comparisons += 1

        similarity = float(similarity)

        self.similarities.append(
            similarity
        )

        if similarity >= self.strong_similarity:

            self.strong_matches += 1

        elif similarity >= self.minimum_similarity:

            self.weak_matches += 1

        if accepted:

            self.accepted_matches += 1

            self.accepted_similarities.append(
                similarity
            )

        else:

            self.rejected_matches += 1

            self.rejected_similarities.append(
                similarity
            )

            if reason:

                self.rejection_reasons[
                    reason
                ] += 1

    @staticmethod
    def _statistics(
        values: list[float],
    ) -> dict:

        if not values:

            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
            }

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": (
                sum(values)
                / len(values)
            ),
        }

    def get_summary(self) -> dict:

        return {
            "candidates_requiring_appearance": (
                self.candidates_requiring_appearance
            ),

            "comparisons": (
                self.comparisons
            ),

            "valid_comparisons": (
                self.valid_comparisons
            ),

            "missing_embeddings": (
                self.missing_embeddings
            ),

            "accepted_matches": (
                self.accepted_matches
            ),

            "rejected_matches": (
                self.rejected_matches
            ),

            "strong_matches": (
                self.strong_matches
            ),

            "weak_matches": (
                self.weak_matches
            ),

            "minimum_similarity": (
                self.minimum_similarity
            ),

            "strong_similarity": (
                self.strong_similarity
            ),

            "similarity": (
                self._statistics(
                    self.similarities
                )
            ),

            "accepted_similarity": (
                self._statistics(
                    self.accepted_similarities
                )
            ),

            "rejected_similarity": (
                self._statistics(
                    self.rejected_similarities
                )
            ),

            "rejection_reasons": dict(
                self.rejection_reasons
            ),
        }


# ============================================================
# GLOBAL IDENTITY MANAGER
# ============================================================


class GlobalIdentityManager:
    """
    Merge ByteTrack fragments into persistent identities.
    """

    DEFAULT_CLASSES = {
        "player",
        "goalkeeper",
        "referee",
    }

    def __init__(
        self,
        matcher: Optional[IdentityMatcher] = None,
        minimum_match_score: Optional[float] = None,
        allowed_classes: Optional[set[str]] = None,
    ) -> None:

        self.matcher = (
            matcher
            or IdentityMatcher()
        )

        if minimum_match_score is None:

            minimum_match_score = getattr(
                self.matcher,
                "minimum_match_score",
                0.55,
            )

        self.minimum_match_score = float(
            minimum_match_score
        )

        self.allowed_classes = (
            set(allowed_classes)
            if allowed_classes is not None
            else set(self.DEFAULT_CLASSES)
        )

        self.identities: Dict[
            str,
            Identity,
        ] = {}

        self._next_identity_number = 1

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self.total_tracks_processed = 0

        self.total_tracks_merged = 0

        self.total_new_identities = 0

        # ----------------------------------------------------
        # Candidate diagnostics
        # ----------------------------------------------------

        self.match_diagnostics: list[
            dict[str, Any]
        ] = []

        self.rejection_counters = defaultdict(int)

        # ----------------------------------------------------
        # Appearance diagnostics
        # ----------------------------------------------------

        minimum_appearance = getattr(
            self.matcher,
            "minimum_appearance_similarity",
            0.68,
        )

        strong_appearance = getattr(
            self.matcher,
            "appearance_strong_similarity",
            0.85,
        )

        self.appearance_diagnostics = (
            AppearanceDiagnostics(
                minimum_similarity=(
                    minimum_appearance
                ),
                strong_similarity=(
                    strong_appearance
                ),
            )
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        self.identity_matching_evaluation = (
            IdentityMatchingEvaluation(
                diagnostics=self.match_diagnostics,
                minimum_appearance_similarity=(
                    minimum_appearance
                ),
                strong_appearance_similarity=(
                    strong_appearance
                ),
            )
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def build_identities(
        self,
        tracks: Iterable[Track],
    ) -> Dict[str, Identity]:

        self.reset()

        valid_tracks = [
            track
            for track in tracks
            if self._is_valid_track(track)
        ]

        valid_tracks.sort(
            key=lambda track: (
                track.first_frame
                if track.first_frame is not None
                else float("inf"),
                track.local_id,
            )
        )

        logger.info(
            "Building global identities from %d tracks",
            len(valid_tracks),
        )

        for track in valid_tracks:

            self.total_tracks_processed += 1

            identity, result = (
                self._find_best_identity(
                    track
                )
            )

            if (
                identity is not None
                and result is not None
                and result.matched
                and result.score
                >= self.minimum_match_score
            ):

                self._merge_track(
                    identity,
                    track,
                )

                self.total_tracks_merged += 1

            else:

                identity = (
                    self._create_identity(
                        track
                    )
                )

                self.total_new_identities += 1

        self.identity_matching_evaluation.diagnostics = list(
            self.match_diagnostics
        )

        return self.identities

    # ========================================================
    # MATCH TRACKS
    # ========================================================

    def build_match_tracks(
        self,
        tracks: Iterable[Track],
        fps: float = 25.0,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> MatchTracks:

        tracks = list(tracks)

        identities = self.build_identities(
            tracks
        )

        return MatchTracks(
            tracks={
                track.local_id: track
                for track in tracks
            },
            identities=identities,
            fps=float(fps),
            frame_width=frame_width,
            frame_height=frame_height,
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:

        self.identities.clear()

        self._next_identity_number = 1

        self.total_tracks_processed = 0

        self.total_tracks_merged = 0

        self.total_new_identities = 0

        self.match_diagnostics.clear()

        self.rejection_counters.clear()

        self.appearance_diagnostics.reset()

        self.identity_matching_evaluation.diagnostics = (
            self.match_diagnostics
        )

    # ========================================================
    # FIND BEST IDENTITY
    # ========================================================

    def _find_best_identity(
        self,
        track: Track,
    ) -> tuple[
        Optional[Identity],
        Optional[MatchResult],
    ]:

        best_identity: Optional[
            Identity
        ] = None

        best_result: Optional[
            MatchResult
        ] = None

        for identity in self.identities.values():

            if not identity.active:
                continue

            if identity.class_name != track.class_name:
                continue

            result = self.matcher.match(
                track,
                identity,
            )

            accepted = (
                result.matched
                and result.score
                >= self.minimum_match_score
            )

            reason = (
                "accepted"
                if accepted
                else result.reason
            )

            self.rejection_counters[
                result.reason
            ] += 1

            # ------------------------------------------------
            # Appearance is only meaningful if matcher
            # reached the appearance stage.
            # ------------------------------------------------

            if result.reason not in {
                "track_overlaps_identity",
                "temporal_gap_too_large",
                "spatial_distance_too_large",
                "motion_difference_too_large",
                "class_mismatch",
                "track_has_no_observations",
                "identity_has_no_observations",
                "spatial_information_unavailable",
            }:

                self.appearance_diagnostics.record(
                    similarity=(
                        result.appearance_similarity
                    ),
                    accepted=accepted,
                    reason=reason,
                )

                self.appearance_diagnostics.candidates_requiring_appearance += 1

            # ------------------------------------------------
            # Diagnostic record
            # ------------------------------------------------

            self._record_diagnostic(
                track,
                identity,
                result,
                accepted,
                reason,
            )

            if not accepted:
                continue

            if (
                best_result is None
                or result.score
                > best_result.score
            ):

                best_identity = identity

                best_result = result

        return (
            best_identity,
            best_result,
        )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def _record_diagnostic(
        self,
        track: Track,
        identity: Identity,
        result: MatchResult,
        accepted: bool,
        reason: str,
    ) -> None:

        self.match_diagnostics.append(
            {
                "track_id": (
                    track.local_id
                ),

                "identity_id": (
                    identity.identity_id
                ),

                "identity_source_track_ids": list(
                    identity.source_track_ids
                ),

                "identity_first_frame": (
                    identity.first_frame
                ),

                "identity_last_frame": (
                    identity.last_frame
                ),

                "matched": bool(
                    result.matched
                ),

                "score": float(
                    result.score
                ),

                "temporal_gap": int(
                    result.temporal_gap
                ),

                "spatial_distance": float(
                    result.spatial_distance
                ),

                "motion_difference": float(
                    result.motion_difference
                ),

                "appearance_similarity": float(
                    result.appearance_similarity
                ),

                "team_match": (
                    result.team_match
                ),

                "jersey_match": (
                    result.jersey_match
                ),

                "accepted": bool(
                    accepted
                ),

                "reason": reason,
            }
        )

    def get_match_diagnostics(self) -> dict:

        diagnostics = (
            self.match_diagnostics
        )

        accepted = [
            item
            for item in diagnostics
            if item["accepted"]
        ]

        rejected = [
            item
            for item in diagnostics
            if not item["accepted"]
        ]

        rejection_reasons: dict[
            str,
            int,
        ] = {}

        for item in rejected:

            reason = item["reason"]

            rejection_reasons[reason] = (
                rejection_reasons.get(
                    reason,
                    0,
                )
                + 1
            )

        accepted_scores = [
            item["score"]
            for item in accepted
        ]

        if accepted_scores:

            score_min = min(
                accepted_scores
            )

            score_max = max(
                accepted_scores
            )

            score_avg = (
                sum(accepted_scores)
                / len(accepted_scores)
            )

        else:

            score_min = 0.0

            score_max = 0.0

            score_avg = 0.0

        appearance_values = [
            item[
                "appearance_similarity"
            ]
            for item in accepted
            if item[
                "appearance_similarity"
            ] >= 0.0
        ]

        appearance_avg = (
            sum(appearance_values)
            / len(appearance_values)
            if appearance_values
            else 0.0
        )

        return {
            "candidates_checked": len(
                diagnostics
            ),

            "accepted_matches": len(
                accepted
            ),

            "rejected_matches": len(
                rejected
            ),

            "rejection_reasons": (
                rejection_reasons
            ),

            "accepted_score": {
                "min": score_min,
                "max": score_max,
                "avg": score_avg,
            },

            "appearance": {
                "matches_with_embeddings": len(
                    appearance_values
                ),

                "average_similarity": (
                    appearance_avg
                ),
            },

            "team_matches": sum(
                1
                for item in accepted
                if item[
                    "team_match"
                ] is True
            ),

            "jersey_matches": sum(
                1
                for item in accepted
                if item[
                    "jersey_match"
                ] is True
            ),
        }

    # ========================================================
    # APPEARANCE DIAGNOSTICS
    # ========================================================

    def get_appearance_diagnostics(
        self,
    ) -> dict:

        return (
            self.appearance_diagnostics
            .get_summary()
        )

    def log_appearance_diagnostics(
        self,
    ) -> None:

        summary = (
            self.get_appearance_diagnostics()
        )

        logger.info(
            "APPEARANCE DIAGNOSTICS"
        )

        logger.info(
            "Candidates requiring appearance: %d",
            summary[
                "candidates_requiring_appearance"
            ],
        )

        logger.info(
            "Appearance comparisons: %d",
            summary["comparisons"],
        )

        logger.info(
            "Valid comparisons: %d",
            summary["valid_comparisons"],
        )

        logger.info(
            "Missing embeddings: %d",
            summary["missing_embeddings"],
        )

        logger.info(
            "Accepted matches: %d",
            summary["accepted_matches"],
        )

        logger.info(
            "Rejected matches: %d",
            summary["rejected_matches"],
        )

        logger.info(
            "Strong matches: %d",
            summary["strong_matches"],
        )

        logger.info(
            "Similarity: %s",
            summary["similarity"],
        )

        logger.info(
            "Accepted similarity: %s",
            summary["accepted_similarity"],
        )

        logger.info(
            "Rejected similarity: %s",
            summary["rejected_similarity"],
        )

        logger.info(
            "Rejection reasons: %s",
            summary["rejection_reasons"],
        )

    # ========================================================
    # EVALUATION
    # ========================================================

    def get_identity_matching_evaluation(
        self,
    ) -> dict[str, Any]:

        self.identity_matching_evaluation.diagnostics = list(
            self.match_diagnostics
        )

        return (
            self.identity_matching_evaluation
            .get_report()
        )

    def get_identity_matching_evaluation_summary(
        self,
    ) -> dict[str, Any]:

        self.identity_matching_evaluation.diagnostics = list(
            self.match_diagnostics
        )

        return (
            self.identity_matching_evaluation
            .get_summary()
        )

    def log_identity_matching_evaluation(
        self,
    ) -> None:

        self.identity_matching_evaluation.diagnostics = list(
            self.match_diagnostics
        )

        self.identity_matching_evaluation.log_report()

    # ========================================================
    # GENERAL LOGGING
    # ========================================================

    def log_match_diagnostics(
        self,
    ) -> None:

        summary = (
            self.get_match_diagnostics()
        )

        logger.info(
            "GLOBAL IDENTITY V3 DIAGNOSTICS"
        )

        logger.info(
            "Candidates checked: %d",
            summary[
                "candidates_checked"
            ],
        )

        logger.info(
            "Accepted matches: %d",
            summary[
                "accepted_matches"
            ],
        )

        logger.info(
            "Rejected matches: %d",
            summary[
                "rejected_matches"
            ],
        )

        logger.info(
            "Rejection reasons: %s",
            summary[
                "rejection_reasons"
            ],
        )

        logger.info(
            "Appearance: %s",
            summary[
                "appearance"
            ],
        )

    # ========================================================
    # MERGE
    # ========================================================

    def _merge_track(
        self,
        identity: Identity,
        track: Track,
    ) -> None:

        if (
            track.local_id
            in identity.source_track_ids
        ):
            return

        identity.add_track(
            track
        )

        identity.active = True

        self._update_identity_metadata(
            identity
        )

    # ========================================================
    # CREATE
    # ========================================================

    def _create_identity(
        self,
        track: Track,
    ) -> Identity:

        identity = Identity(
            identity_id=(
                self._generate_identity_id()
            ),

            class_name=(
                track.class_name
            ),

            source_track_ids=[],

            observations=[],

            team=None,

            jersey_number=None,

            identity_confidence=0.0,

            active=True,

            first_frame=None,

            last_frame=None,

            team_history=[],

            jersey_history=[],
        )

        identity.add_track(
            track
        )

        self._update_identity_metadata(
            identity
        )

        self.identities[
            identity.identity_id
        ] = identity

        return identity

    def _generate_identity_id(
        self,
    ) -> str:

        identity_id = (
            f"GID_{self._next_identity_number:04d}"
        )

        self._next_identity_number += 1

        return identity_id

    # ========================================================
    # METADATA
    # ========================================================

    @staticmethod
    def _update_identity_metadata(
        identity: Identity,
    ) -> None:

        if identity.team_history:

            identity.team = (
                GlobalIdentityManager._most_common(
                    identity.team_history
                )
            )

        else:

            identity.team = None

        if identity.jersey_history:

            identity.jersey_number = (
                GlobalIdentityManager._most_common(
                    identity.jersey_history
                )
            )

        else:

            identity.jersey_number = None

        fragment_count = len(
            identity.source_track_ids
        )

        if fragment_count <= 1:

            identity.identity_confidence = 0.50

        else:

            identity.identity_confidence = min(
                1.0,
                0.50
                + (
                    0.10
                    * (
                        fragment_count - 1
                    )
                ),
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _is_valid_track(
        self,
        track: Track,
    ) -> bool:

        if (
            track.class_name
            not in self.allowed_classes
        ):
            return False

        if not track.observations:
            return False

        if track.first_frame is None:
            return False

        if track.last_frame is None:
            return False

        return True

    # ========================================================
    # UTILITY
    # ========================================================

    @staticmethod
    def _most_common(
        values,
    ):

        if not values:
            return None

        counts = {}

        for value in values:

            counts[value] = (
                counts.get(
                    value,
                    0,
                )
                + 1
            )

        return max(
            counts,
            key=counts.get,
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    def get_statistics(
        self,
    ) -> dict:

        identities = list(
            self.identities.values()
        )

        by_class = {}

        for identity in identities:

            class_name = (
                identity.class_name
            )

            by_class[class_name] = (
                by_class.get(
                    class_name,
                    0,
                )
                + 1
            )

        diagnostics = (
            self.get_match_diagnostics()
        )

        evaluation_summary = (
            self.get_identity_matching_evaluation_summary()
        )

        return {
            "version": "v3",

            "tracks_processed": (
                self.total_tracks_processed
            ),

            "tracks_merged": (
                self.total_tracks_merged
            ),

            "new_identities": (
                self.total_new_identities
            ),

            "global_identities": len(
                identities
            ),

            "identities_by_class": (
                by_class
            ),

            "candidates_checked": (
                diagnostics[
                    "candidates_checked"
                ]
            ),

            "accepted_matches": (
                diagnostics[
                    "accepted_matches"
                ]
            ),

            "rejected_matches": (
                diagnostics[
                    "rejected_matches"
                ]
            ),

            "rejection_reasons": (
                diagnostics[
                    "rejection_reasons"
                ]
            ),

            "accepted_score": (
                diagnostics[
                    "accepted_score"
                ]
            ),

            "appearance": (
                diagnostics[
                    "appearance"
                ]
            ),

            "appearance_diagnostics": (
                self.appearance_diagnostics
                .get_summary()
            ),

            "identity_matching_evaluation": (
                evaluation_summary
            ),

            "team_matches": (
                diagnostics[
                    "team_matches"
                ]
            ),

            "jersey_matches": (
                diagnostics[
                    "jersey_matches"
                ]
            ),
        }

    def get_identity_summary(
        self,
    ) -> dict:

        return self.get_statistics()