
"""
Global Identity Manager V5.

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
from unittest import result

from ai_engine.identity.identity_matcher import (
    IdentityMatcher,
    MatchResult,
)

from ai_engine.identity.identity_matching_evaluation import (
    IdentityMatchingEvaluation,
)

from ai_engine.schemas import track
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

    A candidate requiring appearance comparison is counted in
    candidates_requiring_appearance before the comparison result
    is recorded.
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

    def record_candidate(self) -> None:
        """
        Record that the matcher reached the appearance stage.

        This is intentionally separate from record() because
        candidates_requiring_appearance describes candidates that
        reached appearance evaluation, regardless of whether the
        embedding comparison succeeds.
        """

        self.candidates_requiring_appearance += 1

    def record(
        self,
        similarity: float,
        accepted: bool,
        reason: str = "",
    ) -> None:

        self.comparisons += 1

        # ----------------------------------------------------
        # Appearance unavailable
        # ----------------------------------------------------

        if similarity < 0.0:

            self.missing_embeddings += 1

            return

        # ----------------------------------------------------
        # Valid appearance comparison
        # ----------------------------------------------------

        self.valid_comparisons += 1

        similarity = float(similarity)

        self.similarities.append(
            similarity
        )

        # ----------------------------------------------------
        # Strength classification
        # ----------------------------------------------------

        if similarity >= self.strong_similarity:

            self.strong_matches += 1

        elif similarity >= self.minimum_similarity:

            self.weak_matches += 1

        # ----------------------------------------------------
        # Accepted / rejected
        # ----------------------------------------------------

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

    Current identity matching pipeline: V5.
    """

    VERSION = "v5.1"

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

                self._create_identity(
                    track
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
    ) -> list[MatchTracks]:

        match_tracks = []

        for track in tracks:

            if not self._is_valid_track(track):
                continue

            match_tracks.append(
                MatchTracks(
                    track=track,
                    fps=fps,
                )
            )

        return match_tracks

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:

        self.identities = {}

        self._next_identity_number = 1

        self.total_tracks_processed = 0
        self.total_tracks_merged = 0
        self.total_new_identities = 0

        self.match_diagnostics = []

        self.rejection_counters = defaultdict(
            int
        )

        self.appearance_diagnostics.reset()

        self.identity_matching_evaluation.diagnostics = (
            self.match_diagnostics
        )

    # ========================================================
    # BEST IDENTITY
    # ========================================================

    def _find_best_identity(
        self,
        track: Track,
    ) -> tuple[
        Optional[Identity],
        Optional[MatchResult],
    ]:

        best_identity = None
        best_result = None

        for identity in self.identities.values():

            if (
                identity.class_name
                != track.class_name
            ):
                continue

            result = self.matcher.match(
                track,
                identity,
            )

            self._record_match_diagnostic(
                track,
                identity,
                result,
            )

            if (
                result.matched
                and result.score
                >= self.minimum_match_score
            ):

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

    def _record_match_diagnostic(
    self,
    track: Track,
    identity: Identity,
    result: MatchResult,
    ) -> None:

        appearance_similarity = getattr(
            result,
            "appearance_similarity",
            -1.0,
        )

    # ----------------------------------------------------
    # Detect whether appearance was actually evaluated.
    #
    # A non-negative similarity means the matcher reached
    # the appearance stage.
    # ----------------------------------------------------

        if appearance_similarity >= 0.0:

            self.appearance_diagnostics.record_candidate()

    # ----------------------------------------------------
    # Store the COMPLETE V5.1 MatchResult.
    #
    # This is important for low-score inspection.
    # Previously temporal_gap, spatial_distance,
    # motion_difference, and appearance_evaluated were
    # discarded here.
    # ----------------------------------------------------

        diagnostic = {
            "track_id": track.local_id,

            "identity_id": (
                identity.identity_id
            ),

            "matched": (
                result.matched
            ),

            "score": (
                result.score
            ),

            "temporal_gap": (
                result.temporal_gap
            ),

            "spatial_distance": (
                result.spatial_distance
            ),

            "motion_difference": (
                result.motion_difference
            ),

            "appearance_similarity": (
                result.appearance_similarity
            ),

            "appearance_evaluated": (
                result.appearance_evaluated
            ),

            "team_match": (
                result.team_match
            ),

            "jersey_match": (
                result.jersey_match
            ),

            "reason": (
                result.reason
            ),
        }

        self.match_diagnostics.append(
            diagnostic
        )

        reason = diagnostic["reason"]

        if not result.matched:

            if reason:

                self.rejection_counters[
                    reason
                ] += 1

    # ----------------------------------------------------
    # Record appearance result.
    # ----------------------------------------------------

        if appearance_similarity >= 0.0:

            self.appearance_diagnostics.record(
                similarity=(
                    appearance_similarity
                ),
                accepted=result.matched,
                reason=reason,
            )



  # ========================================================
    # MATCH DIAGNOSTICS SUMMARY
    # ========================================================

    def get_match_diagnostics(
        self,
    ) -> dict[str, Any]:

        candidates_checked = len(
            self.match_diagnostics
        )

        accepted = [
            item
            for item in self.match_diagnostics
            if item["matched"]
        ]

        rejected = [
            item
            for item in self.match_diagnostics
            if not item["matched"]
        ]

        accepted_scores = [
            item["score"]
            for item in accepted
            if item["score"] is not None
        ]

        accepted_appearance = [
            item["appearance_similarity"]
            for item in accepted
            if item["appearance_similarity"] >= 0.0
        ]

        team_matches = sum(
            1
            for item in accepted
            if item["team_match"]
        )

        jersey_matches = sum(
            1
            for item in accepted
            if item["jersey_match"]
        )

        if accepted_scores:

            accepted_score = {
                "min": min(
                    accepted_scores
                ),
                "max": max(
                    accepted_scores
                ),
                "avg": (
                    sum(accepted_scores)
                    / len(accepted_scores)
                ),
            }

        else:

            accepted_score = {
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
            }

        if accepted_appearance:

            appearance = {
                "matches_with_embeddings": (
                    len(
                        accepted_appearance
                    )
                ),
                "average_similarity": (
                    sum(
                        accepted_appearance
                    )
                    / len(
                        accepted_appearance
                    )
                ),
            }

        else:

            appearance = {
                "matches_with_embeddings": 0,
                "average_similarity": 0.0,
            }

        return {
            "candidates_checked": (
                candidates_checked
            ),

            "accepted_matches": (
                len(accepted)
            ),

            "rejected_matches": (
                len(rejected)
            ),

            "rejection_reasons": dict(
                self.rejection_counters
            ),

            "accepted_score": (
                accepted_score
            ),

            "appearance": (
                appearance
            ),

            "team_matches": (
                team_matches
            ),

            "jersey_matches": (
                jersey_matches
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
            "Weak matches: %d",
            summary["weak_matches"],
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
    # V5 EVALUATION
    # ========================================================

    def _build_v5_evaluation_summary(
        self,
    ) -> dict[str, Any]:
        """
        Build the canonical V5 appearance evaluation.

        This deliberately uses AppearanceDiagnostics as the
        source of truth.

        This prevents the evaluation layer from incorrectly
        reporting:

            appearance_accepted = 0

        when appearance diagnostics already show accepted
        appearance matches.
        """

        appearance = (
            self.appearance_diagnostics
            .get_summary()
        )

        candidates = len(
            self.match_diagnostics
        )

        appearance_comparisons = (
            appearance["comparisons"]
        )

        appearance_accepted = (
            appearance["accepted_matches"]
        )

        appearance_rejected = (
            appearance["rejected_matches"]
        )

        strong_matches = (
            appearance["strong_matches"]
        )

        average_similarity = (
            appearance["similarity"]["avg"]
        )

        accepted_average_similarity = (
            appearance[
                "accepted_similarity"
            ]["avg"]
        )

        rejected_average_similarity = (
            appearance[
                "rejected_similarity"
            ]["avg"]
        )

        # ----------------------------------------------------
        # Risk assessment
        # ----------------------------------------------------

        risky_matches = 0

        if appearance_accepted > 0:

            for similarity in (
                self.appearance_diagnostics
                .accepted_similarities
            ):

                # Accepted matches below the normal
                # appearance threshold are considered risky.
                if (
                    similarity
                    < self.appearance_diagnostics
                    .minimum_similarity
                ):
                    risky_matches += 1

        # ----------------------------------------------------
        # Recommendation
        # ----------------------------------------------------

        if appearance_accepted == 0:

            if appearance_comparisons == 0:

                recommendation = (
                    "NO APPEARANCE MATCHES: "
                    "no valid appearance comparisons "
                    "were recorded."
                )

            else:

                recommendation = (
                    "WEAK: no accepted appearance "
                    "matches were recorded."
                )

        elif risky_matches > 0:

            recommendation = (
                "REVIEW: a significant portion of "
                "accepted appearance matches are "
                "potentially risky."
            )

        elif (
            accepted_average_similarity
            >= self.appearance_diagnostics
            .strong_similarity
        ):

            recommendation = (
                "EXCELLENT: appearance matching "
                "quality is strong; keep the current "
                "threshold for further validation."
            )

        elif (
            accepted_average_similarity
            >= self.appearance_diagnostics
            .minimum_similarity
        ):

            recommendation = (
                "GOOD: accepted appearance matches "
                "are above the minimum threshold; "
                "continue validation."
            )

        else:

            recommendation = (
                "WEAK: accepted appearance "
                "similarity is low; review the "
                "appearance threshold and embedding "
                "quality."
            )

        return {
            "candidates": candidates,

            "appearance_comparisons": (
                appearance_comparisons
            ),

            "appearance_accepted": (
                appearance_accepted
            ),

            "appearance_rejected": (
                appearance_rejected
            ),

            "strong_matches": (
                strong_matches
            ),

            "average_similarity": (
                average_similarity
            ),

            "accepted_average_similarity": (
                accepted_average_similarity
            ),

            "rejected_average_similarity": (
                rejected_average_similarity
            ),

            "risky_matches": (
                risky_matches
            ),

            "recommendation": (
                recommendation
            ),
        }

    def get_identity_matching_evaluation(
        self,
    ) -> dict[str, Any]:

        # Keep the underlying evaluator synchronized.
        self.identity_matching_evaluation.diagnostics = list(
            self.match_diagnostics
        )

        report = (
            self.identity_matching_evaluation
            .get_report()
        )

        # ----------------------------------------------------
        # Override the appearance section with the canonical
        # V5 diagnostics.
        # ----------------------------------------------------

        v5_summary = (
            self._build_v5_evaluation_summary()
        )

        report.update(
            v5_summary
        )

        return report

    def get_identity_matching_evaluation_summary(
        self,
    ) -> dict[str, Any]:

        # Keep evaluator state synchronized.
        self.identity_matching_evaluation.diagnostics = list(
            self.match_diagnostics
        )

        summary = (
            self.identity_matching_evaluation
            .get_summary()
        )

        # ----------------------------------------------------
        # Canonical V5 appearance evaluation.
        # ----------------------------------------------------

        v5_summary = (
            self._build_v5_evaluation_summary()
        )

        summary.update(
            v5_summary
        )

        return summary

    def log_identity_matching_evaluation(
        self,
    ) -> None:

        evaluation = (
            self.get_identity_matching_evaluation()
        )

        logger.info(
            "IDENTITY MATCHING %s EVALUATION",
            self.VERSION.upper(),
        )

        logger.info(
            "Candidates: %d",
            evaluation["candidates"],
        )

        logger.info(
            "Appearance comparisons: %d",
            evaluation[
                "appearance_comparisons"
            ],
        )

        logger.info(
            "Appearance accepted: %d",
            evaluation[
                "appearance_accepted"
            ],
        )

        logger.info(
            "Appearance rejected: %d",
            evaluation[
                "appearance_rejected"
            ],
        )

        logger.info(
            "Strong matches: %d",
            evaluation[
                "strong_matches"
            ],
        )

        logger.info(
            "Average appearance similarity: %.4f",
            evaluation[
                "average_similarity"
            ],
        )

        logger.info(
            "Accepted appearance similarity: %.4f",
            evaluation[
                "accepted_average_similarity"
            ],
        )

        logger.info(
            "Rejected appearance similarity: %.4f",
            evaluation[
                "rejected_average_similarity"
            ],
        )

        logger.info(
            "Risky matches: %d",
            evaluation[
                "risky_matches"
            ],
        )

        logger.info(
            "Recommendation: %s",
            evaluation[
                "recommendation"
            ],
        )

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
            "GLOBAL IDENTITY %s DIAGNOSTICS",
            self.VERSION.upper(),
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
            # ------------------------------------------------
            # V5 VERSION
            # ------------------------------------------------

            "version": self.VERSION,

            "matcher_version": getattr(
                self.matcher,
                "VERSION",
                self.VERSION,
            ),

            # ------------------------------------------------
            # Track statistics
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Match statistics
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Appearance diagnostics
            # ------------------------------------------------

            "appearance_diagnostics": (
                self.appearance_diagnostics
                .get_summary()
            ),

            # ------------------------------------------------
            # V5 evaluation
            # ------------------------------------------------

            "identity_matching_evaluation": (
                evaluation_summary
            ),

            # ------------------------------------------------
            # Other matching signals
            # ------------------------------------------------

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

