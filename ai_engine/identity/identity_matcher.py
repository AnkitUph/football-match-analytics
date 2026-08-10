
"""
Global Identity Matcher V4.

Compares a ByteTrack track fragment against an existing
global Identity.

Signals:

1. Class compatibility
2. Temporal continuity
3. Spatial continuity
4. Motion similarity
5. Appearance similarity
6. Team consistency
7. Jersey-number consistency

V4 improvements:

- Strong temporal/spatial/motion gating.
- Multi-embedding OSNet appearance matching.
- More conservative appearance acceptance.
- Strong appearance can provide additional confidence.
- Weak appearance requires stronger geometric evidence.
- Explicit team/jersey contradictions are rejected.
- Missing team/jersey evidence is neutral.
- Appearance is evaluated only after geometric gating.
- Missing appearance is different from appearance mismatch.
- Matcher NEVER mutates Track or Identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

from ai_engine.schemas.track import Identity, Track


# ============================================================
# MATCH RESULT
# ============================================================


@dataclass
class MatchResult:
    """
    Result of comparing a Track against an Identity.
    """

    matched: bool

    score: float

    temporal_gap: int

    spatial_distance: float

    motion_difference: float

    appearance_similarity: float

    team_match: Optional[bool]

    jersey_match: Optional[bool]

    reason: str = ""


# ============================================================
# MATCHER V4
# ============================================================


class IdentityMatcher:
    """
    Global identity matcher V4.

    Matching flow:

        class
          ↓
        temporal gate
          ↓
        spatial gate
          ↓
        motion gate
          ↓
        appearance
          ↓
        team
          ↓
        jersey
          ↓
        weighted score
          ↓
        evidence-strength validation
          ↓
        final decision
    """

    def __init__(
        self,
        max_temporal_gap: int = 75,
        max_spatial_distance: float = 350.0,
        max_motion_difference: float = 100.0,

        # ----------------------------------------------------
        # Appearance
        # ----------------------------------------------------

        minimum_appearance_similarity: float = 0.72,
        appearance_strong_similarity: float = 0.85,
        appearance_recent_embeddings: int = 20,
        appearance_average_top_k: int = 5,

        # ----------------------------------------------------
        # Weights
        # ----------------------------------------------------

        temporal_weight: float = 0.10,
        spatial_weight: float = 0.25,
        motion_weight: float = 0.15,
        appearance_weight: float = 0.43,
        team_weight: float = 0.04,
        jersey_weight: float = 0.03,

        # ----------------------------------------------------
        # Final decision
        # ----------------------------------------------------

        minimum_match_score: float = 0.60,

        # Weak appearance matches need stronger
        # geometric evidence.
        weak_appearance_min_score: float = 0.68,

        # Strong appearance matches can tolerate
        # slightly weaker geometry.
        strong_appearance_min_score: float = 0.55,

        # When appearance is weak, require this combined
        # geometric confidence.
        weak_appearance_geometry_score: float = 0.62,

    ) -> None:

        # ====================================================
        # VALIDATION
        # ====================================================

        if max_temporal_gap < 0:
            raise ValueError(
                "max_temporal_gap must be >= 0."
            )

        if max_spatial_distance <= 0:
            raise ValueError(
                "max_spatial_distance must be > 0."
            )

        if max_motion_difference <= 0:
            raise ValueError(
                "max_motion_difference must be > 0."
            )

        if not 0.0 <= minimum_appearance_similarity <= 1.0:
            raise ValueError(
                "minimum_appearance_similarity "
                "must be between 0 and 1."
            )

        if not 0.0 <= appearance_strong_similarity <= 1.0:
            raise ValueError(
                "appearance_strong_similarity "
                "must be between 0 and 1."
            )

        if (
            appearance_strong_similarity
            < minimum_appearance_similarity
        ):
            raise ValueError(
                "appearance_strong_similarity must be "
                ">= minimum_appearance_similarity."
            )

        if not 0.0 <= minimum_match_score <= 1.0:
            raise ValueError(
                "minimum_match_score must be "
                "between 0 and 1."
            )

        if not 0.0 <= weak_appearance_min_score <= 1.0:
            raise ValueError(
                "weak_appearance_min_score must be "
                "between 0 and 1."
            )

        if not 0.0 <= strong_appearance_min_score <= 1.0:
            raise ValueError(
                "strong_appearance_min_score must be "
                "between 0 and 1."
            )

        if not 0.0 <= weak_appearance_geometry_score <= 1.0:
            raise ValueError(
                "weak_appearance_geometry_score must be "
                "between 0 and 1."
            )

        # ====================================================
        # CONFIGURATION
        # ====================================================

        self.max_temporal_gap = int(
            max_temporal_gap
        )

        self.max_spatial_distance = float(
            max_spatial_distance
        )

        self.max_motion_difference = float(
            max_motion_difference
        )

        self.minimum_appearance_similarity = float(
            minimum_appearance_similarity
        )

        self.appearance_strong_similarity = float(
            appearance_strong_similarity
        )

        self.appearance_recent_embeddings = max(
            1,
            int(appearance_recent_embeddings),
        )

        self.appearance_average_top_k = max(
            1,
            int(appearance_average_top_k),
        )

        self.minimum_match_score = float(
            minimum_match_score
        )

        self.weak_appearance_min_score = float(
            weak_appearance_min_score
        )

        self.strong_appearance_min_score = float(
            strong_appearance_min_score
        )

        self.weak_appearance_geometry_score = float(
            weak_appearance_geometry_score
        )

        # ====================================================
        # NORMALIZE WEIGHTS
        # ====================================================

        weights = {
            "temporal": float(temporal_weight),
            "spatial": float(spatial_weight),
            "motion": float(motion_weight),
            "appearance": float(appearance_weight),
            "team": float(team_weight),
            "jersey": float(jersey_weight),
        }

        if any(
            weight < 0
            for weight in weights.values()
        ):
            raise ValueError(
                "Matcher weights cannot be negative."
            )

        total_weight = sum(
            weights.values()
        )

        if total_weight <= 0:
            raise ValueError(
                "Matcher weights must sum to "
                "a positive value."
            )

        self.temporal_weight = (
            weights["temporal"] / total_weight
        )

        self.spatial_weight = (
            weights["spatial"] / total_weight
        )

        self.motion_weight = (
            weights["motion"] / total_weight
        )

        self.appearance_weight = (
            weights["appearance"] / total_weight
        )

        self.team_weight = (
            weights["team"] / total_weight
        )

        self.jersey_weight = (
            weights["jersey"] / total_weight
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def match(
        self,
        track: Track,
        identity: Identity,
    ) -> MatchResult:
        """
        Compare Track against Identity.

        The matcher never mutates either object.
        """

        # ====================================================
        # BASIC VALIDATION
        # ====================================================

        if not track.observations:
            return self._failed_result(
                reason="track_has_no_observations"
            )

        if not identity.observations:
            return self._failed_result(
                reason="identity_has_no_observations"
            )

        # ====================================================
        # CLASS
        # ====================================================

        if track.class_name != identity.class_name:
            return self._failed_result(
                temporal_gap=self._temporal_gap(
                    track,
                    identity,
                ),
                reason="class_mismatch",
            )

        # ====================================================
        # TEMPORAL GATE
        # ====================================================

        temporal_gap = self._temporal_gap(
            track,
            identity,
        )

        # A global identity and new fragment
        # cannot overlap.
        if temporal_gap < 0:
            return self._failed_result(
                temporal_gap=temporal_gap,
                reason="track_overlaps_identity",
            )

        if temporal_gap > self.max_temporal_gap:
            return self._failed_result(
                temporal_gap=temporal_gap,
                reason="temporal_gap_too_large",
            )

        # ====================================================
        # SPATIAL GATE
        # ====================================================

        spatial_distance = self._spatial_distance(
            track,
            identity,
        )

        if spatial_distance == float("inf"):
            return self._failed_result(
                temporal_gap=temporal_gap,
                spatial_distance=spatial_distance,
                reason="spatial_information_unavailable",
            )

        if spatial_distance > self.max_spatial_distance:
            return self._failed_result(
                temporal_gap=temporal_gap,
                spatial_distance=spatial_distance,
                reason="spatial_distance_too_large",
            )

        # ====================================================
        # MOTION GATE
        # ====================================================

        motion_difference = self._motion_difference(
            track,
            identity,
            temporal_gap,
        )

        # Missing motion is neutral.
        if motion_difference == float("inf"):
            motion_difference = 0.0

        if motion_difference > self.max_motion_difference:
            return self._failed_result(
                temporal_gap=temporal_gap,
                spatial_distance=spatial_distance,
                motion_difference=motion_difference,
                reason="motion_difference_too_large",
            )

        # ====================================================
        # APPEARANCE
        # ====================================================

        appearance_similarity = (
            self._appearance_similarity(
                track,
                identity,
            )
        )

        # ====================================================
        # TEAM
        # ====================================================

        team_match = self._team_match(
            track,
            identity,
        )

        # ====================================================
        # JERSEY
        # ====================================================

        jersey_match = self._jersey_match(
            track,
            identity,
        )

        # ====================================================
        # NORMALIZED SIGNALS
        # ====================================================

        temporal_score = self._temporal_score(
            temporal_gap,
            self.max_temporal_gap,
        )

        spatial_score = self._distance_score(
            spatial_distance,
            self.max_spatial_distance,
        )

        motion_score = self._distance_score(
            motion_difference,
            self.max_motion_difference,
        )

        # ====================================================
        # GEOMETRIC EVIDENCE
        # ====================================================

        geometry_score = (
            temporal_score * self.temporal_weight
            + spatial_score * self.spatial_weight
            + motion_score * self.motion_weight
        )

        geometry_weight = (
            self.temporal_weight
            + self.spatial_weight
            + self.motion_weight
        )

        if geometry_weight > 0:
            geometry_score /= geometry_weight

        # ====================================================
        # DYNAMIC EVIDENCE
        # ====================================================

        components: list[
            tuple[str, float, float]
        ] = [
            (
                "temporal",
                temporal_score,
                self.temporal_weight,
            ),
            (
                "spatial",
                spatial_score,
                self.spatial_weight,
            ),
            (
                "motion",
                motion_score,
                self.motion_weight,
            ),
        ]

        # Appearance is optional.
        if appearance_similarity >= 0.0:
            components.append(
                (
                    "appearance",
                    appearance_similarity,
                    self.appearance_weight,
                )
            )

        # Team evidence is optional.
        if team_match is not None:
            components.append(
                (
                    "team",
                    1.0 if team_match else 0.0,
                    self.team_weight,
                )
            )

        # Jersey evidence is optional.
        if jersey_match is not None:
            components.append(
                (
                    "jersey",
                    1.0 if jersey_match else 0.0,
                    self.jersey_weight,
                )
            )

        total_weight = sum(
            weight
            for _, _, weight in components
        )

        score = (
            sum(
                value * weight
                for _, value, weight in components
            )
            / total_weight
            if total_weight > 0
            else 0.0
        )

        # ====================================================
        # EXPLICIT APPEARANCE MISMATCH
        # ====================================================

        if appearance_similarity >= 0.0:

            if (
                appearance_similarity
                < self.minimum_appearance_similarity
            ):
                return MatchResult(
                    matched=False,
                    score=score,
                    temporal_gap=temporal_gap,
                    spatial_distance=spatial_distance,
                    motion_difference=motion_difference,
                    appearance_similarity=(
                        appearance_similarity
                    ),
                    team_match=team_match,
                    jersey_match=jersey_match,
                    reason="appearance_mismatch",
                )

        # ====================================================
        # TEAM CONTRADICTION
        # ====================================================

        if team_match is False:

            return MatchResult(
                matched=False,
                score=score,
                temporal_gap=temporal_gap,
                spatial_distance=spatial_distance,
                motion_difference=motion_difference,
                appearance_similarity=(
                    appearance_similarity
                ),
                team_match=team_match,
                jersey_match=jersey_match,
                reason="team_mismatch",
            )

        # ====================================================
        # JERSEY CONTRADICTION
        # ====================================================

        if jersey_match is False:

            return MatchResult(
                matched=False,
                score=score,
                temporal_gap=temporal_gap,
                spatial_distance=spatial_distance,
                motion_difference=motion_difference,
                appearance_similarity=(
                    appearance_similarity
                ),
                team_match=team_match,
                jersey_match=jersey_match,
                reason="jersey_mismatch",
            )

        # ====================================================
        # APPEARANCE EVIDENCE STRENGTH
        # ====================================================

        if appearance_similarity < 0.0:

            # No appearance evidence.
            #
            # Because appearance is unavailable, require
            # stronger geometry than the normal threshold.
            if geometry_score < 0.70:

                return MatchResult(
                    matched=False,
                    score=score,
                    temporal_gap=temporal_gap,
                    spatial_distance=spatial_distance,
                    motion_difference=motion_difference,
                    appearance_similarity=(
                        appearance_similarity
                    ),
                    team_match=team_match,
                    jersey_match=jersey_match,
                    reason="insufficient_evidence",
                )

        elif (
            appearance_similarity
            >= self.appearance_strong_similarity
        ):

            # ------------------------------------------------
            # Strong appearance
            # ------------------------------------------------

            if score < self.strong_appearance_min_score:

                return MatchResult(
                    matched=False,
                    score=score,
                    temporal_gap=temporal_gap,
                    spatial_distance=spatial_distance,
                    motion_difference=motion_difference,
                    appearance_similarity=(
                        appearance_similarity
                    ),
                    team_match=team_match,
                    jersey_match=jersey_match,
                    reason="score_below_threshold",
                )

        else:

            # ------------------------------------------------
            # Weak/moderate appearance
            # ------------------------------------------------

            if (
                geometry_score
                < self.weak_appearance_geometry_score
            ):

                return MatchResult(
                    matched=False,
                    score=score,
                    temporal_gap=temporal_gap,
                    spatial_distance=spatial_distance,
                    motion_difference=motion_difference,
                    appearance_similarity=(
                        appearance_similarity
                    ),
                    team_match=team_match,
                    jersey_match=jersey_match,
                    reason="weak_appearance_geometry",
                )

            if score < self.weak_appearance_min_score:

                return MatchResult(
                    matched=False,
                    score=score,
                    temporal_gap=temporal_gap,
                    spatial_distance=spatial_distance,
                    motion_difference=motion_difference,
                    appearance_similarity=(
                        appearance_similarity
                    ),
                    team_match=team_match,
                    jersey_match=jersey_match,
                    reason="score_below_threshold",
                )

        # ====================================================
        # FINAL SCORE
        # ====================================================

        if score < self.minimum_match_score:

            return MatchResult(
                matched=False,
                score=score,
                temporal_gap=temporal_gap,
                spatial_distance=spatial_distance,
                motion_difference=motion_difference,
                appearance_similarity=(
                    appearance_similarity
                ),
                team_match=team_match,
                jersey_match=jersey_match,
                reason="score_below_threshold",
            )

        # ====================================================
        # MATCHED
        # ====================================================

        return MatchResult(
            matched=True,
            score=score,
            temporal_gap=temporal_gap,
            spatial_distance=spatial_distance,
            motion_difference=motion_difference,
            appearance_similarity=(
                appearance_similarity
            ),
            team_match=team_match,
            jersey_match=jersey_match,
            reason="compatible",
        )

    # ========================================================
    # FAILURE RESULT
    # ========================================================

    @staticmethod
    def _failed_result(
        temporal_gap: int = 0,
        spatial_distance: float = float("inf"),
        motion_difference: float = float("inf"),
        reason: str = "not_matched",
    ) -> MatchResult:

        return MatchResult(
            matched=False,
            score=0.0,
            temporal_gap=temporal_gap,
            spatial_distance=spatial_distance,
            motion_difference=motion_difference,
            appearance_similarity=-1.0,
            team_match=None,
            jersey_match=None,
            reason=reason,
        )

    # ========================================================
    # TEMPORAL
    # ========================================================

    @staticmethod
    def _temporal_gap(
        track: Track,
        identity: Identity,
    ) -> int:

        if identity.last_frame is None:
            return 0

        if track.first_frame is None:
            return 0

        return (
            track.first_frame
            - identity.last_frame
            - 1
        )

    @staticmethod
    def _temporal_score(
        temporal_gap: int,
        max_gap: int,
    ) -> float:

        if temporal_gap <= 0:
            return 1.0

        if max_gap <= 0:
            return 0.0

        return max(
            0.0,
            1.0
            - (
                temporal_gap / max_gap
            ),
        )

    # ========================================================
    # SPATIAL
    # ========================================================

    @staticmethod
    def _spatial_distance(
        track: Track,
        identity: Identity,
    ) -> float:

        last_identity = (
            identity.last_observation
        )

        first_track = (
            track.first_observation
        )

        if (
            last_identity is None
            or first_track is None
        ):
            return float("inf")

        x1, y1 = last_identity.position
        x2, y2 = first_track.position

        return sqrt(
            (x2 - x1) ** 2
            + (y2 - y1) ** 2
        )

    @staticmethod
    def _distance_score(
        distance: float,
        maximum: float,
    ) -> float:

        if maximum <= 0:
            return 0.0

        if distance < 0:
            return 0.0

        if distance >= maximum:
            return 0.0

        return 1.0 - (
            distance / maximum
        )

    # ========================================================
    # MOTION
    # ========================================================

    def _motion_difference(
        self,
        track: Track,
        identity: Identity,
        temporal_gap: int,
    ) -> float:

        identity_velocity = (
            self._estimate_end_velocity(
                identity
            )
        )

        track_velocity = (
            self._estimate_start_velocity(
                track
            )
        )

        # Motion unavailable is not a rejection.
        if identity_velocity is None:
            return 0.0

        if track_velocity is None:
            return 0.0

        ivx, ivy = identity_velocity
        tvx, tvy = track_velocity

        velocity_difference = sqrt(
            (tvx - ivx) ** 2
            + (tvy - ivy) ** 2
        )

        if temporal_gap > 0:

            last_identity = (
                identity.last_observation
            )

            first_track = (
                track.first_observation
            )

            if (
                last_identity is not None
                and first_track is not None
            ):

                expected_x = (
                    ivx
                    * (temporal_gap + 1)
                )

                expected_y = (
                    ivy
                    * (temporal_gap + 1)
                )

                actual_x = (
                    first_track.position[0]
                    - last_identity.position[0]
                )

                actual_y = (
                    first_track.position[1]
                    - last_identity.position[1]
                )

                displacement_difference = sqrt(
                    (
                        actual_x
                        - expected_x
                    ) ** 2
                    + (
                        actual_y
                        - expected_y
                    ) ** 2
                )

                return (
                    velocity_difference * 0.6
                    + displacement_difference * 0.4
                )

        return velocity_difference

    @staticmethod
    def _estimate_end_velocity(
        identity: Identity,
    ) -> Optional[
        tuple[float, float]
    ]:

        observations = identity.observations

        if len(observations) < 2:
            return None

        previous = observations[-2]
        current = observations[-1]

        dt = (
            current.frame_index
            - previous.frame_index
        )

        if dt <= 0:
            return None

        px, py = previous.position
        cx, cy = current.position

        return (
            (cx - px) / dt,
            (cy - py) / dt,
        )

    @staticmethod
    def _estimate_start_velocity(
        track: Track,
    ) -> Optional[
        tuple[float, float]
    ]:

        observations = track.observations

        if len(observations) < 2:
            return None

        first = observations[0]
        second = observations[1]

        dt = (
            second.frame_index
            - first.frame_index
        )

        if dt <= 0:
            return None

        px, py = first.position
        cx, cy = second.position

        return (
            (cx - px) / dt,
            (cy - py) / dt,
        )

    # ========================================================
    # APPEARANCE
    # ========================================================

    def _appearance_similarity(
        self,
        track: Track,
        identity: Identity,
    ) -> float:
        """
        Multi-embedding OSNet comparison.

        Returns:

            [0, 1] -> valid appearance
            -1     -> unavailable
        """

        track_embeddings = (
            self._collect_embeddings(
                track.observations
            )
        )

        identity_embeddings = (
            self._collect_embeddings(
                identity.observations
            )
        )

        if (
            not track_embeddings
            or not identity_embeddings
        ):
            return -1.0

        # Only use recent identity embeddings.
        identity_embeddings = (
            identity_embeddings[
                -self.appearance_recent_embeddings:
            ]
        )

        similarities: list[float] = []

        for track_embedding in track_embeddings:

            for identity_embedding in identity_embeddings:

                similarity = (
                    self._cosine_similarity(
                        track_embedding,
                        identity_embedding,
                    )
                )

                if similarity is None:
                    continue

                similarities.append(
                    similarity
                )

        if not similarities:
            return -1.0

        similarities.sort(
            reverse=True
        )

        strongest = similarities[0]

        # ====================================================
        # STRONG SINGLE OBSERVATION
        # ====================================================

        if (
            strongest
            >= self.appearance_strong_similarity
        ):
            return strongest

        # ====================================================
        # TOP-K CONSENSUS
        # ====================================================

        top_k = min(
            self.appearance_average_top_k,
            len(similarities),
        )

        strongest_similarities = (
            similarities[:top_k]
        )

        return (
            sum(strongest_similarities)
            / len(strongest_similarities)
        )

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    @staticmethod
    def _collect_embeddings(
        observations,
    ) -> list[list[float]]:

        embeddings: list[list[float]] = []

        for observation in observations:

            embedding = getattr(
                observation,
                "appearance_embedding",
                None,
            )

            if embedding is None:
                continue

            try:
                vector = [
                    float(value)
                    for value in embedding
                ]

            except (
                TypeError,
                ValueError,
            ):
                continue

            if not vector:
                continue

            # Reject NaN / infinity.
            if not all(
                value == value
                and abs(value) != float("inf")
                for value in vector
            ):
                continue

            norm = sqrt(
                sum(
                    value * value
                    for value in vector
                )
            )

            if norm <= 0:
                continue

            embeddings.append(
                vector
            )

        return embeddings

    # ========================================================
    # COSINE
    # ========================================================

    @staticmethod
    def _cosine_similarity(
        embedding_a: list[float],
        embedding_b: list[float],
    ) -> Optional[float]:

        if not embedding_a or not embedding_b:
            return None

        if len(embedding_a) != len(embedding_b):
            return None

        dot = sum(
            a * b
            for a, b in zip(
                embedding_a,
                embedding_b,
            )
        )

        norm_a = sqrt(
            sum(
                value * value
                for value in embedding_a
            )
        )

        norm_b = sqrt(
            sum(
                value * value
                for value in embedding_b
            )
        )

        if norm_a <= 0 or norm_b <= 0:
            return None

        similarity = (
            dot
            / (norm_a * norm_b)
        )

        return max(
            -1.0,
            min(
                1.0,
                similarity,
            ),
        )

    # ========================================================
    # TEAM
    # ========================================================

    @staticmethod
    def _team_match(
        track: Track,
        identity: Identity,
    ) -> Optional[bool]:

        track_team = (
            IdentityMatcher._most_recent_team(
                track
            )
        )

        identity_team = identity.team

        # No evidence is neutral.
        if (
            track_team is None
            or identity_team is None
        ):
            return None

        return (
            track_team == identity_team
        )

    @staticmethod
    def _most_recent_team(
        track: Track,
    ) -> Optional[str]:

        for observation in reversed(
            track.observations
        ):

            team = getattr(
                observation,
                "team",
                None,
            )

            if team is not None:
                return team

        return None

    # ========================================================
    # JERSEY
    # ========================================================

    @staticmethod
    def _jersey_match(
        track: Track,
        identity: Identity,
    ) -> Optional[bool]:

        track_number = (
            IdentityMatcher._most_recent_jersey(
                track
            )
        )

        identity_number = (
            identity.jersey_number
        )

        # No evidence is neutral.
        if (
            track_number is None
            or identity_number is None
        ):
            return None

        return (
            track_number == identity_number
        )

    @staticmethod
    def _most_recent_jersey(
        track: Track,
    ) -> Optional[int]:

        for observation in reversed(
            track.observations
        ):

            jersey_number = getattr(
                observation,
                "jersey_number",
                None,
            )

            if jersey_number is not None:
                return jersey_number

        return None

