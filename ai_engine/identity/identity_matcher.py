"""
Global Identity Matcher V5.1.

Compares a ByteTrack track fragment against an existing
global Identity.

V5.1 goals
----------
1. Hard class compatibility.
2. Hard temporal continuity gate.
3. Hard spatial continuity gate.
4. Motion consistency gate.
5. Appearance evaluated only after geometry passes.
6. Multi-embedding OSNet appearance matching.
7. Tiered appearance acceptance.
8. Weak appearance cannot rescue weak geometry.
9. Strong appearance may relax geometry slightly.
10. Long temporal gaps require stronger evidence.
11. Team and jersey evidence remain optional.
12. Matcher never mutates Track or Identity.
13. Explicit diagnostic reason codes.
14. Explicit appearance_evaluated flag.

Appearance result
-----------------
appearance_similarity:
    [0, 1] -> valid comparison
    -1.0    -> appearance unavailable

appearance_evaluated:
    True  -> matcher reached appearance stage
    False -> matcher rejected before appearance stage
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
    matched: bool
    score: float

    temporal_gap: int
    spatial_distance: float
    motion_difference: float

    appearance_similarity: float

    # IMPORTANT:
    # Distinguishes:
    #
    #   appearance not evaluated
    #       from
    #
    #   appearance evaluated but unavailable.
    appearance_evaluated: bool

    team_match: Optional[bool]
    jersey_match: Optional[bool]

    reason: str = ""


# ============================================================
# MATCHER
# ============================================================

class IdentityMatcher:

    VERSION = "v5.1"

    def __init__(
        self,

        # ----------------------------------------------------
        # HARD GEOMETRIC LIMITS
        # ----------------------------------------------------

        max_temporal_gap: int = 75,
        max_spatial_distance: float = 300.0,
        max_motion_difference: float = 80.0,

        # ----------------------------------------------------
        # APPEARANCE
        # ----------------------------------------------------

        minimum_appearance_similarity: float = 0.72,
        appearance_normal_similarity: float = 0.78,
        appearance_strong_similarity: float = 0.85,

        appearance_recent_embeddings: int = 20,
        appearance_average_top_k: int = 5,

        # ----------------------------------------------------
        # WEIGHTS
        # ----------------------------------------------------

        temporal_weight: float = 0.10,
        spatial_weight: float = 0.25,
        motion_weight: float = 0.20,
        appearance_weight: float = 0.35,
        team_weight: float = 0.07,
        jersey_weight: float = 0.03,

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        minimum_match_score: float = 0.60,

        # ----------------------------------------------------
        # GEOMETRY QUALITY
        # ----------------------------------------------------

        strong_spatial_ratio: float = 0.35,
        strong_motion_ratio: float = 0.40,

        # ----------------------------------------------------
        # LONG GAP
        # ----------------------------------------------------

        long_gap_ratio: float = 0.50,
        long_gap_min_appearance: float = 0.78,

        # ----------------------------------------------------
        # WEAK APPEARANCE
        # ----------------------------------------------------

        weak_appearance_max_spatial_ratio: float = 0.20,
        weak_appearance_max_motion_ratio: float = 0.25,
        weak_appearance_min_score: float = 0.72,

        # ----------------------------------------------------
        # STRONG APPEARANCE
        # ----------------------------------------------------

        strong_appearance_max_spatial_ratio: float = 0.85,
        strong_appearance_max_motion_ratio: float = 0.90,

        # ----------------------------------------------------
        # SAFETY
        # ----------------------------------------------------

        minimum_observations_for_motion: int = 2,
    ) -> None:

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

        appearance_thresholds = (
            minimum_appearance_similarity,
            appearance_normal_similarity,
            appearance_strong_similarity,
        )

        if any(
            not 0.0 <= value <= 1.0
            for value in appearance_thresholds
        ):
            raise ValueError(
                "Appearance thresholds must be between 0 and 1."
            )

        if not (
            minimum_appearance_similarity
            <= appearance_normal_similarity
            <= appearance_strong_similarity
        ):
            raise ValueError(
                "Appearance thresholds must satisfy "
                "minimum <= normal <= strong."
            )

        if not 0.0 <= minimum_match_score <= 1.0:
            raise ValueError(
                "minimum_match_score must be between 0 and 1."
            )

        ratios = (
            strong_spatial_ratio,
            strong_motion_ratio,
            long_gap_ratio,
            weak_appearance_max_spatial_ratio,
            weak_appearance_max_motion_ratio,
            strong_appearance_max_spatial_ratio,
            strong_appearance_max_motion_ratio,
        )

        if any(
            not 0.0 <= value <= 1.0
            for value in ratios
        ):
            raise ValueError(
                "Geometry ratios must be between 0 and 1."
            )

        if not 0.0 <= long_gap_min_appearance <= 1.0:
            raise ValueError(
                "long_gap_min_appearance must be between 0 and 1."
            )

        if not 0.0 <= weak_appearance_min_score <= 1.0:
            raise ValueError(
                "weak_appearance_min_score must be between 0 and 1."
            )

        if minimum_observations_for_motion < 2:
            raise ValueError(
                "minimum_observations_for_motion must be >= 2."
            )

        self.max_temporal_gap = int(max_temporal_gap)
        self.max_spatial_distance = float(max_spatial_distance)
        self.max_motion_difference = float(max_motion_difference)

        self.minimum_appearance_similarity = float(
            minimum_appearance_similarity
        )

        self.appearance_normal_similarity = float(
            appearance_normal_similarity
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

        self.strong_spatial_ratio = float(
            strong_spatial_ratio
        )

        self.strong_motion_ratio = float(
            strong_motion_ratio
        )

        self.long_gap_ratio = float(
            long_gap_ratio
        )

        self.long_gap_min_appearance = float(
            long_gap_min_appearance
        )

        self.weak_appearance_max_spatial_ratio = float(
            weak_appearance_max_spatial_ratio
        )

        self.weak_appearance_max_motion_ratio = float(
            weak_appearance_max_motion_ratio
        )

        self.weak_appearance_min_score = float(
            weak_appearance_min_score
        )

        self.strong_appearance_max_spatial_ratio = float(
            strong_appearance_max_spatial_ratio
        )

        self.strong_appearance_max_motion_ratio = float(
            strong_appearance_max_motion_ratio
        )

        self.minimum_observations_for_motion = int(
            minimum_observations_for_motion
        )

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

        total_weight = sum(weights.values())

        if total_weight <= 0:
            raise ValueError(
                "Matcher weights must sum to a positive value."
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

        if not track.observations:
            return self._failed_result(
                reason="track_has_no_observations"
            )

        if not identity.observations:
            return self._failed_result(
                reason="identity_has_no_observations"
            )

        # ----------------------------------------------------
        # CLASS
        # ----------------------------------------------------

        if track.class_name != identity.class_name:
            return self._failed_result(
                temporal_gap=self._temporal_gap(
                    track,
                    identity,
                ),
                reason="class_mismatch",
            )

        # ----------------------------------------------------
        # TEMPORAL
        # ----------------------------------------------------

        temporal_gap = self._temporal_gap(
            track,
            identity,
        )

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

        # ----------------------------------------------------
        # SPATIAL
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # MOTION
        # ----------------------------------------------------

        motion_difference = self._motion_difference(
            track,
            identity,
            temporal_gap,
        )

        # Missing motion is not a rejection.
        if motion_difference == float("inf"):
            motion_difference = 0.0

        # ----------------------------------------------------
        # NORMALIZED GEOMETRY
        # ----------------------------------------------------

        spatial_ratio = (
            spatial_distance
            / self.max_spatial_distance
        )

        motion_ratio = (
            motion_difference
            / self.max_motion_difference
        )

        spatial_score = self._distance_score(
            spatial_distance,
            self.max_spatial_distance,
        )

        motion_score = self._distance_score(
            motion_difference,
            self.max_motion_difference,
        )

        temporal_score = self._temporal_score(
            temporal_gap,
            self.max_temporal_gap,
        )

        # ----------------------------------------------------
        # HARD SPATIAL GATE
        # ----------------------------------------------------

        if spatial_distance > self.max_spatial_distance:
            return self._failed_result(
                temporal_gap=temporal_gap,
                spatial_distance=spatial_distance,
                motion_difference=motion_difference,
                reason="spatial_distance_too_large",
            )

        # ----------------------------------------------------
        # HARD MOTION GATE
        # ----------------------------------------------------

        if motion_difference > self.max_motion_difference:
            return self._failed_result(
                temporal_gap=temporal_gap,
                spatial_distance=spatial_distance,
                motion_difference=motion_difference,
                reason="motion_difference_too_large",
            )

        # ----------------------------------------------------
        # APPEARANCE
        # ----------------------------------------------------

        appearance_similarity = (
            self._appearance_similarity(
                track,
                identity,
            )
        )

        appearance_evaluated = True

        # ----------------------------------------------------
        # TEAM
        # ----------------------------------------------------

        team_match = self._team_match(
            track,
            identity,
        )

        # ----------------------------------------------------
        # JERSEY
        # ----------------------------------------------------

        jersey_match = self._jersey_match(
            track,
            identity,
        )

        # ----------------------------------------------------
        # APPEARANCE GATE
        # ----------------------------------------------------

        appearance_reason = (
            self._appearance_gate_reason(
                appearance_similarity=appearance_similarity,
                temporal_gap=temporal_gap,
                spatial_ratio=spatial_ratio,
                motion_ratio=motion_ratio,
                spatial_score=spatial_score,
                motion_score=motion_score,
            )
        )

        if appearance_reason is not None:

            score = self._calculate_score(
                temporal_score=temporal_score,
                spatial_score=spatial_score,
                motion_score=motion_score,
                appearance_similarity=appearance_similarity,
                team_match=team_match,
                jersey_match=jersey_match,
            )

            return MatchResult(
                matched=False,
                score=score,
                temporal_gap=temporal_gap,
                spatial_distance=spatial_distance,
                motion_difference=motion_difference,
                appearance_similarity=appearance_similarity,
                appearance_evaluated=appearance_evaluated,
                team_match=team_match,
                jersey_match=jersey_match,
                reason=appearance_reason,
            )

        # ----------------------------------------------------
        # TEAM HARD EVIDENCE
        # ----------------------------------------------------

        if team_match is False:

            score = self._calculate_score(
                temporal_score,
                spatial_score,
                motion_score,
                appearance_similarity,
                team_match,
                jersey_match,
            )

            return MatchResult(
                matched=False,
                score=score,
                temporal_gap=temporal_gap,
                spatial_distance=spatial_distance,
                motion_difference=motion_difference,
                appearance_similarity=appearance_similarity,
                appearance_evaluated=appearance_evaluated,
                team_match=team_match,
                jersey_match=jersey_match,
                reason="team_mismatch",
            )

        # ----------------------------------------------------
        # JERSEY HARD EVIDENCE
        # ----------------------------------------------------

        if jersey_match is False:

            score = self._calculate_score(
                temporal_score,
                spatial_score,
                motion_score,
                appearance_similarity,
                team_match,
                jersey_match,
            )

            return MatchResult(
                matched=False,
                score=score,
                temporal_gap=temporal_gap,
                spatial_distance=spatial_distance,
                motion_difference=motion_difference,
                appearance_similarity=appearance_similarity,
                appearance_evaluated=appearance_evaluated,
                team_match=team_match,
                jersey_match=jersey_match,
                reason="jersey_mismatch",
            )

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        score = self._calculate_score(
            temporal_score=temporal_score,
            spatial_score=spatial_score,
            motion_score=motion_score,
            appearance_similarity=appearance_similarity,
            team_match=team_match,
            jersey_match=jersey_match,
        )

        if score < self.minimum_match_score:

            return MatchResult(
                matched=False,
                score=score,
                temporal_gap=temporal_gap,
                spatial_distance=spatial_distance,
                motion_difference=motion_difference,
                appearance_similarity=appearance_similarity,
                appearance_evaluated=appearance_evaluated,
                team_match=team_match,
                jersey_match=jersey_match,
                reason="score_below_threshold",
            )

        return MatchResult(
            matched=True,
            score=score,
            temporal_gap=temporal_gap,
            spatial_distance=spatial_distance,
            motion_difference=motion_difference,
            appearance_similarity=appearance_similarity,
            appearance_evaluated=appearance_evaluated,
            team_match=team_match,
            jersey_match=jersey_match,
            reason=self._success_reason(
                appearance_similarity,
                temporal_gap,
                spatial_ratio,
                motion_ratio,
            ),
        )

    # ========================================================
    # SCORE
    # ========================================================

    def _calculate_score(
        self,
        temporal_score: float,
        spatial_score: float,
        motion_score: float,
        appearance_similarity: float,
        team_match: Optional[bool],
        jersey_match: Optional[bool],
    ) -> float:

        components = [
            (
                temporal_score,
                self.temporal_weight,
            ),
            (
                spatial_score,
                self.spatial_weight,
            ),
            (
                motion_score,
                self.motion_weight,
            ),
        ]

        if appearance_similarity >= 0.0:
            components.append(
                (
                    appearance_similarity,
                    self.appearance_weight,
                )
            )

        if team_match is not None:
            components.append(
                (
                    1.0 if team_match else 0.0,
                    self.team_weight,
                )
            )

        if jersey_match is not None:
            components.append(
                (
                    1.0 if jersey_match else 0.0,
                    self.jersey_weight,
                )
            )

        total_weight = sum(
            weight
            for _, weight in components
        )

        if total_weight <= 0:
            return 0.0

        return sum(
            value * weight
            for value, weight in components
        ) / total_weight

    # ========================================================
    # APPEARANCE GATE
    # ========================================================

    def _appearance_gate_reason(
        self,
        appearance_similarity: float,
        temporal_gap: int,
        spatial_ratio: float,
        motion_ratio: float,
        spatial_score: float,
        motion_score: float,
    ) -> Optional[str]:

        if appearance_similarity < 0.0:

            if (
                spatial_ratio
                <= self.strong_spatial_ratio
                and motion_ratio
                <= self.strong_motion_ratio
            ):
                return None

            return (
                "appearance_unavailable_geometry_not_strong"
            )

        if (
            appearance_similarity
            < self.minimum_appearance_similarity
        ):
            return "appearance_mismatch"

        gap_ratio = (
            temporal_gap / self.max_temporal_gap
            if self.max_temporal_gap > 0
            else 0.0
        )

        if gap_ratio >= self.long_gap_ratio:

            if (
                appearance_similarity
                < self.long_gap_min_appearance
            ):
                return "long_gap_weak_appearance"

            if spatial_ratio > 0.70:
                return "long_gap_spatial_uncertainty"

            if motion_ratio > 0.70:
                return "long_gap_motion_uncertainty"

        # ----------------------------------------------------
        # Strong appearance
        # ----------------------------------------------------

        if (
            appearance_similarity
            >= self.appearance_strong_similarity
        ):

            if (
                spatial_ratio
                <= self.strong_appearance_max_spatial_ratio
                and motion_ratio
                <= self.strong_appearance_max_motion_ratio
            ):
                return None

            return (
                "strong_appearance_geometry_too_weak"
            )

        # ----------------------------------------------------
        # Normal appearance
        # ----------------------------------------------------

        if (
            appearance_similarity
            >= self.appearance_normal_similarity
        ):

            if (
                spatial_ratio <= 0.70
                and motion_ratio <= 0.70
            ):
                return None

            return (
                "normal_appearance_geometry_too_weak"
            )

        # ----------------------------------------------------
        # Weak appearance
        # ----------------------------------------------------

        if (
            appearance_similarity
            >= self.minimum_appearance_similarity
        ):

            if (
                spatial_ratio
                <= self.weak_appearance_max_spatial_ratio
                and motion_ratio
                <= self.weak_appearance_max_motion_ratio
                and spatial_score >= 0.80
                and motion_score >= 0.75
            ):
                return None

            return "weak_appearance_geometry"

        return "appearance_mismatch"

    # ========================================================
    # SUCCESS REASON
    # ========================================================

    def _success_reason(
        self,
        appearance_similarity: float,
        temporal_gap: int,
        spatial_ratio: float,
        motion_ratio: float,
    ) -> str:

        if (
            appearance_similarity
            >= self.appearance_strong_similarity
        ):

            if temporal_gap == 0:
                return "strong_appearance_continuity"

            return "strong_appearance_temporal_continuity"

        if (
            appearance_similarity
            >= self.appearance_normal_similarity
        ):
            return "normal_appearance_geometry"

        return "weak_appearance_strong_geometry"

    # ========================================================
    # FAILURE
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
            appearance_evaluated=False,
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
            1.0 - (
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

        last_identity = identity.last_observation
        first_track = track.first_observation

        if (
            last_identity is None
            or first_track is None
        ):
            return float("inf")

        try:
            x1, y1 = last_identity.position
            x2, y2 = first_track.position
        except (
            TypeError,
            ValueError,
        ):
            return float("inf")

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
                    ivx * (temporal_gap + 1)
                )

                expected_y = (
                    ivy * (temporal_gap + 1)
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
                    velocity_difference * 0.60
                    + displacement_difference * 0.40
                )

        return velocity_difference

    @staticmethod
    def _estimate_end_velocity(
        identity: Identity,
    ) -> Optional[tuple[float, float]]:

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
    ) -> Optional[tuple[float, float]]:

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

        identity_embeddings = (
            identity_embeddings[
                -self.appearance_recent_embeddings:
            ]
        )

        similarities = []

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

                # Keep the existing V5 normalized
                # [0, 1] representation.
                normalized = (
                    similarity + 1.0
                ) / 2.0

                similarities.append(
                    normalized
                )

        if not similarities:
            return -1.0

        similarities.sort(reverse=True)

        strongest = similarities[0]

        if (
            strongest
            >= self.appearance_strong_similarity
        ):
            return strongest

        top_k = min(
            self.appearance_average_top_k,
            len(similarities),
        )

        return (
            sum(
                similarities[:top_k]
            )
            / top_k
        )

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    @staticmethod
    def _collect_embeddings(
        observations,
    ) -> list[list[float]]:

        embeddings = []

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

            embeddings.append(vector)

        return embeddings

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
            dot / (norm_a * norm_b)
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

        if (
            track_team is None
            or identity_team is None
        ):
            return None

        return track_team == identity_team

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

        identity_number = identity.jersey_number

        if (
            track_number is None
            or identity_number is None
        ):
            return None

        return track_number == identity_number

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