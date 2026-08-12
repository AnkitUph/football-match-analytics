"""
Home/Away Team Assigner.

Purpose:
Assign detected players to HOME or AWAY based on
dominant jersey-color appearance.

This version focuses on robust team-color classification
rather than continuous player identity tracking.

The classifier:
1. Extracts the torso region of each player.
2. Removes pitch/grass contamination.
3. Removes very dark and very low-saturation pixels.
4. Uses robust HSV statistics for jersey appearance.
5. Learns two team color clusters using K-Means.
6. Uses circular hue distance during classification.
7. Uses saturation/value information together with hue.
8. Rejects ambiguous classifications when the two teams
   are too visually similar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


# =========================================================
# DATA CLASS
# =========================================================

@dataclass
class TeamAssignment:
    player_id: int
    team: str
    confidence: float
    color_distance_home: float
    color_distance_away: float
    sample_count: int


# =========================================================
# TEAM ASSIGNER
# =========================================================

class TeamAssigner:
    """
    Assign players to HOME / AWAY using jersey-color
    features.

    The assigner first learns two dominant team colors
    from player crops and then classifies individual
    players against those learned colors.
    """

    HOME = "HOME"
    AWAY = "AWAY"
    UNKNOWN = "UNKNOWN"

    def __init__(
        self,
        min_confidence: float = 0.58,
        min_margin: float = 0.08,
        sample_size: int = 300,
    ) -> None:

        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.sample_size = int(sample_size)

        self.home_color: Optional[np.ndarray] = None
        self.away_color: Optional[np.ndarray] = None

        self.history: Dict[int, List[str]] = {}

    # =====================================================
    # COLOR EXTRACTION
    # =====================================================

    @staticmethod
    def _extract_jersey_color(
        crop: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Extract a robust HSV jersey-color descriptor.

        Returns:
            [hue, saturation, value]
        """

        if crop is None or crop.size == 0:
            return None

        h, w = crop.shape[:2]

        if h < 20 or w < 10:
            return None

        # -------------------------------------------------
        # Focus on torso.
        #
        # Avoid:
        # - head
        # - legs/shorts
        # - shoes
        # - most background
        # -------------------------------------------------

        x1 = int(w * 0.20)
        x2 = int(w * 0.80)

        y1 = int(h * 0.15)
        y2 = int(h * 0.62)

        torso = crop[y1:y2, x1:x2]

        if torso.size == 0:
            return None

        # -------------------------------------------------
        # Convert to HSV
        # -------------------------------------------------

        hsv = cv2.cvtColor(
            torso,
            cv2.COLOR_BGR2HSV,
        )

        pixels = hsv.reshape(-1, 3)

        if len(pixels) < 20:
            return None

        # -------------------------------------------------
        # Remove very dark pixels.
        #
        # These are often:
        # - shadows
        # - shorts
        # - outlines
        # - background
        # -------------------------------------------------

        valid = pixels[
            pixels[:, 2] > 45
        ]

        if len(valid) < 20:
            return None

        # -------------------------------------------------
        # Remove very low saturation pixels.
        #
        # Gray/white background pixels do not provide
        # useful team-color information.
        #
        # We keep some low-saturation pixels because
        # white jerseys can naturally have low saturation.
        # -------------------------------------------------

        valid = valid[
            valid[:, 1] > 25
        ]

        if len(valid) < 20:
            return None

        # -------------------------------------------------
        # Remove grass/pitch pixels.
        #
        # OpenCV HSV hue range:
        # H = 0..179
        #
        # Grass is commonly around 30..95.
        # We only remove strongly saturated green pixels.
        # -------------------------------------------------

        green_mask = (
            (valid[:, 0] >= 30)
            & (valid[:, 0] <= 95)
            & (valid[:, 1] > 60)
            & (valid[:, 2] > 40)
        )

        valid = valid[~green_mask]

        if len(valid) < 20:
            return None

        # -------------------------------------------------
        # Remove extreme dark/saturated outliers.
        #
        # This reduces the effect of tiny regions such as
        # logos, shadows and background pixels.
        # -------------------------------------------------

        s_low = np.percentile(valid[:, 1], 10)
        s_high = np.percentile(valid[:, 1], 90)

        v_low = np.percentile(valid[:, 2], 10)
        v_high = np.percentile(valid[:, 2], 90)

        filtered = valid[
            (valid[:, 1] >= s_low)
            & (valid[:, 1] <= s_high)
            & (valid[:, 2] >= v_low)
            & (valid[:, 2] <= v_high)
        ]

        if len(filtered) < 20:
            filtered = valid

        # -------------------------------------------------
        # Median is robust against remaining outliers.
        # -------------------------------------------------

        hue = float(
            np.median(filtered[:, 0])
        )

        saturation = float(
            np.median(filtered[:, 1])
        )

        value = float(
            np.median(filtered[:, 2])
        )

        return np.array(
            [
                hue,
                saturation,
                value,
            ],
            dtype=np.float32,
        )

    # =====================================================
    # FEATURE CONVERSION
    # =====================================================

    @staticmethod
    def _hsv_to_feature(
        hsv_color: np.ndarray,
    ) -> np.ndarray:
        """
        Convert HSV into a feature representation suitable
        for clustering.

        Hue is circular, so represent it as sin/cos.

        This avoids the problem where:
            H=179
        and:
            H=1

        are actually very close colors but numerically
        appear far apart.
        """

        hue = float(
            hsv_color[0]
        )

        saturation = float(
            hsv_color[1]
        )

        value = float(
            hsv_color[2]
        )

        angle = (
            2.0
            * np.pi
            * hue
            / 180.0
        )

        hue_x = np.cos(angle)
        hue_y = np.sin(angle)

        # Normalize S/V to approximately 0..1.
        s = saturation / 255.0
        v = value / 255.0

        return np.array(
            [
                hue_x,
                hue_y,
                s,
                v,
            ],
            dtype=np.float32,
        )

    # =====================================================
    # COLOR DISTANCE
    # =====================================================

    @staticmethod
    def _color_distance(
        color_a: np.ndarray,
        color_b: np.ndarray,
    ) -> float:
        """
        Compute weighted HSV distance.

        Hue receives the highest importance because jersey
        hue is generally the strongest team signal.

        Saturation and value still matter because lighting
        can change brightness and color strength.
        """

        h1 = float(color_a[0])
        s1 = float(color_a[1])
        v1 = float(color_a[2])

        h2 = float(color_b[0])
        s2 = float(color_b[1])
        v2 = float(color_b[2])

        # ---------------------------------------------
        # Circular hue difference.
        #
        # Hue range = 0..179
        # ---------------------------------------------

        hue_diff = abs(
            h1 - h2
        )

        hue_diff = min(
            hue_diff,
            180.0 - hue_diff,
        )

        hue_diff /= 90.0

        # ---------------------------------------------
        # Normalize saturation/value.
        # ---------------------------------------------

        saturation_diff = (
            abs(s1 - s2)
            / 255.0
        )

        value_diff = (
            abs(v1 - v2)
            / 255.0
        )

        # ---------------------------------------------
        # Weighted distance.
        #
        # Hue is strongest.
        # Saturation is second.
        # Value is weakest because lighting can change it.
        # ---------------------------------------------

        distance = (
            0.65 * hue_diff
            + 0.25 * saturation_diff
            + 0.10 * value_diff
        )

        return float(distance)

    # =====================================================
    # LEARNING TEAM COLORS
    # =====================================================

    def fit(
        self,
        player_crops: List[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Learn two dominant team colors.

        Each crop produces one robust jersey descriptor.

        K-Means is performed on circular-hue features
        rather than raw HSV values.
        """

        colors: List[np.ndarray] = []

        # -------------------------------------------------
        # Respect sample_size.
        # -------------------------------------------------

        selected_crops = player_crops[
            : self.sample_size
        ]

        for crop in selected_crops:

            color = self._extract_jersey_color(
                crop
            )

            if color is not None:
                colors.append(color)

        if len(colors) < 2:
            raise ValueError(
                "Not enough valid player crops to "
                "learn team colors."
            )

        colors_array = np.asarray(
            colors,
            dtype=np.float32,
        )

        # -------------------------------------------------
        # Convert HSV descriptors into circular hue
        # features.
        # -------------------------------------------------

        features = np.asarray(
            [
                self._hsv_to_feature(color)
                for color in colors_array
            ],
            dtype=np.float32,
        )

        # -------------------------------------------------
        # K-Means.
        # -------------------------------------------------

        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.001,
        )

        _, labels, centers = cv2.kmeans(
            features,
            2,
            None,
            criteria,
            20,
            cv2.KMEANS_PP_CENTERS,
        )

        labels = labels.reshape(-1)

        counts = np.bincount(
            labels,
            minlength=2,
        )

        # -------------------------------------------------
        # Determine which original HSV colors belong
        # to each cluster.
        # -------------------------------------------------

        cluster_colors = []

        for cluster_index in range(2):

            cluster_samples = colors_array[
                labels == cluster_index
            ]

            if len(cluster_samples) == 0:
                cluster_colors.append(
                    np.zeros(
                        3,
                        dtype=np.float32,
                    )
                )
                continue

            # Median HSV of the cluster.
            cluster_color = np.median(
                cluster_samples,
                axis=0,
            ).astype(np.float32)

            cluster_colors.append(
                cluster_color
            )

        # -------------------------------------------------
        # Largest cluster becomes HOME.
        #
        # IMPORTANT:
        # HOME/AWAY are labels, not semantic identities.
        # The system is learning two teams from the video.
        # -------------------------------------------------

        order = np.argsort(
            -counts
        )

        home_idx = int(
            order[0]
        )

        away_idx = int(
            order[1]
        )

        self.home_color = cluster_colors[
            home_idx
        ]

        self.away_color = cluster_colors[
            away_idx
        ]

        # -------------------------------------------------
        # Calculate cluster separation.
        # -------------------------------------------------

        separation = self._color_distance(
            self.home_color,
            self.away_color,
        )

        return {
            "samples": len(colors),
            "home_samples": int(
                counts[home_idx]
            ),
            "away_samples": int(
                counts[away_idx]
            ),
            "home_color_hsv": (
                self.home_color.tolist()
            ),
            "away_color_hsv": (
                self.away_color.tolist()
            ),
            "cluster_separation": float(
                separation
            ),
        }

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    def assign(
        self,
        player_id: int,
        crop: np.ndarray,
    ) -> TeamAssignment:
        """
        Classify one player crop as HOME/AWAY/UNKNOWN.
        """

        if (
            self.home_color is None
            or self.away_color is None
        ):
            raise RuntimeError(
                "TeamAssigner has not been fitted."
            )

        color = self._extract_jersey_color(
            crop
        )

        if color is None:

            return TeamAssignment(
                player_id=int(player_id),
                team=self.UNKNOWN,
                confidence=0.0,
                color_distance_home=999.0,
                color_distance_away=999.0,
                sample_count=0,
            )

        # -------------------------------------------------
        # Compare against both learned teams.
        # -------------------------------------------------

        distance_home = self._color_distance(
            color,
            self.home_color,
        )

        distance_away = self._color_distance(
            color,
            self.away_color,
        )

        # -------------------------------------------------
        # Select closest team.
        # -------------------------------------------------

        if distance_home <= distance_away:

            team = self.HOME
            best_distance = distance_home
            other_distance = distance_away

        else:

            team = self.AWAY
            best_distance = distance_away
            other_distance = distance_home

        # -------------------------------------------------
        # Convert distance into confidence.
        #
        # If the selected team is much closer than the
        # other team, confidence is high.
        # -------------------------------------------------

        total_distance = (
            distance_home
            + distance_away
            + 1e-6
        )

        confidence = (
            1.0
            - best_distance
            / total_distance
        )

        # -------------------------------------------------
        # Calculate relative margin.
        #
        # Example:
        #
        # HOME = 0.20
        # AWAY = 0.21
        #
        # This is ambiguous even though HOME is technically
        # closer.
        # -------------------------------------------------

        margin = (
            other_distance
            - best_distance
        ) / (
            other_distance
            + best_distance
            + 1e-6
        )

        # -------------------------------------------------
        # Reject uncertain classifications.
        #
        # This is intentional:
        #
        # WRONG TEAM
        #
        # is worse for the presentation than:
        #
        # UNKNOWN
        # -------------------------------------------------

        if (
            confidence < self.min_confidence
            or margin < self.min_margin
        ):
            team = self.UNKNOWN

        # -------------------------------------------------
        # Store history.
        #
        # TeamHistory in the runner performs the actual
        # temporal stabilization.
        # -------------------------------------------------

        self.history.setdefault(
            int(player_id),
            [],
        ).append(team)

        return TeamAssignment(
            player_id=int(player_id),
            team=team,
            confidence=float(
                confidence
            ),
            color_distance_home=float(
                distance_home
            ),
            color_distance_away=float(
                distance_away
            ),
            sample_count=1,
        )

    # =====================================================
    # TEMPORAL SMOOTHING
    # =====================================================

    def get_stable_team(
        self,
        player_id: int,
    ) -> str:
        """
        Return the majority team assignment for a player ID.

        This is retained for compatibility with the existing
        code. The main temporal stabilization is performed
        by TeamHistory in team_assignment_runner.py.
        """

        history = self.history.get(
            int(player_id),
            [],
        )

        valid = [
            team
            for team in history
            if team != self.UNKNOWN
        ]

        if not valid:
            return self.UNKNOWN

        home_count = valid.count(
            self.HOME
        )

        away_count = valid.count(
            self.AWAY
        )

        if home_count > away_count:
            return self.HOME

        if away_count > home_count:
            return self.AWAY

        return self.UNKNOWN

    # =====================================================
    # STATISTICS
    # =====================================================

    def get_team_statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Return team-assignment statistics.
        """

        result = {
            self.HOME: 0,
            self.AWAY: 0,
            self.UNKNOWN: 0,
            "total": 0,
        }

        for player_id in self.history:

            team = self.get_stable_team(
                player_id
            )

            result[team] += 1
            result["total"] += 1

        return result