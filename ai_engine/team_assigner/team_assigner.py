import colorsys
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

HUE_WEIGHT = 0.7
SATURATION_WEIGHT = 0.2
VALUE_WEIGHT = 0.1

GK_MATCH_THRESHOLD = 0.45

MIN_VALID_COLOR_PIXELS = 8
MAX_TRACK_COLOR_HISTORY = 40

WHITE_SATURATION_MAX = 0.25
WHITE_VALUE_MIN = 0.55

GREEN_SATURATION_MIN = 0.30
GREEN_VALUE_MIN = 0.18
GREEN_HUE_MIN = 35.0
GREEN_HUE_MAX = 155.0

PLAYER_CONFIDENCE_BIAS = 0.05

GK_SPATIAL_WEIGHT = 0.55
GK_COLOR_WEIGHT = 0.45

TEMPORAL_HOME_MARGIN = 0.08
TEMPORAL_AWAY_MARGIN = 0.08


def hex_to_hsv(hex_color):
    if not hex_color:
        return None

    hex_color = hex_color.lstrip("#")

    if len(hex_color) != 6:
        return None

    try:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
    except ValueError:
        return None

    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    return h * 360.0, s, v


def _bgr_to_hsv_deg(bgr):
    b, g, r = [float(x) for x in bgr]

    h, s, v = colorsys.rgb_to_hsv(
        np.clip(r / 255.0, 0.0, 1.0),
        np.clip(g / 255.0, 0.0, 1.0),
        np.clip(b / 255.0, 0.0, 1.0),
    )

    return h * 360.0, s, v


def _hsv_distance(hsv1, hsv2):
    if hsv1 is None or hsv2 is None:
        return 1.0

    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2

    dh = min(
        abs(h1 - h2),
        360.0 - abs(h1 - h2),
    ) / 180.0

    ds = abs(s1 - s2)
    dv = abs(v1 - v2)

    hue_reliability = min(s1, s2)

    effective_hue_weight = (
        HUE_WEIGHT * hue_reliability
    )

    remaining = 1.0 - effective_hue_weight

    other_total = (
        SATURATION_WEIGHT
        + VALUE_WEIGHT
    )

    eff_sat = (
        remaining
        * SATURATION_WEIGHT
        / other_total
    )

    eff_val = (
        remaining
        * VALUE_WEIGHT
        / other_total
    )

    return (
        dh * effective_hue_weight
        + ds * eff_sat
        + dv * eff_val
    )


def _hue_distance(h1, h2):
    return min(
        abs(h1 - h2),
        360.0 - abs(h1 - h2),
    )


class TeamAssigner:

    def __init__(
        self,
        home_color_hex=None,
        away_color_hex=None,
        home_gk_color_hex=None,
        away_gk_color_hex=None,
    ):
        self.home_hsv = hex_to_hsv(
            home_color_hex
        )

        self.away_hsv = hex_to_hsv(
            away_color_hex
        )

        self.home_gk_hsv = hex_to_hsv(
            home_gk_color_hex
        )

        self.away_gk_hsv = hex_to_hsv(
            away_gk_color_hex
        )

        self.team_colors_bgr = {}
        self.cluster_label_map = {}

        self.player_team_cache = {}

        self.player_color_history = {}

        self.player_team_votes = {}

        self.goalkeeper_team_cache = {}

        self.goalkeeper_color_history = {}

        self.player_positions = {}

        self.team_position_history = {
            "home": [],
            "away": [],
        }

        self.initialized = False

        logger.info(
            "TeamAssigner V10 initialized"
        )

        logger.info(
            "Home kit HSV: %s",
            self.home_hsv,
        )

        logger.info(
            "Away kit HSV: %s",
            self.away_hsv,
        )

        logger.info(
            "Home GK HSV: %s",
            self.home_gk_hsv,
        )

        logger.info(
            "Away GK HSV: %s",
            self.away_gk_hsv,
        )

    def _crop_torso(
        self,
        frame,
        bbox,
    ):
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]

        x1, y1, x2, y2 = [
            int(round(v))
            for v in bbox
        ]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))

        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        box_w = x2 - x1
        box_h = y2 - y1

        torso_x1 = int(
            x1 + box_w * 0.18
        )

        torso_x2 = int(
            x2 - box_w * 0.18
        )

        torso_y1 = int(
            y1 + box_h * 0.18
        )

        torso_y2 = int(
            y1 + box_h * 0.58
        )

        torso_x1 = max(
            x1,
            torso_x1,
        )

        torso_x2 = min(
            x2,
            torso_x2,
        )

        torso_y1 = max(
            y1,
            torso_y1,
        )

        torso_y2 = min(
            y2,
            torso_y2,
        )

        if (
            torso_x2 <= torso_x1
            or torso_y2 <= torso_y1
        ):
            return None

        crop = frame[
            torso_y1:torso_y2,
            torso_x1:torso_x2,
        ]

        if crop.size == 0:
            return None

        return crop

    def _get_valid_hsv_pixels(
        self,
        crop,
    ):
        if crop is None or crop.size == 0:
            return np.empty(
                (0, 3),
                dtype=np.float32,
            )

        hsv = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2HSV,
        )

        pixels = hsv.reshape(
            -1,
            3,
        ).astype(
            np.float32
        )

        if len(pixels) == 0:
            return pixels

        h = pixels[:, 0] * 2.0
        s = pixels[:, 1] / 255.0
        v = pixels[:, 2] / 255.0

        valid = (
            (v > 0.12)
            & (v < 0.98)
        )

        filtered = np.column_stack(
            (
                h,
                s,
                v,
            )
        )

        return filtered[valid]

    def _white_score(
        self,
        hsv_pixels,
    ):
        if len(hsv_pixels) == 0:
            return 0.0

        s = hsv_pixels[:, 1]
        v = hsv_pixels[:, 2]

        white = (
            (s <= WHITE_SATURATION_MAX)
            & (v >= WHITE_VALUE_MIN)
        )

        return float(
            np.mean(white)
        )

    def _green_score(
        self,
        hsv_pixels,
    ):
        if len(hsv_pixels) == 0:
            return 0.0

        h = hsv_pixels[:, 0]
        s = hsv_pixels[:, 1]
        v = hsv_pixels[:, 2]

        hue_mask = (
            (h >= GREEN_HUE_MIN)
            & (h <= GREEN_HUE_MAX)
        )

        green = (
            hue_mask
            & (s >= GREEN_SATURATION_MIN)
            & (v >= GREEN_VALUE_MIN)
        )

        return float(
            np.mean(green)
        )

    def _reference_score(
        self,
        hsv_pixels,
        reference_hsv,
    ):
        if (
            reference_hsv is None
            or len(hsv_pixels) == 0
        ):
            return 1.0

        distances = []

        for pixel in hsv_pixels:
            distances.append(
                _hsv_distance(
                    tuple(pixel),
                    reference_hsv,
                )
            )

        if not distances:
            return 1.0

        distances = np.asarray(
            distances,
            dtype=np.float32,
        )

        distances = np.sort(
            distances
        )

        keep_count = max(
            1,
            int(
                len(distances) * 0.35
            ),
        )

        return float(
            np.mean(
                distances[:keep_count]
            )
        )

    def _get_player_color(
        self,
        frame,
        bbox,
    ):
        crop = self._crop_torso(
            frame,
            bbox,
        )

        if crop is None:
            return None

        hsv_pixels = (
            self._get_valid_hsv_pixels(
                crop
            )
        )

        if (
            len(hsv_pixels)
            < MIN_VALID_COLOR_PIXELS
        ):
            return None

        white_score = (
            self._white_score(
                hsv_pixels
            )
        )

        green_score = (
            self._green_score(
                hsv_pixels
            )
        )

        if (
            self.home_hsv is not None
            and self.away_hsv is not None
        ):
            home_h, home_s, home_v = (
                self.home_hsv
            )

            away_h, away_s, away_v = (
                self.away_hsv
            )

            if (
                home_s < 0.20
                and home_v > 0.75
                and away_s > 0.45
            ):
                if white_score >= 0.20:
                    selected = hsv_pixels[
                        (
                            hsv_pixels[:, 1]
                            <= 0.30
                        )
                        & (
                            hsv_pixels[:, 2]
                            >= 0.50
                        )
                    ]

                    if len(selected) > 0:
                        median_hsv = np.median(
                            selected,
                            axis=0,
                        )

                        return self._hsv_to_bgr(
                            median_hsv
                        )

                if green_score >= 0.20:
                    selected = hsv_pixels[
                        (
                            hsv_pixels[:, 1]
                            >= 0.25
                        )
                        & (
                            hsv_pixels[:, 2]
                            >= 0.15
                        )
                    ]

                    if len(selected) > 0:
                        distances = np.asarray(
                            [
                                _hue_distance(
                                    float(pixel[0]),
                                    away_h,
                                )
                                for pixel in selected
                            ]
                        )

                        order = np.argsort(
                            distances
                        )

                        keep = selected[
                            order[
                                :max(
                                    1,
                                    int(
                                        len(order)
                                        * 0.35
                                    ),
                                )
                            ]
                        ]

                        median_hsv = np.median(
                            keep,
                            axis=0,
                        )

                        return self._hsv_to_bgr(
                            median_hsv
                        )

        median_hsv = np.median(
            hsv_pixels,
            axis=0,
        )

        return self._hsv_to_bgr(
            median_hsv
        )

    def _hsv_to_bgr(
        self,
        hsv,
    ):
        h, s, v = [
            float(x)
            for x in hsv
        ]

        rgb = colorsys.hsv_to_rgb(
            (h % 360.0) / 360.0,
            np.clip(s, 0.0, 1.0),
            np.clip(v, 0.0, 1.0),
        )

        r = int(
            np.clip(
                rgb[0] * 255,
                0,
                255,
            )
        )

        g = int(
            np.clip(
                rgb[1] * 255,
                0,
                255,
            )
        )

        b = int(
            np.clip(
                rgb[2] * 255,
                0,
                255,
            )
        )

        return np.array(
            [b, g, r],
            dtype=np.float32,
        )

    def _player_reference_distance(
        self,
        color,
        team,
    ):
        hsv = _bgr_to_hsv_deg(
            color
        )

        if team == "home":
            reference = self.home_hsv
        else:
            reference = self.away_hsv

        if reference is None:
            return 1.0

        return _hsv_distance(
            hsv,
            reference,
        )

    def _specific_team_score(
        self,
        color,
    ):
        hsv = _bgr_to_hsv_deg(
            color
        )

        home_distance = (
            self._player_reference_distance(
                color,
                "home",
            )
        )

        away_distance = (
            self._player_reference_distance(
                color,
                "away",
            )
        )

        h, s, v = hsv

        home_score = (
            1.0 - home_distance
        )

        away_score = (
            1.0 - away_distance
        )

        if (
            self.home_hsv is not None
            and self.home_hsv[1] < 0.20
            and self.home_hsv[2] > 0.75
        ):
            white_strength = (
                max(
                    0.0,
                    1.0 - s / 0.30,
                )
                * max(
                    0.0,
                    min(
                        1.0,
                        (v - 0.45)
                        / 0.45,
                    ),
                )
            )

            home_score += (
                0.40
                * white_strength
            )

        if (
            self.away_hsv is not None
            and self.away_hsv[1] > 0.40
        ):
            hue_distance = (
                _hue_distance(
                    h,
                    self.away_hsv[0],
                )
                / 180.0
            )

            green_strength = (
                max(
                    0.0,
                    1.0 - hue_distance,
                )
                * min(
                    1.0,
                    s / 0.55,
                )
            )

            away_score += (
                0.40
                * green_strength
            )

        return (
            home_score,
            away_score,
        )

    def assign_team_colors(
        self,
        frame,
        player_tracks_in_frame,
    ):
        player_colors = []

        for (
            track_id,
            info,
        ) in player_tracks_in_frame.items():

            color = self._get_player_color(
                frame,
                info["bbox"],
            )

            if color is None:
                continue

            player_colors.append(
                color
            )

        logger.info(
            "V10 team-color initialization: "
            "%d valid player observations",
            len(player_colors),
        )

        if len(player_colors) < 2:
            logger.warning(
                "Not enough valid player colors "
                "to initialize team colors."
            )
            return

        data = np.float32(
            player_colors
        )

        if (
            self.home_hsv is not None
            and self.away_hsv is not None
        ):
            home_ref = np.array(
                self._hsv_to_bgr(
                    self.home_hsv
                ),
                dtype=np.float32,
            )

            away_ref = np.array(
                self._hsv_to_bgr(
                    self.away_hsv
                ),
                dtype=np.float32,
            )

            self.team_colors_bgr = {
                1: home_ref,
                2: away_ref,
            }

            self.cluster_label_map = {
                1: "home",
                2: "away",
            }

        else:
            criteria = (
                cv2.TERM_CRITERIA_EPS
                + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                0.2,
            )

            _, _, centers = cv2.kmeans(
                data,
                2,
                None,
                criteria,
                10,
                cv2.KMEANS_PP_CENTERS,
            )

            self.team_colors_bgr = {
                1: centers[0],
                2: centers[1],
            }

            self.cluster_label_map = (
                self._label_clusters(
                    centers
                )
            )

        self.initialized = True

        logger.info(
            "V10 team colors: "
            "home=%s away=%s",
            self.home_hsv,
            self.away_hsv,
        )

    def _label_clusters(
        self,
        centers,
    ):
        if (
            self.home_hsv is None
            or self.away_hsv is None
        ):
            return {
                1: "team_1",
                2: "team_2",
            }

        center_hsvs = [
            _bgr_to_hsv_deg(
                center
            )
            for center in centers
        ]

        cost_a = (
            _hsv_distance(
                center_hsvs[0],
                self.home_hsv,
            )
            + _hsv_distance(
                center_hsvs[1],
                self.away_hsv,
            )
        )

        cost_b = (
            _hsv_distance(
                center_hsvs[0],
                self.away_hsv,
            )
            + _hsv_distance(
                center_hsvs[1],
                self.home_hsv,
            )
        )

        if cost_a <= cost_b:
            return {
                1: "home",
                2: "away",
            }

        return {
            1: "away",
            2: "home",
        }

    def _get_temporal_team(
        self,
        track_id,
    ):
        votes = (
            self.player_team_votes.get(
                track_id,
                {},
            )
        )

        home = float(
            votes.get(
                "home",
                0.0,
            )
        )

        away = float(
            votes.get(
                "away",
                0.0,
            )
        )

        total = home + away

        if total <= 0:
            return None

        if (
            home > away
            and (
                home - away
            ) / total
            >= TEMPORAL_HOME_MARGIN
        ):
            return "home"

        if (
            away > home
            and (
                away - home
            ) / total
            >= TEMPORAL_AWAY_MARGIN
        ):
            return "away"

        return None

    def _update_team_vote(
        self,
        track_id,
        label,
        weight=1.0,
    ):
        if label not in (
            "home",
            "away",
        ):
            return

        votes = (
            self.player_team_votes.setdefault(
                track_id,
                {
                    "home": 0.0,
                    "away": 0.0,
                },
            )
        )

        votes[label] += weight

    def get_player_team(
        self,
        frame,
        bbox,
        track_id,
    ):
        color = self._get_player_color(
            frame,
            bbox,
        )

        if color is None:
            temporal = (
                self._get_temporal_team(
                    track_id
                )
            )

            return (
                temporal
                if temporal is not None
                else "unclassified"
            )

        history = (
            self.player_color_history.setdefault(
                track_id,
                [],
            )
        )

        history.append(color)

        if (
            len(history)
            > MAX_TRACK_COLOR_HISTORY
        ):
            del history[
                :-MAX_TRACK_COLOR_HISTORY
            ]

        home_score, away_score = (
            self._specific_team_score(
                color
            )
        )

        if home_score > away_score:
            label = "home"
            margin = (
                home_score
                - away_score
            )
        else:
            label = "away"
            margin = (
                away_score
                - home_score
            )

        if margin < 0.08:
            temporal = (
                self._get_temporal_team(
                    track_id
                )
            )

            if temporal is not None:
                label = temporal
                margin = 0.08

        vote_weight = max(
            0.20,
            min(
                2.0,
                0.50
                + margin * 2.0,
            ),
        )

        self._update_team_vote(
            track_id,
            label,
            vote_weight,
        )

        temporal = (
            self._get_temporal_team(
                track_id
            )
        )

        if temporal is not None:
            label = temporal

        self.player_team_cache[
            track_id
        ] = label

        try:
            x1, y1, x2, y2 = [
                float(v)
                for v in bbox
            ]

            center_x = (
                x1 + x2
            ) / 2.0

            center_y = (
                y1 + y2
            ) / 2.0

            self.player_positions[
                track_id
            ] = (
                center_x,
                center_y,
            )

            self.team_position_history[
                label
            ].append(
                (
                    center_x,
                    center_y,
                )
            )

            if (
                len(
                    self.team_position_history[
                        label
                    ]
                )
                > 200
            ):
                self.team_position_history[
                    label
                ] = (
                    self.team_position_history[
                        label
                    ][-200:]
                )

        except Exception:
            pass

        return label

    def _get_team_spatial_centers(
        self,
    ):
        centers = {}

        for team in (
            "home",
            "away",
        ):
            positions = (
                self.team_position_history[
                    team
                ]
            )

            if not positions:
                continue

            data = np.asarray(
                positions,
                dtype=np.float32,
            )

            centers[team] = np.median(
                data,
                axis=0,
            )

        return centers

    def _goalkeeper_spatial_team(
        self,
        bbox,
    ):
        centers = (
            self._get_team_spatial_centers()
        )

        if len(centers) < 2:
            return None

        x1, y1, x2, y2 = [
            float(v)
            for v in bbox
        ]

        point = np.array(
            [
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
            ],
            dtype=np.float32,
        )

        distances = {
            team: float(
                np.linalg.norm(
                    point - center
                )
            )
            for team, center
            in centers.items()
        }

        ordered = sorted(
            distances.items(),
            key=lambda item: item[1],
        )

        if len(ordered) < 2:
            return ordered[0][0]

        first_team, first_distance = (
            ordered[0]
        )

        _, second_distance = (
            ordered[1]
        )

        if (
            second_distance <= 0
        ):
            return first_team

        ratio = (
            first_distance
            / second_distance
        )

        if ratio <= 0.88:
            return first_team

        return None

    def _goalkeeper_color_team(
        self,
        frame,
        bbox,
    ):
        color = self._get_player_color(
            frame,
            bbox,
        )

        if color is None:
            return None, 1.0

        hsv = _bgr_to_hsv_deg(
            color
        )

        candidates = {}

        if self.home_gk_hsv is not None:
            candidates[
                "home"
            ] = _hsv_distance(
                hsv,
                self.home_gk_hsv,
            )

        if self.away_gk_hsv is not None:
            candidates[
                "away"
            ] = _hsv_distance(
                hsv,
                self.away_gk_hsv,
            )

        if candidates:
            best_team = min(
                candidates,
                key=candidates.get,
            )

            return (
                best_team,
                candidates[
                    best_team
                ],
            )

        return None, 1.0

    def assign_goalkeeper_team(
        self,
        frame,
        bbox,
        track_id=None,
    ):
        color_team, color_distance = (
            self._goalkeeper_color_team(
                frame,
                bbox,
            )
        )

        spatial_team = (
            self._goalkeeper_spatial_team(
                bbox
            )
        )

        if (
            color_team is not None
            and color_distance
            <= GK_MATCH_THRESHOLD
        ):
            if spatial_team is None:
                result = color_team
            elif spatial_team == color_team:
                result = color_team
            else:
                result = color_team

        elif spatial_team is not None:
            result = spatial_team

        else:
            result = "unclassified"

        if track_id is not None:
            history = (
                self.goalkeeper_color_history.setdefault(
                    track_id,
                    [],
                )
            )

            if color_team is not None:
                history.append(
                    (
                        color_team,
                        color_distance,
                    )
                )

            if (
                len(history)
                > MAX_TRACK_COLOR_HISTORY
            ):
                del history[
                    :-MAX_TRACK_COLOR_HISTORY
                ]

            votes = {
                "home": 0.0,
                "away": 0.0,
            }

            for (
                observed_team,
                distance,
            ) in history:
                weight = max(
                    0.1,
                    1.0 - distance,
                )

                votes[
                    observed_team
                ] += weight

            if (
                votes["home"]
                > votes["away"]
                * 1.15
            ):
                result = "home"

            elif (
                votes["away"]
                > votes["home"]
                * 1.15
            ):
                result = "away"

            self.goalkeeper_team_cache[
                track_id
            ] = result

        return result