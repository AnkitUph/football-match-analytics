"""
Assigns each tracked PLAYER to a team, using K-means clustering to
discover the two actual jersey colors present in the footage, then
labels the two discovered clusters as "home"/"away" by comparing them
to the match's stored kit colors.

This only ever operates on the "players" category from Tracker - since
your custom model already separates goalkeepers and referees as their
own classes, there's no contamination risk here anymore (the biggest
recurring problem in earlier attempts at this pipeline was exactly this
kind of contamination, fought after the fact with color heuristics on a
generic person detector - that problem doesn't exist with this model).

Goalkeepers are handled separately (see assign_goalkeeper_team) via
direct distance matching against their own stored kit colors, since
there are only ever 2 GK tracks in a match - far too few to cluster
reliably, and they don't need to be, since the model already tells us
they're goalkeepers.
"""

import colorsys
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

HUE_WEIGHT = 0.7
SATURATION_WEIGHT = 0.2
VALUE_WEIGHT = 0.1

# If a goalkeeper's color isn't at least this close to a stored GK
# color, it's left unassigned rather than force-matched - most likely
# means the stored kit color is wrong/missing, not that the model
# mis-detected them (they're already confirmed goalkeeper-class).
GK_MATCH_THRESHOLD = 0.30


def hex_to_hsv(hex_color):
    if not hex_color:
        return None
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (h * 360, s, v)


def _bgr_to_hsv_deg(bgr):
    b, g, r = bgr
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return (h * 360, s, v)


def _hsv_distance(hsv1, hsv2):
    """
    Weighted HSV distance, hue down-weighted for low-saturation colors
    (hue is unreliable/near-meaningless for near-black/gray/white
    colors). Kept consistent with the distance metric used elsewhere in
    this project.
    """
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2

    dh = min(abs(h1 - h2), 360 - abs(h1 - h2)) / 180
    ds = abs(s1 - s2)
    dv = abs(v1 - v2)

    hue_reliability = min(s1, s2)
    effective_hue_weight = HUE_WEIGHT * hue_reliability
    remaining = 1 - effective_hue_weight
    other_total = SATURATION_WEIGHT + VALUE_WEIGHT
    eff_sat = remaining * (SATURATION_WEIGHT / other_total)
    eff_val = remaining * (VALUE_WEIGHT / other_total)

    return dh * effective_hue_weight + ds * eff_sat + dv * eff_val


class TeamAssigner:
    def __init__(self, home_color_hex=None, away_color_hex=None,
                 home_gk_color_hex=None, away_gk_color_hex=None):
        self.home_hsv = hex_to_hsv(home_color_hex)
        self.away_hsv = hex_to_hsv(away_color_hex)
        self.home_gk_hsv = hex_to_hsv(home_gk_color_hex)
        self.away_gk_hsv = hex_to_hsv(away_gk_color_hex)

        self.team_colors_bgr = {}       # 1 -> bgr, 2 -> bgr (raw cluster centers)
        self.cluster_label_map = {}     # 1 -> "home"/"away", 2 -> "home"/"away"
        self.player_team_cache = {}     # track_id -> label, so repeated calls are cheap

    def _get_player_color(self, frame, bbox):
        """
        Crops the top half of a player's box (torso/head region, most
        jersey-colored, avoiding shorts/legs which are often a different
        color) and separates jersey from background within that crop via
        a quick 2-cluster K-means, using the crop's corner pixels (which
        are almost always background, not the player) to figure out
        which of the 2 clusters is the jersey.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        image = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if image.size == 0:
            return None

        top_half = image[0:max(1, image.shape[0] // 2), :]
        pixels = top_half.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 4:
            return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(top_half.shape[0], top_half.shape[1])

        corner_labels = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        background_cluster = max(set(corner_labels), key=corner_labels.count)
        jersey_cluster = 1 - background_cluster

        return centers[jersey_cluster]  # BGR

    def assign_team_colors(self, frame, player_tracks_in_frame):
        """
        Call once, on a frame with good visibility of both teams (the
        first frame with players detected is usually fine). Discovers
        the two team colors via K-means over all players' jersey colors,
        then labels the two clusters as home/away using the stored kit
        colors as reference.
        """
        player_colors = []
        for track_id, info in player_tracks_in_frame.items():
            color = self._get_player_color(frame, info["bbox"])
            if color is not None:
                player_colors.append(color)

        if len(player_colors) < 2:
            logger.warning("Not enough players with valid colors to cluster teams.")
            return

        data = np.float32(player_colors)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, _, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        self.team_colors_bgr = {1: centers[0], 2: centers[1]}
        self.cluster_label_map = self._label_clusters(centers)
        logger.info("Discovered team colors: cluster 1=%s (%s), cluster 2=%s (%s)",
                    centers[0], self.cluster_label_map.get(1),
                    centers[1], self.cluster_label_map.get(2))

    def _label_clusters(self, centers):
        """
        Labels the 2 cluster centers as home/away by finding whichever
        pairing has the lower total distance to the stored colors -
        guarantees the two clusters get DIFFERENT labels, rather than
        both greedily matching whichever is nominally closer.
        """
        if self.home_hsv is None or self.away_hsv is None:
            return {1: "team_1", 2: "team_2"}

        center_hsvs = [_bgr_to_hsv_deg(c) for c in centers]

        cost_a = _hsv_distance(center_hsvs[0], self.home_hsv) + _hsv_distance(center_hsvs[1], self.away_hsv)
        cost_b = _hsv_distance(center_hsvs[0], self.away_hsv) + _hsv_distance(center_hsvs[1], self.home_hsv)

        if cost_a <= cost_b:
            return {1: "home", 2: "away"}
        return {1: "away", 2: "home"}

    def get_player_team(self, frame, bbox, track_id):
        if track_id in self.player_team_cache:
            return self.player_team_cache[track_id]

        if not self.team_colors_bgr:
            return "unclassified"

        color = self._get_player_color(frame, bbox)
        if color is None:
            return "unclassified"

        color_hsv = _bgr_to_hsv_deg(color)
        distances = {
            cluster_id: _hsv_distance(color_hsv, _bgr_to_hsv_deg(center))
            for cluster_id, center in self.team_colors_bgr.items()
        }
        nearest_cluster = min(distances, key=distances.get)
        label = self.cluster_label_map.get(nearest_cluster, f"team_{nearest_cluster}")

        self.player_team_cache[track_id] = label
        return label

    def assign_goalkeeper_team(self, frame, bbox):
        """
        Direct match against stored GK colors - no clustering, since
        there are only ever 2 GK tracks per match. Returns "home_gk",
        "away_gk", or "unclassified" if neither stored color is close
        enough (most likely means the stored GK color wasn't set).
        """
        if self.home_gk_hsv is None and self.away_gk_hsv is None:
            return "unclassified"

        color = self._get_player_color(frame, bbox)
        if color is None:
            return "unclassified"

        color_hsv = _bgr_to_hsv_deg(color)

        candidates = {}
        if self.home_gk_hsv is not None:
            candidates["home_gk"] = _hsv_distance(color_hsv, self.home_gk_hsv)
        if self.away_gk_hsv is not None:
            candidates["away_gk"] = _hsv_distance(color_hsv, self.away_gk_hsv)

        best_label = min(candidates, key=candidates.get)
        if candidates[best_label] > GK_MATCH_THRESHOLD:
            return "unclassified"
        return best_label