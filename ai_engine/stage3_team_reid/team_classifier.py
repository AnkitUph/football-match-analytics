"""
Stage 3a: Team Classification via jersey color.

Fast coarse filter — build and validate this FIRST, before adding Re-ID
embeddings (team_reid.py) or OCR (jersey_ocr.py). It's enough on its own to
get Stage 5 working on a single continuous camera angle.
"""

import numpy as np

from ai_engine.config import TeamReidConfig
from ai_engine.utils.types import Detection, Team


def mask_out_grass(crop: np.ndarray) -> np.ndarray:
    """
    HSV-threshold a player bounding-box crop to remove green pitch pixels,
    isolating jersey/skin/shorts pixels for color sampling.

    TODO:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)
        non_grass_mask = cv2.bitwise_not(grass_mask)
        return cv2.bitwise_and(crop, crop, mask=non_grass_mask)
    """
    raise NotImplementedError("mask_out_grass is stubbed.")


def sample_torso_color(crop: np.ndarray) -> np.ndarray:
    """
    Crop to the torso region (roughly the middle third vertically, avoiding
    legs/shorts and head) and return the dominant color as an (R,G,B) or
    HSV vector via K-means (k=1 or k=2 to also catch a secondary trim
    color).

    TODO:
        h, w = crop.shape[:2]
        torso = crop[int(h*0.25):int(h*0.6), int(w*0.2):int(w*0.8)]
        pixels = torso.reshape(-1, 3).astype(np.float32)
        # drop near-black pixels left by grass masking
        pixels = pixels[pixels.sum(axis=1) > 30]
        if len(pixels) == 0:
            return np.array([0, 0, 0])
        kmeans = KMeans(n_clusters=1, n_init=3).fit(pixels)
        return kmeans.cluster_centers_[0]
    """
    raise NotImplementedError("sample_torso_color is stubbed.")


def fit_team_color_clusters(
    torso_colors: list[np.ndarray], config: TeamReidConfig
) -> "sklearn.cluster.KMeans":  # noqa: F821
    """
    Run ONCE per shot (or per match, if lighting is consistent) across all
    detected torso colors to find the two team-color centroids (+ referee,
    which usually stands out as black/yellow/neon and can be filtered by a
    simple brightness/saturation rule before clustering).

    TODO:
        from sklearn.cluster import KMeans
        X = np.array(torso_colors)
        return KMeans(n_clusters=config.kmeans_clusters, n_init=10).fit(X)
    """
    raise NotImplementedError("fit_team_color_clusters is stubbed.")


def classify_team(
    torso_color: np.ndarray, team_clusters, config: TeamReidConfig
) -> Team:
    """
    Assign TEAM_A / TEAM_B / REFEREE / UNKNOWN based on nearest cluster
    centroid. Referee detection at the Detector level (Stage 1's separate
    `referee` class) should usually be trusted over color clustering here —
    only fall back to color-based referee guessing if Stage 1 missed it.
    """
    raise NotImplementedError("classify_team is stubbed.")
