"""
Stage 3a: Team Classification via jersey color.

Validated against real footage (test_11.avi) — see notes below on what
did NOT work and why, before landing on this approach.

FAILED APPROACH 1 (HSV grass-masking): this match's away kit is neon
green, whose HSV hue (~52) is nearly identical to the pitch grass (~50).
Any grass-removal mask strips out most of the jersey pixels too, since
they're indistinguishable by hue. Do not "fix" this by tuning the mask
range tighter — there is no hue threshold that separates them, since
they're the same color.

FAILED APPROACH 2 (no masking, wide torso ROI): background grass at the
crop edges pulled every color average toward green, contaminating even
white-kit players' samples.

WORKING APPROACH (this file): skip grass masking entirely. Use a tight
chest-only ROI (avoids arms/edges where background leaks in) and take the
MEDIAN pixel color, not the mean or a 1-cluster K-means centroid — median
is far more robust to the residual few background/shadow pixels that
still creep into even a tight crop.

Known limitation: referees (dark kit) tend to get absorbed into whichever
team's color cluster is darker, since 2-means clustering has no dedicated
"dark/referee" cluster. This is harmless in practice — Stage 1's detector
already identifies referees as their own class, so referee crops should
be excluded from team classification entirely rather than relying on
color to separate them. See classify_team() below.
"""

import numpy as np
from sklearn.cluster import KMeans

from ai_engine.config import TeamReidConfig
from ai_engine.utils.types import Team


def sample_torso_color(crop: np.ndarray) -> np.ndarray | None:
    """
    Samples a tight chest-region crop and returns its median BGR color.
    Returns None if the crop is too small to sample meaningfully.
    """
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 10 or w < 5:
        return None

    # Upper-chest core only — deliberately tight to avoid arms, legs, and
    # crop edges where background pitch color leaks in.
    torso = crop[int(h * 0.20):int(h * 0.42), int(w * 0.28):int(w * 0.72)]
    if torso.size == 0:
        return None

    pixels = torso.reshape(-1, 3).astype(np.float32)
    return np.median(pixels, axis=0)


def fit_team_color_clusters(torso_colors: list[np.ndarray], config: TeamReidConfig) -> KMeans:
    """
    Run ONCE per match (or per shot, if you want to be defensive about
    lighting changes) across all sampled player torso colors, to find the
    two team-color centroids.

    IMPORTANT: only pass PLAYER-class torso colors here, not referees —
    referees should be identified via Stage 1's detection class directly
    (see classify_team below), not folded into this clustering. Including
    referee crops here is what causes them to get absorbed into whichever
    team cluster is darker.
    """
    X = np.array(torso_colors)
    return KMeans(n_clusters=config.kmeans_clusters, n_init=10, random_state=0).fit(X)


def classify_team(
    detection_cls: str,
    torso_color: np.ndarray | None,
    team_clusters: KMeans | None,
) -> Team:
    """
    Assigns a Team given a detection. Referees are identified directly
    from Stage 1's class label — never routed through color clustering,
    since their dark kit doesn't reliably separate from a dark team
    cluster. Only 'player' and 'goalkeeper' class detections use color.
    """
    if detection_cls == "referee":
        return Team.REFEREE

    if torso_color is None or team_clusters is None:
        return Team.UNKNOWN

    cluster_id = int(team_clusters.predict(torso_color.reshape(1, -1))[0])
    return Team.TEAM_A if cluster_id == 0 else Team.TEAM_B


def classify_team_by_known_colors(
    torso_color: np.ndarray, home_color_bgr: np.ndarray, away_color_bgr: np.ndarray
) -> Team:
    """
    Alternative to blind clustering: classify by nearest distance to your
    project's own known per-match kit colors (home_color/away_color hex,
    already captured at upload time per the Match model). More robust
    than blind K-means when you have this data, since you're matching
    against a known reference instead of hoping two clusters separate
    cleanly. Convert hex (e.g. "#39FF14") to BGR before calling this,
    e.g.: bytes.fromhex(hex_str.lstrip('#'))[::-1]
    """
    dist_home = np.linalg.norm(torso_color - home_color_bgr)
    dist_away = np.linalg.norm(torso_color - away_color_bgr)
    return Team.TEAM_A if dist_home < dist_away else Team.TEAM_B
