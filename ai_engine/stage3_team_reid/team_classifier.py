"""
Stage 3a: Team Classification via jersey color.

Validated against real footage (test_11.avi) — see notes below on what
did NOT work and why, before landing on this approach.

FAILED APPROACH 1 (HSV grass-masking): this match's away kit is neon
green, whose HSV hue (~52) is nearly identical to the pitch grass (~50).
Any grass-removal mask strips out most of the jersey pixels too, since
they're indistinguishable by hue.

FAILED APPROACH 2 (no masking, wide torso ROI): background grass at the
crop edges pulled every color average toward green, contaminating even
white-kit players' samples.

WORKING APPROACH (this file): skip grass masking entirely. Use a tight
chest-only ROI (avoids arms/edges where background leaks in) and take
the MEDIAN pixel color, not the mean.

Known limitation: referees (dark kit) tend to get absorbed into
whichever team's color cluster is darker — Stage 1's detector already
identifies referees as their own class, so referee crops are excluded
from team classification entirely at the call site (see
apps/matches/tasks.py), not handled via color here.
"""

import numpy as np
from sklearn.cluster import KMeans

from ai_engine.config import TeamReidConfig
from ai_engine.utils.types import Team

# A known-color split this one-sided is treated as evidence the
# provided reference colors don't match the actual footage (e.g. test
# data pairing made-up teams with unrelated real video) — found via
# real testing: colors #3d8b5f/#274690 (neither matching the true
# white/neon-green kits in test_11.mp4) sent 100% of players to one
# team. Falls back to blind clustering in that case.
FALLBACK_IMBALANCE_THRESHOLD = 0.85


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

    torso = crop[int(h * 0.20):int(h * 0.42), int(w * 0.28):int(w * 0.72)]
    if torso.size == 0:
        return None

    pixels = torso.reshape(-1, 3).astype(np.float32)
    return np.median(pixels, axis=0)


def fit_team_color_clusters(torso_colors: list[np.ndarray], config: TeamReidConfig) -> KMeans:
    """
    Blind clustering fallback — run across all sampled PLAYER (non-
    referee) torso colors to find the two team-color centroids.
    """
    X = np.array(torso_colors)
    return KMeans(n_clusters=config.kmeans_clusters, n_init=10, random_state=0).fit(X)


def classify_team(
    detection_cls: str,
    torso_color: np.ndarray | None,
    team_clusters: KMeans | None,
) -> Team:
    """
    Assigns a Team using blind clusters. Referees are identified
    directly from Stage 1's class label at the call site, not routed
    through this function in practice — the check here is defensive.
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
    Classifies by nearest distance to known per-match kit colors.
    Team.TEAM_A is DEFINED as "closer to home_color_bgr" — this
    definition is relied on elsewhere (e.g. apps/matches/tasks.py maps
    TEAM_A -> match.home_team directly on this basis).
    """
    dist_home = np.linalg.norm(torso_color - home_color_bgr)
    dist_away = np.linalg.norm(torso_color - away_color_bgr)
    return Team.TEAM_A if dist_home < dist_away else Team.TEAM_B


def classify_teams_with_fallback(
    colors: dict[int, np.ndarray],
    home_bgr: np.ndarray | None,
    away_bgr: np.ndarray | None,
    config: TeamReidConfig,
) -> dict[int, Team]:
    """
    Primary entry point for Stage 3 team classification, used by
    apps/matches/tasks.py.

    Tries known-color classification first (when both home_bgr and
    away_bgr are provided — i.e. this match's real kit colors were
    captured at upload). Falls back to blind K-means clustering if:
      (a) known colors weren't provided at all, or
      (b) known-color classification produces an implausibly one-sided
          split (>FALLBACK_IMBALANCE_THRESHOLD fraction on one side) —
          a strong signal the provided colors don't actually match the
          footage, found via real testing (see module docstring).

    colors: {track_id: BGR median torso color}, PLAYER-CLASS ONLY —
    exclude referees before calling this (see apps/matches/tasks.py).

    Returns {track_id: Team} for every key in `colors`.
    """
    track_ids = list(colors.keys())
    color_list = [colors[tid] for tid in track_ids]

    def blind_fallback() -> dict[int, Team]:
        if len(color_list) < 2:
            return {tid: Team.UNKNOWN for tid in track_ids}
        clusters = fit_team_color_clusters(color_list, config)
        return {
            tid: classify_team("player", colors[tid], clusters)
            for tid in track_ids
        }

    if home_bgr is None or away_bgr is None:
        return blind_fallback()

    known_color_result = {
        tid: classify_team_by_known_colors(colors[tid], home_bgr, away_bgr)
        for tid in track_ids
    }

    total = len(track_ids)
    if total == 0:
        return known_color_result

    team_a_count = sum(1 for t in known_color_result.values() if t == Team.TEAM_A)
    team_a_fraction = team_a_count / total

    if team_a_fraction >= FALLBACK_IMBALANCE_THRESHOLD or team_a_fraction <= (1 - FALLBACK_IMBALANCE_THRESHOLD):
        # One-sided split — known colors likely don't match this
        # footage. Fall back to blind clustering instead of trusting a
        # near-100%-one-team result.
        return blind_fallback()

    return known_color_result
