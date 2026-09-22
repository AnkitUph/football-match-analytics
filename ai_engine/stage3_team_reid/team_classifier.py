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

FAILED APPROACH 3 (raw BGR Euclidean distance, single-frame sample):
worked well on saturated kit colors, but on a match with a near-white
home kit (#e6edf3), a meaningful fraction of home-team raw tracklets got
misclassified as away. Root cause: near-white/low-saturation colors
swing much more in raw BGR magnitude under lighting/shadow than a
saturated color does, and raw Euclidean distance conflates that
brightness swing with actual hue difference — a shadowed white crop can
land numerically closer to a mid-brightness green reference than to its
own (much brighter) white reference. Compounded by single-frame
sampling per tracklet (see apps/matches/tasks.py's Stage 3 block): one
unlucky lighting frame could flip an entire raw tracklet's
classification on its own.

WORKING APPROACH (this file): tight chest-only median-color sampling
(from FAILED APPROACH 1/2's lessons) STILL applies — that part wasn't
the problem. Fixed the brightness sensitivity by normalizing colors to
unit length before comparing (classify_team_by_known_colors), so
distance reflects color RATIO (hue/chroma) rather than raw magnitude.
The complementary fix — sampling multiple frames per tracklet instead
of one — lives in apps/matches/tasks.py, not here.

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


def _normalize_color(color: np.ndarray) -> np.ndarray:
    """
    Strips overall brightness from a BGR color, leaving its relative
    channel ratios (a simple chromaticity normalization) — see this
    module's docstring (FAILED APPROACH 3) for why this matters: a
    near-white/pale kit's raw BGR magnitude swings more under lighting
    changes than a saturated color's does, and comparing raw magnitude
    conflates that swing with genuine hue difference. Comparing unit
    vectors instead compares color RATIO, which is far more stable
    across lighting/shadow for the SAME physical kit.
    """
    norm = np.linalg.norm(color)
    if norm < 1e-6:
        return color
    return color / norm


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
    Classifies by nearest distance to known per-match kit colors, in
    BRIGHTNESS-NORMALIZED space (see _normalize_color and this module's
    FAILED APPROACH 3 note) rather than raw BGR magnitude — comparing
    color ratio instead of raw distance is far more robust to the same
    kit appearing brighter/darker under different lighting or shadow,
    which matters most for low-saturation kits (white, pale colors).

    Team.TEAM_A is DEFINED as "closer to home_color_bgr" — this
    definition is relied on elsewhere (e.g. apps/matches/tasks.py maps
    TEAM_A -> match.home_team directly on this basis).
    """
    sample_n = _normalize_color(torso_color)
    home_n = _normalize_color(home_color_bgr)
    away_n = _normalize_color(away_color_bgr)
    dist_home = np.linalg.norm(sample_n - home_n)
    dist_away = np.linalg.norm(sample_n - away_n)
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

    colors: {track_id: BGR color for that track — ideally a per-track
    median of several sampled frames, not a single frame; see
    apps/matches/tasks.py's Stage 3 block}, PLAYER-CLASS ONLY — exclude
    referees before calling this (see apps/matches/tasks.py).

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


def classify_teams_with_dinov2(
    track_embeddings: dict[int, np.ndarray],
    track_colors: dict[int, np.ndarray],
    home_bgr: np.ndarray | None,
    away_bgr: np.ndarray | None,
    config: TeamReidConfig,
) -> dict[int, Team]:
    """
    Advanced Team Classification using Hugging Face DINOv2 visual embeddings.

    Clusters 384-dimensional semantic appearance embeddings using K-Means (k=2).
    Because DINOv2 encodes the player's full kit (jersey color, patterns, sleeves,
    shorts) while being invariant to scale and lighting, it cleanly separates the
    two opposing teams without being misled by shadows or tiny 9x11 torso noise.

    Maps the two clusters to Team A (Home) and Team B (Away) by comparing the
    cluster-median colors against the match reference kit colors.
    """
    track_ids = [tid for tid in track_embeddings.keys() if track_embeddings[tid] is not None]

    # If fewer than 4 tracks have embeddings, fall back to color clustering
    if len(track_ids) < 4:
        return classify_teams_with_fallback(track_colors, home_bgr, away_bgr, config)

    X = np.array([track_embeddings[tid] for tid in track_ids], dtype=np.float32)
    # L2 normalize just in case
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    try:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
        labels = kmeans.labels_
    except Exception:
        return classify_teams_with_fallback(track_colors, home_bgr, away_bgr, config)

    # Separate track IDs into Cluster 0 and Cluster 1
    c0_tids = [tid for tid, l in zip(track_ids, labels) if l == 0]
    c1_tids = [tid for tid, l in zip(track_ids, labels) if l == 1]

    # If one cluster has almost everything (>85%), fall back to color clustering
    c0_ratio = len(c0_tids) / max(len(track_ids), 1)
    if c0_ratio < 0.15 or c0_ratio > 0.85:
        return classify_teams_with_fallback(track_colors, home_bgr, away_bgr, config)

    # Compute average kit colors for both clusters to determine Home vs Away mapping
    def get_cluster_color(tids):
        cols = [track_colors[t] for t in tids if t in track_colors and track_colors[t] is not None]
        return np.median(np.array(cols), axis=0) if cols else np.array([128.0, 128.0, 128.0])

    col_0 = get_cluster_color(c0_tids)
    col_1 = get_cluster_color(c1_tids)

    # Decide whether Cluster 0 is Team A or Team B
    c0_is_team_a = True
    if home_bgr is not None and away_bgr is not None:
        home_n = _normalize_color(home_bgr)
        away_n = _normalize_color(away_bgr)
        c0_n = _normalize_color(col_0)
        c1_n = _normalize_color(col_1)

        # Hypothesis 1: Cluster 0 is Home, Cluster 1 is Away
        cost_direct = np.linalg.norm(c0_n - home_n) + np.linalg.norm(c1_n - away_n)
        # Hypothesis 2: Cluster 1 is Home, Cluster 0 is Away
        cost_swap = np.linalg.norm(c1_n - home_n) + np.linalg.norm(c0_n - away_n)

        c0_is_team_a = (cost_direct <= cost_swap)

    result = {}
    for tid, l in zip(track_ids, labels):
        if l == 0:
            result[tid] = Team.TEAM_A if c0_is_team_a else Team.TEAM_B
        else:
            result[tid] = Team.TEAM_B if c0_is_team_a else Team.TEAM_A

    # For any tracks that didn't have embeddings, fill in from track_colors
    for tid in track_colors:
        if tid not in result:
            result[tid] = Team.UNKNOWN

    return result