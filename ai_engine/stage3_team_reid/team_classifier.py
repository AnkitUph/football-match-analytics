"""
Stage 3a: Universal Team Classification via Perceptual Color & Deep Appearance Fusion.

Addresses real-world broadcast conditions across any match fixture, lighting condition,
and kit color combination.

Core Principles:
1. CIE-L*a*b* Perceptual Uniformity with Lightness Attenuation:
   Stadium shadows, floodlight glare, and sun angles swing perceptual lightness (L*)
   dramatically (±40), but the physical kit dye (a* = green-red, b* = blue-yellow)
   remains constant. Distance calculation down-weights L* (factor 0.25) so lighting
   and shadows cannot flip kit classification.

2. Pitch-Turf Decoupling:
   Bounding boxes for running players inevitably catch green pitch turf at the borders.
   Grass pixels are dynamically filtered out in Lab space before computing the dominant
   torso kit signature.

3. Deep Semantic Appearance Fusion (DINOv2 + Perceptual Chromaticity):
   Combines 384-d DINOv2 appearance features (which encode stripes, textures, and kit patterns)
   with perceptual color vectors, constrained by equal-team cardinality priors.

4. Domain-Invariant Sanity Guardrail & Self-Healing:
   In any 11v11 match, both teams have ~10 outfield players on the pitch simultaneously.
   Any proposed classification with an impossible imbalance (> 75% on one side) is
   automatically rejected and resolved using balanced margin ranking.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

from ai_engine.config import TeamReidConfig
from ai_engine.utils.types import Team

FALLBACK_IMBALANCE_THRESHOLD = 0.75


def bgr_to_standard_lab(bgr: np.ndarray) -> np.ndarray:
    """
    Converts a BGR color (float or uint8) to standard CIE-L*a*b* coordinates:
    L* in [0, 100], a* in [-128, 127], b* in [-128, 127].
    """
    if bgr is None:
        return np.array([50.0, 0.0, 0.0], dtype=np.float32)
    bgr_u8 = np.clip(bgr, 0, 255).astype(np.uint8)
    img = np.reshape(bgr_u8, (1, 1, 3))
    lab_cv = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[0, 0].astype(np.float32)
    # OpenCV scaling: L = L* * 255/100, a = a* + 128, b = b* + 128
    L_std = lab_cv[0] * 100.0 / 255.0
    a_std = lab_cv[1] - 128.0
    b_std = lab_cv[2] - 128.0
    return np.array([L_std, a_std, b_std], dtype=np.float32)


def perceptual_kit_distance(bgr1: np.ndarray, bgr2: np.ndarray, l_weight: float = 0.25) -> float:
    """
    Calculates shadow-attenuated perceptual color distance in CIE-L*a*b* space,
    fused with normalized chromaticity vector distance.
    L* is down-weighted to be robust against pitch shadows and floodlight exposure swings,
    while chromatic axes and relative channel ratios strictly preserve kit distinctions
    (e.g. yellow vs red vs blue vs white) even under broadcast camera desaturation.
    """
    if bgr1 is None or bgr2 is None:
        return 999.0
    lab1 = bgr_to_standard_lab(bgr1)
    lab2 = bgr_to_standard_lab(bgr2)
    dL = (lab1[0] - lab2[0]) * l_weight
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    lab_dist = float(np.sqrt(dL * dL + da * da + db * db))

    # Normalized chromatic distance (channel ratio invariance)
    n1 = bgr1.astype(np.float32) / max(float(np.linalg.norm(bgr1)), 1e-6)
    n2 = bgr2.astype(np.float32) / max(float(np.linalg.norm(bgr2)), 1e-6)
    chrom_dist = float(np.linalg.norm(n1 - n2))

    return lab_dist + 35.0 * chrom_dist


def sample_torso_color(crop: np.ndarray) -> np.ndarray | None:
    """
    Samples a tight chest-region crop, dynamically decouples pitch grass pixels using
    HSV hue isolation, and returns the median BGR color of the jersey fabric.
    Returns None if the crop is too small or empty.
    """
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 15 or w < 8:
        return None

    # Sample chest region: 20%-45% vertical, 25%-75% horizontal
    torso = crop[int(h * 0.20):int(h * 0.45), int(w * 0.25):int(w * 0.75)]
    if torso.size == 0:
        return None

    try:
        # Decouple green pitch turf using HSV hue
        # Pitch grass has hue in [35, 85] with saturation >= 35
        hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        grass = (hsv_torso[:, :, 0] >= 35) & (hsv_torso[:, :, 0] <= 85) & (hsv_torso[:, :, 1] >= 35)
        # If not almost the entire torso is grass (which would happen for a green kit)
        if np.mean(grass) < 0.75:
            non_grass = torso[~grass]
            if len(non_grass) >= 12:
                return np.median(non_grass.astype(np.float32), axis=0)
    except Exception:
        pass

    pixels = torso.reshape(-1, 3).astype(np.float32)
    return np.median(pixels, axis=0)


def _normalize_color(color: np.ndarray) -> np.ndarray:
    """Retained for backwards compatibility where unit vector normalization is expected."""
    norm = np.linalg.norm(color)
    if norm < 1e-6:
        return color
    return color / norm


def fit_team_color_clusters(torso_colors: list[np.ndarray], config: TeamReidConfig) -> KMeans:
    """
    Fits 2 clusters in shadow-attenuated CIE-L*a*b* space across sampled torso colors.
    """
    X_lab = np.array([
        [l[0] * 0.25, l[1], l[2]]
        for l in (bgr_to_standard_lab(c) for c in torso_colors)
    ], dtype=np.float32)
    return KMeans(n_clusters=config.kmeans_clusters, n_init=10, random_state=0).fit(X_lab)


def classify_team(
    detection_cls: str,
    torso_color: np.ndarray | None,
    team_clusters: KMeans | None,
) -> Team:
    """
    Assigns a Team using blind clusters. Referees are identified
    directly from Stage 1's class label at the call site.
    """
    if detection_cls == "referee":
        return Team.REFEREE

    if torso_color is None or team_clusters is None:
        return Team.UNKNOWN

    lab = bgr_to_standard_lab(torso_color)
    feat = np.array([[lab[0] * 0.25, lab[1], lab[2]]], dtype=np.float32)
    cluster_id = int(team_clusters.predict(feat)[0])
    return Team.TEAM_A if cluster_id == 0 else Team.TEAM_B


def classify_team_by_known_colors(
    torso_color: np.ndarray,
    home_color_bgr: np.ndarray,
    away_color_bgr: np.ndarray,
    home_gk_bgr: np.ndarray | None = None,
    away_gk_bgr: np.ndarray | None = None,
    is_gk: bool = False,
) -> Team:
    """
    Classifies a torso sample by nearest distance to known per-match kit colors.
    If is_gk is True and GK kit colors are provided, matches against the goalkeeper kits.
    """
    if is_gk and (home_gk_bgr is not None or away_gk_bgr is not None):
        h_col = home_gk_bgr if home_gk_bgr is not None else home_color_bgr
        a_col = away_gk_bgr if away_gk_bgr is not None else away_color_bgr
        dist_home = perceptual_kit_distance(torso_color, h_col)
        dist_away = perceptual_kit_distance(torso_color, a_col)
        return Team.TEAM_A if dist_home < dist_away else Team.TEAM_B

    dist_home = perceptual_kit_distance(torso_color, home_color_bgr)
    dist_away = perceptual_kit_distance(torso_color, away_color_bgr)
    return Team.TEAM_A if dist_home < dist_away else Team.TEAM_B


def _balanced_margin_partition(
    track_ids: list[int],
    colors: dict[int, np.ndarray],
    home_bgr: np.ndarray,
    away_bgr: np.ndarray,
    home_gk_bgr: np.ndarray | None = None,
    away_gk_bgr: np.ndarray | None = None,
    cls_by_track: dict[int, any] | None = None,
) -> dict[int, Team]:
    """
    Domain-invariant balanced partitioner:
    Scores each outfield track by (dist_away - dist_home).
    Positive margin means closer to Home; negative means closer to Away.
    Guarantees both teams maintain realistic outfield participation (between 30% and 70%),
    preventing any degenerate 'one team swallowed' collapse.
    Goalkeepers are classified directly by GK kit proximity.
    """
    scored = []
    gk_results = {}
    for tid in track_ids:
        col = colors.get(tid)
        if col is None:
            col = np.array([128.0, 128.0, 128.0], dtype=np.float32)

        is_gk = False
        if cls_by_track:
            c = cls_by_track.get(tid)
            is_gk = (c == "goalkeeper" or getattr(c, "value", "") == "goalkeeper")

        if is_gk and (home_gk_bgr is not None or away_gk_bgr is not None):
            gk_results[tid] = classify_team_by_known_colors(
                col, home_bgr, away_bgr, home_gk_bgr, away_gk_bgr, is_gk=True
            )
            continue

        dh = perceptual_kit_distance(col, home_bgr)
        da = perceptual_kit_distance(col, away_bgr)
        margin = da - dh  # higher = more strongly Home
        scored.append((tid, margin))

    # Sort from most Home to most Away
    scored.sort(key=lambda x: -x[1])
    n = len(scored)
    result = dict(gk_results)
    if n == 0:
        return result

    # Natural zero-crossing index
    zero_idx = next((i for i, item in enumerate(scored) if item[1] < 0), n // 2)

    # Constrain split within [30%, 70%] bounds
    min_split = max(1, int(0.30 * n))
    max_split = min(n - 1, int(0.70 * n))
    if min_split <= max_split:
        cut_idx = max(min_split, min(zero_idx, max_split))
    else:
        cut_idx = n // 2

    for i, (tid, _) in enumerate(scored):
        result[tid] = Team.TEAM_A if i < cut_idx else Team.TEAM_B
    return result


def classify_teams_with_fallback(
    colors: dict[int, np.ndarray],
    home_bgr: np.ndarray | None,
    away_bgr: np.ndarray | None,
    config: TeamReidConfig,
    home_gk_bgr: np.ndarray | None = None,
    away_gk_bgr: np.ndarray | None = None,
    cls_by_track: dict[int, any] | None = None,
) -> dict[int, Team]:
    """
    Primary entry point for Stage 3 team classification via perceptual color.
    Uses shadow-attenuated CIE-L*a*b* distance and normalized chromaticity with
    automatic balanced fallback and goalkeeper kit awareness.
    """
    track_ids = list(colors.keys())
    if not track_ids:
        return {}

    def blind_fallback() -> dict[int, Team]:
        color_list = [colors[tid] for tid in track_ids]
        if len(color_list) < 2:
            return {tid: Team.UNKNOWN for tid in track_ids}
        clusters = fit_team_color_clusters(color_list, config)
        return {
            tid: classify_team("player", colors[tid], clusters)
            for tid in track_ids
        }

    if home_bgr is None or away_bgr is None:
        return blind_fallback()

    # Step 1: Direct perceptual classification
    known_color_result = {}
    for tid in track_ids:
        is_gk = False
        if cls_by_track:
            c = cls_by_track.get(tid)
            is_gk = (c == "goalkeeper" or getattr(c, "value", "") == "goalkeeper")
        known_color_result[tid] = classify_team_by_known_colors(
            colors[tid], home_bgr, away_bgr, home_gk_bgr, away_gk_bgr, is_gk=is_gk
        )

    outfield_ids = [
        tid for tid in track_ids
        if not (cls_by_track and (cls_by_track.get(tid) == "goalkeeper" or getattr(cls_by_track.get(tid), "value", "") == "goalkeeper"))
    ]
    total_outfield = max(len(outfield_ids), 1)
    team_a_count = sum(1 for tid in outfield_ids if known_color_result[tid] == Team.TEAM_A)
    team_a_fraction = team_a_count / total_outfield

    # Step 2: Sanity Guardrail & Self-Healing
    # If outfield distribution collapses (> 75% or < 25%), apply balanced margin partitioning
    if team_a_fraction >= FALLBACK_IMBALANCE_THRESHOLD or team_a_fraction <= (1.0 - FALLBACK_IMBALANCE_THRESHOLD):
        return _balanced_margin_partition(
            track_ids, colors, home_bgr, away_bgr, home_gk_bgr, away_gk_bgr, cls_by_track
        )

    return known_color_result


def classify_teams_with_dinov2(
    track_embeddings: dict[int, np.ndarray],
    track_colors: dict[int, np.ndarray],
    home_bgr: np.ndarray | None,
    away_bgr: np.ndarray | None,
    config: TeamReidConfig,
    home_gk_bgr: np.ndarray | None = None,
    away_gk_bgr: np.ndarray | None = None,
    cls_by_track: dict[int, any] | None = None,
) -> dict[int, Team]:
    """
    Advanced Team Classification using fused DINOv2 embeddings and perceptual color.
    Clusters semantic visual embeddings, checks cluster validity against perceptual kit colors,
    and enforces balanced cardinality guardrails with goalkeeper kit awareness.
    """
    track_ids = [tid for tid in track_embeddings.keys() if track_embeddings[tid] is not None]

    if len(track_ids) < 4:
        return classify_teams_with_fallback(
            track_colors, home_bgr, away_bgr, config, home_gk_bgr, away_gk_bgr, cls_by_track
        )

    # Normalized DINOv2 embeddings
    X_emb = np.array([track_embeddings[tid] for tid in track_ids], dtype=np.float32)
    norms = np.linalg.norm(X_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_emb = X_emb / norms

    try:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_emb)
        labels = kmeans.labels_
    except Exception:
        return classify_teams_with_fallback(
            track_colors, home_bgr, away_bgr, config, home_gk_bgr, away_gk_bgr, cls_by_track
        )

    c0_tids = [tid for tid, l in zip(track_ids, labels) if l == 0]
    c1_tids = [tid for tid, l in zip(track_ids, labels) if l == 1]

    c0_ratio = len(c0_tids) / max(len(track_ids), 1)
    # If DINOv2 cluster split is degenerate (< 25% or > 75%), fall back to robust color classification
    if c0_ratio < 0.25 or c0_ratio > 0.75:
        return classify_teams_with_fallback(
            track_colors, home_bgr, away_bgr, config, home_gk_bgr, away_gk_bgr, cls_by_track
        )

    def get_cluster_color(tids):
        cols = [track_colors[t] for t in tids if t in track_colors and track_colors[t] is not None]
        return np.median(np.array(cols), axis=0) if cols else np.array([128.0, 128.0, 128.0])

    col_0 = get_cluster_color(c0_tids)
    col_1 = get_cluster_color(c1_tids)

    c0_is_team_a = True
    if home_bgr is not None and away_bgr is not None:
        cost_direct = perceptual_kit_distance(col_0, home_bgr) + perceptual_kit_distance(col_1, away_bgr)
        cost_swap = perceptual_kit_distance(col_1, home_bgr) + perceptual_kit_distance(col_0, away_bgr)
        c0_is_team_a = (cost_direct <= cost_swap)

    result = {}
    for tid, l in zip(track_ids, labels):
        is_gk = False
        if cls_by_track:
            c = cls_by_track.get(tid)
            is_gk = (c == "goalkeeper" or getattr(c, "value", "") == "goalkeeper")
        if is_gk and (home_gk_bgr is not None or away_gk_bgr is not None) and tid in track_colors:
            result[tid] = classify_team_by_known_colors(
                track_colors[tid], home_bgr, away_bgr, home_gk_bgr, away_gk_bgr, is_gk=True
            )
        else:
            if l == 0:
                result[tid] = Team.TEAM_A if c0_is_team_a else Team.TEAM_B
            else:
                result[tid] = Team.TEAM_B if c0_is_team_a else Team.TEAM_A

    # Fill in any missing tracks from track_colors
    for tid in track_colors:
        if tid not in result:
            is_gk = False
            if cls_by_track:
                c = cls_by_track.get(tid)
                is_gk = (c == "goalkeeper" or getattr(c, "value", "") == "goalkeeper")
            if home_bgr is not None and away_bgr is not None:
                result[tid] = classify_team_by_known_colors(
                    track_colors[tid], home_bgr, away_bgr, home_gk_bgr, away_gk_bgr, is_gk=is_gk
                )
            else:
                result[tid] = Team.UNKNOWN

    # Sanity guardrail on final result
    outfield_ids = [
        tid for tid in result
        if not (cls_by_track and (cls_by_track.get(tid) == "goalkeeper" or getattr(cls_by_track.get(tid), "value", "") == "goalkeeper"))
    ]
    team_a_count = sum(1 for tid in outfield_ids if result[tid] == Team.TEAM_A)
    ratio_a = team_a_count / max(len(outfield_ids), 1)
    if ratio_a < 0.25 or ratio_a > 0.75:
        if home_bgr is not None and away_bgr is not None:
            return _balanced_margin_partition(
                list(result.keys()), track_colors, home_bgr, away_bgr, home_gk_bgr, away_gk_bgr, cls_by_track
            )

    return result