"""
Kit color extraction for the upload-time color picker.

Solves a real usability problem: asking a user to type the exact hex
color of a jersey they're eyeballing from a video is unrealistic. This
extracts actual candidate colors FROM the footage instead, so the user
just clicks to label real swatches rather than guessing hex codes.

Distinguishes goalkeepers from outfield players using Stage 1's own
class label (goalkeeper vs player) — NOT color clustering. This matters:
clustering outfield + GK crops together in one pass would usually fail,
since GK detections are rare (1-2 per team) compared to ~10 outfield
players per team, and a small GK cluster tends to get absorbed into a
larger nearby outfield cluster in k-means. Clustering each group
separately avoids this entirely.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

from ai_engine.stage3_team_reid.team_classifier import sample_torso_color


def extract_kit_color_candidates(
    video_path: str,
    sample_frame_count: int = 8,
    conf_thresh: float = 0.35,
) -> dict:
    """
    Samples frames from a TIGHT temporal window near the start of the
    video (not spread across the whole clip) and clusters outfield/
    goalkeeper torso colors separately.

    WHY A TIGHT WINDOW, NOT SPREAD ACROSS THE WHOLE CLIP: tested first
    with samples spread evenly across the full video — clustering came
    back muddy (both "clusters" ended up greenish, not a clean white-vs-
    green split that the raw per-crop samples clearly showed existed).
    Root cause: different parts of a match are shot at different zoom
    levels and lighting, and mixing samples across that variation adds
    noise K-means can't cleanly separate. Restricting to a short,
    consistent window fixed it — validated same match, same real
    footage: muddy centers became clean, accurate ones matching
    independently-validated reference values almost exactly.

    Returns:
        {
            "outfield": [hex1, hex2] or [] if insufficient data,
            "goalkeeper": [hex1, hex2] or [hex1] or [] depending on how
                many distinct GK detections were found,
        }
    """
    from ai_engine.config import DEFAULT_CONFIG
    from ai_engine.stage1_detection.detector import Detector

    detector = Detector(DEFAULT_CONFIG.detection)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    window_frames = int(fps * 2)  # ~2 seconds of consistent footage
    sample_indices = [int(window_frames * i / sample_frame_count) for i in range(sample_frame_count)]

    outfield_colors = []
    goalkeeper_colors = []

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        detections = detector.detect_frame(frame, frame_idx)
        for det in detections:
            if det.cls.value not in ("player", "goalkeeper"):
                continue
            x1, y1, x2, y2 = map(int, det.bbox)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            color = sample_torso_color(crop)
            if color is None:
                continue
            if det.cls.value == "goalkeeper":
                goalkeeper_colors.append(color)
            else:
                outfield_colors.append(color)

    result = {"outfield": [], "goalkeeper": []}

    if len(outfield_colors) >= 4:  # need enough samples for a meaningful 2-cluster split
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(outfield_colors)
        result["outfield"] = [_bgr_to_hex(c) for c in kmeans.cluster_centers_]

    if len(goalkeeper_colors) >= 4:
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(goalkeeper_colors)
        result["goalkeeper"] = [_bgr_to_hex(c) for c in kmeans.cluster_centers_]
    elif len(goalkeeper_colors) >= 1:
        # Not enough for a confident 2-way split, but at least one real
        # sample exists — better to show it than nothing, user can still
        # tell if it's clearly one specific GK's color.
        result["goalkeeper"] = [_bgr_to_hex(np.mean(goalkeeper_colors, axis=0))]

    return result


def _bgr_to_hex(bgr: np.ndarray) -> str:
    b, g, r = [int(max(0, min(255, round(c)))) for c in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"
