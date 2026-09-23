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
    Samples broadcast frames from across the video where full squads are visible
    (excluding close-up replays, referee close-ups, and graphic wipes), removes
    pitch grass reflection from player chest crops, and clusters outfield and
    goalkeeper kit colors separately into clean, identifiable kit swatches.

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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Sample candidate frames distributed across the video
    if total_frames > 150:
        num_candidates = min(36, total_frames // 25)
        candidates = [int(total_frames * (i + 1) / (num_candidates + 1)) for i in range(num_candidates)]
    else:
        candidates = [int(total_frames * i / max(1, sample_frame_count)) for i in range(sample_frame_count)]

    # Collect valid broadcast frames (where at least 6 players of normal broadcast scale are visible)
    valid_frames = []
    fallback_frames = []
    for f_idx in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        detections = detector.detect_frame(frame, f_idx)
        broadcast_players = [
            d for d in detections
            if d.cls.value in ("player", "goalkeeper") and 35 <= (d.bbox[3] - d.bbox[1]) <= 220
        ]
        if len(broadcast_players) >= 6:
            valid_frames.append((f_idx, frame, detections))
            if len(valid_frames) >= 12:
                break
        elif broadcast_players:
            fallback_frames.append((f_idx, frame, detections))

    frames_to_process = valid_frames if len(valid_frames) >= 3 else (valid_frames + fallback_frames)[:8]

    outfield_colors = []
    gk_colors = []

    for f_idx, frame, detections in frames_to_process:
        for det in detections:
            if det.cls.value not in ("player", "goalkeeper"):
                continue
            h_box = det.bbox[3] - det.bbox[1]
            if h_box < 35 or h_box > 220:
                continue
            x1, y1, x2, y2 = map(int, det.bbox)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            color = _sample_torso_no_grass(crop)
            if color is None:
                continue

            lum = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            if det.cls.value == "goalkeeper" or (lum < 75 and det.conf >= 0.55):
                gk_colors.append(color)
            else:
                outfield_colors.append(color)

    result = {"outfield": [], "goalkeeper": []}

    if len(outfield_colors) >= 4:
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(outfield_colors)
        result["outfield"] = [_enhance_kit_hex(c) for c in kmeans.cluster_centers_]

    if gk_colors:
        raw_gk_hexes = [_enhance_kit_hex(c) for c in gk_colors]
        # Filter out GK colors that are identical to outfield kit colors (goalkeeper never shares outfield kit)
        distinct_gk = [h for h in dict.fromkeys(raw_gk_hexes) if h not in result["outfield"]]
        result["goalkeeper"] = distinct_gk[:2] if distinct_gk else (raw_gk_hexes[:1] if raw_gk_hexes else [])

    return result


def _sample_torso_no_grass(crop: np.ndarray) -> np.ndarray | None:
    """
    Samples torso median color while filtering out background pitch grass
    hue contamination (H between 32 and 85 with saturation >= 40).
    """
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 15 or w < 8:
        return None

    torso = crop[int(h * 0.20):int(h * 0.42), int(w * 0.28):int(w * 0.72)]
    if torso.size == 0:
        return None

    hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    grass = (hsv_torso[:, :, 0] >= 32) & (hsv_torso[:, :, 0] <= 85) & (hsv_torso[:, :, 1] >= 40)
    non_grass = torso[~grass]
    if len(non_grass) >= 8:
        return np.median(non_grass.astype(np.float32), axis=0)
    return np.median(torso.reshape(-1, 3).astype(np.float32), axis=0)


def _enhance_kit_hex(bgr: np.ndarray) -> str:
    """
    Enhances raw camera-captured BGR colors into clean, distinct kit swatches,
    counteracting broadcast camera desaturation and lighting shifts.
    """
    b, g, r = float(bgr[0]), float(bgr[1]), float(bgr[2])
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    bgr_u8 = np.uint8([[[int(np.clip(b, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(r, 0, 255))]]])
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])

    # Black / Dark kit (Goalkeeper)
    if lum < 75 and s < 90:
        return "#222222"
    # White / Light kit
    if lum > 195 and s < 40:
        return "#FFFFFF"
    # Red kit (hue in red spectrum: 0..12 or 168..180)
    if (h <= 12 or h >= 168) and s > 35:
        return "#D71920"
    # Yellow kit (hue in yellow spectrum: 15..32)
    if (15 <= h <= 32) and s > 35:
        return "#FFD700"
    # Royal / Navy Blue kit (hue 95..130)
    if (95 <= h <= 130) and s > 35:
        return "#0055A5"
    # Green kit (hue 35..85)
    if (35 <= h <= 85) and s > 35:
        return "#00A859"

    # Fallback: boost saturation and value so it doesn't look washed out by camera
    s = min(255.0, max(120.0, s * 1.6))
    v = min(255.0, max(140.0, v * 1.2))
    enh_hsv = np.uint8([[[int(h), int(s), int(v)]]])
    enh_bgr = cv2.cvtColor(enh_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    eb, eg, er = [int(max(0, min(255, round(c)))) for c in enh_bgr]
    return f"#{er:02x}{eg:02x}{eb:02x}"


def _bgr_to_hex(bgr: np.ndarray) -> str:
    b, g, r = [int(max(0, min(255, round(c)))) for c in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"
