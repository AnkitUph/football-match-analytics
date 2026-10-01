"""Extract match-specific outfield and goalkeeper kit swatches from video.

The upload picker returns candidate colors only. The user assigns each
candidate to Home/Away; this module does not infer team identity from color.
"""

from dataclasses import replace

import cv2
import numpy as np
from sklearn.cluster import KMeans

from ai_engine.stage3_team_reid.team_classifier import bgr_to_standard_lab


def extract_kit_color_candidates(
    video_path: str,
    sample_frame_count: int = 8,
    conf_thresh: float = 0.35,
) -> dict:
    """Sample player torsos across a match and return up to two kit colors per group.

    Candidate frames are spread across the entire video. Each detection
    contributes one torso color, so a large close-up or a single frame cannot
    dominate the palette. HSV saturation isolates jersey fabric from white
    sponsors and shadows; no fixed hue palette is used, so green and unusual
    kit colors remain valid.
    """
    from ai_engine.config import DEFAULT_CONFIG
    from ai_engine.stage1_detection.detector import Detector

    detector = Detector(replace(DEFAULT_CONFIG.detection, conf_thresh=conf_thresh))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frame count: {video_path}")

    # Sample every ~5 seconds (up to 120 frames) across the whole timeline.
    # A 20-second stride skipped the clearly visible green goalkeeper at 1:08
    # in 10_min_match.mp4, so rare kits need denser temporal coverage.
    desired = min(120, max(sample_frame_count, int(frame_count / max(fps * 5, 1))))
    frame_indices = np.linspace(0, frame_count - 1, num=desired, dtype=int)
    outfield_samples: list[np.ndarray] = []
    goalkeeper_samples: list[np.ndarray] = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        detections = detector.detect_frame(frame, int(frame_idx))
        frame_outfield = []
        frame_goalkeepers = []
        for det in detections:
            cls = getattr(det.cls, "value", str(det.cls))
            if cls not in ("player", "goalkeeper"):
                continue
            x1, y1, x2, y2 = map(int, det.bbox)
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            box_h, box_w = y2 - y1, x2 - x1
            # Close-ups, crowd detections, and merged person boxes have a
            # different scale/domain from the wide-shot players whose kits we
            # need. They were introducing skin, boards, and background colors.
            if box_h < 30 or box_h > 220 or box_w < 8 or box_h / box_w < 1.1:
                continue
            color = _sample_jersey_color(frame[y1:y2, x1:x2])
            if color is None:
                continue
            if cls == "goalkeeper":
                if det.conf >= 0.50:
                    frame_goalkeepers.append(color)
            else:
                frame_outfield.append(color)

        # Retain only frames where the detector found enough outfield players
        # to make a team palette plausible. Limit every frame's contribution.
        if len(frame_outfield) >= 4:
            outfield_samples.extend(frame_outfield[:18])
        goalkeeper_samples.extend(frame_goalkeepers[:2])

    cap.release()
    return {
        "outfield": _cluster_swatches(outfield_samples, min_samples=4),
        "goalkeeper": _cluster_swatches(goalkeeper_samples, min_samples=2),
    }


def _sample_jersey_color(crop: np.ndarray) -> np.ndarray | None:
    """Return a robust BGR torso color from a full-person detection crop."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 15 or w < 8:
        return None

    # Central chest patch avoids heads, shorts, and most grass around loose
    # boxes. Keep the vertical band broad enough for small distant players.
    torso = crop[int(h * 0.24):int(h * 0.44), int(w * 0.38):int(w * 0.62)]
    if torso.size == 0:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    # Use the chromatic fabric mode rather than the raw median, which is often
    # pulled toward white sponsor marks. Hue is deliberately unrestricted.
    colored = torso[(saturation >= 55) & (value >= 35)]
    if len(colored) >= max(6, int(torso.shape[0] * torso.shape[1] * 0.06)):
        hsv_colored = cv2.cvtColor(colored.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        hue = hsv_colored[:, 0]
        # Find the dominant hue using circular distance, avoiding the 0/180
        # discontinuity for red and claret shirts.
        bins = np.arange(0, 181, 5)
        hist, edges = np.histogram(hue, bins=bins, weights=hsv_colored[:, 1])
        center = (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]) / 2
        delta = np.abs(hue - center)
        delta = np.minimum(delta, 180 - delta)
        fabric = colored[delta <= 10]
        if len(fabric) >= 4:
            return np.median(fabric.astype(np.float32), axis=0)
        return np.median(colored.astype(np.float32), axis=0)

    # Achromatic kits (white/black/gray) have no stable hue. The narrow chest
    # patch is the best available signal; don't discard it for low saturation.
    return np.median(torso.reshape(-1, 3).astype(np.float32), axis=0)


def _cluster_swatches(colors: list[np.ndarray], min_samples: int) -> list[str]:
    """Cluster per-player colors in shadow-attenuated Lab and return medoids."""
    if len(colors) < min_samples:
        return []
    bgr = np.asarray(colors, dtype=np.float32)
    lab = np.asarray([bgr_to_standard_lab(c) for c in bgr], dtype=np.float32)
    lab[:, 0] *= 0.25

    if len(colors) < 4:
        centers = [np.median(bgr, axis=0)]
    else:
        model = KMeans(n_clusters=2, n_init=10, random_state=0).fit(lab)
        groups = [bgr[model.labels_ == i] for i in range(2)]
        # Tiny clusters usually represent a referee, a crop failure, or a
        # single noisy detection. Do not advertise them as a second team kit.
        if min(map(len, groups)) < max(2, int(len(colors) * 0.08)):
            groups = [groups[int(np.argmax([len(g) for g in groups]))]]
        centers = [np.median(group, axis=0) for group in groups]

    # Stable visual ordering only; the UI still asks the user to assign sides.
    centers.sort(key=lambda c: int(cv2.cvtColor(
        np.uint8([[np.clip(c, 0, 255)]]), cv2.COLOR_BGR2HSV
    )[0, 0, 0]))
    return [_bgr_to_hex(c) for c in centers]


def _bgr_to_hex(bgr: np.ndarray) -> str:
    b, g, r = [int(np.clip(round(float(c)), 0, 255)) for c in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"
