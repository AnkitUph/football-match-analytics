"""
Stage 3c: Jersey Number OCR (secondary/confirming signal).

Use opportunistically — when it works, it's a near-instant confirm/override
for identity matching in Stage 5. Don't rely on it as a primary signal;
broadcast resolution and player orientation make it unreliable frame to
frame.

HONEST EXPECTATION: on wide broadcast footage (players a few dozen pixels
tall), most individual OCR reads on a torso crop will fail outright or
misread digits. aggregate_jersey_number() is deliberately conservative
(requires multiple independent frames to agree before accepting a number)
specifically because of this — a single lucky high-confidence read is a
known failure mode, not a rare edge case, at this resolution. Expect a
real, possibly large, fraction of tracklets to end up with
jersey_number=None. That's the aggregation working as intended.

HISTORICAL ONE-CROP OBSERVATION (test_11.mp4): tested against the single LARGEST player
bounding box in the entire match (~104x76px, full bbox, no torso crop) —
EasyOCR found zero legible text regions at all (empty result, not a
low-confidence miss). This suggests resolution limited OCR on that crop;
it is not a benchmark across footage. A jersey number is maybe 15-20% of torso height, which on a 104px
full-body box is a handful of pixels, well below what any OCR model can
resolve through broadcast compression. DEFAULT_CONFIG.enable_ocr is set
to False by default; higher-resolution footage still needs a reviewed
accuracy estimate before roster identification relies on OCR output.

An earlier separate Tesseract-based
implementation (CLAHE contrast enhancement, dual-polarity Otsu
binarization, cubic upscaling to 200px height, confidence-weighted
majority voting — a well-designed module in its own right) was tested
against the exact same crop. Zero raw digit candidates on EITHER
polarity, before any confidence filtering. This was a limited manual spot
check, not a comparison of OCR systems or preprocessing methods. Keep OCR
disabled by default and evaluate it only on footage where shirt numbers
are visibly legible. Do not infer accuracy from upscaling or confidence
thresholds alone.
"""

from collections import defaultdict

import numpy as np

from ai_engine.config import TeamReidConfig
from ai_engine.jersey_number_ocr import JerseyNumberReader
from ai_engine.utils.types import Team, Tracklet


class JerseyOCR:
    def __init__(self, config: TeamReidConfig):
        self.config = config
        self._reader = None
        self._temporal_reader = JerseyNumberReader()

    def _load_reader(self):
        if self._reader is not None:
            return self._reader

        import easyocr
        self._reader = easyocr.Reader(["en"], gpu=False)
        return self._reader

    def read_number(self, crop: np.ndarray) -> tuple[int | None, float]:
        """
        Returns (number, confidence). Returns (None, 0.0) if nothing
        legible was found, the read wasn't purely digits, or confidence
        was below config.ocr_min_confidence.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 120:
            return None, 0.0

        reader = self._load_reader()
        try:
            results = reader.readtext(crop, allowlist="0123456789")
        except Exception:
            return None, 0.0

        if not results:
            return None, 0.0

        # Each result is (bbox, text, confidence).
        _, text, conf = max(results, key=lambda r: r[2])
        if conf < self.config.ocr_min_confidence or not text.isdigit():
            return None, 0.0
        return int(text), conf


def aggregate_jersey_number(
    readings: list[tuple[int, float]], min_agreeing_reads: int = 2
) -> tuple[int | None, float]:
    """
    Majority vote across independent per-frame OCR reads for one
    tracklet.
    """
    if not readings:
        return None, 0.0

    votes: dict[int, list[float]] = defaultdict(list)
    for number, conf in readings:
        votes[number].append(conf)

    candidates = [
        (number, len(confs), sum(confs))
        for number, confs in votes.items()
        if len(confs) >= min_agreeing_reads
    ]
    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda c: (c[1], c[2]), reverse=True)
    best_number, best_count, best_conf_sum = candidates[0]
    return best_number, best_conf_sum / best_count


def run_jersey_ocr_for_tracklets(
    tracklets: list[Tracklet],
    video_path: str,
    config: TeamReidConfig,
    max_samples_per_tracklet: int | None = None,
    min_agreeing_reads: int | None = None,
) -> None:
    """
    Mutates each PLAYER tracklet's .jersey_number/.jersey_number_conf in place.
    Uses deep temporal network (JerseyNumberTemporalNet) when available, or
    per-frame OCR with voting.
    
    Optimized with single monotonic forward video pass (eliminating thousands
    of random backward seeks that previously caused 45+ minute bottlenecks).
    """
    import cv2
    from collections import defaultdict

    if max_samples_per_tracklet is None:
        max_samples_per_tracklet = config.ocr_max_samples_per_tracklet
    if min_agreeing_reads is None:
        min_agreeing_reads = config.ocr_min_agreeing_reads

    temporal_reader = JerseyNumberReader()
    ocr = JerseyOCR(config) if temporal_reader.deep_model is None else None

    # Filter to real player tracklets with meaningful duration (ignore 1-4 frame noise)
    player_tracklets = [
        t for t in tracklets 
        if t.team in (Team.TEAM_A, Team.TEAM_B) and (len(t.detections) >= 8 or len(tracklets) <= 40)
    ]
    if not player_tracklets:
        return

    # Build chronological frame mapping for single monotonic video sweep
    crops_needed_by_frame: dict[int, list[tuple[Tracklet, tuple]]] = defaultdict(list)
    tracklet_sampled_count: dict[int, int] = defaultdict(int)

    for t in player_tracklets:
        if not t.detections:
            continue
        n = len(t.detections)
        sample_count = min(max_samples_per_tracklet, n)
        sample_indices = sorted(set(
            round(i * (n - 1) / max(sample_count - 1, 1)) for i in range(sample_count)
        ))
        for idx in sample_indices:
            det = t.detections[idx]
            crops_needed_by_frame[det.frame_idx].append((t, det.bbox))

    if not crops_needed_by_frame:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return

    track_crops: dict[int, list] = defaultdict(list)
    track_readings: dict[int, list] = defaultdict(list)

    try:
        sorted_frames = sorted(crops_needed_by_frame.keys())
        curr_pos = 0

        for target_f in sorted_frames:
            # Monotonic forward positioning (no backward seeks)
            if target_f != curr_pos:
                if 0 < (target_f - curr_pos) <= 8:
                    while curr_pos < target_f:
                        cap.grab()
                        curr_pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
                    curr_pos = target_f

            ok, frame = cap.read()
            curr_pos += 1
            if not ok or frame is None:
                continue

            for t, bbox in crops_needed_by_frame[target_f]:
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(x1, 0), max(y1, 0)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 8:
                    continue

                torso_height = max(int(crop.shape[0] * 0.65), 1)
                torso_crop = crop[:torso_height, :].copy()
                if torso_crop.size == 0:
                    continue

                if temporal_reader.deep_model is not None:
                    track_crops[t.track_id].append(torso_crop)
                elif ocr is not None:
                    number, conf = ocr.read_number(torso_crop)
                    if number is not None:
                        track_readings[t.track_id].append((number, conf))

    finally:
        cap.release()

    # Fast neural prediction pass per player tracklet
    for t in player_tracklets:
        if temporal_reader.deep_model is not None:
            crops = track_crops.get(t.track_id, [])
            if crops:
                num, conf = temporal_reader.predict_tracklet_deep(crops)
                t.jersey_number = num
                t.jersey_number_conf = conf
        else:
            readings = track_readings.get(t.track_id, [])
            if readings:
                jersey_number, jersey_conf = aggregate_jersey_number(readings, min_agreeing_reads)
                t.jersey_number = jersey_number
                t.jersey_number_conf = jersey_conf
