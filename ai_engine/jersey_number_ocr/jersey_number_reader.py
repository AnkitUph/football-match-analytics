"""
Jersey-number OCR for tracked outfield players.

Operates ONLY on ``tracks["players"]`` — referees and goalkeepers are already
separated by the custom YOLO model and are not given numbers (outfield jersey
numbers are the identity signal this stage consumes).

Pipeline per track_id:
  1. Sample a subset of frames (``sample_every``) where the player is visible.
  2. Crop the upper-torso region of the bbox (where shirt numbers sit) and
     preprocess it for Tesseract (resize, grayscale, CLAHE contrast
     enhancement, Gaussian blur, Otsu binarization in both polarities).
  3. OCR digits via pytesseract, keeping only plausible 1-99 readings with
     their per-word confidence.
  4. Temporal voting across all frames: the winning number must hold a
     majority of the *weighted* vote AND come from enough independent OCR
     readings. A single misread (or a noisy crop) therefore cannot flip a
     player's identity.

The module degrades gracefully: when pytesseract is missing or the Tesseract
binary is not installed, ``available`` is False and the reader is a no-op, so
the rest of the pipeline keeps working unchanged.

Tests inject a fake ``ocr_func`` so the voting logic can be exercised without
a Tesseract installation.
"""

import logging
import re
from collections import defaultdict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_EVERY = 3      # OCR every Nth frame a track is visible
MIN_CROP_HEIGHT_PX = 60       # crops smaller than this are useless for OCR
UPSCALE_HEIGHT_PX = 200       # standardised crop height fed to Tesseract
MAX_JERSEY_NUMBER = 99
MIN_OCR_CONFIDENCE = 40.0     # per-word Tesseract confidence floor
MIN_OCR_READINGS = 3          # a number needs this many independent readings
VOTE_MAJORITY_FRACTION = 0.5  # winner must hold this share of weighted votes
DISPLAY_CONFIDENCE_THRESHOLD = 0.6  # video overlay only shows numbers at/above this share

_DIGITS_ONLY = re.compile(r"\D+")

# Common Tesseract install locations checked when the binary is not on PATH.
_TESSERACT_CANDIDATES = (
    "tesseract",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def _resolve_tesseract_cmd():
    import shutil

    for candidate in _TESSERACT_CANDIDATES:
        if shutil.which(candidate):
            return candidate
    return None


def _pytesseract_read(image):
    """Run Tesseract on one preprocessed crop, returning [(number, confidence)]."""
    import pytesseract

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config="--psm 7 -c tessedit_char_whitelist=0123456789",
    )
    readings = []
    for text, conf_text in zip(data.get("text", []), data.get("conf", [])):
        try:
            conf = float(conf_text or 0)
        except (TypeError, ValueError):
            conf = 0.0
        digits = _DIGITS_ONLY.sub("", text or "")
        if not digits:
            continue
        number = int(digits)
        if 1 <= number <= MAX_JERSEY_NUMBER:
            readings.append((number, conf))
    return readings


def _tesseract_available():
    try:
        import pytesseract

        tesseract_cmd = _resolve_tesseract_cmd()
        if tesseract_cmd is not None:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        pytesseract.get_tesseract_version()
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Tesseract OCR unavailable (%s); jersey numbers will not be read.", exc)
        return False


class JerseyNumberReader:
    def __init__(self, sample_every=DEFAULT_SAMPLE_EVERY, ocr_func=None):
        self.sample_every = max(1, int(sample_every))
        if ocr_func is not None:
            self.ocr_func = ocr_func
            self.available = True
        elif _tesseract_available():
            self.ocr_func = _pytesseract_read
            self.available = True
        else:
            self.ocr_func = None
            self.available = False

        # track_id -> {jersey_number: weighted_votes}
        self.track_votes = defaultdict(lambda: defaultdict(float))
        # track_id -> number of confident OCR readings seen (low-confidence
        # results are ignored and never count as evidence)
        self.track_reading_count = defaultdict(int)
        # final outcome: track_id -> int jersey number
        self.track_jersey_numbers = {}
        # final outcome: track_id -> weighted vote share of the winning number
        self.track_jersey_confidence = {}

    # ------------------------------------------------------------------ #
    # Crop + preprocess
    # ------------------------------------------------------------------ #
    @staticmethod
    def crop_jersey(frame, bbox):
        """Return the upper-torso crop of a player bbox, or None if too small."""
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        height = y2 - y1
        width = x2 - x1
        if height < MIN_CROP_HEIGHT_PX or width <= 0:
            return None
        top = max(0, y1)
        bottom = max(top, y1 + int(height * 0.55))  # chest/shoulder region
        left = max(0, x1)
        right = max(left, x2)
        crop = frame[top:bottom, left:right]
        return crop if crop.size else None

    @staticmethod
    def _binarize(crop):
        """Return (normal, inverted) Otsu-binarized grayscale versions.

        Preprocessing chain: resize -> grayscale -> CLAHE contrast
        enhancement -> slight blur -> Otsu threshold. Both polarities are
        produced because shirt numbers can be either lighter or darker than
        the shirt fabric.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scale = UPSCALE_HEIGHT_PX / gray.shape[0]
        if scale != 1:
            new_width = max(1, int(round(gray.shape[1] * scale)))
            gray = cv2.resize(gray, (new_width, UPSCALE_HEIGHT_PX), interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh, cv2.bitwise_not(thresh)

    def _ocr_image(self, image):
        if self.ocr_func is None:
            return []
        try:
            return self.ocr_func(image)
        except Exception:  # pragma: no cover - engine exceptions must not kill tracking
            logger.debug("OCR failed for a jersey crop", exc_info=True)
            return []

    def read_track(self, frame, bbox):
        """OCR one player's jersey once; returns [(number, confidence)]."""
        crop = self.crop_jersey(frame, bbox)
        if crop is None:
            return []
        readings = []
        for variant in self._binarize(crop):
            readings.extend(self._ocr_image(variant))
        return readings

    # ------------------------------------------------------------------ #
    # Frame sweep + temporal voting
    # ------------------------------------------------------------------ #
    def add_jersey_numbers_to_tracks(self, frames, tracks, analysis_indices=None):
        """Read numbers across frames, vote per track_id, write results back.

        When ``analysis_indices`` is given (the frames YOLO actually analysed,
        from ``Tracker.analysis_indices``), OCR runs only on those selected
        frames - interpolated frames never produce crops. The voted number is
        written into every frame entry of the player's track
        (``info["jersey_number"]``) so downstream drawing/persistence can rely
        on it without needing this reader object. No-op if OCR is unavailable.
        """
        self.track_votes = defaultdict(lambda: defaultdict(float))
        self.track_reading_count = defaultdict(int)

        if not self.available:
            logger.warning("Jersey OCR is unavailable; skipping number detection.")
            return

        candidate_frames = set(analysis_indices) if analysis_indices is not None else None
        for frame_number, frame_tracks in enumerate(tracks.get("players", [])):
            if candidate_frames is not None and frame_number not in candidate_frames:
                continue
            if frame_number % self.sample_every != 0:
                continue
            frame = frames[frame_number]
            for track_id, info in frame_tracks.items():
                readings = self.read_track(frame, info.get("bbox"))
                if not readings:
                    continue
                for number, confidence in readings:
                    if confidence < MIN_OCR_CONFIDENCE:
                        # Low-confidence results are ignored entirely: they
                        # neither vote nor count as evidence.
                        continue
                    self.track_reading_count[track_id] += 1
                    self.track_votes[track_id][number] += confidence

        self.track_jersey_numbers = self._temporal_vote()

        for frame_tracks in tracks.get("players", []):
            for track_id, info in frame_tracks.items():
                number = self.track_jersey_numbers.get(track_id)
                if number is not None:
                    info["jersey_number"] = number
                    info["jersey_confidence"] = self.track_jersey_confidence.get(track_id, 0.0)

        if self.track_jersey_numbers:
            logger.info("Jersey numbers assigned: %s", self.track_jersey_numbers)

    def _temporal_vote(self):
        """Decide each track's number from weighted, majority-guarded votes."""
        results = {}
        self.track_jersey_confidence = {}
        for track_id, votes in self.track_votes.items():
            total_weight = sum(votes.values())
            reading_count = self.track_reading_count.get(track_id, 0)
            if total_weight <= 0 or reading_count < MIN_OCR_READINGS:
                continue
            winner, winner_weight = max(votes.items(), key=lambda item: item[1])
            share = winner_weight / total_weight
            if share > VOTE_MAJORITY_FRACTION:
                results[track_id] = winner
                self.track_jersey_confidence[track_id] = round(share, 3)
        return results
