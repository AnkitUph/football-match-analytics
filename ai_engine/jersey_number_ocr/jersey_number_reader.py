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
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_EVERY = 3      # OCR every Nth frame a track is visible
MIN_CROP_HEIGHT_PX = 60       # crops smaller than this are useless for OCR
UPSCALE_HEIGHT_PX = 200       # standardised crop height fed to Tesseract
TARGET_CROP_SIZE = (128, 64)  # (Height, Width) for temporal neural network
MAX_JERSEY_NUMBER = 99
MIN_OCR_CONFIDENCE = 40.0     # per-word fallback confidence floor
MIN_DEEP_CONFIDENCE = 0.25    # deep temporal model confidence threshold (0.0 - 1.0)
MIN_OCR_READINGS = 3          # a number needs this many independent readings
VOTE_MAJORITY_FRACTION = 0.5  # winner must hold this share of weighted votes
DISPLAY_CONFIDENCE_THRESHOLD = 0.6  # video overlay only shows numbers at/above this share

_DIGITS_ONLY = re.compile(r"\D+")

_MODEL_PATHS = (
    os.path.join(os.path.dirname(__file__), "..", "models", "jersey_number_temporal_best.pth"),
    os.path.join(os.path.dirname(__file__), "..", "models", "jersey_number_temporal.pth"),
)


def _deep_model_available():
    return any(os.path.exists(p) for p in _MODEL_PATHS)


class JerseyNumberTemporalNet(nn.Module):
    """
    Spatio-temporal neural network for football jersey number recognition.
    - Backbone: EfficientNet-B0 feature extractor (1280-dim)
    - Temporal: 2-layer BiLSTM + attention pooling over tracklet sequence
    - Heads: Independent tens (0-9 + none) and units (0-9) digit classifiers
    """
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        import timm
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.fc_tens = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 11)  # 0-9 and 10=None (single-digit)
        )
        self.fc_units = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)  # 0-9
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.backbone(x_flat)
        feats = feats.view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        pooled = torch.sum(lstm_out * attn_weights, dim=1)
        tens_logits = self.fc_tens(pooled)
        units_logits = self.fc_units(pooled)
        return tens_logits, units_logits


def _load_deep_model():
    """Attempt to load trained PyTorch temporal jersey model weights."""
    if not _deep_model_available():
        return None
    for p in _MODEL_PATHS:
        if os.path.exists(p):
            try:
                model = JerseyNumberTemporalNet()
                state_dict = torch.load(p, map_location="cpu")
                model.load_state_dict(state_dict)
                model.eval()
                logger.info("Loaded trained JerseyNumberTemporalNet from %s", p)
                return model
            except Exception as e:
                logger.warning("Failed to load jersey model from %s: %s", p, e)
    return None


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
    def __init__(self, sample_every=DEFAULT_SAMPLE_EVERY, ocr_func=None, use_deep_model=True):
        self.sample_every = max(1, int(sample_every))
        self.deep_model = _load_deep_model() if (use_deep_model and ocr_func is None) else None
        
        if ocr_func is not None:
            self.ocr_func = ocr_func
            self.available = True
        elif self.deep_model is not None:
            self.ocr_func = None
            self.available = True
        elif _tesseract_available():
            self.ocr_func = _pytesseract_read
            self.available = True
        else:
            self.ocr_func = None
            self.available = False

        # track_id -> {jersey_number: weighted_votes}
        self.track_votes = defaultdict(lambda: defaultdict(float))
        self.track_reading_count = defaultdict(int)
        # final outcome: track_id -> int jersey number
        self.track_jersey_numbers = {}
        # final outcome: track_id -> weighted vote share of the winning number
        self.track_jersey_confidence = {}

    def predict_tracklet_deep(self, crops: list[np.ndarray]) -> tuple[int | None, float]:
        """
        Run sequence inference through JerseyNumberTemporalNet on T player crops.
        Returns (predicted_jersey_number, confidence).
        """
        if self.deep_model is None or not crops:
            return None, 0.0

        try:
            # Preprocess and stack T crops into (1, T, 3, 128, 64) tensor
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3, 1, 1)

            processed_frames = []
            for crop in crops:
                if crop is None or crop.size == 0:
                    continue
                # Resize to (128, 64) Height x Width
                resized = cv2.resize(crop, (TARGET_CROP_SIZE[1], TARGET_CROP_SIZE[0]))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                chw = np.transpose(rgb, (2, 0, 1))  # (3, H, W)
                processed_frames.append(chw)

            if not processed_frames:
                return None, 0.0

            # Pad or truncate to 8 frames
            while len(processed_frames) < 8:
                processed_frames.append(processed_frames[-1])
            processed_frames = processed_frames[:8]

            tensor = torch.from_numpy(np.array(processed_frames, dtype=np.float32)).unsqueeze(0)  # (1, 8, 3, H, W)
            # Normalize with ImageNet stats
            tensor = (tensor - torch.from_numpy(mean)) / torch.from_numpy(std)

            with torch.no_grad():
                tens_logits, units_logits = self.deep_model(tensor)
                p_tens = F.softmax(tens_logits, dim=1)
                p_units = F.softmax(units_logits, dim=1)

                tens_idx = torch.argmax(p_tens, dim=1).item()
                units_idx = torch.argmax(p_units, dim=1).item()
                conf_tens = p_tens[0, tens_idx].item()
                conf_units = p_units[0, units_idx].item()

                if tens_idx == 10:  # No tens digit
                    number = units_idx
                    total_conf = conf_units
                else:
                    number = tens_idx * 10 + units_idx
                    # Geometric mean of tens & units confidence: sqrt(p_tens * p_units)
                    total_conf = float(np.sqrt(conf_tens * conf_units))

                if 1 <= number <= MAX_JERSEY_NUMBER and total_conf >= MIN_DEEP_CONFIDENCE:
                    return number, float(round(total_conf, 3))
                return None, 0.0
        except Exception as e:
            logger.debug("Deep temporal prediction failed: %s", e)
            return None, 0.0

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

        if self.deep_model is not None:
            # Deep temporal network: collect sequence of torso crops per track
            track_crops = defaultdict(list)
            for frame_number, frame_tracks in enumerate(tracks.get("players", [])):
                if candidate_frames is not None and frame_number not in candidate_frames:
                    continue
                if frame_number % self.sample_every != 0:
                    continue
                frame = frames[frame_number]
                for track_id, info in frame_tracks.items():
                    crop = self.crop_jersey(frame, info.get("bbox"))
                    if crop is not None and crop.size > 0:
                        track_crops[track_id].append(crop)

            self.track_jersey_numbers = {}
            self.track_jersey_confidence = {}
            for track_id, crops in track_crops.items():
                if crops:
                    num, conf = self.predict_tracklet_deep(crops)
                    if num is not None:
                        self.track_jersey_numbers[track_id] = num
                        self.track_jersey_confidence[track_id] = conf
        else:
            # Traditional OCR fallback with temporal voting
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
