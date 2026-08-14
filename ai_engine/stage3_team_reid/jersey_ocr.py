"""
Stage 3c: Jersey Number OCR (secondary/confirming signal).

Use opportunistically — when it works, it's a near-instant confirm/override
for identity matching in Stage 5. Don't rely on it as a primary signal;
broadcast resolution and player orientation make it unreliable frame to
frame. Skip this file entirely for your first working version; add it once
Re-ID (reid.py) is already running and you want to sharpen matches further.
"""

import numpy as np

from ai_engine.config import TeamReidConfig


class JerseyOCR:
    def __init__(self, config: TeamReidConfig):
        self.config = config
        self._reader = None

    def _load_reader(self):
        if self._reader is not None:
            return self._reader
        # TODO:
        # import easyocr
        # self._reader = easyocr.Reader(["en"], gpu=True)
        raise NotImplementedError("JerseyOCR._load_reader is stubbed.")

    def read_number(self, crop: np.ndarray) -> tuple[int | None, float]:
        """
        Returns (number, confidence). Returns (None, 0.0) if nothing
        legible was found, or confidence was below config.ocr_min_confidence.

        TODO:
            reader = self._load_reader()
            # crop to upper-back region if you have pose/orientation info;
            # otherwise just try the full torso crop.
            results = reader.readtext(crop, allowlist="0123456789")
            if not results:
                return None, 0.0
            text, conf = max(results, key=lambda r: r[2])[1:3]
            if conf < self.config.ocr_min_confidence or not text.isdigit():
                return None, 0.0
            return int(text), conf
        """
        raise NotImplementedError("JerseyOCR.read_number is stubbed.")
