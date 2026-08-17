"""
Shared YOLO model loading, cached so Stage 1 (Detector) and Stage 2
(Tracker) — which both need the same underlying weights — don't load
best.pt into memory twice.
"""

from functools import lru_cache
from pathlib import Path

from ai_engine.config import DetectionConfig


@lru_cache(maxsize=4)
def _load_yolo(model_path_str: str):
    from ultralytics import YOLO
    return YOLO(model_path_str)


def get_yolo_model(config: DetectionConfig):
    if not Path(config.model_path).exists():
        raise FileNotFoundError(
            f"Detection weights not found at {config.model_path}. "
            "Copy your trained best.pt into ai_engine/models/."
        )
    return _load_yolo(str(config.model_path))
