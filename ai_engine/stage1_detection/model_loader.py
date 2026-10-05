"""
Shared YOLO model loading, cached so Stage 1 (Detector) and Stage 2
(Tracker) — which both need the same underlying weights — don't load
best.pt into memory twice.
"""

from functools import lru_cache
from pathlib import Path

from ai_engine.config import CLASS_NAMES, DetectionConfig


@lru_cache(maxsize=4)
def _load_yolo(model_path_str: str):
    from ultralytics import YOLO
    return YOLO(model_path_str)


def get_yolo_model(config: DetectionConfig):
    model_path = Path(config.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Detection model not found at {model_path}. Set DetectionConfig.model_path "
            "to a local Ultralytics checkpoint or exported model directory."
        )

    model = _load_yolo(str(model_path.resolve()))
    model_names = getattr(model, "names", None)
    if model_names is not None:
        items = model_names.items() if isinstance(model_names, dict) else enumerate(model_names)
        actual_names = {int(class_id): str(name).strip().lower() for class_id, name in items}
        expected_names = {int(class_id): name.strip().lower() for class_id, name in CLASS_NAMES.items()}
        if actual_names != expected_names:
            raise ValueError(
                "Detection model class names do not match ai_engine.config.CLASS_NAMES. "
                f"Expected {expected_names}, got {actual_names} for {model_path}."
            )
    return model
