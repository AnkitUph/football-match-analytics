"""
Stage 1: Player & Ball Detection.

Wraps your trained YOLO11s checkpoint (best.pt from the Kaggle notebook —
mAP50 0.969 player / 0.880 referee / 0.828 goalkeeper / 0.792 ball).
Nothing fancy here on purpose: this stage's only job is clean per-frame
detections. Ball-specific improvements belong in Stage 4, not here.

Drop your trained weights at: ai_engine/models/best.pt
"""

from pathlib import Path

import numpy as np

from ai_engine.config import DetectionConfig, CLASS_NAMES
from ai_engine.utils.types import Detection, ObjectClass
from ai_engine.utils.video_io import iter_frames_downsampled


class Detector:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self._model = None  # lazy-loaded, see _load_model

    def _load_model(self):
        if self._model is not None:
            return self._model

        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(
                f"Detection weights not found at {self.config.model_path}. "
                "Copy your trained best.pt from the Kaggle checkpoint into "
                "ai_engine/models/."
            )

        # TODO: uncomment once ultralytics is installed in this environment
        # from ultralytics import YOLO
        # self._model = YOLO(str(self.config.model_path))

        raise NotImplementedError(
            "Detector._load_model is stubbed — install `ultralytics` and "
            "uncomment the YOLO(...) load above."
        )

        return self._model

    def detect_frame(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """
        Run detection on a single frame. Returns a list of Detection objects
        in pixel space (xyxy).
        """
        model = self._load_model()

        # TODO: replace with real inference:
        # results = model.predict(
        #     frame,
        #     imgsz=self.config.imgsz,
        #     conf=self.config.conf_thresh,
        #     iou=self.config.iou_thresh,
        #     device=self.config.device,
        #     verbose=False,
        # )[0]
        #
        # detections = []
        # for box in results.boxes:
        #     cls_id = int(box.cls.item())
        #     x1, y1, x2, y2 = box.xyxy[0].tolist()
        #     detections.append(Detection(
        #         frame_idx=frame_idx,
        #         cls=ObjectClass(CLASS_NAMES[cls_id]),
        #         conf=float(box.conf.item()),
        #         x1=x1, y1=y1, x2=x2, y2=y2,
        #     ))
        # return detections

        raise NotImplementedError("Detector.detect_frame is stubbed.")

    def detect_video(
        self, video_path: str
    ) -> dict[int, list[Detection]]:
        """
        Runs detection across a whole clip at the configured downsample
        rate. Returns {frame_idx: [Detection, ...]}.

        For a 15-min clip at 12 fps this is ~10,800 frames — for the full
        pipeline you'll likely want to stream this (yield per-frame results
        to Stage 2 as you go) rather than materializing the whole dict in
        memory. This dict-returning version is fine for validating Stage 1
        alone on a short test clip first.
        """
        all_detections: dict[int, list[Detection]] = {}
        for frame_idx, frame in iter_frames_downsampled(
            video_path, target_fps=self.config.target_fps
        ):
            all_detections[frame_idx] = self.detect_frame(frame, frame_idx)
        return all_detections
