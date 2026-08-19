"""
Stage 1: Player & Ball Detection.

Wraps your trained YOLO11s checkpoint (best.pt from the Kaggle notebook —
mAP50 0.969 player / 0.880 referee / 0.828 goalkeeper / 0.792 ball).
Validated on real broadcast footage (test_11.avi) — clean detections
across all classes, ball detection working as expected (lower confidence,
which Stage 4's interpolation is designed for).
"""

from pathlib import Path

import numpy as np

from ai_engine.config import DetectionConfig, CLASS_NAMES
from ai_engine.stage1_detection.model_loader import get_yolo_model
from ai_engine.utils.types import Detection, ObjectClass
from ai_engine.utils.video_io import iter_frames_downsampled


class Detector:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = get_yolo_model(self.config)
        return self._model

    def detect_frame(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """Run detection on a single frame. Returns Detections in pixel space.

        Uses a lower confidence threshold for the ball class specifically
        (config.ball_conf_thresh) — see that field's docstring for the
        real-footage validation behind this. Same post-filtering approach
        as Tracker.track_video(), kept consistent between the two paths.
        """
        model = self._load_model()

        effective_conf = min(self.config.conf_thresh, self.config.ball_conf_thresh)
        results = model.predict(
            frame,
            imgsz=self.config.imgsz,
            conf=effective_conf,
            iou=self.config.iou_thresh,
            device=self.config.device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            cls_name = CLASS_NAMES[cls_id]
            conf = float(box.conf.item())

            if cls_name != "ball" and conf < self.config.conf_thresh:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    frame_idx=frame_idx,
                    cls=ObjectClass(cls_name),
                    conf=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections

    def detect_video(self, video_path: str) -> dict[int, list[Detection]]:
        """
        Runs detection across a whole clip at the configured downsample
        rate. Returns {frame_idx: [Detection, ...]}.

        Fine for a short validation clip. For full 15-min clips, prefer
        Tracker.track_video() instead (Stage 2) — ultralytics' .track()
        does detection + tracking in one pass, so running this separately
        first would mean decoding and running inference on the video twice.
        Kept here for isolated Stage 1 testing/debugging.
        """
        all_detections: dict[int, list[Detection]] = {}
        for frame_idx, frame in iter_frames_downsampled(
            video_path, target_fps=self.config.target_fps
        ):
            all_detections[frame_idx] = self.detect_frame(frame, frame_idx)
        return all_detections
