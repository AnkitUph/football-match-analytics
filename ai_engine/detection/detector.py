import logging
from pathlib import Path

from ultralytics import YOLO

from ai_engine.schemas.track import BoundingBox, Detection

logger = logging.getLogger(__name__)


class Detector:
    def __init__(
        self,
        model_path,
        imgsz=1280,
        confidence=0.25,
        device=None,
    ):
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.confidence = confidence
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names

        logger.info(
            "Loaded YOLO model: %s",
            self.model_path,
        )

        logger.info(
            "Model classes: %s",
            self.class_names,
        )

        logger.info(
            "Image size: %d",
            self.imgsz,
        )

        logger.info(
            "Confidence threshold: %.2f",
            self.confidence,
        )

    def detect_frame(
        self,
        frame,
        frame_index,
    ):
        kwargs = {
            "imgsz": self.imgsz,
            "conf": self.confidence,
            "verbose": False,
        }

        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(
            frame,
            **kwargs,
        )

        if not results:
            return []

        return self._parse_result(
            results[0],
            frame_index,
        )

    def detect_frames(
        self,
        frames,
        batch_size=16,
    ):
        all_detections = []

        if not frames:
            return all_detections

        total_frames = len(frames)

        for start in range(
            0,
            total_frames,
            batch_size,
        ):
            end = min(
                start + batch_size,
                total_frames,
            )

            logger.info(
                "Detecting frames %d-%d/%d",
                start,
                end - 1,
                total_frames,
            )

            batch = frames[start:end]

            kwargs = {
                "imgsz": self.imgsz,
                "conf": self.confidence,
                "verbose": False,
            }

            if self.device is not None:
                kwargs["device"] = self.device

            results = self.model.predict(
                batch,
                **kwargs,
            )

            for offset, result in enumerate(
                results
            ):
                frame_index = start + offset

                detections = self._parse_result(
                    result,
                    frame_index,
                )

                all_detections.append(
                    detections
                )

        return all_detections

    def _parse_result(
        self,
        result,
        frame_index,
    ):
        detections = []

        if result.boxes is None:
            return detections

        if len(result.boxes) == 0:
            return detections

        boxes = result.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for bbox, confidence, class_id in zip(
            xyxy,
            confidences,
            class_ids,
        ):
            class_name = self.class_names.get(
                int(class_id),
                str(class_id),
            )

            detections.append(
                Detection(
                    frame_index=frame_index,
                    class_id=int(class_id),
                    class_name=class_name,
                    confidence=float(confidence),
                    bbox=BoundingBox.from_xyxy(
                        bbox
                    ),
                )
            )

        return detections

    def get_class_id(
        self,
        class_name,
    ):
        for class_id, name in self.class_names.items():
            if name == class_name:
                return class_id

        return None

    def get_class_name(
        self,
        class_id,
    ):
        return self.class_names.get(
            class_id,
            str(class_id),
        )