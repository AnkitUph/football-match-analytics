"""
Stage 2: Multi-Object Tracking within a single continuous camera shot.

Uses ultralytics' built-in BoT-SORT (via .track()) rather than calling
supervision's ByteTrack separately — ultralytics' tracker call does
detection + tracking together, so we don't decode/run-inference on the
video twice (once in Stage 1, once here). Stage 1's Detector is still
useful standalone for quick single-frame debugging, but for real clips
this module is the one that actually drives the pipeline.

IMPORTANT SCOPE NOTE: track_id here is only guaranteed unique WITHIN one
continuous shot. It is expected to reset/not persist across camera cuts —
that's by design. Stage 5's identity association (using Stage 3's Re-ID
embeddings) is what produces the permanent cross-cut Master ID.
"""

from collections import defaultdict, Counter

from ai_engine.config import DetectionConfig, TrackingConfig, CLASS_NAMES
from ai_engine.stage1_detection.model_loader import get_yolo_model
from ai_engine.utils.types import Detection, ObjectClass, Tracklet


class Tracker:
    def __init__(self, detection_config: DetectionConfig, tracking_config: TrackingConfig):
        self.detection_config = detection_config
        self.tracking_config = tracking_config
        self._model = None
        self.ball_by_frame: dict[int, Detection | None] = {}

    def _load_model(self):
        if self._model is None:
            self._model = get_yolo_model(self.detection_config)
        return self._model

    def track_video(self, video_path: str) -> dict[int, Tracklet]:
        """
        Runs BoT-SORT tracking across an entire video (or shot segment —
        for now, treat the whole clip as one shot; Stage 2.5 will later
        call this once per shot segment and reset between them).

        Returns {track_id: Tracklet}, with each Tracklet's `detections`
        list populated in frame order.

        Note on frame indexing: ultralytics' `.track(source=video_path)`
        walks every frame in the source video by default (no built-in
        downsample), unlike Stage 1's iter_frames_downsampled helper. For
        a 15-min clip at native fps this means far more frames processed
        than Stage 1's config.target_fps would suggest. Pass `vid_stride`
        below to approximate the same downsampling Stage 1 uses, so
        frame_idx stays comparable across stages.
        """
        model = self._load_model()

        # Approximate the Stage 1 downsample rate by frame-skipping.
        # ultralytics' vid_stride: keep 1 in every N frames.
        # We don't have the source fps here without opening the video
        # separately — do that once, cheaply, just for the stride calc.
        from ai_engine.utils.video_io import get_video_info

        info = get_video_info(video_path)
        source_fps = info.fps or self.detection_config.target_fps
        vid_stride = max(1, round(source_fps / self.detection_config.target_fps))

        tracklets: dict[int, Tracklet] = {}
        self.ball_by_frame = {}
        raw_balls_queue = []

        def _capture_raw_detections(predictor):
            if predictor.results:
                res = predictor.results[0]
                if res.boxes is not None:
                    ball_boxes = [b for b in res.boxes if int(b.cls.item()) == 0]
                    raw_balls_queue.append(ball_boxes)
                else:
                    raw_balls_queue.append([])
            else:
                raw_balls_queue.append([])

        # Ultralytics doesn't support per-class confidence thresholds in
        # one call — run the model at the LOWER of the two thresholds
        # (catches marginal ball detections), then post-filter non-ball
        # detections back up to the stricter general threshold below.
        # See config.py's ball_conf_thresh docstring for why this exists.
        effective_conf = min(self.detection_config.conf_thresh, self.detection_config.ball_conf_thresh)

        results_generator = model.track(
            source=video_path,
            imgsz=self.detection_config.imgsz,
            conf=effective_conf,
            iou=self.detection_config.iou_thresh,
            device=self.detection_config.device,
            tracker=str(self.tracking_config.tracker_yaml),
            persist=True,
            stream=True,
            vid_stride=vid_stride,
            verbose=False,
        )

        # Hook our callback at index 0 of on_predict_postprocess_end
        # so we extract raw ball detections BEFORE BoT-SORT discards them
        if "on_predict_postprocess_end" in model.callbacks:
            model.callbacks["on_predict_postprocess_end"].insert(0, _capture_raw_detections)

        frame_idx = 0
        for result in results_generator:
            # Harvest raw ball detection for this frame before tracker filtering
            if raw_balls_queue:
                balls = raw_balls_queue.pop(0)
                if balls:
                    best = max(balls, key=lambda b: float(b.conf.item()))
                    x1, y1, x2, y2 = best.xyxy[0].tolist()
                    self.ball_by_frame[frame_idx] = Detection(
                        frame_idx=frame_idx,
                        cls=ObjectClass.BALL,
                        conf=float(best.conf.item()),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                else:
                    self.ball_by_frame[frame_idx] = None
            else:
                self.ball_by_frame[frame_idx] = None

            if result.boxes is None or result.boxes.id is None:
                frame_idx += vid_stride
                continue

            for box in result.boxes:
                track_id = int(box.id.item())
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                cls_name = CLASS_NAMES[cls_id]

                # Post-filter: ball already passed effective_conf (the
                # lower bar) if we're here; non-ball classes need to
                # additionally clear the stricter general threshold.
                if cls_name != "ball" and conf < self.detection_config.conf_thresh:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detection = Detection(
                    frame_idx=frame_idx,
                    cls=ObjectClass(cls_name),
                    conf=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )

                if track_id not in tracklets:
                    tracklets[track_id] = Tracklet(
                        track_id=track_id,
                        shot_id=0,  # single-shot assumption for now
                        cls=detection.cls,
                    )
                tracklets[track_id].detections.append(detection)

            frame_idx += vid_stride

        # Majority-vote tracklet class across all its detections
        # Prevents a single noisy first-frame detection from locking a player into referee or vice-versa
        for t in tracklets.values():
            if t.detections:
                classes = [d.cls for d in t.detections if d.cls]
                if classes:
                    t.cls = Counter(classes).most_common(1)[0][0]

        return tracklets

    def summarize(self, tracklets: dict[int, Tracklet]) -> None:
        """Quick sanity-check printout — not part of the pipeline proper,
        just handy while validating Stage 2 on a test clip."""
        by_cls: dict[str, int] = defaultdict(int)
        for t in tracklets.values():
            by_cls[t.cls.value if t.cls else "unknown"] += 1

        print(f"Total tracklets: {len(tracklets)}")
        for cls_name, count in sorted(by_cls.items()):
            print(f"  {cls_name}: {count}")

        durations = sorted(
            ((t.track_id, t.duration_frames) for t in tracklets.values()),
            key=lambda x: -x[1],
        )
        print("Longest-lived tracklets (track_id, frames_tracked):")
        for track_id, duration in durations[:10]:
            print(f"  {track_id}: {duration}")
