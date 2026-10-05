"""
Stage 2: Multi-Object Tracking within a single continuous camera shot.

Uses ultralytics' built-in BoT-SORT (via .track()) rather than calling
supervision's ByteTrack separately — ultralytics' tracker call does
detection + tracking together, so we don't decode/run-inference on the
video twice (once in Stage 1, once here). Stage 1's Detector is still
useful standalone for quick single-frame debugging, but for real clips
this module is the one that actually drives the pipeline.

Track IDs are made unique across one processed clip, while BoT-SORT's local
state resets at camera cuts. IDs do not persist across separate videos;
cross-cut identity association is a separate Stage 5 operation.
"""

from collections import defaultdict, Counter
from bisect import bisect_right
from dataclasses import replace
import logging

from ai_engine.config import DetectionConfig, TrackingConfig, CLASS_NAMES
from ai_engine.stage1_detection.model_loader import get_yolo_model
from ai_engine.utils.types import Detection, ObjectClass, ShotSegment, Tracklet

logger = logging.getLogger(__name__)


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

    def reset(self) -> None:
        """Clear this tracker's per-video state and cached BoT-SORT state.

        The YOLO model loader is shared across ``Tracker`` instances. Without
        resetting its predictor, a long-lived worker can carry track IDs and
        motion state from one video or shot into the next.
        """
        self.ball_by_frame.clear()
        if self._model is None:
            return

        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()):
            reset_tracker = getattr(tracker, "reset", None)
            if callable(reset_tracker):
                reset_tracker()
            else:
                logger.warning(
                    "Cached tracker %s does not expose reset(); IDs may carry "
                    "over between videos",
                    type(tracker).__name__,
                )

    def track_video(
        self,
        video_path: str,
        shot_segments: list[ShotSegment] | None = None,
    ) -> dict[int, Tracklet]:
        """
        Runs BoT-SORT across a clip. When Stage 2.5 shot segments are
        supplied, resets tracker state at each boundary while decoding the
        source only once. Returned track IDs are unique across the clip;
        each Tracklet retains its local shot_id.

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
        self.reset()

        # Approximate the Stage 1 downsample rate by frame-skipping.
        # ultralytics' vid_stride: keep 1 in every N frames.
        # We don't have the source fps here without opening the video
        # separately — do that once, cheaply, just for the stride calc.
        from ai_engine.utils.video_io import get_video_info

        info = get_video_info(video_path)
        source_fps = info.fps or self.detection_config.target_fps
        vid_stride = max(1, round(source_fps / self.detection_config.target_fps))

        segments = sorted(shot_segments or [], key=lambda shot: shot.start_frame)
        if segments:
            previous_end = 0
            for segment in segments:
                if segment.start_frame != previous_end or segment.end_frame <= segment.start_frame:
                    raise ValueError("shot_segments must be contiguous, ordered, non-empty half-open ranges")
                previous_end = segment.end_frame
            if segments[0].start_frame != 0:
                raise ValueError("shot_segments must start at source frame 0")
            if segments[-1].end_frame != info.frame_count:
                logger.warning(
                    "Shot detector end frame %d differs from OpenCV frame count %d; "
                    "aligning the final shot to the source video",
                    segments[-1].end_frame, info.frame_count,
                )
                final_shot = segments[-1]
                if final_shot.start_frame >= info.frame_count:
                    raise ValueError("final detected shot begins after the source video's last frame")
                segments[-1] = replace(final_shot, end_frame=info.frame_count)
            segment_starts = [segment.start_frame for segment in segments]

            def shot_at(frame_idx: int) -> ShotSegment:
                idx = bisect_right(segment_starts, frame_idx) - 1
                if idx < 0 or frame_idx >= segments[idx].end_frame:
                    raise ValueError(f"frame {frame_idx} is not covered by shot_segments")
                return segments[idx]
        else:
            segment_starts = []

            def shot_at(frame_idx: int) -> ShotSegment | None:
                return None

        tracklets: dict[int, Tracklet] = {}
        self.ball_by_frame = {}
        raw_balls_queue = []

        model_names = model.names
        if isinstance(model_names, dict):
            name_items = model_names.items()
        else:
            name_items = enumerate(model_names)
        ball_class_ids = {
            int(class_id)
            for class_id, class_name in name_items
            if str(class_name).strip().lower() == "ball"
        }
        if not ball_class_ids:
            raise ValueError(
                f"Detection model has no 'ball' class; model classes are {model_names!r}"
            )

        def _capture_raw_detections(predictor):
            if segments:
                # Ultralytics invokes this callback before its own tracking
                # callback and before incrementing predictor.seen. Convert
                # the current sampled-frame counter back to source coordinates.
                source_frame_idx = int(getattr(predictor, "seen", 0)) * vid_stride
                current_segment = shot_at(source_frame_idx)
                if current_segment.shot_id != current_shot_id[0]:
                    if current_shot_id[0] is not None:
                        for tracker in getattr(predictor, "trackers", ()):
                            reset_tracker = getattr(tracker, "reset", None)
                            if callable(reset_tracker):
                                reset_tracker()
                            else:
                                logger.warning(
                                    "Tracker %s cannot reset at shot boundary %d",
                                    type(tracker).__name__, current_segment.shot_id,
                                )
                    current_shot_id[0] = current_segment.shot_id

            if predictor.results:
                res = predictor.results[0]
                if res.boxes is not None:
                    ball_boxes = [
                        b for b in res.boxes if int(b.cls.item()) in ball_class_ids
                    ]
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

        # Hook before BoT-SORT filters detections so the weak, fast-moving
        # ball remains available to Stage 4. The YOLO instance is cached and
        # reused, so clean up the hook even if inference raises; otherwise a
        # long-lived Celery worker retains old per-video queues and callbacks.
        callback_list = model.callbacks.get("on_predict_postprocess_end")
        current_shot_id = [None]
        if callback_list is None:
            if segments:
                raise RuntimeError(
                    "Ultralytics has no on_predict_postprocess_end callback; "
                    "cannot guarantee BoT-SORT resets at shot boundaries"
                )
            logger.warning(
                "Ultralytics has no on_predict_postprocess_end callback; "
                "ball detections will be unavailable for this video"
            )
        else:
            # Also clear hooks left by a previous interrupted invocation.
            callback_list[:] = [
                callback
                for callback in callback_list
                if not getattr(callback, "_ai_engine_ball_capture", False)
            ]
            _capture_raw_detections._ai_engine_ball_capture = True
            callback_list.insert(0, _capture_raw_detections)

        frame_idx = 0
        global_id_by_local: dict[tuple[int, int], int] = {}
        next_global_track_id = 1
        try:
            for result in results_generator:
                if frame_idx % 500 == 0:
                    logger.info("Tracking progress: frame %d", frame_idx)
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

                current_segment = shot_at(frame_idx) if segments else None
                shot_id = current_segment.shot_id if current_segment else 0

                if result.boxes is None or result.boxes.id is None:
                    frame_idx += vid_stride
                    continue

                for box in result.boxes:
                    local_track_id = int(box.id.item())
                    id_key = (shot_id, local_track_id)
                    if id_key not in global_id_by_local:
                        global_id_by_local[id_key] = next_global_track_id
                        next_global_track_id += 1
                    track_id = global_id_by_local[id_key]
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
                            shot_id=shot_id,
                            cls=detection.cls,
                        )
                    tracklets[track_id].detections.append(detection)

                frame_idx += vid_stride
        finally:
            if callback_list is not None:
                callback_list[:] = [
                    callback
                    for callback in callback_list
                    if callback is not _capture_raw_detections
                ]

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
