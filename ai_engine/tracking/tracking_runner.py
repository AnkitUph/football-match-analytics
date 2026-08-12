
"""
Tracking runner for the football AI pipeline.

Pipeline:

    Video
      ↓
    YOLO Detector
      ↓
    ByteTrack
      ↓
    Track / TrackObservation schemas
      ↓
    Appearance Embeddings
      ↓
    Tracking statistics

This phase intentionally does NOT perform:

- global identity merging
- team assignment
- jersey-number recognition
- homography
- heatmap generation
- event detection
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict

import cv2

from ai_engine.detection.detector import Detector
from ai_engine.tracking.bytetrack_tracker import ByteTrackTracker

logger = logging.getLogger(__name__)


def run_tracking(
    video_path: str,
    model_path: str,
    fps: float | None = None,
    imgsz: int = 1280,
    confidence: float = 0.25,

    # ========================================================
    # BYTE TRACK CONFIGURATION
    # ========================================================

    track_activation_threshold: float = 0.40,
    lost_track_buffer: int = 100,
    minimum_matching_threshold: float = 0.70,
    minimum_consecutive_frames: int = 2,

    # ========================================================
    # APPEARANCE CONFIGURATION
    # ========================================================

    appearance_model_name: str = "osnet_x1_0",
    appearance_device: str = "cpu",
    appearance_interval: int = 10,
    appearance_refresh_on_new_track: bool = True,
) -> ByteTrackTracker:
    """
    Run YOLO detection + ByteTrack over a video.

    ByteTrack configuration is exposed here so that we can
    run controlled experiments without modifying the tracker
    implementation every time.

    Args:
        video_path:
            Input football video.

        model_path:
            YOLO model path.

        fps:
            Optional FPS override.

        imgsz:
            YOLO inference image size.

        confidence:
            YOLO detection confidence threshold.

        track_activation_threshold:
            ByteTrack track activation threshold.

        lost_track_buffer:
            Number of frames a lost track remains recoverable.

        minimum_matching_threshold:
            ByteTrack matching threshold.

        minimum_consecutive_frames:
            Consecutive detections required before activation.

        appearance_model_name:
            OSNet/TorchReID model.

        appearance_device:
            Appearance extraction device.

        appearance_interval:
            OSNet refresh interval.

        appearance_refresh_on_new_track:
            Immediately extract appearance for new tracks.
    """

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_path}"
        )

    try:

        # ====================================================
        # VIDEO METADATA
        # ====================================================

        video_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps is None or fps <= 0:
            fps = video_fps

        if fps is None or fps <= 0:
            fps = 25.0

        fps = float(fps)

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        video_width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        video_height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        logger.info(
            "Video opened: %s",
            video_path,
        )

        logger.info(
            "Resolution: %dx%d | FPS: %.2f | Frames: %d",
            video_width,
            video_height,
            fps,
            total_frames,
        )

        # ====================================================
        # DETECTOR
        # ====================================================

        detector = Detector(
            model_path=model_path,
            imgsz=imgsz,
            confidence=confidence,
        )

        logger.info(
            "Detector initialized"
        )

        logger.info(
            "Classes: %s",
            detector.class_names,
        )

        # ====================================================
        # BYTE TRACK
        # ====================================================

        tracker = ByteTrackTracker(
            fps=fps,

            # ------------------------------------------------
            # ByteTrack
            # ------------------------------------------------

            track_activation_threshold=(
                track_activation_threshold
            ),

            lost_track_buffer=(
                lost_track_buffer
            ),

            minimum_matching_threshold=(
                minimum_matching_threshold
            ),

            minimum_consecutive_frames=(
                minimum_consecutive_frames
            ),

            # ------------------------------------------------
            # Appearance
            # ------------------------------------------------

            appearance_model_name=(
                appearance_model_name
            ),

            appearance_device=(
                appearance_device
            ),

            appearance_interval=(
                appearance_interval
            ),

            appearance_refresh_on_new_track=(
                appearance_refresh_on_new_track
            ),
        )

        logger.info(
            "ByteTrack configuration: "
            "activation=%.2f | "
            "lost_buffer=%d | "
            "matching=%.2f | "
            "min_consecutive=%d",
            track_activation_threshold,
            lost_track_buffer,
            minimum_matching_threshold,
            minimum_consecutive_frames,
        )

        # ====================================================
        # STATISTICS
        # ====================================================

        frame_number = 0
        total_detections = 0

        unique_ids = set()

        id_frame_counts: Dict[int, int] = (
            defaultdict(int)
        )

        id_classes: Dict[int, str] = {}

        # ====================================================
        # FRAME LOOP
        # ====================================================

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            detections = detector.detect_frame(
                frame,
                frame_number,
            )

            total_detections += len(
                detections
            )

            # ------------------------------------------------
            # BYTE TRACK
            # ------------------------------------------------

            tracks = tracker.update(
                detections=detections,
                frame_index=frame_number,
                frame=frame,
            )

            # ------------------------------------------------
            # CURRENT FRAME TRACKS
            # ------------------------------------------------

            for track in tracks:

                track_id = track.local_id

                unique_ids.add(
                    track_id
                )

                id_frame_counts[
                    track_id
                ] += 1

                id_classes[
                    track_id
                ] = track.class_name

            # ------------------------------------------------
            # Advance frame
            # ------------------------------------------------

            frame_number += 1

            # ------------------------------------------------
            # Progress logging
            # ------------------------------------------------

            if frame_number % 100 == 0:

                logger.info(
                    "Processed %d/%d frames | "
                    "detections=%d | "
                    "active tracks=%d | "
                    "unique ByteTrack IDs=%d",
                    frame_number,
                    total_frames,
                    total_detections,
                    len(tracks),
                    len(unique_ids),
                )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        class_id_counts = defaultdict(
            int
        )

        for (
            track_id,
            class_name,
        ) in id_classes.items():

            class_id_counts[
                class_name
            ] += 1

        print()
        print("=" * 70)
        print("BYTE TRACK BASELINE")
        print("=" * 70)

        print(
            f"Frames: {frame_number}"
        )

        print(
            f"FPS: {fps:.2f}"
        )

        print(
            f"Resolution: "
            f"{video_width}x{video_height}"
        )

        print(
            f"Total detections: "
            f"{total_detections}"
        )

        print(
            f"Unique ByteTrack IDs: "
            f"{len(unique_ids)}"
        )

        print()

        # ====================================================
        # CONFIGURATION
        # ====================================================

        print(
            "ByteTrack configuration:"
        )

        print(
            f"  Activation threshold: "
            f"{track_activation_threshold:.2f}"
        )

        print(
            f"  Lost track buffer: "
            f"{lost_track_buffer} frames "
            f"({lost_track_buffer / fps:.2f}s)"
        )

        print(
            f"  Matching threshold: "
            f"{minimum_matching_threshold:.2f}"
        )

        print(
            f"  Minimum consecutive frames: "
            f"{minimum_consecutive_frames}"
        )

        print()

        # ====================================================
        # UNIQUE IDS BY CLASS
        # ====================================================

        print(
            "Unique IDs by class:"
        )

        for (
            class_name,
            count,
        ) in sorted(
            class_id_counts.items()
        ):

            print(
                f"  {class_name}: {count}"
            )

        print()

        # ====================================================
        # TRACKING QUALITY
        # ====================================================

        tracking_stats = (
            tracker.get_tracking_statistics()
        )

        print(
            "Tracking quality:"
        )

        print(
            f"  Average lifetime: "
            f"{tracking_stats['average_track_lifetime_frames']:.2f} frames "
            f"({tracking_stats['average_track_lifetime_seconds']:.2f}s)"
        )

        print(
            f"  Median lifetime: "
            f"{tracking_stats['median_track_lifetime_frames']:.0f} frames"
        )

        print(
            f"  < 1 second: "
            f"{tracking_stats['short_tracks_under_1_second']}"
        )

        print(
            f"  1-5 seconds: "
            f"{tracking_stats['medium_tracks_1_to_5_seconds']}"
        )

        print(
            f"  >= 5 seconds: "
            f"{tracking_stats['long_tracks_over_5_seconds']}"
        )

        print()

        # ====================================================
        # FINAL ACTIVE TRACKS
        # ====================================================

        print(
            "Final active tracks:"
        )

        final_tracks = (
            tracker.get_current_tracks()
        )

        final_by_class = defaultdict(
            int
        )

        for track in final_tracks:

            final_by_class[
                track.class_name
            ] += 1

        if final_by_class:

            for (
                class_name,
                count,
            ) in sorted(
                final_by_class.items()
            ):

                print(
                    f"  {class_name}: {count}"
                )

        else:

            print(
                "  None"
            )

        print()

        # ====================================================
        # LONGEST TRACKS
        # ====================================================

        print(
            "Top 20 longest tracks:"
        )

        longest_tracks = sorted(
            id_frame_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:20]

        if longest_tracks:

            for (
                track_id,
                lifetime,
            ) in longest_tracks:

                class_name = (
                    id_classes.get(
                        track_id,
                        "unknown",
                    )
                )

                history = (
                    tracker.get_track(
                        track_id
                    )
                )

                first_frame = (
                    history.first_frame
                    if history is not None
                    else None
                )

                last_frame = (
                    history.last_frame
                    if history is not None
                    else None
                )

                print(
                    f"  ID {track_id:3d} | "
                    f"{class_name:12s} | "
                    f"{lifetime:5d} frames | "
                    f"{first_frame} -> "
                    f"{last_frame}"
                )

        else:

            print(
                "  None"
            )

        print()

        # ====================================================
        # TRAJECTORY HISTORY
        # ====================================================

        history = (
            tracker.get_track_history()
        )

        total_observations = sum(
            len(track.observations)
            for track in history.values()
        )

        print(
            "Trajectory history:"
        )

        print(
            f"  Stored tracks: "
            f"{len(history)}"
        )

        print(
            f"  Stored observations: "
            f"{total_observations}"
        )

        print()

        # ====================================================
        # APPEARANCE STATISTICS
        # ====================================================

        print(
            "Appearance statistics:"
        )

        appearance_stats = (
            tracker.get_appearance_statistics()
        )

        print(
            f"  Model: "
            f"{appearance_stats['model']}"
        )

        print(
            f"  Device: "
            f"{appearance_stats['device']}"
        )

        print(
            f"  Appearance interval: "
            f"{appearance_stats['appearance_interval']} frames"
        )

        print(
            f"  Extraction attempts: "
            f"{appearance_stats['extraction_attempts']}"
        )

        print(
            f"  OSNet extractions performed: "
            f"{appearance_stats['osnet_extractions_performed']}"
        )

        print(
            f"  Embeddings extracted: "
            f"{appearance_stats['embeddings_extracted']}"
        )

        print(
            f"  Embeddings propagated: "
            f"{appearance_stats['embeddings_propagated']}"
        )

        print(
            f"  Extraction failures: "
            f"{appearance_stats['extraction_failures']}"
        )

        print(
            f"  Tracks without embedding: "
            f"{appearance_stats['tracks_without_embedding']}"
        )

        print(
            f"  Non-person observations skipped: "
            f"{appearance_stats['skipped_non_person']}"
        )

        print(
            f"  Extraction success rate: "
            f"{appearance_stats['success_rate']:.2%}"
        )

        print()

        # ====================================================
        # APPEARANCE CACHE
        # ====================================================

        cached_embeddings = len(
            tracker.last_appearance_embeddings
        )

        print(
            "Appearance cache:"
        )

        print(
            f"  Tracks with cached embeddings: "
            f"{cached_embeddings}"
        )

        print()

        print("=" * 70)
        print("BYTE TRACK BASELINE COMPLETE")
        print("=" * 70)

        logger.info(
            "Tracking completed successfully: "
            "frames=%d, detections=%d, "
            "unique_ids=%d, stored_tracks=%d",
            frame_number,
            total_detections,
            len(unique_ids),
            len(history),
        )

        return tracker

    finally:

        cap.release()

