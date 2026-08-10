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
    Tracking statistics

This phase intentionally does NOT perform:
    - global identity merging
    - appearance embeddings
    - team assignment
    - jersey-number recognition
    - homography
    - heatmap generation

Those will be added as separate pipeline stages.
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
):
    """
    Run YOLO detection + ByteTrack over a video.

    Returns:
        ByteTrackTracker instance.

    The returned tracker contains:

        tracker.current_tracks
            Tracks active in the latest frame.

        tracker.track_history
            Complete trajectory history of every ByteTrack ID.
    """

    # ============================================================
    # OPEN VIDEO
    # ============================================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_path}"
        )

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = video_fps

    if fps is None or fps <= 0:
        fps = 25.0

    fps = float(fps)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    video_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    video_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
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

    # ============================================================
    # DETECTOR
    # ============================================================

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

    # ============================================================
    # BYTE TRACK
    # ============================================================

    tracker = ByteTrackTracker(
        fps=fps,
        appearance_model_name="osnet_x1_0",
        appearance_device="cpu",
        appearance_interval=10,
        appearance_refresh_on_new_track=True,
    )

    logger.info(
        "ByteTrack initialized"
    )

    # ============================================================
    # STATISTICS
    # ============================================================

    frame_number = 0

    total_detections = 0

    # Every unique ByteTrack ID.
    unique_ids = set()

    # Track ID -> number of frames observed.
    id_frame_counts: Dict[int, int] = defaultdict(int)

    # Track ID -> class name.
    id_classes: Dict[int, str] = {}

    # Number of active tracks per class per frame.
    active_per_frame = defaultdict(list)

    # ============================================================
    # FRAME LOOP
    # ============================================================

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        # --------------------------------------------------------
        # YOLO DETECTION
        # --------------------------------------------------------

        detections = detector.detect_frame(
            frame,
            frame_number,
        )

        total_detections += len(detections)

        # --------------------------------------------------------
        # BYTE TRACK
        # --------------------------------------------------------

        tracks = tracker.update(
            detections,
            frame_number,
            frame,
        )

        # --------------------------------------------------------
        # CURRENT FRAME TRACKS
        #
        # IMPORTANT:
        # `tracks` is already the current-frame list.
        #
        # We don't use:
        #
        #     track.track_id
        #
        # because our Track schema uses:
        #
        #     track.local_id
        # --------------------------------------------------------

        for track in tracks:

            track_id = track.local_id

            unique_ids.add(track_id)

            id_frame_counts[track_id] += 1

            id_classes[track_id] = (
                track.class_name
            )

            active_per_frame[
                track.class_name
            ].append(track_id)

        # --------------------------------------------------------
        # PROGRESS LOGGING
        # --------------------------------------------------------

        frame_number += 1

        if frame_number % 100 == 0:

            active_tracks = len(tracks)

            logger.info(
                "Processed %d/%d frames | "
                "detections=%d | "
                "active tracks=%d | "
                "unique ByteTrack IDs=%d",
                frame_number,
                total_frames,
                total_detections,
                active_tracks,
                len(unique_ids),
            )

    cap.release()

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    class_id_counts = defaultdict(int)

    for track_id, class_name in id_classes.items():
        class_id_counts[class_name] += 1

    # ============================================================
    # PRINT SUMMARY
    # ============================================================

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
        f"Resolution: {video_width}x{video_height}"
    )

    print(
        f"Total detections: {total_detections}"
    )

    print(
        f"Unique ByteTrack IDs: {len(unique_ids)}"
    )

    print()

    # ============================================================
    # UNIQUE IDS BY CLASS
    # ============================================================

    print(
        "Unique IDs by class:"
    )

    for class_name, count in sorted(
        class_id_counts.items()
    ):
        print(
            f"  {class_name}: {count}"
        )

    print()

    # ============================================================
    # TRACK LIFETIME STATISTICS
    # ============================================================

    print(
        "Track lifetime statistics:"
    )

    if id_frame_counts:

        lifetimes = list(
            id_frame_counts.values()
        )

        print(
            f"  Shortest: {min(lifetimes)} frames"
        )

        print(
            f"  Longest: {max(lifetimes)} frames"
        )

        print(
            f"  Average: "
            f"{sum(lifetimes) / len(lifetimes):.2f} frames"
        )

        # --------------------------------------------------------
        # Median
        # --------------------------------------------------------

        sorted_lifetimes = sorted(
            lifetimes
        )

        middle = len(
            sorted_lifetimes
        ) // 2

        if len(sorted_lifetimes) % 2 == 0:

            median = (
                sorted_lifetimes[middle - 1]
                + sorted_lifetimes[middle]
            ) / 2

        else:
            median = sorted_lifetimes[middle]

        print(
            f"  Median: {median:.0f} frames"
        )

    else:

        print(
            "  No tracks created."
        )

    print()

    # ============================================================
    # ACTIVE TRACK STATISTICS
    # ============================================================

    print(
        "Final active tracks:"
    )

    final_tracks = tracker.get_current_tracks()

    final_by_class = defaultdict(int)

    for track in final_tracks:

        final_by_class[
            track.class_name
        ] += 1

    for class_name, count in sorted(
        final_by_class.items()
    ):
        print(
            f"  {class_name}: {count}"
        )

    print()

    # ============================================================
    # LONGEST TRACKS
    # ============================================================

    print(
        "Top 20 longest tracks:"
    )

    longest_tracks = sorted(
        id_frame_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]

    for track_id, lifetime in longest_tracks:

        class_name = id_classes.get(
            track_id,
            "unknown",
        )

        history = tracker.get_track(
            track_id
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
            f"{first_frame} -> {last_frame}"
        )

    print()

    # ============================================================
    # TRAJECTORY HISTORY SUMMARY
    # ============================================================

    print(
        "Trajectory history:"
    )

    history = tracker.get_track_history()

    print(
        f"  Stored tracks: {len(history)}"
    )

    total_observations = sum(
        len(track.observations)
        for track in history.values()
    )

    print(
        f"  Stored observations: "
        f"{total_observations}"
    )

    print()

    # ============================================================
    # TRACK QUALITY BREAKDOWN
    # ============================================================

    print(
        "Track quality:"
    )

    if id_frame_counts:

        # 1 second = approximately FPS frames
        one_second = max(
            1,
            int(round(fps)),
        )

        short_tracks = sum(
            1
            for lifetime in id_frame_counts.values()
            if lifetime < one_second
        )

        medium_tracks = sum(
            1
            for lifetime in id_frame_counts.values()
            if one_second <= lifetime < 5 * one_second
        )

        long_tracks = sum(
            1
            for lifetime in id_frame_counts.values()
            if lifetime >= 5 * one_second
        )

        print(
            f"  < 1 second: "
            f"{short_tracks}"
        )

        print(
            f"  1-5 seconds: "
            f"{medium_tracks}"
        )

        print(
            f"  >= 5 seconds: "
            f"{long_tracks}"
        )

    print()

    print(
        "=" * 70
    )

    print(
        "BYTE TRACK BASELINE COMPLETE"
    )

    print(
        "=" * 70
    )

    # ============================================================
    # RETURN TRACKER
    # ============================================================

    return tracker