"""Shot boundary detection and conservative pitch-view classification.

The classifier marks a segment MAIN_WIDE only when both turf coverage and
pitch-line structure are visible. It does not guess whether other views are
replays from a single frame.
"""

import cv2
import numpy as np

from ai_engine.config import ShotDetectionConfig
from ai_engine.utils.types import ShotSegment, ShotType


def detect_shot_boundaries(
    video_path: str, config: ShotDetectionConfig
) -> list[tuple[int, int]]:
    """Return half-open ``(start_frame, end_frame)`` scene ranges.

    PySceneDetect content cuts are expanded to cover the complete video. If
    no cut is found, ``start_in_scene=True`` returns one segment for the clip.
    """
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path))
    try:
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=config.content_threshold,
                min_scene_len=config.min_scene_len_frames,
            )
        )
        scene_manager.detect_scenes(video=video)
        boundaries = [
            (int(start.get_frames()), int(end.get_frames()))
            for start, end in scene_manager.get_scene_list(start_in_scene=True)
        ]
    finally:
        video.close()

    if not boundaries:
        raise IOError(f"No frames could be read while detecting shots in {video_path}")
    return boundaries


def classify_shot(frame: np.ndarray, config: ShotDetectionConfig) -> ShotType:
    """Classify a representative BGR frame using grass and line evidence.

    Frames are reduced to at most 640 pixels wide so line thresholds behave
    consistently across source resolutions. CLOSE_UP means some turf is
    visible but there is not enough field-line structure to trust homography.
    OTHER covers non-pitch views. REPLAY is intentionally not inferred here.
    """
    if frame is None or frame.ndim != 3 or frame.shape[0] < 16 or frame.shape[1] < 16:
        return ShotType.OTHER

    height, width = frame.shape[:2]
    if width > 640:
        scale = 640.0 / width
        frame = cv2.resize(
            frame,
            (640, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(
        hsv,
        np.array((25, 35, 30), dtype=np.uint8),
        np.array((100, 255, 255), dtype=np.uint8),
    )
    green_ratio = cv2.countNonZero(green_mask) / float(green_mask.size)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    field_region = cv2.dilate(
        green_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    field_edges = cv2.bitwise_and(edges, edges, mask=field_region)
    lines = cv2.HoughLinesP(
        field_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=20,
        maxLineGap=8,
    )
    line_pixel_count = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            line_pixel_count += int(round(np.hypot(x2 - x1, y2 - y1)))

    if (
        green_ratio >= config.min_green_ratio
        and line_pixel_count >= config.min_line_pixels
    ):
        return ShotType.MAIN_WIDE
    if green_ratio >= config.min_green_ratio * 0.5:
        return ShotType.CLOSE_UP
    return ShotType.OTHER


def build_shot_segments(
    video_path: str, config: ShotDetectionConfig
) -> list[ShotSegment]:
    """Detect cuts, classify each segment's middle frame, and return ranges."""
    boundaries = detect_shot_boundaries(video_path, config)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video to classify shots: {video_path}")

    segments: list[ShotSegment] = []
    try:
        for shot_id, (start, end) in enumerate(boundaries):
            if end <= start:
                continue
            mid_frame_idx = start + (end - start - 1) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
            ok, frame = cap.read()
            shot_type = classify_shot(frame, config) if ok else ShotType.OTHER
            segments.append(
                ShotSegment(
                    shot_id=shot_id,
                    start_frame=start,
                    end_frame=end,
                    shot_type=shot_type,
                )
            )
    finally:
        cap.release()
    return segments
