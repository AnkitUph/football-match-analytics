"""
Stage 2.5: Shot Boundary Detection & Classification.

Not in your original 7-stage roadmap — added because broadcast football
footage constantly cuts between wide shots, close-ups, and replays, and
BoT-SORT (Stage 2) has no way to know a cut happened. This stage's only
job: tell you WHEN cuts happen and WHICH segments are usable for pitch
mapping. It does not fix identity continuity itself — that's Stage 5,
using Stage 3's Re-ID embeddings.

Second-pass feature: leave `enable_shot_detection=False` in config until
Stage 1+2+3(color)+4+5(single-angle)+6+7 all work on a single continuous
angle end-to-end.
"""

import cv2
import numpy as np

from ai_engine.config import ShotDetectionConfig
from ai_engine.utils.types import ShotSegment, ShotType


def detect_shot_boundaries(
    video_path: str, config: ShotDetectionConfig
) -> list[tuple[int, int]]:
    """
    Returns a list of (start_frame, end_frame) tuples for each detected
    shot segment, in SOURCE video frame indices.

    TODO: implement with PySceneDetect, e.g.:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=config.content_threshold,
                min_scene_len=config.min_scene_len_frames,
            )
        )
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        return [
            (start.get_frames(), end.get_frames())
            for start, end in scene_list
        ]
    """
    raise NotImplementedError("detect_shot_boundaries is stubbed.")


def classify_shot(frame: np.ndarray, config: ShotDetectionConfig) -> ShotType:
    """
    Heuristic classifier for a representative frame of a shot segment.
    High green-pixel coverage + visible line structure -> MAIN_WIDE
    (usable for homography). Otherwise CLOSE_UP / REPLAY / OTHER.

    TODO: implement, roughly:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        green_ratio = green_mask.sum() / (255 * green_mask.size)

        # Line detection as a secondary signal (pitch markings).
        edges = cv2.Canny(frame, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                 minLineLength=100, maxLineGap=10)
        line_pixel_count = 0 if lines is None else len(lines)

        if (green_ratio >= config.min_green_ratio
                and line_pixel_count >= config.min_line_pixels):
            return ShotType.MAIN_WIDE

        # Distinguish close-up (high green, low structure) from replay
        # (often has on-screen graphics/letterboxing) — needs tuning on
        # your actual footage; start simple and iterate.
        return ShotType.CLOSE_UP if green_ratio >= config.min_green_ratio else ShotType.OTHER
    """
    raise NotImplementedError("classify_shot is stubbed.")


def build_shot_segments(
    video_path: str, config: ShotDetectionConfig
) -> list[ShotSegment]:
    """
    Full Stage 2.5 entry point: detect boundaries, sample a representative
    frame from each, classify it, and return ShotSegment objects ready to
    drive Stage 2's reset() calls and Stage 5's homography gating.
    """
    boundaries = detect_shot_boundaries(video_path, config)

    cap = cv2.VideoCapture(video_path)
    segments: list[ShotSegment] = []
    for shot_id, (start, end) in enumerate(boundaries):
        mid_frame_idx = (start + end) // 2
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
    cap.release()
    return segments
