"""
Stage 5 Pre-processing: Broadcast Camera View & Scene Cut Classifier.

Distinguishes between:
1. MAIN_TACTICAL: Elevated wide sideline tactical broadcast view.
2. BEHIND_GOAL: Endline or behind-the-goal perspective (requires secondary homography).
3. CLOSEUP_OR_REPLAY: Dugout, bench, player/referee closeups, crowd, or replay graphics.

Prevents homography drift, tracking explosion, and invalid coordinate generation
during non-tactical cuts.
"""

from dataclasses import dataclass
from enum import Enum
import cv2
import numpy as np


class CameraViewType(str, Enum):
    MAIN_TACTICAL = "main_tactical"
    BEHIND_GOAL = "behind_goal"
    CLOSEUP_OR_REPLAY = "closeup_or_replay"


@dataclass
class ViewClassification:
    frame_idx: int
    view_type: CameraViewType
    is_cut: bool
    green_ratio: float
    top_green_ratio: float
    scene_diff: float


class ViewClassifier:
    """
    Stateful real-time frame-by-frame camera view and scene cut classifier.
    """
    def __init__(
        self,
        cut_diff_threshold: float = 32.0,
        min_grass_ratio: float = 0.50,
        behind_goal_top_grass: float = 0.85,
    ):
        self.cut_diff_threshold = cut_diff_threshold
        self.min_grass_ratio = min_grass_ratio
        self.behind_goal_top_grass = behind_goal_top_grass
        self._prev_gray: np.ndarray | None = None
        self._prev_view: CameraViewType = CameraViewType.MAIN_TACTICAL

    def classify_frame(self, frame: np.ndarray, frame_idx: int) -> ViewClassification:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Instantaneous Scene Cut Detection
        scene_diff = 0.0
        is_cut = False
        if self._prev_gray is not None:
            scene_diff = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
            if scene_diff > self.cut_diff_threshold:
                is_cut = True
        self._prev_gray = gray

        # 2. Green Grass Pixel Analysis (HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv, (35, 38, 38), (85, 255, 255))
        total_pixels = h * w
        green_ratio = float(np.count_nonzero(grass_mask) / total_pixels)

        # Top 25% of the frame (crowd/stands vs full pitch coverage)
        top_h = max(1, int(h * 0.25))
        top_green_ratio = float(np.count_nonzero(grass_mask[:top_h, :]) / (top_h * w))

        # 3. View Type Classification
        if green_ratio < self.min_grass_ratio:
            view_type = CameraViewType.CLOSEUP_OR_REPLAY
        else:
            # High grass ratio overall
            # Behind-the-goal cameras typically look straight down the pitch,
            # showing grass across the top quarter (> 85%), whereas sideline
            # cameras have the stadium stands/boards in the top 25%.
            if top_green_ratio > self.behind_goal_top_grass:
                view_type = CameraViewType.BEHIND_GOAL
            else:
                view_type = CameraViewType.MAIN_TACTICAL

        self._prev_view = view_type
        return ViewClassification(
            frame_idx=frame_idx,
            view_type=view_type,
            is_cut=is_cut,
            green_ratio=round(green_ratio, 3),
            top_green_ratio=round(top_green_ratio, 3),
            scene_diff=round(scene_diff, 2),
        )


def classify_video_views(
    video_path: str,
    stride: int = 1,
    max_frames: int | None = None,
) -> dict[int, ViewClassification]:
    """
    Scans a video and returns a dictionary of frame_idx -> ViewClassification.
    """
    cap = cv2.VideoCapture(video_path)
    classifier = ViewClassifier()
    results: dict[int, ViewClassification] = {}

    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        if stride == 1 or frame_idx % stride == 0:
            classification = classifier.classify_frame(frame, frame_idx)
            results[frame_idx] = classification

        frame_idx += 1

    cap.release()
    return results


def get_non_tactical_frame_ranges(video_path: str, stride: int = 10) -> list[tuple[int, int]]:
    """
    Fast single-pass scanner to find all non-tactical broadcast intervals
    (replays, close-ups, dugout/crowd cuts, graphic wipes).
    """
    cap = cv2.VideoCapture(video_path)
    classifier = ViewClassifier()
    non_tactical_ranges: list[tuple[int, int]] = []
    f_idx = 0
    in_non_tactical = False
    start_f = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx % stride == 0:
            res = classifier.classify_frame(frame, f_idx)
            # Tactical sideline view requires sufficient pitch grass and stadium stands in top quarter
            is_tactical = (res.green_ratio >= 0.45 and res.top_green_ratio <= 0.35)
            if not is_tactical:
                if not in_non_tactical:
                    in_non_tactical = True
                    start_f = f_idx
            else:
                if in_non_tactical:
                    in_non_tactical = False
                    non_tactical_ranges.append((start_f, f_idx))
        f_idx += 1

    cap.release()
    if in_non_tactical:
        non_tactical_ranges.append((start_f, f_idx))
    return non_tactical_ranges

