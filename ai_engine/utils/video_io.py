"""
Video reading helpers: downsampling to target FPS, frame iteration.

Shared by Stage 1 (detection) and Stage 2.5 (shot detection) so both stages
walk the exact same frame sequence — mismatched frame indices between
stages is a classic source of silent bugs in this kind of pipeline.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


def get_video_info(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoInfo(
        path=Path(path),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=frame_count / fps if fps else 0.0,
    )


def iter_frames_downsampled(
    path: str | Path,
    target_fps: int = 12,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yields (original_frame_idx, frame) at approximately target_fps, by
    dropping frames from the source rather than re-encoding.

    original_frame_idx is the index in the SOURCE video, not a compacted
    0..N counter — keep using this everywhere downstream (Stage 2.5's shot
    boundaries, Stage 6's event timestamps) so you can always map back to
    "what second of the original clip was this".
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(1, round(source_fps / target_fps))

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()
