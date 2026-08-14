"""
Pipeline orchestrator — runs Stages 1-6 end to end on a single clip.

Usage (once stages are implemented, not stubs):
    python -m ai_engine.main /path/to/clip.mp4

Build/test order (see ai_engine/__init__.py docstring):
    Get run_pipeline() working with config.enable_shot_detection=False and
    config.enable_reid=False first — that's the "single continuous angle"
    happy path (Stages 1, 2, 3-color-only, 4, 5-single-shot, 6). Only flip
    those flags on once that path is solid on a real test clip.
"""

import sys
from dataclasses import dataclass, field

from ai_engine.config import PipelineConfig, DEFAULT_CONFIG
from ai_engine.stage1_detection.detector import Detector
from ai_engine.stage2_tracking.tracker import Tracker
from ai_engine.stage5_pitch_mapping.identity_association import IdentityGallery
from ai_engine.utils.types import Event, MasterIdentity, ShotSegment


@dataclass
class PipelineResult:
    identities: list[MasterIdentity] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    shot_segments: list[ShotSegment] = field(default_factory=list)


def run_pipeline(video_path: str, config: PipelineConfig = DEFAULT_CONFIG) -> PipelineResult:
    """
    Orchestrates Stages 1-6 on one clip. This is intentionally a straight-
    line sketch, not a streaming/memory-optimized implementation — get the
    logic right on a short test clip first, optimize for full 15-min clips
    once each stage is validated individually.
    """

    # --- Stage 2.5: shot boundaries (skip entirely for single-angle testing) ---
    if config.enable_shot_detection:
        from ai_engine.stage2_5_shot_detection.shot_detector import build_shot_segments

        shot_segments = build_shot_segments(video_path, config.shot_detection)
    else:
        # Treat the whole clip as one shot — the single-angle happy path.
        from ai_engine.utils.video_io import get_video_info
        from ai_engine.utils.types import ShotType

        info = get_video_info(video_path)
        shot_segments = [
            ShotSegment(
                shot_id=0,
                start_frame=0,
                end_frame=info.frame_count,
                shot_type=ShotType.MAIN_WIDE,
            )
        ]

    # --- Stage 1 + 2: detect and track within each shot ---
    detector = Detector(config.detection)
    tracker = Tracker(config.tracking)
    gallery = IdentityGallery(config.pitch_mapping)

    all_tracklets_by_shot: dict[int, list] = {}

    # TODO: this should stream frames once (via utils.video_io) and hand
    # each frame to both the detector and, when a shot boundary is
    # crossed, call tracker.reset(). Sketched as a loop-per-shot here for
    # clarity; revisit for a real single-pass streaming implementation
    # before running on full 15-min clips (re-decoding the video per shot
    # would be wasteful).
    for shot in shot_segments:
        tracker.reset()
        # TODO: detect_video / tracker.update calls scoped to this shot's
        # frame range, then:
        # tracklets = tracker.finalize_shot()
        # all_tracklets_by_shot[shot.shot_id] = tracklets
        pass

    # --- Stage 3: team + Re-ID (fill in once tracklets exist) ---
    # TODO: for each tracklet, classify_team(...) and, if
    # config.enable_reid, ReidEmbedder().embed(...) per crop.

    # --- Stage 4: ball tracking (parallel to player tracking above) ---
    # TODO: extract_ball_detections + interpolate_gaps.

    # --- Stage 5: pitch mapping + identity association ---
    # TODO: for MAIN_WIDE shots, compute_homography + map tracklet
    # positions to pitch coords. For every shot after the first, call
    # match_tracklets_across_cut(..., gallery, ...) to assign master IDs.

    # --- Stage 6: event detection ---
    # TODO: detect_possession / detect_passes / detect_shots using
    # gallery.identities' trajectories and the ball trajectory.

    return PipelineResult(
        identities=gallery.identities,
        events=[],
        shot_segments=shot_segments,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m ai_engine.main /path/to/clip.mp4")
        sys.exit(1)

    result = run_pipeline(sys.argv[1])
    print(f"Identities: {len(result.identities)}")
    print(f"Events: {len(result.events)}")
    print(f"Shot segments: {len(result.shot_segments)}")
