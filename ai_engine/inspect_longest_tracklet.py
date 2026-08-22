"""
Diagnostic: inspects the longest tracklet from a full-clip run to check
whether it's a real continuously-tracked player, or bogus data bridged
across a camera cut BoT-SORT didn't detect.

Usage:
    docker compose exec web python -m ai_engine.inspect_longest_tracklet \
        media/test_clips/test_11.mp4
"""

import argparse

from ai_engine.config import DEFAULT_CONFIG
from ai_engine.stage2_tracking.tracker import Tracker


def main(args):
    tracker = Tracker(DEFAULT_CONFIG.detection, DEFAULT_CONFIG.tracking)
    tracklets = tracker.track_video(args.input_video)

    longest = max(tracklets.values(), key=lambda t: t.duration_frames)
    print(f"Longest tracklet: track_id={longest.track_id}, duration={longest.duration_frames} frames")
    print()

    prev_center = None
    prev_frame = None
    max_jump = 0
    max_jump_frame = None

    for det in longest.detections:
        cx, cy = det.center
        if prev_center is not None:
            dist = ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5
            frame_gap = det.frame_idx - prev_frame
            # normalize by frame gap so a legit fast-but-continuous jump
            # over a big gap isn't confused with a real teleport
            per_frame_dist = dist / max(frame_gap, 1)
            if per_frame_dist > max_jump:
                max_jump = per_frame_dist
                max_jump_frame = det.frame_idx
        prev_center = (cx, cy)
        prev_frame = det.frame_idx

    print(f"Largest per-frame position jump: {max_jump:.1f} px/frame, at frame {max_jump_frame}")
    print()
    print("Positions around the largest jump:")
    for det in longest.detections:
        if max_jump_frame - 6 <= det.frame_idx <= max_jump_frame + 6:
            print(f"  frame {det.frame_idx}: center=({det.center[0]:.0f},{det.center[1]:.0f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    args = parser.parse_args()
    main(args)
