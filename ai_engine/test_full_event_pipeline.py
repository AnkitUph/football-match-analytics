"""
Full integration test: Stage 2 tracking -> Stage 3 team classification ->
Stage 4 ball interpolation -> Stage 5 homography -> Stage 6 possession,
passes, and shots, all wired together end to end.

Usage (inside your Docker container):
    docker compose exec web python -m ai_engine.test_full_event_pipeline \
        media/test_clips/test_11.mp4 --end 650

NOTE ON CALIBRATION POINTS: same caveat as test_possession_pipeline.py —
the 4 bootstrap points are validated specifically against frame 540 of
test_11.mp4. Re-calibrate via the click tool if using different footage.
"""

import argparse
from collections import defaultdict

import cv2

from ai_engine.config import DEFAULT_CONFIG
from ai_engine.stage2_tracking.tracker import Tracker
from ai_engine.stage3_team_reid.team_classifier import (
    classify_team,
    fit_team_color_clusters,
    sample_torso_color,
)
from ai_engine.stage4_ball_tracking.ball_tracker import extract_ball_detections, interpolate_gaps
from ai_engine.stage5_pitch_mapping.homography_tracker import HomographyTracker
from ai_engine.stage5_pitch_mapping.homography import image_point_to_pitch
from ai_engine.stage6_event_detection.events import detect_possession, detect_passes, detect_shots
from ai_engine.utils.types import MasterIdentity, PitchPoint, BallTrajectoryPoint

BOOTSTRAP_FRAME = 540
IMAGE_PTS = [(1022, 348), (1467, 561), (1309, 329), (1864, 520)]
PITCH_PTS = [(36, 20.16), (36, -20.16), (52.5, 20.16), (52.5, -20.16)]
GOAL_CENTER_PITCH = (52.5, 0.0)


def main(args):
    print("Stage 1+2: detection + tracking...")
    tracker = Tracker(DEFAULT_CONFIG.detection, DEFAULT_CONFIG.tracking)
    tracklets = tracker.track_video(args.input_video)
    print(f"  {len(tracklets)} raw tracklets")

    min_duration_frames = int(DEFAULT_CONFIG.pitch_mapping.min_tracklet_duration_sec * 25)
    survivors = {tid: t for tid, t in tracklets.items() if t.duration_frames >= min_duration_frames}
    print(f"  {len(survivors)} survive duration filter")

    print("Stage 3: team classification...")
    cap = cv2.VideoCapture(args.input_video)
    colors, valid = {}, []
    for t in survivors.values():
        det = t.detections[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, det.frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = map(int, det.bbox)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        color = sample_torso_color(crop)
        if color is not None:
            colors[t.track_id] = color
            valid.append(t)
    clusters = fit_team_color_clusters(list(colors.values()), DEFAULT_CONFIG.team_reid)
    for t in valid:
        t.team = classify_team(t.cls.value if t.cls else "player", colors[t.track_id], clusters)

    print("Stage 4: ball extraction + interpolation...")
    all_detections_by_frame = defaultdict(list)
    for t in tracklets.values():
        for det in t.detections:
            all_detections_by_frame[det.frame_idx].append(det)
    ball_by_frame = extract_ball_detections(dict(all_detections_by_frame))
    ball_trajectory_px = interpolate_gaps(ball_by_frame, DEFAULT_CONFIG.ball_tracking)

    print("Stage 5: homography bootstrap + propagation...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, BOOTSTRAP_FRAME)
    ok, bootstrap_frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read bootstrap frame {BOOTSTRAP_FRAME}")

    homography_tracker = HomographyTracker(DEFAULT_CONFIG.pitch_mapping)
    if not homography_tracker.bootstrap(bootstrap_frame, IMAGE_PTS, PITCH_PTS):
        raise RuntimeError("Homography bootstrap failed")

    homography_by_frame = {BOOTSTRAP_FRAME: homography_tracker.current_H.copy()}
    cap.set(cv2.CAP_PROP_POS_FRAMES, BOOTSTRAP_FRAME + 1)
    for frame_idx in range(BOOTSTRAP_FRAME + 1, args.end):
        ok, frame = cap.read()
        if not ok:
            break
        H = homography_tracker.update(frame)
        if H is not None:
            homography_by_frame[frame_idx] = H.copy()
    print(f"  {len(homography_by_frame)} frames with valid homography")

    print("Converting ball + player positions to pitch space...")
    ball_pitch_trajectory = []
    for point in ball_trajectory_px:
        if point.x_m is None or point.frame_idx not in homography_by_frame:
            continue
        pitch_pt = image_point_to_pitch(point.x_m, point.y_m, homography_by_frame[point.frame_idx])
        ball_pitch_trajectory.append(
            BallTrajectoryPoint(frame_idx=point.frame_idx, x_m=pitch_pt.x_m, y_m=pitch_pt.y_m, interpolated=point.interpolated)
        )

    identities = []
    for t in valid:
        traj = {}
        for det in t.detections:
            if det.frame_idx not in homography_by_frame:
                continue
            foot_x, foot_y = (det.x1 + det.x2) / 2, det.y2
            pt = image_point_to_pitch(foot_x, foot_y, homography_by_frame[det.frame_idx])
            traj[det.frame_idx] = PitchPoint(x_m=pt.x_m, y_m=pt.y_m)
        if traj:
            identities.append(MasterIdentity(master_id=t.track_id, team=t.team, reid_embedding=[], trajectory=traj))
    print(f"  {len(identities)} identities with pitch trajectories")

    print()
    print("Stage 6: event detection...")
    possession_events = detect_possession(ball_pitch_trajectory, identities, DEFAULT_CONFIG.event_detection)
    print(f"  possession: {len(possession_events)} events")
    for e in possession_events:
        print(f"    frame {e.frame_idx}: player {e.player_master_id}")

    pass_events = detect_passes(possession_events, identities)
    print(f"  passes: {len(pass_events)} events")
    for e in pass_events:
        print(f"    frame {e.frame_idx}: {e.player_master_id} -> {e.target_master_id}")

    shot_events = detect_shots(ball_pitch_trajectory, GOAL_CENTER_PITCH)
    print(f"  shots: {len(shot_events)} events")
    for e in shot_events:
        m = e.metadata
        print(f"    frame {e.frame_idx}: speed={m['speed_mps']:.1f}m/s align={m['alignment']:.2f}")

    print()
    print("PIPELINE COMPLETE — no errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("--end", type=int, default=650)
    args = parser.parse_args()
    main(args)
