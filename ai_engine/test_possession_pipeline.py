"""
Integration test: Stage 2 tracking -> Stage 3 team classification ->
Stage 4 ball interpolation -> Stage 5 homography (pixel->pitch) ->
Stage 6 possession detection, all wired together end to end.

Usage (inside your Docker container):
    docker compose exec web python -m ai_engine.test_possession_pipeline \
        media/test_clips/test_11.mp4 --start 540 --end 650

NOTE ON CALIBRATION POINTS: the 4 bootstrap points below are the ones we
validated by hand against frame 540 of test_11.mp4 specifically (holdout
penalty-spot test: 35cm/15cm error). If you run this against a DIFFERENT
frame or a different clip, these pixel coordinates will be wrong for that
footage — you'd need to re-calibrate using the click tool against a
similarly close penalty-box view in the new footage. This script doesn't
(yet) automate finding a good calibration frame; that's still a manual
step per shot/clip until Stage 2.5 + auto re-calibration exists.
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
from ai_engine.stage6_event_detection.events import detect_possession
from ai_engine.utils.types import MasterIdentity, PitchPoint, BallTrajectoryPoint

# Validated calibration for test_11.mp4, frame 540 specifically — see
# module docstring.
BOOTSTRAP_FRAME = 540
IMAGE_PTS = [(1022, 348), (1467, 561), (1309, 329), (1864, 520)]
PITCH_PTS = [(36, 20.16), (36, -20.16), (52.5, 20.16), (52.5, -20.16)]


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

    print("Stage 6: possession detection...")
    events = detect_possession(ball_pitch_trajectory, identities, DEFAULT_CONFIG.event_detection)
    print(f"  {len(events)} possession events detected")
    for e in events:
        print(f"    {e.event_type} at frame {e.frame_idx}: player {e.player_master_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("--end", type=int, default=650)
    args = parser.parse_args()
    main(args)
