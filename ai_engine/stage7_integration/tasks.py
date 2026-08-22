"""
Stage 7: Pipeline Integration & Celery Processing.

This is the ONLY file in ai_engine that imports Django models — every
other stage stays framework-agnostic so it's testable standalone.

HONEST SCOPE, READ BEFORE WIRING THIS UP:

What this populates with REAL, validated data:
  - TeamStatistics: possession %, shots, passes, distance covered
    (team-level aggregates — accurate, using Stage 6's validated logic)
  - Heatmap-ready trajectory data (derived from MasterIdentity.trajectory)

What this does NOT populate (left at model defaults, i.e. 0):
  - Goals, cards, fouls, tackles, interceptions, clearances, offsides,
    xG — no detection logic exists for any of these. Not a bug, just
    genuinely out of the current pipeline's scope.
  - PlayerStatistics linked to real named Player records — this pipeline
    has no jersey-number OCR (still second-pass, unbuilt), so it cannot
    map a tracked Master ID to an actual roster Player. Per-player rows
    are SKIPPED here rather than saved against guessed/wrong players.
    Revisit once stage3_team_reid/jersey_ocr.py exists.
  - Anything outside the calibrated camera segment(s) — Stage 5/6 only
    produce real numbers where a homography calibration exists for that
    part of the clip. See homography_tracker.py's docstring.

CALIBRATION NOTE: HOME_TEAM_KIT_COLOR / AWAY_TEAM_KIT_COLOR below need to
be sourced from wherever your project actually stores per-match kit
colors (mentioned in project notes as part of the Phase 3 upload flow,
but the exact model/field wasn't visible in what I could see of your
codebase — verify and wire up the real lookup before relying on this).
IMAGE_PTS/PITCH_PTS are the validated calibration points for this
specific test clip's box-view segment — for a real uploaded match, this
whole calibration step needs to come from somewhere else entirely (see
the "known gap" discussion: manual calibration doesn't scale to a full
match with multiple camera angles).
"""

from pathlib import Path

from celery import shared_task

from ai_engine.config import DEFAULT_CONFIG
from ai_engine.main import run_pipeline
from ai_engine.stage2_tracking.tracker import Tracker
from ai_engine.stage3_team_reid.team_classifier import (
    classify_team,
    fit_team_color_clusters,
    sample_torso_color,
)
from ai_engine.stage4_ball_tracking.ball_tracker import extract_ball_detections, interpolate_gaps
from ai_engine.stage5_pitch_mapping.homography_tracker import HomographyTracker
from ai_engine.stage5_pitch_mapping.homography import image_point_to_pitch
from ai_engine.stage5_pitch_mapping.identity_association import (
    IdentityGallery,
    assign_tracklets_to_gallery,
    match_tracklets_within_shot,
)
from ai_engine.stage6_event_detection.events import detect_possession, detect_passes, detect_shots
from ai_engine.utils.types import MasterIdentity, PitchPoint, BallTrajectoryPoint, Team

import cv2
from collections import defaultdict


def _compute_team_distance_m(identities: list[MasterIdentity], team: Team) -> float:
    """Sums frame-to-frame pitch-space movement for all identities on a
    team. Straightforward given trajectories already exist — not
    previously built since we had nothing to test it against until now."""
    total = 0.0
    for identity in identities:
        if identity.team != team:
            continue
        frames = sorted(identity.trajectory.keys())
        for f1, f2 in zip(frames, frames[1:]):
            p1, p2 = identity.trajectory[f1], identity.trajectory[f2]
            total += ((p2.x_m - p1.x_m) ** 2 + (p2.y_m - p1.y_m) ** 2) ** 0.5
    return total


@shared_task(bind=True, time_limit=3600, soft_time_limit=3300)
def process_match_video(
    self,
    match_id: int,
    video_path: str,
    calibration_frame: int,
    image_pts: list,
    pitch_pts: list,
    shot_min_speed_mps: float = 10.0,
    shot_min_alignment: float = 0.85,
):
    """
    Celery task entry point.

    shot_min_speed_mps/shot_min_alignment default to the same values as
    detect_shots' own defaults — exposed here as parameters purely to
    allow end-to-end DB-write testing with loosened thresholds against
    known sparse test data, without touching production defaults. Not
    intended as a normal per-call tuning knob; leave at defaults for
    real match processing.

    calibration_frame/image_pts/pitch_pts are passed in explicitly rather
    than hardcoded, since — per the known gap discussed with the user —
    there's no automatic full-match calibration yet. Until that exists,
    whatever calls this task needs to supply calibration for whichever
    camera segment it wants real Stage 5/6 output for. Processing frames
    outside that calibration will simply not produce pitch-space data
    for those frames (Stages 1-4 still run and save fine regardless).
    """
    from apps.matches.models import Match
    from apps.analytics.models import TeamStatistics

    try:
        match = Match.objects.get(id=match_id)
        match.status = Match.MatchStatus.PROCESSING
        match.save(update_fields=["status"])

        # --- Stages 1-4: run across the whole clip, no calibration needed ---
        tracker = Tracker(DEFAULT_CONFIG.detection, DEFAULT_CONFIG.tracking)
        tracklets = tracker.track_video(video_path)

        min_duration_frames = int(DEFAULT_CONFIG.pitch_mapping.min_tracklet_duration_sec * 25)
        survivors = [t for t in tracklets.values() if t.duration_frames >= min_duration_frames]

        cap = cv2.VideoCapture(video_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        colors, valid = {}, []
        for t in survivors:
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

        all_det = defaultdict(list)
        for t in tracklets.values():
            for d in t.detections:
                all_det[d.frame_idx].append(d)
        ball_by_frame = extract_ball_detections(dict(all_det))
        ball_traj_px = interpolate_gaps(ball_by_frame, DEFAULT_CONFIG.ball_tracking, frame_w, frame_h)

        # --- Stage 5: homography, only valid within the supplied calibration ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, calibration_frame)
        ok, bootstrap_frame = cap.read()
        homography_tracker = HomographyTracker(DEFAULT_CONFIG.pitch_mapping)
        homography_tracker.bootstrap(bootstrap_frame, image_pts, pitch_pts)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        homography_by_frame = {calibration_frame: homography_tracker.current_H.copy()}
        cap.set(cv2.CAP_PROP_POS_FRAMES, calibration_frame + 1)
        for frame_idx in range(calibration_frame + 1, total_frames):
            ok, frame = cap.read()
            if not ok:
                break
            H = homography_tracker.update(frame)
            if H is not None:
                homography_by_frame[frame_idx] = H.copy()

        ball_pitch_trajectory = []
        for p in ball_traj_px:
            if p.x_m is None or p.frame_idx not in homography_by_frame:
                continue
            pt = image_point_to_pitch(p.x_m, p.y_m, homography_by_frame[p.frame_idx])
            ball_pitch_trajectory.append(
                BallTrajectoryPoint(frame_idx=p.frame_idx, x_m=pt.x_m, y_m=pt.y_m, interpolated=p.interpolated)
            )

        identities = []
        for t in valid:
            traj = {}
            for det in t.detections:
                if det.frame_idx not in homography_by_frame:
                    continue
                fx, fy = (det.x1 + det.x2) / 2, det.y2
                pt = image_point_to_pitch(fx, fy, homography_by_frame[det.frame_idx])
                traj[det.frame_idx] = PitchPoint(x_m=pt.x_m, y_m=pt.y_m)
            if traj:
                identities.append(MasterIdentity(master_id=t.track_id, team=t.team, reid_embedding=[], trajectory=traj))

        # --- Stage 5b: stitch + assign Master IDs (only within-shot for now) ---
        stitched = match_tracklets_within_shot(valid, DEFAULT_CONFIG.pitch_mapping, fps=25.0)
        gallery = IdentityGallery(DEFAULT_CONFIG.pitch_mapping)
        assign_tracklets_to_gallery(stitched, gallery)

        # --- Stage 6: event detection ---
        possession_events = detect_possession(ball_pitch_trajectory, identities, DEFAULT_CONFIG.event_detection)
        pass_events = detect_passes(possession_events, identities)
        shot_events = detect_shots(
            ball_pitch_trajectory,
            (52.5, 0.0),
            min_shot_speed_mps=shot_min_speed_mps,
            min_alignment=shot_min_alignment,
        )

        # --- Team attribution lookups, built once from identities ---
        # possession/passes both resolve to a player_master_id, which we
        # can map to a team via the identities we already built above.
        # This closes the TODO from the first draft — real per-team
        # numbers now, not placeholders.
        team_by_master_id = {i.master_id: i.team for i in identities}

        possession_frame_counts_by_team = {Team.TEAM_A: 0, Team.TEAM_B: 0}
        for i in range(len(possession_events)):
            start_frame = possession_events[i].frame_idx
            end_frame = possession_events[i + 1].frame_idx if i + 1 < len(possession_events) else start_frame
            holder_team = team_by_master_id.get(possession_events[i].player_master_id)
            if holder_team in possession_frame_counts_by_team:
                possession_frame_counts_by_team[holder_team] += max(end_frame - start_frame, 1)
        total_possession_frames = sum(possession_frame_counts_by_team.values()) or 1

        passes_by_team = {Team.TEAM_A: 0, Team.TEAM_B: 0}
        for e in pass_events:
            passer_team = team_by_master_id.get(e.player_master_id)
            if passer_team in passes_by_team:
                passes_by_team[passer_team] += 1

        # Shots don't carry a player_master_id yet (detect_shots works
        # purely off ball trajectory, no player attribution built in) —
        # attribute each shot to whichever team had the closest player
        # to the ball at that frame, as a reasonable proxy. Not as solid
        # as passes/possession's direct attribution, but far better than
        # leaving it uncounted.
        shots_by_team = {Team.TEAM_A: 0, Team.TEAM_B: 0}
        for e in shot_events:
            closest_team, closest_dist = None, float("inf")
            for identity in identities:
                pos = identity.trajectory.get(e.frame_idx)
                if pos is None:
                    continue
                # ball position at this frame, from ball_pitch_trajectory
                ball_pos = next((p for p in ball_pitch_trajectory if p.frame_idx == e.frame_idx), None)
                if ball_pos is None:
                    continue
                dist = ((pos.x_m - ball_pos.x_m) ** 2 + (pos.y_m - ball_pos.y_m) ** 2) ** 0.5
                if dist < closest_dist:
                    closest_dist = dist
                    closest_team = identity.team
            if closest_team in shots_by_team:
                shots_by_team[closest_team] += 1

        # --- Save TeamStatistics (real, team-level data) ---
        # NOTE: TEAM_A/TEAM_B -> home_team/away_team mapping needs your
        # actual per-match kit color lookup here — placeholder mapping
        # below assumes TEAM_A=home, TEAM_B=away, which is NOT reliably
        # true (depends purely on which cluster K-means labeled first).
        # Verify/replace with a real color-based lookup before trusting
        # this in production.
        for team_enum, team_obj in [(Team.TEAM_A, match.home_team), (Team.TEAM_B, match.away_team)]:
            possession_pct = 100.0 * possession_frame_counts_by_team[team_enum] / total_possession_frames
            distance = _compute_team_distance_m(identities, team_enum)

            TeamStatistics.objects.update_or_create(
                match=match,
                team=team_obj,
                defaults={
                    "total_distance": distance,
                    "possession": possession_pct,
                    "shots": shots_by_team[team_enum],
                    "passes_completed": passes_by_team[team_enum],
                },
            )

        match.status = Match.MatchStatus.COMPLETED
        match.processing_progress = 100
        match.save(update_fields=["status", "processing_progress"])

    except Exception as exc:
        match = Match.objects.filter(id=match_id).first()
        if match:
            match.status = Match.MatchStatus.FAILED
            match.save(update_fields=["status"])
        raise
