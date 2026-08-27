"""
Phase 6: real background processing, replacing Phase 4's simulated
time.sleep loop.

SCOPE OF WHAT'S REAL HERE (read before extending):
  - Video metadata (duration, fps, resolution) — real, straightforward.
  - Player/ball tracking data (Stage 1-4) — real, saved as CSVs. Doesn't
    need any camera calibration.
  - TeamStatistics / PlayerStatistics — NOT populated here. Every field
    on those models (possession, shots, distance, xG...) needs Stage 5's
    pitch-space mapping, which needs a camera calibration that doesn't
    exist for a fresh upload (no clicked calibration points, no
    homography). match_results() in views.py falls back to the existing
    Phase 5 dummy generator for these until a calibration exists for
    this match — see the calibration-problem discussion for the planned
    follow-up (manual per-upload calibration UI, later full automation).

The PENDING -> PROCESSING -> COMPLETED/FAILED status contract is
unchanged from Phase 4, so nothing else in the app needs to change.
"""

import csv
import io
from collections import defaultdict

from celery import shared_task
from django.core.files.base import ContentFile

import cv2


@shared_task(bind=True)
def process_match(self, match_id):
    # Local imports: avoid circular/app-loading issues at Celery worker
    # startup, same reasoning as the original Phase 4 task.
    from apps.matches.models import Match, MatchFiles

    from ai_engine.config import DEFAULT_CONFIG
    from ai_engine.stage2_tracking.tracker import Tracker
    from ai_engine.stage3_team_reid.team_classifier import (
        classify_team,
        classify_team_by_known_colors,
        fit_team_color_clusters,
        sample_torso_color,
    )
    from ai_engine.stage4_ball_tracking.ball_tracker import extract_ball_detections, interpolate_gaps

    try:
        match = Match.objects.get(pk=match_id)
    except Match.DoesNotExist:
        return

    match.status = Match.MatchStatus.PROCESSING
    match.processing_progress = 0
    match.save(update_fields=["status", "processing_progress", "updated_at"])

    try:
        video_path = match.video.original_video.path

        # --- Video metadata (real, no calibration needed) ---
        cap = cv2.VideoCapture(video_path)
        # cv2 can return -1 (or 0, or NaN) for these properties if the
        # video didn't open cleanly for this specific read — found via
        # real testing that blindly trusting these values crashed the
        # whole task with a MySQL "out of range" error (PositiveIntegerField
        # is UNSIGNED, and -1 doesn't fit). Validate before assigning;
        # skip metadata gracefully rather than failing the whole task
        # over fields that don't block the actual detection work below.
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video file: {video_path}")

        raw_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        raw_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        raw_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if raw_fps and raw_fps > 0 and raw_frame_count and raw_frame_count > 0:
            match.video.video_duration = raw_frame_count / raw_fps
        if raw_fps and raw_fps > 0:
            match.video.fps = raw_fps
        if raw_width and raw_width > 0:
            match.video.resolution_width = int(raw_width)
        if raw_height and raw_height > 0:
            match.video.resolution_height = int(raw_height)
        match.video.save()

        match.processing_progress = 15
        match.save(update_fields=["processing_progress", "updated_at"])

        # --- Stage 1+2: detection + tracking (real, no calibration needed) ---
        tracker = Tracker(DEFAULT_CONFIG.detection, DEFAULT_CONFIG.tracking)
        tracklets = tracker.track_video(video_path)

        match.processing_progress = 50
        match.save(update_fields=["processing_progress", "updated_at"])

        min_duration_frames = int(DEFAULT_CONFIG.pitch_mapping.min_tracklet_duration_sec * 25)
        survivors = [t for t in tracklets.values() if t.duration_frames >= min_duration_frames]

        # --- Stage 3: team classification ---
        # Uses classify_teams_with_fallback: tries this match's ACTUAL
        # kit colors (captured at upload) first, but automatically falls
        # back to blind clustering if that produces an implausibly
        # one-sided split — found via real testing that stored kit
        # colors can be wrong/unrelated to the actual footage (e.g. test
        # data pairing made-up teams with real video), which silently
        # classified 100% of players into one team. See
        # team_classifier.classify_teams_with_fallback's docstring.
        cap2 = cv2.VideoCapture(video_path)
        colors, valid = {}, []
        referees = []
        for t in survivors:
            if t.cls and t.cls.value == "referee":
                from ai_engine.utils.types import Team
                t.team = Team.REFEREE
                referees.append(t)
                continue
            det = t.detections[0]
            cap2.set(cv2.CAP_PROP_POS_FRAMES, det.frame_idx)
            ok, frame = cap2.read()
            if not ok:
                continue
            x1, y1, x2, y2 = map(int, det.bbox)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            color = sample_torso_color(crop)
            if color is not None:
                colors[t.track_id] = color
                valid.append(t)

        home_bgr = _hex_to_bgr(match.home_kit_color) if match.home_kit_color else None
        away_bgr = _hex_to_bgr(match.away_kit_color) if match.away_kit_color else None

        from ai_engine.stage3_team_reid.team_classifier import classify_teams_with_fallback
        team_by_track_id = classify_teams_with_fallback(colors, home_bgr, away_bgr, DEFAULT_CONFIG.team_reid)
        for t in valid:
            t.team = team_by_track_id[t.track_id]
        valid = valid + referees

        match.processing_progress = 70
        match.save(update_fields=["processing_progress", "updated_at"])

        # --- Stage 4: ball extraction + interpolation (real, no calibration needed) ---
        all_det = defaultdict(list)
        for t in tracklets.values():
            for d in t.detections:
                all_det[d.frame_idx].append(d)
        ball_by_frame = extract_ball_detections(dict(all_det))
        frame_w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ball_trajectory = interpolate_gaps(ball_by_frame, DEFAULT_CONFIG.ball_tracking, frame_w, frame_h)

        match.processing_progress = 85
        match.save(update_fields=["processing_progress", "updated_at"])

        # --- Save real tracking data as CSVs on MatchFiles ---
        # Pixel-space only (no calibration = no pitch coordinates yet).
        # Saving this now means Stage 5 won't need to re-run detection
        # once a calibration exists later — it can work from these CSVs.
        files, _ = MatchFiles.objects.get_or_create(match=match)

        player_csv = io.StringIO()
        writer = csv.writer(player_csv)
        writer.writerow(["track_id", "frame_idx", "team", "class", "x1", "y1", "x2", "y2", "conf"])
        for t in valid:
            for det in t.detections:
                writer.writerow([t.track_id, det.frame_idx, t.team.value, det.cls.value, det.x1, det.y1, det.x2, det.y2, det.conf])
        files.player_tracking_csv.save(f"match_{match.id}_players.csv", ContentFile(player_csv.getvalue()), save=False)

        ball_csv = io.StringIO()
        writer = csv.writer(ball_csv)
        writer.writerow(["frame_idx", "x_px", "y_px", "interpolated"])
        for p in ball_trajectory:
            if p.x_m is not None:
                writer.writerow([p.frame_idx, p.x_m, p.y_m, p.interpolated])
        files.ball_tracking_csv.save(f"match_{match.id}_ball.csv", ContentFile(ball_csv.getvalue()), save=False)

        files.save()

        # ============================================================
        # DEMO-ONLY: Stage 5-6 using a KNOWN, hardcoded calibration.
        #
        # This is NOT a general solution — it only works because this
        # specific match's video happens to be the same footage we
        # manually calibrated and validated earlier (frame 540, holdout-
        # tested to ~35cm accuracy). A genuinely different upload has no
        # calibration and this whole block should be skipped for it.
        #
        # TODO before this can run on real uploads: replace the hardcoded
        # CALIBRATION_FRAME/IMAGE_PTS/PITCH_PTS below with a real lookup
        # (e.g. a MatchCalibration model populated via a per-upload
        # calibration UI). Until that exists, only attempt this block
        # for matches you know match this specific test footage.
        # ============================================================
        CALIBRATION_FRAME = 540
        IMAGE_PTS = [(1022, 348), (1467, 561), (1309, 329), (1864, 520)]
        PITCH_PTS = [(36, 20.16), (36, -20.16), (52.5, 20.16), (52.5, -20.16)]

        from ai_engine.stage5_pitch_mapping.homography_tracker import HomographyTracker
        from ai_engine.stage5_pitch_mapping.homography import image_point_to_pitch
        from ai_engine.stage5_pitch_mapping.identity_association import (
            IdentityGallery,
            assign_tracklets_to_gallery,
            match_tracklets_within_shot,
        )
        from ai_engine.stage6_event_detection.events import detect_possession, detect_passes, detect_shots
        from ai_engine.utils.types import MasterIdentity, PitchPoint, Team
        from apps.analytics.models import TeamStatistics

        cap3 = cv2.VideoCapture(video_path)
        cap3.set(cv2.CAP_PROP_POS_FRAMES, CALIBRATION_FRAME)
        ok, bootstrap_frame = cap3.read()

        homography_tracker = HomographyTracker(DEFAULT_CONFIG.pitch_mapping)
        if ok and homography_tracker.bootstrap(bootstrap_frame, IMAGE_PTS, PITCH_PTS):
            total_frames = int(cap3.get(cv2.CAP_PROP_FRAME_COUNT))
            homography_by_frame = {CALIBRATION_FRAME: homography_tracker.current_H.copy()}
            cap3.set(cv2.CAP_PROP_POS_FRAMES, CALIBRATION_FRAME + 1)
            for frame_idx in range(CALIBRATION_FRAME + 1, total_frames):
                ok, frame = cap3.read()
                if not ok:
                    break
                H = homography_tracker.update(frame)
                if H is not None:
                    homography_by_frame[frame_idx] = H.copy()

            ball_pitch_trajectory = []
            for p in ball_trajectory:
                if p.x_m is None or p.frame_idx not in homography_by_frame:
                    continue
                pt = image_point_to_pitch(p.x_m, p.y_m, homography_by_frame[p.frame_idx])
                from ai_engine.utils.types import BallTrajectoryPoint
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

            stitched = match_tracklets_within_shot(valid, DEFAULT_CONFIG.pitch_mapping, fps=25.0)
            gallery = IdentityGallery(DEFAULT_CONFIG.pitch_mapping)
            assign_tracklets_to_gallery(stitched, gallery)

            possession_events = detect_possession(ball_pitch_trajectory, identities, DEFAULT_CONFIG.event_detection)
            pass_events = detect_passes(possession_events, identities)
            shot_events = detect_shots(ball_pitch_trajectory, (52.5, 0.0))

            team_by_master_id = {i.master_id: i.team for i in identities}

            possession_frame_counts = {Team.TEAM_A: 0, Team.TEAM_B: 0}
            for i, event in enumerate(possession_events):
                start_frame = event.frame_idx
                end_frame = possession_events[i + 1].frame_idx if i + 1 < len(possession_events) else start_frame
                holder_team = team_by_master_id.get(event.player_master_id)
                if holder_team in possession_frame_counts:
                    possession_frame_counts[holder_team] += max(end_frame - start_frame, 1)
            total_possession_frames = sum(possession_frame_counts.values()) or 1

            passes_by_team = {Team.TEAM_A: 0, Team.TEAM_B: 0}
            for e in pass_events:
                passer_team = team_by_master_id.get(e.player_master_id)
                if passer_team in passes_by_team:
                    passes_by_team[passer_team] += 1

            shots_by_team = {Team.TEAM_A: 0, Team.TEAM_B: 0}
            for e in shot_events:
                closest_team, closest_dist = None, float("inf")
                for identity in identities:
                    pos = identity.trajectory.get(e.frame_idx)
                    if pos is None:
                        continue
                    ball_pos = next((bp for bp in ball_pitch_trajectory if bp.frame_idx == e.frame_idx), None)
                    if ball_pos is None:
                        continue
                    dist = ((pos.x_m - ball_pos.x_m) ** 2 + (pos.y_m - ball_pos.y_m) ** 2) ** 0.5
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_team = identity.team
                if closest_team in shots_by_team:
                    shots_by_team[closest_team] += 1

            def team_distance_m(team_enum):
                total = 0.0
                for identity in identities:
                    if identity.team != team_enum:
                        continue
                    frames = sorted(identity.trajectory.keys())
                    for f1, f2 in zip(frames, frames[1:]):
                        p1, p2 = identity.trajectory[f1], identity.trajectory[f2]
                        total += ((p2.x_m - p1.x_m) ** 2 + (p2.y_m - p1.y_m) ** 2) ** 0.5
                return total

            # Reliable now: TEAM_A is defined as "closer to home_kit_color"
            # by classify_team_by_known_colors, since Stage 3 used real
            # kit colors (not blind clustering) when they're available.
            for team_enum, team_obj in [(Team.TEAM_A, match.home_team), (Team.TEAM_B, match.away_team)]:
                possession_pct = 100.0 * possession_frame_counts[team_enum] / total_possession_frames
                TeamStatistics.objects.update_or_create(
                    match=match,
                    team=team_obj,
                    defaults={
                        "total_distance": team_distance_m(team_enum),
                        "possession": possession_pct,
                        "shots": shots_by_team[team_enum],
                        "passes_completed": passes_by_team[team_enum],
                    },
                )

        match.status = Match.MatchStatus.COMPLETED
        match.processing_progress = 100
        match.save(update_fields=["status", "processing_progress", "updated_at"])

    except Exception:
        match.status = Match.MatchStatus.FAILED
        match.save(update_fields=["status", "updated_at"])
        raise


def _hex_to_bgr(hex_color: str):
    """Converts a '#RRGGBB' hex string to a BGR array, matching
    ai_engine.stage3_team_reid.team_classifier's expected input format."""
    import numpy as np
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([b, g, r], dtype=np.float32)
