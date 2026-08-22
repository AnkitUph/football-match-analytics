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
        # Uses this match's ACTUAL kit colors (captured at upload) when
        # available — more reliable than blind clustering, and avoids
        # the "which cluster is home vs away" ambiguity blind K-means
        # has. Falls back to blind clustering only if kit colors weren't
        # captured for this match.
        cap2 = cv2.VideoCapture(video_path)
        colors, valid = {}, []
        for t in survivors:
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

        if match.home_kit_color and match.away_kit_color:
            home_bgr = _hex_to_bgr(match.home_kit_color)
            away_bgr = _hex_to_bgr(match.away_kit_color)
            for t in valid:
                if t.cls and t.cls.value == "referee":
                    from ai_engine.utils.types import Team
                    t.team = Team.REFEREE
                else:
                    t.team = classify_team_by_known_colors(colors[t.track_id], home_bgr, away_bgr)
        else:
            clusters = fit_team_color_clusters(list(colors.values()), DEFAULT_CONFIG.team_reid)
            for t in valid:
                t.team = classify_team(t.cls.value if t.cls else "player", colors[t.track_id], clusters)

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
