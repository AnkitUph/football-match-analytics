"""
Phase 6: real background processing, replacing Phase 4's simulated
time.sleep loop.

SCOPE OF WHAT'S REAL HERE (read before extending):
  - Video metadata (duration, fps, resolution) — real, straightforward.
  - Player/ball tracking data (Stage 1-4, this file's process_match) —
    real, saved as CSVs. Doesn't need any camera calibration, runs at
    upload time unconditionally.
  - TeamStatistics / PlayerStatistics — populated by compute_pitch_mapping
    (this file), a SEPARATE task that only runs once a MatchCalibration
    exists for the match (manual per-match calibration UI — see
    apps/matches/views.py: calibrate_match/calibrate_frame/calibrate_save).
    Until a match is calibrated, match_results() in views.py falls back
    to the Phase 5 dummy generator for these fields. Recalibrating a
    match re-runs ONLY compute_pitch_mapping (reads the CSVs process_match
    already wrote) — it never re-runs YOLO/tracking.

The PENDING -> PROCESSING -> COMPLETED/FAILED status contract covers
process_match (Stage 1-4) only. compute_pitch_mapping doesn't change
Match.status — a match can be COMPLETED with or without real pitch stats.
"""

import csv
import io
import logging
import math
import random
from collections import defaultdict

from celery import shared_task
from django.core.files.base import ContentFile

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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

        # --- Stage 3: Team Re-ID, Stitching & Classification (DINOv2 + Color Fallback) ---
        colors, valid = {}, []
        track_embeddings = {}
        referees = []

        from ai_engine.stage3_team_reid.reid import ReidEmbedder
        embedder = ReidEmbedder(DEFAULT_CONFIG.team_reid)

        # High-impact optimization:
        # Instead of 15 samples per tracklet and thousands of random seeking operations,
        # sample up to 3 high-quality crops per tracklet (start, middle, end) and read
        # video frames in a single monotonic forward pass.
        MAX_COLOR_SAMPLES_PER_TRACK = 3
        crops_needed_by_frame = defaultdict(list)
        for t in survivors:
            n = len(t.detections)
            sample_count = min(MAX_COLOR_SAMPLES_PER_TRACK, n)
            sample_indices = sorted(set(
                round(i * (n - 1) / max(sample_count - 1, 1)) for i in range(sample_count)
            ))
            for idx in sample_indices:
                det = t.detections[idx]
                crops_needed_by_frame[det.frame_idx].append((t.track_id, det.bbox))

        # Single monotonic forward pass through video
        cap2 = cv2.VideoCapture(video_path)
        track_crops = defaultdict(list)
        track_colors = defaultdict(list)
        sorted_frames = sorted(crops_needed_by_frame.keys())
        curr_pos = 0

        for target_f in sorted_frames:
            if target_f != curr_pos:
                if 0 < (target_f - curr_pos) <= 8:
                    while curr_pos < target_f:
                        cap2.grab()
                        curr_pos += 1
                else:
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, target_f)
                    curr_pos = target_f

            ok, frame = cap2.read()
            curr_pos += 1
            if not ok:
                continue

            for tid, bbox in crops_needed_by_frame[target_f]:
                x1, y1, x2, y2 = map(int, bbox)
                # CRITICAL: .copy() creates an isolated small array (~15 KB)
                # instead of retaining a view on the entire 6.2 MB 1080p frame!
                crop = frame[max(0, y1):y2, max(0, x1):x2].copy()
                if crop.size > 0 and crop.shape[0] >= 15 and crop.shape[1] >= 8:
                    track_crops[tid].append(crop)
                color = sample_torso_color(crop)
                if color is not None:
                    track_colors[tid].append(color)

        cap2.release()
        del crops_needed_by_frame

        # Compute median colors per tracklet
        for tid, col_list in track_colors.items():
            if col_list:
                colors[tid] = np.median(np.array(col_list), axis=0)
        del track_colors

        # Batch embed crops through DINOv2 (flattened into one batched forward pass)
        all_flat_crops = []
        crop_slices = {}
        for t in survivors:
            crops = track_crops.get(t.track_id, [])
            # Embed tracklets with duration >= 12 frames (or all if very few tracks)
            if crops and (t.duration_frames >= 12 or len(survivors) <= 40):
                start = len(all_flat_crops)
                all_flat_crops.extend(crops)
                crop_slices[t.track_id] = (start, start + len(crops))

        del track_crops

        if all_flat_crops and getattr(DEFAULT_CONFIG.team_reid, "use_hf_dinov2", True):
            logger.info("Stage 3: Embedding %d crops across %d tracks in batch mode...", len(all_flat_crops), len(crop_slices))
            all_feats = embedder.embed_batch(all_flat_crops)
            for tid, (start, end) in crop_slices.items():
                valid_feats = [f for f in all_feats[start:end] if f is not None]
                if valid_feats:
                    mean_feat = np.mean(np.array(valid_feats), axis=0)
                    norm = np.linalg.norm(mean_feat)
                    track_embeddings[tid] = (mean_feat / norm) if norm > 1e-6 else mean_feat

        del all_flat_crops, crop_slices
        import gc
        gc.collect()

        for t in survivors:
            valid.append(t)

        # Tracklet Stitching across brief occlusions using DINOv2 Re-ID similarity
        if getattr(DEFAULT_CONFIG.tracking, "enable_stitching", True) and track_embeddings:
            from ai_engine.stage2_tracking.stitcher import stitch_tracklets
            track_dict = {t.track_id: t for t in valid}
            stitched_dict, id_mapping = stitch_tracklets(
                track_dict,
                track_embeddings,
                max_gap_frames=getattr(DEFAULT_CONFIG.tracking, "stitch_max_gap_frames", 125),
                similarity_thresh=getattr(DEFAULT_CONFIG.tracking, "stitch_similarity_thresh", 0.82),
            )
            valid = list(stitched_dict.values())

            # Pool embeddings and colors across all constituent merged fragments
            emb_groups = defaultdict(list)
            col_groups = defaultdict(list)
            for old_id, root_id in id_mapping.items():
                if old_id in track_embeddings:
                    emb_groups[root_id].append(track_embeddings[old_id])
                if old_id in colors:
                    col_groups[root_id].append(colors[old_id])

            new_embeddings = {}
            for root_id, embs in emb_groups.items():
                mean_e = np.mean(embs, axis=0)
                norm = np.linalg.norm(mean_e)
                new_embeddings[root_id] = (mean_e / norm) if norm > 1e-6 else mean_e

            new_colors = {}
            for root_id, cols in col_groups.items():
                new_colors[root_id] = np.median(cols, axis=0)

            track_embeddings = new_embeddings
            colors = new_colors

        home_bgr = _hex_to_bgr(match.home_kit_color) if match.home_kit_color else None
        away_bgr = _hex_to_bgr(match.away_kit_color) if match.away_kit_color else None
        home_gk_bgr = _hex_to_bgr(match.home_gk_kit_color) if match.home_gk_kit_color else None
        away_gk_bgr = _hex_to_bgr(match.away_gk_kit_color) if match.away_gk_kit_color else None
        cls_by_track = {t.track_id: (t.cls.value if t.cls else None) for t in valid}

        from ai_engine.stage3_team_reid.team_classifier import classify_teams_with_dinov2, classify_teams_with_fallback, perceptual_kit_distance
        if getattr(DEFAULT_CONFIG.team_reid, "use_hf_dinov2", True) and len(track_embeddings) >= 4:
            team_by_track_id = classify_teams_with_dinov2(
                track_embeddings, colors, home_bgr, away_bgr, DEFAULT_CONFIG.team_reid,
                home_gk_bgr=home_gk_bgr, away_gk_bgr=away_gk_bgr, cls_by_track=cls_by_track,
            )
        else:
            team_by_track_id = classify_teams_with_fallback(
                colors, home_bgr, away_bgr, DEFAULT_CONFIG.team_reid,
                home_gk_bgr=home_gk_bgr, away_gk_bgr=away_gk_bgr, cls_by_track=cls_by_track,
            )

        from ai_engine.utils.types import Team, ObjectClass

        # Referee Disambiguation:
        # Identify true on-pitch referee track(s) using referee detection counts,
        # distinct kit color (outlier from both team kits), and single-referee exclusivity.
        ref_candidates = []
        for t in valid:
            ref_detections = sum(1 for d in t.detections if d.cls and d.cls.value == "referee")
            tot_detections = len(t.detections)
            if ref_detections >= 10 or (t.cls and t.cls.value == "referee") or (ref_detections >= 5 and ref_detections / max(tot_detections, 1) >= 0.15):
                col = colors.get(t.track_id)
                matches_home = home_bgr is not None and col is not None and perceptual_kit_distance(col, home_bgr) < 45.0
                matches_away = away_bgr is not None and col is not None and perceptual_kit_distance(col, away_bgr) < 45.0
                if not (matches_home or matches_away):
                    score = tot_detections * (ref_detections / max(tot_detections, 1))
                    ref_candidates.append((t, score))

        ref_candidates.sort(key=lambda x: -x[1])
        verified_ref_ids = set()
        occupied_ref_frames = set()

        for rt, score in ref_candidates:
            r_frames = {d.frame_idx for d in rt.detections}
            # Strictly at most one on-pitch referee: check temporal exclusivity
            if len(r_frames & occupied_ref_frames) <= 2:
                verified_ref_ids.add(rt.track_id)
                occupied_ref_frames.update(r_frames)

        for t in valid:
            if t.track_id in verified_ref_ids or t.cls == ObjectClass.REFEREE or (t.cls and getattr(t.cls, "value", "") == "referee"):
                t.team = Team.REFEREE
            else:
                t.team = team_by_track_id.get(t.track_id, Team.UNKNOWN)

        # --- Stage 3c: jersey number OCR (secondary signal, gated by config) ---
        # Runs on player/goalkeeper tracklets only (team_a/team_b) —
        # referees are skipped, their number isn't relevant here. Mutates
        # each tracklet's .jersey_number/.jersey_number_conf in place.
        # Expect a real, honest failure rate: broadcast-resolution torso
        # crops mean many tracklets will end up with jersey_number=None
        # (the aggregation function requires multiple agreeing reads
        # before accepting a number — see jersey_ocr.aggregate_jersey_number).
        # That's working as intended, not a bug to chase by loosening the
        # confidence bar. This is purely additive to Stage 1-4 — never
        # blocks CSV writing, COMPLETED status, or anything downstream.
        if DEFAULT_CONFIG.enable_ocr:
            from ai_engine.stage3_team_reid.jersey_ocr import run_jersey_ocr_for_tracklets
            try:
                run_jersey_ocr_for_tracklets(valid, video_path, DEFAULT_CONFIG.team_reid)
            except Exception:
                # OCR is a bonus signal — a failure here (e.g. EasyOCR
                # model download issue) should not fail the whole match.
                # Tracklets simply keep jersey_number=None (their default).
                pass

        match.processing_progress = 70
        match.save(update_fields=["processing_progress", "updated_at"])

        # --- Stage 4: ball extraction + interpolation (real, no calibration needed) ---
        all_det = defaultdict(list)
        for t in tracklets.values():
            for d in t.detections:
                all_det[d.frame_idx].append(d)
        ball_by_frame = getattr(tracker, "ball_by_frame", None) or extract_ball_detections(dict(all_det))
        frame_w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ball_trajectory = interpolate_gaps(ball_by_frame, DEFAULT_CONFIG.ball_tracking, frame_w, frame_h)

        match.processing_progress = 85
        match.save(update_fields=["processing_progress", "updated_at"])

        # --- Save real tracking data as CSVs on MatchFiles ---
        # No pitch_x/pitch_y here — this task no longer knows or cares
        # about calibration. Those columns get added (and the CSV
        # rewritten) by compute_pitch_mapping, once a MatchCalibration
        # exists for this match (see apps/matches/views.py:calibrate_save
        # and apps/matches/tasks.py:compute_pitch_mapping). A match with
        # no calibration simply has tracking data with no pitch mapping —
        # match_results() in views.py falls back to dummy stats for it.
        #
        # jersey_number/jersey_conf are per-TRACKLET (same value repeated
        # on every row for that track_id, like team already is) — blank
        # when Stage 3c OCR is disabled or couldn't confidently read a
        # number for that tracklet.
        files, _ = MatchFiles.objects.get_or_create(match=match)

        player_csv = io.StringIO()
        writer = csv.writer(player_csv)
        writer.writerow(["track_id", "frame_idx", "team", "class", "x1", "y1", "x2", "y2", "conf", "jersey_number", "jersey_conf"])
        for t in valid:
            jersey_number = t.jersey_number if t.jersey_number is not None else ""
            jersey_conf = round(t.jersey_number_conf, 3) if t.jersey_number is not None else ""
            for det in t.detections:
                writer.writerow([t.track_id, det.frame_idx, t.team.value, det.cls.value, det.x1, det.y1, det.x2, det.y2, det.conf, jersey_number, jersey_conf])
        files.player_tracking_csv.save(f"match_{match.id}_players.csv", ContentFile(player_csv.getvalue()), save=False)

        ball_csv = io.StringIO()
        writer = csv.writer(ball_csv)
        writer.writerow(["frame_idx", "x_px", "y_px", "interpolated"])
        for p in ball_trajectory:
            # Save ALL frames, including None gaps — this preserves the full
            # frame timeline so compute_pitch_mapping can re-interpolate over
            # the pitch-projected trajectory instead of losing gap timestamps.
            x_val = round(p.x_m, 2) if p.x_m is not None else ""
            y_val = round(p.y_m, 2) if p.y_m is not None else ""
            writer.writerow([p.frame_idx, x_val, y_val, p.interpolated])
        files.ball_tracking_csv.save(f"match_{match.id}_ball.csv", ContentFile(ball_csv.getvalue()), save=False)

        files.save()

        # Automatic calibration: runs for every match, no human step —
        # see _run_automatic_calibration's docstring for exactly what it
        # does and when it gives up (fewer than 4 confident keypoints).
        # Skipped if a calibration ALREADY exists (e.g. from a previous
        # run, or a manual fix via the hidden /calibrate/ page) — a
        # human's override should never be silently clobbered by a
        # later reprocessing run.
        from apps.matches.models import MatchCalibration
        if not MatchCalibration.objects.filter(match=match).exists():
            try:
                _run_automatic_calibration(match, video_path)
            except Exception:
                logger.exception("Automatic calibration failed for match=%s, continuing without it", match.id)

        if MatchCalibration.objects.filter(match=match).exists():
            try:
                # Run synchronously so real TeamStatistics, PlayerStatistics,
                # passes, shots, and heatmaps are fully computed BEFORE status=COMPLETED
                compute_pitch_mapping(match.id)
            except Exception:
                logger.exception("compute_pitch_mapping failed for match=%s, continuing", match.id)

        # Generate annotated tracking video with CV bounding boxes and ball trail
        # Rendered BEFORE status=COMPLETED so results page loads with Tracked (AI) immediately ready
        match.processing_progress = 92
        match.save(update_fields=["processing_progress", "updated_at"])
        try:
            from ai_engine.stage7_visualization.annotated_video import render_annotated_match_video
            render_annotated_match_video(match)
        except Exception:
            logger.exception("Failed to render annotated video for match=%s", match.id)

        # PDF report, generated with REAL stats
        try:
            from apps.reports.generator import generate_match_report
            generate_match_report(match)
        except Exception:
            pass

        match.status = Match.MatchStatus.COMPLETED
        match.processing_progress = 100
        match.save(update_fields=["status", "processing_progress", "updated_at"])

    except Exception:
        match.status = Match.MatchStatus.FAILED
        match.save(update_fields=["status", "updated_at"])
        raise


def _run_automatic_calibration(match, video_path, num_anchors=4, samples_per_window=3):
    """
    Runs immediately after Stage 1-4 finishes, for every match, no human
    involved. Divides the clip into num_anchors equal windows and, within
    EACH window, samples several frames — keeping whichever sampled
    frame has the highest SUM OF CONFIDENCES across its detected points
    (see "WHY SCORE BY SUMMED CONFIDENCE" below — this replaced an
    x-span-based scoring approach that was tried and found to actively
    backfire). Saves a MatchCalibration anchor for each window that has
    at least one usable sample (>=4 confident points) — a homography is
    mathematically undefined below that (see
    ai_engine/stage5_pitch_mapping/homography.py:compute_homography_from_points's
    own `len(image_points) < 4` check), so a window either gets a real
    anchor or is skipped outright, never a partial/degraded one.

    WHY MULTIPLE ANCHORS: tested single-anchor automatic calibration
    against match 10 (test_1.mp4/Mainz) — the one match with a
    known-good MANUAL 3-anchor calibration (frames 100, 440, 600) that
    had already produced real passes/shots/xG. A single middle-frame
    anchor produced near-zero team distances (16-36m across a whole
    clip) and zero passes/shots — a real, confirmed regression. Root
    cause: compute_pitch_mapping's HomographyTracker only tracks FORWARD
    from an anchor's own frame to the next anchor's frame (or the end of
    the clip) — see compute_pitch_mapping's segment logic. One anchor
    covering an entire clip fails once the camera pans/zooms enough for
    its tracked points to drift or leave frame.

    TRIED AND REJECTED: scoring candidate frames by real-world x-span
    (how far apart the detected points are along the pitch length),
    reasoning that a spatially spread-out set of points would produce a
    homography that extrapolates well across its whole segment instead
    of just near wherever it was calibrated. This was motivated by a
    real finding — mapping the same ball detection through match 10 and
    a single-anchor automatic calibration at the same frame showed a
    >10m disagreement (52.55m vs 42.88m) that plausibly hid a real shot
    — but a visual reprojection-overlay check (same technique used
    earlier this session to catch the class_id/"class" string bug)
    proved x-span scoring backfired: it directly rewards including
    spurious far-field detections, since a WRONGLY-labeled
    "right_box_top"/"right_corner_top"/"right_penalty_spot" cluster
    inflates x-span exactly as effectively as a real one would. The
    overlay showed this cluster confidently drawn on empty grass with no
    actual box or corner anywhere near it, while that same anchor's
    halfway-line/center-circle points — the one cluster independently
    validated via reprojection tests on two separate clips earlier this
    session — were correctly aligned. Maximizing spread specifically
    hunts for the least-validated, most failure-prone landmark types
    (box/goal-line corners, which sit far from center by construction)
    over the well-validated center cluster.

    WHY SCORE BY SUMMED CONFIDENCE INSTEAD: a more neutral proxy for
    "this is a clean, unambiguous frame" — it rewards having many
    confident detections without specifically going looking for distant
    landmark types the way x-span does. Doesn't fully solve the
    underlying problem (a frame could still combine several confident
    center points with one confident-but-wrong far point, and summed
    confidence wouldn't know the difference) — see KNOWN LIMITATION
    below.

    num_anchors=4 and samples_per_window=3 are general-purpose defaults
    (not tuned to test_1.mp4 specifically) — 12 Roboflow calls per match
    total. More of either means smaller/better-conditioned segments at
    the cost of more API calls per match.

    KNOWN LIMITATION: still no human look at any candidate frame before
    it's used, and no per-landmark trust weighting — every landmark ID
    is treated as equally reliable even though only the halfway-line/
    center-circle cluster has been independently validated via visual
    reprojection so far (see pitch_landmarks_32.py and this function's
    own rejected x-span attempt above for the concrete evidence that the
    box/goal-line landmarks are NOT equally trustworthy on every clip).
    A future improvement worth considering: down-weight or exclude
    landmark IDs outside the validated center cluster unless a
    per-match visual check confirms them. If a specific match's
    auto-calibration still comes out visibly wrong, the hidden
    /calibrate/ page (not linked from results.html, but still live at
    its URL) can add/override anchors manually — see MatchCalibration's
    docstring, and note process_match will not clobber existing
    calibrations on a later reprocessing run (see the check at this
    function's call site, above).

    Never raises — any failure here (network, bad frame, no API key)
    should not fail Stage 1-4 over a Roboflow problem, same resilience
    pattern as PDF report generation above. The caller still wraps this
    in try/except as a second layer of safety.
    """
    import os
    from django.conf import settings
    from apps.matches.models import MatchCalibration
    from ai_engine.stage5_pitch_mapping.smart_assist import detect_pitch_keypoints

    # 0. Check if a previously calibrated match used the same video file (by file size).
    # If so, inherit the existing verified calibration anchors (including multi-view behind-the-goal anchors).
    try:
        from apps.matches.models import MatchVideo
        curr_size = os.path.getsize(video_path) if os.path.exists(video_path) else None
        if curr_size:
            for mv in MatchVideo.objects.exclude(match=match).select_related("match"):
                if mv.original_video and os.path.exists(mv.original_video.path):
                    if os.path.getsize(mv.original_video.path) == curr_size:
                        prev_cals = MatchCalibration.objects.filter(match=mv.match)
                        if prev_cals.exists():
                            for pc in prev_cals:
                                MatchCalibration.objects.update_or_create(
                                    match=match,
                                    calibration_frame=pc.calibration_frame,
                                    defaults={"points": pc.points},
                                )
                            logger.info(
                                "match=%s: inherited %d calibration anchors from matching video in match=%s",
                                match.id, prev_cals.count(), mv.match_id,
                            )
                            return
    except Exception:
        logger.exception("Failed to check for matching video calibrations for match=%s", match.id)

    if not settings.ROBOFLOW_API_KEY:
        logger.info("match=%s: ROBOFLOW_API_KEY not configured, skipping auto-calibration", match.id)
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return

    # Scale anchor count with video duration: 1 anchor per ~60 seconds (1500 frames)
    # Ensures long clips have sufficient calibration coverage even when optical flow loses tracking
    num_anchors = max(num_anchors, total_frames // 1500)

    saved_count = 0
    for i in range(num_anchors):
        window_start = int(total_frames * i / num_anchors)
        window_end = max(int(total_frames * (i + 1) / num_anchors), window_start + 1)

        candidate_offsets = sorted(set(
            min(window_start + int((window_end - window_start) * (j + 0.5) / samples_per_window), total_frames - 1)
            for j in range(samples_per_window)
        ))
        if i == 0 and 0 not in candidate_offsets:
            candidate_offsets = [0] + candidate_offsets

        best_suggestions = None
        best_frame_idx = None
        best_score = -1

        for frame_idx in candidate_offsets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            # 1. Reject non-tactical broadcast frames (replays, close-ups, crowd, graphics)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            grass_mask = cv2.inRange(hsv, (35, 38, 38), (85, 255, 255))
            h_f, w_f = frame.shape[:2]
            green_ratio = float(np.count_nonzero(grass_mask) / (h_f * w_f))
            top_h = max(1, int(h_f * 0.25))
            top_green_ratio = float(np.count_nonzero(grass_mask[:top_h, :]) / (top_h * w_f))
            # Tactical sideline camera requires >= 45% grass, and top quarter must NOT be all grass (replays/zooms)
            if green_ratio < 0.45 or top_green_ratio > 0.35:
                continue

            suggestions = detect_pitch_keypoints(
                frame, settings.ROBOFLOW_API_KEY, confidence_threshold=0.5, max_points=12
            )
            if len(suggestions) < 4:
                suggestions = detect_pitch_keypoints(
                    frame, settings.ROBOFLOW_API_KEY, confidence_threshold=0.35, max_points=12
                )
            if len(suggestions) < 4:
                continue

            # 2. Mathematical Conditioning Guard: Ensure points span 2D space and form a non-degenerate H
            img_pts = [(s["pixel_x"], s["pixel_y"]) for s in suggestions]
            pitch_pts = [(s["pitch_x"], s["pitch_y"]) for s in suggestions]
            from ai_engine.stage5_pitch_mapping.homography import compute_homography_from_points
            from ai_engine.config import DEFAULT_CONFIG
            H_cand = compute_homography_from_points(img_pts, pitch_pts, DEFAULT_CONFIG.pitch_mapping)
            if H_cand is None or abs(H_cand[2, 2]) < 1e-4:
                continue
            H_norm = H_cand / H_cand[2, 2]
            det_cand = abs(np.linalg.det(H_norm))
            cond_cand = np.linalg.cond(H_norm)
            x_span = max(p[0] for p in pitch_pts) - min(p[0] for p in pitch_pts)
            y_span = max(p[1] for p in pitch_pts) - min(p[1] for p in pitch_pts)
            if det_cand < 1e-4 or cond_cand > 250000 or x_span < 12.0 or y_span < 12.0:
                continue

            score = sum(s["confidence"] for s in suggestions)
            if score > best_score:
                best_score = score
                best_suggestions = suggestions
                best_frame_idx = frame_idx

        if best_suggestions is None:
            logger.info(
                "match=%s: no candidate frame in window [%d, %d) yielded >=4 confident keypoints, skipping this anchor",
                match.id, window_start, window_end,
            )
            continue

        points = [
            {
                "landmark_id": s["landmark_id"],
                "pixel_x": s["pixel_x"],
                "pixel_y": s["pixel_y"],
                "pitch_x": s["pitch_x"],
                "pitch_y": s["pitch_y"],
            }
            for s in best_suggestions
        ]

        MatchCalibration.objects.update_or_create(
            match=match,
            calibration_frame=best_frame_idx,
            defaults={"points": points},
        )
        saved_count += 1
        logger.info(
            "match=%s: auto-calibrated anchor at frame %d (window [%d,%d), confidence_sum=%.2f) with %d points",
            match.id, best_frame_idx, window_start, window_end, best_score, len(points),
        )

    cap.release()

    if saved_count == 0:
        logger.info("match=%s: no window yielded a usable calibration anchor, no auto-calibration saved", match.id)
    else:
        logger.info("match=%s: auto-calibration saved %d/%d window anchors", match.id, saved_count, num_anchors)


@shared_task(bind=True)
def compute_pitch_mapping(self, match_id):
    """
    Stage 5-6, fully decoupled from Stage 1-4 (process_match above).

    Triggered whenever a MatchCalibration is saved/updated for a match
    (see views.calibrate_save), or automatically at the end of
    process_match if a calibration already existed before Stage 1-4
    finished. Reads the CSVs process_match already wrote instead of
    re-running YOLO/tracking — a recalibration only changes the pixel<->
    pitch mapping, never the underlying detections, so there's no reason
    to pay for detection again.

    Rewrites player_tracking_csv in place with fresh pitch_x/pitch_y
    columns (from the new calibration), and recomputes TeamStatistics.
    Does NOT touch Match.status/processing_progress — those describe
    Stage 1-4 only; a match can be COMPLETED with or without real pitch
    stats.
    """
    from apps.matches.models import Match
    from apps.analytics.models import TeamStatistics

    from ai_engine.config import DEFAULT_CONFIG
    from ai_engine.stage5_pitch_mapping.homography_tracker import HomographyTracker
    from ai_engine.stage5_pitch_mapping.homography import compute_homography_from_points, image_point_to_pitch
    from ai_engine.stage5_pitch_mapping.identity_association import match_tracklets_within_shot
    from ai_engine.stage6_event_detection.events import (
        detect_possession,
        detect_passes,
        detect_shots,
        estimate_shot_xg,
        detect_extended_match_events,
        compute_player_physical_metrics,
        compute_player_rating,
        compute_continuous_possession,
        detect_passes_with_metadata,
        detect_corner_kicks,
        infer_team_defending_goals,
    )
    from ai_engine.utils.types import (
        BallTrajectoryPoint,
        Detection,
        MasterIdentity,
        ObjectClass,
        PitchPoint,
        Team,
        Tracklet,
    )

    try:
        match = Match.objects.get(pk=match_id)
    except Match.DoesNotExist:
        return

    calibrations = list(match.calibrations.order_by("calibration_frame"))
    if not calibrations:
        return  # nothing to do without at least one calibration anchor

    files = getattr(match, "files", None)
    if not files or not files.player_tracking_csv:
        # Stage 1-4 hasn't produced tracking data yet (still processing,
        # or failed). Not fatal — process_match re-triggers this task
        # once it finishes, if a calibration exists by then.
        return

    video_path = match.video.original_video.path

    # --- Track each calibration anchor's homography across ITS OWN
    # segment of the video (not the whole video from one anchor) ---
    # See MatchCalibration's docstring for why: a single anchor's
    # optical-flow tracking can be lost partway through a long pan (a
    # calibration point leaves frame, gets occluded, no auto-recovery),
    # silently leaving every later frame with no real pitch mapping.
    # Multiple anchors fix this WITHOUT chaining across them either —
    # each anchor bootstraps fresh from its own 4 points and tracks
    # forward only until the NEXT anchor's frame (exclusive), where a
    # completely fresh bootstrap takes over. One anchor's tracking
    # failure only costs that anchor's own segment, not everything after
    # it in the match.
    homography_by_frame = {}
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i, calibration in enumerate(calibrations):
        segment_end_frame = (
            calibrations[i + 1].calibration_frame if i + 1 < len(calibrations) else total_frames
        )
        if calibration.calibration_frame >= segment_end_frame:
            # Shouldn't happen given unique_together + ordering, but
            # skip defensively rather than looping backward.
            continue

        image_pts = [(p["pixel_x"], p["pixel_y"]) for p in calibration.points]
        pitch_pts = [(p["pitch_x"], p["pitch_y"]) for p in calibration.points]

        cap.set(cv2.CAP_PROP_POS_FRAMES, calibration.calibration_frame)
        ok, bootstrap_frame = cap.read()

        homography_tracker = HomographyTracker(DEFAULT_CONFIG.pitch_mapping)
        if not (ok and homography_tracker.bootstrap(bootstrap_frame, image_pts, pitch_pts)):
            # This ONE anchor failed to bootstrap (bad frame/points) —
            # skip just its segment, not the whole match. Other anchors'
            # segments still get real pitch mapping.
            continue

        seg_frames = [calibration.calibration_frame]
        seg_H = [homography_tracker.current_H.copy()]
        cap.set(cv2.CAP_PROP_POS_FRAMES, calibration.calibration_frame + 1)
        was_cut_paused = False
        anchor_H = homography_tracker.current_H.copy()
        for frame_idx in range(calibration.calibration_frame + 1, segment_end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            H = homography_tracker.update(frame)
            if H is not None and abs(H[2, 2]) > 1e-4:
                H_norm = H / H[2, 2]
                det_H = abs(np.linalg.det(H_norm))
                cond_H = np.linalg.cond(H_norm)
                if det_H > 1e-4 and cond_H < 250000:
                    if was_cut_paused:
                        # Resumed from cut: re-anchor optical flow on the fresh frame
                        homography_tracker.resume_after_cut(frame, H_target=anchor_H)
                        was_cut_paused = False
                    seg_frames.append(frame_idx)
                    seg_H.append(H_norm.copy())
                else:
                    was_cut_paused = True
            else:
                was_cut_paused = True

        # If next anchor has valid bootstrap homography, apply smooth boundary drift correction
        next_calib = calibrations[i + 1] if i + 1 < len(calibrations) else None
        next_H = None
        if next_calib is not None:
            n_img = [(p["pixel_x"], p["pixel_y"]) for p in next_calib.points]
            n_pitch = [(p["pitch_x"], p["pitch_y"]) for p in next_calib.points]
            H_cand = compute_homography_from_points(n_img, n_pitch, DEFAULT_CONFIG.pitch_mapping)
            if H_cand is not None:
                next_H = H_cand / H_cand[2, 2]

        has_intervening_cut = was_cut_paused or (
            next_calib is not None
            and len(seg_frames) > 0
            and seg_frames[-1] < next_calib.calibration_frame - 5
        )
        if next_H is not None and len(seg_H) > 1 and not has_intervening_cut:
            try:
                Delta = next_H @ np.linalg.inv(seg_H[-1])
                det_delta = np.linalg.det(Delta)
                # Only apply Delta blend if Delta has positive determinant and is well-conditioned
                if 0.05 < det_delta < 20.0:
                    N = len(seg_H) - 1
                    for k, f_idx in enumerate(seg_frames):
                        alpha = k / N
                        H_corr = ((1.0 - alpha) * np.eye(3) + alpha * Delta) @ seg_H[k]
                        det_corr = abs(np.linalg.det(H_corr))
                        if det_corr > 0.005 and abs(H_corr[2, 2]) > 1e-4 and np.isfinite(H_corr).all():
                            homography_by_frame[f_idx] = H_corr / H_corr[2, 2]
                        else:
                            homography_by_frame[f_idx] = seg_H[k]
                else:
                    for f_idx, H_val in zip(seg_frames, seg_H):
                        homography_by_frame[f_idx] = H_val
            except Exception:
                for f_idx, H_val in zip(seg_frames, seg_H):
                    homography_by_frame[f_idx] = H_val
        else:
            for f_idx, H_val in zip(seg_frames, seg_H):
                homography_by_frame[f_idx] = H_val

    cap.release()

    # Backfill homography for any early frames preceding the first tracked anchor
    if homography_by_frame:
        earliest_frame = min(homography_by_frame.keys())
        if earliest_frame > 0:
            first_H = homography_by_frame[earliest_frame]
            for f in range(0, earliest_frame):
                homography_by_frame[f] = first_H.copy()

    # TEMPORARY diagnostic — separate from last_tracked_frame (which is
    # ball-trajectory-derived and can be capped by Stage 1-4 ball
    # detection gaps having nothing to do with homography coverage).
    # This reports homography_by_frame's ACTUAL coverage directly, so we
    # can tell whether a given anchor's tracking succeeded independent
    # of whether the ball itself was detected that far.
    covered_frames = sorted(homography_by_frame.keys())
    logger.info(
        "compute_pitch_mapping match=%s: homography covers %d frames, range=%s-%s",
        match.id, len(covered_frames),
        covered_frames[0] if covered_frames else None,
        covered_frames[-1] if covered_frames else None,
    )
    for calibration in calibrations:
        anchor_frame = calibration.calibration_frame
        frames_from_this_anchor_onward = [f for f in covered_frames if f >= anchor_frame]
        logger.info(
            "  anchor@%s: bootstrapped=%s, max_covered_frame_at_or_after_anchor=%s",
            anchor_frame,
            anchor_frame in homography_by_frame,
            max(frames_from_this_anchor_onward) if frames_from_this_anchor_onward else None,
        )

    if not homography_by_frame:
        return

    # --- Reconstruct tracklets from the saved player_tracking_csv ---
    # Team/class/every detection are already in there from Stage 1-4 —
    # no re-detection needed. shot_id is fixed at 0 for all of them:
    # Stage 2.5 (shot-boundary detection) is deferred, so the whole clip
    # is still treated as one continuous shot, same assumption
    # process_match's original inline version made.
    with files.player_tracking_csv.open("rb") as f:
        content = f.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    has_jersey_columns = reader.fieldnames and "jersey_number" in reader.fieldnames

    detections_by_track = defaultdict(list)
    team_by_track = {}
    jersey_by_track = {}  # track_id -> (jersey_number or None, jersey_conf)
    for row in reader:
        track_id = int(row["track_id"])
        team_by_track[track_id] = Team(row["team"])
        detections_by_track[track_id].append(
            Detection(
                frame_idx=int(row["frame_idx"]),
                cls=ObjectClass(row["class"]),
                conf=float(row["conf"]),
                x1=float(row["x1"]), y1=float(row["y1"]),
                x2=float(row["x2"]), y2=float(row["y2"]),
            )
        )
        if track_id not in jersey_by_track:
            raw_number = row.get("jersey_number") if has_jersey_columns else ""
            raw_conf = row.get("jersey_conf") if has_jersey_columns else ""
            jersey_number = int(raw_number) if raw_number not in (None, "") else None
            jersey_conf = float(raw_conf) if raw_conf not in (None, "") else 0.0
            jersey_by_track[track_id] = (jersey_number, jersey_conf)

    valid = []
    for track_id, dets in detections_by_track.items():
        dets.sort(key=lambda d: d.frame_idx)
        jersey_number, jersey_conf = jersey_by_track.get(track_id, (None, 0.0))
        valid.append(Tracklet(
            track_id=track_id,
            shot_id=0,
            detections=dets,
            cls=dets[0].cls,
            team=team_by_track[track_id],
            jersey_number=jersey_number,
            jersey_number_conf=jersey_conf,
        ))

    # --- Reconstruct ball trajectory from the saved ball_tracking_csv ---
    ball_trajectory = []
    if files.ball_tracking_csv:
        with files.ball_tracking_csv.open("rb") as f:
            ball_content = f.read().decode("utf-8")
        for row in csv.DictReader(io.StringIO(ball_content)):
            raw_x = row.get("x_px", "")
            raw_y = row.get("y_px", "")
            # Gracefully handle None/empty gap rows written by process_match
            x_m = float(raw_x) if raw_x not in ("", None) else None
            y_m = float(raw_y) if raw_y not in ("", None) else None
            ball_trajectory.append(BallTrajectoryPoint(
                frame_idx=int(row["frame_idx"]),
                x_m=x_m,
                y_m=y_m,
                interpolated=row["interpolated"].strip().lower() == "true",
            ))

    # --- Stage 5-6: possession/pass/shot detection + TeamStatistics ---
    # Build raw pitch trajectory from pixel detections + homography
    ball_pitch_raw: list[BallTrajectoryPoint] = []
    for p in ball_trajectory:
        if p.x_m is not None and p.frame_idx in homography_by_frame:
            pt = image_point_to_pitch(p.x_m, p.y_m, homography_by_frame[p.frame_idx])
            if abs(pt.x_m) <= 55.0 and abs(pt.y_m) <= 37.0:
                ball_pitch_raw.append(
                    BallTrajectoryPoint(frame_idx=p.frame_idx, x_m=pt.x_m, y_m=pt.y_m, interpolated=p.interpolated)
                )

    # Second-pass Kalman interpolation in pitch-metric space:
    # Gaps survive the homography projection when the ball pixel position is
    # valid but the frame's homography is missing (e.g. during close-ups /
    # replay cuts). Bridging these in pitch space is safer — the pitch
    # coordinate system has hard bounds (±55m, ±37m) that immediately reject
    # any extrapolation drifting off the playing surface.
    from ai_engine.stage4_ball_tracking.ball_tracker import interpolate_gaps as _interpolate_pitch_gaps
    from ai_engine.utils.types import Detection as _Det
    if ball_pitch_raw:
        all_frames = sorted({p.frame_idx for p in ball_trajectory})
        raw_by_frame = {p.frame_idx: p for p in ball_pitch_raw}
        pitch_by_frame: dict = {}
        for fid in all_frames:
            rpt = raw_by_frame.get(fid)
            if rpt is not None:
                # Wrap pitch meters in a fake Detection so interpolate_gaps can handle it
                pitch_by_frame[fid] = _Det(
                    frame_idx=fid, cls=ObjectClass.BALL, conf=1.0,
                    x1=rpt.x_m - 0.1, y1=rpt.y_m - 0.1,
                    x2=rpt.x_m + 0.1, y2=rpt.y_m + 0.1,
                )
            else:
                pitch_by_frame[fid] = None
        # frame_width/height guard in pitch-meter units (±55m x, ±37m y)
        # Re-map to [0, 110] x [0, 74] so the frame-bounds guard works correctly
        for fid, det in pitch_by_frame.items():
            if det is not None:
                det.x1 += 55.0; det.x2 += 55.0
                det.y1 += 37.0; det.y2 += 37.0
        pitch_interp = _interpolate_pitch_gaps(
            pitch_by_frame, DEFAULT_CONFIG.ball_tracking,
            frame_width=110, frame_height=74,
        )
        ball_pitch_trajectory: list[BallTrajectoryPoint] = []
        for tp in pitch_interp:
            if tp.x_m is not None:
                ball_pitch_trajectory.append(BallTrajectoryPoint(
                    frame_idx=tp.frame_idx,
                    x_m=tp.x_m - 55.0,   # undo the offset back to [-55, 55]
                    y_m=tp.y_m - 37.0,   # undo the offset back to [-37, 37]
                    interpolated=tp.interpolated,
                ))
    else:
        ball_pitch_trajectory = []

    # Rewrite ball_tracking_csv with pitch coordinates included
    ball_csv = io.StringIO()
    ball_writer = csv.writer(ball_csv)
    ball_writer.writerow(["frame_idx", "x_px", "y_px", "interpolated", "pitch_x", "pitch_y"])
    pitch_traj_by_frame = {p.frame_idx: p for p in ball_pitch_trajectory}
    for p in ball_trajectory:
        px, py = "", ""
        ptp = pitch_traj_by_frame.get(p.frame_idx)
        if ptp is not None:
            px, py = round(ptp.x_m, 2), round(ptp.y_m, 2)
        x_val = round(p.x_m, 2) if p.x_m is not None else ""
        y_val = round(p.y_m, 2) if p.y_m is not None else ""
        ball_writer.writerow([p.frame_idx, x_val, y_val, p.interpolated, px, py])
    files.ball_tracking_csv.save(f"match_{match.id}_ball.csv", ContentFile(ball_csv.getvalue()), save=True)


    # --- Merge fragmented raw tracklets using metric pitch distance and Unified Master ID Table ---
    from ai_engine.stage5_pitch_mapping.identity_association import build_unified_master_identity_table
    stitched, id_mapping = build_unified_master_identity_table(
        valid, DEFAULT_CONFIG.pitch_mapping, fps=25.0, homography_by_frame=homography_by_frame
    )

    # Rewrite player_tracking_csv using STITCHED tracklets so track IDs, teams, and crop bounding boxes match 100%
    player_csv = io.StringIO()
    writer = csv.writer(player_csv)
    writer.writerow(["track_id", "frame_idx", "team", "class", "x1", "y1", "x2", "y2", "conf", "jersey_number", "jersey_conf", "pitch_x", "pitch_y"])
    for t in stitched:
        jersey_number = t.jersey_number if t.jersey_number is not None else ""
        jersey_conf = round(t.jersey_number_conf, 3) if t.jersey_number is not None else ""
        for det in t.detections:
            pitch_x = pitch_y = ""
            H = homography_by_frame.get(det.frame_idx)
            if H is not None:
                fx, fy = (det.x1 + det.x2) / 2, det.y2
                pt = image_point_to_pitch(fx, fy, H)
                pitch_x, pitch_y = round(pt.x_m, 2), round(pt.y_m, 2)
            writer.writerow([t.track_id, det.frame_idx, t.team.value, det.cls.value, det.x1, det.y1, det.x2, det.y2, det.conf, jersey_number, jersey_conf, pitch_x, pitch_y])
    files.player_tracking_csv.save(f"match_{match.id}_players.csv", ContentFile(player_csv.getvalue()), save=True)

    identities = []
    for t in stitched:
        if getattr(t, "cls", None) == ObjectClass.BALL:
            continue
        traj = {}
        for det in t.detections:
            if det.frame_idx not in homography_by_frame:
                continue
            fx, fy = (det.x1 + det.x2) / 2, det.y2
            pt = image_point_to_pitch(fx, fy, homography_by_frame[det.frame_idx])
            if abs(pt.x_m) <= 55.0 and abs(pt.y_m) <= 37.0:
                traj[det.frame_idx] = PitchPoint(x_m=pt.x_m, y_m=pt.y_m)
        if traj:
            identities.append(MasterIdentity(
                master_id=t.track_id, team=t.team, reid_embedding=[],
                jersey_number=t.jersey_number, cls=t.cls, trajectory=traj,
            ))

    cls_by_track = {t.track_id: t.cls for t in stitched}
    try:
        _auto_assign_unidentified_tracks(match, identities, cls_by_track=cls_by_track)
    except Exception:
        logger.exception("Failed to auto-assign tracks for match=%s", match.id)

    # Filter broadcast footage: exclude replays, crowd/dugout cuts, and graphic transitions
    from ai_engine.stage5_pitch_mapping.view_classifier import get_non_tactical_frame_ranges
    try:
        non_tactical_ranges = get_non_tactical_frame_ranges(video_path, stride=10)
    except Exception:
        non_tactical_ranges = []

    def is_non_tactical(f_idx):
        return any(s <= f_idx <= e for s, e in non_tactical_ranges)

    tactical_ball_trajectory = [p for p in ball_pitch_trajectory if not is_non_tactical(p.frame_idx)]

    # Continuous frame-level possession + discrete possession change events
    cont_possession = compute_continuous_possession(tactical_ball_trajectory, identities, fps=25.0)
    possession_events = cont_possession["possession_events"]
    if not possession_events:
        possession_events = detect_possession(tactical_ball_trajectory, identities, DEFAULT_CONFIG.event_detection)
    possession_events = [e for e in possession_events if not is_non_tactical(e.frame_idx)]
    possession_pct_by_team = cont_possession["percentages"]

    pass_events = detect_passes(possession_events, identities)
    recorded_passes = detect_passes_with_metadata(possession_events, identities, tactical_ball_trajectory, fps=25.0)
    corner_events = detect_corner_kicks(tactical_ball_trajectory, identities, fps=25.0)
    pass_intervals = [
        (p["start_frame"], p["frame_idx"])
        for p in recorded_passes
        if p.get("is_completed") and not (
            abs(float(p.get("end_x", 0.0))) >= 47.0
            and abs(float(p.get("end_y", 0.0))) <= 12.0
            and float(p.get("speed_mps", 0.0)) >= 14.0
        )
    ]
    shot_events = detect_shots(
        tactical_ball_trajectory,
        ((52.5, 0.0), (-52.5, 0.0)),
        identities=identities,
        pass_intervals=pass_intervals,
        celebration_intervals=non_tactical_ranges,
    )

    team_by_master_id = {i.master_id: i.team for i in identities}

    passes_by_team = {Team.TEAM_A: 0, Team.TEAM_B: 0}
    for e in pass_events:
        passer_team = team_by_master_id.get(e.player_master_id)
        if passer_team in passes_by_team:
            passes_by_team[passer_team] += 1

    # Per-player completed-pass count, same track_id space as
    # player_stats_csv below (identity.master_id). NOTE: detect_passes
    # only ever emits a COMPLETED pass — a failed/intercepted pass is
    # architecturally invisible to this stage (see events.py), so this
    # is a real "passes completed" count but there is deliberately no
    # matching "passes attempted"/"pass accuracy" anywhere real — don't
    # try to derive one, it would be fabricated. See _load_real_distance_by_jersey
    # / _load_real_distance_by_assignment and _dummy_player_rows in
    # views.py for how this is kept separate from the still-dummy
    # attempted/accuracy fields on the results page.
    passes_completed_by_track_id = defaultdict(int)
    for e in pass_events:
        passes_completed_by_track_id[e.player_master_id] += 1

    # Per-shot: closest identity (not just team) to the ball at the shot
    # frame — same "closest identity to ball at that frame" inference
    # already used for team-level shot counting, just also keeping the
    # specific identity around now for per-player attribution. This is
    # an INFERENCE, not a directly-detected fact (detect_shots' Event
    # carries no player_master_id at all — see events.py) — flagged as
    # such wherever it surfaces on the results page (shots_is_real /
    # xg_is_real in views.py).
    #
    # xG per shot comes from estimate_shot_xg — a simplified distance/
    # angle heuristic, NOT a trained model. See that function's
    # docstring in events.py for exactly what it does and doesn't claim.
    ball_pitch_by_frame = {bp.frame_idx: bp for bp in ball_pitch_trajectory}

    shots_by_team = {Team.TEAM_A: 0, Team.TEAM_B: 0}
    shots_on_target_by_team = {Team.TEAM_A: 0, Team.TEAM_B: 0}
    xg_by_team = {Team.TEAM_A: 0.0, Team.TEAM_B: 0.0}
    shots_by_track_id = defaultdict(int)
    shots_on_target_by_track_id = defaultdict(int)
    xg_by_track_id = defaultdict(float)
    recorded_shots = []
    team_defending_goal = infer_team_defending_goals(identities, cls_by_track=cls_by_track, ball_trajectory=ball_pitch_trajectory)

    for e in shot_events:
        ball_pos = ball_pitch_by_frame.get(e.frame_idx)
        if ball_pos is None:
            continue

        target_goal = e.metadata.get("target_goal", (52.5, 0.0))
        attacking_teams = [t for t, def_goal in team_defending_goal.items() if def_goal != target_goal]
        target_attacking_team = attacking_teams[0] if attacking_teams else None

        closest_identity, closest_dist = None, float("inf")
        # Search strike window [e.frame_idx - 4, e.frame_idx + 1]
        for identity in identities:
            if getattr(identity, "cls", None) in (ObjectClass.REFEREE, ObjectClass.BALL) or identity.team == Team.REFEREE:
                continue
            if identity.team not in shots_by_team:
                continue
            for f in range(max(0, e.frame_idx - 4), e.frame_idx + 2):
                pos = identity.trajectory.get(f)
                b_pt = ball_pitch_by_frame.get(f)
                if pos is None or b_pt is None:
                    continue
                dist = ((pos.x_m - b_pt.x_m) ** 2 + (pos.y_m - b_pt.y_m) ** 2) ** 0.5
                if dist < closest_dist:
                    closest_dist = dist
                    closest_identity = identity

        # If no striker within close physical proximity (<= 2.2m), fall back to closest attacking team player
        if closest_identity is None or closest_dist > 2.2:
            attacker_id, attacker_dist = None, float("inf")
            for identity in identities:
                if getattr(identity, "cls", None) in (ObjectClass.REFEREE, ObjectClass.BALL) or identity.team == Team.REFEREE:
                    continue
                if target_attacking_team and identity.team != target_attacking_team:
                    continue
                if identity.team not in shots_by_team:
                    continue
                for f in range(max(0, e.frame_idx - 6), e.frame_idx + 3):
                    pos = identity.trajectory.get(f)
                    b_pt = ball_pitch_by_frame.get(f)
                    if pos is None or b_pt is None:
                        continue
                    dist = ((pos.x_m - b_pt.x_m) ** 2 + (pos.y_m - b_pt.y_m) ** 2) ** 0.5
                    if dist < attacker_dist:
                        attacker_dist = dist
                        attacker_id = identity
            if attacker_id is not None:
                closest_identity = attacker_id
                closest_dist = attacker_dist

        # Determine shot team:
        # A shot directed towards target_goal is an offensive attack on that goal by target_attacking_team.
        shot_team = target_attacking_team if target_attacking_team else (closest_identity.team if closest_identity else None)
        if shot_team is None or shot_team not in shots_by_team:
            continue

        # If closest_identity is from the defending team, attribute to closest striker from shot_team
        if closest_identity and closest_identity.team != shot_team:
            best_striker = None
            best_striker_dist = float("inf")
            for identity in identities:
                if identity.team != shot_team or getattr(identity, "cls", None) in (ObjectClass.REFEREE, ObjectClass.BALL):
                    continue
                for f in range(max(0, e.frame_idx - 6), e.frame_idx + 3):
                    pos = identity.trajectory.get(f)
                    b_pt = ball_pitch_by_frame.get(f)
                    if pos and b_pt:
                        d = ((pos.x_m - b_pt.x_m) ** 2 + (pos.y_m - b_pt.y_m) ** 2) ** 0.5
                        if d < best_striker_dist:
                            best_striker_dist = d
                            best_striker = identity
            if best_striker and best_striker_dist <= 4.0:
                closest_identity = best_striker
                closest_dist = best_striker_dist

        shot_xg = estimate_shot_xg(e.metadata["origin_distance_m"], e.metadata["alignment"])
        is_on_target = bool(e.metadata.get("is_on_target", False))
        is_goal = bool(e.metadata.get("is_goal", False))
        if is_goal:
            is_on_target = True

        shots_by_team[shot_team] += 1
        if is_on_target:
            shots_on_target_by_team[shot_team] += 1
            if closest_identity:
                shots_on_target_by_track_id[closest_identity.master_id] += 1

        xg_by_team[shot_team] += shot_xg
        if closest_identity:
            shots_by_track_id[closest_identity.master_id] += 1
            xg_by_track_id[closest_identity.master_id] += shot_xg

        video_minute = max(1, round((e.frame_idx / 25.0) / 60.0))
        track_id = closest_identity.master_id if closest_identity else None
        if is_goal:
            outcome = "Goal"
        elif is_on_target:
            outcome = "On Target"
        else:
            outcome = "Off Target"

        recorded_shots.append({
            "frame_idx": e.frame_idx,
            "minute": video_minute,
            "track_id": track_id,
            "team": shot_team.value,
            "pitch_x": round(ball_pos.x_m, 2),
            "pitch_y": round(ball_pos.y_m, 2),
            "target_goal_x": target_goal[0],
            "target_goal_y": target_goal[1],
            "distance_m": round(e.metadata["origin_distance_m"], 1),
            "speed_mps": round(e.metadata.get("speed_mps", 0.0), 1),
            "alignment": round(e.metadata["alignment"], 3),
            "xg": round(shot_xg, 3),
            "is_on_target": is_on_target,
            "is_goal": is_goal,
            "outcome": outcome,
        })

    # Extended event detection: attempted passes, duels (tackles/interceptions), clearances, dribbles, key passes
    extended_events = detect_extended_match_events(
        ball_trajectory=ball_pitch_trajectory,
        identities=identities,
        possession_events=possession_events,
        shot_events=shot_events,
        fps=25.0,
    )
    passes_attempted_by_team = extended_events["team_passes_attempted"]
    passes_attempted_by_track_id = extended_events["passes_attempted"]
    tackles_by_track_id = extended_events["tackles"]
    interceptions_by_track_id = extended_events["interceptions"]
    clearances_by_track_id = extended_events["clearances"]
    dribbles_by_track_id = extended_events["dribbles"]
    key_passes_by_track_id = extended_events["key_passes"]

    # Physical metrics calculation: distance covered, top speed, average speed
    physical_by_track_id = {}
    for identity in identities:
        physical_by_track_id[identity.master_id] = compute_player_physical_metrics(identity, fps=25.0)

    def team_distance_m(team_enum):
        return sum(
            physical_by_track_id[ident.master_id]["distance_m"]
            for ident in identities
            if ident.team == team_enum
        )

    def team_avg_speed(team_enum):
        speeds = [
            physical_by_track_id[ident.master_id].get("average_speed_kmh", 0.0)
            for ident in identities
            if ident.team == team_enum and physical_by_track_id[ident.master_id].get("average_speed_kmh", 0.0) > 0
        ]
        return round(sum(speeds) / len(speeds), 1) if speeds else 0.0

    corners_by_team = {Team.TEAM_A: 0, Team.TEAM_B: 0}
    for c in corner_events:
        t_enum = Team.TEAM_A if c.get("team") == "team_a" else Team.TEAM_B
        corners_by_team[t_enum] += 1

    # TeamStatistics: real possession, shots, shots on target, passes completed/attempted, accuracy, corners, speed
    for team_enum, team_obj in [(Team.TEAM_A, match.home_team), (Team.TEAM_B, match.away_team)]:
        possession_pct = possession_pct_by_team.get(team_enum, 50.0)
        completed = passes_by_team[team_enum]
        attempted = passes_attempted_by_team.get(team_enum, completed)
        if attempted < completed:
            attempted = completed
        accuracy = round(100.0 * completed / attempted, 1) if attempted > 0 else 0.0
        s_count = shots_by_team[team_enum]
        sot_count = shots_on_target_by_team[team_enum]
        if sot_count > s_count:
            sot_count = s_count

        team_goals = sum(1 for s in recorded_shots if s.get("is_goal") and s["team"] == team_enum.value)

        TeamStatistics.objects.update_or_create(
            match=match,
            team=team_obj,
            defaults={
                "goals": team_goals,
                "total_distance": team_distance_m(team_enum),
                "possession": possession_pct,
                "shots": s_count,
                "shots_on_target": sot_count,
                "passes_completed": completed,
                "passes_attempted": attempted,
                "pass_accuracy": accuracy,
                "corners": corners_by_team.get(team_enum, 0),
                "xg": round(xg_by_team[team_enum], 2),
                "average_team_speed": team_avg_speed(team_enum),
            },
        )

    # Compute algorithmic player ratings
    from apps.matches.models import MatchGoal, TrackPlayerIdentification
    from apps.matches.views import _recompute_match_score

    track_to_lineup = dict(TrackPlayerIdentification.objects.filter(match=match).values_list("track_id", "lineup_entry_id"))

    # Auto-sync confirmed goals to MatchGoal and update scoreboard
    # Clear stale auto-synced goals for this match to ensure clean sync
    MatchGoal.objects.filter(match=match).delete()
    for s in recorded_shots:
        if s.get("is_goal"):
            goal_team = match.home_team if s["team"] == "team_a" else match.away_team
            scorer_lid = track_to_lineup.get(s.get("track_id"))
            goal_obj, created = MatchGoal.objects.get_or_create(
                match=match,
                team=goal_team,
                minute=s["minute"],
                defaults={
                    "scorer_id": scorer_lid,
                    "is_own_goal": False,
                },
            )
            if not created and scorer_lid and not goal_obj.scorer_id:
                goal_obj.scorer_id = scorer_lid
                goal_obj.save(update_fields=["scorer"])
    _recompute_match_score(match)

    goals_by_track_id = defaultdict(int)
    lineup_goals = defaultdict(int)
    for g in MatchGoal.objects.filter(match=match, is_own_goal=False):
        if g.scorer_id is not None:
            lineup_goals[g.scorer_id] += 1
    for tid, lid in track_to_lineup.items():
        if lid in lineup_goals:
            goals_by_track_id[tid] = lineup_goals[lid]

    ratings_by_track_id = {}
    for identity in identities:
        tid = identity.master_id
        phys = physical_by_track_id.get(tid, {})
        comp = passes_completed_by_track_id.get(tid, 0)
        att = passes_attempted_by_track_id.get(tid, comp)
        if att < comp:
            att = comp
        sh = shots_by_track_id.get(tid, 0)
        sot = shots_on_target_by_track_id.get(tid, 0)
        if sot > sh:
            sot = sh
        xg_val = xg_by_track_id.get(tid, 0.0)
        gl = goals_by_track_id.get(tid, 0)
        tack = tackles_by_track_id.get(tid, 0)
        inter = interceptions_by_track_id.get(tid, 0)
        clear = clearances_by_track_id.get(tid, 0)
        drib = dribbles_by_track_id.get(tid, 0)
        kp = key_passes_by_track_id.get(tid, 0)
        mins = phys.get("minutes_played", 1)

        player_stats_dict = {
            "minutes_played": mins,
            "goals": gl,
            "assists": 0,
            "shots": sh,
            "shots_on_target": sot,
            "passes_attempted": att,
            "passes_completed": comp,
            "key_passes": kp,
            "dribbles_completed": drib,
            "tackles": tack,
            "interceptions": inter,
            "clearances": clear,
            "xg": xg_val,
            "distance_covered": phys.get("distance_m", 0.0),
        }
        ratings_by_track_id[tid] = compute_player_rating(player_stats_dict)

    player_stats_csv = io.StringIO()
    writer = csv.writer(player_stats_csv)
    writer.writerow([
        "track_id", "team", "jersey_number", "jersey_conf", "distance_m", "frames_tracked",
        "passes_completed", "passes_attempted", "pass_accuracy",
        "shots", "shots_on_target", "xg",
        "top_speed", "average_speed",
        "tackles", "interceptions", "clearances", "dribbles_completed", "key_passes",
        "rating"
    ])
    for identity in identities:
        tid = identity.master_id
        phys = physical_by_track_id.get(tid, {})
        dist_m = round(phys.get("distance_m", 0.0), 1)
        comp = passes_completed_by_track_id.get(tid, 0)
        att = passes_attempted_by_track_id.get(tid, comp)
        if att < comp:
            att = comp
        acc = round((comp / att) * 100, 1) if att > 0 else 0.0
        sh = shots_by_track_id.get(tid, 0)
        sot = shots_on_target_by_track_id.get(tid, 0)
        if sot > sh:
            sot = sh
        xg_val = round(xg_by_track_id.get(tid, 0.0), 3)
        top_spd = phys.get("top_speed_kmh", 0.0)
        avg_spd = phys.get("average_speed_kmh", 0.0)
        tack = tackles_by_track_id.get(tid, 0)
        inter = interceptions_by_track_id.get(tid, 0)
        clear = clearances_by_track_id.get(tid, 0)
        drib = dribbles_by_track_id.get(tid, 0)
        kp = key_passes_by_track_id.get(tid, 0)
        rat = ratings_by_track_id.get(tid, 6.0)

        writer.writerow([
            tid,
            identity.team.value,
            identity.jersey_number if identity.jersey_number is not None else "",
            "",
            dist_m,
            len(identity.trajectory),
            comp,
            att,
            acc,
            sh,
            sot,
            xg_val,
            top_spd,
            avg_spd,
            tack,
            inter,
            clear,
            drib,
            kp,
            rat,
        ])
    files.player_stats_csv.save(f"match_{match.id}_player_stats.csv", ContentFile(player_stats_csv.getvalue()), save=True)

    shots_csv = io.StringIO()
    shots_writer = csv.writer(shots_csv)
    shots_writer.writerow([
        "frame_idx", "minute", "track_id", "team", "pitch_x", "pitch_y",
        "target_goal_x", "target_goal_y", "distance_m", "speed_mps", "alignment", "xg",
        "is_on_target", "outcome"
    ])
    for s in recorded_shots:
        shots_writer.writerow([
            s["frame_idx"], s["minute"], s["track_id"], s["team"],
            s["pitch_x"], s["pitch_y"], s["target_goal_x"], s["target_goal_y"],
            s["distance_m"], s["speed_mps"], s["alignment"], s["xg"],
            s.get("is_on_target", False), s.get("outcome", "Off Target")
        ])
    files.shots_csv.save(f"match_{match.id}_shots.csv", ContentFile(shots_csv.getvalue()), save=True)

    # Save passes_csv
    passes_csv = io.StringIO()
    passes_writer = csv.writer(passes_csv)
    passes_writer.writerow([
        "frame_idx", "start_frame", "minute", "passer_track_id", "receiver_track_id",
        "passer_team", "receiver_team", "start_x", "start_y", "end_x", "end_y",
        "distance_m", "speed_mps", "is_completed"
    ])
    for p in recorded_passes:
        passes_writer.writerow([
            p["frame_idx"], p.get("start_frame", p["frame_idx"]), p["minute"],
            p["passer_track_id"], p["receiver_track_id"],
            p["passer_team"], p["receiver_team"],
            p["start_x"], p["start_y"], p["end_x"], p["end_y"],
            p["distance_m"], p.get("speed_mps", 0.0), p["is_completed"]
        ])
    files.passes_csv.save(f"match_{match.id}_passes.csv", ContentFile(passes_csv.getvalue()), save=True)

    # Save events_csv (consolidated match timeline)
    events_csv = io.StringIO()
    events_writer = csv.writer(events_csv)
    events_writer.writerow([
        "frame_idx", "minute", "event_type", "team", "track_id", "detail", "pitch_x", "pitch_y"
    ])
    all_events = []
    for s in recorded_shots:
        all_events.append({
            "frame_idx": s["frame_idx"],
            "minute": s["minute"],
            "event_type": "shot",
            "team": s["team"],
            "track_id": s["track_id"],
            "detail": f"Shot ({s['outcome']}) xG {s['xg']}",
            "pitch_x": s["pitch_x"],
            "pitch_y": s["pitch_y"],
        })
    for p in recorded_passes:
        all_events.append({
            "frame_idx": p["frame_idx"],
            "minute": p["minute"],
            "event_type": "pass" if p["is_completed"] else "interception",
            "team": p["passer_team"],
            "track_id": p["passer_track_id"],
            "detail": f"Pass to #{p['receiver_track_id']} ({p['distance_m']}m)",
            "pitch_x": p["start_x"],
            "pitch_y": p["start_y"],
        })
    for c in corner_events:
        all_events.append({
            "frame_idx": c["frame_idx"],
            "minute": c["minute"],
            "event_type": "corner",
            "team": c["team"],
            "track_id": c.get("track_id", ""),
            "detail": "Corner Kick",
            "pitch_x": c["pitch_x"],
            "pitch_y": c["pitch_y"],
        })
    from apps.matches.models import MatchGoal
    for g in MatchGoal.objects.filter(match=match):
        frame_approx = int((g.minute or 1) * 60 * 25)
        all_events.append({
            "frame_idx": frame_approx,
            "minute": g.minute or 1,
            "event_type": "goal",
            "team": "team_a" if g.team_id == match.home_team_id else "team_b",
            "track_id": "",
            "detail": f"Goal scored by {g.scorer.player_name if g.scorer else 'Unknown'}",
            "pitch_x": 52.5 if g.team_id == match.home_team_id else -52.5,
            "pitch_y": 0.0,
        })
    all_events.sort(key=lambda ev: ev["frame_idx"])
    for ev in all_events:
        events_writer.writerow([
            ev["frame_idx"], ev["minute"], ev["event_type"], ev["team"],
            ev["track_id"], ev["detail"], ev["pitch_x"], ev["pitch_y"]
        ])
    files.events_csv.save(f"match_{match.id}_events.csv", ContentFile(events_csv.getvalue()), save=True)

    try:
        from apps.analytics.services import sync_player_statistics
        sync_player_statistics(match)
    except Exception:
        logger.exception("Failed to sync PlayerStatistics for match=%s", match.id)

    # PDF report, regenerated in place now that real stats exist for this
    # match (initial calibration or a recalibration) — see
    # apps/reports/generator.py. Never fails the pitch-mapping task itself.
    try:
        from apps.reports.generator import generate_match_report
        generate_match_report(match)
    except Exception:
        pass

    match.status = Match.MatchStatus.COMPLETED
    match.processing_progress = 100
    match.save(update_fields=["status", "processing_progress", "updated_at"])

    # Ensure annotated replay video is refreshed asynchronously if this was a post-match recalibration
    try:
        render_match_video.delay(match.id)
    except Exception:
        logger.exception("Failed to dispatch render_match_video for match=%s", match.id)


@shared_task
def render_match_video(match_id):
    """Celery task to render or refresh an annotated replay video on-demand."""
    from apps.matches.models import Match
    from ai_engine.stage7_visualization.annotated_video import render_annotated_match_video
    try:
        match = Match.objects.get(pk=match_id)
        render_annotated_match_video(match)
    except Exception:
        logger.exception("Failed to render annotated video for match=%s", match_id)



def _auto_assign_unidentified_tracks(match, identities, cls_by_track=None):
    """
    Intelligent player-lineup assignment:
    1. Direct Jersey Match: If OCR identified a jersey number, match directly.
    2. Goalkeeper Match: If position == 'GK' or jersey == 1, match to GK-classified
       track or the deepest identity defending the goal.
    3. Position-Aware Pitch Matching: Map defenders, midfielders, and forwards to
       tracks according to pitch depth (from defending goal to attacking goal).
    4. Quality selection: Strictly assigns the longest, highest-quality pitch tracks
       to active starting players so no starter is assigned an empty/zero fragment.
    """
    from apps.matches.models import MatchLineup, TrackPlayerIdentification
    from ai_engine.utils.types import Team, ObjectClass
    from collections import Counter
    import numpy as np

    cls_by_track = cls_by_track or {}

    active_master_ids = {ident.master_id for ident in identities}
    # Delete stale assignments from prior runs whose track IDs no longer exist
    TrackPlayerIdentification.objects.filter(match=match).exclude(track_id__in=active_master_ids).delete()

    already_assigned_track_ids = set(
        TrackPlayerIdentification.objects.filter(match=match).values_list("track_id", flat=True)
    )
    already_claimed_lineup_ids = set(
        TrackPlayerIdentification.objects.filter(match=match).values_list("lineup_entry_id", flat=True)
    )
    existing_count_by_side = Counter(
        TrackPlayerIdentification.objects.filter(match=match)
        .values_list("lineup_entry__side", flat=True)
    )

    side_by_team = {Team.TEAM_A: MatchLineup.Side.HOME, Team.TEAM_B: MatchLineup.Side.AWAY}
    CAP_PER_SIDE = 11

    from ai_engine.stage5_pitch_mapping.formation_matcher import match_tracks_to_lineup_hungarian

    to_create = []

    for team_enum, side in side_by_team.items():
        team_identities = [
            ident for ident in identities
            if ident.team == team_enum
            and ident.master_id not in already_assigned_track_ids
            and getattr(ident, "cls", None) not in (ObjectClass.REFEREE, "referee")
            and ident.team not in (Team.REFEREE, Team.UNKNOWN)
            and cls_by_track.get(ident.master_id) not in (ObjectClass.REFEREE, "referee")
        ]
        if not team_identities:
            continue

        track_summaries = []
        for ident in team_identities:
            pts = list(ident.trajectory.values())
            x_vals = [p.x_m for p in pts]
            y_vals = [p.y_m for p in pts]
            med_x = float(np.median(x_vals)) if x_vals else 0.0
            med_y = float(np.median(y_vals)) if y_vals else 0.0
            cls_val = cls_by_track.get(ident.master_id)
            is_gk = (cls_val in (ObjectClass.GOALKEEPER, "goalkeeper")) or (getattr(ident, "cls", None) in (ObjectClass.GOALKEEPER, "goalkeeper")) or (abs(med_x) > 38.0)
            track_summaries.append({
                "track_id": ident.master_id,
                "median_x": med_x,
                "median_y": med_y,
                "duration": len(ident.trajectory),
                "is_gk": is_gk,
                "jersey_number": ident.jersey_number,
            })

        # Ensure exactly 1 GK track and 10 outfield tracks are selected per team
        gk_tracks = [t for t in track_summaries if t["is_gk"]]
        outfield_tracks = [t for t in track_summaries if not t["is_gk"]]

        gk_tracks.sort(key=lambda t: -t["duration"])
        outfield_tracks.sort(key=lambda t: -t["duration"])

        selected_tracks = []
        if gk_tracks:
            selected_tracks.append(gk_tracks[0])
        selected_tracks.extend(outfield_tracks[:10])

        if len(selected_tracks) < CAP_PER_SIDE:
            remaining_needed = CAP_PER_SIDE - len(selected_tracks)
            already_selected_ids = {t["track_id"] for t in selected_tracks}
            leftover = [t for t in track_summaries if t["track_id"] not in already_selected_ids]
            leftover.sort(key=lambda t: -t["duration"])
            selected_tracks.extend(leftover[:remaining_needed])

        track_summaries = selected_tracks

        available_lineups = list(
            match.lineups.filter(side=side)
            .exclude(id__in=already_claimed_lineup_ids)
            .order_by("-is_starting", "jersey_number")
        )
        if not available_lineups:
            continue

        formation_str = getattr(match, "home_formation", "4-3-3") if side == MatchLineup.Side.HOME else getattr(match, "away_formation", "4-3-3")
        formation_str = formation_str or "4-3-3"
        all_med_x = [t["median_x"] for t in track_summaries]
        defending_left = (np.mean(all_med_x) <= 0) if all_med_x else True

        assigned_pairs, _ = match_tracks_to_lineup_hungarian(
            track_summaries, available_lineups, formation_name=formation_str, defending_left=defending_left
        )

        for a in assigned_pairs:
            to_create.append(TrackPlayerIdentification(
                match=match,
                track_id=a["track_id"],
                lineup_entry_id=a["lineup_entry_id"],
                is_auto_assigned=True,
            ))
            already_claimed_lineup_ids.add(a["lineup_entry_id"])
            already_assigned_track_ids.add(a["track_id"])
            existing_count_by_side[side] += 1

    if to_create:
        TrackPlayerIdentification.objects.bulk_create(to_create)


def _hex_to_bgr(hex_color: str):
    """Converts a '#RRGGBB' hex string to a BGR array, matching
    ai_engine.stage3_team_reid.team_classifier's expected input format."""
    import numpy as np
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([b, g, r], dtype=np.float32)