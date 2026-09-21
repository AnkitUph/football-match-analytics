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
        # Sample up to a few frames per tracklet, not just the first —
        # a single unlucky frame (shadow, motion blur, partial
        # occlusion) shouldn't be able to flip an entire tracklet's team
        # on its own. VALIDATED NEED (real footage, match with a near-
        # white home kit): single-frame sampling let a large fraction of
        # home-team raw tracklets get misclassified as away — see
        # team_classifier.py's module docstring (FAILED APPROACH 3) for
        # the complementary fix (brightness-normalized distance) that
        # addresses the same root cause from the other side.
        MAX_COLOR_SAMPLES_PER_TRACK = 5
        for t in survivors:
            if t.cls and t.cls.value == "referee":
                from ai_engine.utils.types import Team
                t.team = Team.REFEREE
                referees.append(t)
                continue

            n = len(t.detections)
            sample_count = min(MAX_COLOR_SAMPLES_PER_TRACK, n)
            sample_indices = sorted(set(
                round(i * (n - 1) / max(sample_count - 1, 1)) for i in range(sample_count)
            ))

            sampled_colors = []
            for idx in sample_indices:
                det = t.detections[idx]
                cap2.set(cv2.CAP_PROP_POS_FRAMES, det.frame_idx)
                ok, frame = cap2.read()
                if not ok:
                    continue
                x1, y1, x2, y2 = map(int, det.bbox)
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                color = sample_torso_color(crop)
                if color is not None:
                    sampled_colors.append(color)

            if sampled_colors:
                colors[t.track_id] = np.median(np.array(sampled_colors), axis=0)
                valid.append(t)

        home_bgr = _hex_to_bgr(match.home_kit_color) if match.home_kit_color else None
        away_bgr = _hex_to_bgr(match.away_kit_color) if match.away_kit_color else None

        from ai_engine.stage3_team_reid.team_classifier import classify_teams_with_fallback
        team_by_track_id = classify_teams_with_fallback(colors, home_bgr, away_bgr, DEFAULT_CONFIG.team_reid)
        for t in valid:
            t.team = team_by_track_id[t.track_id]
        valid = valid + referees

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
        ball_by_frame = extract_ball_detections(dict(all_det))
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
            if p.x_m is not None:
                writer.writerow([p.frame_idx, p.x_m, p.y_m, p.interpolated])
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
            compute_pitch_mapping.delay(match.id)

        match.status = Match.MatchStatus.COMPLETED
        match.processing_progress = 100
        match.save(update_fields=["status", "processing_progress", "updated_at"])

        # PDF report, generated automatically (not behind a button) — see
        # apps/reports/generator.py. Uses whatever data exists right now
        # (real if compute_pitch_mapping already ran/is about to via the
        # calibration check above, Phase-5 dummy otherwise) — it gets
        # regenerated in place once real stats land, same
        # update_or_create-by-match pattern as everywhere else real data
        # replaces dummy data in this project. Report generation failing
        # should never fail the underlying match processing.
        try:
            from apps.reports.generator import generate_match_report
            generate_match_report(match)
        except Exception:
            pass

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
    from django.conf import settings
    from apps.matches.models import MatchCalibration
    from ai_engine.stage5_pitch_mapping.smart_assist import detect_pitch_keypoints

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

    saved_count = 0
    for i in range(num_anchors):
        window_start = int(total_frames * i / num_anchors)
        window_end = max(int(total_frames * (i + 1) / num_anchors), window_start + 1)

        candidate_offsets = sorted(set(
            min(window_start + int((window_end - window_start) * (j + 0.5) / samples_per_window), total_frames - 1)
            for j in range(samples_per_window)
        ))

        best_suggestions = None
        best_frame_idx = None
        best_score = -1

        for frame_idx in candidate_offsets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            suggestions = detect_pitch_keypoints(
                frame, settings.ROBOFLOW_API_KEY, confidence_threshold=0.5, max_points=12
            )
            if len(suggestions) < 4:
                continue

            # Score = sum of confidences across all detected points, NOT
            # real-world x-span. An earlier version of this function
            # scored by x-span specifically to avoid picking frames whose
            # confident points all cluster in one place -- but a real
            # side-by-side visual check (homography reprojection overlay
            # against match 10's known-good manual calibration on this
            # same clip) showed x-span maximization backfired: it
            # actively prefers spurious far-field detections, since a
            # WRONG "right box corner" detection sitting near a goal line
            # inflates x-span just as effectively as a real one would.
            # Confirmed concretely -- the overlay showed a confidently
            # labeled "right_box_top"/"right_corner_top"/"right_penalty_spot"
            # cluster drawn on empty grass with no real box or corner
            # anywhere near it, while this same anchor's halfway-line/
            # center-circle points (a well-validated cluster, confirmed
            # via reprojection tests on two independent clips earlier
            # this session) were correctly aligned. Summed confidence is
            # a more neutral proxy for "this is a clean, unambiguous
            # frame" -- it doesn't specifically go looking for distant
            # landmark types the way x-span does.
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
    from ai_engine.stage5_pitch_mapping.homography import image_point_to_pitch
    from ai_engine.stage5_pitch_mapping.identity_association import match_tracklets_within_shot
    from ai_engine.stage6_event_detection.events import detect_possession, detect_passes, detect_shots, estimate_shot_xg
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

        homography_by_frame[calibration.calibration_frame] = homography_tracker.current_H.copy()
        cap.set(cv2.CAP_PROP_POS_FRAMES, calibration.calibration_frame + 1)
        for frame_idx in range(calibration.calibration_frame + 1, segment_end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            H = homography_tracker.update(frame)
            if H is not None:
                homography_by_frame[frame_idx] = H.copy()

    cap.release()

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
            ball_trajectory.append(BallTrajectoryPoint(
                frame_idx=int(row["frame_idx"]),
                x_m=float(row["x_px"]),
                y_m=float(row["y_px"]),
                interpolated=row["interpolated"].strip().lower() == "true",
            ))

    # --- Rewrite player_tracking_csv with pitch_x/pitch_y from the NEW calibration ---
    # jersey_number/jersey_conf are carried through unchanged — this task
    # never re-runs OCR, only re-runs the pitch mapping.
    player_csv = io.StringIO()
    writer = csv.writer(player_csv)
    writer.writerow(["track_id", "frame_idx", "team", "class", "x1", "y1", "x2", "y2", "conf", "jersey_number", "jersey_conf", "pitch_x", "pitch_y"])
    for t in valid:
        jersey_number = t.jersey_number if t.jersey_number is not None else ""
        jersey_conf = round(t.jersey_number_conf, 3) if t.jersey_number is not None else ""
        for det in t.detections:
            pitch_x = pitch_y = ""
            H = homography_by_frame.get(det.frame_idx)
            if H is not None:
                fx, fy = (det.x1 + det.x2) / 2, det.y2
                pt = image_point_to_pitch(fx, fy, H)
                pitch_x, pitch_y = pt.x_m, pt.y_m
            writer.writerow([t.track_id, det.frame_idx, t.team.value, det.cls.value, det.x1, det.y1, det.x2, det.y2, det.conf, jersey_number, jersey_conf, pitch_x, pitch_y])
    files.player_tracking_csv.save(f"match_{match.id}_players.csv", ContentFile(player_csv.getvalue()), save=True)

    # --- Stage 5-6: possession/pass/shot detection + TeamStatistics ---
    # (Same logic as the old inline block that used to live in
    # process_match — just fed from the reconstructed valid/
    # ball_trajectory above instead of a live detection run.)
    ball_pitch_trajectory = []
    for p in ball_trajectory:
        if p.x_m is None or p.frame_idx not in homography_by_frame:
            continue
        pt = image_point_to_pitch(p.x_m, p.y_m, homography_by_frame[p.frame_idx])
        ball_pitch_trajectory.append(
            BallTrajectoryPoint(frame_idx=p.frame_idx, x_m=pt.x_m, y_m=pt.y_m, interpolated=p.interpolated)
        )

    # --- Merge fragmented raw tracklets into one-per-real-player first ---
    # VALIDATED FINDING (see project handoff): raw BoT-SORT tracklets
    # fragment heavily — ~37 fragments for what should be ~22 real
    # players/ref on test_11.mp4. Building per-player output (distance,
    # trajectory, the identify-players crops below) from `valid` directly
    # would silently split one real player's data across 2-3 rows. Fixed
    # here by stitching first and using the STITCHED tracklet's own
    # track_id as the stable per-match player id everywhere downstream.
    #
    # Deliberately NOT running assign_tracklets_to_gallery here: that
    # function's whole job is preserving identity across a camera CUT
    # (Stage 2.5), which doesn't apply — this whole clip is one
    # continuous shot (shot_id=0 throughout, Stage 2.5 deferred). Running
    # it anyway would only add the ≤22/≤11-per-team cap, which
    # match_tracklets_within_shot's own duration filter already
    # accomplishes well enough for a single shot.
    stitched = match_tracklets_within_shot(valid, DEFAULT_CONFIG.pitch_mapping, fps=25.0)

    identities = []
    for t in stitched:
        traj = {}
        for det in t.detections:
            if det.frame_idx not in homography_by_frame:
                continue
            fx, fy = (det.x1 + det.x2) / 2, det.y2
            pt = image_point_to_pitch(fx, fy, homography_by_frame[det.frame_idx])
            traj[det.frame_idx] = PitchPoint(x_m=pt.x_m, y_m=pt.y_m)
        if traj:
            identities.append(MasterIdentity(
                master_id=t.track_id, team=t.team, reid_embedding=[],
                jersey_number=t.jersey_number, trajectory=traj,
            ))

    # Fallback identification: guess a name for any tracked player who
    # still has none, so every identity shows up as SOMEONE in the
    # results/report rather than staying anonymous. Created with
    # is_auto_assigned=False (treated as confirmed immediately, per
    # project decision) — see that function's docstring below. A human
    # can still correct any of these via /identify/; this never
    # overwrites an existing assignment from an earlier run.
    try:
        _auto_assign_unidentified_tracks(match, identities)
    except Exception:
        pass

    possession_events = detect_possession(ball_pitch_trajectory, identities, DEFAULT_CONFIG.event_detection)
    pass_events = detect_passes(possession_events, identities)
    # Checks BOTH goals now — see events.py's detect_shots docstring.
    # Previously hardcoded to (52.5, 0.0), the right goal only, which
    # made every left-goal shot structurally invisible regardless of
    # detection quality. Found via real-data review, not testing —
    # this project's footage happened not to have a left-goal shot in
    # any clip validated so far, so the bug never surfaced in output.
    shot_events = detect_shots(ball_pitch_trajectory, ((52.5, 0.0), (-52.5, 0.0)))

    team_by_master_id = {i.master_id: i.team for i in identities}

    # FOUND VIA REAL DATA: a short tracked segment (~190 frames from a
    # single calibration anchor) produced a degenerate 100%/0% possession
    # split. Root cause was here — the LAST possession event in the list
    # had its end_frame fall back to its OWN start_frame (a zero-length
    # interval, floored to 1 frame by the max() below), while every
    # EARLIER event correctly got its real duration up to the NEXT
    # event's frame. On a short clip with few possession-change events,
    # that one truncated interval can dominate the total, producing
    # exactly this kind of skewed/degenerate split. Fix: the last event's
    # interval should extend to the actual last analyzed frame (the last
    # frame with a valid ball pitch position), not collapse to nothing.
    valid_ball_frames = [p.frame_idx for p in ball_pitch_trajectory if p.x_m is not None]
    last_tracked_frame = max(valid_ball_frames) if valid_ball_frames else 0

    possession_frame_counts = {Team.TEAM_A: 0, Team.TEAM_B: 0}
    for i, event in enumerate(possession_events):
        start_frame = event.frame_idx
        end_frame = (
            possession_events[i + 1].frame_idx if i + 1 < len(possession_events) else last_tracked_frame
        )
        holder_team = team_by_master_id.get(event.player_master_id)
        if holder_team in possession_frame_counts:
            possession_frame_counts[holder_team] += max(end_frame - start_frame, 1)
    total_possession_frames = sum(possession_frame_counts.values()) or 1

    # Diagnostic, not behavior — helps distinguish "genuinely few
    # possession changes in a short/scrappy segment" (expected, see
    # detect_possession's own docstring on contested-ball flicker) from
    # an actual detection problem, without needing a debugger.
    logger.info(
        "compute_pitch_mapping match=%s: %d possession events, %d pass events, last_tracked_frame=%s",
        match.id, len(possession_events), len(pass_events), last_tracked_frame,
    )
    # TEMPORARY, verbose — remove once the 100/0 possession investigation
    # is closed out. Dumps every event so we can see the raw picture
    # instead of just aggregate counts.
    for i, event in enumerate(possession_events):
        logger.info(
            "  possession_event[%d]: frame=%s player_master_id=%s team=%s",
            i, event.frame_idx, event.player_master_id, team_by_master_id.get(event.player_master_id),
        )

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
    xg_by_team = {Team.TEAM_A: 0.0, Team.TEAM_B: 0.0}
    shots_by_track_id = defaultdict(int)
    xg_by_track_id = defaultdict(float)
    for e in shot_events:
        ball_pos = ball_pitch_by_frame.get(e.frame_idx)
        if ball_pos is None:
            continue

        closest_identity, closest_dist = None, float("inf")
        for identity in identities:
            pos = identity.trajectory.get(e.frame_idx)
            if pos is None:
                continue
            dist = ((pos.x_m - ball_pos.x_m) ** 2 + (pos.y_m - ball_pos.y_m) ** 2) ** 0.5
            if dist < closest_dist:
                closest_dist = dist
                closest_identity = identity

        if closest_identity is None or closest_identity.team not in shots_by_team:
            continue

        shot_xg = estimate_shot_xg(e.metadata["origin_distance_m"], e.metadata["alignment"])

        shots_by_team[closest_identity.team] += 1
        xg_by_team[closest_identity.team] += shot_xg
        shots_by_track_id[closest_identity.master_id] += 1
        xg_by_track_id[closest_identity.master_id] += shot_xg

    # Max plausible human sprint speed, generously above a real peak
    # (~12.4 m/s for an elite sprinter at full tilt) — used to reject
    # single-frame tracking glitches, not to model real player speed.
    # FOUND VIA REAL DATA: a tracked identity on real footage summed to
    # 1259m over 119 frames (~4.8s at 25fps) = ~264 m/s, obviously
    # impossible. Root cause: team_distance_m/identity_distance_m summed
    # raw frame-to-frame displacement with NO sanity bound — a single
    # bad per-frame homography solve (see homography_tracker.py: each
    # frame's homography is resolved completely fresh, with no
    # consistency check against its neighbors — that's what avoids
    # chained drift, but it also means nothing catches a one-off bad
    # solve) can silently inflate distance covered. detect_shots already
    # guards the ball the same way (max_plausible_speed_mps) — this is
    # the same pattern applied to player movement. SKIPS the impossible
    # segment entirely (doesn't cap it to a plausible-but-still-invented
    # value) — same "honest gap over fabricated number" pattern as
    # Stage 4's ball-interpolation bounds guard.
    MAX_PLAYER_SPEED_MPS = 12.0

    def _plausible_segment_distance_m(p1, p2, dt_frames):
        if dt_frames <= 0:
            return 0.0
        dt_sec = dt_frames / 25.0
        dist = ((p2.x_m - p1.x_m) ** 2 + (p2.y_m - p1.y_m) ** 2) ** 0.5
        if dist / dt_sec > MAX_PLAYER_SPEED_MPS:
            return 0.0  # tracking glitch, not real movement — don't count it
        return dist

    def team_distance_m(team_enum):
        total = 0.0
        for identity in identities:
            if identity.team != team_enum:
                continue
            frames = sorted(identity.trajectory.keys())
            for f1, f2 in zip(frames, frames[1:]):
                total += _plausible_segment_distance_m(identity.trajectory[f1], identity.trajectory[f2], f2 - f1)
        return total

    # TEAM_A is "closer to home_kit_color" per classify_team_by_known_colors
    # from Stage 3, which already ran (and is baked into the CSV's team
    # column) — no need to redo that classification here.
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
                "xg": round(xg_by_team[team_enum], 2),
            },
        )

    # --- Real per-player distance, keyed by jersey_number for matching ---
    # against MatchLineup at request time (see views.py). NOT a full
    # PlayerStatistics row — no goals/passes/etc. per player exist yet,
    # this is deliberately just distance + identity, kept in its own CSV
    # (MatchFiles.player_stats_csv, previously unused) rather than
    # conflated with the still-fully-dummy per-player stats table.
    def identity_distance_m(identity):
        frames = sorted(identity.trajectory.keys())
        total = 0.0
        for f1, f2 in zip(frames, frames[1:]):
            total += _plausible_segment_distance_m(identity.trajectory[f1], identity.trajectory[f2], f2 - f1)
        return total

    player_stats_csv = io.StringIO()
    writer = csv.writer(player_stats_csv)
    writer.writerow(["track_id", "team", "jersey_number", "jersey_conf", "distance_m", "frames_tracked", "passes_completed", "shots", "xg"])
    for identity in identities:
        writer.writerow([
            identity.master_id,
            identity.team.value,
            identity.jersey_number if identity.jersey_number is not None else "",
            "",  # per-identity conf not carried past Tracklet -> MasterIdentity today; jersey_number's presence already implies it cleared the aggregation bar
            round(identity_distance_m(identity), 1),
            len(identity.trajectory),
            passes_completed_by_track_id.get(identity.master_id, 0),
            shots_by_track_id.get(identity.master_id, 0),
            round(xg_by_track_id.get(identity.master_id, 0.0), 3),
        ])
    files.player_stats_csv.save(f"match_{match.id}_player_stats.csv", ContentFile(player_stats_csv.getvalue()), save=True)

    # PDF report, regenerated in place now that real stats exist for this
    # match (initial calibration or a recalibration) — see
    # apps/reports/generator.py. Never fails the pitch-mapping task itself.
    try:
        from apps.reports.generator import generate_match_report
        generate_match_report(match)
    except Exception:
        pass


def _auto_assign_unidentified_tracks(match, identities):
    """
    Fallback identification: for any tracked player-side identity that
    still has no TrackPlayerIdentification, randomly pair it with an
    unclaimed same-side MatchLineup entry — so every tracked player shows
    up as SOME named lineup player rather than staying anonymous. The
    guess may well be wrong; that's fine for this project (not
    production-grade).

    NOTE (intentional per-project choice): these rows are created with
    is_auto_assigned=False — i.e. treated as CONFIRMED from the moment
    they're created, not flagged as an unreviewed guess. This means
    every tracked player shows real per-player data (distance, passes)
    with the green "(tracked)" label immediately after processing,
    rather than the yellow "(tracked, unverified)" one, and does NOT
    get the "always shown past the cap" / guess-vs-confirmed visual
    split on /identify/ that is_auto_assigned=True used to trigger there
    too — a human (Ankit) reviews and corrects any wrong jersey/identity
    assignments manually via /identify/ without relying on that
    distinction. This is a single-line behavioral choice, not a change
    to the matching/ranking logic below, which is unchanged.

    "Random" here is match+side-seeded, not truly random each call — so
    re-running compute_pitch_mapping (e.g. after a recalibration) doesn't
    reshuffle guesses that already exist. This function only ever fills
    gaps: any track_id or lineup_entry already claimed by ANY existing
    row is left untouched. A human's correction on /identify/ can
    therefore never be silently undone by a later recalibration.

    If there are more tracked identities than lineup entries on a side
    (expected — residual tracklet fragmentation typically outnumbers the
    real ~11 players), the extras simply stay unassigned; there's no
    lineup slot left to guess for them.
    """
    from apps.matches.models import MatchLineup, TrackPlayerIdentification
    from ai_engine.utils.types import Team

    already_assigned_track_ids = set(
        TrackPlayerIdentification.objects.filter(match=match).values_list("track_id", flat=True)
    )
    already_claimed_lineup_ids = set(
        TrackPlayerIdentification.objects.filter(match=match).values_list("lineup_entry_id", flat=True)
    )
    # Per-side count of EXISTING assignments (guessed or confirmed) —
    # the cap below is a TOTAL per side, not "how many new guesses this
    # call adds". Missing this the first time around let a side with
    # some already-confirmed rows exceed 10 total once fresh guesses
    # were added on top, since the guess count alone was capped at 10
    # without checking what was already there.
    from collections import Counter
    existing_count_by_side = Counter(
        TrackPlayerIdentification.objects.filter(match=match)
        .values_list("lineup_entry__side", flat=True)
    )

    side_by_team = {Team.TEAM_A: MatchLineup.Side.HOME, Team.TEAM_B: MatchLineup.Side.AWAY}
    CAP_PER_SIDE = 10

    # Most-seen first, not arbitrary track_id order — len(trajectory) is
    # this identity's pitch-mapped frame count, a direct proxy for how
    # much of the match it was actually visible/tracked for. A short
    # noise fragment shouldn't be able to out-rank a well-tracked real
    # player just because its track_id happens to sort lower.
    duration_by_master_id = {identity.master_id: len(identity.trajectory) for identity in identities}

    to_create = []
    for team_enum, side in side_by_team.items():
        team_track_ids = sorted(
            {
                identity.master_id for identity in identities
                if identity.team == team_enum and identity.master_id not in already_assigned_track_ids
            },
            key=lambda tid: -duration_by_master_id.get(tid, 0),
        )
        if not team_track_ids:
            continue

        available_lineup_ids = list(
            match.lineups.filter(side=side)
            .exclude(id__in=already_claimed_lineup_ids)
            .values_list("id", flat=True)
        )
        if not available_lineup_ids:
            continue

        rng = random.Random(f"{match.public_id}-{side}")
        rng.shuffle(available_lineup_ids)

        # Hard cap: TOTAL identities per side (existing + new) never
        # exceeds CAP_PER_SIDE — keeps the identify-players page to a
        # clean, professor-presentable set instead of every residual
        # fragment (residual fragmentation typically outnumbers the
        # real ~11 players — see this function's earlier notes).
        slots_remaining = max(CAP_PER_SIDE - existing_count_by_side.get(side, 0), 0)
        for track_id, lineup_id in zip(team_track_ids[:slots_remaining], available_lineup_ids):
            to_create.append(TrackPlayerIdentification(
                match=match, track_id=track_id, lineup_entry_id=lineup_id, is_auto_assigned=False,
            ))
            already_claimed_lineup_ids.add(lineup_id)
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