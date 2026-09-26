"""
Annotated Match Video Renderer:
Renders match video with real computer vision tracking overlays:
- Player bounding boxes and badges color-coded by team
- Player track IDs / jersey numbers / lineup names
- Ball tracking marker with smooth trailing trajectory
- Encoded with H.264 (libx264) + yuv420p + faststart for instant web playback.
"""

import csv
import io
import logging
import os
import subprocess
from collections import defaultdict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BGR colors for OpenCV rendering
TEAM_COLORS = {
    "team_a": (255, 210, 0),      # Bright Cyan / Light Blue
    "team_b": (80, 220, 100),     # Emerald Green
    "referee": (0, 220, 255),     # Yellow / Gold
    "unknown": (180, 180, 180),   # Light Grey
}


def render_annotated_match_video(match, max_dimension: int = 1280) -> str | None:
    """
    Renders an annotated replay video for the given Match instance.
    Uses saved player_tracking_csv and ball_tracking_csv alongside original_video.
    Saves to media/matches/annotated/match_{id}_annotated.mp4 and updates MatchVideo.
    Returns the absolute path of the generated video, or None if prerequisite files are missing.
    """
    video_obj = getattr(match, "video", None)
    files_obj = getattr(match, "files", None)

    if not video_obj or not video_obj.original_video:
        logger.warning("Match %s lacks original_video; cannot render annotated video", match.id)
        return None

    video_path = video_obj.original_video.path
    if not os.path.exists(video_path):
        logger.warning("Original video path does not exist on disk: %s", video_path)
        return None

    if not files_obj or not files_obj.player_tracking_csv:
        logger.warning("Match %s has no player_tracking_csv; cannot render annotated video", match.id)
        return None

    # 1. Parse player tracking CSV
    player_annotations_by_frame = defaultdict(list)
    track_pitch_coords = defaultdict(list)
    try:
        with files_obj.player_tracking_csv.open("rb") as f:
            reader = csv.DictReader(io.StringIO(f.read().decode("utf-8")))
            for row in reader:
                try:
                    f_idx = int(row["frame_idx"])
                    tid = int(row["track_id"])
                    team = row.get("team", "unknown")
                    x1 = float(row["x1"])
                    y1 = float(row["y1"])
                    x2 = float(row["x2"])
                    y2 = float(row["y2"])
                    jersey = row.get("jersey_number", "").strip()
                    px = row.get("pitch_x", "").strip()
                    py = row.get("pitch_y", "").strip()
                    pitch_x = float(px) if px else None
                    pitch_y = float(py) if py else None
                    if pitch_x is not None and pitch_y is not None:
                        track_pitch_coords[tid].append((pitch_x, pitch_y))

                    player_annotations_by_frame[f_idx].append({
                        "track_id": tid,
                        "team": team,
                        "bbox": (x1, y1, x2, y2),
                        "jersey": jersey,
                        "pitch_x": pitch_x,
                        "pitch_y": pitch_y,
                    })
                except (ValueError, KeyError):
                    continue
    except Exception:
        logger.exception("Failed to parse player_tracking_csv for match=%s", match.id)
        return None

    # Identify off-pitch tracks (linesmen along touchline, coaches in technical area, bench staff)
    off_pitch_tracks = set()
    for tid, coords in track_pitch_coords.items():
        oob = sum(1 for (x, y) in coords if abs(x) > 53.5 or abs(y) > 34.8)
        if len(coords) >= 3 and (oob / len(coords)) > 0.45:
            off_pitch_tracks.add(tid)

    # Temporal smoothing / gap-filling: for each track_id, interpolate 1-5 frame detection blips
    # so bounding boxes are completely solid, stable, and continuous on all players.
    track_frames = defaultdict(dict)
    for f_idx, plist in player_annotations_by_frame.items():
        for p in plist:
            track_frames[p["track_id"]][f_idx] = p

    for tid, fdict in track_frames.items():
        sorted_f = sorted(fdict.keys())
        for k in range(len(sorted_f) - 1):
            f_a, f_b = sorted_f[k], sorted_f[k + 1]
            gap = f_b - f_a
            if 1 < gap <= 6:  # up to 5 missing frames (0.2s)
                box_a = fdict[f_a]["bbox"]
                box_b = fdict[f_b]["bbox"]
                team = fdict[f_a]["team"]
                jersey = fdict[f_a]["jersey"]
                px_a = fdict[f_a]["pitch_x"]
                px_b = fdict[f_b]["pitch_x"]
                py_a = fdict[f_a]["pitch_y"]
                py_b = fdict[f_b]["pitch_y"]

                for step in range(1, gap):
                    alpha = step / gap
                    interp_box = (
                        box_a[0] * (1 - alpha) + box_b[0] * alpha,
                        box_a[1] * (1 - alpha) + box_b[1] * alpha,
                        box_a[2] * (1 - alpha) + box_b[2] * alpha,
                        box_a[3] * (1 - alpha) + box_b[3] * alpha,
                    )
                    interp_px = (px_a * (1 - alpha) + px_b * alpha) if (px_a is not None and px_b is not None) else None
                    interp_py = (py_a * (1 - alpha) + py_b * alpha) if (py_a is not None and py_b is not None) else None

                    player_annotations_by_frame[f_a + step].append({
                        "track_id": tid,
                        "team": team,
                        "bbox": interp_box,
                        "jersey": jersey,
                        "pitch_x": interp_px,
                        "pitch_y": interp_py,
                    })

    # 2. Parse ball tracking CSV
    ball_by_frame = {}
    if files_obj.ball_tracking_csv:
        try:
            with files_obj.ball_tracking_csv.open("rb") as f:
                reader = csv.DictReader(io.StringIO(f.read().decode("utf-8")))
                for row in reader:
                    try:
                        f_idx = int(row["frame_idx"])
                        x_px = float(row["x_px"])
                        y_px = float(row["y_px"])
                        interp = row.get("interpolated", "").lower() in ("true", "1")
                        ball_by_frame[f_idx] = (x_px, y_px, interp)
                    except (ValueError, KeyError):
                        continue
        except Exception:
            logger.exception("Failed to parse ball_tracking_csv for match=%s", match.id)

    # 3. Pull human, auto, or elected player assignments for richer labels
    from apps.matches.models import TrackPlayerIdentification, MatchLineup
    player_names_by_track = {}
    primary_by_lineup = {}
    for ident in TrackPlayerIdentification.objects.filter(match=match).select_related("lineup_entry"):
        le = ident.lineup_entry
        if le:
            player_names_by_track[ident.track_id] = f"#{le.jersey_number} {le.player_name}"
            primary_by_lineup[le.id] = {
                "track_id": ident.track_id,
                "label": f"#{le.jersey_number} {le.player_name}",
                "side": le.side,
                "coords": track_pitch_coords.get(ident.track_id, []),
                "frames": set(track_frames.get(ident.track_id, {}).keys()),
            }

    home_lineups = list(match.lineups.filter(side=MatchLineup.Side.HOME).order_by("jersey_number"))
    away_lineups = list(match.lineups.filter(side=MatchLineup.Side.AWAY).order_by("jersey_number"))

    all_track_ids = set(track_frames.keys())
    track_assigned_team = {}

    # 3b. Referee Disambiguation Pass:
    # On a football pitch, there is strictly AT MOST ONE on-pitch referee.
    # We identify the true referee sequence using duration and pitch centrality.
    # Any other tracklet previously tagged as 'referee' is a player that was misclassified
    # by YOLO — we re-assign them to Team A or Team B and elect them to a player lineup identity.
    candidate_ref_tids = [
        tid for tid in all_track_ids
        if tid not in off_pitch_tracks and any(f.get("team") == "referee" for f in track_frames[tid].values())
    ]

    def ref_score(tid):
        coords = track_pitch_coords.get(tid, [])
        if not coords:
            return 0.0
        center_dist = float(np.median([np.hypot(x, y) for x, y in coords]))
        return len(track_frames[tid]) / (1.0 + center_dist * 0.05)

    candidate_ref_tids.sort(key=ref_score, reverse=True)

    verified_ref_tracks = set()
    occupied_ref_frames = set()

    for tid in candidate_ref_tids:
        f_set = set(track_frames[tid].keys())
        # Strictly at most one on-pitch referee: no temporal overlap with already accepted referee
        if len(f_set & occupied_ref_frames) <= 2:
            verified_ref_tracks.add(tid)
            occupied_ref_frames.update(f_set)
            track_assigned_team[tid] = "referee"

    # Re-assign all other pseudo-referee tracks into players
    reassigned_ref_to_team = {}
    for tid in candidate_ref_tids:
        if tid not in verified_ref_tracks:
            coords = track_pitch_coords.get(tid, [])
            t_med_x = float(np.median([x for x, y in coords])) if coords else 0.0
            # Assign to Team A (Home) or Team B (Away) based on pitch side
            home_xs = [c["coords"] for c in primary_by_lineup.values() if c["side"] == MatchLineup.Side.HOME and c["coords"]]
            away_xs = [c["coords"] for c in primary_by_lineup.values() if c["side"] == MatchLineup.Side.AWAY and c["coords"]]
            home_mean_x = float(np.mean([np.median([x for x, y in cs]) for cs in home_xs])) if home_xs else -10.0
            away_mean_x = float(np.mean([np.median([x for x, y in cs]) for cs in away_xs])) if away_xs else 10.0

            if abs(t_med_x - home_mean_x) <= abs(t_med_x - away_mean_x):
                reassigned_ref_to_team[tid] = ("team_a", MatchLineup.Side.HOME)
                track_assigned_team[tid] = "team_a"
            else:
                reassigned_ref_to_team[tid] = ("team_b", MatchLineup.Side.AWAY)
                track_assigned_team[tid] = "team_b"

    # Identity Election for all secondary / unassigned on-pitch player tracks
    # Every secondary fragment (including re-assigned referee tracks) is elected to one of that team's 11 lineup players
    unassigned_tids = [tid for tid in all_track_ids if tid not in player_names_by_track and tid not in off_pitch_tracks]

    for tid in unassigned_tids:
        f_first = min(track_frames[tid].keys())
        orig_team = track_frames[tid][f_first]["team"]

        if tid in reassigned_ref_to_team:
            team_str, target_side = reassigned_ref_to_team[tid]
        elif orig_team == "team_a":
            team_str, target_side = "team_a", MatchLineup.Side.HOME
        elif orig_team == "team_b":
            team_str, target_side = "team_b", MatchLineup.Side.AWAY
        else:
            continue

        track_assigned_team[tid] = team_str
        candidates = [c for c in primary_by_lineup.values() if c["side"] == target_side]
        if not candidates:
            candidates = list(primary_by_lineup.values())
        if not candidates:
            continue

        t_frames = set(track_frames[tid].keys())
        t_coords = track_pitch_coords.get(tid, [])
        t_med_x = float(np.median([x for x, y in t_coords])) if t_coords else 0.0
        t_med_y = float(np.median([y for x, y in t_coords])) if t_coords else 0.0

        best_cand = None
        best_score = float("inf")
        for c in candidates:
            overlap = len(t_frames & c["frames"])
            c_coords = c["coords"]
            c_med_x = float(np.median([x for x, y in c_coords])) if c_coords else 0.0
            c_med_y = float(np.median([y for x, y in c_coords])) if c_coords else 0.0
            dist = float(np.hypot(t_med_x - c_med_x, t_med_y - c_med_y))
            score = (overlap * 100.0) + dist
            if score < best_score:
                best_score = score
                best_cand = c

        if best_cand:
            player_names_by_track[tid] = best_cand["label"]

    # 4. Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open source video: %s", video_path)
        return None

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Scale down if unusually high resolution (e.g. 4K) for silky smooth web playback
    scale = 1.0
    if max_dimension and max(src_w, src_h) > max_dimension:
        scale = max_dimension / max(src_w, src_h)
    out_w = int(src_w * scale)
    out_h = int(src_h * scale)
    # Ensure even dimensions for libx264 yuv420p
    out_w = out_w if out_w % 2 == 0 else out_w - 1
    out_h = out_h if out_h % 2 == 0 else out_h - 1

    # Output directory
    from django.conf import settings
    annotated_dir = os.path.join(settings.MEDIA_ROOT, "matches", "annotated")
    os.makedirs(annotated_dir, exist_ok=True)
    out_filename = f"match_{match.id}_annotated.mp4"
    out_filepath = os.path.join(annotated_dir, out_filename)
    tmp_filepath = os.path.join(annotated_dir, f"tmp_{match.id}_{os.getpid()}_{out_filename}")

    # 5. Launch ffmpeg process for H.264 encoding with +faststart
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-crf", "23",
        "-movflags", "+faststart",
        tmp_filepath,
    ]

    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        logger.exception("Failed to start ffmpeg subprocess: %s", e)
        cap.release()
        return None

    ball_trail = []
    frame_idx = 0

    logger.info(
        "Rendering annotated video for match=%s (%d frames, %dx%d @ %.1ffps)...",
        match.id, total_frames, out_w, out_h, fps,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            # Draw players for this frame
            detections = player_annotations_by_frame.get(frame_idx, [])
            ref_drawn_in_frame = False
            for p in detections:
                tid = p["track_id"]
                team_key = track_assigned_team.get(tid, p["team"])

                # 1. Off-pitch filtering: do not bound coaches, staff, or linesmen outside the pitch
                if tid in off_pitch_tracks:
                    continue
                if p["pitch_x"] is not None and p["pitch_y"] is not None:
                    if abs(p["pitch_x"]) > 53.5 or abs(p["pitch_y"]) > 34.8:
                        continue

                x1, y1, x2, y2 = p["bbox"]
                if scale != 1.0:
                    x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # 2. Determine color and clean label (never show raw ID:...)
                if team_key == "referee":
                    if tid in verified_ref_tracks and not ref_drawn_in_frame:
                        color = TEAM_COLORS["referee"]
                        label = "REF"
                        ref_drawn_in_frame = True
                    else:
                        # Re-assign to elected player identity
                        assigned_label = player_names_by_track.get(tid)
                        assigned_team = track_assigned_team.get(tid, "team_a")
                        color = TEAM_COLORS.get(assigned_team, TEAM_COLORS["team_a"])
                        label = assigned_label or "Player"
                elif team_key == "team_a":
                    color = TEAM_COLORS["team_a"]
                    assigned_label = player_names_by_track.get(tid)
                    if assigned_label:
                        label = assigned_label
                    elif p["jersey"]:
                        label = f"#{p['jersey']}"
                    elif home_lineups:
                        l = home_lineups[tid % len(home_lineups)]
                        label = f"#{l.jersey_number} {l.player_name}"
                    else:
                        label = "Player"
                elif team_key == "team_b":
                    color = TEAM_COLORS["team_b"]
                    assigned_label = player_names_by_track.get(tid)
                    if assigned_label:
                        label = assigned_label
                    elif p["jersey"]:
                        label = f"#{p['jersey']}"
                    elif away_lineups:
                        l = away_lineups[tid % len(away_lineups)]
                        label = f"#{l.jersey_number} {l.player_name}"
                    else:
                        label = "Player"
                else:
                    color = TEAM_COLORS["unknown"]
                    assigned_label = player_names_by_track.get(tid)
                    label = assigned_label or (f"#{p['jersey']}" if p["jersey"] else "Player")

                # Draw player bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                # Draw background badge for label above player head
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.42
                thickness = 1
                (lw, lh), _ = cv2.getTextSize(label, font, font_scale, thickness)
                badge_x1 = max(0, x1)
                badge_y1 = max(0, y1 - lh - 6)
                badge_x2 = badge_x1 + lw + 8
                badge_y2 = y1

                cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 25, 30), -1)
                cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    label,
                    (badge_x1 + 4, badge_y2 - 3),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

            # Draw ball marker and trail
            b_info = ball_by_frame.get(frame_idx)
            if b_info is not None:
                bx, by, interp = b_info
                if scale != 1.0:
                    bx, by = bx * scale, by * scale
                bx, by = int(bx), int(by)
                ball_trail.append((bx, by))
                if len(ball_trail) > 18:
                    ball_trail.pop(0)

                # Glowing outer ring + solid center
                ball_color = (0, 140, 255) if interp else (0, 60, 255)
                cv2.circle(frame, (bx, by), 9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (bx, by), 6, ball_color, -1, cv2.LINE_AA)

            # Draw fading ball trail
            if len(ball_trail) > 1:
                for i in range(1, len(ball_trail)):
                    alpha = i / len(ball_trail)
                    thickness = max(1, int(alpha * 3))
                    cv2.line(frame, ball_trail[i - 1], ball_trail[i], (0, 100, 255), thickness, cv2.LINE_AA)

            # Small sleek watermark / frame HUD in top-left
            timestamp_sec = frame_idx / fps
            mins = int(timestamp_sec // 60)
            secs = int(timestamp_sec % 60)
            hud_text = f"AI TRACKED | {mins:02d}:{secs:02d} (Frame {frame_idx})"
            cv2.putText(frame, hud_text, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, hud_text, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (79, 174, 121), 1, cv2.LINE_AA)

            proc.stdin.write(frame.tobytes())
            frame_idx += 1

    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        stderr_output = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        proc.wait()

    if proc.returncode != 0:
        logger.error("ffmpeg failed with code %d: %s", proc.returncode, stderr_output)
        if os.path.exists(tmp_filepath):
            try:
                os.remove(tmp_filepath)
            except OSError:
                pass
        return None

    # Atomically move the completed video into place
    try:
        os.replace(tmp_filepath, out_filepath)
    except OSError:
        import shutil
        shutil.move(tmp_filepath, out_filepath)

    # Update MatchVideo field
    rel_path = f"matches/annotated/{out_filename}"
    video_obj.annotated_video = rel_path
    video_obj.save(update_fields=["annotated_video"])

    logger.info("Successfully generated annotated video for match=%s at %s", match.id, out_filepath)
    return out_filepath
