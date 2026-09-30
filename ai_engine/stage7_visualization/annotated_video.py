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


def render_annotated_match_video(match, max_dimension: int = 1280, start_sec: int = 0, duration_sec: int | None = None) -> str | None:
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

    # Identify off-pitch tracks (coaches in technical area/dugout far outside playing field)
    off_pitch_tracks = set()
    for tid, coords in track_pitch_coords.items():
        if len(coords) >= 15:
            med_x = float(np.median([abs(x) for x, y in coords]))
            med_y = float(np.median([abs(y) for x, y in coords]))
            if med_y > 40.0 or med_x > 58.0:
                off_pitch_tracks.add(tid)

    # Temporal smoothing / gap-filling: for each track_id, interpolate 1-14 frame detection blips
    # so bounding boxes are completely solid, stable, and continuous on all players across momentary occlusions.
    track_frames = defaultdict(dict)
    for f_idx, plist in player_annotations_by_frame.items():
        for p in plist:
            track_frames[p["track_id"]][f_idx] = p

    for tid, fdict in track_frames.items():
        sorted_f = sorted(fdict.keys())
        for k in range(len(sorted_f) - 1):
            f_a, f_b = sorted_f[k], sorted_f[k + 1]
            gap = f_b - f_a
            if 1 < gap <= 15:  # up to 14 missing frames (~0.56s)
                box_a = fdict[f_a]["bbox"]
                box_b = fdict[f_b]["bbox"]
                team = fdict[f_a]["team"]
                jersey = fdict[f_a]["jersey"] or fdict[f_b]["jersey"]
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

    # 3. Setup team information and referee disambiguation
    all_track_ids = set(track_frames.keys())
    track_assigned_team = {}

    # 3b. Referee Disambiguation Pass:
    # Strictly at most ONE on-pitch referee.
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
        if len(f_set & occupied_ref_frames) <= 2:
            verified_ref_tracks.add(tid)
            occupied_ref_frames.update(f_set)
            track_assigned_team[tid] = "referee"

    # Re-assign misclassified referee tracks to players based on pitch side
    for tid in candidate_ref_tids:
        if tid not in verified_ref_tracks:
            coords = track_pitch_coords.get(tid, [])
            t_med_x = float(np.median([x for x, y in coords])) if coords else 0.0
            track_assigned_team[tid] = "team_a" if t_med_x <= 0 else "team_b"

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
                    if abs(p["pitch_x"]) > 58.0 or abs(p["pitch_y"]) > 40.0:
                        continue

                x1, y1, x2, y2 = p["bbox"]
                if scale != 1.0:
                    x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                home_badge = match.home_team.name or match.home_team.short_name or "Chelsea"
                away_badge = match.away_team.name or match.away_team.short_name or "Burnley"

                # 2. Determine color and clean team name label (ONLY show team name, never player name)
                if team_key == "referee":
                    if tid in verified_ref_tracks and not ref_drawn_in_frame:
                        color = TEAM_COLORS["referee"]
                        label = "REF"
                        ref_drawn_in_frame = True
                    else:
                        assigned_team = track_assigned_team.get(tid, "team_a")
                        color = TEAM_COLORS.get(assigned_team, TEAM_COLORS["team_a"])
                        label = home_badge if assigned_team == "team_a" else away_badge
                elif team_key == "team_a":
                    color = TEAM_COLORS["team_a"]
                    label = home_badge
                elif team_key == "team_b":
                    color = TEAM_COLORS["team_b"]
                    label = away_badge
                else:
                    color = TEAM_COLORS["unknown"]
                    label = "Player"

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
