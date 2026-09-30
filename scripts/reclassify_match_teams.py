"""
Fast, single-pass sequential video decoder to extract pure turf-isolated torso colors,
re-classify all tracks with zero flips, and update player_tracking_csv and pitch mapping.
"""

import os
import sys
import csv
import io
import cv2
import numpy as np
import django
from collections import defaultdict, Counter

sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from apps.matches.models import Match, MatchFiles, TrackPlayerIdentification
from apps.matches.tasks import _hex_to_bgr, compute_pitch_mapping
from ai_engine.stage3_team_reid.team_classifier import sample_torso_color, classify_teams_with_fallback
from ai_engine.config import DEFAULT_CONFIG
from ai_engine.utils.types import Team
from django.core.files.base import ContentFile

def fast_reclassify_match(match_id: int):
    match = Match.objects.get(id=match_id)
    files = getattr(match, "files", None)
    if not files or not files.player_tracking_csv:
        print(f"Error: Match #{match_id} has no player_tracking_csv")
        return

    video_path = match.video.original_video.path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    with files.player_tracking_csv.open("rb") as f:
        content = f.read().decode("utf-8")
    
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    
    track_dets = defaultdict(list)
    for r in rows:
        tid = int(r["track_id"])
        track_dets[tid].append(r)

    home_bgr = _hex_to_bgr(match.home_kit_color)
    away_bgr = _hex_to_bgr(match.away_kit_color)
    home_gk_bgr = _hex_to_bgr(match.home_gk_kit_color)
    away_gk_bgr = _hex_to_bgr(match.away_gk_kit_color)

    # Build an inverted index of frame_idx -> list of (track_id, bbox)
    frame_requests = defaultdict(list)
    cls_by_track = {}
    
    for tid, dets in track_dets.items():
        cls_vals = [d["class"] for d in dets]
        cls_by_track[tid] = Counter(cls_vals).most_common(1)[0][0]
        
        sample_indices = np.linspace(0, len(dets)-1, min(4, len(dets)), dtype=int)
        for idx in sample_indices:
            d = dets[idx]
            f_idx = int(d["frame_idx"])
            x1, y1, x2, y2 = int(float(d["x1"])), int(float(d["y1"])), int(float(d["x2"])), int(float(d["y2"]))
            frame_requests[f_idx].append((tid, x1, y1, x2, y2))

    sorted_frames = sorted(frame_requests.keys())
    print(f"Fast sequential extraction across {len(sorted_frames)} unique video frames for {len(track_dets)} tracks...")
    
    sampled_colors_by_track = defaultdict(list)
    curr_f = 0
    
    for f_idx in sorted_frames:
        if curr_f != f_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            curr_f = f_idx
            
        ok, frame = cap.read()
        curr_f += 1
        if not ok:
            continue
            
        h_frame, w_frame = frame.shape[:2]
        for tid, x1, y1, x2, y2 in frame_requests[f_idx]:
            crop = frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]
            col = sample_torso_color(crop)
            if col is not None:
                sampled_colors_by_track[tid].append(col)

    cap.release()

    colors = {}
    for tid, cols in sampled_colors_by_track.items():
        if cols:
            colors[tid] = np.median(cols, axis=0)

    # Re-classify strictly by ground truth kit color
    team_by_track = classify_teams_with_fallback(
        colors, home_bgr, away_bgr, DEFAULT_CONFIG.team_reid,
        home_gk_bgr=home_gk_bgr, away_gk_bgr=away_gk_bgr, cls_by_track=cls_by_track
    )

    # Re-write CSV with updated team column
    updated_csv = io.StringIO()
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(updated_csv, fieldnames=fieldnames)
    writer.writeheader()
    
    for r in rows:
        tid = int(r["track_id"])
        c_mode = cls_by_track.get(tid, "player")
        if c_mode == "referee":
            r["team"] = "referee"
        else:
            t_enum = team_by_track.get(tid, Team.UNKNOWN)
            r["team"] = t_enum.value
        writer.writerow(r)

    files.player_tracking_csv.save(f"match_{match.id}_players.csv", ContentFile(updated_csv.getvalue().encode("utf-8")), save=True)
    print("Saved updated player_tracking_csv with true kit team assignments.")

    # Invalidate stale identifications & recompute pitch mapping
    TrackPlayerIdentification.objects.filter(match=match).delete()
    print("Recomputing pitch mapping, topological lineup assignments, and stats...")
    compute_pitch_mapping(match.id)
    print(f"Match #{match_id} successfully reclassified and rebuilt.")

if __name__ == "__main__":
    import sys
    mid = int(sys.argv[1]) if len(sys.argv) > 1 else 75
    fast_reclassify_match(mid)
