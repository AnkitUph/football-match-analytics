"""Preview or apply conservative team relabeling from sampled jersey crops."""

import os
import sys
import csv
import io
import argparse
import cv2
import numpy as np
import django
from collections import defaultdict, Counter

sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from apps.matches.models import Match, MatchFiles, MatchLineup, TrackPlayerIdentification
from apps.matches.tasks import _hex_to_bgr, compute_pitch_mapping
from ai_engine.stage3_team_reid.team_classifier import sample_torso_color, classify_teams_with_fallback
from ai_engine.config import DEFAULT_CONFIG
from ai_engine.utils.types import Team
from django.core.files.base import ContentFile

def fast_reclassify_match(match_id: int, apply: bool = False):
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

        # Referees are assigned from their detector class below; don't spend
        # video reads sampling their kits as if they were team players.
        if cls_by_track[tid] not in ("player", "goalkeeper") or len(dets) < 12:
            continue

        sample_indices = np.linspace(0, len(dets)-1, min(3, len(dets)), dtype=int)
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
            gap = f_idx - curr_f
            if 0 < gap <= 12:
                while curr_f < f_idx:
                    cap.grab()
                    curr_f += 1
            else:
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
        home_gk_bgr=home_gk_bgr, away_gk_bgr=away_gk_bgr, cls_by_track=cls_by_track,
        color_samples_by_track=sampled_colors_by_track,
    )

    # Compare the preview with the saved labels before changing anything.
    old_by_track = {}
    for tid, dets in track_dets.items():
        old_by_track[tid] = Counter(d["team"] for d in dets).most_common(1)[0][0]

    new_by_track = {}
    for tid, old_team in old_by_track.items():
        if old_team == "referee" or cls_by_track.get(tid) == "referee":
            new_by_track[tid] = "referee"
        elif tid in team_by_track:
            new_by_track[tid] = team_by_track[tid].value
        else:
            # Keep the saved value where this run could not get a usable
            # torso sample; missing evidence must not cause a team flip.
            new_by_track[tid] = old_team

    # Human-confirmed identities are stronger evidence than a noisy jersey
    # crop. Keep their team side authoritative in the preview and on apply.
    confirmed_side_by_track = {
        row.track_id: row.lineup_entry.side
        for row in TrackPlayerIdentification.objects.filter(
            match=match, is_auto_assigned=False
        ).select_related("lineup_entry")
    }
    for tid, side in confirmed_side_by_track.items():
        new_by_track[tid] = "team_a" if side == MatchLineup.Side.HOME else "team_b"

    transitions = Counter()
    for tid, old_team in old_by_track.items():
        transitions[(old_team, new_by_track[tid])] += 1
    print("Track label preview (saved -> proposed):")
    for key, count in sorted(transitions.items()):
        print(f"  {key[0]} -> {key[1]}: {count}")
    print(f"Human-confirmed tracks preserved: {len(confirmed_side_by_track)}")

    if not apply:
        print("Preview only; no match data was changed. Re-run with --apply to save.")
        return

    # Re-write CSV with updated team column only after explicit --apply.
    updated_csv = io.StringIO()
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(updated_csv, fieldnames=fieldnames)
    writer.writeheader()
    
    for r in rows:
        tid = int(r["track_id"])
        r["team"] = new_by_track[tid]
        writer.writerow(r)

    files.player_tracking_csv.save(f"match_{match.id}_players.csv", ContentFile(updated_csv.getvalue().encode("utf-8")), save=True)
    print("Saved updated player_tracking_csv with true kit team assignments.")

    # Automatic lineup guesses are tied to the former team partition and
    # should be rebuilt. Preserve every human-confirmed identification.
    TrackPlayerIdentification.objects.filter(
        match=match, is_auto_assigned=True
    ).delete()
    print("Recomputing pitch mapping and automatic lineup assignments...")
    compute_pitch_mapping(match.id)
    # Team relabeling changes the overlay even when the underlying boxes do
    # not change. Rebuild it so the results page doesn't keep showing stale
    # labels/boxes from the previous processing run.
    from ai_engine.stage7_visualization.annotated_video import render_annotated_match_video
    match.refresh_from_db()
    render_annotated_match_video(match)
    print(f"Match #{match_id} successfully reclassified and rebuilt.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview or reclassify team labels from saved player crops.")
    parser.add_argument("match_id", type=int, nargs="?", default=75)
    parser.add_argument("--apply", action="store_true", help="Save the new labels and recompute derived match data.")
    args = parser.parse_args()
    fast_reclassify_match(args.match_id, apply=args.apply)
