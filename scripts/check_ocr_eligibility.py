"""
Measures bounding box height percentiles across match tracking CSVs
and evaluates Jersey OCR resolution eligibility based on broadcast Nyquist limits.
"""

import sys
import os
import csv
import numpy as np

def evaluate_csv_eligibility(csv_path: str, match_label: str = "Match"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    box_heights = []
    track_heights = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("class") in ("ball", "referee"):
                continue
            try:
                y1 = float(row["y1"])
                y2 = float(row["y2"])
                tid = int(row["track_id"])
                h = abs(y2 - y1)
                if h > 5.0:
                    box_heights.append(h)
                    if tid not in track_heights:
                        track_heights[tid] = []
                    track_heights[tid].append(h)
            except (ValueError, KeyError):
                continue

    if not box_heights:
        print(f"No valid player boxes found in {csv_path}")
        return

    arr = np.array(box_heights)
    p10, p25, p50, p75, p90, p95, p99 = np.percentile(arr, [10, 25, 50, 75, 90, 95, 99])
    max_h = np.max(arr)
    min_h = np.min(arr)

    # Track-level evaluation (>= 50 frames with box height >= 150px)
    eligible_tracks = 0
    total_tracks = len(track_heights)
    for tid, h_list in track_heights.items():
        tall_frames = sum(1 for h in h_list if h >= 150.0)
        if tall_frames >= 50:
            eligible_tracks += 1

    eligible_pct = (eligible_tracks / max(total_tracks, 1)) * 100.0

    if eligible_pct >= 40.0:
        verdict = "ELIGIBLE (>= 40% of tracks pass 150px threshold)"
    elif eligible_pct >= 15.0:
        verdict = "PARTIAL (15% - 40% of tracks pass)"
    else:
        verdict = "NOT ELIGIBLE (< 15% of tracks pass 150px threshold)"

    print("=" * 70)
    print(f"📊 RESOLUTION & OCR ELIGIBILITY REPORT: {match_label}")
    print(f"Source: {os.path.basename(csv_path)}")
    print("-" * 70)
    print(f"Total Player Bounding Boxes: {len(box_heights):,}")
    print(f"Total Unique Player Tracks:  {total_tracks:,}")
    print(f"\nBounding Box Height Percentiles (Pixels):")
    print(f"  - Min Height:        {min_h:6.1f} px  (Est. Digit: {min_h*0.16:4.1f} px)")
    print(f"  - 10th Percentile:   {p10:6.1f} px  (Est. Digit: {p10*0.16:4.1f} px)")
    print(f"  - 25th Percentile:   {p25:6.1f} px  (Est. Digit: {p25*0.16:4.1f} px)")
    print(f"  - 50th (Median):     {p50:6.1f} px  (Est. Digit: {p50*0.16:4.1f} px)")
    print(f"  - 75th Percentile:   {p75:6.1f} px  (Est. Digit: {p75*0.16:4.1f} px)")
    print(f"  - 90th Percentile:   {p90:6.1f} px  (Est. Digit: {p90*0.16:4.1f} px)")
    print(f"  - 95th Percentile:   {p95:6.1f} px  (Est. Digit: {p95*0.16:4.1f} px)")
    print(f"  - 99th Percentile:   {p99:6.1f} px  (Est. Digit: {p99*0.16:4.1f} px)")
    print(f"  - Max Height:        {max_h:6.1f} px  (Est. Digit: {max_h*0.16:4.1f} px)")
    print(f"\nGate Threshold Check (>= 150 px height for >= 50 frames):")
    print(f"  - Passing Tracks:    {eligible_tracks} / {total_tracks} ({eligible_pct:.2f}%)")
    print(f"  - Final Verdict:     >>> {verdict} <<<")
    print("=" * 70)

if __name__ == "__main__":
    import django
    sys.path.insert(0, "/app")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    django.setup()
    from apps.matches.models import Match

    match_ids = [75, 74, 32]
    for mid in match_ids:
        try:
            m = Match.objects.get(id=mid)
            if m.files and m.files.player_tracking_csv:
                csv_path = m.files.player_tracking_csv.path
                evaluate_csv_eligibility(csv_path, f"Match #{mid} ({m.home_team.name} vs {m.away_team.name})")
                print("\n")
        except Exception as e:
            print(f"Could not evaluate match #{mid}: {e}")
