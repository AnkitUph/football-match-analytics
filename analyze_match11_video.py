import cv2
import csv
import io
from apps.matches.models import Match

def analyze():
    m = Match.objects.get(pk=11)
    video_path = m.video.original_video.path
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Match 11 video: {total_frames} frames, {fps} fps, {w}x{h}")

    # Load ball detections
    with m.files.ball_tracking_csv.open("rb") as f:
        ball_rows = list(csv.DictReader(io.StringIO(f.read().decode("utf-8"))))
    ball_by_frame = {int(r["frame_idx"]): (float(r["x_px"]), float(r["y_px"]), r["interpolated"].strip().lower() == "true") for r in ball_rows}

    # Load player detections
    with m.files.player_tracking_csv.open("rb") as f:
        player_rows = list(csv.DictReader(io.StringIO(f.read().decode("utf-8"))))
    
    dets_by_frame = {}
    for r in player_rows:
        f_idx = int(r["frame_idx"])
        if f_idx not in dets_by_frame:
            dets_by_frame[f_idx] = []
        dets_by_frame[f_idx].append({
            "track_id": int(r["track_id"]),
            "team": r["team"],
            "bbox": (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])),
            "pitch": (r.get("pitch_x"), r.get("pitch_y")),
        })

    # Key moments to inspect:
    # 1. Kickoff / buildup (frame 155)
    # 2. Attack progression (frame 400, 530)
    # 3. Shot sequence (frames 660, 666, 672)
    # 4. Long ball sequence (frames 692, 694, 698)
    inspect_frames = [155, 400, 530, 660, 666, 672, 694]

    for f_idx in inspect_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        # Draw players
        for p in dets_by_frame.get(f_idx, []):
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            color = (0, 0, 255) if p["team"] == "team_a" else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"T{p['track_id']}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw ball
        if f_idx in ball_by_frame:
            bx, by, interp = ball_by_frame[f_idx]
            b_color = (0, 255, 255) if not interp else (0, 165, 255)
            cv2.circle(frame, (int(bx), int(by)), 6, b_color, -1)
            cv2.putText(frame, "BALL" if not interp else "BALL(interp)", (int(bx) + 8, int(by) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, b_color, 2)

        out_path = f"inspect_frame_{f_idx}.jpg"
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path} with {len(dets_by_frame.get(f_idx, []))} players, ball={'YES' if f_idx in ball_by_frame else 'NO'}")

    cap.release()

if __name__ == "__main__":
    analyze()
