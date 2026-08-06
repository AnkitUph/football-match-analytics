"""
Stage A validation: detection + tracking with the custom-trained model,
drawn with a distinct color per category (player/goalkeeper/referee/
ball). No team assignment, speed/distance, or camera compensation yet -
those come next, once this stage is confirmed working on real footage.

Usage:
    python -m ai_engine.main <video_path> <output_dir>
"""

import logging
import sys
from pathlib import Path

import cv2

from ai_engine.team_assigner import TeamAssigner
from ai_engine.trackers import Tracker
from ai_engine.utils import get_video_fps, read_video, save_video

logger = logging.getLogger(__name__)

CATEGORY_COLORS = {
    "players": (60, 179, 113),      # green (BGR) - fallback if team not yet assigned
    "goalkeepers": (0, 200, 255),   # yellow - fallback
    "referees": (60, 60, 220),      # red
}
TEAM_DRAW_COLORS = {
    "home": (255, 100, 40),
    "away": (40, 40, 220),
    "home_gk": (255, 200, 40),
    "away_gk": (40, 140, 220),
    "team_1": (200, 200, 40),
    "team_2": (40, 200, 200),
    "unclassified": (180, 180, 180),
}
BALL_COLOR = (0, 165, 255)


def draw_tracks(frames, tracks, team_assigner=None, player_teams=None, gk_teams=None):
    output_frames = []
    for frame_num, frame in enumerate(frames):
        frame = frame.copy()

        for track_id, info in tracks["players"][frame_num].items():
            x1, y1, x2, y2 = [int(v) for v in info["bbox"]]
            if player_teams is not None:
                label = player_teams.get(track_id, "unclassified")
                color = TEAM_DRAW_COLORS.get(label, TEAM_DRAW_COLORS["unclassified"])
            else:
                label, color = "player", CATEGORY_COLORS["players"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{track_id} {label}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        for track_id, info in tracks["goalkeepers"][frame_num].items():
            x1, y1, x2, y2 = [int(v) for v in info["bbox"]]
            if gk_teams is not None:
                label = gk_teams.get(track_id, "unclassified")
                color = TEAM_DRAW_COLORS.get(label, TEAM_DRAW_COLORS["unclassified"])
            else:
                label, color = "goalkeeper", CATEGORY_COLORS["goalkeepers"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{track_id} {label}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        for track_id, info in tracks["referees"][frame_num].items():
            x1, y1, x2, y2 = [int(v) for v in info["bbox"]]
            color = CATEGORY_COLORS["referees"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{track_id} referee", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        ball_info = tracks["ball"][frame_num].get(1)
        if ball_info:
            x1, y1, x2, y2 = ball_info["bbox"]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            is_interp = ball_info.get("is_interpolated", False)
            cv2.circle(frame, (cx, cy), 6, BALL_COLOR, 2 if is_interp else -1)

        output_frames.append(frame)
    return output_frames


def main(video_path, output_dir, model_path="ai_engine/models/best.pt",
         home_color=None, away_color=None, home_gk_color=None, away_gk_color=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading video: %s", video_path)
    frames = read_video(video_path)
    fps = get_video_fps(video_path)
    logger.info("%d frames at %.1f fps", len(frames), fps)

    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(frames, fps=fps)
    tracker.add_position_to_tracks(tracks)

    category_counts = {
        cat: len({tid for frame in tracks[cat] for tid in frame})
        for cat in ("players", "goalkeepers", "referees")
    }
    ball_frames_known = sum(1 for f in tracks["ball"] if f)

    print("--- Detection/Tracking Summary ---")
    print(f"Frames: {len(frames)}  FPS: {fps:.1f}")
    for cat, count in category_counts.items():
        print(f"Unique {cat}: {count}")
    print(f"Ball position known in {ball_frames_known}/{len(frames)} frames "
          f"({100*ball_frames_known/len(frames):.1f}%)")

    player_teams = None
    gk_teams = None

    if home_color and away_color:
        team_assigner = TeamAssigner(home_color, away_color, home_gk_color, away_gk_color)

        # Find the first frame with at least 2 players to calibrate team colors from
        calibration_frame_num = next(
            (i for i, f in enumerate(tracks["players"]) if len(f) >= 2), None
        )
        if calibration_frame_num is not None:
            team_assigner.assign_team_colors(frames[calibration_frame_num], tracks["players"][calibration_frame_num])

            player_teams = {}
            for frame in tracks["players"]:
                for track_id, info in frame.items():
                    if track_id not in player_teams:
                        player_teams[track_id] = team_assigner.get_player_team(
                            frames[calibration_frame_num], info["bbox"], track_id
                        )

            gk_teams = {}
            for frame_num, frame in enumerate(tracks["goalkeepers"]):
                for track_id, info in frame.items():
                    if track_id not in gk_teams:
                        gk_teams[track_id] = team_assigner.assign_goalkeeper_team(frames[frame_num], info["bbox"])

            print("--- Team Assignment ---")
            print(f"Players: {player_teams}")
            print(f"Goalkeepers: {gk_teams}")
        else:
            print("No frame with 2+ players found - skipping team assignment.")

    annotated = draw_tracks(frames, tracks, team_assigner if home_color else None, player_teams, gk_teams)
    output_path = output_dir / "stage_b_output.mp4"
    save_video(annotated, str(output_path), fps=fps)
    print(f"Saved annotated video to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m ai_engine.main <video_path> <output_dir> [model_path] "
              "[home_color] [away_color] [home_gk_color] [away_gk_color]")
        print('Example: python -m ai_engine.main clip.mp4 out/ ai_engine/models/best.pt "#646736" "#AFAA98"')
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    video_arg = sys.argv[1]
    output_arg = sys.argv[2]
    model_arg = sys.argv[3] if len(sys.argv) > 3 else "ai_engine/models/best.pt"
    home_arg = sys.argv[4] if len(sys.argv) > 4 else None
    away_arg = sys.argv[5] if len(sys.argv) > 5 else None
    home_gk_arg = sys.argv[6] if len(sys.argv) > 6 else None
    away_gk_arg = sys.argv[7] if len(sys.argv) > 7 else None

    main(video_arg, output_arg, model_arg, home_arg, away_arg, home_gk_arg, away_gk_arg)