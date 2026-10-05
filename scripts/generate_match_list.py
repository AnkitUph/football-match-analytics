"""Select SoccerNet Ball matches for domain-adaptation fine-tuning.

Candidates come from the checkpoint's SoccerNet Ball train split, preserving
the ball-action label vocabulary expected by the checkpoint.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNB_DIR = ROOT / "ai_engine/stage6_event_detection/tdeed/data/soccernetball"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "data/fine_tune_videos.txt")
    args = parser.parse_args()

    candidates = json.loads((SNB_DIR / "train.json").read_text())
    # English broadcast footage is the closest available labeled source domain
    # to the project's English-league match clips.
    games = [x["video"] for x in candidates if x["video"].startswith(("england_epl/", "england_efl/"))]
    games = games[: args.count]
    if not games:
        raise RuntimeError("No English games found in the SoccerNet Ball train split")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(games) + "\n")
    print(f"Selected {len(games)} labeled SoccerNet Ball games in {args.out}")


if __name__ == "__main__":
    main()
