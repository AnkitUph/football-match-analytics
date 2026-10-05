"""Download SoccerNet Ball videos and their matching ball-action labels."""

import os
import sys
import zipfile
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader

ROOT = Path(__file__).resolve().parents[1]
VIDEO_LIST = ROOT / "data/fine_tune_videos.txt"
OUTPUT_DIR = ROOT / "media/fine_tune_clips"
CACHE_DIR = ROOT / "media/soccernet_ball_downloads"
TASK_DIR = CACHE_DIR / "spotting-ball-2024"


def load_project_hf_token():
    """Pass only HF_TOKEN from .env to the official Hugging Face client."""
    if os.environ.get("HF_TOKEN"):
        return
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "HF_TOKEN":
            os.environ["HF_TOKEN"] = value.strip().strip("\"'")
            return


def main():
    if not VIDEO_LIST.is_file():
        sys.exit(f"Match list missing: {VIDEO_LIST}. Run scripts/generate_match_list.py first.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    load_project_hf_token()
    downloader = SoccerNetDownloader(LocalDirectory=str(CACHE_DIR))
    # SoccerNet Ball is distributed as split archives, not per-game assets.
    # This task-specific HF route contains the corresponding Labels-ball data.
    print("Downloading the SoccerNet Ball 2024 training split")
    downloader.downloadDataTask(task="spotting-ball-2024", split=["train"], source="HuggingFace")
    archives = sorted(TASK_DIR.glob("*.zip"))
    if not archives:
        sys.exit(f"SoccerNet downloader produced no split archive in {TASK_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    allowed_games = {line.strip() for line in VIDEO_LIST.read_text().splitlines() if line.strip()}
    for archive in archives:
        print(f"Extracting {archive.name}")
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                # Archives are trusted official data, but reject path traversal
                # entries before extracting into the project media directory.
                target = (OUTPUT_DIR / member.filename).resolve()
                if target != OUTPUT_DIR.resolve() and OUTPUT_DIR.resolve() not in target.parents:
                    raise RuntimeError(f"Unsafe path in SoccerNet archive: {member.filename}")
                if not any(member.filename.startswith(game + "/") for game in allowed_games):
                    continue
                # Only keep 720p training footage and annotations. The 224p
                # copies are unnecessary for this model.
                if not (member.filename.endswith("/720p.mp4") or member.filename.endswith("/Labels-ball.json")):
                    continue
                try:
                    bundle.extract(member, OUTPUT_DIR)
                except (RuntimeError, NotImplementedError) as exc:
                    raise RuntimeError(
                        f"HF access succeeded, but {archive.name} cannot be unpacked with Python's standard ZIP reader: {exc}. "
                        "Check the current SoccerNet HF archive instructions; this workflow does not use the legacy EXRCS password."
                    ) from exc

    labels = list(OUTPUT_DIR.rglob("Labels-ball.json"))
    if not labels:
        sys.exit(f"No Labels-ball.json extracted under {OUTPUT_DIR}; inspect {archives[0]}")
    print(f"Prepared {len(labels)} labeled SoccerNet Ball games under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
