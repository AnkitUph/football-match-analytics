"""Download SN-PCBAS-2026 train videos and unpack into the training dataset folder."""

import argparse
import os
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "media/pcbas2026/downloads"
TRAIN_VIDEOS_DIR = ROOT / "scratch/pcbas2026/prepared/train/videos"


def get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() == "HF_TOKEN":
                return v.strip().strip("\"'")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="SoccerNet/SN-PCBAS-2026")
    parser.add_argument("--filename", default="videos_352x640_TRAIN.zip")
    parser.add_argument("--download-dir", type=Path, default=DOWNLOAD_DIR)
    parser.add_argument("--output-dir", type=Path, default=TRAIN_VIDEOS_DIR)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.download_dir / args.filename

    if not args.skip_download:
        token = get_hf_token()
        if not token:
            sys.exit("Error: HF_TOKEN not found in environment or .env")
        print(f"Starting download of {args.filename} (~21.3 GB) from {args.repo_id}...")
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type="dataset",
            token=token,
            local_dir=str(args.download_dir),
        )
        print(f"Download complete: {zip_path} ({zip_path.stat().st_size / (1024**3):.2f} GB)")

    if not zip_path.is_file():
        sys.exit(f"Zip archive not found at {zip_path}")

    print(f"Extracting {zip_path.name} to {args.output_dir}...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [m for m in archive.infolist() if not m.is_dir() and m.filename.lower().endswith(".mp4")]
        print(f"Found {len(members)} video files in archive")
        for i, member in enumerate(members, 1):
            target = args.output_dir / Path(member.filename).name
            if target.is_file() and target.stat().st_size > 0:
                print(f"[{i}/{len(members)}] {target.name} already exists, skipping")
                continue
            print(f"[{i}/{len(members)}] Extracting {member.filename} -> {target.name}", flush=True)
            with archive.open(member) as src, target.open("wb") as dst:
                while chunk := src.read(16 * 1024 * 1024):
                    dst.write(chunk)

    extracted = list(args.output_dir.glob("game_*.mp4"))
    print(f"Successfully unpacked {len(extracted)} game videos in {args.output_dir}")


if __name__ == "__main__":
    main()
