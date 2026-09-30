"""
Downloads a full match from SoccerNet via Hugging Face and extracts two
continuous 20-minute segments into media/test_clips/ for testing and training.
"""

import os
import sys
import subprocess
from pathlib import Path
from SoccerNet.Downloader import SoccerNetDownloader


def download_and_extract(
    game: str = "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
    video_file: str = "1_720p.mkv",
    output_dir: str = "/app/media/test_clips",
):
    if not os.environ.get("HF_TOKEN"):
        print("Error: set HF_TOKEN in the environment to download from Hugging Face.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    download_dir = "/app/media/soccernet_raw"
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"1. Downloading SoccerNet match '{game}' ({video_file})...")
    dl = SoccerNetDownloader(LocalDirectory=download_dir)
    dl.downloadGame(game=game, files=[video_file, "Labels-v2.json"], source="HuggingFace")
    
    raw_video = Path(download_dir) / game / video_file
    if not raw_video.exists():
        print(f"Error: Expected video file at {raw_video} was not found.")
        return
    
    print(f"Downloaded match video successfully: {raw_video} ({raw_video.stat().st_size / (1024*1024):.1f} MB)")
    
    # Save full match copy or symlink in test_clips
    full_target = out_path / "soccernet_chelsea_burnley_full_half1.mkv"
    if not full_target.exists():
        os.link(str(raw_video), str(full_target))
        print(f"Linked full match to {full_target}")
        
    # Clip Segment 1 (00:00 to 20:00)
    seg1_out = out_path / "soccernet_chelsea_burnley_seg1_20min.mp4"
    print(f"2. Extracting Segment 1 (00:00 - 20:00) -> {seg1_out}...")
    cmd1 = [
        "ffmpeg", "-y",
        "-ss", "00:00:00",
        "-i", str(raw_video),
        "-t", "00:20:00",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac",
        str(seg1_out),
    ]
    subprocess.run(cmd1, check=True)
    print(f"Segment 1 extracted ({seg1_out.stat().st_size / (1024*1024):.1f} MB)")

    # Clip Segment 2 (20:00 to 40:00)
    seg2_out = out_path / "soccernet_chelsea_burnley_seg2_20min.mp4"
    print(f"3. Extracting Segment 2 (20:00 - 40:00) -> {seg2_out}...")
    cmd2 = [
        "ffmpeg", "-y",
        "-ss", "00:20:00",
        "-i", str(raw_video),
        "-t", "00:20:00",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac",
        str(seg2_out),
    ]
    subprocess.run(cmd2, check=True)
    print(f"Segment 2 extracted ({seg2_out.stat().st_size / (1024*1024):.1f} MB)")
    
    print("\nAll downloads and test clips extracted successfully!")

if __name__ == "__main__":
    download_and_extract()
