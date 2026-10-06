"""Safely extract HF SN-PCBAS video ZIPs after supplying their current archive password."""

import argparse
import os
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_hf_archive_password():
    password = os.environ.get("SOCCERNET_HF_ARCHIVE_PASSWORD")
    if password:
        return password
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "SOCCERNET_HF_ARCHIVE_PASSWORD":
                return value.strip().strip("\"'")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try standard unencrypted zip first
    try:
        with zipfile.ZipFile(args.archive, "r") as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".mp4"):
                    continue
                target = (output_dir / Path(member.filename).name).resolve()
                print(f"Extracting {member.filename} -> {target.name}", flush=True)
                with archive.open(member) as source, target.open("wb") as destination:
                    while chunk := source.read(8 * 1024 * 1024):
                        destination.write(chunk)
        print(f"Extracted videos to {output_dir}")
        return
    except (zipfile.BadZipFile, RuntimeError):
        pass

    password = read_hf_archive_password()
    if not password:
        sys.exit(
            "Archive is encrypted and SOCCERNET_HF_ARCHIVE_PASSWORD is not set."
        )

    try:
        import pyzipper
    except ImportError as exc:
        raise SystemExit("Install pyzipper to extract the AES-encrypted Hugging Face video archive.") from exc

    with pyzipper.AESZipFile(args.archive) as archive:
        archive.pwd = password.encode("utf-8")
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if target != output_dir and output_dir not in target.parents:
                raise RuntimeError(f"Unsafe path in SoccerNet archive: {member.filename}")
            if not member.filename.lower().endswith(".mp4"):
                continue
            print(f"Extracting {member.filename}", flush=True)
            with archive.open(member) as source, target.open("wb") as destination:
                while chunk := source.read(8 * 1024 * 1024):
                    destination.write(chunk)
    print(f"Extracted videos to {output_dir}")


if __name__ == "__main__":
    main()
