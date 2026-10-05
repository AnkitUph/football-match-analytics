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

    password = read_hf_archive_password()
    if not password:
        sys.exit(
            "SN-PCBAS video ZIP entries are AES encrypted. Set "
            "SOCCERNET_HF_ARCHIVE_PASSWORD to the password for this Hugging Face archive; "
            "do not use the legacy EXRCS server password here."
        )

    try:
        import pyzipper
    except ImportError as exc:
        raise SystemExit("Install pyzipper to extract the AES-encrypted Hugging Face video archive.") from exc

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
