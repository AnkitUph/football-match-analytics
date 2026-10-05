"""Convert SN-PCBAS-2026 tactical HDF5 labels to trainer events.json sidecars."""

import argparse
import json
from pathlib import Path

import h5py


# FOOTPASS TAAD class IDs: background, drive, pass, cross, throw-in, shot,
# header, tackle, block. Map only actions represented in the T-DEED checkpoint.
PCBAS_CLASS_TO_TDEED = {
    1: "DRIVE",
    2: "PASS",
    3: "CROSS",
    4: "THROW IN",
    5: "SHOT",
    6: "HEADER",
    7: "PLAYER SUCCESSFUL TACKLE",
    8: "BALL PLAYER BLOCK",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, type=Path, help="Extracted train/val tactical HDF5 file")
    parser.add_argument("--video-dir", required=True, type=Path, help="Directory containing game_<id>.mp4 files")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where annotations/<game>/events.json will be written")
    parser.add_argument("--allow-missing-videos", action="store_true", help="Write sidecars before encrypted video archives are extracted")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.h5, "r") as tactical:
        for key in tactical.keys():
            game_id, half = key.split("_")[-2:]
            video = (args.video_dir / f"game_{game_id}.mp4").resolve()
            if not video.is_file() and not args.allow_missing_videos:
                raise FileNotFoundError(f"Missing video for {key}: {video}")

            # The game video concatenates halves. HDF5 frame indices already use
            # that shared video timeline, so H1 and H2 events can be combined.
            values = tactical[key][:, [0, 13]]
            events = {}
            for frame_value, class_value in values:
                frame = int(round(float(frame_value)))
                class_id = int(round(float(class_value)))
                if class_id == 0:
                    continue
                label = PCBAS_CLASS_TO_TDEED.get(class_id)
                if label is None:
                    raise ValueError(f"Unrecognized SN-PCBAS class id {class_id} in {key}")
                events[(frame, label)] = {"frame": frame, "label": label}

            annotation_dir = args.output_dir / "annotations" / f"game_{game_id}"
            annotation_dir.mkdir(parents=True, exist_ok=True)
            sidecar = {
                "video": str(video),
                "fps": 25,
                "events": list(events.values()),
            }
            path = annotation_dir / "events.json"
            if path.exists():
                previous = json.loads(path.read_text())
                events.update({(int(e["frame"]), e["label"]): e for e in previous["events"]})
                sidecar["events"] = list(events.values())
            path.write_text(json.dumps(sidecar, separators=(",", ":")))
            print(f"{key}: {len(sidecar['events'])} unique labeled events -> {path}")

    sidecars = list((args.output_dir / "annotations").glob("game_*/events.json"))
    total_events = sum(len(json.loads(path.read_text())["events"]) for path in sidecars)
    print(f"Wrote {total_events} unique labeled events across {len(sidecars)} games")


if __name__ == "__main__":
    main()
