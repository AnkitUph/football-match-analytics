from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class GroundTruthIdentity:
    identity_id: str
    class_name: str = "player"
    description: str = ""


class GroundTruth:
    """
    Loads and validates ground-truth identity annotations.

    Supported annotations:

        track_labels:
            ByteTrack ID -> GT identity

        frame_labels:
            frame index -> {
                ByteTrack ID -> GT identity
            }
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Ground-truth file not found: {self.path}"
            )

        with self.path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)

        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if not isinstance(self.data, dict):
            raise ValueError("Ground truth must be a JSON object.")

        version = self.data.get("version")

        if version != 1:
            raise ValueError(
                f"Unsupported ground-truth version: {version}"
            )

        if "identities" not in self.data:
            raise ValueError(
                "Ground truth must contain 'identities'."
            )

        if not isinstance(self.data["identities"], dict):
            raise ValueError(
                "'identities' must be an object."
            )

        self._validate_identities()
        self._validate_track_labels()
        self._validate_frame_labels()

    def _validate_identities(self) -> None:
        for identity_id, identity in self.data["identities"].items():

            if not isinstance(identity, dict):
                raise ValueError(
                    f"Identity '{identity_id}' must be an object."
                )

            class_name = identity.get("class_name", "player")

            if not isinstance(class_name, str):
                raise ValueError(
                    f"Identity '{identity_id}' has invalid class_name."
                )

    def _validate_track_labels(self) -> None:
        labels = self.data.get("track_labels", {})

        if not isinstance(labels, dict):
            raise ValueError(
                "'track_labels' must be an object."
            )

        for track_id, identity_id in labels.items():

            self._validate_track_id(track_id)

            if identity_id not in self.data["identities"]:
                raise ValueError(
                    f"Track {track_id} references unknown "
                    f"identity '{identity_id}'."
                )

    def _validate_frame_labels(self) -> None:
        labels = self.data.get("frame_labels", {})

        if not isinstance(labels, dict):
            raise ValueError(
                "'frame_labels' must be an object."
            )

        for frame_index, frame_data in labels.items():

            self._validate_frame_index(frame_index)

            if not isinstance(frame_data, dict):
                raise ValueError(
                    f"Frame '{frame_index}' must contain "
                    f"an object of track labels."
                )

            for track_id, identity_id in frame_data.items():

                self._validate_track_id(track_id)

                if identity_id not in self.data["identities"]:
                    raise ValueError(
                        f"Frame {frame_index}, track {track_id} "
                        f"references unknown identity "
                        f"'{identity_id}'."
                    )

    @staticmethod
    def _validate_track_id(track_id: str) -> None:
        try:
            value = int(track_id)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid track ID: {track_id}"
            )

        if value < 0:
            raise ValueError(
                f"Track ID cannot be negative: {track_id}"
            )

    @staticmethod
    def _validate_frame_index(frame_index: str) -> None:
        try:
            value = int(frame_index)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid frame index: {frame_index}"
            )

        if value < 0:
            raise ValueError(
                f"Frame index cannot be negative: {frame_index}"
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def video(self) -> Optional[str]:
        return self.data.get("video")

    @property
    def fps(self) -> Optional[float]:
        value = self.data.get("fps")

        if value is None:
            return None

        return float(value)

    # ------------------------------------------------------------------
    # Identity access
    # ------------------------------------------------------------------

    def get_identity(
        self,
        identity_id: str,
    ) -> Optional[GroundTruthIdentity]:

        identity = self.data["identities"].get(identity_id)

        if identity is None:
            return None

        return GroundTruthIdentity(
            identity_id=identity_id,
            class_name=identity.get("class_name", "player"),
            description=identity.get("description", ""),
        )

    def get_identity_ids(self) -> list[str]:
        return list(self.data["identities"].keys())

    # ------------------------------------------------------------------
    # Track-level GT
    # ------------------------------------------------------------------

    def get_track_identity(
        self,
        track_id: int,
    ) -> Optional[str]:

        return self.data.get(
            "track_labels",
            {},
        ).get(str(track_id))

    def get_track_labels(self) -> Dict[int, str]:

        return {
            int(track_id): identity_id
            for track_id, identity_id
            in self.data.get("track_labels", {}).items()
        }

    # ------------------------------------------------------------------
    # Frame-level GT
    # ------------------------------------------------------------------

    def get_frame_identity(
        self,
        frame_index: int,
        track_id: int,
    ) -> Optional[str]:

        frame = self.data.get(
            "frame_labels",
            {},
        ).get(str(frame_index), {})

        return frame.get(str(track_id))

    def get_frame_labels(
        self,
        frame_index: int,
    ) -> Dict[int, str]:

        frame = self.data.get(
            "frame_labels",
            {},
        ).get(str(frame_index), {})

        return {
            int(track_id): identity_id
            for track_id, identity_id
            in frame.items()
        }

    def get_all_frame_labels(self) -> Dict[int, Dict[int, str]]:

        return {
            int(frame_index): {
                int(track_id): identity_id
                for track_id, identity_id
                in frame_data.items()
            }
            for frame_index, frame_data
            in self.data.get("frame_labels", {}).items()
        }

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def annotated_track_ids(self) -> set[int]:
        return set(self.get_track_labels().keys())

    def annotated_frame_indices(self) -> set[int]:
        return {
            int(frame_index)
            for frame_index
            in self.data.get("frame_labels", {})
        }

    def total_track_annotations(self) -> int:
        return len(self.get_track_labels())

    def total_frame_annotations(self) -> int:

        return sum(
            len(frame_data)
            for frame_data
            in self.data.get("frame_labels", {}).values()
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:

        return {
            "path": str(self.path),
            "version": self.data.get("version"),
            "video": self.video,
            "fps": self.fps,
            "identities": len(self.get_identity_ids()),
            "track_annotations": self.total_track_annotations(),
            "annotated_tracks": len(self.annotated_track_ids()),
            "annotated_frames": len(
                self.annotated_frame_indices()
            ),
            "frame_annotations": self.total_frame_annotations(),
        }

    def print_summary(self) -> None:

        summary = self.summary()

        print()
        print("=" * 60)
        print("GROUND TRUTH SUMMARY")
        print("=" * 60)

        print(
            f"File:                    {summary['path']}"
        )

        print(
            f"Video:                   {summary['video']}"
        )

        print(
            f"FPS:                     {summary['fps']}"
        )

        print(
            f"GT identities:            {summary['identities']}"
        )

        print(
            f"Annotated tracks:         {summary['annotated_tracks']}"
        )

        print(
            f"Track annotations:        {summary['track_annotations']}"
        )

        print(
            f"Annotated frames:         {summary['annotated_frames']}"
        )

        print(
            f"Frame annotations:        {summary['frame_annotations']}"
        )

        print("=" * 60)