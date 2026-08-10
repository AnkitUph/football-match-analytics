from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IdentityRecord:
    """
    Persistent representation of one global football identity.

    A Global Identity may contain multiple ByteTrack track IDs
    belonging to the same real-world player.
    """

    global_id: str

    track_ids: list[int] = field(default_factory=list)

    class_name: str = "player"

    team: Optional[str] = None

    jersey_number: Optional[int] = None

    first_frame: Optional[int] = None

    last_frame: Optional[int] = None

    total_frames: int = 0

    confidence: float = 0.0

    identity_quality: float = 0.0

    metadata: dict = field(default_factory=dict)

    def add_track(
        self,
        track_id: int,
        first_frame: Optional[int] = None,
        last_frame: Optional[int] = None,
        frame_count: int = 0,
    ) -> None:
        """Add a ByteTrack ID to this global identity."""

        if track_id not in self.track_ids:
            self.track_ids.append(track_id)

        if first_frame is not None:
            if self.first_frame is None:
                self.first_frame = first_frame
            else:
                self.first_frame = min(
                    self.first_frame,
                    first_frame,
                )

        if last_frame is not None:
            if self.last_frame is None:
                self.last_frame = last_frame
            else:
                self.last_frame = max(
                    self.last_frame,
                    last_frame,
                )

        self.total_frames += frame_count

    @property
    def track_count(self) -> int:
        return len(self.track_ids)

    @property
    def duration_frames(self) -> int:
        if self.first_frame is None or self.last_frame is None:
            return 0

        return self.last_frame - self.first_frame + 1

    @property
    def is_fragmented(self) -> bool:
        return self.track_count > 1

    def set_team(self, team: Optional[str]) -> None:
        if team is not None:
            self.team = team

    def set_jersey_number(
        self,
        jersey_number: Optional[int],
    ) -> None:
        if jersey_number is not None:
            self.jersey_number = jersey_number

    def to_dict(self) -> dict:
        return {
            "global_id": self.global_id,
            "track_ids": list(self.track_ids),
            "class_name": self.class_name,
            "team": self.team,
            "jersey_number": self.jersey_number,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "total_frames": self.total_frames,
            "duration_frames": self.duration_frames,
            "track_count": self.track_count,
            "confidence": self.confidence,
            "identity_quality": self.identity_quality,
            "metadata": dict(self.metadata),
        }