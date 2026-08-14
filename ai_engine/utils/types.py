"""
Shared data types passed between pipeline stages.

Keeping these in one module means every stage speaks the same "language" —
Stage 3 doesn't need to know how Stage 2 built a Tracklet, only what fields
it has.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ObjectClass(str, Enum):
    BALL = "ball"
    GOALKEEPER = "goalkeeper"
    PLAYER = "player"
    REFEREE = "referee"


class Team(str, Enum):
    TEAM_A = "team_a"
    TEAM_B = "team_b"
    REFEREE = "referee"
    UNKNOWN = "unknown"


class ShotType(str, Enum):
    MAIN_WIDE = "main_wide"
    CLOSE_UP = "close_up"
    REPLAY = "replay"
    OTHER = "other"


@dataclass
class Detection:
    """Single detection in a single frame. Output of Stage 1."""
    frame_idx: int
    cls: ObjectClass
    conf: float
    # Pixel-space bounding box, xyxy.
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class Tracklet:
    """
    A short spatial track from Stage 2, scoped to ONE continuous camera shot.
    Does NOT persist across cuts on its own — that's what Stage 5's identity
    association (backed by Stage 3's embeddings) is for.
    """
    track_id: int              # local ID, only unique within its shot
    shot_id: int                # which shot segment this belongs to (Stage 2.5)
    detections: list[Detection] = field(default_factory=list)
    cls: Optional[ObjectClass] = None

    # Populated by Stage 3.
    team: Team = Team.UNKNOWN
    reid_embedding: Optional[list[float]] = None   # running-average vector
    jersey_number: Optional[int] = None
    jersey_number_conf: float = 0.0

    @property
    def duration_frames(self) -> int:
        if not self.detections:
            return 0
        return self.detections[-1].frame_idx - self.detections[0].frame_idx + 1


@dataclass
class ShotSegment:
    """Output of Stage 2.5."""
    shot_id: int
    start_frame: int
    end_frame: int
    shot_type: ShotType


@dataclass
class PitchPoint:
    """Top-down pitch coordinate in meters, origin at pitch center."""
    x_m: float
    y_m: float


@dataclass
class MasterIdentity:
    """
    A permanent player identity in the ≤22-slot gallery. Output of Stage 5's
    identity association. This is the thing that must NOT change across
    camera cuts.
    """
    master_id: int
    team: Team
    reid_embedding: list[float]
    jersey_number: Optional[int] = None
    # frame_idx -> PitchPoint, only populated for frames inside main_wide shots
    trajectory: dict[int, PitchPoint] = field(default_factory=dict)


@dataclass
class BallTrajectoryPoint:
    frame_idx: int
    x_m: Optional[float]
    y_m: Optional[float]
    interpolated: bool = False


@dataclass
class Event:
    """Output of Stage 6."""
    event_type: str            # "pass", "shot", "possession_change", ...
    frame_idx: int
    player_master_id: Optional[int] = None
    target_master_id: Optional[int] = None  # e.g. pass recipient
    metadata: dict = field(default_factory=dict)
