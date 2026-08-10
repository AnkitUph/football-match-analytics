from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2.0,
            self.y2,
        )

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xyxy(self) -> list[float]:
        return [
            self.x1,
            self.y1,
            self.x2,
            self.y2,
        ]

    @classmethod
    def from_xyxy(cls, bbox) -> "BoundingBox":
        return cls(
            x1=float(bbox[0]),
            y1=float(bbox[1]),
            x2=float(bbox[2]),
            y2=float(bbox[3]),
        )


@dataclass
class Detection:
    frame_index: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass
class TrackObservation:
    frame_index: int
    bbox: BoundingBox
    confidence: float
    class_name: str
    team: Optional[str] = None
    jersey_number: Optional[int] = None
    appearance_embedding: Optional[list[float]] = None

    @property
    def position(self) -> tuple[float, float]:
        return self.bbox.bottom_center


@dataclass
class Track:
    local_id: int
    class_name: str
    observations: list[TrackObservation] = field(
        default_factory=list
    )
    active: bool = True
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None

    def add_observation(
        self,
        observation: TrackObservation,
    ) -> None:
        self.observations.append(observation)

        if self.first_frame is None:
            self.first_frame = observation.frame_index

        self.last_frame = observation.frame_index

    @property
    def frame_count(self) -> int:
        return len(self.observations)

    @property
    def last_observation(
        self,
    ) -> Optional[TrackObservation]:
        if not self.observations:
            return None

        return self.observations[-1]

    @property
    def first_observation(
        self,
    ) -> Optional[TrackObservation]:
        if not self.observations:
            return None

        return self.observations[0]

    @property
    def trajectory(self) -> list[tuple[float, float]]:
        return [
            observation.position
            for observation in self.observations
        ]


@dataclass
class Identity:
    identity_id: str
    class_name: str

    source_track_ids: list[int] = field(
        default_factory=list
    )

    observations: list[TrackObservation] = field(
        default_factory=list
    )

    team: Optional[str] = None
    jersey_number: Optional[int] = None

    identity_confidence: float = 0.0

    active: bool = True

    first_frame: Optional[int] = None
    last_frame: Optional[int] = None

    team_history: list[str] = field(
        default_factory=list
    )

    jersey_history: list[int] = field(
        default_factory=list
    )

    def add_track(
        self,
        track: Track,
    ) -> None:
        if track.local_id not in self.source_track_ids:
            self.source_track_ids.append(
                track.local_id
            )

        for observation in track.observations:
            self.add_observation(
                observation
            )

    def add_observation(
        self,
        observation: TrackObservation,
    ) -> None:
        self.observations.append(observation)

        if self.first_frame is None:
            self.first_frame = observation.frame_index

        self.last_frame = observation.frame_index

        if observation.team is not None:
            self.team_history.append(
                observation.team
            )

        if observation.jersey_number is not None:
            self.jersey_history.append(
                observation.jersey_number
            )

    @property
    def frame_count(self) -> int:
        return len(self.observations)

    @property
    def trajectory(self) -> list[tuple[float, float]]:
        return [
            observation.position
            for observation in self.observations
        ]

    @property
    def last_observation(
        self,
    ) -> Optional[TrackObservation]:
        if not self.observations:
            return None

        return self.observations[-1]


@dataclass
class FrameTracks:
    frame_index: int

    players: dict[int, TrackObservation] = field(
        default_factory=dict
    )

    goalkeepers: dict[int, TrackObservation] = field(
        default_factory=dict
    )

    referees: dict[int, TrackObservation] = field(
        default_factory=dict
    )

    ball: Optional[TrackObservation] = None


@dataclass
class MatchTracks:
    frames: list[FrameTracks] = field(
        default_factory=list
    )

    tracks: dict[int, Track] = field(
        default_factory=dict
    )

    identities: dict[str, Identity] = field(
        default_factory=dict
    )

    fps: float = 25.0

    frame_width: Optional[int] = None
    frame_height: Optional[int] = None

    def add_frame(
        self,
        frame: FrameTracks,
    ) -> None:
        self.frames.append(frame)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def player_identities(
        self,
    ) -> list[Identity]:
        return [
            identity
            for identity in self.identities.values()
            if identity.class_name == "player"
        ]

    @property
    def goalkeeper_identities(
        self,
    ) -> list[Identity]:
        return [
            identity
            for identity in self.identities.values()
            if identity.class_name == "goalkeeper"
        ]

    @property
    def referee_identities(
        self,
    ) -> list[Identity]:
        return [
            identity
            for identity in self.identities.values()
            if identity.class_name == "referee"
        ]