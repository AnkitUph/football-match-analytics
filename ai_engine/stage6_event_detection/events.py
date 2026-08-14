"""
Stage 6: Rule-Based Event Detection.

Operates only on frames with valid pitch coordinates — i.e. frames inside
MAIN_WIDE shot segments where Stage 5's homography succeeded. Events
during close-ups/replays are skipped here; if you want continuity across
those gaps later, that's a future enhancement (e.g. carrying possession
state across a cut using the last known state), not part of this stub.
"""

from ai_engine.config import EventDetectionConfig
from ai_engine.utils.types import BallTrajectoryPoint, Event, MasterIdentity


def detect_possession(
    ball_trajectory: list[BallTrajectoryPoint],
    identities: list[MasterIdentity],
    config: EventDetectionConfig,
) -> list[Event]:
    """
    For each frame with a valid ball position, find the closest player
    within config.possession_radius_m. A possession event is only
    confirmed if the same player stays closest for
    config.possession_min_frames consecutive frames (avoids flickering
    on every frame where two players are near the ball).

    TODO: implement — for each ball point, compute distance to every
    identity's trajectory[frame_idx] (skip identities with no pitch
    position that frame), track a rolling "current closest player" state
    machine, and emit a "possession_change" Event whenever the confirmed
    closest player changes.
    """
    raise NotImplementedError("detect_possession is stubbed.")


def detect_passes(
    ball_trajectory: list[BallTrajectoryPoint],
    possession_events: list[Event],
    config: EventDetectionConfig,
) -> list[Event]:
    """
    A pass = a rapid ball vector/speed change originating near one player
    and ending near a DIFFERENT player on the SAME team, inferred from
    consecutive possession_change events plus the ball's velocity profile
    between them (config.pass_min_ball_speed_change as the trigger
    threshold to distinguish an intentional pass from a loose ball roll).

    TODO: implement using possession_events as anchor points.
    """
    raise NotImplementedError("detect_passes is stubbed.")


def detect_shots(
    ball_trajectory: list[BallTrajectoryPoint],
    goal_box_pitch_coords: tuple[float, float, float, float],
) -> list[Event]:
    """
    A shot = ball trajectory accelerating toward the goal box area from
    within pitch boundaries. goal_box_pitch_coords is
    (x_min, y_min, x_max, y_max) in pitch meters for the target goal.

    TODO: implement — look for a ball velocity vector pointed at the goal
    box with increasing speed, originating from open play (not already
    inside the box, which would just be an in-box touch).
    """
    raise NotImplementedError("detect_shots is stubbed.")
