"""
Stage 6: Rule-Based Event Detection.

Operates only on frames with valid pitch coordinates — i.e. frames inside
MAIN_WIDE shot segments where Stage 5's homography succeeded. Events
during close-ups/replays are skipped here; if you want continuity across
those gaps later, that's a future enhancement (e.g. carrying possession
state across a cut using the last known state), not part of this stub.
"""

from collections import Counter, deque

from ai_engine.config import EventDetectionConfig
from ai_engine.utils.types import BallTrajectoryPoint, Event, MasterIdentity


def detect_possession(
    ball_trajectory: list[BallTrajectoryPoint],
    identities: list[MasterIdentity],
    config: EventDetectionConfig,
) -> list[Event]:
    """
    For each frame with a valid ball position, finds the closest player
    within config.possession_radius_m. A possession event confirms when
    one player is the majority closest-within-radius player across a
    rolling time window — NOT strict consecutive-frame matching.

    WHY MAJORITY VOTE, NOT STRICT CONSECUTIVE MATCHING (found via testing
    on real footage): a naive "same closest player for N consecutive
    frames, reset on any change" approach fails on real contested-ball
    situations. Tested on a real 50/50 challenge in match footage: two
    players' distances to the ball flickered back and forth (0.9m-3.7m)
    as they both contested it — the closest player legitimately
    alternated between them frame to frame. Strict consecutive matching
    never confirmed possession at all, resetting its counter on every
    flicker. Majority vote across a rolling window tolerates this kind
    of brief flicker while still correctly NOT firing during genuinely
    unresolved scrappy passages (validated: an earlier contested moment
    in the same clip, where the closest player changed between 4
    different tracklets with no clear majority, correctly produced no
    event).

    WHY A TIME-BASED WINDOW, NOT A SAMPLE-COUNT WINDOW: ball_trajectory
    may be sampled at a different rate than the player tracking data
    (e.g. Stage 4's ball extraction at a lower target_fps than Stage 2's
    tracker) — using config.possession_min_frames as a raw sample count
    silently changes the real time window depending on ball_trajectory's
    sample rate. This function converts to a sample count internally
    using fps, so behavior stays consistent regardless of how densely
    ball_trajectory happens to be sampled.
    """
    RADIUS = config.possession_radius_m
    MIN_WINDOW_SEC = config.possession_min_frames / 25.0  # config value assumed relative to 25fps baseline
    ball_fps_estimate = _estimate_sample_rate(ball_trajectory)
    window_size = max(2, round(MIN_WINDOW_SEC * ball_fps_estimate))

    identity_by_id = {i.master_id: i for i in identities}
    recent: deque = deque(maxlen=window_size)
    events: list[Event] = []
    current_holder: int | None = None

    valid_points = [p for p in ball_trajectory if p.x_m is not None]
    for point in valid_points:
        closest_id, closest_dist = None, float("inf")
        for identity in identities:
            pos = identity.trajectory.get(point.frame_idx)
            if pos is None:
                continue
            dist = ((pos.x_m - point.x_m) ** 2 + (pos.y_m - point.y_m) ** 2) ** 0.5
            if dist < closest_dist:
                closest_dist = dist
                closest_id = identity.master_id

        within_radius = closest_id if closest_dist <= RADIUS else None
        recent.append(within_radius)

        if len(recent) == window_size:
            candidates = [c for c in recent if c is not None]
            if candidates:
                winner, count = Counter(candidates).most_common(1)[0]
                if count >= window_size * 0.6 and winner != current_holder:
                    current_holder = winner
                    events.append(
                        Event(
                            event_type="possession_change",
                            frame_idx=point.frame_idx,
                            player_master_id=winner,
                        )
                    )

    return events


def _estimate_sample_rate(trajectory: list[BallTrajectoryPoint]) -> float:
    """Estimates the effective sample rate of a trajectory from its frame
    index spacing, so time-based windows stay correct regardless of the
    trajectory's actual sampling density."""
    frame_indices = sorted(p.frame_idx for p in trajectory if p.x_m is not None)
    if len(frame_indices) < 2:
        return 25.0  # fallback assumption
    diffs = [b - a for a, b in zip(frame_indices, frame_indices[1:])]
    avg_step = sum(diffs) / len(diffs)
    return 25.0 / avg_step if avg_step > 0 else 25.0


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
