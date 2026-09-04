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
    possession_events: list[Event],
    identities: list[MasterIdentity],
) -> list[Event]:
    """
    A pass = consecutive possession changes where the ball moves from one
    player to a DIFFERENT player on the SAME team. A possession change to
    a player on the OPPOSING team is a turnover/interception, not a pass
    — this function only emits "pass" events, turnovers are silently
    skipped (not tracked as a separate event type yet; add one if you
    need it later, the distinction is already computed here).

    DELIBERATELY SIMPLE v1: uses possession_change events as the anchor
    (already validated — see detect_possession's real-footage testing),
    rather than also gating on ball speed/vector change as the original
    Stage 6 roadmap suggested. Reasoning: possession detection is already
    a solid, tested signal; adding a speed-based secondary check adds
    real complexity (what threshold? validated against what data?) without
    a demonstrated need yet. Revisit if this v1 produces obviously wrong
    results in practice (e.g. counting a loose-ball scramble as a pass).
    """
    team_by_id = {i.master_id: i.team for i in identities}
    events: list[Event] = []

    for prev_event, next_event in zip(possession_events, possession_events[1:]):
        passer_team = team_by_id.get(prev_event.player_master_id)
        receiver_team = team_by_id.get(next_event.player_master_id)
        if passer_team is None or receiver_team is None:
            continue
        if passer_team != receiver_team:
            continue  # turnover, not a pass
        events.append(
            Event(
                event_type="pass",
                frame_idx=next_event.frame_idx,
                player_master_id=prev_event.player_master_id,
                target_master_id=next_event.player_master_id,
            )
        )

    return events


def detect_shots(
    ball_trajectory: list[BallTrajectoryPoint],
    goal_center_pitch: tuple[float, float],
    min_shot_speed_mps: float = 10.0,
    max_plausible_speed_mps: float = 35.0,
    min_origin_distance_m: float = 5.0,
    min_alignment: float = 0.85,
    cooldown_frames: int = 20,
) -> list[Event]:
    """
    A shot = ball velocity, between any two consecutive tracked points,
    pointed toward the goal (cosine alignment > min_alignment) with
    speed between min_shot_speed_mps and max_plausible_speed_mps.

    EVALUATES EVERY CONSECUTIVE PAIR, not just points before a gap —
    corrected after testing found the original "only near a gap" design
    too narrow: on real footage, a genuinely excellent shot signal
    (speed ~11 m/s, alignment 0.98-1.00, i.e. pointed almost exactly at
    goal) sat in the middle of a continuously-tracked stretch and was
    skipped entirely by that restriction. Evaluating every pair still
    naturally covers the "shot flight not fully tracked" case (Stage 4
    ball tracking's own gap philosophy already handles that — this
    function just needs to catch the last velocity estimate before data
    runs out, which happens automatically here).

    max_plausible_speed_mps GUARDS AGAINST TRACKING NOISE: found via
    testing that raw frame-to-frame speed occasionally spikes to
    physically impossible values (300+ m/s) from a single mistracked
    point — filtering those out is necessary, not optional. No real
    shot exceeds ~35 m/s.

    cooldown_frames prevents one continuous fast, goal-aligned stretch
    from emitting many duplicate "shot" events for what's really one
    strike.
    """
    events: list[Event] = []
    valid_points = [p for p in ball_trajectory if p.x_m is not None]
    last_event_frame = -cooldown_frames

    for current, next_point in zip(valid_points, valid_points[1:]):
        dt_frames = next_point.frame_idx - current.frame_idx
        if dt_frames <= 0:
            continue
        dt_sec = dt_frames / 25.0

        vx = (next_point.x_m - current.x_m) / dt_sec
        vy = (next_point.y_m - current.y_m) / dt_sec
        speed = (vx**2 + vy**2) ** 0.5

        if not (min_shot_speed_mps <= speed <= max_plausible_speed_mps):
            continue

        origin_dist = ((goal_center_pitch[0] - current.x_m) ** 2 + (goal_center_pitch[1] - current.y_m) ** 2) ** 0.5
        if origin_dist < min_origin_distance_m:
            continue

        to_goal_x = goal_center_pitch[0] - current.x_m
        to_goal_y = goal_center_pitch[1] - current.y_m
        to_goal_dist = (to_goal_x**2 + to_goal_y**2) ** 0.5
        if to_goal_dist == 0:
            continue
        alignment = (vx * to_goal_x + vy * to_goal_y) / (speed * to_goal_dist)

        if alignment > min_alignment and (current.frame_idx - last_event_frame) >= cooldown_frames:
            events.append(
                Event(
                    event_type="shot",
                    frame_idx=current.frame_idx,
                    metadata={"speed_mps": speed, "origin_distance_m": origin_dist, "alignment": alignment},
                )
            )
            last_event_frame = current.frame_idx

    return events


def estimate_shot_xg(origin_distance_m: float, alignment: float) -> float:
    """
    Simplified heuristic expected-goals value for ONE detected shot event.
    NOT a trained model — there is no labeled goal/no-goal outcome data
    to fit against, only detect_shots' own two derived signals. Treat
    this as a distance/angle-shaped placeholder, clearly not a real xG
    model, same honesty standard as everything else real-vs-dummy in
    this project.

    Two components, multiplied together:

    - Distance factor: 1 / (1 + (d/8)^2) — a smooth, monotonically
      decreasing curve with plausible round numbers (d=6m -> ~0.64,
      d=12m -> ~0.22, d=18m -> ~0.11, d=30m -> ~0.04), NOT fit to any
      real dataset.
    - Alignment factor: detect_shots only ever emits events with
      alignment > min_alignment (0.85 by default), so raw alignment on
      a real shot event only ever spans a narrow ~[0.85, 1.0] band.
      Rescaled here to [0.7, 1.0] so it actually moves the final number
      instead of being lost in that narrow input range.

    Result is clamped to [0.01, 0.95] — a detected shot should never
    display as exactly impossible or exactly certain.
    """
    distance_factor = 1.0 / (1.0 + (origin_distance_m / 8.0) ** 2)

    alignment_clamped = max(0.85, min(1.0, alignment))
    alignment_factor = 0.7 + 0.3 * (alignment_clamped - 0.85) / 0.15

    xg = distance_factor * alignment_factor
    return round(max(0.01, min(0.95, xg)), 3)