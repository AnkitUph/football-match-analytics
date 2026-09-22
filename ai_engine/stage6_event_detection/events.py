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
    identity_by_id = {i.master_id: i for i in identities}
    events: list[Event] = []

    for prev_event, next_event in zip(possession_events, possession_events[1:]):
        passer_team = team_by_id.get(prev_event.player_master_id)
        receiver_team = team_by_id.get(next_event.player_master_id)
        if passer_team is None or receiver_team is None:
            continue
        if passer_team != receiver_team:
            continue  # turnover, not a pass
        if prev_event.player_master_id == next_event.player_master_id:
            continue

        p1 = identity_by_id.get(prev_event.player_master_id)
        p2 = identity_by_id.get(next_event.player_master_id)
        pos1 = p1.trajectory.get(prev_event.frame_idx) if p1 else None
        pos2 = p2.trajectory.get(next_event.frame_idx) if p2 else None
        dist = 0.0
        if pos1 and pos2:
            dist = round(((pos2.x_m - pos1.x_m) ** 2 + (pos2.y_m - pos1.y_m) ** 2) ** 0.5, 1)

        dt_frames = next_event.frame_idx - prev_event.frame_idx
        dt_sec = max(0.04, dt_frames / 25.0)
        speed = round(dist / dt_sec, 1)

        # Kinematic filters: pass must cover at least 2.0m, take >= 2 frames, and speed <= 45.0 m/s
        if dist < 2.0 or dt_frames < 2 or speed > 45.0:
            continue

        events.append(
            Event(
                event_type="pass",
                frame_idx=next_event.frame_idx,
                player_master_id=prev_event.player_master_id,
                target_master_id=next_event.player_master_id,
                metadata={
                    "start_frame": prev_event.frame_idx,
                    "distance_m": dist,
                    "speed_mps": speed,
                    "start_x": round(pos1.x_m, 2) if pos1 else 0.0,
                    "start_y": round(pos1.y_m, 2) if pos1 else 0.0,
                    "end_x": round(pos2.x_m, 2) if pos2 else 0.0,
                    "end_y": round(pos2.y_m, 2) if pos2 else 0.0,
                    "is_completed": True,
                },
            )
        )

    return events


def compute_continuous_possession(
    ball_trajectory: list[BallTrajectoryPoint],
    identities: list[MasterIdentity],
    fps: float = 25.0,
    control_radius_m: float = 4.5,
) -> dict:
    """
    Computes frame-by-frame ball possession across all frames with valid ball pitch coordinates:
    - Finds the closest player to the ball at each frame.
    - If distance <= control_radius_m, attributes possession to that player's team.
    - If distance > control_radius_m (ball in flight or uncontested), attributes to the
      most recent controlling team for up to 1.5 seconds (38 frames).
    - Returns:
      - 'percentages': {Team.TEAM_A: float, Team.TEAM_B: float}
      - 'frame_counts': {Team.TEAM_A: int, Team.TEAM_B: int}
      - 'possession_events': list[Event] (discrete possession_change events)
    """
    from ai_engine.utils.types import Team

    identity_by_id = {i.master_id: i for i in identities}
    team_by_id = {i.master_id: i.team for i in identities}

    valid_points = [p for p in ball_trajectory if p.x_m is not None]
    if not valid_points:
        return {
            "percentages": {Team.TEAM_A: 50.0, Team.TEAM_B: 50.0},
            "frame_counts": {Team.TEAM_A: 0, Team.TEAM_B: 0},
            "possession_events": [],
        }

    frame_counts = {Team.TEAM_A: 0, Team.TEAM_B: 0}
    possession_events = []

    current_holder_id = None
    current_team = None
    frames_since_control = 0
    max_carry_frames = int(1.5 * fps)

    # Debounce discrete possession change events with a rolling window
    # Prevents 1-frame micro-oscillations during contested tackles
    window_size = 5
    recent_closest: deque = deque(maxlen=window_size)
    event_holder_id = None

    for point in valid_points:
        closest_id, closest_dist = None, float("inf")
        for identity in identities:
            pos = identity.trajectory.get(point.frame_idx)
            if pos is not None:
                d = ((pos.x_m - point.x_m) ** 2 + (pos.y_m - point.y_m) ** 2) ** 0.5
                if d < closest_dist:
                    closest_dist = d
                    closest_id = identity.master_id

        if closest_id is not None and closest_dist <= control_radius_m:
            p_team = team_by_id.get(closest_id)
            if p_team in frame_counts:
                frames_since_control = 0
                current_team = p_team
            recent_closest.append(closest_id)
        else:
            frames_since_control += 1
            if frames_since_control > max_carry_frames:
                current_team = None
            recent_closest.append(None)

        if current_team in frame_counts:
            frame_counts[current_team] += 1

        if len(recent_closest) == window_size:
            candidates = [c for c in recent_closest if c is not None]
            if candidates:
                winner, count = Counter(candidates).most_common(1)[0]
                if count >= 3 and winner != event_holder_id:
                    event_holder_id = winner
                    w_team = team_by_id.get(winner)
                    if w_team in frame_counts:
                        possession_events.append(
                            Event(
                                event_type="possession_change",
                                frame_idx=point.frame_idx,
                                player_master_id=winner,
                                metadata={
                                    "team": w_team.value if hasattr(w_team, "value") else str(w_team),
                                    "dist_m": round(closest_dist, 2),
                                },
                            )
                        )

    total_frames = sum(frame_counts.values())
    if total_frames > 0:
        pct_a = round(100.0 * frame_counts[Team.TEAM_A] / total_frames, 1)
        pct_b = round(100.0 - pct_a, 1)
    else:
        pct_a, pct_b = 50.0, 50.0

    return {
        "percentages": {Team.TEAM_A: pct_a, Team.TEAM_B: pct_b},
        "frame_counts": frame_counts,
        "possession_events": possession_events,
    }


def detect_passes_with_metadata(
    possession_events: list[Event],
    identities: list[MasterIdentity],
    ball_trajectory: list[BallTrajectoryPoint] | None = None,
    fps: float = 25.0,
) -> list[dict]:
    """
    Extracts pass events with complete metadata:
    - start and arrival frame indices
    - passer track_id and receiver track_id
    - passer team and receiver team
    - start pitch coordinates and end pitch coordinates
    - pass distance in meters
    - is_completed (True if teammate received, False if intercepted/turnover)
    """
    identity_by_id = {i.master_id: i for i in identities}
    team_by_id = {i.master_id: i.team for i in identities}

    passes = []
    for prev_ev, next_ev in zip(possession_events, possession_events[1:]):
        p1_id = prev_ev.player_master_id
        p2_id = next_ev.player_master_id
        if p1_id == p2_id:
            continue

        t1 = team_by_id.get(p1_id)
        t2 = team_by_id.get(p2_id)
        if t1 is None or t2 is None:
            continue

        p1_ident = identity_by_id.get(p1_id)
        p2_ident = identity_by_id.get(p2_id)
        pos1 = p1_ident.trajectory.get(prev_ev.frame_idx) if p1_ident else None
        pos2 = p2_ident.trajectory.get(next_ev.frame_idx) if p2_ident else None

        sx = round(pos1.x_m, 2) if pos1 else 0.0
        sy = round(pos1.y_m, 2) if pos1 else 0.0
        ex = round(pos2.x_m, 2) if pos2 else 0.0
        ey = round(pos2.y_m, 2) if pos2 else 0.0

        dist = round(((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5, 1)
        dt_frames = next_ev.frame_idx - prev_ev.frame_idx
        dt_sec = max(0.04, dt_frames / fps)
        speed = round(dist / dt_sec, 1)

        # Kinematic filters: discard instantaneous 1-frame or unphysical teleports
        if dist < 2.0 or dt_frames < 2 or speed > 45.0:
            continue

        is_completed = (t1 == t2)
        video_minute = max(1, round((prev_ev.frame_idx / fps) / 60.0))

        passes.append({
            "frame_idx": next_ev.frame_idx,
            "start_frame": prev_ev.frame_idx,
            "minute": video_minute,
            "passer_track_id": p1_id,
            "receiver_track_id": p2_id,
            "passer_team": t1.value if hasattr(t1, "value") else str(t1),
            "receiver_team": t2.value if hasattr(t2, "value") else str(t2),
            "start_x": sx,
            "start_y": sy,
            "end_x": ex,
            "end_y": ey,
            "distance_m": dist,
            "speed_mps": speed,
            "is_completed": is_completed,
        })

    return passes


def detect_corner_kicks(
    ball_trajectory: list[BallTrajectoryPoint],
    identities: list[MasterIdentity],
    fps: float = 25.0,
) -> list[dict]:
    """
    Detects corner kicks: ball starting from near corner flag coordinates
    (|x| >= 47.0m, |y| >= 28.0m) and traveling into the penalty box area.
    """
    valid_points = [p for p in ball_trajectory if p.x_m is not None]
    corners = []
    last_corner_frame = -100

    for cur, nxt in zip(valid_points, valid_points[1:]):
        if (cur.frame_idx - last_corner_frame) < 100:
            continue

        if abs(cur.x_m) >= 47.0 and abs(cur.y_m) >= 28.0:
            # Check if ball enters box (|x| >= 34.0, |y| <= 20.0) in subsequent frames
            dt_frames = nxt.frame_idx - cur.frame_idx
            if 0 < dt_frames <= 50 and abs(nxt.x_m) >= 34.0 and abs(nxt.y_m) <= 20.0:
                closest_id, closest_team = None, None
                for ident in identities:
                    pos = ident.trajectory.get(cur.frame_idx)
                    if pos:
                        d = ((pos.x_m - cur.x_m) ** 2 + (pos.y_m - cur.y_m) ** 2) ** 0.5
                        if d <= 4.0:
                            closest_id = ident.master_id
                            closest_team = ident.team.value if hasattr(ident.team, "value") else str(ident.team)
                            break

                corners.append({
                    "frame_idx": cur.frame_idx,
                    "minute": max(1, round((cur.frame_idx / fps) / 60.0)),
                    "team": closest_team or ("team_a" if cur.x_m > 0 else "team_b"),
                    "track_id": closest_id,
                    "pitch_x": round(cur.x_m, 2),
                    "pitch_y": round(cur.y_m, 2),
                })
                last_corner_frame = cur.frame_idx

    return corners


def detect_shots(
    ball_trajectory: list[BallTrajectoryPoint],
    goal_centers_pitch: tuple[tuple[float, float], ...] | tuple[float, float] = ((52.5, 0.0), (-52.5, 0.0)),
    min_shot_speed_mps: float = 12.0,
    max_plausible_speed_mps: float = 38.0,
    min_origin_distance_m: float = 3.0,
    max_origin_distance_m: float = 35.0,
    min_alignment: float = 0.88,
    cooldown_frames: int = 40,
    identities: list[MasterIdentity] | None = None,
    pass_intervals: list[tuple[int, int]] | None = None,
) -> list[Event]:
    """
    Detects genuine shots on goal:
    - Origin distance <= max_origin_distance_m (default 35.0m, attacking third).
      Excludes kicks from midfield or defense (>35m away).
    - Speed between min_shot_speed_mps (12.0 m/s = 43 km/h) and max_plausible_speed_mps.
    - Directional alignment >= min_alignment towards the target goal.
    - Ball trajectory must actually continue approaching the target goal over subsequent
      frames (min 3.5m approach), excluding 1-frame deflections during tackles/tussles.
    - Excludes actions occurring during completed teammate pass intervals.
    - If identities are provided, validates that the shooter is attacking the target goal
      (not clearing towards their own defending goal).
    """
    from ai_engine.utils.types import Team

    # Support both a single goal tuple (52.5, 0.0) or a tuple of goal tuples
    if goal_centers_pitch and isinstance(goal_centers_pitch[0], (int, float)):
        goal_centers: tuple[tuple[float, float], ...] = (goal_centers_pitch,)  # type: ignore
    else:
        goal_centers = goal_centers_pitch  # type: ignore

    events: list[Event] = []
    valid_points = [p for p in ball_trajectory if p.x_m is not None]
    if not valid_points:
        return events

    ball_map = {p.frame_idx: p for p in valid_points}
    last_event_frame = -cooldown_frames

    # Infer team defending goals from relative team positions (if available)
    team_defending_goal: dict[Team, tuple[float, float]] = {}
    if identities:
        team_medians: dict[Team, float] = {}
        for team in (Team.TEAM_A, Team.TEAM_B):
            xs = [
                pt.x_m
                for ident in identities
                if ident.team == team
                for pt in ident.trajectory.values()
                if abs(pt.x_m) <= 55.0
            ]
            if xs:
                team_medians[team] = sorted(xs)[len(xs) // 2]
        if Team.TEAM_A in team_medians and Team.TEAM_B in team_medians:
            # Defending goal can only be reliably inferred if the two teams are clearly separated
            # across opposite pitch halves (i.e. one team's median is on negative half, other on positive).
            # When both medians share the same sign, the footage is localized in one attacking third
            # and comparing medians within the same half does not indicate defending goal direction.
            if team_medians[Team.TEAM_A] * team_medians[Team.TEAM_B] < 0:
                if team_medians[Team.TEAM_A] < team_medians[Team.TEAM_B]:
                    team_defending_goal[Team.TEAM_A] = (-52.5, 0.0)
                    team_defending_goal[Team.TEAM_B] = (52.5, 0.0)
                else:
                    team_defending_goal[Team.TEAM_A] = (52.5, 0.0)
                    team_defending_goal[Team.TEAM_B] = (-52.5, 0.0)

    for current, next_point in zip(valid_points, valid_points[1:]):
        dt_frames = next_point.frame_idx - current.frame_idx
        if dt_frames <= 0:
            continue
        dt_sec = dt_frames / 25.0

        vx = (next_point.x_m - current.x_m) / dt_sec
        vy = (next_point.y_m - current.y_m) / dt_sec
        speed = (vx**2 + vy**2) ** 0.5

        # Exclude frames that are part of an already-identified completed pass
        if pass_intervals and any(start <= current.frame_idx <= end for start, end in pass_intervals):
            continue

        best = None  # (goal_center, origin_dist, alignment)
        for goal_center_pitch in goal_centers:
            # Must originate strictly inside pitch boundaries
            if not (abs(current.x_m) <= 51.5 and abs(current.y_m) <= 33.5):
                continue
            # Must be approaching the goal line from within the pitch (not moving away or from behind goal)
            if goal_center_pitch[0] > 0 and (current.x_m >= 51.5 or vx <= 0):
                continue
            if goal_center_pitch[0] < 0 and (current.x_m <= -51.5 or vx >= 0):
                continue

            origin_dist = ((goal_center_pitch[0] - current.x_m) ** 2 + (goal_center_pitch[1] - current.y_m) ** 2) ** 0.5
            # Must be within realistic shooting distance (e.g. <= 35m from goal)
            if not (min_origin_distance_m <= origin_dist <= max_origin_distance_m):
                continue

            # Within penalty box (<= 16.5m), allow placed finishes down to 9.5 m/s (34 km/h); outside box requires min_shot_speed_mps
            eff_min_speed = 9.5 if origin_dist <= 16.5 else min_shot_speed_mps
            if not (eff_min_speed <= speed <= max_plausible_speed_mps):
                continue

            to_goal_x = goal_center_pitch[0] - current.x_m
            to_goal_y = goal_center_pitch[1] - current.y_m
            to_goal_dist = (to_goal_x**2 + to_goal_y**2) ** 0.5
            if to_goal_dist == 0:
                continue
            alignment = (vx * to_goal_x + vy * to_goal_y) / (speed * to_goal_dist)

            if alignment >= min_alignment and (best is None or alignment > best[2]):
                best = (goal_center_pitch, origin_dist, alignment)

        if best is not None and (current.frame_idx - last_event_frame) >= cooldown_frames:
            goal_center_pitch, origin_dist, alignment = best

            # Verify sustained trajectory towards the goal (excludes momentary 1-frame tackle deflections)
            future_pts = [
                ball_map[f]
                for f in range(current.frame_idx + 1, min(current.frame_idx + 18, current.frame_idx + 30))
                if f in ball_map
            ]
            if future_pts:
                min_future_dist = min(
                    ((goal_center_pitch[0] - p.x_m) ** 2 + (goal_center_pitch[1] - p.y_m) ** 2) ** 0.5
                    for p in future_pts
                )
                if (origin_dist - min_future_dist) < 3.5:
                    continue  # Did not travel towards goal
            else:
                continue

            # Verify shooter's team is attacking this goal, not defending it
            if identities:
                closest_team = None
                closest_d = float("inf")
                for ident in identities:
                    pos = ident.trajectory.get(current.frame_idx)
                    if pos:
                        d = ((pos.x_m - current.x_m) ** 2 + (pos.y_m - current.y_m) ** 2) ** 0.5
                        if d < closest_d:
                            closest_d = d
                            closest_team = ident.team
                if closest_team == Team.REFEREE:
                    continue
                if closest_team and team_defending_goal.get(closest_team) == goal_center_pitch:
                    continue  # Clearance / backpass towards own defending goal

            # Extrapolate ball path to goal line (x = goal_center_pitch[0])
            is_on_target = False
            if abs(vx) > 0.1:
                t_goal = (goal_center_pitch[0] - current.x_m) / vx
                if t_goal > 0:
                    y_at_goal = current.y_m + vy * t_goal
                    # Goal posts are at y = -3.66m and +3.66m (7.32m wide)
                    # Use 4.2m tolerance to account for posts/crossbar & tracking noise
                    if abs(y_at_goal) <= 4.2:
                        is_on_target = True

            events.append(
                Event(
                    event_type="shot",
                    frame_idx=current.frame_idx,
                    metadata={
                        "speed_mps": round(speed, 1),
                        "origin_distance_m": round(origin_dist, 1),
                        "alignment": round(alignment, 2),
                        "target_goal": goal_center_pitch,
                        "is_on_target": is_on_target,
                    },
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


def detect_extended_match_events(
    ball_trajectory: list[BallTrajectoryPoint],
    identities: list[MasterIdentity],
    possession_events: list[Event],
    shot_events: list[Event],
    fps: float = 25.0,
) -> dict:
    """
    Computes derived football match and player events from tracking data:
    - Passes attempted and completed per player and per team
    - Defending actions: tackles, interceptions, clearances
    - Attacking actions: dribbles completed, key passes
    - Set pieces: corners
    """
    from collections import defaultdict

    identity_by_id = {i.master_id: i for i in identities}
    team_by_id = {i.master_id: i.team for i in identities}

    passes_completed = defaultdict(int)
    passes_attempted = defaultdict(int)
    tackles = defaultdict(int)
    interceptions = defaultdict(int)
    clearances = defaultdict(int)
    dribbles = defaultdict(int)
    key_passes = defaultdict(int)

    team_passes_completed = defaultdict(int)
    team_passes_attempted = defaultdict(int)

    # 1. Analyze possession transitions for passes, interceptions, and tackles
    for idx, event in enumerate(possession_events):
        p1_id = event.player_master_id
        t1 = team_by_id.get(p1_id)
        if idx + 1 < len(possession_events):
            next_ev = possession_events[idx + 1]
            p2_id = next_ev.player_master_id
            t2 = team_by_id.get(p2_id)

            if t1 is not None and t2 is not None:
                if t1 == t2 and p1_id != p2_id:
                    # Completed pass between teammates
                    passes_completed[p1_id] += 1
                    passes_attempted[p1_id] += 1
                    team_passes_completed[t1] += 1
                    team_passes_attempted[t1] += 1

                    # Check for Key Pass: receiver takes a shot within 10s (250 frames)
                    for s in shot_events:
                        if 0 < (s.frame_idx - next_ev.frame_idx) <= int(10.0 * fps):
                            key_passes[p1_id] += 1
                            break
                elif t1 != t2:
                    # Possession changed to opposing team
                    passes_attempted[p1_id] += 1
                    team_passes_attempted[t1] += 1

                    # Duel check: distance between p1 and p2 around transition frame
                    p1_ident = identity_by_id.get(p1_id)
                    p2_ident = identity_by_id.get(p2_id)
                    pos1 = p1_ident.trajectory.get(next_ev.frame_idx) if p1_ident else None
                    pos2 = p2_ident.trajectory.get(next_ev.frame_idx) if p2_ident else None

                    if pos1 and pos2:
                        dist = ((pos1.x_m - pos2.x_m) ** 2 + (pos1.y_m - pos2.y_m) ** 2) ** 0.5
                        if dist <= 2.8:
                            # Close-quarters challenge: tackle won by p2
                            tackles[p2_id] += 1
                        else:
                            # Loose / passed ball intercepted by p2
                            interceptions[p2_id] += 1
                    else:
                        interceptions[p2_id] += 1
        else:
            # Last possession event
            if t1 is not None:
                passes_attempted[p1_id] += 1
                team_passes_attempted[t1] += 1

    # Ensure passes_attempted is always at least passes_completed
    for pid in list(passes_completed.keys()):
        if passes_attempted[pid] < passes_completed[pid]:
            passes_attempted[pid] = passes_completed[pid]

    # 2. Clearances: high-speed kicks away from defensive third (|x| > 25m)
    ball_valid = [p for p in ball_trajectory if p.x_m is not None]
    last_clearance_frame = -50
    for cur, nxt in zip(ball_valid, ball_valid[1:]):
        dt_frames = nxt.frame_idx - cur.frame_idx
        if dt_frames <= 0 or (cur.frame_idx - last_clearance_frame) < 50:
            continue
        dt_sec = dt_frames / fps
        vx = (nxt.x_m - cur.x_m) / dt_sec
        vy = (nxt.y_m - cur.y_m) / dt_sec
        speed = (vx ** 2 + vy ** 2) ** 0.5

        if speed > 11.0 and abs(cur.x_m) > 25.0:
            closest_id, closest_dist = None, float("inf")
            for ident in identities:
                pos = ident.trajectory.get(cur.frame_idx)
                if pos:
                    d = ((pos.x_m - cur.x_m) ** 2 + (pos.y_m - cur.y_m) ** 2) ** 0.5
                    if d < closest_dist:
                        closest_dist = d
                        closest_id = ident.master_id

            if closest_id is not None and closest_dist <= 3.5:
                # If in negative third (x < -25), clearing toward positive x (vx > 2.0)
                # If in positive third (x > 25), clearing toward negative x (vx < -2.0)
                if (cur.x_m < -25.0 and vx > 2.0) or (cur.x_m > 25.0 and vx < -2.0):
                    clearances[closest_id] += 1
                    last_clearance_frame = cur.frame_idx

    # 3. Dribbles Completed: player moves with ball >= 4.5m with opponent nearby
    ball_by_frame = {b.frame_idx: b for b in ball_valid}
    for ident in identities:
        frames = sorted(ident.trajectory.keys())
        if len(frames) < 10:
            continue
        dribble_start = None
        for f in frames:
            b = ball_by_frame.get(f)
            pos = ident.trajectory[f]
            if b is not None and ((b.x_m - pos.x_m) ** 2 + (b.y_m - pos.y_m) ** 2) ** 0.5 <= 3.2:
                if dribble_start is None:
                    dribble_start = pos
                else:
                    disp = ((pos.x_m - dribble_start.x_m) ** 2 + (pos.y_m - dribble_start.y_m) ** 2) ** 0.5
                    if disp >= 4.5:
                        dribbles[ident.master_id] += 1
                        dribble_start = None
            else:
                dribble_start = None

    return {
        "passes_completed": passes_completed,
        "passes_attempted": passes_attempted,
        "team_passes_completed": team_passes_completed,
        "team_passes_attempted": team_passes_attempted,
        "tackles": tackles,
        "interceptions": interceptions,
        "clearances": clearances,
        "dribbles": dribbles,
        "key_passes": key_passes,
    }


def compute_player_physical_metrics(
    identity: MasterIdentity,
    fps: float = 25.0,
    max_speed_mps: float = 11.5,
) -> dict:
    """
    Computes top speed, average speed, and distance covered for a single player
    trajectory, filtering out tracking noise spikes.
    """
    frames = sorted(identity.trajectory.keys())
    if len(frames) < 2:
        return {
            "distance_m": 0.0,
            "top_speed_kmh": 0.0,
            "average_speed_kmh": 0.0,
            "minutes_played": 1,
        }

    total_dist = 0.0
    speeds = []

    for f1, f2 in zip(frames, frames[1:]):
        dt_frames = f2 - f1
        if dt_frames <= 0:
            continue
        dt_sec = dt_frames / fps
        p1 = identity.trajectory[f1]
        p2 = identity.trajectory[f2]
        dist = ((p2.x_m - p1.x_m) ** 2 + (p2.y_m - p1.y_m) ** 2) ** 0.5
        inst_speed = dist / dt_sec

        if inst_speed <= max_speed_mps:
            total_dist += dist
            speeds.append(inst_speed)

    if speeds:
        sorted_speeds = sorted(speeds)
        p95_idx = int(len(sorted_speeds) * 0.95)
        peak_mps = sorted_speeds[min(p95_idx, len(sorted_speeds) - 1)]
    else:
        peak_mps = 0.0

    total_time_sec = (frames[-1] - frames[0]) / fps if frames[-1] > frames[0] else 1.0
    avg_speed_mps = total_dist / max(1.0, total_time_sec)

    top_speed_kmh = round(peak_mps * 3.6, 1)
    avg_speed_kmh = round(avg_speed_mps * 3.6, 1)
    if total_dist > 5.0:
        top_speed_kmh = min(34.5, max(12.0, top_speed_kmh))
        avg_speed_kmh = min(15.0, max(3.0, avg_speed_kmh))
    else:
        top_speed_kmh = 0.0
        avg_speed_kmh = 0.0

    minutes_played = max(1, round(total_time_sec / 60.0))

    return {
        "distance_m": round(total_dist, 1),
        "top_speed_kmh": top_speed_kmh,
        "average_speed_kmh": avg_speed_kmh,
        "minutes_played": minutes_played,
    }


def compute_player_rating(
    stats: dict | None = None,
    *,
    goals: int = 0,
    assists: int = 0,
    shots: int = 0,
    shots_on_target: int = 0,
    passes_completed: int = 0,
    passes_attempted: int = 0,
    tackles: int = 0,
    interceptions: int = 0,
    clearances: int = 0,
    dribbles: int = 0,
    key_passes: int = 0,
    distance_m: float = 0.0,
    xg: float = 0.0,
) -> float:
    """
    Computes an algorithmic performance rating (FotMob / WhoScored style)
    from real match action signals. Baseline is 6.0, clamped between 5.5 and 9.5.
    Accepts either a dictionary of stats or keyword arguments.
    """
    if isinstance(stats, dict):
        goals = stats.get("goals", goals)
        assists = stats.get("assists", assists)
        shots = stats.get("shots", shots)
        shots_on_target = stats.get("shots_on_target", shots_on_target)
        passes_completed = stats.get("passes_completed", passes_completed)
        passes_attempted = stats.get("passes_attempted", passes_attempted)
        tackles = stats.get("tackles", tackles)
        interceptions = stats.get("interceptions", interceptions)
        clearances = stats.get("clearances", clearances)
        dribbles = stats.get("dribbles", stats.get("dribbles_completed", dribbles))
        key_passes = stats.get("key_passes", key_passes)
        distance_m = stats.get("distance_m", stats.get("distance_covered", distance_m))
        xg = stats.get("xg", xg)

    score = 6.0

    # Attacking
    score += goals * 1.0
    score += assists * 0.6
    score += shots_on_target * 0.35
    score += max(0, shots - shots_on_target) * 0.15
    score += xg * 0.4
    score += key_passes * 0.3
    score += dribbles * 0.25

    # Passing
    score += passes_completed * 0.08
    if passes_attempted > 0:
        incompletions = passes_attempted - passes_completed
        score -= incompletions * 0.12
        acc = passes_completed / passes_attempted
        if acc >= 0.85 and passes_attempted >= 2:
            score += 0.2

    # Defending
    score += tackles * 0.3
    score += interceptions * 0.25
    score += clearances * 0.15

    # Physical activity bonus
    score += min(0.5, (distance_m / 100.0) * 0.05)

    return round(max(5.5, min(9.5, score)), 1)