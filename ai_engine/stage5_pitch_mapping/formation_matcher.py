"""
Stage 5b: 2D Tactical Formation Graph Matcher
Matches tracked player trajectories to squad lineup profiles using 2D pitch topology,
tactical formation anchors (e.g. 4-3-3, 3-4-3), and minimum-cost bipartite Hungarian optimization.
"""

from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment


# Standard 105m x 68m FIFA pitch formation slot anchors (oriented defending left -> attacking right)
FORMATION_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "4-3-3": [
        {"role": "GK", "pos_name": "GK", "x": -48.0, "y": 0.0},
        {"role": "DEF", "pos_name": "LB", "x": -28.0, "y": 22.0},
        {"role": "DEF", "pos_name": "LCB", "x": -33.0, "y": 8.0},
        {"role": "DEF", "pos_name": "RCB", "x": -33.0, "y": -8.0},
        {"role": "DEF", "pos_name": "RB", "x": -28.0, "y": -22.0},
        {"role": "MID", "pos_name": "DM", "x": -14.0, "y": 0.0},
        {"role": "MID", "pos_name": "LCM", "x": -2.0, "y": 14.0},
        {"role": "MID", "pos_name": "RCM", "x": -2.0, "y": -14.0},
        {"role": "FWD", "pos_name": "LW", "x": 24.0, "y": 22.0},
        {"role": "FWD", "pos_name": "ST", "x": 32.0, "y": 0.0},
        {"role": "FWD", "pos_name": "RW", "x": 24.0, "y": -22.0},
    ],
    "3-4-3": [
        {"role": "GK", "pos_name": "GK", "x": -48.0, "y": 0.0},
        {"role": "DEF", "pos_name": "LCB", "x": -33.0, "y": 16.0},
        {"role": "DEF", "pos_name": "CB", "x": -35.0, "y": 0.0},
        {"role": "DEF", "pos_name": "RCB", "x": -33.0, "y": -16.0},
        {"role": "MID", "pos_name": "LWB", "x": -8.0, "y": 26.0},
        {"role": "MID", "pos_name": "LCM", "x": -5.0, "y": 8.0},
        {"role": "MID", "pos_name": "RCM", "x": -5.0, "y": -8.0},
        {"role": "MID", "pos_name": "RWB", "x": -8.0, "y": -26.0},
        {"role": "FWD", "pos_name": "LW", "x": 24.0, "y": 20.0},
        {"role": "FWD", "pos_name": "ST", "x": 32.0, "y": 0.0},
        {"role": "FWD", "pos_name": "RW", "x": 24.0, "y": -20.0},
    ],
    "4-4-2": [
        {"role": "GK", "pos_name": "GK", "x": -48.0, "y": 0.0},
        {"role": "DEF", "pos_name": "LB", "x": -28.0, "y": 22.0},
        {"role": "DEF", "pos_name": "LCB", "x": -33.0, "y": 8.0},
        {"role": "DEF", "pos_name": "RCB", "x": -33.0, "y": -8.0},
        {"role": "DEF", "pos_name": "RB", "x": -28.0, "y": -22.0},
        {"role": "MID", "pos_name": "LM", "x": 2.0, "y": 24.0},
        {"role": "MID", "pos_name": "LCM", "x": 0.0, "y": 8.0},
        {"role": "MID", "pos_name": "RCM", "x": 0.0, "y": -8.0},
        {"role": "MID", "pos_name": "RM", "x": 2.0, "y": -24.0},
        {"role": "FWD", "pos_name": "LST", "x": 30.0, "y": 8.0},
        {"role": "FWD", "pos_name": "RST", "x": 30.0, "y": -8.0},
    ],
    "4-2-3-1": [
        {"role": "GK", "pos_name": "GK", "x": -48.0, "y": 0.0},
        {"role": "DEF", "pos_name": "LB", "x": -28.0, "y": 22.0},
        {"role": "DEF", "pos_name": "LCB", "x": -33.0, "y": 8.0},
        {"role": "DEF", "pos_name": "RCB", "x": -33.0, "y": -8.0},
        {"role": "DEF", "pos_name": "RB", "x": -28.0, "y": -22.0},
        {"role": "MID", "pos_name": "LDM", "x": -12.0, "y": 10.0},
        {"role": "MID", "pos_name": "RDM", "x": -12.0, "y": -10.0},
        {"role": "MID", "pos_name": "LAM", "x": 16.0, "y": 20.0},
        {"role": "MID", "pos_name": "CAM", "x": 14.0, "y": 0.0},
        {"role": "MID", "pos_name": "RAM", "x": 16.0, "y": -20.0},
        {"role": "FWD", "pos_name": "ST", "x": 32.0, "y": 0.0},
    ],
    "3-5-2": [
        {"role": "GK", "pos_name": "GK", "x": -48.0, "y": 0.0},
        {"role": "DEF", "pos_name": "LCB", "x": -33.0, "y": 16.0},
        {"role": "DEF", "pos_name": "CB", "x": -35.0, "y": 0.0},
        {"role": "DEF", "pos_name": "RCB", "x": -33.0, "y": -16.0},
        {"role": "MID", "pos_name": "LWB", "x": -5.0, "y": 26.0},
        {"role": "MID", "pos_name": "LDM", "x": -10.0, "y": 8.0},
        {"role": "MID", "pos_name": "RDM", "x": -10.0, "y": -8.0},
        {"role": "MID", "pos_name": "CAM", "x": 12.0, "y": 0.0},
        {"role": "MID", "pos_name": "RWB", "x": -5.0, "y": -26.0},
        {"role": "FWD", "pos_name": "LST", "x": 30.0, "y": 8.0},
        {"role": "FWD", "pos_name": "RST", "x": 30.0, "y": -8.0},
    ],
}


def get_formation_slots(formation_name: str, defending_left: bool = True) -> list[dict[str, Any]]:
    """
    Returns reference 2D pitch coordinates for the requested formation.
    If defending_left is False, mirrors the coordinates horizontally.
    """
    clean_fmt = formation_name.strip() if formation_name else "4-3-3"
    slots = FORMATION_TEMPLATES.get(clean_fmt, FORMATION_TEMPLATES["4-3-3"])
    
    if defending_left:
        return slots
    
    # Mirror coordinates horizontally if defending right goal
    mirrored = []
    for s in slots:
        mirrored.append({
            "role": s["role"],
            "pos_name": s["pos_name"],
            "x": -s["x"],
            "y": -s["y"],
        })
    return mirrored


def map_lineup_to_formation_slots(lineup_entries: list[Any], formation_slots: list[dict[str, Any]]) -> list[tuple[Any, dict[str, Any]]]:
    """
    Pairs real squad lineup players to formation slots based on position role hierarchy.
    """
    starters = [l for l in lineup_entries if getattr(l, "is_starting", True)]
    subs = [l for l in lineup_entries if not getattr(l, "is_starting", True)]
    
    def role_of(l):
        p = (getattr(l, "position", "") or "").upper()
        if "GK" in p:
            return "GK"
        if "DEF" in p or "DF" in p or "BACK" in p:
            return "DEF"
        if "MID" in p or "MF" in p:
            return "MID"
        return "FWD"

    # Match starters to formation slots
    paired = []
    unmatched_slots = list(formation_slots)
    unmatched_lineups = list(starters)

    # 1. Match Goalkeeper
    gk_player = next((l for l in unmatched_lineups if role_of(l) == "GK"), None)
    gk_slot = next((s for s in unmatched_slots if s["role"] == "GK"), None)
    if gk_player and gk_slot:
        paired.append((gk_player, gk_slot))
        unmatched_lineups.remove(gk_player)
        unmatched_slots.remove(gk_slot)

    # 2. Match outfield starters by position role
    for role in ("DEF", "MID", "FWD"):
        role_players = [l for l in unmatched_lineups if role_of(l) == role]
        role_slots = [s for s in unmatched_slots if s["role"] == role]
        for p, s in zip(role_players, role_slots):
            paired.append((p, s))
            unmatched_lineups.remove(p)
            unmatched_slots.remove(s)

    # 3. Fill remaining slots with any available starters or subs
    remaining_players = unmatched_lineups + subs
    for p, s in zip(remaining_players, unmatched_slots):
        paired.append((p, s))

    return paired


def match_tracks_to_lineup_hungarian(
    track_summaries: list[dict[str, Any]],
    lineup_entries: list[Any],
    formation_name: str = "4-3-3",
    defending_left: bool = True,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    """
    Solves global 2D Hungarian maximum bipartite matching between tracked player identities
    and squad lineup players using pitch-space topological Voronoi slots.
    
    Returns:
    - assignments: list of dicts with track_id, lineup_entry_id, confidence, is_uncertain, role
    - top_candidates: dict mapping track_id -> top 3 candidate lineup players with scores
    """
    if not track_summaries or not lineup_entries:
        return [], {}

    slots = get_formation_slots(formation_name, defending_left=defending_left)
    player_slots = map_lineup_to_formation_slots(lineup_entries, slots)

    n_tracks = len(track_summaries)
    n_players = len(player_slots)
    cost_matrix = np.full((n_tracks, n_players), 999.0, dtype=np.float32)

    for i, t in enumerate(track_summaries):
        tx = float(t.get("median_x", 0.0))
        ty = float(t.get("median_y", 0.0))
        is_gk_track = bool(t.get("is_gk", False))
        t_jersey = t.get("jersey_number")

        for j, (player, slot) in enumerate(player_slots):
            sx = slot["x"]
            sy = slot["y"]
            s_role = slot["role"]

            # 1. 2D Euclidean spatial distance in pitch meters
            dist_2d = float(np.sqrt((tx - sx) ** 2 + (ty - sy) ** 2))

            # 2. Role compatibility constraint
            role_cost = 0.0
            if s_role == "GK":
                if not is_gk_track:
                    role_cost += 150.0
            else:
                if is_gk_track:
                    role_cost += 150.0

            # 3. Soft Jersey Evidence
            jersey_cost = 0.0
            if t_jersey is not None and getattr(player, "jersey_number", None) is not None:
                if t_jersey == player.jersey_number:
                    jersey_cost = -45.0  # Strong positive evidence bonus
                else:
                    jersey_cost = 15.0   # Soft mismatch penalty

            cost_matrix[i, j] = dist_2d + role_cost + jersey_cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = []
    top_candidates = {}

    for i, j in zip(row_ind, col_ind):
        track = track_summaries[i]
        player, slot = player_slots[j]
        cost = cost_matrix[i, j]

        # Candidate ranking for this track
        sorted_player_indices = np.argsort(cost_matrix[i])
        best_cost = cost_matrix[i, sorted_player_indices[0]]
        second_cost = cost_matrix[i, sorted_player_indices[1]] if len(sorted_player_indices) > 1 else best_cost + 50.0

        # Confidence metric: normalized inverse cost + margin
        confidence = float(np.clip(1.0 - (best_cost / 100.0), 0.20, 0.98))
        is_uncertain = (second_cost - best_cost) < 6.0  # Close pairing (e.g. 2 center backs)

        assignments.append({
            "track_id": track["track_id"],
            "lineup_entry_id": player.id,
            "player_name": player.player_name,
            "jersey_number": player.jersey_number,
            "position": player.position,
            "slot_role": slot["role"],
            "confidence": round(confidence, 2),
            "is_uncertain": is_uncertain,
        })

        top_candidates[track["track_id"]] = [
            {
                "lineup_entry_id": player_slots[idx][0].id,
                "player_name": player_slots[idx][0].player_name,
                "jersey_number": player_slots[idx][0].jersey_number,
                "score": round(float(cost_matrix[i, idx]), 1),
            }
            for idx in sorted_player_indices[:3]
        ]

    return assignments, top_candidates
