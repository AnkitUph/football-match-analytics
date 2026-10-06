"""
Stage 5b: Tracklet Merging & Cross-Cut Identity Association.

This is the module that actually solves "player IDs shouldn't change" —
everything else in the pipeline feeds into this.

Two matching regimes:
  1. WITHIN a continuous main_wide shot: cheap spatial + Hungarian matching
     on tracklet proximity (no cut to worry about).
  2. AT a shot boundary, returning to main_wide: no spatial continuity to
     lean on. Match against the persistent gallery using team color (hard
     filter) + Re-ID embedding cosine similarity (primary signal) +
     jersey number (strong override when OCR caught one).

Build (1) first — get single-angle tracklet merging working before adding
gallery-based cross-cut matching in (2).
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ai_engine.config import PitchMappingConfig
from ai_engine.stage3_team_reid.reid import cosine_similarity
from ai_engine.utils.types import MasterIdentity, Team, Tracklet


class IdentityGallery:
    """
    The persistent store of up to `max_gallery_size` MasterIdentity
    records for one match. Lives for the whole clip, not just one shot.
    """

    def __init__(self, config: PitchMappingConfig):
        self.config = config
        self._identities: dict[int, MasterIdentity] = {}
        self._next_id = 1

    @property
    def identities(self) -> list[MasterIdentity]:
        return list(self._identities.values())

    def count_for_team(self, team: Team) -> int:
        return sum(1 for i in self._identities.values() if i.team == team)

    def can_add(self, team: Team) -> bool:
        if len(self._identities) >= self.config.max_gallery_size:
            return False
        if self.count_for_team(team) >= self.config.max_per_team:
            return False
        return True

    def add(self, tracklet: Tracklet) -> MasterIdentity:
        if not self.can_add(tracklet.team):
            raise ValueError(
                f"Gallery full for team {tracklet.team} "
                f"(cap={self.config.max_per_team}/team, "
                f"{self.config.max_gallery_size} total) — this should have "
                "been caught before calling add(); route to nearest-match "
                "fallback instead."
            )
        identity = MasterIdentity(
            master_id=self._next_id,
            team=tracklet.team,
            reid_embedding=tracklet.reid_embedding or [],
            jersey_number=tracklet.jersey_number,
            cls=tracklet.cls,
        )
        self._identities[identity.master_id] = identity
        self._next_id += 1
        return identity

    def update_embedding(self, master_id: int, new_embedding: np.ndarray, alpha: float = 0.3):
        identity = self._identities[master_id]
        existing = np.array(identity.reid_embedding) if identity.reid_embedding else None
        if existing is None:
            identity.reid_embedding = new_embedding.tolist()
        else:
            blended = alpha * new_embedding + (1 - alpha) * existing
            identity.reid_embedding = blended.tolist()


def build_unified_master_identity_table(
    tracklets: list[Tracklet],
    config: PitchMappingConfig,
    fps: float = 25.0,
    homography_by_frame: dict[int, np.ndarray] | None = None,
    max_gap_frames: int = 75,
    max_stitch_distance_px: float = 100.0,
    max_stitch_distance_m: float = 6.5,
) -> tuple[list[Tracklet], dict[int, int]]:
    """
    Area 3a & 3d: Unified Master Identity Table & Metric Pitch Graph Stitcher.

    Constructs a 100% deterministic map `raw_track_id -> stitched_id` while
    consolidating fragmented tracklets across camera cuts and tracking gaps.

    Hard Constraints:
    - Overlapping tracklets (active in same frame) can NEVER merge.
    - Tracklets belonging to different teams can NEVER merge.
    - Referees (Team.REFEREE) are isolated from player tracklet merging.
    """
    from ai_engine.stage5_pitch_mapping.homography import image_point_to_pitch

    min_duration_frames = int(config.min_tracklet_duration_sec * fps)
    survivors = [t for t in tracklets if t.duration_frames >= min_duration_frames]

    id_mapping: dict[int, int] = {t.track_id: t.track_id for t in tracklets}

    n = len(survivors)
    if n == 0:
        return tracklets, id_mapping

    INF = 1e6
    cost = np.full((n, n), INF)

    for i, a in enumerate(survivors):
        a_frames = {d.frame_idx for d in a.detections}
        a_end_det = a.detections[-1]
        a_end_frame = a_end_det.frame_idx
        a_end_pos = a_end_det.center

        # Pitch-space coordinate if homography is available
        a_pitch_pt = None
        if homography_by_frame and a_end_frame in homography_by_frame:
            fx, fy = a_end_pos[0], a_end_det.y2
            a_pitch_pt = image_point_to_pitch(fx, fy, homography_by_frame[a_end_frame])

        for j, b in enumerate(survivors):
            if i == j or a.team != b.team or a.cls != b.cls:
                continue
            if a.team not in (Team.TEAM_A, Team.TEAM_B):
                # Unknown side labels are not evidence that two fragments
                # belong to the same player; referees are not player IDs.
                continue

            # Hard Constraint: Conflicting jersey numbers on the same team can NEVER merge
            if (
                a.jersey_number is not None
                and b.jersey_number is not None
                and a.jersey_number != b.jersey_number
                and getattr(a, "jersey_number_conf", 0.0) >= 0.6
                and getattr(b, "jersey_number_conf", 0.0) >= 0.6
            ):
                continue

            # A negative shot_id marks an already-stitched master whose
            # detections span multiple camera views. Its endpoint pair is
            # not a single-shot transition, so it cannot seed another merge.
            if a.shot_id < 0 or b.shot_id < 0:
                continue

            b_frames = {d.frame_idx for d in b.detections}
            # Hard constraint 1: Anti-overlap (mutual temporal exclusion)
            if a_frames & b_frames:
                continue

            b_start_det = b.detections[0]
            gap = b_start_det.frame_idx - a_end_frame
            if gap <= 0:
                continue

            # Within-shot vs Cross-cut gap limits:
            # Within shot: up to 125 frames (5.0s).
            # Across cuts (different shot_ids): up to 375 frames (15.0s) if pitch coordinates exist.
            max_allowed_gap = 375 if a.shot_id != b.shot_id else max(125, max_gap_frames)
            if gap > max_allowed_gap:
                continue

            b_start_pos = b_start_det.center
            pixel_dist = float(np.hypot(a_end_pos[0] - b_start_pos[0], a_end_pos[1] - b_start_pos[1]))

            # Check pitch distance if available
            pitch_dist = None
            if homography_by_frame and b_start_det.frame_idx in homography_by_frame and a_pitch_pt is not None:
                bfx, bfy = b_start_pos[0], b_start_det.y2
                b_pitch_pt = image_point_to_pitch(bfx, bfy, homography_by_frame[b_start_det.frame_idx])
                pitch_dist = float(np.hypot(a_pitch_pt.x_m - b_pitch_pt.x_m, a_pitch_pt.y_m - b_pitch_pt.y_m))

            dt_sec = gap / fps
            # Human sprinting speed cap: 10.0 m/s (36 km/h) anti-hallucination limit
            max_feasible_dist_m = min(35.0, 2.5 + 8.5 * dt_sec)

            if pitch_dist is not None:
                speed_mps = pitch_dist / max(dt_sec, 0.04)
                if speed_mps <= 10.0 and pitch_dist <= max_feasible_dist_m:
                    base_cost = pitch_dist * 8.0 + (gap * 0.08)
                    # Jersey number confirmation bonus
                    if (
                        a.jersey_number is not None
                        and b.jersey_number is not None
                        and a.jersey_number == b.jersey_number
                    ):
                        base_cost = max(1.0, base_cost - 40.0)
                    cost[i, j] = base_cost
            elif a.shot_id == b.shot_id and gap < max_gap_frames:
                # Same shot pixel distance fallback
                if pixel_dist < max_stitch_distance_px:
                    cost[i, j] = pixel_dist + (gap * 0.2)

    row_ind, col_ind = linear_sum_assignment(cost)
    merges = [(i, j) for i, j in zip(row_ind, col_ind) if cost[i, j] < INF]

    parent = {t.track_id: t.track_id for t in survivors}
    group_frames = {t.track_id: {d.frame_idx for d in t.detections} for t in survivors}

    def find(x: int) -> int:
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        for node in path:
            parent[node] = x
        return x

    for i, j in merges:
        root_i = find(survivors[i].track_id)
        root_j = find(survivors[j].track_id)
        if root_i == root_j:
            continue
        # Hard transitive anti-overlap check: root groups must never share any active frame!
        if group_frames[root_i] & group_frames[root_j]:
            continue
        parent[root_j] = root_i
        group_frames[root_i].update(group_frames[root_j])

    groups: dict[int, list[Tracklet]] = {}
    for t in survivors:
        root = find(t.track_id)
        id_mapping[t.track_id] = root
        groups.setdefault(root, []).append(t)

    merged_tracklets = []
    for root, members in groups.items():
        members.sort(key=lambda m: m.detections[0].frame_idx)
        team_votes = Counter(m.team for m in members)

        best_jersey = None
        best_conf = 0.0
        for m in members:
            j_num = getattr(m, "jersey_number", None)
            if j_num is not None:
                j_conf = getattr(m, "jersey_number_conf", 0.0) or 0.5
                if j_conf >= best_conf:
                    best_jersey = j_num
                    best_conf = j_conf

        merged = Tracklet(
            track_id=root,
            shot_id=members[0].shot_id,
            team=team_votes.most_common(1)[0][0],
            cls=members[0].cls,
            jersey_number=best_jersey,
            jersey_number_conf=best_conf,
        )
        for m in members:
            merged.detections.extend(m.detections)
        merged_tracklets.append(merged)

    # Ensure any tracklets that were below min_duration_frames are retained as standalone tracklets
    short_tracklets = [t for t in tracklets if t.duration_frames < min_duration_frames]
    merged_tracklets.extend(short_tracklets)

    return merged_tracklets, id_mapping


def match_tracklets_within_shot(
    tracklets: list[Tracklet],
    config: PitchMappingConfig,
    fps: float,
    max_gap_frames: int = 30,
    max_stitch_distance_px: float = 80.0,
) -> list[Tracklet]:
    """
    Backwards-compatible wrapper around build_unified_master_identity_table.
    """
    merged, _ = build_unified_master_identity_table(
        tracklets, config, fps=fps, max_gap_frames=max_gap_frames, max_stitch_distance_px=max_stitch_distance_px
    )
    return merged


def assign_tracklets_to_gallery(
    tracklets: list[Tracklet], gallery: IdentityGallery
) -> tuple[int, int]:
    """
    Feeds within-shot-stitched tracklets into the gallery, respecting the
    <=22 cap. The current heuristic processes LONGEST tracklets first, not
    insertion order — since match_tracklets_within_shot's output usually
    still has more entries than real players (residual fragmentation),
    processing in arbitrary order risks a short leftover fragment
    grabbing a gallery slot before a genuinely distinct player's
    tracklet, incorrectly bumping them out when the cap is hit. Sorting
    by duration first means the cap preferentially rejects short,
    likely-fragment tracklets. The duration ordering is a heuristic; its
    effect on identity accuracy has not been benchmarked in the current
    worktree.

    REAL BUG FOUND during production testing: referees must be excluded
    here. The <=22/<=11-per-team cap is specifically about the two
    playing teams (per the project's own stated constraint) — but
    Team.REFEREE passed through this function unchecked would compete
    for the same shared 22-slot budget, since IdentityGallery.can_add()
    only checks per-team and total caps generically. On real footage
    this caused Team B to lose 2 legitimate player slots to referee
    tracklets (11 assigned to Team A, only 9 to Team B, 2 to Referee —
    should have been 11/11). Referees don't need persistent cross-cut
    Master IDs for player-stats purposes anyway (Stage 1's class label
    already identifies them per-frame), so the simplest correct fix is
    excluding them from this gallery entirely rather than giving them
    their own separate sub-cap.

    Returns (assigned_count, rejected_count). Rejected count includes
    both cap-exceeded and excluded unknown/referee tracklets.
    """
    player_tracklets = [t for t in tracklets if t.team in (Team.TEAM_A, Team.TEAM_B)]
    excluded_tracklet_count = len(tracklets) - len(player_tracklets)

    ordered = sorted(player_tracklets, key=lambda t: -t.duration_frames)
    assigned, rejected = 0, excluded_tracklet_count
    for t in ordered:
        if gallery.can_add(t.team):
            gallery.add(t)
            assigned += 1
        else:
            rejected += 1
    return assigned, rejected


def match_tracklets_across_cut(
    new_shot_tracklets: list[Tracklet],
    gallery: IdentityGallery,
    similarity_threshold: float,
) -> dict[int, int]:
    """
    Regime 2: the core cross-cut identity fix.

    Returns {tracklet.track_id: master_id} for every tracklet in the new
    shot, either matched to an existing gallery identity or (if the
    gallery has room) assigned a brand new master_id.

    Steps:
      1. Hard-filter by team: only compare tracklets against gallery
         identities of the same team.
      2. Build a cosine-similarity matrix between tracklet embeddings and
         same-team gallery embeddings.
      3. If a tracklet has a confident OCR jersey number that matches a
         gallery identity's known number, treat that as a near-certain
         match and skip the embedding comparison for it.
      4. Solve the remaining similarity matrix with Hungarian assignment
         (maximize total similarity == minimize 1 - similarity as cost).
      5. For any tracklet whose best match is below similarity_threshold:
         only create a new identity if gallery.can_add(team) is True;
         otherwise force-assign to its best (even if weak) match — the
         cap must never be exceeded.
    """
    assignments: dict[int, int] = {}
    player_tracklets = [t for t in new_shot_tracklets if t.team != Team.REFEREE]

    for team in (Team.TEAM_A, Team.TEAM_B):
        team_tracklets = [t for t in player_tracklets if t.team == team]
        team_identities = [i for i in gallery.identities if i.team == team]
        if not team_tracklets:
            continue

        unassigned_tracklets = []
        available_identities = list(team_identities)

        # 1. High-confidence Jersey Number matching
        for t in team_tracklets:
            matched_id = None
            if t.jersey_number is not None:
                matched_id = next(
                    (
                        ident for ident in available_identities
                        if ident.jersey_number == t.jersey_number
                        and (ident.cls is None or t.cls is None or ident.cls == t.cls)
                    ),
                    None
                )
            if matched_id is not None:
                assignments[t.track_id] = matched_id.master_id
                available_identities.remove(matched_id)
                if t.reid_embedding:
                    gallery.update_embedding(matched_id.master_id, np.array(t.reid_embedding))
            else:
                unassigned_tracklets.append(t)

        if not unassigned_tracklets:
            continue

        # 2. Batched Hungarian Matching on Re-ID Embeddings
        valid_tracklets = [t for t in unassigned_tracklets if t.reid_embedding is not None]
        no_emb_tracklets = [t for t in unassigned_tracklets if t.reid_embedding is None]

        if valid_tracklets and available_identities:
            cost_matrix = np.zeros((len(valid_tracklets), len(available_identities)), dtype=np.float32)
            for i, t in enumerate(valid_tracklets):
                t_emb = np.array(t.reid_embedding)
                for j, ident in enumerate(available_identities):
                    if t.cls is not None and ident.cls is not None and t.cls != ident.cls:
                        cost_matrix[i, j] = 1e6
                        continue
                    i_emb = np.array(ident.reid_embedding) if ident.reid_embedding else None
                    if i_emb is not None:
                        sim = cosine_similarity(t_emb, i_emb)
                        cost_matrix[i, j] = 1.0 - sim
                    else:
                        cost_matrix[i, j] = 1.0

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_t_indices = set()
            matched_ident_indices = set()

            for r, c in zip(row_ind, col_ind):
                sim = 1.0 - cost_matrix[r, c]
                t = valid_tracklets[r]
                ident = available_identities[c]

                if t.cls is not None and ident.cls is not None and t.cls != ident.cls:
                    continue

                if sim >= similarity_threshold or not gallery.can_add(team):
                    assignments[t.track_id] = ident.master_id
                    gallery.update_embedding(ident.master_id, np.array(t.reid_embedding))
                    matched_t_indices.add(r)
                    matched_ident_indices.add(c)
                elif gallery.can_add(team):
                    new_id = gallery.add(t)
                    assignments[t.track_id] = new_id.master_id
                    matched_t_indices.add(r)

            # Remaining tracklets that weren't assigned
            for r, t in enumerate(valid_tracklets):
                if r not in matched_t_indices:
                    if gallery.can_add(team):
                        new_id = gallery.add(t)
                        assignments[t.track_id] = new_id.master_id
                    elif available_identities:
                        unmatched_c = [
                            c for c, ident in enumerate(available_identities)
                            if c not in matched_ident_indices
                            and (ident.cls is None or t.cls is None or ident.cls == t.cls)
                        ]
                        if unmatched_c:
                            best_c = min(unmatched_c, key=lambda c: cost_matrix[r, c])
                            ident = available_identities[best_c]
                            assignments[t.track_id] = ident.master_id
                            matched_ident_indices.add(best_c)

        # For tracklets with no embedding:
        for t in no_emb_tracklets:
            if gallery.can_add(team):
                new_id = gallery.add(t)
                assignments[t.track_id] = new_id.master_id

    return assignments
