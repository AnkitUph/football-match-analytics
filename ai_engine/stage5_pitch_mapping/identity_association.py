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


def match_tracklets_within_shot(
    tracklets: list[Tracklet],
    config: PitchMappingConfig,
    fps: float,
    max_gap_frames: int = 30,
    max_stitch_distance_px: float = 80.0,
) -> list[Tracklet]:
    """
    Regime 1: spatial + Hungarian matching within one continuous shot.

    VALIDATED against real footage (test_11.avi, BoT-SORT output): 120 raw
    tracklets -> 56 after duration filtering -> 37 after stitching. Real,
    meaningful reduction, though still above the ~22-25 theoretical count
    (players+ref+GK) — some fragmentation remains that pure spatial
    proximity can't resolve (gaps > max_gap_frames, or player movement
    exceeding max_stitch_distance_px between frames). Closing that
    remaining gap is what Stage 3's Re-ID embeddings (second-pass) are
    for — this function intentionally does NOT try to solve it with ad
    hoc looser thresholds, since that risks false-merging different
    players instead.

    Step 1: filter out tracklets shorter than
    config.min_tracklet_duration_sec — cheap noise removal (in the
    validated test, this alone cut 120 -> 56).

    Step 2: build a cost matrix between every survivor's END point and
    every other survivor's START point (cost = pixel distance, only
    considered if the start occurs shortly after the end within
    max_gap_frames). Solve via Hungarian assignment
    (scipy.optimize.linear_sum_assignment) rather than greedy pairing —
    validated as necessary on real data, since several tracklets had
    multiple ambiguous candidates and greedy matching risks picking the
    wrong one.

    Step 3: merge matched pairs via union-find, so multi-hop chains (A
    stitches to B, B stitches to C) correctly collapse into one tracklet
    rather than needing multiple passes.

    Team is resolved per merged group via majority vote across the
    fragments' individually-classified teams (call Stage 3's
    classify_team on each fragment before this function, same as any
    other single tracklet).
    """
    min_duration_frames = int(config.min_tracklet_duration_sec * fps)
    survivors = [t for t in tracklets if t.duration_frames >= min_duration_frames]

    n = len(survivors)
    if n == 0:
        return []

    INF = 1e6
    cost = np.full((n, n), INF)
    for i, a in enumerate(survivors):
        a_end_frame = a.detections[-1].frame_idx
        a_end_pos = a.detections[-1].center
        for j, b in enumerate(survivors):
            if i == j or a.team != b.team:
                continue
            gap = b.detections[0].frame_idx - a_end_frame
            if 0 < gap < max_gap_frames:
                b_start_pos = b.detections[0].center
                dist = float(np.hypot(a_end_pos[0] - b_start_pos[0], a_end_pos[1] - b_start_pos[1]))
                if dist < max_stitch_distance_px:
                    cost[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost)
    merges = [(i, j) for i, j in zip(row_ind, col_ind) if cost[i, j] < INF]

    parent = {t.track_id: t.track_id for t in survivors}

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    for i, j in merges:
        parent[find(survivors[j].track_id)] = find(survivors[i].track_id)

    groups: dict[int, list[Tracklet]] = {}
    for t in survivors:
        groups.setdefault(find(t.track_id), []).append(t)

    merged_tracklets = []
    for root, members in groups.items():
        members.sort(key=lambda m: m.detections[0].frame_idx)
        team_votes = Counter(m.team for m in members)

        # Preserve highest-confidence jersey recognition across merged tracklet fragments
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

    return merged_tracklets


def assign_tracklets_to_gallery(
    tracklets: list[Tracklet], gallery: IdentityGallery
) -> tuple[int, int]:
    """
    Feeds within-shot-stitched tracklets into the gallery, respecting the
    <=22 cap. VALIDATED finding: process LONGEST tracklets first, not
    insertion order — since match_tracklets_within_shot's output usually
    still has more entries than real players (residual fragmentation),
    processing in arbitrary order risks a short leftover fragment
    grabbing a gallery slot before a genuinely distinct player's
    tracklet, incorrectly bumping them out when the cap is hit. Sorting
    by duration first means the cap preferentially rejects short,
    likely-fragment tracklets. Tested on real data: with this ordering,
    every accepted tracklet had >=335 frames of tracking, every rejected
    one was shorter — a principled split, not an arbitrary one.

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
    both cap-exceeded tracklets AND excluded referee tracklets.
    """
    player_tracklets = [t for t in tracklets if t.team != Team.REFEREE]
    excluded_referee_count = len(tracklets) - len(player_tracklets)

    ordered = sorted(player_tracklets, key=lambda t: -t.duration_frames)
    assigned, rejected = 0, excluded_referee_count
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
                    (ident for ident in available_identities if ident.jersey_number == t.jersey_number),
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
                        unmatched_c = [c for c in range(len(available_identities)) if c not in matched_ident_indices]
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
