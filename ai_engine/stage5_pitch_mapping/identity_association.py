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
    tracklets: list[Tracklet], config: PitchMappingConfig
) -> list[Tracklet]:
    """
    Regime 1: spatial + Hungarian matching within one continuous shot.
    Filters out transient tracklets (< min_tracklet_duration_sec) before
    they're considered at all — cheap noise removal before the expensive
    gallery matching in match_tracklets_across_cut.

    TODO: implement using existing pixel/pitch-space proximity between
    tracklet end-points and start-points (for tracklets that briefly
    dropped due to occlusion within the same shot, per Stage 2's
    track_buffer). This is standard tracklet-stitching — Stage 2's own
    track_buffer already handles most short gaps, so this may end up being
    a thin pass-through plus the duration filter.
    """
    raise NotImplementedError("match_tracklets_within_shot is stubbed.")


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

    for tracklet in new_shot_tracklets:
        same_team_identities = [
            identity for identity in gallery.identities if identity.team == tracklet.team
        ]

        # Strong override: confident jersey number match.
        if tracklet.jersey_number is not None:
            number_match = next(
                (
                    identity
                    for identity in same_team_identities
                    if identity.jersey_number == tracklet.jersey_number
                ),
                None,
            )
            if number_match:
                assignments[tracklet.track_id] = number_match.master_id
                continue

        # TODO: replace this per-tracklet loop with a proper batched
        # Hungarian assignment (build the full similarity matrix across
        # ALL of new_shot_tracklets vs. same-team gallery identities at
        # once, then linear_sum_assignment on the whole matrix) — matching
        # one at a time greedily can produce suboptimal/conflicting
        # assignments when two tracklets both best-match the same identity.
        if not same_team_identities or tracklet.reid_embedding is None:
            if gallery.can_add(tracklet.team):
                new_identity = gallery.add(tracklet)
                assignments[tracklet.track_id] = new_identity.master_id
            # else: no signal to match on AND gallery full — this tracklet
            # is unassigned; log it for manual review rather than guessing.
            continue

        similarities = [
            cosine_similarity(
                np.array(tracklet.reid_embedding), np.array(identity.reid_embedding)
            )
            for identity in same_team_identities
        ]
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]

        if best_score >= similarity_threshold or not gallery.can_add(tracklet.team):
            best_identity = same_team_identities[best_idx]
            assignments[tracklet.track_id] = best_identity.master_id
            gallery.update_embedding(
                best_identity.master_id, np.array(tracklet.reid_embedding)
            )
        else:
            new_identity = gallery.add(tracklet)
            assignments[tracklet.track_id] = new_identity.master_id

    return assignments
