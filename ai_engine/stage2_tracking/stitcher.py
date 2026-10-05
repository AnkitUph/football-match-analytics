"""
ai_engine/stage2_tracking/stitcher.py

Post-tracking tracklet stitcher. Merges fragmented tracklets across occlusions,
camera motion, and rapid direction changes using spatial-temporal gating and
Hugging Face DINOv2 Re-ID appearance similarity.
"""

import logging
import numpy as np

from ai_engine.stage3_team_reid.reid import cosine_similarity
from ai_engine.utils.types import Tracklet

logger = logging.getLogger(__name__)


def stitch_tracklets(
    tracklets: dict[int, Tracklet],
    track_embeddings: dict[int, np.ndarray],
    max_gap_frames: int = 75,
    similarity_thresh: float = 0.84,
    fps: float = 25.0,
    max_speed_mps: float = 12.0,
) -> tuple[dict[int, Tracklet], dict[int, int]]:
    """
    Stitches broken tracklets that belong to the same player into a single continuous track.

    Parameters:
    - tracklets: {track_id: Tracklet}
    - track_embeddings: {track_id: normalized 384-d DINOv2 feature vector}
    - max_gap_frames: max frame gap between end of T1 and start of T2 (default 75 = 3.0s)
    - similarity_thresh: min cosine similarity for appearance match (default 0.84)

    Returns:
    - (stitched_tracklets, id_mapping):
        - stitched_tracklets: {new_track_id: Tracklet}
        - id_mapping: {old_track_id: new_track_id}
    """
    if not tracklets or len(tracklets) <= 1:
        return tracklets, {tid: tid for tid in tracklets}

    # Extract metadata for each tracklet
    spans = {}
    for tid, t in tracklets.items():
        if not t.detections:
            continue
        dets = sorted(t.detections, key=lambda d: d.frame_idx)
        start_f = dets[0].frame_idx
        end_f = dets[-1].frame_idx
        cls_val = t.cls.value if t.cls else "player"
        # Bounding box centers at start and end
        start_bbox = dets[0].bbox
        end_bbox = dets[-1].bbox
        start_center = ((start_bbox[0] + start_bbox[2]) / 2, (start_bbox[1] + start_bbox[3]) / 2)
        end_center = ((end_bbox[0] + end_bbox[2]) / 2, (end_bbox[1] + end_bbox[3]) / 2)
        spans[tid] = {
            "start_f": start_f,
            "end_f": end_f,
            "shot_id": t.shot_id,
            "cls": cls_val,
            "start_center": start_center,
            "end_center": end_center,
            "dets": dets,
        }

    # Sort track IDs chronologically by start frame
    sorted_tids = sorted(spans.keys(), key=lambda tid: spans[tid]["start_f"])

    # Disjoint Set / Union-Find for track IDs
    parent = {tid: tid for tid in sorted_tids}
    group_frames = {tid: {d.frame_idx for d in spans[tid]["dets"]} for tid in sorted_tids}

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i == root_j:
            return False
        # HARD ANTI-OVERLAP CONSTRAINT: Never merge tracklets that share any active frame
        if group_frames[root_i] & group_frames[root_j]:
            return False
        # Keep the earlier track as root
        if spans[root_i]["start_f"] <= spans[root_j]["start_f"]:
            parent[root_j] = root_i
            group_frames[root_i].update(group_frames[root_j])
        else:
            parent[root_i] = root_j
            group_frames[root_j].update(group_frames[root_i])
        return True

    merges_done = 0

    # Greedy best-match stitching
    for i, tid1 in enumerate(sorted_tids):
        emb1 = track_embeddings.get(tid1)
        if emb1 is None:
            continue

        meta1 = spans[tid1]
        root1 = find(tid1)
        best_tid2 = None
        best_sim = similarity_thresh

        for j in range(i + 1, len(sorted_tids)):
            tid2 = sorted_tids[j]
            root2 = find(tid2)
            # Cannot merge if already part of same master track
            if root1 == root2:
                continue

            # Hard anti-overlap check: root groups must not share any active frames
            if group_frames[root1] & group_frames[root2]:
                continue

            emb2 = track_embeddings.get(tid2)
            if emb2 is None:
                continue

            meta2 = spans[tid2]

            # This stitcher uses image-space proximity, which is only valid
            # inside one continuous camera view. Cross-cut identity matching
            # belongs to the calibrated Stage 5 matcher.
            if meta1["shot_id"] != meta2["shot_id"]:
                continue

            # 1. Class consistency
            if meta1["cls"] != meta2["cls"]:
                continue

            # 2. Strict temporal ordering & gap constraint (no negative gaps or overlaps)
            gap = meta2["start_f"] - meta1["end_f"]
            if gap <= 0 or gap > max_gap_frames:
                continue

            # 3. Spatial displacement sanity (in pixels)
            dx = meta2["start_center"][0] - meta1["end_center"][0]
            dy = meta2["start_center"][1] - meta1["end_center"][1]
            dist_px = np.hypot(dx, dy)
            max_allowed_dist_px = min(180.0, max(60.0, gap * 8.0))
            if dist_px > max_allowed_dist_px:
                continue

            # 4. Appearance similarity
            sim = cosine_similarity(emb1, emb2)
            if sim > best_sim:
                best_sim = sim
                best_tid2 = tid2

        if best_tid2 is not None:
            if union(tid1, best_tid2):
                merges_done += 1

    logger.info(f"Stitched {merges_done} tracklet fragments into persistent tracks")

    # Group original tracklets by root ID
    groups = {}
    id_mapping = {}
    for tid in sorted_tids:
        root = find(tid)
        id_mapping[tid] = root
        if root not in groups:
            groups[root] = []
        groups[root].append(tid)

    # Reconstruct merged Tracklets
    stitched_tracklets: dict[int, Tracklet] = {}
    for root_tid, member_tids in groups.items():
        base_tracklet = tracklets[root_tid]
        all_dets = []
        for tid in member_tids:
            all_dets.extend(tracklets[tid].detections)
        all_dets.sort(key=lambda d: d.frame_idx)

        # Remove any duplicate frames just in case
        seen_frames = set()
        deduped_dets = []
        for d in all_dets:
            if d.frame_idx not in seen_frames:
                seen_frames.add(d.frame_idx)
                deduped_dets.append(d)

        new_t = Tracklet(
            track_id=root_tid,
            shot_id=base_tracklet.shot_id,
            cls=base_tracklet.cls,
            team=base_tracklet.team,
        )
        new_t.detections = deduped_dets
        new_t.jersey_number = base_tracklet.jersey_number
        new_t.jersey_number_conf = base_tracklet.jersey_number_conf
        stitched_tracklets[root_tid] = new_t

    return stitched_tracklets, id_mapping
