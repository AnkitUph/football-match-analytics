"""
Stage 2: Multi-Object Tracking within a single continuous camera shot.

IMPORTANT SCOPE NOTE: track_id here is only guaranteed unique WITHIN one
shot (one continuous camera angle). It is expected to reset/collide across
cuts — that's fine and by design. Stage 5's identity association (using
Stage 3's Re-ID embeddings) is what produces the permanent cross-cut
Master ID. Don't try to make this stage smarter than that; it'll fight
Stage 5's job.
"""

import numpy as np

from ai_engine.config import TrackingConfig
from ai_engine.utils.types import Detection, Tracklet


class Tracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self._active_tracklets: dict[int, Tracklet] = {}

    def reset(self):
        """Call this at the start of every new shot segment (Stage 2.5
        boundary), so track IDs don't leak continuity across a cut."""
        self._active_tracklets = {}

    def update(
        self, frame: np.ndarray, detections: list[Detection], shot_id: int
    ) -> list[Tracklet]:
        """
        Feed one frame's detections in, get back the current set of active
        tracklets (updated in place with the new frame's boxes).

        TODO: This should call ultralytics' BoT-SORT tracker directly, e.g.
        via `model.track(frame, tracker="botsort.yaml", persist=True)`
        instead of `model.predict(...)` in Stage 1 — ultralytics bundles
        detection + tracking in one call when you use `.track()`. Consider
        merging Stage 1 + Stage 2 into a single `.track()` loop rather than
        running detection and tracking as fully separate passes; it's less
        code and avoids re-implementing BoT-SORT's matching yourself.

        Kept as separate stages/modules here for clarity of responsibility,
        even if the underlying call ends up being one ultralytics method.
        """
        raise NotImplementedError(
            "Tracker.update is stubbed — wire up ultralytics' BoT-SORT "
            "tracker (see docstring)."
        )

    def finalize_shot(self) -> list[Tracklet]:
        """
        Call at the end of a shot segment. Filters out transient tracklets
        (handled again more thoroughly in Stage 5, but cheap to prune here
        too) and returns the final tracklet list for this shot.
        """
        tracklets = list(self._active_tracklets.values())
        self.reset()
        return tracklets
