"""
Stage 7: Pipeline Integration & Celery Processing.

This is the ONLY file in ai_engine that should import Django models — every
other stage stays framework-agnostic so it's testable standalone
(`python -m ai_engine.main clip.mp4`) without spinning up Django at all.

Wire this up as a Celery task in whichever app owns match processing
(likely apps/matches or apps/analytics, per your project structure) — e.g.
by importing `process_match_video` from there, or moving this file into
that app once it's ready. Left here for now since it's pipeline-adjacent.
"""

from pathlib import Path

# TODO: uncomment once wired into your Celery app
# from celery import shared_task

from ai_engine.config import DEFAULT_CONFIG
from ai_engine.main import run_pipeline


# @shared_task(bind=True, time_limit=3600, soft_time_limit=3300)
def process_match_video(self, match_id: int, video_path: str):
    """
    Celery task entry point. Set generous time_limit/soft_time_limit —
    per Stage 7's own milestone note, expect low tens of minutes of
    processing per 15-min clip once OCR + Re-ID + homography are all
    running, not near-real-time.

    TODO:
      1. Load the Match model instance (apps.matches.models.Match).
      2. Run the pipeline: result = run_pipeline(video_path, DEFAULT_CONFIG)
      3. Persist results to your DB schema:
         - result.events -> whatever Event/Pass/Shot models you design
         - result.identities -> Player/PlayerMatchStats-type models,
           keyed by MasterIdentity.master_id (NOT Stage 2's per-shot
           track_id, which is not stable/meaningful outside this run)
         - heatmap data -> derived from MasterIdentity.trajectory
      4. Update match.status (e.g. "processing" -> "complete" or "failed")
         so the dashboard (Phase 2) can reflect progress.
      5. Wrap in try/except; on failure, set match.status = "failed" and
         log the exception rather than leaving it stuck in "processing".
    """
    raise NotImplementedError(
        "process_match_video is stubbed — see docstring for the wiring steps."
    )
