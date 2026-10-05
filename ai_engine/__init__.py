"""
ai_engine
=========

Standalone computer-vision pipeline for the Football Analytics project.

Lives alongside the Django apps/ directory (accounts, dashboard, matches,
etc.) but has no Django dependency itself — it's a plain Python package.
Stage 7 (Celery integration) is the only place that talks to Django models,
and it does so through a thin adapter so the rest of the pipeline stays
testable in isolation (e.g. `python -m ai_engine.main clip.mp4`).

Suggested build order (see project docs):
    1. stage1_detection      — YOLO player/ball/ref/GK detection
    2. stage2_tracking       — BoT-SORT within a continuous shot
    2.5 stage2_5_shot_detection — camera cut detection, shot classification
    3. stage3_team_reid      — team color + Re-ID embeddings
    4. stage4_ball_tracking  — ball-specific detection + interpolation
    5. stage5_pitch_mapping  — homography + cross-cut identity association
    6. stage6_event_detection — possession / passes / shots
    7. stage7_integration    — Celery task wiring into the Django DB

The standalone orchestrator runs Stages 1-4 on a clip and can run Stages 5-6
when given per-shot calibration points. Enable shot detection to reset Stage 2
tracking at cuts; enable Re-ID to attempt cross-shot player matching. The web
application stores Stages 1-4 outputs and runs calibrated Stages 5-6 later,
then renders the annotated video in Stage 7. Accuracy still needs review on
the target footage; the standalone path does not create pitch outputs without
explicit calibration.
"""
