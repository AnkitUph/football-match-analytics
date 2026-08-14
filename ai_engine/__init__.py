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

Get 1 -> 2 -> 3(color only) -> 4 -> 5(single-angle) -> 6 -> 7 working on a
single continuous camera angle FIRST. Add 2.5 + Re-ID (stage 3) + the
multi-signal matching in stage 5 as a second pass.
"""
