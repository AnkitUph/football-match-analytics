# Tracking & visualization status

Stakeholder / issue-tracker brief for the annotated-replay pipeline. Tunables below match `ai_engine/config.py` and `ai_engine/stage7_visualization/annotated_video.py` as of this write-up.

## Project overview

**Goal:** Automatically generate an annotated replay video for a football match.

**Pipeline (relevant stages):**

| Stage | Role |
| --- | --- |
| **2 – Tracking / stitching** | Detect players and the ball, assign per-frame `track_id`s, stitch short tracklets into longer trajectories. |
| **5 – Pitch mapping / identity association** | Project detections onto the calibrated pitch, classify each track as home, away, referee, or unknown, and resolve duplicate identities (e.g. overlapping teammates). |
| **6 – Event detection** | Infer passes, shots, and related events from ball + player trajectories (CSV / analytics, not the overlay itself). |
| **7 – Visualization** | Render the original video with bounding boxes, **team-badge labels only** (no player names or jersey numbers), ball trajectory, and a HUD. |

## Current problems

| Area | Symptom | Likely cause(s) | Impact |
| --- | --- | --- | --- |
| Bounding-box coverage | Many on-pitch players have no box (especially on the wings). | • Off-pitch filter in visualization: `med_x > 58 m` **or** `med_y > 40 m` (half-length ≈ 52.5 m, half-width ≈ 34 m — the y threshold is especially tight for wide players). • Player-box **gap fill only interpolates gaps of 2–15 frames** (~0.56 s at 25 fps); longer occlusions leave holes. • Tracklets shorter than `min_tracklet_duration_sec = 0.08 s` are dropped in pitch mapping. | Incomplete visualization; analysts cannot see full player movement. |
| Team classification | Some players appear with the wrong team colour or as `UNKNOWN`. | • Union-Find stitching can still merge similar kits across identities if anti-overlap does not fire. • Referee disambiguation can mis-label a player as referee, then fall back to a default team. • Stage 3 kit clustering remains turf-contaminated (see `docs/CURRENT_STACK_ANALYSIS_AND_FAILURES.md`). | Misleading team-badge labels; confusing downstream analytics. |
| Pass detection | Pass events missed or duplicated in generated CSV. | • **Player** temporal smoothing in the renderer still maxes at 14 missing frames; that does not fix Stage 6 identity gaps. • Tight `min_tracklet_duration_sec` and `stitch_similarity_thresh` can drop valid tracks before events run. • Ball-tracking CSV can have out-of-sync timestamps. **Note:** ball interpolation in config is already `interpolation_max_gap_frames = 30` (~1.2 s) — longer than player-box fill in Stage 7. | Inaccurate pass statistics and heat-maps. |
| Goal detection | Goal frames not highlighted correctly; ball sometimes disappears near the net. | • Ball smoothing can still drop the last frames if occlusion exceeds the interpolator. • `ball_color` / interp vs non-interp can hide the ball when the interpolation flag is wrong. • Stage 6 has **shot-on-goal** logic (goal centres at ±52.5 m) but the overlay has **no explicit “goal zone” highlight** — review is still visual. | Missed or mis-recorded goal events in dashboards. |
| General robustness | Occasional FFmpeg seek warnings; permission errors writing temp files. | • Output under `scratch/` (or similar) without write permission. • Segment rendering (`start_sec` / `duration_sec` on `render_annotated_match_video`) needs careful seek + loop-break. Studio subclips in `apps/matches/views.py` already write under a temp directory. | Pipeline stops mid-run; manual restart. |

## What we have done so far

- Anti-overlap checks (`group_frames` set intersection) in Union-Find in `ai_engine/stage2_tracking/stitcher.py` and `ai_engine/stage5_pitch_mapping/identity_association.py`.
- Stitching tunables: `stitch_max_gap_frames = 50`, `stitch_similarity_thresh = 0.85`.
- `min_tracklet_duration_sec = 0.08` so very short detections are kept (≥ ~2 frames at 25 fps).
- Annotated video shows **team badges only** (no names / jersey numbers).
- Ball tracker max interpolation gap raised to **30 frames** in `BallTrackingConfig` (player overlay gap fill is still 15).

## Suggested next steps

1. **Relax off-pitch visualization thresholds** (e.g. `med_x > 62 m`, `med_y > 45 m`) in `annotated_video.py` so wing players are not treated as dugout/coaches.
2. **Extend player-box gap fill to ~30 frames** (≈ 1 s at 25 fps), aligned with ball interpolation.
3. **Post-process sanity check:** flag any `track_id` missing for more than ~5 seconds of on-pitch time.
4. **Goal-zone overlay / event:** treat ball centroid entering the last ~5 m of pitch length (and/or goalmouth y-band) as a candidate goal highlight, in addition to existing Stage 6 shot logic.
5. **FFmpeg temp paths:** always write intermediate segment files to a user-writable directory (`/tmp` or an explicit writable `scratch/`).

## Related docs

- Architecture failure analysis (team ID, OCR, stitching desync): `docs/CURRENT_STACK_ANALYSIS_AND_FAILURES.md`
- Longer-term SOTA roadmap: `docs/ALTERNATIVE_APPROACHES_AND_SOTA_ROADMAP.md`
