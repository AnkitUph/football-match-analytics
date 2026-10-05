# AI Engine checkpoint and handoff

**Checkpoint date:** 2026-10-05  
**Status:** Paused at the user's request until the monthly usage limit resets.

## Website status

The Docker Compose services were running at the checkpoint: `web`, `db`,
`redis`, and `celery_worker`. A live request to `http://127.0.0.1:8000/`
returned a redirect to `/accounts/login/?next=/`; following it returned HTTP
200 and 7,991 bytes of login-page HTML. The site is reachable and serving its
login page. This checks page availability, not authenticated workflows or
background AI processing.

## Work completed in this work period

The AI engine was divided into sequential stages and the current state, data
limits, and proposed sequence were recorded in
[`AI_ENGINE_STAGE_WORKPLAN.md`](AI_ENGINE_STAGE_WORKPLAN.md). Source changes
were made, but no test suite, benchmark, or model inference was run. Treat the
changes as static implementation work pending verification.

- **Stage 1 — detection:** The loader now checks checkpoint class names against
  configured class IDs before inference.
- **Stage 2 — tracking:** BoT-SORT is reset at detected shot boundaries; IDs are
  unique across shots and tracklets retain shot IDs.
- **Stage 2.5 — shot detection:** PySceneDetect was added and wired into the
  production tracking path. Cut labels are conservative, and cross-shot work
  is guarded against uncalibrated pitch positions.
- **Stage 3 — team and Re-ID:** Model selection, batch sizing, OCR sampling,
  missing kit-color handling, and unknown-cluster handling were wired through.
  Unknown gallery entries are excluded.
- **Stage 4 — ball tracking:** Interpolation is shot-local in both production
  and standalone paths. The filter restarts after long or out-of-frame gaps.
- **Stage 5 — pitch mapping and identity:** Homography propagation is limited to
  calibrated main-wide shots and stops on invalid estimates. Calibration
  reuse requires a full source-file SHA256 match. Local pitch keypoint weights
  are tried before Roboflow. Cross-shot association requires calibrated metric
  proximity and a known team label. The calibration UI can use the local
  model without an API key.
- **Stage 6 — events:** Possession, passes, corners, and shots run within each
  shot; pass lookahead cannot cross a shot boundary. Unknown players are not
  assigned to Team A by default.
- **Stage 7 — integration and rendering:** CSV data carries shot IDs;
  recalibration handles merged IDs; rendered box interpolation and ball
  trails do not bridge cuts. The production Django path and standalone path
  remain separate and still need comparison.
- Unsupported validation and accuracy claims in reviewed code comments were
  removed or qualified. The separate `ai_engine/stage7_integration/tasks.py`
  helper is not the production Django path.

## Data and model availability

The user reports Hugging Face access to `SoccerNet_raw_HQ_challenge`,
`SoccerNet-Tracking-Raw-Video`, `SoccerNet_raw_HQ`, and `SN-BAS-2023`. Access
to PC-BAS-2026 has not been granted. Do not treat BAS-2023 as a replacement
for PC-BAS-2026 labels.

The workplan records seven retained clips under `media/test_clips/`, plus
detection-labeled images in `training_data/dataset2/test/`. That dataset's
class order is Ball, Goalkeeper, Player, Referee; it contains detection labels,
not tracking identities or action-spotting annotations. SoccerNet raw tracking
video/annotation trees and T-DEED source videos/frames are not currently in
the workspace. T-DEED split metadata is present.

Local, Git-ignored model artifacts found during the audit include `best.pt`,
an OpenVINO `best.xml`/`.bin` export, `pitch_keypoints_best.pt`,
`action_spotter_soccernet.pt`, `tdeed_ball_action_spotter.pt` and its JSON
metadata, `tracknetv3_ball_best.pt`, `jersey_number_temporal_best.pth`, and
`xg_lightgbm_model.pkl`. Checkpoint compatibility and output quality have not
been established by this work period. Ignored model artifacts are not included
in a normal Git checkout.

## Worktree and preservation

Changes are **uncommitted**. Modified paths at checkpoint:

```text
ai_engine/__init__.py
ai_engine/config.py
ai_engine/main.py
ai_engine/stage1_detection/detector.py
ai_engine/stage1_detection/model_loader.py
ai_engine/stage2_5_shot_detection/shot_detector.py
ai_engine/stage2_tracking/stitcher.py
ai_engine/stage2_tracking/tracker.py
ai_engine/stage3_team_reid/jersey_ocr.py
ai_engine/stage3_team_reid/reid.py
ai_engine/stage3_team_reid/team_classifier.py
ai_engine/stage4_ball_tracking/ball_tracker.py
ai_engine/stage5_pitch_mapping/homography.py
ai_engine/stage5_pitch_mapping/homography_tracker.py
ai_engine/stage5_pitch_mapping/identity_association.py
ai_engine/stage5_pitch_mapping/smart_assist.py
ai_engine/stage6_event_detection/events.py
ai_engine/stage7_integration/tasks.py
ai_engine/stage7_visualization/annotated_video.py
ai_engine/utils/types.py
apps/matches/tasks.py
apps/matches/views.py
requirements.txt
templates/matches/calibrate.html
```

`docs/AI_ENGINE_STAGE_WORKPLAN.md` and this handoff are new documentation.
`inference_output/` appeared as untracked in Git status; it was not inspected,
modified, or deleted. No videos or images were deleted or overwritten during
this work period. Preserve original videos and all of `media/test_clips/`.

## Resume plan

When the monthly limit resets, continue from this checkpoint rather than
restarting the audit:

1. Read this handoff and `AI_ENGINE_STAGE_WORKPLAN.md`; inspect `git status`
   and the diffs so any new work is based on the current worktree.
2. Start at Stage 1. Run the relevant checks and a small, authorized evaluation
   against the held-out detection images. Confirm the local model's class map
   and checkpoint/export compatibility before using video inference.
3. Proceed in order through tracking and shot boundaries, then team/Re-ID and
   ball tracking. Validate frame indices and shot boundaries on retained
   footage. Use SoccerNet tracking ground truth only after the needed files are
   locally available and the evaluation split is understood.
4. Review the local pitch keypoint checkpoint, its class/index mapping, and
   calibration on an actual clip. Require per-view calibration before trusting
   metric coordinates or cross-shot identity association.
5. Review events with suitable local video/labels. Keep SN-BAS-2023 experiments
   separate from PC-BAS-specific tasks; resume those tasks only if access to
   PC-BAS-2026 is granted.
6. Compare the production Django processing path with standalone output on the
   same clip and calibration, then inspect Stage 7 CSV/video output. Only make
   accuracy claims after a reproducible benchmark or reviewed footage.
7. Record results and remaining gaps in the workplan. Preserve all source
   footage and test clips. Commit only if requested by the user.

## Outstanding verification and risks

- None of the AI source changes were exercised. Syntax, imports, runtime API
  compatibility, performance, and output correctness remain unverified.
- The shot-boundary tracker callback depends on the installed Ultralytics API;
  the multi-shot path should fail clearly if the callback is unavailable.
- The local pitch checkpoint exists but its accuracy, class/index mapping, and
  compatibility with the current calibration code are unknown.
- Cut detection can miss or misclassify views. Review boundaries and labels;
  a calibration for one camera view must not be assumed valid for another.
- The Django and standalone pipelines have not been compared end to end.
- `git status` emitted a harmless Git fsmonitor IPC warning while listing the
  worktree. It did still return the modified and untracked paths listed above.

