# AI Engine staged work plan

This plan moves through the `ai_engine` stages in order and uses only footage
and annotations available to this project. SoccerNet PC-BAS-2026 is not a
dependency for the work below.

## Available data and access boundary

- The account has access to SoccerNet_raw_HQ_challenge,
  SoccerNet-Tracking-Raw-Video, SoccerNet_raw_HQ, and SN-BAS-2023.
- The workspace has seven retained clips in `media/test_clips/`: the 10-minute
  match, the full-half Chelsea/Burnley clip, its two 20-minute segments, and
  `test_1.mp4`, `test_11.mp4`, and `test (9).mp4`.
- `training_data/dataset2/test/` contains labelled detection images. Its
  `data.yaml` class order is Ball, Goalkeeper, Player, Referee, matching the
  Stage 1 class IDs after case normalization. These are object-detection
  labels, not tracking identities or action-spotting annotations.
- The workspace includes T-DEED SoccerNet split metadata and class lists, but
  its README says source videos/frames must be downloaded separately. The raw
  SoccerNet tracking annotations and video trees are not currently present in
  the workspace, so tracking evaluation still needs those local assets.
- `SN-BAS-2023` can support a separate ball-action spotting evaluation when
  its matching video assets are available.
- PC-BAS-2026 player-centric labels are not currently available. Do not use
  BAS-2023 as a drop-in source of PC-BAS labels or claim equivalent results.
- Keep project source videos and `media/test_clips/` as local evaluation inputs.
- Several model checkpoints are present locally but ignored by Git, including
  detection, pitch keypoints, T-DEED, and TrackNet weights. Treat them as local
  runtime assets; do not assume they are included in a source checkout.

## Stage order

| Stage | Current state | Work we can do now | Data/access limit |
|---|---|---|---|
| 1. Detection | YOLO weights and an OpenVINO export are present. The loader checks model class names against the configured ball, goalkeeper, player, and referee IDs before inference. | Review model loading and inference settings; the held-out labelled images in `training_data/dataset2/test/` are available for detection evaluation when requested. | These detection images do not measure video tracking quality or provide PC-BAS labels. |
| 2. Tracking | Ultralytics BoT-SORT resets at Stage 2.5 shot boundaries. Track IDs are made unique across shots and tracklets retain their `shot_id`; image-space Re-ID stitching is restricted to a single shot. | Review cut boundaries and tracking continuity on footage, then evaluate against SoccerNet tracking annotations. | SoccerNet tracking annotations can support evaluation but do not provide PC-BAS labels. |
| 2.5 Shot detection | PySceneDetect cut detection and a conservative turf/line classifier are implemented and connected to the production tracking call. Cross-cut Stage 5 stitching now requires valid metric pitch positions. | Review cut boundaries and view labels on broadcast footage; replay identity is still not inferred. | Can use available broadcast footage; no PC-BAS labels needed. |
| 3. Team and Re-ID | Color classification, DINOv2 embedding code, and optional jersey OCR exist. Configured model selection, bounded inference batches, OCR sample settings, missing-color handling, and safe cluster sizing are wired through. | Review team color palettes and embeddings on accessible match footage; keep OCR off unless crop resolution supports it. | Tracking footage supports track-based inspection; player-centric PC-BAS labels are unnecessary. OCR accuracy still needs suitable high-resolution footage. |
| 4. Ball tracking | Production and standalone paths interpolate independently within each detected shot. Interpolation advances across omitted source-frame indices, uses configured measurement noise, and starts a fresh filter after a long or out-of-frame gap. | Review trajectories against available tracking annotations and make sure video frame indexing and pitch mapping use the same source-frame coordinates. | Use tracking data where ball annotations are present; BAS-2023 can support spotting experiments, not substitute for PC-BAS. |
| 5. Pitch mapping and identity association | Production and standalone homography propagation are bounded to calibrated main-wide shots and stop after lost or invalid estimates. Automatic calibration uses detected main-wide shots and only reuses calibrations when source file hashes match. Keypoint detection tries the local `pitch_keypoints_best.pt` checkpoint before Roboflow; the checkpoint is present locally but Git-ignored. Cross-shot matching requires calibrated metric proximity and known team labels. | Review the local checkpoint’s output on retained clips and calibration assumptions; provide a fresh calibration for each main-wide view that needs pitch coordinates. | Accuracy and class-index compatibility of the local checkpoint remain unverified. Shot detection does not create a calibration for a new view. |
| 6. Event detection | Rule-based possession, pass, corner, and shot detection processes each detected camera shot separately. Possession matching skips unknowns and referees, and pass lookahead is bounded to its shot. Local T-DEED checkpoints and SoccerNet split metadata are present. | Review rule-based events on retained project clips; use matching BAS-2023 video assets for spotting evaluation when available. | T-DEED source videos/frames are absent locally; PC-BAS-specific player-action supervision remains unavailable. |
| 7. Integration and visualization | Production uses `apps/matches/tasks.py`: Stage 1–4 tracking is saved, then calibration-triggered Stage 5–6 processing writes pitch and event data, and Stage 7 renders the annotated video. Player and ball CSVs retain shot IDs. Recalibration recognizes already-merged cross-shot player IDs, and rendering avoids box interpolation and ball trails across cuts. Cross-cut matching requires calibrated metric distance. Missing either outfield kit color leaves player teams unknown. Standalone `ai_engine/main.py` orchestrates Stages 1–6 when per-shot calibrations are supplied and reports skipped stages otherwise. `ai_engine/stage7_integration/tasks.py` remains a separate, unused helper. | Compare standalone and Django outputs on the same reviewed clip, then reconcile output IDs and event/statistics serialization. | Pitch/event outputs need valid calibration for each camera view. Accuracy claims still need benchmarks or reviewed footage. |

## Working sequence

1. Stage 1: confirm configuration, checkpoint/export compatibility, and class
   mapping. Then measure detections on authorized clips.
2. Stage 2: validate tracking and frame numbering against SoccerNet tracking
   ground truth.
3. Implement Stage 2.5 before enabling multi-shot processing.
4. Review Stage 3 and Stage 4 with tracking outputs and ball trajectories.
5. Review Stage 5 calibration and identity association.
6. Review Stage 6 with available labels, keeping BAS-2023 experiments separate
   from the PC-BAS-2026 plan.
7. Finish by unifying and documenting Stage 7 processing and outputs.

## Implementation notes

- Static review and source changes do not establish model accuracy. No test,
  benchmark, or inference run has been performed as part of this work.
- Older module comments included validation counts and accuracy claims not
  backed by a reproducible evaluation in the current worktree; unsupported
  claims have been removed from the Stage 1, 4, and 5 descriptions.
- Stage 3's DINOv2 and 384-bin color-histogram outputs are different feature
  spaces. Do not mix embeddings created by different backends in one gallery;
  keep a match's Re-ID backend consistent.
- The existing Stage 5 calibration still requires known image-to-pitch point
  pairs for each camera view. A first-shot homography cannot be assumed valid
  after a broadcast cut.
- Stage 5 production recalculates shot boundaries before propagating saved
  calibrations. A missed cut or incorrect shot classification can still
  affect mapping, so pitch coordinates require manual review.
- The web application's production path is `apps/matches/tasks.py`, not the
  separate `ai_engine/stage7_integration/tasks.py` helper. Keep the two paths
  from being treated as interchangeable until their outputs have been compared
  on the same calibrated clip.
- Shot-boundary resets use Ultralytics' `on_predict_postprocess_end` callback
  before its tracker update, keyed by the current source-frame position. A
  multi-shot inference run must fail if that callback is unavailable rather
  than silently continuing with stale BoT-SORT state.
- Standalone usage accepts `--detect-shots`, `--reid`, team kit colors, and a
  JSON file of per-shot image-to-pitch calibration points. Without calibration
  it returns detection/tracking/team/ball outputs and does not invent pitch
  coordinates or events.

Do not remove or overwrite original videos or `media/test_clips/`. Do not claim
model improvement without a benchmark or manually reviewed project footage.
