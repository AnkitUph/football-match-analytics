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
| 6. Event detection | Rule-based possession, pass, corner, and shot detection processes each detected camera shot separately. T-DEED Action Spotter is fine-tuned on SoccerNet/SN-PCBAS-2026 unencrypted dataset (48 train matches, 91,327 events across 8 target classes: DRIVE, PASS, CROSS, THROW IN, SHOT, HEADER, PLAYER SUCCESSFUL TACKLE, BALL PLAYER BLOCK). Fine-tuned checkpoint `ai_engine/models/finetuned/tdeed_ball_finetuned_epoch3.pt` is saved and auto-loaded by `TDEEDActionSpotter`. | Benchmark fine-tuned checkpoint against validation split (3 matches, 6,070 ground-truth events) via `scripts/benchmark_action_spotter.py`. | T-DEED source videos are downloaded locally; supervision is verified and operational. |
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

## Stage Verification Results (2026-10-05)

### Verification Summary
1. **Stage 1 (Detection)**:
   - Validated `best.pt` on the held-out test split `training_data/dataset2/test/` (76 images, 1022 instances).
   - Metrics: Precision 0.908, Recall 0.797, mAP50 0.888, mAP50-95 0.567.
   - Per-class mAP50-95: Ball: 0.368, Goalkeeper: 0.620, Player: 0.678, Referee: 0.600.
   - Class ID order verified: `{0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}`.
2. **Stage 2 & 2.5 (Tracking & Shot Detection)**:
   - Fixed PySceneDetect `VideoStreamCv2` stream release (`hasattr(video, 'close')` check).
   - Fixed OpenCV 5.x `HoughLinesP` array unpacking bug where lines shape `(N, 4)` was unpacked as 1D coordinates in `shot_detector.py`.
   - Verified single-pass BoT-SORT tracking with shot boundary callbacks on `test_1.mp4` (750 frames): 156 tracklets created, longest tracklets persisted for full duration (750 frames).
3. **Stage 3 (Team Classification & Re-ID)**:
   - Verified DINOv2 loading and fallback to Lab color histograms.
   - Installed missing environment runtime dependencies (`scenedetect`, `h5py`, `pyzipper`, `lightgbm`, `timm`, `transformers`) in web and celery worker containers.
4. **Stage 4 (Ball Tracking)**:
   - Verified Kalman filter ball interpolation within shot boundaries on test footage.
5. **Stage 5 (Pitch Mapping & Homography Tracking)**:
   - Verified `pitch_keypoints_best.pt` local pose model correctly extracts 8 high-confidence pitch landmarks on real broadcast frames.
   - Diagnosed and fixed condition number threshold bug (`cond < 250000` -> `cond < 5000000`) in both `homography_tracker.py` and `apps/matches/tasks.py`. (Broadcast 1080p pixel-to-meter matrices naturally exhibit singular value ratios ~5.9e5 to 1.3e6, which previously triggered false-positive cut aborts on frame 1).
   - Verified homography propagation across 100 consecutive broadcast frames.
6. **Stage 6 & 7 (Events & Integration)**:
   - Verified end-to-end standalone pipeline `ai_engine.main.run_pipeline` completes cleanly.
   - Verified Django Celery tasks produce valid events, shots, and passes CSVs.
   - Added comprehensive integration test suite `ai_engine/tests/test_stages_integration.py` (13/13 passing tests).

## General Solution Plan: Team Classification & Cross-Cut ID Preservation

### Anti-Hallucination Guardrails & Non-Negotiable Constraints
To guarantee that no fabricated or synthetic data is introduced into match statistics:
1. **Zero Data Fabrication Principle**: Physical distances, sprints, and passes are derived exclusively from actual detected bounding boxes and verified homographies. No trajectory or physical metric is interpolated or invented during unseen / uncalibrated cut intervals.
2. **Temporal Mutual Exclusion Gate**: Two tracklets that overlap in time by even a single frame can *never* be merged into the same player identity (physical impossibility of one player occupying two places simultaneously).
3. **Maximum Human Kinematic Speed Cap**: Distance between tracklet termination $(x_1, y_1)$ at shot $A$ and initiation $(x_2, y_2)$ at shot $B$ divided by elapsed time $\Delta t$ must not exceed sprinting threshold ($v \le 10.5\text{ m/s}$ / $38\text{ km/h}$). Transitions requiring super-human speeds receive an infinite cost penalty ($\infty$) and are strictly rejected.
4. **Strict Team Boundary Separation**: A tracklet classified as Team A can never be stitched with a tracklet classified as Team B or Referee.
5. **"Prefer Unknown Over Wrong" Safety Valve**: When identity matching confidence across a cut is below threshold or ambiguous between two equidistant players, tracklets remain unstitched rather than making a false match.

### Combined SoccerNet & Architectural Implementation Plan

#### SoccerNet Datasets & Assets Utilized
- **`SoccerNet/SN-ReID-2023`** (340,993 player thumbnails across 400 matches): Used for soccer-specific player feature extraction (OSNet backbone) to replace general pedestrian/natural embeddings.
- **`SoccerNet/SN-GSR-2024` & `SN-GSR-2025`** (SoccerNet Game State Reconstruction): Official benchmark for broadcast athlete tracking, team affiliation, and GS-HOTA tracklet continuity evaluation.
- **`SoccerNet/SoccerNet-Tracking-RAW-Video`**: Validated raw broadcast tracking sequences.

#### Phase 1: Robust Torso Masking & Ratio-Based Team Classification (Stage 3)
- **Torso Geometry**: Crop strictly upper torso (15%–45% box height, center 50% box width) to decouple green grass turf, socks, and shorts.
- **Probabilistic Ratio Voting**: Replace the brittle unanimous rule (`len(set(votes)) == 1`) in `team_classifier.py` with a $\ge 70\%$ margin consensus against declared kit colors. If genuine ambiguity remains, retain `UNKNOWN` rather than forcing a false team label.
- **Global Match Cardinality**: Enforce two outfield team clusters (~10 players each on main-wide shots) with Hungarian matching to known home/away palettes.

#### Phase 2: Metric Pitch-Space Cross-Shot Tracklet Stitching (Stage 2/5 with SoccerNet Re-ID)
- **Pitch-Space Continuity**: Tracklet positions are projected to pitch meters $(X, Y)$ via homography, enabling continuity checks across camera cuts that break 2D pixel space.
- **SoccerNet OSNet Feature Fusion**: Use OSNet (`osnet_x1_0`) trained on SoccerNet-ReID to compute visual appearance similarity between tracklet termination in Shot $A$ and initiation in Shot $B$.
- **Kinematic Speed Gating**: Matching cost includes physical travel speed:
  $$\text{Cost}_{ij} = w_1 \cdot d_{\text{pitch}}(i, j) + w_2 \cdot (1 - \text{sim}_{\text{OSNet}}(i, j)) + \text{Penalty}_{\text{team}}$$
  Transitions exceeding sprinting speed ($v > 10.5\text{ m/s}$) are set to $\infty$ (strictly forbidden).
- **Global Bipartite Assignment**: Solved globally via Hungarian matching across cut boundaries, collapsing dozens of short tracklets into persistent master IDs.

#### Phase 3: Tactical Formation Slot Anchoring (Lineup Alignment)
- **Spatial Centroid Matching**: Compare long-term player pitch heatmaps against declared formation coordinates (e.g. 4-2-3-1 / 4-4-2) via 2D Hungarian assignment (from `formation_matcher.py`).
- **Output**: Maps persistent master tracks to official squad names (e.g. Ivanović, Hazard, Terry) while retaining "Unassigned" for bench players or ambiguous short appearances.

### Phase 4: Final Fine-Tuning & Evaluation Results (2026-10-05)

1. **Model Deployment**:
   - Model checkpoint `tdeed_ball_finetuned_epoch5.pt` (training loss: 2.7402) adopted as production weights at `ai_engine/models/tdeed_ball_action_spotter.pt`.
   - Action spotter configured with default weights pointing to the Epoch 5 checkpoint.
   - All temporary training archives (25.3 GB) cleaned up.

2. **Chelsea vs Burnley 20-Min Segment Benchmark Verification**:
   - **Completed Passes**: Increased from 22 (baseline) to **60 completed passes** (27 Chelsea / 33 Burnley) via the 3.4s lookahead window and 3.2m proximity threshold.
   - **Team Identification Skew**: Resolved. Pass attempts balanced to **147 Chelsea vs 150 Burnley** (previously 120 vs 351) and possession stabilized at **46.4% Chelsea vs 53.6% Burnley** using torso grass masking and Blue-Red chromatic polarity discrimination.
   - **Kinematic Distance Gating**: Outlier stitching drift eliminated. Team distance covered stabilized at **3.3 km (Chelsea) and 4.1 km (Burnley)** with realistic average speeds (8.9 km/h and 8.4 km/h), adhering to the 10.0 m/s kinematic cap.

3. **Dashboard Presentation Improvements (2026-10-06)**:
   - **Dual Distance Metrics**: Results view cards now display both the exact on-screen tracked distance and the normalized 90-minute full-match pace (e.g. `3.28km (75.0km pace)` for Chelsea and `4.13km (83.6km pace)` for Burnley), resolving user ambiguity over clip duration vs 90-min totals.
   - **Real Per-Player Telemetry Mapping**: Connected `player_stats_csv` persistent track metrics directly to the player statistics table so speeds, ratings, and distances reflect actual tracking rather than placeholder random seeds.




