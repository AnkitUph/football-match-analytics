# Implementation Procedure: Eligibility Check + Step-by-Step Work Plan

This document covers the five areas from the Hugging Face / SoccerNet suggestion:

1. Team classification + ReID
2. Jersey number OCR
3. Tracking, stitching and identity assignment
4. Event detection
5. Camera calibration

For each area: an **eligibility gate** (do we go ahead?) and a **working procedure**.
It builds on `SOLUTION_PLAN_TEAM_ID_OCR.md`.

---

## Part A. Jersey OCR eligibility check

### What I could check, and what I could not

- I do **not** have your match videos or tracking output here, so I could not measure your footage directly.
- Your failure analysis reports player boxes of **80–130 px** high in wide broadcast shots, with jersey digits of **12–22 px**.
- I wrote `check_ocr_eligibility.py` so you can measure the real numbers in one minute.

### Result from the numbers you reported

**Verdict on wide-shot footage: NOT ELIGIBLE.**

| Item | Value |
|---|---|
| Your box height | 80–130 px |
| Digit height (about 16% of box) | 12–22 px |
| Height where I would start trying OCR | about 150 px (digit about 24 px) |
| Result | Every box in the reported range is below the gate |

Why the gate is not lower:
- Published work on the SoccerNet jersey data reports about **87% accuracy at tracklet level** for the best cases. That is on tracklets of many frames, and it includes a "no number visible" class.
- That same work found that only about **1 frame in 8** shows a usable number. After keyframe selection, roughly 88% of frames were dropped.
- So even with good footage, OCR gives you a number for a share of tracks, not for all of them.

### What to run (one minute)

```bash
# from the tracker's pickle stub
python check_ocr_eligibility.py --pickle stubs/track_stubs.pkl

# or from the tracking CSV
python check_ocr_eligibility.py --csv path/to/player_tracking.csv

# compare with the SoccerNet Jersey crops (after you download them)
python check_ocr_eligibility.py --images-dir path/to/sn-jersey/train
```

The script prints size percentiles, how many tracks pass the size gate, and one of three verdicts:

| Verdict | Rule (my judgement, tune it) | What to do |
|---|---|---|
| ELIGIBLE | at least 40% of tracks have 50+ frames with box height ≥ 150 px | Do the full OCR procedure (Area 2) |
| PARTIAL | 15–40% of tracks pass | OCR as a bonus for those tracks only |
| NOT ELIGIBLE | under 15% pass | Skip OCR. Rely on color + position + human review |

Then do one manual check: open 30 tall crops of players from the back and try to read the numbers yourself. **If you cannot read them, the model will not either.**

### Decision for this project

- Run the script on at least 3 matches.
- If the verdict is NOT ELIGIBLE, mark Area 2 as "documented limitation" and put your effort into Areas 1, 3 and the review page. This is a valid, defensible engineering decision. Explain it in your report with the measured numbers.

---

## Part B. Order of work

| Order | Area | Gate | Effort |
|---|---|---|---|
| 0 | Ground truth + measurement | none | 1 day |
| 1 | Area 1: Team classification | always go | 2 days |
| 2 | Area 3a: Single ID table | always go | 0.5 day |
| 3 | Area 5: Calibration stability check | go if unstable | 1 day check, more if needed |
| 4 | Area 3b: Tracking + stitching + lineup assignment | always go | 1–2 weeks |
| 5 | Area 2: Jersey OCR | eligibility script | 1–2 weeks, only if ELIGIBLE/PARTIAL |
| 6 | Area 4: Events | after 1–4 hit targets | later |

---

## Part C. Step 0: Ground truth (needed for every gate below)

1. Pick 2–3 matches with different kits and stadiums.
2. Hand-label about 200 stitched tracks. Columns: `match_id, stitched_id, true_team, true_role, true_player`.
   - `true_team` is one of Home, Away, Referee, Other.
   - `true_player` can stay blank when you can't tell. That is normal.
3. Write `evaluate.py` that prints separately:
   - team accuracy and referee accuracy
   - stitched identities per team (target: close to 11 plus substitutes)
   - role accuracy and exact-player accuracy
   - share of tracks marked "Unknown"
4. Run it on the **current** pipeline and save the output as `baseline.txt`.

**Done when:** you have baseline numbers written down.

---

## Area 1. Team classification (+ ReID for later use)

**Gate:** always go. This is independent of video size.

**Important:** use ReID for stitching, not for team labels. ReID separates individual players. It is not built to separate two kits.

### Procedure

1. **Torso crop.** Take y from 15% to 50% and x from 25% to 75% of the bounding box.
2. **Estimate the pitch color per match.** Take the first frames, find the dominant green hue, and store it. Do not hard-code one HSV range for all stadiums.
3. **Remove grass.** Drop pixels close to that hue. Better version: a person segmentation model (YOLO11-seg or SAM 2), keep body pixels only.
4. **Get the color.** Compute the median Lab color of the remaining pixels, per frame.
5. **Aggregate per track.** Take the median over all frames of the track.
6. **Reference colors.** Sample them from your own video by using the tracks where you know the team (your kit swatch tool). Use the declared hex codes only as a fallback.
7. **Assign classes.** Classes: Home, Away, Home GK, Away GK, Referee. Use nearest reference, and use Hungarian matching if you cluster first. This makes a flip impossible.
8. **Referee gate.** Trust the detector `referee` class. Add a fallback: a track far from every kit color becomes "Other".
9. **Goalkeeper rule.** Use the detector `goalkeeper` class, the GK kit color and the pitch half.
10. **Edge cases to test by eye:** green or lime kits, white kits, yellow kits, night matches, heavy shadow.
11. **Evaluate** with `evaluate.py` and compare to `baseline.txt`.

**Done when:** team and referee accuracy on your labeled set clearly beat the baseline, and no Home/Away flip appears in any test match.

### Optional upgrade (needs a GPU)

1. Auto-label 5+ matches with the steps above.
2. Fine-tune a small classifier (ConvNeXt-Tiny or ViT-S) on masked crops as Home / Away / Referee / Other, with strong color augmentation.
3. Compare it to the rule-based version on held-out matches. Keep whichever wins.

### ReID model (used in Area 3)

1. Download SN-ReID-2023 from the Hugging Face `SoccerNet` organization (check the dataset card for the license).
2. Fine-tune a ReID backbone (for example OSNet, or a ViT) with ID loss + triplet loss.
3. Check quality with rank-1 and mAP on the validation split.
4. Export embeddings for use in stitching.

---

## Area 3. Tracking, stitching and identity assignment

**Gate:** always go. This is where most of your player-ID quality will come from.

### 3a. Single ID table (do early, half a day)

1. Create one table: `raw_track_id -> stitched_id`, plus `team` and later `lineup_player`.
2. Generate the tracking CSV, crops, stats and database rows **only** from this table.
3. Add a check that fails loudly if any output contains a raw ID that is not in the table.

**Done when:** the crop, the stats and the ID shown on an identification card all come from the same stitched ID.

### 3b. Benchmark set

1. Download a slice of SN-GSR-2025 from Hugging Face. The validation zip is about 11 GB, so start there. Check the dataset card first.
2. Run your pipeline on some clips and score tracking (for example HOTA and IDF1 with TrackEval).
3. Keep these scores next to your own labeled matches. Your own matches matter most, because they show your real footage.

### 3c. Tracking

1. Enable ReID and camera-motion compensation in BoT-SORT, or test Deep-EIoU / BoostTrack.
2. Change one thing at a time and re-score.
3. **Do not swap the detector for DETR.** Detection is not your bottleneck. If you ever want a transformer detector, try RT-DETR.

### 3d. Stitching

1. Convert every tracklet to **pitch meters** (needs stable calibration, see Area 5).
2. Build a cost between tracklets from: distance in meters, time gap, and ReID similarity.
3. Add hard rules:
   - overlapping tracklets cannot be the same player
   - same team only (from Area 1)
   - at most 11 players per team on the pitch at once
4. Solve globally (min-cost flow or constrained clustering), not greedily.

**Done when:** stitched identities per team are close to 11 (plus subs) on your labeled matches.

### 3e. Lineup assignment

1. Compute each identity's average position in pitch meters over long windows. Flip attack direction at half time.
2. Use the team's formation to make slot coordinates for each role.
3. Build a cost matrix per team: position distance in 2D + role penalty. Add a jersey-number term only if Area 2 provides one.
4. Solve one Hungarian assignment per team.
5. Attach a confidence. If the top two options are close, mark the identity "Unknown / review".
6. Show the top 3 candidates on the identify page. The user confirms only the uncertain ones.

**Done when:** role accuracy and exact-player accuracy are measured separately, and wrong forced guesses became "Unknown".

---

## Area 5. Camera calibration

**Gate:** measure first. Do not train anything until you see a problem.

### Stability check (1 day)

1. Find 5–10 short segments (5–10 s) where the camera is almost still.
2. In each segment, project one fixed pitch point (for example the penalty spot) back into the image for every frame, using your per-frame homography.
3. Measure how much it moves. On a still camera it should barely move.
4. Also check for sudden jumps, for example on camera cuts.

| Result | Action |
|---|---|
| Small movement, no jumps | Go on. No calibration work needed |
| Jitter but no wrong lines | Add temporal smoothing (RANSAC per frame, then a filter over time) |
| Wrong or missing keypoints | Train a keypoint model (HRNet) on SN-Calibration-2023 |

### If training is needed

1. Download SN-Calibration-2023 from Hugging Face (check the card for the license).
2. Train a pitch keypoint detector (HRNet or similar).
3. Fit the homography with RANSAC and smooth over time.
4. Re-run the stability check. Compare before and after.

**Done when:** the fixed-point test shows small movement and no jumps.

---

## Area 2. Jersey number OCR

**Gate:** run `check_ocr_eligibility.py` (Part A). Go only if ELIGIBLE or PARTIAL.

If PARTIAL, build only the parts that use size and orientation gates, and treat OCR as a bonus.

### Procedure

1. **Data.** Download SN-Jersey-2023 (SoccerNet downloader task `jersey-2023`, or the Hugging Face copy). Run the script in `--images-dir` mode and compare crop heights with yours. If yours are much smaller, stop here.
2. **Reference code.** Study the public jersey-number pipeline by Koshkina et al. It has: legibility classifier, pose-guided crop, scene-text recognition, and tracklet consolidation. Note that its README lists an old PyTorch version (1.9), so expect dependency work.
3. **Size gate.** Use only boxes ≥ the height you chose from the script.
4. **Orientation gate.** Use a pose model (YOLO-pose, RTMPose or ViTPose). Keep back views only.
5. **Legibility classifier.** Train a small yes/no model (for example ResNet-34): "is a number readable here?"
6. **Number crop.** Crop the region between shoulders and hips using pose keypoints.
7. **Reader.** Fine-tune PARSeq (not TrOCR) on SN-Jersey crops. Use heavy blur and compression augmentation.
8. **Constrain the output** to the numbers in that team's lineup, plus "unreadable". The team comes from Area 1, never from OCR.
9. **Track-level voting.** Sum probabilities over all frames that pass the gates. Accept a number only if several frames agree.
10. **Calibrate confidence.** Use your labeled set to find the threshold that gives high precision (for example 95%+), even if few tracks get a number.
11. **Integrate as soft evidence** in the Area 3e cost matrix. Never lock an identity from OCR alone.

Do not use super-resolution on tiny digits (it invents digits).

**Done when:** accepted numbers reach your precision target on your labeled set, and lineup accuracy improves with OCR on versus off.

**Expectation:** even good published systems reach about 87% at tracklet level on the SoccerNet data. On your wide footage, expect lower.

---

## Area 4. Event detection (passes, shots)

**Gate:** start only after Areas 1, 3 and 5 meet your targets. Event stats depend on knowing who is who, and events are not one of your current failures.

### Procedure (later phase)

1. **Measure the current method.** Label about 100 events (passes, shots) by hand from 2–3 matches. Score precision and recall of your physics-based rules.
2. **Improve the rules first.** Use ball possession from the ID table and pitch-coordinate speeds. Re-score.
3. **Only if the ceiling is too low,** try a learned spotting model on SN-BAS-2025 (check the dataset card and license).
   - Ball action spotting needs precise timing. A model that labels a whole 3-second clip is a weak fit. Use a model built for spotting (for example the T-DEED type).
   - These models are heavy on CPU, so plan for GPU inference or offline batch processing.
4. **Compare** the learned model against the improved rules on your matches.

**Done when:** the event F1 on your labeled matches is measured for both methods and you keep the better one.

---

## Part D. Final checklist

- [ ] Ground truth and `evaluate.py` ready, baseline saved
- [ ] `check_ocr_eligibility.py` run on 3+ matches, verdict recorded
- [ ] 30 tall crops checked by eye
- [ ] Turf-masked, per-track team classification, no flips
- [ ] Referee and goalkeeper rules in place
- [ ] Single ID table feeds CSV, crops, stats and database
- [ ] Calibration stability measured (smoothing or training only if needed)
- [ ] ReID model trained and used in stitching
- [ ] Stitching in pitch meters with hard rules
- [ ] Hungarian lineup assignment with "Unknown" and top-3 review
- [ ] OCR added only if eligible, as soft evidence
- [ ] Events phase only after the above hit targets
- [ ] Old vs new comparison written with real numbers

## Notes for your report

- State measured numbers only. Avoid claims like "98%" without your own test.
- If OCR is not eligible, say so with the measured box heights. It shows you tested the idea and chose the right design.
- Datasets on Hugging Face were listed under GPL-3.0 on the cards I saw. Check each card for the current terms.
