# Solution Plan: Team Classification, Tracking, Player Identification and Jersey OCR

Goal: fix the failures listed in `CURRENT_STACK_ANALYSIS_AND_FAILURES.md` with a pipeline that is accurate, honest about what it does not know, and easy to measure.

---

## 1. The Big Picture

There are three root causes behind all the failures:

1. **Wrong pixels.** Grass fills 50–75% of each player crop, so it controls the result.
2. **Weak decisions.** The system decides frame by frame and greedily. It does not vote across a whole track and does not enforce rules (one player per number, one team per track).
3. **Missing information.** In wide shots the digits are 12–22 px tall. No model can read what is not in the pixels.

Design rules for the new pipeline:

- Decide per **track**, not per frame.
- Use **known information** (declared kit colors, lineup, formation) as anchors.
- Combine **weak signals** (team, position, number) into one global assignment.
- Allow **"Unknown"**. A missing stat is better than a wrong stat.
- Keep a **human review step** for uncertain cases.
- **Measure** every change on hand-labeled data.

---

## 2. Target Pipeline

```
Detection (player / goalkeeper / referee / ball)
        |
Tracking (BoT-SORT + ReID, or Deep-EIoU)
        |
Homography -> pitch coordinates (meters)
        |
Tracklet stitching (pitch coords + ReID + hard rules)
        |
Single ID table: raw_track_id -> stitched_id      <- one source of truth
        |
Team classification (turf-masked color, per stitched track)
        |
Lineup assignment (Hungarian: team + role/position + optional number)
        |
Human review page (only uncertain cases)
        |
Stats saved to the database
```

Jersey OCR is an **optional side input** to the lineup assignment. It is not a required stage.

---

## 3. Step-by-Step Plan

### Step 0: Build a small ground-truth set (do this first)

Without labels you cannot prove anything improved.

- Pick 2–3 matches (different kits and stadiums).
- Hand-label about 200 tracks:
  - team: Home / Away / Referee / Other
  - lineup player, where you can tell
- Save as a simple CSV: `match_id, stitched_id, true_team, true_player`.
- Write one script that prints: team accuracy, referee accuracy, role accuracy, exact-player accuracy.

**Done when:** the script runs on the current pipeline and gives baseline numbers.

---

### Step 1: Team classification (biggest, fastest win)

**Problem fixed:** grass contamination, Home/Away flip, referee mixed into a team.

1. **Crop the torso.** Use roughly y from 15% to 50% and x from 25% to 75% of the bounding box.
2. **Remove grass.** Use an HSV mask as a first version. Do not hard-code the range. Estimate the pitch hue from the first frames of the match, then remove pixels near it.
   - Better version: use a person segmentation model (YOLO11-seg or SAM 2) and keep only body pixels.
3. **Compute the color.** Take the median color (CIE-Lab) of the remaining pixels in each frame.
4. **Aggregate per track.** Take the median over many frames of the same track.
5. **Reference colors.** Use the kit colors **sampled from your own video** (your swatch tool already does this). Use the declared hex codes only as a backup, because TV lighting changes colors.
6. **Assign with fixed classes.** Classes: Home, Away, Home GK, Away GK, Referee. Match each track to the nearest reference. Use Hungarian matching if you cluster first, so a flip cannot happen.
7. **Referee gate.**
   - Trust the detector `referee` class first.
   - Add a backup: if a track is far from every kit color, label it "Other".
8. **Goalkeepers.** Use the detector `goalkeeper` class, the GK kit color and the pitch half the player stays in.

**Watch out for:**
- Green or lime kits are erased by a grass mask. Add a per-match option to turn the mask off or change its range.
- White or yellow kits can look close to bright grass. Check these cases by eye.

**Done when:** team accuracy and referee accuracy on your labeled set are clearly better than the baseline, and no flips appear on any test match.

**Optional upgrade:** auto-label a match with the steps above, then fine-tune a small classifier (ConvNeXt-Tiny or ViT-S) on masked crops as Home / Away / Referee / Other, with strong color augmentation. Train it on a GPU (Colab, Kaggle or rented) and run it on CPU.

---

### Step 2: One source of truth for IDs

**Problem fixed:** the ID desync between stitched IDs, the tracking CSV, crops and database rows.

- Create one table: `raw_track_id -> stitched_id` (plus team and, later, lineup player).
- Generate **everything else** from this table: CSV, crops, stats, database rows.
- Never write raw IDs in one place and stitched IDs in another.

**Done when:** for any player card, the crop, the stats and the ID all come from the same stitched ID.

---

### Step 3: Better tracking and tracklet stitching

**Problem fixed:** 30–60 fragments per team instead of about 11.

**Tracking**
- Turn on ReID and camera-motion compensation in BoT-SORT, or try Deep-EIoU / BoostTrack.
- Train or fine-tune an appearance model on SoccerNet-ReID.

**Stitching**
1. Compute each tracklet's position in **pitch meters** (after homography), not pixels. This survives camera pans and cuts.
2. Build a cost between tracklets from: distance in meters, time gap, and ReID appearance similarity.
3. Add **hard rules:**
   - Two tracklets that overlap in time cannot be the same player.
   - Both must be on the same team (from Step 1).
   - No more than 11 players per team on the pitch at once.
4. Solve as one global problem (min-cost flow or constrained clustering), not greedily.

**Done when:** the number of stitched identities per team is close to 11 (plus substitutes), measured on your labeled matches.

---

### Step 4: Lineup assignment (player identification)

**Problem fixed:** the cascade error and the weak 1D depth sort.

1. For each stitched identity, compute the **average position** in pitch meters over the whole match (or a long window). Flip the attack direction at half time.
2. Use the team's formation (4-3-3, 4-2-3-1, 3-5-2, ...) to create slot coordinates for each role (GK, LB, CB, RB, CM, LW, ST, ...).
3. For each team, build a cost matrix: identity vs lineup player.
   - Cost from position: distance to the formation slot, in 2D.
   - Cost from role: penalty if the role does not match the lineup position.
   - Cost from jersey number (only if Step 5 gives one): strongly favors that number, but is not final.
   - Team rule: only players from that team's lineup are allowed.
4. Solve one **Hungarian assignment per team**.
5. Give each assignment a confidence. If the best and second-best options are close, mark it **Unknown / needs review**. Two center backs or two central midfielders will often be close, and that is normal.
6. Show the top 3 candidates per identity on your `/identify/` page and ask the user to confirm only the uncertain ones.

**Expect:** good accuracy for the role, weaker accuracy for the exact player inside a line, until the user confirms. Report both numbers separately.

**Known weak points:**
- Substitutions, players changing position, set pieces.
- Broadcast cameras follow the ball, so average positions are biased.
- Formation may change during the match.

**Done when:** role accuracy and exact-player accuracy on the labeled set are measured, and wrong forced guesses are replaced by "Unknown".

---

### Step 5: Jersey OCR (optional, do last)

Only do this if your footage has close-ups, replays or zoomed shots. The physical limit does not change: if the digits are 12–22 px, no model will read them.

1. **Size gate.** Run OCR only when the player box is tall enough (start with 150–200 px and tune on your data).
2. **Orientation gate.** Train a tiny classifier (front / back / side) on your crops, or use a pose model. Do not rely on aspect ratio alone. Skip everything that is not a back view.
3. **Legibility check.** Train a small yes/no classifier: "is a number readable in this crop?" Only readable crops go on.
4. **Reader.** Fine-tune a text-recognition model (PARSeq is a good choice) on real crops, with heavy blur and compression augmentation.
   - Restrict the output to **numbers in that team's lineup**, plus an "unreadable" class.
   - The team comes from Step 1, never from OCR.
5. **Track-level voting.** Combine probabilities over all good frames of a track. Accept a number only if several frames agree.
6. **Use as soft evidence** in the Step 4 cost, never as a locked answer. A confidence value such as 0.70 is usually not calibrated, so calibrate it on your labeled set before trusting it.

**Do not use:**
- Super-resolution on tiny digits. It invents digits.
- Synthetic-only training data.

**Data:** SoccerNet Jersey Number dataset. It has number labels per tracklet, not boxes, so a recognition model is easier to train than a box detector.

**Done when:** on tracks that pass all the gates, precision of accepted numbers is very high (choose your target, for example 95%+), even if only a small share of tracks get a number.

---

### Step 6: Compare and report

- Run the old pipeline and the new one on the same labeled matches.
- Report these separately: team accuracy, referee accuracy, stitched identities per team, role accuracy, exact-player accuracy, and how many tracks were left "Unknown".
- Be honest in your report that wide broadcast footage limits jersey OCR, and explain why a hybrid design (color + position + optional number + human check) is the right answer.

---

## 4. Order of Work

| Order | Step | Effort | Needs GPU training? |
|---|---|---|---|
| 1 | Step 0: labeled set and metric script | 1 day | No |
| 2 | Step 1: turf-masked team classification | 1–2 days | No (optional later) |
| 3 | Step 2: single ID table | 0.5 day | No |
| 4 | Step 3: stitching in pitch coordinates + ReID | 3–5 days | Optional (ReID) |
| 5 | Step 4: Hungarian lineup assignment + review page | 3–4 days | No |
| 6 | Step 5: gated OCR | 1–2 weeks | Yes |
| 7 | Step 6: comparison and report | 1–2 days | No |

If time is short, Steps 0–2 and Step 4 already give the biggest improvement.

---

## 5. Practical Notes

- **Compute:** your pipeline runs on CPU-only PyTorch. Train models on Colab, Kaggle or a rented GPU, export them, and run inference on CPU. Prefer small models for inference (ViT-S, ConvNeXt-Tiny, ResNet) or distill bigger ones.
- **Reference project:** SoccerNet Game State Reconstruction (`sn-gamestate` / TrackLab) is close to your pipeline. Use it to compare design choices.
- **Datasets:** SoccerNet-ReID (appearance), SoccerNet Jersey Number (OCR), SoccerNet Game State Reconstruction (whole pipeline).
- **Claims to avoid:** do not quote accuracy numbers such as "98%" or "100% deterministic" unless your own labeled set shows them.

---

## 6. Checklist

- [ ] Ground-truth labels and metric script ready
- [ ] Baseline numbers recorded for the old pipeline
- [ ] Turf-masked, per-track team classification working, no flips
- [ ] Referee and goalkeeper rules in place
- [ ] Single ID table used by CSV, crops, stats and database
- [ ] Stitching in pitch coordinates with ReID and hard rules
- [ ] Hungarian lineup assignment with "Unknown" and top-3 review
- [ ] (Optional) Gated, track-level OCR added as soft evidence
- [ ] Old vs new comparison written up with real numbers
