# Current Stack Architecture & Root Cause Failure Analysis

---

## 1. Overview of the Current Pipeline

The current AI analytics pipeline processes broadcast match video across the following sequential stages:

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│ 1. Detection    │ ──> │ 2. Tracking      │ ──> │ 3. Team Re-ID & OCR   │
│ YOLOv8x         │     │ BoT-SORT         │     │ DINOv2 + BiLSTM Net   │
└─────────────────┘     └──────────────────┘     └───────────────────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│ 6. Reporting    │ <── │ 5. Events & xG   │ <── │ 4. Pitch Homography   │
│ PDF & Web Studio│     │ Ball/Player Logic│     │ Keypoints + Stitching │
└─────────────────┘     └──────────────────┘     └───────────────────────┘
```

---

## 2. Component-by-Component Failure Analysis

### Component 1: Team Classification (Stage 3)
* **Current Tech**: DINOv2 ViT-Small/14 embeddings (384-d) + K-Means clustering ($k=2$) with CIE-L\*a\*b\* perceptual kit distance fallback.
* **Intended Goal**: Separate all detected player tracklets into Home (Team A) and Away (Team B) kits.
* **Why It Fails**:
  1. **Pitch Turf Dominance**: Player bounding boxes in broadcast camera angles contain 50% to 75% green grass turf pixels. Because DINOv2 computes a global spatial feature embedding across the full bounding box, pitch turf features dominate the embedding space over jersey fabric colors.
  2. **Cluster Inversion / Swap**: Unsupervised K-Means clustering creates two partitions ($C_0, C_1$) but does not know which cluster corresponds to Home vs Away. When mapping clusters to declared team kit colors (`#d71920` Red vs `#ffd700` Yellow), shadows and turf noise distort the cluster medians, frequently flipping the assignment so Yellow players are labeled `Team A (Home)` and Red players are labeled `Team B (Away)`.
  3. **Referee Conflation**: Referees wearing contrasting shirts (e.g. Cyan, Black, Neon Green) do not fit either team cluster and get pulled into whichever team cluster has higher variance, causing referees to be assigned to outfield squad players.

---

### Component 2: Jersey Number Recognition (Stage 3c)
* **Current Tech**: Sampled torso crops + `JerseyNumberTemporalNet` (EfficientNet-B0 backbone + 2-layer BiLSTM + Temporal Attention + Independent Tens/Units digit heads).
* **Intended Goal**: Read player jersey numbers from broadcast video to automatically identify players.
* **Why It Fails**:
  1. **Extreme Low Resolution**: In a wide 1080p tactical broadcast shot, player bounding boxes are only 80–130 pixels high. The torso region is ~40×50 pixels, and the jersey number itself spans only **12 to 22 pixels**.
  2. **Angle & Occlusion Blindness**: Jersey numbers are only visible when a player's back is facing the camera (~15–20% of match time). In the other 80% of frames, players face forward, sideways, or are occluded. Passing front/side crops into an OCR or classification network forces the model to hallucinate digits from kit sponsors, crests, fabric folds, or shadows.
  3. **Synthetic Training Domain Gap**: Models trained on high-contrast synthetic datasets or cropped digit images fail when encountering real-world broadcast artifacts like H.264 compression blocks, motion blur, and interlacing.

---

### Component 3: Tracklet Stitching (Stage 5)
* **Current Tech**: Spatial-temporal Hungarian matching (`match_tracklets_within_shot`) using pixel proximity and duration thresholds.
* **Intended Goal**: Merge short, fragmented ByteTrack detections of the same player into a single continuous trajectory per match.
* **Why It Fails**:
  1. **Disconnection from Tracking Output**: When tracklets are merged into `stitched` root IDs, the underlying `player_tracking_csv` was previously retaining unstitched raw IDs, creating desynchronization between crop coordinates, track stats, and database IDs.
  2. **Single-Shot Limitation**: When camera cuts or rapid pans occur, purely spatial distance thresholds exceed the allowed gap, leaving 30–60 separate tracklet fragments per team instead of the theoretical 11 players.

---

### Component 4: Player Identification & Lineup Mapping
* **Current Tech**: Greedy direct jersey matching fallback to pitch depth ($x$-coordinate ranking).
* **Intended Goal**: Link each tracked identity to a real lineup player (e.g., #10 Morgan Rogers).
* **Why It Fails**:
  1. **Cascading Hallucination Effect**: If the OCR model erroneously predicts number `#11` on a Yellow away player with low confidence, the system greedily assigns that track to Home #11 (Isaiah Jones). This removes the correct player from the pool and displaces all subsequent outfield players.
  2. **Unconstrained Spatial Depth**: Simple 1D $x$-coordinate sorting assumes teams play in static horizontal lines, ignoring lateral width ($y$-coordinates for Left Back vs Right Back) and dynamic tactical shifts.

---

## 3. Summary of Core Failure Dynamics

| Failure Mode | Root Trigger | Cascade Effect |
| :--- | :--- | :--- |
| **Team Color Flip** | Turf contamination in DINOv2 bounding boxes | Away players labeled Home; Home players labeled Away |
| **Referee as Player** | Lack of strict referee rejection gate | Referee assigned to a starting outfield player |
| **Digit Hallucination** | Running OCR on front/side player angles | Wrong player profile attached; corrupts real stats |
| **ID Desync** | Stitching tracklets without updating tracking CSV | Wrong crops displayed on identification cards |
