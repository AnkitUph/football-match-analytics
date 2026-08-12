from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ai_engine.tracking.tracking_runner import run_tracking
from ai_engine.team_assignment.team_assigner import TeamAssigner
from ai_engine.team_assignment.team_history import TeamHistory


HOME_COLOR = (255, 80, 80)
AWAY_COLOR = (80, 80, 255)
UNKNOWN_COLOR = (180, 180, 180)


# =========================================================
# PLAYER CROP
# =========================================================

def crop_player(frame, bbox, padding=5):
    x1 = max(0, int(bbox.x1) - padding)
    y1 = max(0, int(bbox.y1) - padding)
    x2 = min(frame.shape[1], int(bbox.x2) + padding)
    y2 = min(frame.shape[0], int(bbox.y2) + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


# =========================================================
# DRAW PLAYER
# =========================================================

def draw_player(
    frame,
    bbox,
    team,
    confidence,
    player_id,
):
    if team == TeamHistory.HOME:
        color = HOME_COLOR
    elif team == TeamHistory.AWAY:
        color = AWAY_COLOR
    else:
        color = UNKNOWN_COLOR

    x1 = int(bbox.x1)
    y1 = int(bbox.y1)
    x2 = int(bbox.x2)
    y2 = int(bbox.y2)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    label = (
        f"{team} "
        f"{confidence:.2f} "
        f"ID:{player_id}"
    )

    label_width = max(
        150,
        len(label) * 9,
    )

    cv2.rectangle(
        frame,
        (
            x1,
            max(0, y1 - 30),
        ),
        (
            x1 + label_width,
            y1,
        ),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (
            x1 + 5,
            y1 - 8,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# =========================================================
# DRAW SUMMARY
# =========================================================

def draw_summary(
    frame,
    home,
    away,
    unknown,
):
    text = (
        f"HOME: {home}   "
        f"AWAY: {away}   "
        f"UNKNOWN: {unknown}"
    )

    cv2.rectangle(
        frame,
        (15, 15),
        (500, 60),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# =========================================================
# COLLECT BALANCED PLAYER CROPS
# =========================================================

def collect_player_crops(
    video_path,
    tracks,
    max_frames=500,
    max_crops=300,
    samples_per_track=10,
):
    """
    Collect balanced player crops for team-color learning.

    Sampling strategy:

    1. Build an index of player observations by frame.
    2. Sample frames throughout the selected video range.
    3. Limit the number of crops contributed by each track.
    4. Prevent a small number of tracks from dominating
       K-Means.
    5. Return (track_id, crop) pairs for diagnostics.
    """

    capture = cv2.VideoCapture(
        video_path
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_path}"
        )

    # -----------------------------------------------------
    # Build frame -> observations index.
    # -----------------------------------------------------

    observations_by_frame = {}

    for track_id, track in tracks.items():

        observations = getattr(
            track,
            "observations",
            [],
        ) or []

        for obs in observations:

            if getattr(
                obs,
                "class_name",
                "",
            ) != "player":
                continue

            frame_index = int(
                obs.frame_index
            )

            if frame_index >= max_frames:
                continue

            observations_by_frame.setdefault(
                frame_index,
                [],
            ).append(
                (
                    int(track_id),
                    obs,
                )
            )

    # -----------------------------------------------------
    # Available frames.
    # -----------------------------------------------------

    available_frames = sorted(
        observations_by_frame.keys()
    )

    if not available_frames:
        capture.release()
        return []

    # -----------------------------------------------------
    # Sample frames across the video.
    #
    # This avoids training only on the first part of
    # the video.
    # -----------------------------------------------------

    target_frame_count = min(
        max_frames,
        len(available_frames),
    )

    if target_frame_count <= 0:
        capture.release()
        return []

    if target_frame_count == len(
        available_frames
    ):
        selected_frames = available_frames

    else:
        indices = np.linspace(
            0,
            len(available_frames) - 1,
            target_frame_count,
            dtype=int,
        )

        selected_frames = [
            available_frames[index]
            for index in indices
        ]

    selected_frame_set = set(
        selected_frames
    )

    # -----------------------------------------------------
    # Track-level sample counters.
    # -----------------------------------------------------

    track_counts = {}

    samples = []

    frame_index = 0

    # -----------------------------------------------------
    # Read video.
    # -----------------------------------------------------

    while True:

        ok, frame = capture.read()

        if not ok:
            break

        if frame_index in selected_frame_set:

            observations = (
                observations_by_frame.get(
                    frame_index,
                    [],
                )
            )

            for track_id, obs in observations:

                current_count = (
                    track_counts.get(
                        track_id,
                        0,
                    )
                )

                # -----------------------------------------
                # Prevent one player from dominating.
                # -----------------------------------------

                if (
                    current_count
                    >= samples_per_track
                ):
                    continue

                crop = crop_player(
                    frame,
                    obs.bbox,
                )

                if crop is None:
                    continue

                samples.append(
                    (
                        track_id,
                        crop,
                    )
                )

                track_counts[track_id] = (
                    current_count + 1
                )

                if len(samples) >= max_crops:
                    break

        if len(samples) >= max_crops:
            break

        frame_index += 1

        if frame_index >= max_frames:
            break

    capture.release()

    return samples


# =========================================================
# MAIN VISUAL TEST
# =========================================================

def run_team_assignment_visual_test(
    video_path: str,
    model_path: str,
    output_path: str,
    imgsz: int = 640,
):
    print("=" * 70)
    print("TEAM ASSIGNMENT VISUAL TEST")
    print("=" * 70)

    print("Video :", video_path)
    print("Model :", model_path)
    print("Output:", output_path)

    # -----------------------------------------------------
    # 1. Existing tracking
    # -----------------------------------------------------

    print()
    print(
        "[1/4] Running existing tracking pipeline..."
    )

    tracker = run_tracking(
        video_path,
        model_path,
        imgsz=imgsz,
    )

    tracks = getattr(
        tracker,
        "track_history",
        {},
    ) or {}

    print(
        "Tracked objects:",
        len(tracks),
    )

    # -----------------------------------------------------
    # 2. Collect balanced jersey samples
    # -----------------------------------------------------

    print()
    print(
        "[2/4] Collecting balanced player crops..."
    )

    samples = collect_player_crops(
        video_path,
        tracks,
        max_frames=500,
        max_crops=300,
        samples_per_track=10,
    )

    print(
        "Training samples:",
        len(samples),
    )

    # -----------------------------------------------------
    # Sample distribution diagnostics
    # -----------------------------------------------------

    sample_counts = Counter(
        track_id
        for track_id, _ in samples
    )

    print(
        "Training players:",
        len(sample_counts),
    )

    print()
    print(
        "SAMPLE DISTRIBUTION"
    )
    print("-" * 50)

    for track_id in sorted(
        sample_counts
    ):
        print(
            f"Track {track_id:>4}: "
            f"{sample_counts[track_id]:>2} samples"
        )

    # -----------------------------------------------------
    # Extract crops from samples.
    # -----------------------------------------------------

    crops = [
        crop
        for _, crop in samples
    ]

    if len(crops) < 10:
        raise RuntimeError(
            "Not enough player crops "
            "for team assignment."
        )

    # -----------------------------------------------------
    # 3. Fit team colors
    # -----------------------------------------------------

    print()
    print(
        "[3/4] Learning team colors..."
    )

    assigner = TeamAssigner(
        min_confidence=0.55,
    )

    fit_report = assigner.fit(
        crops
    )

    print()
    print(
        "TEAM COLOR CLUSTERS"
    )
    print("-" * 50)

    print(
        "Valid samples :",
        fit_report["samples"],
    )

    print(
        "HOME samples  :",
        fit_report["home_samples"],
    )

    print(
        "AWAY samples  :",
        fit_report["away_samples"],
    )

    print(
        "HOME HSV      :",
        [
            round(x, 2)
            for x in fit_report[
                "home_color_hsv"
            ]
        ],
    )

    print(
        "AWAY HSV      :",
        [
            round(x, 2)
            for x in fit_report[
                "away_color_hsv"
            ]
        ],
    )

    print(
        "Cluster separation:",
        round(
            fit_report[
                "cluster_separation"
            ],
            4,
        ),
    )

    # -----------------------------------------------------
    # 4. Render video
    # -----------------------------------------------------

    print()
    print(
        "[4/4] Rendering team assignment video..."
    )

    history = TeamHistory(
        minimum_votes=3,
    )

    capture = cv2.VideoCapture(
        video_path
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()

        raise RuntimeError(
            f"Unable to create output video: "
            f"{output_path}"
        )

    frame_index = 0
    frames_processed = 0

    totals = Counter()

    while True:

        ok, frame = capture.read()

        if not ok:
            break

        frame_assignments = []

        for track_id, track in tracks.items():

            observations = getattr(
                track,
                "observations",
                [],
            ) or []

            for obs in observations:

                if (
                    int(obs.frame_index)
                    != frame_index
                ):
                    continue

                if getattr(
                    obs,
                    "class_name",
                    "",
                ) != "player":
                    continue

                crop = crop_player(
                    frame,
                    obs.bbox,
                )

                if crop is None:
                    continue

                # -----------------------------------------
                # Instantaneous color assignment
                # -----------------------------------------

                assignment = assigner.assign(
                    int(track_id),
                    crop,
                )

                # -----------------------------------------
                # Temporal history
                # -----------------------------------------

                entry = history.update(
                    player_id=int(track_id),
                    team=assignment.team,
                    confidence=assignment.confidence,
                    frame_index=frame_index,
                )

                # -----------------------------------------
                # Stable team after voting
                # -----------------------------------------

                stable_team = history.get_team(
                    int(track_id)
                )

                if (
                    stable_team
                    == TeamHistory.UNKNOWN
                ):
                    display_team = (
                        assignment.team
                    )

                    display_confidence = (
                        assignment.confidence
                    )

                else:
                    display_team = (
                        stable_team
                    )

                    display_confidence = (
                        entry.confidence
                    )

                frame_assignments.append(
                    (
                        display_team,
                        display_confidence,
                        int(track_id),
                    )
                )

                draw_player(
                    frame,
                    obs.bbox,
                    display_team,
                    display_confidence,
                    int(track_id),
                )

        home = sum(
            1
            for team, _, _
            in frame_assignments
            if team
            == TeamHistory.HOME
        )

        away = sum(
            1
            for team, _, _
            in frame_assignments
            if team
            == TeamHistory.AWAY
        )

        unknown = sum(
            1
            for team, _, _
            in frame_assignments
            if team
            == TeamHistory.UNKNOWN
        )

        totals[
            TeamHistory.HOME
        ] += home

        totals[
            TeamHistory.AWAY
        ] += away

        totals[
            TeamHistory.UNKNOWN
        ] += unknown

        draw_summary(
            frame,
            home,
            away,
            unknown,
        )

        writer.write(
            frame
        )

        frame_index += 1
        frames_processed += 1

        if frame_index % 100 == 0:
            print(
                f"Processed frame "
                f"{frame_index}"
            )

    capture.release()
    writer.release()

    # -----------------------------------------------------
    # Final terminal report
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TEAM ASSIGNMENT RESULTS"
    )
    print("=" * 70)

    print(
        "Frames processed:",
        frames_processed,
    )

    home = totals[
        TeamHistory.HOME
    ]

    away = totals[
        TeamHistory.AWAY
    ]

    unknown = totals[
        TeamHistory.UNKNOWN
    ]

    total = (
        home
        + away
        + unknown
    )

    print()
    print(
        "FRAME-LEVEL ASSIGNMENTS"
    )
    print("-" * 50)

    print(
        "HOME   :",
        home,
    )

    print(
        "AWAY   :",
        away,
    )

    print(
        "UNKNOWN:",
        unknown,
    )

    print(
        "TOTAL  :",
        total,
    )

    if total:

        print()

        print(
            f"HOME   : "
            f"{home / total * 100:.2f}%"
        )

        print(
            f"AWAY   : "
            f"{away / total * 100:.2f}%"
        )

        print(
            f"UNKNOWN: "
            f"{unknown / total * 100:.2f}%"
        )

    # -----------------------------------------------------
    # Track-level results
    # -----------------------------------------------------

    statistics = history.statistics()

    print()
    print(
        "TRACK-LEVEL ASSIGNMENTS"
    )
    print("-" * 50)

    print(
        "HOME   :",
        statistics[
            TeamHistory.HOME
        ],
    )

    print(
        "AWAY   :",
        statistics[
            TeamHistory.AWAY
        ],
    )

    print(
        "UNKNOWN:",
        statistics[
            TeamHistory.UNKNOWN
        ],
    )

    print(
        "TOTAL  :",
        statistics["total"],
    )

    print()
    print(
        "TRACK RESULTS"
    )
    print("-" * 50)

    for player_id in sorted(
        history.entries
    ):

        entry = history.get_entry(
            player_id
        )

        if entry is None:
            continue

        print(
            f"Track {player_id:>4} -> "
            f"{entry.stable_team:<7} "
            f"confidence={entry.confidence:.3f} "
            f"assignments="
            f"{len(entry.assignments):>4} "
            f"last_frame="
            f"{entry.last_frame}"
        )

    # -----------------------------------------------------
    # DB-ready records
    # -----------------------------------------------------

    records = history.to_records()

    print()
    print(
        "DB-READY TEAM RECORDS"
    )
    print("-" * 50)

    for record in records:
        print(record)

    print()
    print("=" * 70)
    print(
        "TEAM ASSIGNMENT VIDEO SAVED"
    )
    print("=" * 70)

    print(
        output_path
    )

    return {
        "output_path": output_path,
        "frames_processed": frames_processed,
        "fit_report": fit_report,
        "frame_statistics": {
            "HOME": home,
            "AWAY": away,
            "UNKNOWN": unknown,
            "TOTAL": total,
        },
        "track_statistics": statistics,
        "team_records": records,
    }