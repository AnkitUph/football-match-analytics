"""
Pairwise Identity Visual Inspector.

Purpose:
    Generate side-by-side visual evidence for HIGH_RISK / CRITICAL
    accepted identity assignments.

For every risky accepted match:

    previous ByteTrack fragment
                +
    newly merged ByteTrack fragment
                ↓
        side-by-side image

The inspector does NOT modify:
    - IdentityManager
    - Identity
    - Track
    - TrackObservation
    - matcher configuration

It only creates external image evidence and a JSON manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import cv2

from ai_engine.validation.identity_visual.identity_visual_validator import (
    IdentityVisualValidator,
)


VIDEO_PATH = "media/test_clips/test_11.mp4"
OUTPUT_DIR = "media/identity_pairwise_inspection"

# Number of risky accepted matches to inspect.
# None = all risky matches.
MAX_RISKY_MATCHES: Optional[int] = None

# Add some padding around the player bounding box.
CROP_PADDING = 80

# Size of each cropped player panel.
CROP_WIDTH = 500
CROP_HEIGHT = 500


def read_frame(
    capture: cv2.VideoCapture,
    frame_index: int,
) -> Optional[Any]:
    """Read one video frame."""
    if frame_index < 0:
        return None

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_index),
    )

    ok, frame = capture.read()

    if not ok:
        return None

    return frame


def get_track_observation(
    track: Any,
    frame_index: int,
) -> Optional[Any]:
    """Find the observation nearest to a requested frame."""
    if track is None:
        return None

    observations = list(
        getattr(track, "observations", []) or []
    )

    if not observations:
        return None

    exact = [
        observation
        for observation in observations
        if getattr(
            observation,
            "frame_index",
            None,
        ) == frame_index
    ]

    if exact:
        return exact[0]

    return min(
        observations,
        key=lambda observation: abs(
            int(
                getattr(
                    observation,
                    "frame_index",
                    0,
                )
            )
            - int(frame_index)
        ),
    )


def crop_observation(
    frame: Any,
    observation: Any,
    padding: int = CROP_PADDING,
) -> Any:
    """
    Crop around a player's bounding box.

    Keeps the crop square-ish and includes context around
    the player so jersey/color/background remain visible.
    """

    if observation is None:
        return frame

    bbox = getattr(
        observation,
        "bbox",
        None,
    )

    if bbox is None:
        return frame

    x1 = int(bbox.x1)
    y1 = int(bbox.y1)
    x2 = int(bbox.x2)
    y2 = int(bbox.y2)

    height, width = frame.shape[:2]

    x1 = max(
        0,
        x1 - padding,
    )

    y1 = max(
        0,
        y1 - padding,
    )

    x2 = min(
        width,
        x2 + padding,
    )

    y2 = min(
        height,
        y2 + padding,
    )

    if x2 <= x1 or y2 <= y1:
        return frame

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    return crop


def resize_crop(
    crop: Any,
    width: int = CROP_WIDTH,
    height: int = CROP_HEIGHT,
) -> Any:
    """Resize crop to a consistent inspection size."""

    return cv2.resize(
        crop,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def add_panel_header(
    image: Any,
    lines: list[str],
) -> Any:
    """Add readable metadata above a crop."""

    header_height = 42 + (
        28 * len(lines)
    )

    output = cv2.copyMakeBorder(
        image,
        header_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )

    y = 28

    for line in lines:
        cv2.putText(
            output,
            str(line)[:90],
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 28

    return output


def add_center_metadata(
    image: Any,
    lines: list[str],
) -> Any:
    """Add diagnostic information between the two panels."""

    width = image.shape[1]
    height = image.shape[0]

    overlay_height = 180

    overlay = cv2.copyMakeBorder(
        image,
        0,
        overlay_height,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(10, 10, 10),
    )

    y = height + 30

    for line in lines:
        cv2.putText(
            overlay,
            str(line)[:140],
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 28

    return overlay


def choose_previous_track(
    identity: Any,
    current_track_id: int,
    tracks: dict[Any, Any],
) -> Optional[Any]:
    """
    Find the ByteTrack fragment immediately preceding the
    newly accepted fragment.

    This is the important part of the pairwise inspection.

    Example:

        GID_0012
        tracks = [12, 68, 73, 83, 85, 123]

        current = 83

        previous candidate:
            track 73

        because track 73 ended immediately before track 83.
    """

    current_track = tracks.get(
        current_track_id
    )

    if current_track is None:
        return None

    current_first = getattr(
        current_track,
        "first_frame",
        None,
    )

    if current_first is None:
        return None

    source_track_ids = list(
        getattr(
            identity,
            "source_track_ids",
            [],
        )
        or []
    )

    candidates = []

    for track_id in source_track_ids:

        if int(track_id) == int(
            current_track_id
        ):
            continue

        track = tracks.get(track_id)

        if track is None:
            continue

        last_frame = getattr(
            track,
            "last_frame",
            None,
        )

        if last_frame is None:
            continue

        # Prefer tracks that ended before the
        # current track started.
        if last_frame <= current_first:
            gap = (
                current_first
                - last_frame
            )

            candidates.append(
                (
                    gap,
                    track,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[0][1]


def draw_bbox(
    frame: Any,
    observation: Any,
    label: str,
) -> Any:
    """Draw bounding box and label."""

    if observation is None:
        return frame

    bbox = getattr(
        observation,
        "bbox",
        None,
    )

    if bbox is None:
        return frame

    x1 = int(bbox.x1)
    y1 = int(bbox.y1)
    x2 = int(bbox.x2)
    y2 = int(bbox.y2)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        3,
    )

    cv2.putText(
        frame,
        label,
        (
            x1,
            max(
                30,
                y1 - 10,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def generate_pairwise_image(
    *,
    result: dict[str, Any],
    identity: Any,
    previous_track: Any,
    current_track: Any,
    capture: cv2.VideoCapture,
    output_path: Path,
) -> Optional[dict[str, Any]]:
    """
    Generate:

        previous fragment | current fragment

    using the end of the previous track and the beginning
    of the current track.
    """

    previous_frame_index = getattr(
        previous_track,
        "last_frame",
        None,
    )

    current_frame_index = getattr(
        current_track,
        "first_frame",
        None,
    )

    if (
        previous_frame_index is None
        or current_frame_index is None
    ):
        return None

    previous_frame = read_frame(
        capture,
        previous_frame_index,
    )

    current_frame = read_frame(
        capture,
        current_frame_index,
    )

    if (
        previous_frame is None
        or current_frame is None
    ):
        return None

    previous_observation = (
        get_track_observation(
            previous_track,
            previous_frame_index,
        )
    )

    current_observation = (
        get_track_observation(
            current_track,
            current_frame_index,
        )
    )

    previous_frame = draw_bbox(
        previous_frame,
        previous_observation,
        (
            f"Track {previous_track.local_id}"
            " | PREVIOUS"
        ),
    )

    current_frame = draw_bbox(
        current_frame,
        current_observation,
        (
            f"Track {current_track.local_id}"
            " | ACCEPTED"
        ),
    )

    previous_crop = crop_observation(
        previous_frame,
        previous_observation,
    )

    current_crop = crop_observation(
        current_frame,
        current_observation,
    )

    previous_crop = resize_crop(
        previous_crop
    )

    current_crop = resize_crop(
        current_crop
    )

    previous_crop = add_panel_header(
        previous_crop,
        [
            (
                f"PREVIOUS FRAGMENT | "
                f"track={previous_track.local_id}"
            ),
            (
                f"frame={previous_frame_index}"
            ),
        ],
    )

    current_crop = add_panel_header(
        current_crop,
        [
            (
                f"NEW FRAGMENT | "
                f"track={current_track.local_id}"
            ),
            (
                f"frame={current_frame_index}"
            ),
        ],
    )

    # Both panels now have the same height.
    panel_height = max(
        previous_crop.shape[0],
        current_crop.shape[0],
    )

    def pad_height(
        image: Any,
    ) -> Any:
        difference = (
            panel_height
            - image.shape[0]
        )

        if difference <= 0:
            return image

        return cv2.copyMakeBorder(
            image,
            0,
            difference,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(20, 20, 20),
        )

    previous_crop = pad_height(
        previous_crop
    )

    current_crop = pad_height(
        current_crop
    )

    combined = cv2.hconcat(
        [
            previous_crop,
            current_crop,
        ]
    )

    temporal_gap = (
        current_frame_index
        - previous_frame_index
    )

    metadata_lines = [
        (
            f"{result['identity_id']} | "
            f"{result.get('risk_level', 'UNKNOWN')}"
        ),
        (
            f"score={result.get('score', 0.0):.3f} | "
            f"appearance="
            f"{result.get('appearance_similarity', -1.0):.3f}"
        ),
        (
            f"temporal_gap={result.get('temporal_gap', 0)} | "
            f"actual_fragment_gap={temporal_gap}"
        ),
        (
            f"spatial={result.get('spatial_distance', 0.0):.2f} | "
            f"motion={result.get('motion_difference', 0.0):.2f}"
        ),
        (
            f"team_match={result.get('team_match')} | "
            f"jersey_match={result.get('jersey_match')}"
        ),
    ]

    combined = add_center_metadata(
        combined,
        metadata_lines,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ok = cv2.imwrite(
        str(output_path),
        combined,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            94,
        ],
    )

    if not ok:
        return None

    return {
        "identity_id": result[
            "identity_id"
        ],
        "current_track_id": int(
            current_track.local_id
        ),
        "previous_track_id": int(
            previous_track.local_id
        ),
        "previous_frame": int(
            previous_frame_index
        ),
        "current_frame": int(
            current_frame_index
        ),
        "actual_fragment_gap": int(
            temporal_gap
        ),
        "validator_temporal_gap": int(
            result.get(
                "temporal_gap",
                0,
            )
        ),
        "score": float(
            result.get(
                "score",
                0.0,
            )
        ),
        "appearance_similarity": float(
            result.get(
                "appearance_similarity",
                -1.0,
            )
        ),
        "spatial_distance": float(
            result.get(
                "spatial_distance",
                0.0,
            )
        ),
        "motion_difference": float(
            result.get(
                "motion_difference",
                0.0,
            )
        ),
        "risk_level": result.get(
            "risk_level"
        ),
        "status": result.get(
            "status"
        ),
        "validation_warnings": list(
            result.get(
                "validation_warnings",
                [],
            )
        ),
        "matcher_reasons": list(
            result.get(
                "matcher_reasons",
                [],
            )
        ),
        "team_match": result.get(
            "team_match"
        ),
        "jersey_match": result.get(
            "jersey_match"
        ),
        "image_path": str(
            output_path
        ),
    }


def build_pairwise_inspection(
    manager: Any,
    tracks: dict[Any, Any],
    video_path: str,
    output_dir: str,
    max_risky_matches: Optional[int] = None,
) -> dict[str, Any]:
    """
    Main pairwise inspection function.
    """

    output_path = Path(
        output_dir
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    validator = IdentityVisualValidator()

    validator.validate_manager(
        manager
    )

    risky_matches = [
        result.to_dict()
        for result in validator.match_results
        if (
            result.matched
            and result.risk_level
            in {
                "HIGH",
                "CRITICAL",
            }
        )
    ]

    # Highest-risk first.
    risky_matches.sort(
        key=lambda result: (
            0
            if result.get(
                "risk_level"
            )
            == "CRITICAL"
            else 1,
            -float(
                result.get(
                    "spatial_distance",
                    0.0,
                )
            ),
            -float(
                result.get(
                    "motion_difference",
                    0.0,
                )
            ),
        )
    )

    if max_risky_matches is not None:
        risky_matches = risky_matches[
            :max_risky_matches
        ]

    print()
    print(
        "=" * 70
    )
    print(
        "PAIRWISE IDENTITY VISUAL INSPECTION"
    )
    print(
        "=" * 70
    )
    print(
        "Risky accepted matches:",
        len(risky_matches),
    )

    capture = cv2.VideoCapture(
        video_path
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    evidence = []

    try:

        for index, result in enumerate(
            risky_matches,
            start=1,
        ):

            identity_id = result[
                "identity_id"
            ]

            current_track_id = result[
                "track_id"
            ]

            identity = (
                manager.identities.get(
                    identity_id
                )
            )

            current_track = tracks.get(
                current_track_id
            )

            print()
            print(
                f"[{index}/{len(risky_matches)}] "
                f"{identity_id}"
            )
            print(
                "  current track:",
                current_track_id,
            )
            print(
                "  risk:",
                result["risk_level"],
            )
            print(
                "  appearance:",
                result[
                    "appearance_similarity"
                ],
            )
            print(
                "  score:",
                result["score"],
            )
            print(
                "  warnings:",
                result[
                    "validation_warnings"
                ],
            )

            if identity is None:
                print(
                    "  SKIP: identity not found"
                )
                continue

            if current_track is None:
                print(
                    "  SKIP: current track not found"
                )
                continue

            previous_track = (
                choose_previous_track(
                    identity,
                    current_track_id,
                    tracks,
                )
            )

            if previous_track is None:
                print(
                    "  SKIP: previous fragment "
                    "could not be determined"
                )
                continue

            filename = (
                f"{index:03d}_"
                f"{identity_id}_"
                f"prev_{previous_track.local_id}_"
                f"current_{current_track.local_id}_"
                f"{result['risk_level'].lower()}.jpg"
            )

            image_path = (
                output_path
                / filename
            )

            item = generate_pairwise_image(
                result=result,
                identity=identity,
                previous_track=previous_track,
                current_track=current_track,
                capture=capture,
                output_path=image_path,
            )

            if item is None:
                print(
                    "  SKIP: image generation failed"
                )
                continue

            evidence.append(item)

            print(
                "  previous track:",
                previous_track.local_id,
            )
            print(
                "  previous last frame:",
                previous_track.last_frame,
            )
            print(
                "  current first frame:",
                current_track.first_frame,
            )
            print(
                "  generated:",
                image_path,
            )

    finally:
        capture.release()

    manifest = {
        "video": video_path,
        "total_risky_matches": len(
            risky_matches
        ),
        "generated_pairs": len(
            evidence
        ),
        "evidence": evidence,
    }

    manifest_path = (
        output_path
        / "pairwise_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 70
    )
    print(
        "PAIRWISE INSPECTION COMPLETE"
    )
    print(
        "=" * 70
    )
    print(
        "Risky matches:",
        len(risky_matches),
    )
    print(
        "Pairwise images:",
        len(evidence),
    )
    print(
        "Output:",
        output_path,
    )
    print(
        "Manifest:",
        manifest_path,
    )

    return manifest