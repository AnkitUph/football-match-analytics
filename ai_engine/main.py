"""Standalone orchestration for the framework-independent AI stages.

CLI example: ``python -m ai_engine.main clip.mp4 --detect-shots --reid``

For pitch-space identities and events, pass a per-shot calibration mapping:
``{shot_id: (calibration_frame, image_points, pitch_points)}``. Without it,
the pipeline still returns shot-scoped tracklets, team labels, and a pixel-space
ball trajectory, while explicitly reporting that Stages 5-6 were skipped.
"""

import logging
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from ai_engine.config import DEFAULT_CONFIG, PipelineConfig
from ai_engine.stage2_tracking.tracker import Tracker
from ai_engine.stage5_pitch_mapping.identity_association import (
    IdentityGallery,
    match_tracklets_across_cut,
)
from ai_engine.utils.types import (
    BallTrajectoryPoint,
    Event,
    MasterIdentity,
    ObjectClass,
    PitchPoint,
    ShotSegment,
    ShotType,
    Team,
    Tracklet,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    identities: list[MasterIdentity] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    shot_segments: list[ShotSegment] = field(default_factory=list)
    tracklets: list[Tracklet] = field(default_factory=list)
    # Stage 4 uses BallTrajectoryPoint for pixel positions as well as pitch
    # positions; these are source-video pixel coordinates until calibrated.
    ball_trajectory_px: list[BallTrajectoryPoint] = field(default_factory=list)
    ball_trajectory_pitch: list[BallTrajectoryPoint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    team_sides_confirmed: bool = False


def _read_tracklet_crops(
    video_path: str,
    tracklets: list[Tracklet],
    max_samples_per_tracklet: int = 3,
) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
    """Read a few temporally spread crops per tracklet in one forward pass."""
    frames_needed: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    for tracklet in tracklets:
        detections = tracklet.detections
        if not detections or tracklet.cls not in (ObjectClass.PLAYER, ObjectClass.GOALKEEPER):
            continue
        sample_count = min(max_samples_per_tracklet, len(detections))
        sample_indices = sorted({
            round(i * (len(detections) - 1) / max(sample_count - 1, 1))
            for i in range(sample_count)
        })
        for idx in sample_indices:
            detection = detections[idx]
            frames_needed[detection.frame_idx].append((tracklet.track_id, detection.bbox))

    if not frames_needed:
        return {}, {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Could not open video for Stage 3 sampling: {video_path}")

    crops_by_track: dict[int, list[np.ndarray]] = defaultdict(list)
    colors_by_track: dict[int, list[np.ndarray]] = defaultdict(list)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    current_pos = 0
    try:
        from ai_engine.stage3_team_reid.team_classifier import sample_torso_color

        for target_frame in sorted(frames_needed):
            if target_frame != current_pos:
                if 0 < target_frame - current_pos <= 8:
                    while current_pos < target_frame:
                        cap.grab()
                        current_pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    current_pos = target_frame

            ok, frame = cap.read()
            current_pos += 1
            if not ok or frame is None:
                continue

            for track_id, bbox in frames_needed[target_frame]:
                x1, y1, x2, y2 = map(int, bbox)
                x1, x2 = max(0, x1), min(frame_width, x2)
                y1, y2 = max(0, y1), min(frame_height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2].copy()
                if crop.shape[0] < 15 or crop.shape[1] < 8:
                    continue
                crops_by_track[track_id].append(crop)
                color = sample_torso_color(crop)
                if color is not None:
                    colors_by_track[track_id].append(color)
    finally:
        cap.release()

    return dict(crops_by_track), dict(colors_by_track)


def _embed_tracklets(
    tracklets: list[Tracklet],
    crops_by_track: dict[int, list[np.ndarray]],
    config: PipelineConfig,
) -> dict[int, np.ndarray]:
    if not config.enable_reid:
        return {}

    from ai_engine.stage3_team_reid.reid import ReidEmbedder

    embedder = ReidEmbedder(config.team_reid)
    embeddings: dict[int, np.ndarray] = {}
    for tracklet in tracklets:
        crops = crops_by_track.get(tracklet.track_id, [])
        if not crops:
            continue
        vectors = [v for v in embedder.embed_batch(crops) if v is not None]
        if not vectors:
            continue
        vector = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            embeddings[tracklet.track_id] = vector / norm
    return embeddings


def _stitch_tracklets_within_shots(
    tracklets: dict[int, Tracklet],
    embeddings: dict[int, np.ndarray],
    colors: dict[int, np.ndarray],
    color_samples: dict[int, list[np.ndarray]],
    config: PipelineConfig,
) -> tuple[dict[int, Tracklet], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, list[np.ndarray]]]:
    if not config.tracking.enable_stitching or len(tracklets) < 2:
        return tracklets, embeddings, colors, color_samples

    from ai_engine.stage2_tracking.stitcher import stitch_tracklets

    stitched_all: dict[int, Tracklet] = {}
    stitched_embeddings: dict[int, np.ndarray] = {}
    stitched_colors: dict[int, np.ndarray] = {}
    stitched_color_samples: dict[int, list[np.ndarray]] = {}
    groups_by_shot: dict[int, dict[int, Tracklet]] = defaultdict(dict)
    for track_id, tracklet in tracklets.items():
        groups_by_shot[tracklet.shot_id][track_id] = tracklet

    for shot_tracks in groups_by_shot.values():
        merged, id_mapping = stitch_tracklets(
            shot_tracks,
            embeddings,
            max_gap_frames=config.tracking.stitch_max_gap_frames,
            similarity_thresh=config.tracking.stitch_similarity_thresh,
        )
        members_by_root: dict[int, list[int]] = defaultdict(list)
        for old_id, root_id in id_mapping.items():
            members_by_root[root_id].append(old_id)
        for root_id, tracklet in merged.items():
            stitched_all[root_id] = tracklet
            member_ids = members_by_root.get(root_id, [root_id])
            member_vectors = [embeddings[tid] for tid in member_ids if tid in embeddings]
            if member_vectors:
                vector = np.mean(np.asarray(member_vectors, dtype=np.float32), axis=0)
                norm = np.linalg.norm(vector)
                if norm > 1e-6:
                    stitched_embeddings[root_id] = vector / norm
                    tracklet.reid_embedding = stitched_embeddings[root_id].tolist()
            member_colors = [c for tid in member_ids for c in color_samples.get(tid, [])]
            if member_colors:
                stitched_color_samples[root_id] = member_colors
                stitched_colors[root_id] = np.median(np.asarray(member_colors), axis=0)

    return stitched_all, stitched_embeddings, stitched_colors, stitched_color_samples


def _add_tracklets_to_gallery(
    tracklets: list[Tracklet], gallery: IdentityGallery
) -> dict[int, int]:
    """Create first-shot IDs for classifiable players, preferring long tracks."""
    assignments: dict[int, int] = {}
    eligible = [t for t in tracklets if t.team in (Team.TEAM_A, Team.TEAM_B)]
    for tracklet in sorted(eligible, key=lambda t: -t.duration_frames):
        if gallery.can_add(tracklet.team):
            identity = gallery.add(tracklet)
            assignments[tracklet.track_id] = identity.master_id
    return assignments


def _build_identities(
    video_path: str,
    ball_trajectory_px: list[BallTrajectoryPoint],
    tracklets: list[Tracklet],
    shot_segments: list[ShotSegment],
    config: PipelineConfig,
    calibration_by_shot: dict[int, tuple[int, list[tuple[float, float]], list[tuple[float, float]]]] | None,
    warnings: list[str],
) -> tuple[list[MasterIdentity], list[BallTrajectoryPoint], list[Event]]:
    from ai_engine.stage5_pitch_mapping.homography import image_point_to_pitch
    from ai_engine.stage5_pitch_mapping.homography_tracker import HomographyTracker
    from ai_engine.stage6_event_detection.events import detect_passes, detect_possession, detect_shots

    calibration_by_shot = calibration_by_shot or {}
    shots_by_id = {shot.shot_id: shot for shot in shot_segments}
    homography_by_frame: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(video_path)
    try:
        for shot_id, (calibration_frame, image_points, pitch_points) in calibration_by_shot.items():
            shot = shots_by_id.get(shot_id)
            if shot is None:
                raise ValueError(f"Calibration provided for unknown shot_id={shot_id}")
            if shot.shot_type != ShotType.MAIN_WIDE:
                raise ValueError(f"Shot {shot_id} is {shot.shot_type.value}; only main_wide shots can be calibrated")
            if not shot.start_frame <= calibration_frame < shot.end_frame:
                raise ValueError(f"Calibration frame {calibration_frame} is outside shot {shot_id}")
            if not cap.isOpened():
                raise IOError("Could not open source video for pitch calibration")
            cap.set(cv2.CAP_PROP_POS_FRAMES, calibration_frame)
            ok, frame = cap.read()
            tracker = HomographyTracker(config.pitch_mapping)
            if not ok or not tracker.bootstrap(frame, image_points, pitch_points):
                warnings.append(f"Stage 5 skipped shot {shot_id}: calibration could not be initialized")
                continue
            homography_by_frame[calibration_frame] = tracker.current_H.copy()
            cap.set(cv2.CAP_PROP_POS_FRAMES, calibration_frame + 1)
            for frame_idx in range(calibration_frame + 1, shot.end_frame):
                ok, frame = cap.read()
                if not ok:
                    break
                matrix = tracker.update(frame)
                if matrix is None or abs(matrix[2, 2]) <= 1e-6:
                    warnings.append(
                        f"Shot {shot_id}: pitch mapping stopped at frame {frame_idx} "
                        "after tracking lost a trustworthy homography"
                    )
                    break
                normalized = matrix / matrix[2, 2]
                if not np.isfinite(normalized).all():
                    warnings.append(
                        f"Shot {shot_id}: pitch mapping stopped at frame {frame_idx} "
                        "after a non-finite homography estimate"
                    )
                    break
                homography_by_frame[frame_idx] = normalized.copy()
            expected_frames = shot.end_frame - calibration_frame
            mapped_frames = sum(
                1 for frame_idx in range(calibration_frame, shot.end_frame)
                if frame_idx in homography_by_frame
            )
            if mapped_frames < expected_frames:
                warnings.append(
                    f"Shot {shot_id}: pitch mapping is unavailable on "
                    f"{expected_frames - mapped_frames} frames after calibration"
                )
    finally:
        cap.release()

    gallery = IdentityGallery(config.pitch_mapping)
    track_to_master: dict[int, int] = {}
    tracks_by_shot: dict[int, list[Tracklet]] = defaultdict(list)
    for tracklet in tracklets:
        if shots_by_id.get(tracklet.shot_id, None) and shots_by_id[tracklet.shot_id].shot_type == ShotType.MAIN_WIDE:
            tracks_by_shot[tracklet.shot_id].append(tracklet)

    if config.enable_reid:
        for shot_id in sorted(tracks_by_shot):
            shot_tracks = tracks_by_shot[shot_id]
            assignments = (
                _add_tracklets_to_gallery(shot_tracks, gallery)
                if not gallery.identities
                else match_tracklets_across_cut(
                    shot_tracks, gallery, config.team_reid.reid_similarity_threshold
                )
            )
            track_to_master.update(assignments)
        identities_by_id = {i.master_id: i for i in gallery.identities}
    else:
        identities_by_id = {}
        for tracklet in tracklets:
            shot = shots_by_id.get(tracklet.shot_id)
            if shot and shot.shot_type == ShotType.MAIN_WIDE and tracklet.team != Team.REFEREE:
                identities_by_id[tracklet.track_id] = MasterIdentity(
                    master_id=tracklet.track_id,
                    team=tracklet.team,
                    reid_embedding=tracklet.reid_embedding or [],
                    jersey_number=tracklet.jersey_number,
                    cls=tracklet.cls,
                )
                track_to_master[tracklet.track_id] = tracklet.track_id
        warnings.append("Re-ID is disabled; identities are track IDs and are not matched across shots")

    if not calibration_by_shot:
        warnings.append("Stage 5-6 skipped: provide calibration points for each main_wide shot")
    elif not homography_by_frame:
        warnings.append("Stage 5-6 skipped: no calibration produced a usable homography")
    else:
        uncalibrated_wide = [
            shot.shot_id for shot in shot_segments
            if shot.shot_type == ShotType.MAIN_WIDE and shot.shot_id not in calibration_by_shot
        ]
        if uncalibrated_wide:
            warnings.append(f"No pitch calibration supplied for main_wide shots: {uncalibrated_wide}")

    unassigned_wide = [
        tracklet.track_id for tracklet in tracklets
        if shots_by_id.get(tracklet.shot_id)
        and shots_by_id[tracklet.shot_id].shot_type == ShotType.MAIN_WIDE
        and tracklet.team in (Team.TEAM_A, Team.TEAM_B)
        and tracklet.track_id not in track_to_master
    ]
    if unassigned_wide:
        warnings.append(
            f"{len(unassigned_wide)} main_wide tracklets have no gallery identity "
            "(for example, the per-team gallery limit may have been reached)"
        )

    for tracklet in tracklets:
        master_id = track_to_master.get(tracklet.track_id)
        identity = identities_by_id.get(master_id) if master_id is not None else None
        if identity is None:
            continue
        shot = shots_by_id.get(tracklet.shot_id)
        if shot is None or shot.shot_type != ShotType.MAIN_WIDE:
            continue
        for detection in tracklet.detections:
            matrix = homography_by_frame.get(detection.frame_idx)
            if matrix is None:
                continue
            point = image_point_to_pitch((detection.x1 + detection.x2) / 2, detection.y2, matrix)
            if abs(point.x_m) <= config.pitch_mapping.pitch_length_m / 2 and abs(point.y_m) <= config.pitch_mapping.pitch_width_m / 2:
                identity.trajectory[detection.frame_idx] = PitchPoint(point.x_m, point.y_m)

    ball_pitch: list[BallTrajectoryPoint] = []
    if homography_by_frame:
        for pixel_point in ball_trajectory_px:
            shot = next((s for s in shot_segments if s.start_frame <= pixel_point.frame_idx < s.end_frame), None)
            if shot is None or shot.shot_type != ShotType.MAIN_WIDE or pixel_point.x_m is None:
                continue
            matrix = homography_by_frame.get(pixel_point.frame_idx)
            if matrix is None:
                continue
            point = image_point_to_pitch(pixel_point.x_m, pixel_point.y_m, matrix)
            if abs(point.x_m) <= config.pitch_mapping.pitch_length_m / 2 and abs(point.y_m) <= config.pitch_mapping.pitch_width_m / 2:
                ball_pitch.append(BallTrajectoryPoint(pixel_point.frame_idx, point.x_m, point.y_m, pixel_point.interpolated))

    events: list[Event] = []
    if ball_pitch and identities_by_id:
        identities = list(identities_by_id.values())
        for shot in shot_segments:
            if shot.shot_type != ShotType.MAIN_WIDE:
                continue
            shot_ball = [p for p in ball_pitch if shot.start_frame <= p.frame_idx < shot.end_frame]
            if not shot_ball:
                continue
            possession = detect_possession(shot_ball, identities, config.event_detection)
            passes = detect_passes(possession, identities)
            pass_intervals = [
                (event.metadata.get("start_frame", event.frame_idx), event.frame_idx)
                for event in passes
            ]
            shots = detect_shots(
                shot_ball,
                ((config.pitch_mapping.pitch_length_m / 2, 0.0), (-config.pitch_mapping.pitch_length_m / 2, 0.0)),
                identities=identities,
                pass_intervals=pass_intervals,
            )
            events.extend(possession)
            events.extend(passes)
            events.extend(shots)
        events.sort(key=lambda event: event.frame_idx)

    return list(identities_by_id.values()), ball_pitch, events


def run_pipeline(
    video_path: str,
    config: PipelineConfig = DEFAULT_CONFIG,
    calibration_by_shot: dict[int, tuple[int, list[tuple[float, float]], list[tuple[float, float]]]] | None = None,
    home_kit_color_bgr: np.ndarray | None = None,
    away_kit_color_bgr: np.ndarray | None = None,
    home_gk_color_bgr: np.ndarray | None = None,
    away_gk_color_bgr: np.ndarray | None = None,
) -> PipelineResult:
    """Run Stages 1-4 and, with per-shot calibration, Stages 5-6.

    Kit colors are BGR triples. If either outfield color is missing, team
    clusters remain anonymous A/B labels and ``team_sides_confirmed`` is false.
    Calibration values are ``(frame_idx, image_points, pitch_points)`` keyed by
    the Stage 2.5 shot ID; pitch points are in metres around pitch center.
    """
    from ai_engine.stage3_team_reid.team_classifier import classify_teams_with_fallback
    from ai_engine.stage4_ball_tracking.ball_tracker import interpolate_gaps
    from ai_engine.utils.video_io import get_video_info

    info = get_video_info(video_path)
    warnings: list[str] = []
    if config.enable_shot_detection:
        from ai_engine.stage2_5_shot_detection.shot_detector import build_shot_segments

        shot_segments = build_shot_segments(video_path, config.shot_detection)
    else:
        warnings.append("Shot detection is disabled; the full clip is treated as one continuous camera view")
        shot_segments = [ShotSegment(0, 0, info.frame_count, ShotType.MAIN_WIDE)]

    tracker = Tracker(config.detection, config.tracking)
    raw_tracklets = tracker.track_video(video_path, shot_segments=shot_segments)
    tracklets = [
        t for t in raw_tracklets.values()
        if t.cls in (ObjectClass.PLAYER, ObjectClass.GOALKEEPER, ObjectClass.REFEREE)
        and t.duration_frames >= int(config.pitch_mapping.min_tracklet_duration_sec * (info.fps or 25.0))
    ]

    crops_by_track, color_samples = _read_tracklet_crops(video_path, tracklets)
    embeddings = _embed_tracklets(tracklets, crops_by_track, config)
    colors = {
        tid: np.median(np.asarray(samples, dtype=np.float32), axis=0)
        for tid, samples in color_samples.items() if samples
    }
    tracklets_dict = {t.track_id: t for t in tracklets}
    tracklets_dict, embeddings, colors, color_samples = _stitch_tracklets_within_shots(
        tracklets_dict, embeddings, colors, color_samples, config
    )
    tracklets = list(tracklets_dict.values())

    classes = {t.track_id: t.cls.value if t.cls else None for t in tracklets}
    team_by_track = classify_teams_with_fallback(
        colors,
        home_kit_color_bgr,
        away_kit_color_bgr,
        config.team_reid,
        home_gk_bgr=home_gk_color_bgr,
        away_gk_bgr=away_gk_color_bgr,
        cls_by_track=classes,
        color_samples_by_track=color_samples,
    )
    for tracklet in tracklets:
        tracklet.team = team_by_track.get(tracklet.track_id, Team.UNKNOWN)
        if tracklet.track_id in embeddings:
            tracklet.reid_embedding = embeddings[tracklet.track_id].tolist()

    if home_kit_color_bgr is None or away_kit_color_bgr is None:
        warnings.append("Home/Away kit colors were not both provided; TEAM_A/TEAM_B are anonymous clusters")
    if config.enable_ocr and config.team_reid.ocr_enabled:
        from ai_engine.stage3_team_reid.jersey_ocr import run_jersey_ocr_for_tracklets

        try:
            run_jersey_ocr_for_tracklets(tracklets, video_path, config.team_reid)
        except Exception as exc:
            logger.exception("Optional jersey OCR failed")
            warnings.append(f"Jersey OCR failed and was skipped: {exc}")

    pixel_ball: list[BallTrajectoryPoint] = []
    for shot in shot_segments:
        shot_ball_by_frame = {
            frame_idx: detection
            for frame_idx, detection in tracker.ball_by_frame.items()
            if shot.start_frame <= frame_idx < shot.end_frame
        }
        if shot_ball_by_frame:
            pixel_ball.extend(
                interpolate_gaps(
                    shot_ball_by_frame,
                    config.ball_tracking,
                    info.width or None,
                    info.height or None,
                )
            )
    pixel_ball.sort(key=lambda point: point.frame_idx)

    identities, ball_pitch, events = _build_identities(
        video_path, pixel_ball, tracklets, shot_segments,
        config, calibration_by_shot, warnings,
    )

    return PipelineResult(
        identities=identities,
        events=events,
        shot_segments=shot_segments,
        tracklets=tracklets,
        ball_trajectory_px=pixel_ball,
        ball_trajectory_pitch=ball_pitch,
        warnings=warnings,
        team_sides_confirmed=(home_kit_color_bgr is not None and away_kit_color_bgr is not None),
    )


def _parse_hex_color(value: str | None) -> np.ndarray | None:
    if not value:
        return None
    color = value.strip().lstrip("#")
    if len(color) != 6:
        raise argparse.ArgumentTypeError("kit colors must be six-digit hex, such as #1a2b3c")
    try:
        red, green, blue = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid hex kit color: {value}") from exc
    return np.asarray((blue, green, red), dtype=np.float32)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run football video analytics stages on a clip")
    parser.add_argument("video_path")
    parser.add_argument("--detect-shots", action="store_true", help="Detect cuts and reset tracking at each shot")
    parser.add_argument(
        "--reid",
        action="store_true",
        help="Enable appearance embeddings; cross-shot matching also requires --detect-shots",
    )
    parser.add_argument(
        "--calibration-json",
        help="JSON mapping shot IDs to {frame_idx, image_points, pitch_points}; needed for pitch outputs/events",
    )
    parser.add_argument("--home-kit", type=_parse_hex_color, help="Home outfield kit color in hex, e.g. #123456")
    parser.add_argument("--away-kit", type=_parse_hex_color, help="Away outfield kit color in hex, e.g. #abcdef")
    parser.add_argument("--home-gk-kit", type=_parse_hex_color, help="Home goalkeeper kit color in hex")
    parser.add_argument("--away-gk-kit", type=_parse_hex_color, help="Away goalkeeper kit color in hex")
    args = parser.parse_args()

    calibration_by_shot = None
    if args.calibration_json:
        with open(args.calibration_json, encoding="utf-8") as source:
            raw = json.load(source)
        calibration_by_shot = {
            int(shot_id): (
                int(calibration["frame_idx"]),
                [tuple(map(float, point)) for point in calibration["image_points"]],
                [tuple(map(float, point)) for point in calibration["pitch_points"]],
            )
            for shot_id, calibration in raw.items()
        }

    config = replace(
        DEFAULT_CONFIG,
        enable_shot_detection=args.detect_shots,
        enable_reid=args.reid,
    )
    result = run_pipeline(
        args.video_path,
        config=config,
        calibration_by_shot=calibration_by_shot,
        home_kit_color_bgr=args.home_kit,
        away_kit_color_bgr=args.away_kit,
        home_gk_color_bgr=args.home_gk_kit,
        away_gk_color_bgr=args.away_gk_kit,
    )
    print(f"Tracklets: {len(result.tracklets)}")
    print(f"Identities: {len(result.identities)}")
    print(f"Ball trajectory points (pixels): {len(result.ball_trajectory_px)}")
    print(f"Pitch trajectory points: {len(result.ball_trajectory_pitch)}")
    print(f"Events: {len(result.events)}")
    print(f"Shot segments: {len(result.shot_segments)}")
    for warning in result.warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    _cli()
