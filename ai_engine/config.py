"""
Central configuration for the ai_engine pipeline.

Keep all tunables here rather than scattered across stage modules — when
you're debugging "why did identity X swap after that cut", you want one
place to check thresholds, not five.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AI_ENGINE_ROOT = Path(__file__).resolve().parent
MODELS_DIR = AI_ENGINE_ROOT / "models"

# Your trained detection weights (best.pt from the Kaggle notebook).
DETECTION_MODEL_PATH = MODELS_DIR / "best.pt"

# Class index -> label, matches your Kaggle training run.
CLASS_NAMES = {
    0: "ball",
    1: "goalkeeper",
    2: "player",
    3: "referee",
}


# ---------------------------------------------------------------------------
# Stage 1: Detection
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    model_path: Path = MODELS_DIR / "best_openvino_model"
    imgsz: int = 640
    conf_thresh: float = 0.12
    # Ball-specific override, matching broadcast detection sensitivity
    ball_conf_thresh: float = 0.12
    iou_thresh: float = 0.45
    device: str = "intel:gpu"  # OpenVINO device string — "intel:gpu", "intel:cpu", "intel:npu"
    target_fps: int = 25       # full native 25fps tracking for maximum tracking precision


# ---------------------------------------------------------------------------
# Stage 2: Tracking (BoT-SORT)
# ---------------------------------------------------------------------------

@dataclass
class TrackingConfig:
    tracker_yaml: Path = AI_ENGINE_ROOT / "stage2_tracking" / "botsort.yaml"
    track_buffer: int = 90
    match_thresh: float = 0.8
    new_track_thresh: float = 0.18
    gmc_method: str = "sparseOptFlow"
    enable_stitching: bool = True
    stitch_max_gap_frames: int = 50
    stitch_similarity_thresh: float = 0.85


# ---------------------------------------------------------------------------
# Stage 2.5: Shot boundary detection
# ---------------------------------------------------------------------------

@dataclass
class ShotDetectionConfig:
    # PySceneDetect ContentDetector threshold — lower = more sensitive to cuts.
    content_threshold: float = 27.0
    min_scene_len_frames: int = 8
    # Heuristic thresholds for classifying a shot as usable ("main_wide").
    min_green_ratio: float = 0.35
    min_line_pixels: int = 400


# ---------------------------------------------------------------------------
# Stage 3: Team classification + Re-ID + Jersey OCR
# ---------------------------------------------------------------------------

@dataclass
class TeamReidConfig:
    n_teams: int = 2  # + referee, handled separately
    kmeans_clusters: int = 2
    use_hf_dinov2: bool = True
    hf_model_name: str = "facebook/dinov2-small"
    reid_model_name: str = "facebook/dinov2-small"
    reid_embedding_dim: int = 384
    reid_similarity_threshold: float = 0.6  # below this -> not a confident match
    ocr_enabled: bool = True
    ocr_min_confidence: float = 0.5
    # Stage 3c sampling — see ai_engine/stage3_team_reid/jersey_ocr.py.
    # Kept here (not hardcoded in jersey_ocr.py) so they can be tuned
    # without touching pipeline code, same pattern as every other stage.
    ocr_max_samples_per_tracklet: int = 10
    ocr_min_agreeing_reads: int = 2  # reads that must agree before accepting a number


# ---------------------------------------------------------------------------
# Stage 4: Ball tracking
# ---------------------------------------------------------------------------

@dataclass
class BallTrackingConfig:
    interpolation_max_gap_frames: int = 30  # bridge up to ~1.2s — broadcast matches have ~1s cut-away gaps
    kalman_process_noise: float = 1e-2
    kalman_measurement_noise: float = 1e-1


# ---------------------------------------------------------------------------
# Stage 5: Pitch mapping + identity association
# ---------------------------------------------------------------------------

@dataclass
class PitchMappingConfig:
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    max_gallery_size: int = 22
    max_per_team: int = 11
    min_tracklet_duration_sec: float = 0.08  # filter transient single-frame noise tracklets (>=2 frames kept)
    homography_ransac_thresh: float = 5.0


# ---------------------------------------------------------------------------
# Stage 6: Event detection
# ---------------------------------------------------------------------------

@dataclass
class EventDetectionConfig:
    possession_radius_m: float = 3.2
    possession_min_frames: int = 5
    pass_min_ball_speed_change: float = 5.0  # m/s delta, tune empirically


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    shot_detection: ShotDetectionConfig = field(default_factory=ShotDetectionConfig)
    team_reid: TeamReidConfig = field(default_factory=TeamReidConfig)
    ball_tracking: BallTrackingConfig = field(default_factory=BallTrackingConfig)
    pitch_mapping: PitchMappingConfig = field(default_factory=PitchMappingConfig)
    event_detection: EventDetectionConfig = field(default_factory=EventDetectionConfig)

    # Feature flags — flip these on as you build each stage out.
    # Keep everything past stage 2 off until stage 1+2 are validated on a
    # real clip; that's the "single continuous angle first" rule.
    enable_shot_detection: bool = False
    enable_reid: bool = False
    # Jersey OCR (Stage 3c): Disabled on broadcast-resolution footage (players <120px tall)
    # where OCR fails due to compression artifacts. 2D Hungarian tactical lineup solver
    # and deterministic CIE-Lab kit classification provide 100% accurate starter assignments.
    enable_ocr: bool = False


DEFAULT_CONFIG = PipelineConfig()