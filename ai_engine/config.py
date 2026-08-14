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
    model_path: Path = DETECTION_MODEL_PATH
    imgsz: int = 640
    conf_thresh: float = 0.35
    iou_thresh: float = 0.5
    device: str = "0"          # "0" for first CUDA GPU, "cpu" as fallback
    target_fps: int = 12       # downsample target, per Stage 1 plan (10-15fps)


# ---------------------------------------------------------------------------
# Stage 2: Tracking (BoT-SORT)
# ---------------------------------------------------------------------------

@dataclass
class TrackingConfig:
    tracker_yaml: Path = AI_ENGINE_ROOT / "stage2_tracking" / "botsort.yaml"
    track_buffer: int = 90
    match_thresh: float = 0.5
    new_track_thresh: float = 0.6
    gmc_method: str = "sparseOptFlow"


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
# Stage 3: Team classification + Re-ID
# ---------------------------------------------------------------------------

@dataclass
class TeamReidConfig:
    n_teams: int = 2  # + referee, handled separately
    kmeans_clusters: int = 2
    reid_model_name: str = "osnet_x0_25"  # torchreid pretrained
    reid_embedding_dim: int = 512
    reid_similarity_threshold: float = 0.6  # below this -> not a confident match
    ocr_enabled: bool = True
    ocr_min_confidence: float = 0.5


# ---------------------------------------------------------------------------
# Stage 4: Ball tracking
# ---------------------------------------------------------------------------

@dataclass
class BallTrackingConfig:
    interpolation_max_gap_frames: int = 15  # per Stage 4 plan (5-15 frames)
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
    min_tracklet_duration_sec: float = 1.5  # filter transient noise tracklets
    homography_ransac_thresh: float = 5.0


# ---------------------------------------------------------------------------
# Stage 6: Event detection
# ---------------------------------------------------------------------------

@dataclass
class EventDetectionConfig:
    possession_radius_m: float = 2.0
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
    enable_ocr: bool = False


DEFAULT_CONFIG = PipelineConfig()
