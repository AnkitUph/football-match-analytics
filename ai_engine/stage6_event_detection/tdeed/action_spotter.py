import os
import sys
from pathlib import Path
import torch
import numpy as np

# Ensure tdeed package directory is in sys.path for internal imports
TDEED_DIR = Path(__file__).resolve().parent
if str(TDEED_DIR) not in sys.path:
    sys.path.insert(0, str(TDEED_DIR))

from model.model import TDEEDModel
from util.io import load_json
from util.dataset import load_classes
from util.eval import inference
from dataset.frame import ActionSpotInferenceDataset
from torch.utils.data import DataLoader

# Default paths
BASE_DIR = TDEED_DIR.parent.parent.parent
DEFAULT_WEIGHTS = BASE_DIR / "ai_engine" / "models" / "tdeed_ball_action_spotter.pt"
DEFAULT_CONFIG = BASE_DIR / "ai_engine" / "models" / "tdeed_ball_action_spotter.json"


class TDEEDActionSpotter:
    """
    Production wrapper for T-DEED Ball Action Spotting model.
    Spots 12 football event classes (Pass, Drive, Header, High Pass, Out, Cross,
    Throw In, Shot, Ball-Player Block, Tackle, Free Kick, Goal).
    """

    def __init__(self, weights_path=DEFAULT_WEIGHTS, config_path=DEFAULT_CONFIG, device="cpu"):
        self.device = torch.device(device)
        self.config_path = Path(config_path)
        self.weights_path = Path(weights_path)
        
        cfg = load_json(str(self.config_path))
        class Args: pass
        self.args = Args()
        for k, v in cfg.items():
            setattr(self.args, k, v)
        if getattr(self.args, "crop_dim", 0) <= 0:
            self.args.crop_dim = None
        self.args.seed = 1
        self.args.acc_grad_iter = 1
        self.args.save_dir = "checkpoints"
        self.args.model = "SoccerNetBall_challenge1"

        # Load classes
        self.classes = load_classes(str(TDEED_DIR / "data" / self.args.dataset / "class.txt"))
        pretrain_classes = load_classes(str(TDEED_DIR / "data" / self.args.pretrain["dataset"] / "class.txt"))

        # Instantiate model
        self.model = TDEEDModel(args=self.args)
        n_classes = [len(self.classes) + 1, len(pretrain_classes) + 1]
        self.model._model.update_pred_head(n_classes)
        self.model._num_classes = sum(n_classes)

        # Load trained weights
        ckpt = torch.load(str(self.weights_path), map_location=self.device)
        self.model.load(ckpt)
        self.model._model.to(self.device)
        self.model._model.eval()
        if self.device.type == "cpu":
            torch.set_num_threads(min(8, os.cpu_count() or 4))

    def spot_events(self, video_path, threshold=0.25, frame_width=398, frame_height=224, stride=2):
        """
        Runs action spotting on a video file and returns a list of detected events:
        [{'frame': int, 'label': str, 'confidence': float}]
        """
        dataset = ActionSpotInferenceDataset(
            str(video_path),
            clip_len=self.args.clip_len,
            overlap_len=self.args.clip_len // 2,
            stride=stride,
            dataset=self.args.dataset,
            size=(frame_width, frame_height),
        )

        loader = DataLoader(
            dataset,
            batch_size=getattr(self.args, "batch_size", 1),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

        # Run inference – store results in a user-writable directory
        output_dir = Path("user_inference_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_dict = inference(self.model, loader, self.classes, threshold=threshold)

        if pred_dict and isinstance(pred_dict, dict) and "events" in pred_dict:
            predictions = []
            for event in pred_dict["events"]:
                predictions.append({
                    "frame": int(event["frame"]) * stride,
                    "label": event["label"],
                    "confidence": float(event["score"]),
                })
            return predictions

        results_path = output_dir / "results_inference.json"
        if results_path.exists():
            data = load_json(str(results_path))
            predictions = data.get("predictions", []) if isinstance(data, dict) else data
            return predictions
        return []
