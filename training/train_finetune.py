"""Fine-tune the SoccerNet Ball head on labeled soccer broadcast matches.

Inputs can be SoccerNet Ball clips, project events.json sidecars, or converted
SN-PCBAS-2026 clips prepared by scripts/prepare_pc_bas_events.py.
The pretrained ball-action head is fine-tuned; the second SoccerNet-v2 head and
the image backbone stay frozen. Output checkpoints retain the production
model's exact state-dict structure and can be selected in the app config.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
TDEED_DIR = ROOT / "ai_engine/stage6_event_detection/tdeed"
if str(TDEED_DIR) not in sys.path:
    sys.path.insert(0, str(TDEED_DIR))

from action_spotter import TDEEDActionSpotter  # noqa: E402
from util.io import load_text  # noqa: E402


class BallMatchDataset(Dataset):
    """Random temporal windows from SoccerNet's first-half 720p game videos."""

    def __init__(self, root, classes, clip_len=100, stride=2, windows_per_half=128,
                 radius=4, size=(398, 224)):
        self.root = Path(root)
        self.classes = classes
        self.clip_len = clip_len
        self.stride = stride
        self.windows_per_half = windows_per_half
        self.radius = radius
        self.size = size
        self.matches = []

        # Project-owned clips can use a small events.json beside each video:
        # {"video": "clip.mp4", "fps": 25,
        #  "events": [{"time_seconds": 12.4, "label": "PASS"}]}
        for annotation_path in self.root.rglob("events.json"):
            annotation = json.loads(annotation_path.read_text())
            video_value = annotation.get("video")
            if video_value:
                cand_path = Path(video_value)
                if cand_path.is_file():
                    video = cand_path.resolve()
                elif str(video_value).startswith("/app/"):
                    video = (ROOT / str(video_value)[5:]).resolve()
                elif (annotation_path.parent / video_value).is_file():
                    video = (annotation_path.parent / video_value).resolve()
                elif (self.root / "videos" / Path(video_value).name).is_file():
                    video = (self.root / "videos" / Path(video_value).name).resolve()
                else:
                    video = cand_path
            else:
                candidates = sorted(annotation_path.parent.glob("*.mp4")) + sorted(
                    annotation_path.parent.glob("*.mkv")
                )
                if len(candidates) != 1:
                    raise RuntimeError(f"Set 'video' in {annotation_path} when the folder has zero or multiple videos")
                video = candidates[0].resolve()
            if not video.is_file():
                raise RuntimeError(f"Video listed in {annotation_path} does not exist: {video}")
            annotation_fps = float(annotation.get("fps", 25))
            events = []
            for event in annotation.get("events", []):
                label = event.get("label", "").strip().upper().replace("_", " ")
                if label not in classes:
                    raise RuntimeError(f"Unknown Ball event class '{label}' in {annotation_path}")
                if "time_seconds" in event:
                    source_frame = float(event["time_seconds"]) * annotation_fps
                elif "frame" in event:
                    source_frame = float(event["frame"])
                else:
                    raise RuntimeError(f"Each event needs time_seconds or frame in {annotation_path}")
                events.append((round(source_frame / self.stride), classes[label]))
            capture = cv2.VideoCapture(str(video))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if frame_count:
                self.matches.append((video, frame_count, events))

        for label_path in self.root.rglob("Labels-ball.json"):
            data = json.loads(label_path.read_text())
            annotations = data.get("annotations", data.get("labels", []))
            game_dir = label_path.parent
            video = game_dir / "720p.mp4"
            if not video.is_file():
                continue
            events = []
            for event in annotations:
                game_time = event.get("gameTime", "")
                if not game_time or int(game_time.split(" - ", 1)[0]) != 1:
                    continue
                label = event.get("label", "").strip()
                if label not in classes:
                    continue
                # The checkpoint's SoccerNet Ball training split treats each
                # provided 720p.mp4 as the first-half video.
                # Labels use the 25 fps timeline; this trainer samples every
                # `stride`th source frame, matching the production inference path.
                frame = round(float(event["position"]) * 25 / (1000 * self.stride))
                events.append((frame, classes[label]))
            capture = cv2.VideoCapture(str(video))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if frame_count:
                self.matches.append((video, frame_count, events))

        if not self.matches:
            raise RuntimeError(f"No labeled SoccerNet Ball matches found under {self.root}")

    def __len__(self):
        return len(self.matches) * self.windows_per_half

    def __getitem__(self, index):
        video, frame_count, events = self.matches[index % len(self.matches)]
        sample_count = (frame_count + self.stride - 1) // self.stride
        max_start = max(0, sample_count - self.clip_len)

        # Half the windows include a labeled event, improving rare-event
        # exposure while preserving background-only examples.
        eligible = [f for f, _ in events if f <= max_start + self.clip_len - 1]
        if eligible and random.random() < 0.5:
            event_frame = random.choice(eligible)
            start = max(0, min(max_start, event_frame - random.randrange(self.clip_len)))
        else:
            start = random.randint(0, max_start) if max_start else 0

        cap = cv2.VideoCapture(str(video))
        first_raw_frame = start * self.stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_raw_frame)
        frames = []
        for _ in range(self.clip_len):
            ok, frame = cap.read()
            if not ok:
                frame = frames[-1].copy() if frames else None
                if frame is None:
                    cap.release()
                    raise RuntimeError(f"Could not decode video window: {video} at {start}")
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))
            if len(frames) < self.clip_len:
                cap.set(cv2.CAP_PROP_POS_FRAMES, first_raw_frame + len(frames) * self.stride)
        cap.release()

        target = torch.zeros(self.clip_len, dtype=torch.long)  # class 0 is background
        displacement = torch.zeros(self.clip_len, dtype=torch.float32)
        end = start + self.clip_len
        for frame, class_index in events:
            if start <= frame < end:
                local = frame - start
                if 0 <= local < self.clip_len:
                    # Match T-DEED's radius-displacement supervision: nearby
                    # support frames predict the class and point back to event.
                    for support in range(max(0, local - self.radius), min(self.clip_len, local + self.radius + 1)):
                        target[support] = class_index
                        displacement[support] = support - local
        return torch.stack(frames).float(), target, displacement


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "media/fine_tune_clips")
    parser.add_argument("--weights", type=Path, default=ROOT / "ai_engine/models/tdeed_ball_action_spotter.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "ai_engine/models/tdeed_ball_action_spotter.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "ai_engine/models/finetuned")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or xpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--start-epoch", type=int, default=1, help="First epoch number (for resuming)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--windows-per-half", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--trainable-classes",
        nargs="*",
        default=None,
        help="Only include these class names and background in the supervised loss.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    spotter = TDEEDActionSpotter(args.weights, args.config, str(device))
    labels = load_text(str(TDEED_DIR / "data/soccernetball/class.txt"))
    class_to_index = {name: index + 1 for index, name in enumerate(labels)}
    if args.trainable_classes:
        unknown = sorted(set(args.trainable_classes) - set(class_to_index))
        if unknown:
            parser.error(f"Unknown trainable classes: {', '.join(unknown)}")
        active_ids = [0] + [class_to_index[name] for name in args.trainable_classes]
    else:
        active_ids = list(range(len(labels) + 1))
    target_remap = torch.full((len(labels) + 1,), -1, dtype=torch.long, device=device)
    target_remap[active_ids] = torch.arange(len(active_ids), device=device)
    dataset = BallMatchDataset(args.data_dir, class_to_index, clip_len=spotter.args.clip_len,
                              stride=2, windows_per_half=args.windows_per_half,
                              radius=spotter.args.radi_displacement)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    net = spotter.model._model
    for parameter in net.parameters():
        parameter.requires_grad = False
    # Adapt the temporal representation and the SoccerNet Ball output head.
    net.temp_enc.requires_grad = True
    for parameter in net._temp_fine.parameters():
        parameter.requires_grad = True
    for parameter in net._pred_fine._fc1.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=args.lr)
    # Background is frequent; moderate class weighting helps retain rare events.
    class_weights = torch.ones(len(active_ids), device=device)
    class_weights[1:] = 4.0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = args.start_epoch
    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        net.train()
        total_loss = 0.0
        n_steps = len(loader)
        for step, (frames, target, displacement) in enumerate(loader, 1):
            frames = frames.to(device)
            target = target.to(device)
            displacement = displacement.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = net(frames)
            logits = prediction["im_feat"] if isinstance(prediction, dict) else prediction
            logits = logits.index_select(2, torch.tensor(active_ids, device=device))
            remapped_target = target_remap[target]
            if torch.any(remapped_target < 0):
                raise RuntimeError("A training event belongs to a class excluded from --trainable-classes")
            loss = F.cross_entropy(logits.reshape(-1, len(active_ids)), remapped_target.reshape(-1),
                                   weight=class_weights)
            if isinstance(prediction, dict):
                loss = loss + F.mse_loss(prediction["displ_feat"], displacement)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in net.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            step_loss = float(loss.detach())
            total_loss += step_loss
            if step % 10 == 0 or step == n_steps:
                running_avg = total_loss / step
                print(f"[Epoch {epoch}/{end_epoch}] Step {step}/{n_steps}: loss={step_loss:.4f} (running avg: {running_avg:.4f})", flush=True)
        mean_loss = total_loss / max(1, len(loader))
        output = args.out_dir / f"tdeed_ball_finetuned_epoch{epoch}.pt"
        torch.save(spotter.model.state_dict(), output)
        print(f"Epoch {epoch}/{end_epoch}: loss={mean_loss:.4f}; saved {output}", flush=True)


if __name__ == "__main__":
    main()
