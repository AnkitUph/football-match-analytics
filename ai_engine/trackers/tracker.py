"""
Detection + tracking using your custom-trained YOLOv8 model
(ai_engine/models/best.pt), trained on the Roboflow football dataset
with 4 classes: ball, goalkeeper, player, referee.

Key difference from earlier iterations of this pipeline (which used
generic pretrained COCO weights): class names are read DYNAMICALLY from
the model itself (model.names) rather than hardcoded indices. This
matters because:

  1. Class index ordering can vary between training runs/dataset
     exports - hardcoding indices would silently break if you ever
     retrain.
  2. Most importantly: referees are now their own detected class. We no
     longer need pitch-boundary filtering or color-based heuristics to
     exclude them from player tracking - the model was trained to tell
     them apart directly. This solves the biggest recurring problem from
     earlier attempts at this pipeline, at its actual source.

Players, goalkeepers, and referees all go through ONE shared ByteTrack
instance (for a consistent track_id space across a whole clip), then get
split into separate output categories based on the class each detection
was tagged with. The ball is handled separately - not an identity
problem (there's only one), just a gap-filling problem, same approach
proven out earlier in this project (best detection per frame +
interpolation across short gaps).
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import supervision as sv
from ultralytics import YOLO

logger = logging.getLogger(__name__)

DEFAULT_IMGSZ = 1280  # keeps detail on small/distant players - verify still optimal for your specific trained model
DEFAULT_CONFIDENCE = 0.3
MAX_BALL_GAP_SECONDS = 1.5


class Tracker:
    def __init__(self, model_path, imgsz=DEFAULT_IMGSZ, confidence=DEFAULT_CONFIDENCE):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.confidence = confidence
        self.class_names = self.model.names  # e.g. {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
        self.class_name_to_id = {name: idx for idx, name in self.class_names.items()}

        logger.info("Loaded model with classes: %s", self.class_names)
        for required in ("ball", "player", "referee"):
            if required not in self.class_name_to_id:
                logger.warning(
                    "Expected class '%s' not found in model.names (%s) - "
                    "check this is really the football-trained model, not "
                    "a generic/COCO one.", required, self.class_names
                )

    def detect_frames(self, frames, batch_size=20):
        detections = []
        total_batches = (len(frames) + batch_size - 1) // batch_size
        for batch_num, i in enumerate(range(0, len(frames), batch_size), start=1):
            logger.info("Detecting batch %d/%d (frames %d-%d of %d)...",
                        batch_num, total_batches, i, min(i + batch_size, len(frames)), len(frames))
            batch = self.model.predict(
                frames[i:i + batch_size],
                imgsz=self.imgsz,
                conf=self.confidence,
                verbose=False,
            )
            detections += batch
        return detections

    def get_object_tracks(self, frames, fps=25, read_from_stub=False, stub_path=None):
        """
        Returns a dict with keys "players", "goalkeepers", "referees",
        "ball" - each a list (one entry per frame) of {track_id: {...}}
        dicts, matching the structure expected by team_assigner.py etc.
        """
        if read_from_stub and stub_path is not None and Path(stub_path).exists():
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        # IMPORTANT: minimum_matching_threshold in supervision's ByteTrack
        # behaves INVERSE to normal IoU-threshold intuition - this is a
        # documented quirk (see roboflow/supervision issue #1670), not
        # something we got wrong by guessing. Community reports (e.g.
        # roboflow/supervision discussion #1001) confirm empirically that
        # RAISING this value toward ~0.9-0.95 stabilizes tracking - even
        # through player collisions - while lowering it causes MORE
        # fragmentation, not less. We tried lowering it first based on
        # normal IoU intuition and it made things dramatically worse
        # (175 -> 552 unique players), which is consistent with this
        # being the actual (if unintuitive) documented behavior.
        tracker = sv.ByteTrack(
            frame_rate=int(fps),
            lost_track_buffer=int(fps * 4),
            track_activation_threshold=self.confidence,
            minimum_matching_threshold=0.95,
        )

        ball_class_id = self.class_name_to_id.get("ball")
        goalkeeper_class_id = self.class_name_to_id.get("goalkeeper")
        player_class_id = self.class_name_to_id.get("player")
        referee_class_id = self.class_name_to_id.get("referee")

        tracks = {"players": [], "goalkeepers": [], "referees": [], "ball": []}
        raw_ball_points = {}  # frame_index -> (x, y, confidence), for interpolation afterward

        # Collected first, categorized after the full pass - see the
        # majority-vote step below for why.
        raw_person_frames = []  # list of {track_id: (class_id, bbox, confidence)} per frame

        for frame_num, detection in enumerate(detections):
            sv_detections = sv.Detections.from_ultralytics(detection)

            if ball_class_id is not None:
                ball_mask = sv_detections.class_id == ball_class_id
                ball_detections = sv_detections[ball_mask]
                if len(ball_detections) > 0:
                    best_idx = int(np.argmax(ball_detections.confidence))
                    x1, y1, x2, y2 = ball_detections.xyxy[best_idx]
                    conf = float(ball_detections.confidence[best_idx])
                    raw_ball_points[frame_num] = ((x1 + x2) / 2, (y1 + y2) / 2, conf)
                person_mask = ~ball_mask
                person_detections = sv_detections[person_mask]
            else:
                person_detections = sv_detections

            tracked = tracker.update_with_detections(person_detections)

            frame_entries = {}
            for i in range(len(tracked)):
                track_id = int(tracked.tracker_id[i])
                class_id = int(tracked.class_id[i])
                bbox = tracked.xyxy[i].tolist()
                conf = float(tracked.confidence[i])
                frame_entries[track_id] = (class_id, bbox, conf)

            raw_person_frames.append(frame_entries)
            tracks["players"].append({})
            tracks["goalkeepers"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

        # ---- Majority-vote category per track, THEN bucket every frame ----
        # A single physical person shouldn't be able to flip between
        # "player" and "referee" from one frame to the next just because
        # the model's classification wavered momentarily - that's noise,
        # not a real category change. Deciding each track's category once
        # (by whichever class it was predicted as most often across its
        # whole lifetime) and applying that consistently avoids inflating
        # per-category track counts with what's actually the same
        # continuously-tracked person.
        track_class_votes = {}
        for frame_entries in raw_person_frames:
            for track_id, (class_id, _, _) in frame_entries.items():
                track_class_votes.setdefault(track_id, {}).setdefault(class_id, 0)
                track_class_votes[track_id][class_id] += 1

        track_category = {}
        for track_id, votes in track_class_votes.items():
            majority_class_id = max(votes, key=votes.get)
            if majority_class_id == goalkeeper_class_id:
                track_category[track_id] = "goalkeepers"
            elif majority_class_id == referee_class_id:
                track_category[track_id] = "referees"
            elif majority_class_id == player_class_id:
                track_category[track_id] = "players"
            else:
                track_category[track_id] = None  # unexpected class, drop it

        for frame_num, frame_entries in enumerate(raw_person_frames):
            for track_id, (class_id, bbox, conf) in frame_entries.items():
                category = track_category.get(track_id)
                if category is not None:
                    tracks[category][frame_num][track_id] = {"bbox": bbox, "confidence": conf}

        tracks["ball"] = self._ball_dicts_from_interpolated_points(
            raw_ball_points, len(frames), fps
        )

        if stub_path is not None:
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def _ball_dicts_from_interpolated_points(self, raw_ball_points, frame_count, fps):
        """
        Converts raw per-frame ball detections into the same
        {frame: {1: {"bbox": [...], "is_interpolated": bool}}} shape as
        the other categories (ball always uses track_id=1, since there's
        only ever one), filling short gaps with linear interpolation.
        """
        max_gap_frames = int(MAX_BALL_GAP_SECONDS * fps)
        ball_track = [{} for _ in range(frame_count)]

        if not raw_ball_points:
            return ball_track

        known_frames = sorted(raw_ball_points.keys())

        def make_entry(x, y, conf, is_interp):
            half_box = 8  # ball boxes are small; synthesize a small bbox around the point for downstream code expecting one
            return {
                "bbox": [x - half_box, y - half_box, x + half_box, y + half_box],
                "confidence": conf,
                "is_interpolated": is_interp,
            }

        for i in range(len(known_frames) - 1):
            f1, f2 = known_frames[i], known_frames[i + 1]
            x1, y1, conf1 = raw_ball_points[f1]
            x2, y2, _ = raw_ball_points[f2]

            ball_track[f1][1] = make_entry(x1, y1, conf1, False)

            gap = f2 - f1
            if 1 < gap <= max_gap_frames:
                for step in range(1, gap):
                    t = step / gap
                    x = x1 + (x2 - x1) * t
                    y = y1 + (y2 - y1) * t
                    ball_track[f1 + step][1] = make_entry(x, y, None, True)

        last_frame = known_frames[-1]
        x, y, conf = raw_ball_points[last_frame]
        ball_track[last_frame][1] = make_entry(x, y, conf, False)

        return ball_track

    def add_position_to_tracks(self, tracks):
        """Adds a 'position' (foot position for people, center for the ball) to every tracked box."""
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object_type == "ball":
                        position = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    else:
                        position = ((bbox[0] + bbox[2]) / 2, bbox[3])
                    tracks[object_type][frame_num][track_id]["position"] = position