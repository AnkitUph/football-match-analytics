import logging
import pickle
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

logger = logging.getLogger(__name__)

DEFAULT_IMGSZ = 1280
DEFAULT_CONFIDENCE = 0.25

PLAYER_CONFIDENCE = 0.30
GOALKEEPER_CONFIDENCE = 0.48
REFEREE_CONFIDENCE = 0.42
BALL_CONFIDENCE = 0.25

LOST_TRACK_BUFFER_SECONDS = 2.5

PLAYER_MATCH_THRESHOLD = 0.85
GOALKEEPER_MATCH_THRESHOLD = 0.80
REFEREE_MATCH_THRESHOLD = 0.80

MIN_PLAYER_TRACK_FRAMES = 35
MIN_GOALKEEPER_TRACK_FRAMES = 20
MIN_REFEREE_TRACK_FRAMES = 20

MAX_PLAYER_TRACKS = 22
MAX_PLAYERS_PER_TEAM = 11
MAX_GOALKEEPER_TRACKS = 2
MAX_REFEREE_TRACKS = 3

MAX_FRAGMENT_GAP_FRAMES = 50
MAX_BALL_GAP_SECONDS = 1.5


class Tracker:

    def __init__(
        self,
        model_path,
        imgsz=DEFAULT_IMGSZ,
        confidence=DEFAULT_CONFIDENCE,
    ):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.confidence = confidence

        self.class_names = self.model.names

        self.class_name_to_id = {
            name: idx
            for idx, name in self.class_names.items()
        }

        logger.info(
            "Loaded model with classes: %s",
            self.class_names,
        )

        logger.info(
            "Model path: %s",
            model_path,
        )

        logger.info(
            "Inference image size: %d",
            self.imgsz,
        )

        logger.info(
            "YOLO confidence threshold: %.2f",
            self.confidence,
        )

        logger.info(
            "Player confidence threshold: %.2f",
            PLAYER_CONFIDENCE,
        )

        logger.info(
            "Goalkeeper confidence threshold: %.2f",
            GOALKEEPER_CONFIDENCE,
        )

        logger.info(
            "Referee confidence threshold: %.2f",
            REFEREE_CONFIDENCE,
        )

        logger.info(
            "Ball confidence threshold: %.2f",
            BALL_CONFIDENCE,
        )

        for required in (
            "ball",
            "goalkeeper",
            "player",
            "referee",
        ):
            if required not in self.class_name_to_id:
                logger.warning(
                    "Expected class '%s' not found in model.names=%s",
                    required,
                    self.class_names,
                )

    def detect_frames(
        self,
        frames,
        batch_size=20,
    ):
        detections = []

        if not frames:
            return detections

        total_batches = (
            len(frames) + batch_size - 1
        ) // batch_size

        for batch_num, i in enumerate(
            range(0, len(frames), batch_size),
            start=1,
        ):
            logger.info(
                "Detecting batch %d/%d (frames %d-%d of %d)...",
                batch_num,
                total_batches,
                i,
                min(i + batch_size, len(frames)),
                len(frames),
            )

            batch = self.model.predict(
                frames[i:i + batch_size],
                imgsz=self.imgsz,
                conf=self.confidence,
                verbose=False,
            )

            detections.extend(batch)

        return detections

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        intersection = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(
            0.0,
            ay2 - ay1,
        )

        area_b = max(0.0, bx2 - bx1) * max(
            0.0,
            by2 - by1,
        )

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _appearance_signature(
        self,
        frame,
        bbox,
    ):
        h, w = frame.shape[:2]

        x1, y1, x2, y2 = [
            int(round(v))
            for v in bbox
        ]

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return (
                0.0,
                0.0,
                0.0,
                0.0,
            )

        crop = frame[
            y1:y1 + max(1, int((y2 - y1) * 0.68)),
            x1 + int((x2 - x1) * 0.15):
            x2 - int((x2 - x1) * 0.15),
        ]

        if crop.size == 0:
            return (
                0.0,
                0.0,
                0.0,
                0.0,
            )

        crop = cv2.resize(
            crop,
            (32, 32),
            interpolation=cv2.INTER_AREA,
        )

        hsv = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2HSV,
        )

        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        valid = v_channel > 35

        if not np.any(valid):
            return (
                0.0,
                0.0,
                0.0,
                0.0,
            )

        green = (
            (h_channel >= 35)
            & (h_channel <= 90)
            & (s_channel >= 70)
            & (v_channel >= 60)
            & valid
        )

        white = (
            (s_channel <= 75)
            & (v_channel >= 135)
            & valid
        )

        orange = (
            (h_channel <= 25)
            & (s_channel >= 90)
            & (v_channel >= 80)
            & valid
        )

        black = (
            (v_channel <= 75)
            & valid
        )

        total = float(
            np.sum(valid)
        )

        return (
            float(np.sum(green)) / total,
            float(np.sum(white)) / total,
            float(np.sum(orange)) / total,
            float(np.sum(black)) / total,
        )

    @staticmethod
    def _appearance_label(signature):
        green, white, orange, black = signature

        if orange >= 0.12:
            return "orange"

        if green >= 0.18 and green >= white * 1.15:
            return "green"

        if white >= 0.20 and white >= green * 1.15:
            return "white"

        if black >= 0.45:
            return "black"

        return "unknown"

    @staticmethod
    def _appearance_distance(
        signature_a,
        signature_b,
    ):
        a = np.asarray(
            signature_a,
            dtype=np.float32,
        )

        b = np.asarray(
            signature_b,
            dtype=np.float32,
        )

        return float(
            np.linalg.norm(a - b)
        )

    def _resolve_person_overlaps(
        self,
        detections,
    ):
        if len(detections) == 0:
            return detections

        class_ids = detections.class_id
        confidences = detections.confidence
        boxes = detections.xyxy

        player_id = self.class_name_to_id.get(
            "player"
        )

        goalkeeper_id = self.class_name_to_id.get(
            "goalkeeper"
        )

        referee_id = self.class_name_to_id.get(
            "referee"
        )

        keep = np.ones(
            len(detections),
            dtype=bool,
        )

        player_indices = []

        goalkeeper_indices = []

        referee_indices = []

        for i, class_id in enumerate(
            class_ids
        ):
            if class_id == player_id:
                player_indices.append(i)

            elif class_id == goalkeeper_id:
                goalkeeper_indices.append(i)

            elif class_id == referee_id:
                referee_indices.append(i)

        for gi in goalkeeper_indices:
            g_conf = float(
                confidences[gi]
            )

            for pi in player_indices:
                if not keep[pi]:
                    continue

                overlap = self._iou(
                    boxes[gi],
                    boxes[pi],
                )

                p_conf = float(
                    confidences[pi]
                )

                if (
                    overlap >= 0.40
                    and g_conf >= p_conf - 0.05
                ):
                    keep[pi] = False

        for ri in referee_indices:
            r_conf = float(
                confidences[ri]
            )

            for pi in player_indices:
                if not keep[pi]:
                    continue

                overlap = self._iou(
                    boxes[ri],
                    boxes[pi],
                )

                p_conf = float(
                    confidences[pi]
                )

                if (
                    overlap >= 0.40
                    and r_conf >= p_conf - 0.03
                ):
                    keep[pi] = False

        return detections[keep]

    def _filter_class(
        self,
        detections,
        class_id,
        threshold,
    ):
        if (
            class_id is None
            or len(detections) == 0
        ):
            return detections[:0]

        mask = (
            detections.class_id == class_id
        )

        mask &= (
            detections.confidence
            >= threshold
        )

        return detections[mask]

    @staticmethod
    def _new_tracker(
        fps,
        lost_seconds,
        activation_threshold,
        matching_threshold,
    ):
        return sv.ByteTrack(
            frame_rate=max(
                int(round(fps)),
                1,
            ),
            lost_track_buffer=max(
                int(
                    fps
                    * lost_seconds
                ),
                1,
            ),
            track_activation_threshold=(
                activation_threshold
            ),
            minimum_matching_threshold=(
                matching_threshold
            ),
        )

    def _tracked_entries(
        self,
        tracked,
        frame,
    ):
        entries = {}

        if (
            tracked.tracker_id is None
            or len(tracked) == 0
        ):
            return entries

        for i in range(
            len(tracked)
        ):
            if (
                tracked.tracker_id[i]
                is None
            ):
                continue

            if (
                tracked.xyxy is None
                or tracked.confidence is None
            ):
                continue

            track_id = int(
                tracked.tracker_id[i]
            )

            bbox = (
                tracked.xyxy[i]
                .astype(float)
                .tolist()
            )

            confidence = float(
                tracked.confidence[i]
            )

            signature = (
                self._appearance_signature(
                    frame,
                    bbox,
                )
            )

            label = (
                self._appearance_label(
                    signature
                )
            )

            entries[track_id] = (
                bbox,
                confidence,
                label,
                signature,
            )

        return entries

    def _build_track_stats(
        self,
        raw_frames,
    ):
        stats = {}

        for frame_num, frame_entries in enumerate(
            raw_frames
        ):
            for (
                track_id,
                (
                    bbox,
                    confidence,
                    label,
                    signature,
                ),
            ) in frame_entries.items():

                x1, y1, x2, y2 = bbox

                position = (
                    (x1 + x2) / 2.0,
                    y2,
                )

                height = max(
                    1.0,
                    y2 - y1,
                )

                if track_id not in stats:
                    stats[track_id] = {
                        "frames": [],
                        "positions": [],
                        "heights": [],
                        "confidences": [],
                        "labels": [],
                        "signatures": [],
                    }

                stats[
                    track_id
                ]["frames"].append(
                    frame_num
                )

                stats[
                    track_id
                ]["positions"].append(
                    position
                )

                stats[
                    track_id
                ]["heights"].append(
                    height
                )

                stats[
                    track_id
                ]["confidences"].append(
                    confidence
                )

                stats[
                    track_id
                ]["labels"].append(
                    label
                )

                stats[
                    track_id
                ]["signatures"].append(
                    signature
                )

        return stats

    @staticmethod
    def _majority_label(
        labels,
    ):
        if not labels:
            return "unknown"

        counts = {}

        for label in labels:
            counts[label] = (
                counts.get(label, 0) + 1
            )

        return max(
            counts,
            key=counts.get,
        )

    @staticmethod
    def _label_purity(
        labels,
        label,
    ):
        if not labels:
            return 0.0

        return (
            labels.count(label)
            / len(labels)
        )

    @staticmethod
    def _average_signature(
        signatures,
    ):
        if not signatures:
            return (
                0.0,
                0.0,
                0.0,
                0.0,
            )

        return tuple(
            np.mean(
                np.asarray(
                    signatures,
                    dtype=np.float32,
                ),
                axis=0,
            ).tolist()
        )

    @staticmethod
    def _predict_position(
        track,
        target_frame,
    ):
        frames = track["frames"]
        positions = track["positions"]

        if not frames:
            return None

        last_frame = frames[-1]
        last_position = np.asarray(
            positions[-1],
            dtype=np.float32,
        )

        if (
            len(frames) < 3
            or target_frame <= last_frame
        ):
            return last_position

        start = max(
            0,
            len(frames) - 8,
        )

        velocities = []

        for i in range(
            start + 1,
            len(frames),
        ):
            df = (
                frames[i]
                - frames[i - 1]
            )

            if df <= 0:
                continue

            p1 = np.asarray(
                positions[i - 1],
                dtype=np.float32,
            )

            p2 = np.asarray(
                positions[i],
                dtype=np.float32,
            )

            velocity = (
                p2 - p1
            ) / float(df)

            velocities.append(
                velocity
            )

        if not velocities:
            return last_position

        velocity = np.median(
            np.asarray(
                velocities,
                dtype=np.float32,
            ),
            axis=0,
        )

        velocity = np.clip(
            velocity,
            -15.0,
            15.0,
        )

        gap = (
            target_frame
            - last_frame
        )

        return (
            last_position
            + velocity * gap
        )

    def _merge_track_fragments(
        self,
        raw_frames,
        max_gap_frames=MAX_FRAGMENT_GAP_FRAMES,
    ):
        stats = self._build_track_stats(
            raw_frames
        )

        track_ids = list(
            stats.keys()
        )

        parent = {
            track_id: track_id
            for track_id in track_ids
        }

        def find(x):
            while parent[x] != x:
                parent[x] = parent[
                    parent[x]
                ]
                x = parent[x]

            return x

        def union(a, b):
            ra = find(a)
            rb = find(b)

            if ra == rb:
                return

            parent[rb] = ra

        ordered = sorted(
            track_ids,
            key=lambda track_id: (
                stats[track_id]["frames"][0]
            ),
        )

        for later_id in ordered:
            later = stats[later_id]

            later_first = later[
                "frames"
            ][0]

            later_position = np.asarray(
                later["positions"][0],
                dtype=np.float32,
            )

            later_height = float(
                np.median(
                    later["heights"]
                )
            )

            later_label = (
                self._majority_label(
                    later["labels"]
                )
            )

            best_candidate = None
            best_score = float("inf")

            for earlier_id in ordered:
                if earlier_id == later_id:
                    continue

                earlier = stats[
                    earlier_id
                ]

                earlier_last = earlier[
                    "frames"
                ][-1]

                if (
                    earlier_last
                    >= later_first
                ):
                    continue

                gap = (
                    later_first
                    - earlier_last
                    - 1
                )

                if (
                    gap < 0
                    or gap > max_gap_frames
                ):
                    continue

                earlier_height = float(
                    np.median(
                        earlier[
                            "heights"
                        ]
                    )
                )

                height_ratio = (
                    later_height
                    / max(
                        earlier_height,
                        1.0,
                    )
                )

                if (
                    height_ratio < 0.55
                    or height_ratio > 1.80
                ):
                    continue

                earlier_label = (
                    self._majority_label(
                        earlier[
                            "labels"
                        ]
                    )
                )

                if (
                    earlier_label
                    in (
                        "green",
                        "white",
                    )
                    and later_label
                    in (
                        "green",
                        "white",
                    )
                    and earlier_label
                    != later_label
                ):
                    continue

                predicted = (
                    self._predict_position(
                        earlier,
                        later_first,
                    )
                )

                if predicted is None:
                    continue

                distance = float(
                    np.linalg.norm(
                        predicted
                        - later_position
                    )
                )

                max_distance = max(
                    90.0,
                    70.0
                    + 7.0 * gap,
                )

                if distance > max_distance:
                    continue

                earlier_signature = (
                    self._average_signature(
                        earlier[
                            "signatures"
                        ]
                    )
                )

                later_signature = (
                    self._average_signature(
                        later[
                            "signatures"
                        ]
                    )
                )

                appearance_distance = (
                    self._appearance_distance(
                        earlier_signature,
                        later_signature,
                    )
                )

                if appearance_distance > 0.42:
                    continue

                distance_score = (
                    distance
                    / max_distance
                )

                score = (
                    distance_score
                    + appearance_distance
                )

                if score < best_score:
                    best_score = score
                    best_candidate = (
                        earlier_id
                    )

            if best_candidate is not None:
                union(
                    best_candidate,
                    later_id,
                )

        mapping = {}

        groups = {}

        for track_id in track_ids:
            root = find(track_id)

            groups.setdefault(
                root,
                [],
            ).append(track_id)

        for root, members in groups.items():
            canonical_id = min(
                members
            )

            for member in members:
                mapping[
                    member
                ] = canonical_id

        return (
            mapping,
            stats,
            groups,
        )

    def _merged_statistics(
        self,
        raw_stats,
        groups,
    ):
        merged = {}

        for root, members in groups.items():
            frames = []
            positions = []
            heights = []
            confidences = []
            labels = []
            signatures = []

            for member in members:
                stat = raw_stats[
                    member
                ]

                frames.extend(
                    stat["frames"]
                )

                positions.extend(
                    stat["positions"]
                )

                heights.extend(
                    stat["heights"]
                )

                confidences.extend(
                    stat["confidences"]
                )

                labels.extend(
                    stat["labels"]
                )

                signatures.extend(
                    stat["signatures"]
                )

            order = np.argsort(
                np.asarray(
                    frames
                )
            )

            frames = [
                frames[i]
                for i in order
            ]

            positions = [
                positions[i]
                for i in order
            ]

            merged[root] = {
                "frames": frames,
                "positions": positions,
                "heights": heights,
                "confidences": confidences,
                "labels": labels,
                "signatures": signatures,
                "average_confidence": (
                    float(
                        np.mean(
                            confidences
                        )
                    )
                    if confidences
                    else 0.0
                ),
                "label": (
                    self._majority_label(
                        labels
                    )
                ),
            }

        return merged

    def _track_quality(
        self,
        stat,
    ):
        frames = len(
            stat["frames"]
        )

        confidence = stat[
            "average_confidence"
        ]

        label = stat["label"]

        purity = self._label_purity(
            stat["labels"],
            label,
        )

        return (
            frames
            * confidence
            * (
                1.0
                + 0.35 * purity
            )
        )

    def _select_player_tracks(
        self,
        merged_stats,
    ):
        candidates = []

        for track_id, stat in merged_stats.items():
            if (
                len(stat["frames"])
                < MIN_PLAYER_TRACK_FRAMES
            ):
                continue

            label = stat["label"]

            candidates.append(
                (
                    track_id,
                    stat,
                    self._track_quality(
                        stat
                    ),
                    label,
                )
            )

        if len(candidates) <= MAX_PLAYER_TRACKS:
            return {
                track_id
                for (
                    track_id,
                    _,
                    _,
                    _,
                ) in candidates
            }

        green = sorted(
            [
                item
                for item in candidates
                if item[3] == "green"
            ],
            key=lambda item: item[2],
            reverse=True,
        )

        white = sorted(
            [
                item
                for item in candidates
                if item[3] == "white"
            ],
            key=lambda item: item[2],
            reverse=True,
        )

        unknown = sorted(
            [
                item
                for item in candidates
                if item[3]
                not in (
                    "green",
                    "white",
                )
            ],
            key=lambda item: item[2],
            reverse=True,
        )

        selected = []

        selected.extend(
            green[
                :MAX_PLAYERS_PER_TEAM
            ]
        )

        selected.extend(
            white[
                :MAX_PLAYERS_PER_TEAM
            ]
        )

        selected_ids = {
            item[0]
            for item in selected
        }

        remaining_slots = (
            MAX_PLAYER_TRACKS
            - len(selected_ids)
        )

        if remaining_slots > 0:
            remaining = sorted(
                [
                    item
                    for item in candidates
                    if item[0]
                    not in selected_ids
                ],
                key=lambda item: item[2],
                reverse=True,
            )

            selected.extend(
                remaining[
                    :remaining_slots
                ]
            )

        return {
            item[0]
            for item in selected[
                :MAX_PLAYER_TRACKS
            ]
        }

    def _select_special_tracks(
        self,
        merged_stats,
        maximum,
        minimum_frames,
    ):
        candidates = []

        for track_id, stat in merged_stats.items():
            frames = len(
                stat["frames"]
            )

            if frames < minimum_frames:
                continue

            label = stat["label"]

            purity = self._label_purity(
                stat["labels"],
                label,
            )

            quality = (
                frames
                * stat[
                    "average_confidence"
                ]
                * (
                    1.0
                    + 0.30 * purity
                )
            )

            candidates.append(
                (
                    track_id,
                    quality,
                    frames,
                    stat,
                )
            )

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return {
            item[0]
            for item in candidates[
                :maximum
            ]
        }

    def _apply_category(
        self,
        tracks,
        raw_frames,
        category,
        mapping,
        selected_ids,
    ):
        for frame_num, frame_entries in enumerate(
            raw_frames
        ):
            for (
                raw_id,
                (
                    bbox,
                    confidence,
                    label,
                    signature,
                ),
            ) in frame_entries.items():

                canonical_id = mapping.get(
                    raw_id,
                    raw_id,
                )

                if (
                    canonical_id
                    not in selected_ids
                ):
                    continue

                tracks[
                    category
                ][frame_num][
                    canonical_id
                ] = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "team_hint": label,
                    "appearance": signature,
                }

    def _ball_dicts_from_interpolated_points(
        self,
        raw_ball_points,
        frame_count,
        fps,
    ):
        max_gap_frames = int(
            MAX_BALL_GAP_SECONDS
            * fps
        )

        ball_track = [
            {}
            for _ in range(
                frame_count
            )
        ]

        if not raw_ball_points:
            return ball_track

        known_frames = sorted(
            raw_ball_points.keys()
        )

        def make_entry(
            x,
            y,
            conf,
            is_interp,
        ):
            half_box = 8

            return {
                "bbox": [
                    float(x - half_box),
                    float(y - half_box),
                    float(x + half_box),
                    float(y + half_box),
                ],
                "confidence": conf,
                "is_interpolated": is_interp,
            }

        for i in range(
            len(known_frames) - 1
        ):
            f1 = known_frames[i]
            f2 = known_frames[i + 1]

            x1, y1, conf1 = (
                raw_ball_points[f1]
            )

            x2, y2, _ = (
                raw_ball_points[f2]
            )

            ball_track[
                f1
            ][1] = make_entry(
                x1,
                y1,
                conf1,
                False,
            )

            gap = f2 - f1

            if (
                1 < gap
                <= max_gap_frames
            ):
                for step in range(
                    1,
                    gap,
                ):
                    t = (
                        step
                        / gap
                    )

                    x = (
                        x1
                        + (x2 - x1)
                        * t
                    )

                    y = (
                        y1
                        + (y2 - y1)
                        * t
                    )

                    ball_track[
                        f1 + step
                    ][1] = make_entry(
                        x,
                        y,
                        None,
                        True,
                    )

        last_frame = known_frames[
            -1
        ]

        x, y, conf = (
            raw_ball_points[
                last_frame
            ]
        )

        ball_track[
            last_frame
        ][1] = make_entry(
            x,
            y,
            conf,
            False,
        )

        return ball_track

    def get_object_tracks(
        self,
        frames,
        fps=25,
        read_from_stub=False,
        stub_path=None,
    ):
        if (
            read_from_stub
            and stub_path is not None
            and Path(stub_path).exists()
        ):
            logger.info(
                "Loading tracks from stub: %s",
                stub_path,
            )

            with open(
                stub_path,
                "rb",
            ) as f:
                return pickle.load(f)

        if not frames:
            logger.warning(
                "No frames supplied to tracker."
            )

            return {
                "players": [],
                "goalkeepers": [],
                "referees": [],
                "ball": [],
            }

        detections = self.detect_frames(
            frames
        )

        player_class_id = (
            self.class_name_to_id.get(
                "player"
            )
        )

        goalkeeper_class_id = (
            self.class_name_to_id.get(
                "goalkeeper"
            )
        )

        referee_class_id = (
            self.class_name_to_id.get(
                "referee"
            )
        )

        ball_class_id = (
            self.class_name_to_id.get(
                "ball"
            )
        )

        player_tracker = (
            self._new_tracker(
                fps,
                LOST_TRACK_BUFFER_SECONDS,
                PLAYER_CONFIDENCE,
                PLAYER_MATCH_THRESHOLD,
            )
        )

        goalkeeper_tracker = (
            self._new_tracker(
                fps,
                LOST_TRACK_BUFFER_SECONDS,
                GOALKEEPER_CONFIDENCE,
                GOALKEEPER_MATCH_THRESHOLD,
            )
        )

        referee_tracker = (
            self._new_tracker(
                fps,
                LOST_TRACK_BUFFER_SECONDS,
                REFEREE_CONFIDENCE,
                REFEREE_MATCH_THRESHOLD,
            )
        )

        raw_player_frames = []
        raw_goalkeeper_frames = []
        raw_referee_frames = []

        raw_ball_points = {}

        raw_counts = {
            "player": 0,
            "goalkeeper": 0,
            "referee": 0,
            "ball": 0,
        }

        filtered_counts = {
            "player": 0,
            "goalkeeper": 0,
            "referee": 0,
            "ball": 0,
        }

        logger.info("=" * 70)
        logger.info(
            "V9 FOOTBALL TRACKING"
        )
        logger.info("=" * 70)

        logger.info(
            "Video configuration: 1920x1080 @ 25 FPS"
        )

        logger.info(
            "Inference image size: %d",
            self.imgsz,
        )

        logger.info(
            "Player threshold: %.2f",
            PLAYER_CONFIDENCE,
        )

        logger.info(
            "Goalkeeper threshold: %.2f",
            GOALKEEPER_CONFIDENCE,
        )

        logger.info(
            "Referee threshold: %.2f",
            REFEREE_CONFIDENCE,
        )

        logger.info(
            "Player matching threshold: %.2f",
            PLAYER_MATCH_THRESHOLD,
        )

        logger.info(
            "Fragment gap: %d frames",
            MAX_FRAGMENT_GAP_FRAMES,
        )

        for frame_num, detection in enumerate(
            detections
        ):
            sv_detections = (
                sv.Detections.from_ultralytics(
                    detection
                )
            )

            if len(sv_detections) == 0:
                raw_player_frames.append({})
                raw_goalkeeper_frames.append({})
                raw_referee_frames.append({})
                continue

            if ball_class_id is not None:
                ball_mask = (
                    sv_detections.class_id
                    == ball_class_id
                )

                ball_detections = (
                    sv_detections[
                        ball_mask
                    ]
                )

                raw_counts[
                    "ball"
                ] += len(
                    ball_detections
                )

                if len(
                    ball_detections
                ) > 0:
                    best_idx = int(
                        np.argmax(
                            ball_detections.confidence
                        )
                    )

                    conf = float(
                        ball_detections.confidence[
                            best_idx
                        ]
                    )

                    if conf >= BALL_CONFIDENCE:
                        x1, y1, x2, y2 = (
                            ball_detections.xyxy[
                                best_idx
                            ]
                        )

                        raw_ball_points[
                            frame_num
                        ] = (
                            float(
                                (x1 + x2)
                                / 2
                            ),
                            float(
                                (y1 + y2)
                                / 2
                            ),
                            conf,
                        )

                        filtered_counts[
                            "ball"
                        ] += 1

                person_mask = ~ball_mask

                person_detections = (
                    sv_detections[
                        person_mask
                    ]
                )

            else:
                person_detections = (
                    sv_detections
                )

            if len(
                person_detections
            ) > 0:
                class_ids = (
                    person_detections.class_id
                )

                raw_counts[
                    "player"
                ] += int(
                    np.sum(
                        class_ids
                        == player_class_id
                    )
                )

                raw_counts[
                    "goalkeeper"
                ] += int(
                    np.sum(
                        class_ids
                        == goalkeeper_class_id
                    )
                )

                raw_counts[
                    "referee"
                ] += int(
                    np.sum(
                        class_ids
                        == referee_class_id
                    )
                )

            person_detections = (
                self._resolve_person_overlaps(
                    person_detections
                )
            )

            player_detections = (
                self._filter_class(
                    person_detections,
                    player_class_id,
                    PLAYER_CONFIDENCE,
                )
            )

            goalkeeper_detections = (
                self._filter_class(
                    person_detections,
                    goalkeeper_class_id,
                    GOALKEEPER_CONFIDENCE,
                )
            )

            referee_detections = (
                self._filter_class(
                    person_detections,
                    referee_class_id,
                    REFEREE_CONFIDENCE,
                )
            )

            filtered_counts[
                "player"
            ] += len(
                player_detections
            )

            filtered_counts[
                "goalkeeper"
            ] += len(
                goalkeeper_detections
            )

            filtered_counts[
                "referee"
            ] += len(
                referee_detections
            )

            tracked_players = (
                player_tracker.update_with_detections(
                    player_detections
                )
            )

            tracked_goalkeepers = (
                goalkeeper_tracker.update_with_detections(
                    goalkeeper_detections
                )
            )

            tracked_referees = (
                referee_tracker.update_with_detections(
                    referee_detections
                )
            )

            raw_player_frames.append(
                self._tracked_entries(
                    tracked_players,
                    frames[frame_num],
                )
            )

            raw_goalkeeper_frames.append(
                self._tracked_entries(
                    tracked_goalkeepers,
                    frames[frame_num],
                )
            )

            raw_referee_frames.append(
                self._tracked_entries(
                    tracked_referees,
                    frames[frame_num],
                )
            )

            if frame_num % 25 == 0:
                logger.info(
                    "TRACK frame %d: players=%d "
                    "goalkeepers=%d referees=%d",
                    frame_num,
                    len(
                        raw_player_frames[
                            -1
                        ]
                    ),
                    len(
                        raw_goalkeeper_frames[
                            -1
                        ]
                    ),
                    len(
                        raw_referee_frames[
                            -1
                        ]
                    ),
                )

        logger.info("-" * 70)
        logger.info(
            "V9 FRAGMENT RECONSTRUCTION"
        )
        logger.info("-" * 70)

        (
            player_mapping,
            player_raw_stats,
            player_groups,
        ) = self._merge_track_fragments(
            raw_player_frames
        )

        (
            goalkeeper_mapping,
            goalkeeper_raw_stats,
            goalkeeper_groups,
        ) = self._merge_track_fragments(
            raw_goalkeeper_frames
        )

        (
            referee_mapping,
            referee_raw_stats,
            referee_groups,
        ) = self._merge_track_fragments(
            raw_referee_frames
        )

        player_merged_stats = (
            self._merged_statistics(
                player_raw_stats,
                player_groups,
            )
        )

        goalkeeper_merged_stats = (
            self._merged_statistics(
                goalkeeper_raw_stats,
                goalkeeper_groups,
            )
        )

        referee_merged_stats = (
            self._merged_statistics(
                referee_raw_stats,
                referee_groups,
            )
        )

        selected_players = (
            self._select_player_tracks(
                player_merged_stats
            )
        )

        selected_goalkeepers = (
            self._select_special_tracks(
                goalkeeper_merged_stats,
                MAX_GOALKEEPER_TRACKS,
                MIN_GOALKEEPER_TRACK_FRAMES,
            )
        )

        selected_referees = (
            self._select_special_tracks(
                referee_merged_stats,
                MAX_REFEREE_TRACKS,
                MIN_REFEREE_TRACK_FRAMES,
            )
        )

        tracks = {
            "players": [
                {}
                for _ in range(
                    len(frames)
                )
            ],
            "goalkeepers": [
                {}
                for _ in range(
                    len(frames)
                )
            ],
            "referees": [
                {}
                for _ in range(
                    len(frames)
                )
            ],
            "ball": [
                {}
                for _ in range(
                    len(frames)
                )
            ],
        }

        self._apply_category(
            tracks,
            raw_player_frames,
            "players",
            player_mapping,
            selected_players,
        )

        self._apply_category(
            tracks,
            raw_goalkeeper_frames,
            "goalkeepers",
            goalkeeper_mapping,
            selected_goalkeepers,
        )

        self._apply_category(
            tracks,
            raw_referee_frames,
            "referees",
            referee_mapping,
            selected_referees,
        )

        tracks[
            "ball"
        ] = self._ball_dicts_from_interpolated_points(
            raw_ball_points,
            len(frames),
            fps,
        )

        player_ids = {
            track_id
            for frame in tracks["players"]
            for track_id in frame
        }

        goalkeeper_ids = {
            track_id
            for frame in tracks["goalkeepers"]
            for track_id in frame
        }

        referee_ids = {
            track_id
            for frame in tracks["referees"]
            for track_id in frame
        }

        player_active = [
            len(frame)
            for frame in tracks[
                "players"
            ]
        ]

        goalkeeper_active = [
            len(frame)
            for frame in tracks[
                "goalkeepers"
            ]
        ]

        referee_active = [
            len(frame)
            for frame in tracks[
                "referees"
            ]
        ]

        ball_known = sum(
            bool(frame)
            for frame in tracks[
                "ball"
            ]
        )

        logger.info("=" * 70)
        logger.info(
            "V9 FINAL TRACKING SUMMARY"
        )
        logger.info("=" * 70)

        logger.info(
            "RAW player detections:       %d",
            raw_counts["player"],
        )

        logger.info(
            "FILTERED player detections:  %d",
            filtered_counts["player"],
        )

        logger.info(
            "RAW goalkeeper detections:   %d",
            raw_counts["goalkeeper"],
        )

        logger.info(
            "FILTERED goalkeeper detections: %d",
            filtered_counts["goalkeeper"],
        )

        logger.info(
            "RAW referee detections:      %d",
            raw_counts["referee"],
        )

        logger.info(
            "FILTERED referee detections: %d",
            filtered_counts["referee"],
        )

        logger.info(
            "RAW ball detections:         %d",
            raw_counts["ball"],
        )

        logger.info(
            "FILTERED ball detections:    %d",
            filtered_counts["ball"],
        )

        logger.info(
            "Raw player track IDs:        %d",
            len(player_raw_stats),
        )

        logger.info(
            "Merged player tracks:        %d",
            len(player_merged_stats),
        )

        logger.info(
            "Final player IDs:            %d",
            len(player_ids),
        )

        logger.info(
            "Raw goalkeeper track IDs:    %d",
            len(goalkeeper_raw_stats),
        )

        logger.info(
            "Merged goalkeeper tracks:    %d",
            len(goalkeeper_merged_stats),
        )

        logger.info(
            "Final goalkeeper IDs:        %d",
            len(goalkeeper_ids),
        )

        logger.info(
            "Raw referee track IDs:       %d",
            len(referee_raw_stats),
        )

        logger.info(
            "Merged referee tracks:       %d",
            len(referee_merged_stats),
        )

        logger.info(
            "Final referee IDs:           %d",
            len(referee_ids),
        )

        logger.info(
            "Player appearance groups: %s",
            {
                track_id: player_merged_stats[
                    track_id
                ]["label"]
                for track_id in sorted(
                    selected_players
                )
            },
        )

        if player_active:
            logger.info(
                "Players active/frame: "
                "min=%d max=%d avg=%.2f",
                min(player_active),
                max(player_active),
                np.mean(player_active),
            )

        if goalkeeper_active:
            logger.info(
                "Goalkeepers active/frame: "
                "min=%d max=%d avg=%.2f",
                min(goalkeeper_active),
                max(goalkeeper_active),
                np.mean(
                    goalkeeper_active
                ),
            )

        if referee_active:
            logger.info(
                "Referees active/frame: "
                "min=%d max=%d avg=%.2f",
                min(referee_active),
                max(referee_active),
                np.mean(
                    referee_active
                ),
            )

        logger.info(
            "Ball position known: %d/%d frames (%.1f%%)",
            ball_known,
            len(frames),
            (
                100.0
                * ball_known
                / len(frames)
                if frames
                else 0.0
            ),
        )

        if stub_path is not None:
            Path(
                stub_path
            ).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                stub_path,
                "wb",
            ) as f:
                pickle.dump(
                    tracks,
                    f,
                )

            logger.info(
                "Saved tracking stub: %s",
                stub_path,
            )

        return tracks

    def add_position_to_tracks(
        self,
        tracks,
    ):
        for (
            object_type,
            object_tracks,
        ) in tracks.items():

            for frame_num, track in enumerate(
                object_tracks
            ):
                for (
                    track_id,
                    track_info,
                ) in track.items():

                    bbox = track_info[
                        "bbox"
                    ]

                    if (
                        object_type
                        == "ball"
                    ):
                        position = (
                            (
                                bbox[0]
                                + bbox[2]
                            ) / 2,
                            (
                                bbox[1]
                                + bbox[3]
                            ) / 2,
                        )
                    else:
                        position = (
                            (
                                bbox[0]
                                + bbox[2]
                            ) / 2,
                            bbox[3],
                        )

                    track_info[
                        "position"
                    ] = position