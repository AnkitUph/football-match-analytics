from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import csv


@dataclass
class GroundTruthDetection:
    frame_id: int
    identity_id: str
    bbox: tuple[float, float, float, float]
    class_name: str = "player"


@dataclass
class PredictionDetection:
    frame_id: int
    track_id: int
    global_identity: Optional[str]
    bbox: tuple[float, float, float, float]
    class_name: str = "player"


@dataclass
class IdentityQualityResult:
    total_gt_detections: int
    matched_detections: int
    unmatched_gt_detections: int
    unmatched_predictions: int

    id_precision: float
    id_recall: float
    idf1: float

    identity_switches: int

    false_merges: int
    false_splits: int

    gt_identities: int
    predicted_identities: int

    gid_purity: float
    average_gt_identities_per_gid: float

    gt_fragmentation: float

    per_identity: dict[str, dict[str, Any]]


class IdentityQualityEvaluator:
    """
    Ground-truth based evaluator for the fused identity pipeline.

    Ground truth CSV format:

        frame_id,identity_id,x1,y1,x2,y2,class_name

    Example:

        100,P001,412,210,475,390,player
        100,P002,700,190,760,380,player
    """

    def __init__(
        self,
        manager,
        ground_truth_path: str,
        iou_threshold: float = 0.5,
        class_filter: Optional[set[str]] = None,
    ):
        self.manager = manager
        self.ground_truth_path = Path(ground_truth_path)
        self.iou_threshold = iou_threshold
        self.class_filter = class_filter

        self.ground_truth: list[GroundTruthDetection] = []
        self.predictions: list[PredictionDetection] = []

        self.matches: list[
            tuple[GroundTruthDetection, PredictionDetection]
        ] = []

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def evaluate(self) -> IdentityQualityResult:
        self._load_ground_truth()
        self._extract_predictions()
        self._match_detections()

        return self._calculate_metrics()

    # ------------------------------------------------------------------
    # GROUND TRUTH
    # ------------------------------------------------------------------

    def _load_ground_truth(self) -> None:
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(
                f"Ground truth file not found: "
                f"{self.ground_truth_path}"
            )

        self.ground_truth.clear()

        with self.ground_truth_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as f:

            reader = csv.DictReader(f)

            required = {
                "frame_id",
                "identity_id",
                "x1",
                "y1",
                "x2",
                "y2",
            }

            missing = required - set(reader.fieldnames or [])

            if missing:
                raise ValueError(
                    "Ground truth CSV is missing columns: "
                    f"{sorted(missing)}"
                )

            for row in reader:

                class_name = row.get(
                    "class_name",
                    "player",
                )

                if (
                    self.class_filter is not None
                    and class_name not in self.class_filter
                ):
                    continue

                self.ground_truth.append(
                    GroundTruthDetection(
                        frame_id=int(row["frame_id"]),
                        identity_id=str(row["identity_id"]),
                        bbox=(
                            float(row["x1"]),
                            float(row["y1"]),
                            float(row["x2"]),
                            float(row["y2"]),
                        ),
                        class_name=class_name,
                    )
                )

        print(
            f"Loaded ground-truth detections: "
            f"{len(self.ground_truth)}"
        )

    # ------------------------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------------------------

    def _extract_predictions(self) -> None:
        self.predictions.clear()

        track_history = self.manager.tracker.get_track_history()

        track_identity_map = (
            self.manager.get_track_identity_map()
        )

        for track_id, track in track_history.items():

            gid = track_identity_map.get(track_id)

            if gid is None:
                continue

            class_name = getattr(
                track,
                "class_name",
                "player",
            )

            if (
                self.class_filter is not None
                and class_name not in self.class_filter
            ):
                continue

            for observation in track.observations:

                frame_id = self._get_frame_id(
                    observation
                )

                bbox = self._get_bbox(
                    observation
                )

                if frame_id is None or bbox is None:
                    continue

                self.predictions.append(
                    PredictionDetection(
                        frame_id=frame_id,
                        track_id=track_id,
                        global_identity=gid,
                        bbox=bbox,
                        class_name=class_name,
                    )
                )

        print(
            f"Extracted predicted observations: "
            f"{len(self.predictions)}"
        )

    # ------------------------------------------------------------------
    # OBSERVATION HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _get_frame_id(
        observation,
    ) -> Optional[int]:

        for name in (
            "frame_id",
            "frame_index",
            "frame",
        ):
            if hasattr(observation, name):
                value = getattr(observation, name)

                if value is not None:
                    return int(value)

        return None

    @staticmethod
    def _get_bbox(
        observation,
    ) -> Optional[tuple[float, float, float, float]]:

        # Direct bbox-like attributes
        for name in (
            "bbox",
            "xyxy",
            "box",
            "bounding_box",
        ):

            if hasattr(observation, name):

                value = getattr(observation, name)

                if value is None:
                    continue

                try:
                    values = list(value)

                    if len(values) >= 4:
                        return (
                            float(values[0]),
                            float(values[1]),
                            float(values[2]),
                            float(values[3]),
                        )

                except (TypeError, ValueError):
                    pass

        # x1/y1/x2/y2 representation
        names = (
            "x1",
            "y1",
            "x2",
            "y2",
        )

        if all(hasattr(observation, name) for name in names):

            return (
                float(observation.x1),
                float(observation.y1),
                float(observation.x2),
                float(observation.y2),
            )

        return None

    # ------------------------------------------------------------------
    # MATCHING
    # ------------------------------------------------------------------

    def _match_detections(self) -> None:

        self.matches.clear()

        gt_by_frame = defaultdict(list)
        pred_by_frame = defaultdict(list)

        for gt in self.ground_truth:
            gt_by_frame[gt.frame_id].append(gt)

        for pred in self.predictions:
            pred_by_frame[pred.frame_id].append(pred)

        for frame_id, gt_items in gt_by_frame.items():

            pred_items = pred_by_frame.get(
                frame_id,
                [],
            )

            if not pred_items:
                continue

            candidates = []

            for gt in gt_items:

                for pred in pred_items:

                    if gt.class_name != pred.class_name:
                        continue

                    score = self._iou(
                        gt.bbox,
                        pred.bbox,
                    )

                    if score >= self.iou_threshold:

                        candidates.append(
                            (
                                score,
                                gt,
                                pred,
                            )
                        )

            # Highest IoU first
            candidates.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            used_gt = set()
            used_pred = set()

            for score, gt, pred in candidates:

                gt_key = id(gt)
                pred_key = id(pred)

                if gt_key in used_gt:
                    continue

                if pred_key in used_pred:
                    continue

                used_gt.add(gt_key)
                used_pred.add(pred_key)

                self.matches.append(
                    (
                        gt,
                        pred,
                    )
                )

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------

    def _calculate_metrics(
        self,
    ) -> IdentityQualityResult:

        matched = len(self.matches)

        total_gt = len(
            self.ground_truth
        )

        total_predictions = len(
            self.predictions
        )

        unmatched_gt = total_gt - matched

        unmatched_predictions = (
            total_predictions - matched
        )

        id_precision = (
            matched / total_predictions
            if total_predictions
            else 0.0
        )

        id_recall = (
            matched / total_gt
            if total_gt
            else 0.0
        )

        if id_precision + id_recall > 0:

            idf1 = (
                2
                * id_precision
                * id_recall
                / (id_precision + id_recall)
            )

        else:
            idf1 = 0.0

        identity_switches = (
            self._calculate_identity_switches()
        )

        false_merges = (
            self._calculate_false_merges()
        )

        false_splits = (
            self._calculate_false_splits()
        )

        gid_purity = (
            self._calculate_gid_purity()
        )

        average_gt_per_gid = (
            self._calculate_average_gt_per_gid()
        )

        fragmentation = (
            self._calculate_fragmentation()
        )

        per_identity = (
            self._calculate_per_identity()
        )

        gt_identities = len(
            {
                gt.identity_id
                for gt in self.ground_truth
            }
        )

        predicted_identities = len(
            {
                pred.global_identity
                for pred in self.predictions
                if pred.global_identity is not None
            }
        )

        return IdentityQualityResult(
            total_gt_detections=total_gt,
            matched_detections=matched,
            unmatched_gt_detections=unmatched_gt,
            unmatched_predictions=unmatched_predictions,
            id_precision=id_precision,
            id_recall=id_recall,
            idf1=idf1,
            identity_switches=identity_switches,
            false_merges=false_merges,
            false_splits=false_splits,
            gt_identities=gt_identities,
            predicted_identities=predicted_identities,
            gid_purity=gid_purity,
            average_gt_identities_per_gid=average_gt_per_gid,
            gt_fragmentation=fragmentation,
            per_identity=per_identity,
        )

    # ------------------------------------------------------------------
    # ID SWITCHES
    # ------------------------------------------------------------------

    def _calculate_identity_switches(self) -> int:

        identity_history = defaultdict(list)

        for gt, pred in self.matches:

            if pred.global_identity is None:
                continue

            identity_history[
                gt.identity_id
            ].append(
                (
                    gt.frame_id,
                    pred.global_identity,
                )
            )

        switches = 0

        for history in identity_history.values():

            history.sort(
                key=lambda x: x[0]
            )

            previous = None

            for frame_id, gid in history:

                if (
                    previous is not None
                    and gid != previous
                ):
                    switches += 1

                previous = gid

        return switches

    # ------------------------------------------------------------------
    # FALSE MERGES
    # ------------------------------------------------------------------

    def _calculate_false_merges(self) -> int:

        gid_to_gt_ids = defaultdict(set)

        for gt, pred in self.matches:

            if pred.global_identity is None:
                continue

            gid_to_gt_ids[
                pred.global_identity
            ].add(gt.identity_id)

        false_merges = 0

        for gt_ids in gid_to_gt_ids.values():

            if len(gt_ids) > 1:

                false_merges += (
                    len(gt_ids) - 1
                )

        return false_merges

    # ------------------------------------------------------------------
    # FALSE SPLITS
    # ------------------------------------------------------------------

    def _calculate_false_splits(self) -> int:

        gt_to_gids = defaultdict(set)

        for gt, pred in self.matches:

            if pred.global_identity is None:
                continue

            gt_to_gids[
                gt.identity_id
            ].add(
                pred.global_identity
            )

        false_splits = 0

        for gids in gt_to_gids.values():

            if len(gids) > 1:

                false_splits += (
                    len(gids) - 1
                )

        return false_splits

    # ------------------------------------------------------------------
    # GID PURITY
    # ------------------------------------------------------------------

    def _calculate_gid_purity(self) -> float:

        gid_to_gt = defaultdict(Counter)

        for gt, pred in self.matches:

            if pred.global_identity is None:
                continue

            gid_to_gt[
                pred.global_identity
            ][gt.identity_id] += 1

        if not gid_to_gt:
            return 0.0

        total = 0
        correct = 0

        for counts in gid_to_gt.values():

            total += sum(counts.values())

            correct += counts.most_common(1)[0][1]

        return (
            correct / total
            if total
            else 0.0
        )

    # ------------------------------------------------------------------
    # AVERAGE GT IDENTITIES PER GID
    # ------------------------------------------------------------------

    def _calculate_average_gt_per_gid(
        self,
    ) -> float:

        gid_to_gt = defaultdict(set)

        for gt, pred in self.matches:

            if pred.global_identity is None:
                continue

            gid_to_gt[
                pred.global_identity
            ].add(gt.identity_id)

        if not gid_to_gt:
            return 0.0

        return (
            sum(
                len(ids)
                for ids in gid_to_gt.values()
            )
            / len(gid_to_gt)
        )

    # ------------------------------------------------------------------
    # FRAGMENTATION
    # ------------------------------------------------------------------

    def _calculate_fragmentation(
        self,
    ) -> float:

        gt_to_gids = defaultdict(set)

        for gt, pred in self.matches:

            if pred.global_identity is None:
                continue

            gt_to_gids[
                gt.identity_id
            ].add(pred.global_identity)

        if not gt_to_gids:
            return 0.0

        fragmented = sum(
            1
            for gids in gt_to_gids.values()
            if len(gids) > 1
        )

        return (
            fragmented
            / len(gt_to_gids)
        )

    # ------------------------------------------------------------------
    # PER IDENTITY
    # ------------------------------------------------------------------

    def _calculate_per_identity(
        self,
    ) -> dict[str, dict[str, Any]]:

        result = defaultdict(
            lambda: {
                "matched": 0,
                "gids": Counter(),
                "frames": [],
            }
        )

        for gt, pred in self.matches:

            entry = result[
                gt.identity_id
            ]

            entry["matched"] += 1

            if pred.global_identity:
                entry["gids"][
                    pred.global_identity
                ] += 1

            entry["frames"].append(
                gt.frame_id
            )

        final = {}

        for identity_id, data in result.items():

            gids = data["gids"]

            dominant_gid = (
                gids.most_common(1)[0][0]
                if gids
                else None
            )

            dominant_count = (
                gids.most_common(1)[0][1]
                if gids
                else 0
            )

            final[identity_id] = {
                "matched": data["matched"],
                "global_identities": dict(gids),
                "dominant_gid": dominant_gid,
                "purity": (
                    dominant_count
                    / data["matched"]
                    if data["matched"]
                    else 0.0
                ),
                "fragmented": len(gids) > 1,
                "frame_count": len(
                    set(data["frames"])
                ),
            }

        return final

    # ------------------------------------------------------------------
    # IOU
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(
        box_a,
        box_b,
    ) -> float:

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        intersection = iw * ih

        area_a = max(
            0.0,
            ax2 - ax1,
        ) * max(
            0.0,
            ay2 - ay1,
        )

        area_b = max(
            0.0,
            bx2 - bx1,
        ) * max(
            0.0,
            by2 - by1,
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------

    def print_report(
        self,
        result: IdentityQualityResult,
    ) -> None:

        print()
        print("=" * 60)
        print("IDENTITY QUALITY EVALUATION")
        print("=" * 60)

        print()
        print("DETECTION COVERAGE")
        print("-" * 60)

        print(
            f"Ground-truth detections: "
            f"{result.total_gt_detections}"
        )

        print(
            f"Matched detections: "
            f"{result.matched_detections}"
        )

        print(
            f"Unmatched GT detections: "
            f"{result.unmatched_gt_detections}"
        )

        print(
            f"Unmatched predictions: "
            f"{result.unmatched_predictions}"
        )

        print()
        print("IDENTITY METRICS")
        print("-" * 60)

        print(
            f"ID Precision: "
            f"{result.id_precision:.4f}"
        )

        print(
            f"ID Recall:    "
            f"{result.id_recall:.4f}"
        )

        print(
            f"IDF1:         "
            f"{result.idf1:.4f}"
        )

        print()
        print("IDENTITY CONSISTENCY")
        print("-" * 60)

        print(
            f"GT identities: "
            f"{result.gt_identities}"
        )

        print(
            f"Predicted GIDs: "
            f"{result.predicted_identities}"
        )

        print(
            f"Identity switches: "
            f"{result.identity_switches}"
        )

        print(
            f"False merges: "
            f"{result.false_merges}"
        )

        print(
            f"False splits: "
            f"{result.false_splits}"
        )

        print(
            f"GID purity: "
            f"{result.gid_purity:.4f}"
        )

        print(
            f"Average GT identities/GID: "
            f"{result.average_gt_identities_per_gid:.4f}"
        )

        print(
            f"GT fragmentation: "
            f"{result.gt_fragmentation:.4f}"
        )

        print()
        print("=" * 60)

    # ------------------------------------------------------------------
    # PROBLEM IDENTITIES
    # ------------------------------------------------------------------

    def print_problem_identities(
        self,
        result: IdentityQualityResult,
        purity_threshold: float = 0.8,
    ) -> None:

        print()
        print("=" * 60)
        print("PROBLEM IDENTITIES")
        print("=" * 60)

        found = False

        for identity_id, data in sorted(
            result.per_identity.items()
        ):

            if (
                data["purity"]
                < purity_threshold
                or data["fragmented"]
            ):

                found = True

                print()
                print(
                    f"{identity_id}:"
                )

                print(
                    f"  Matched frames: "
                    f"{data['matched']}"
                )

                print(
                    f"  Purity: "
                    f"{data['purity']:.3f}"
                )

                print(
                    f"  Dominant GID: "
                    f"{data['dominant_gid']}"
                )

                print(
                    f"  GIDs: "
                    f"{data['global_identities']}"
                )

                print(
                    f"  Fragmented: "
                    f"{data['fragmented']}"
                )

        if not found:
            print(
                "No problematic identities detected."
            )