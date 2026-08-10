from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import math


@dataclass
class IdentityQualityResult:
    # Basic counts
    total_tracks: int = 0
    evaluated_tracks: int = 0
    total_frames: int = 0
    evaluated_frames: int = 0

    predicted_identities: int = 0
    ground_truth_identities: int = 0

    # Track-level metrics
    track_accuracy: float = 0.0
    track_purity: float = 0.0

    # Frame-level metrics
    frame_accuracy: float = 0.0
    identity_switches: int = 0

    # Identity metrics
    id_precision: float = 0.0
    id_recall: float = 0.0
    id_f1: float = 0.0
    idf1: float = 0.0

    # Structural errors
    false_merges: int = 0
    false_splits: int = 0

    # Counts
    correct_track_assignments: int = 0
    incorrect_track_assignments: int = 0

    correct_frame_assignments: int = 0
    incorrect_frame_assignments: int = 0

    # Detailed reports
    predicted_gid_to_gt: dict[str, str | None] = field(default_factory=dict)
    gid_purity: dict[str, float] = field(default_factory=dict)

    gt_to_predicted_gids: dict[str, list[str]] = field(
        default_factory=dict
    )

    suspicious_gids: list[str] = field(default_factory=list)
    merged_gids: list[str] = field(default_factory=list)
    fragmented_gt_identities: dict[str, list[str]] = field(
        default_factory=dict
    )

    identity_switch_details: list[dict[str, Any]] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tracks": self.total_tracks,
            "evaluated_tracks": self.evaluated_tracks,
            "total_frames": self.total_frames,
            "evaluated_frames": self.evaluated_frames,

            "predicted_identities": self.predicted_identities,
            "ground_truth_identities": self.ground_truth_identities,

            "track_accuracy": self.track_accuracy,
            "track_purity": self.track_purity,

            "frame_accuracy": self.frame_accuracy,
            "identity_switches": self.identity_switches,

            "id_precision": self.id_precision,
            "id_recall": self.id_recall,
            "id_f1": self.id_f1,
            "idf1": self.idf1,

            "false_merges": self.false_merges,
            "false_splits": self.false_splits,

            "correct_track_assignments":
                self.correct_track_assignments,

            "incorrect_track_assignments":
                self.incorrect_track_assignments,

            "correct_frame_assignments":
                self.correct_frame_assignments,

            "incorrect_frame_assignments":
                self.incorrect_frame_assignments,

            "predicted_gid_to_gt":
                self.predicted_gid_to_gt,

            "gid_purity":
                self.gid_purity,

            "gt_to_predicted_gids":
                self.gt_to_predicted_gids,

            "suspicious_gids":
                self.suspicious_gids,

            "merged_gids":
                self.merged_gids,

            "fragmented_gt_identities":
                self.fragmented_gt_identities,

            "identity_switch_details":
                self.identity_switch_details,
        }


class FusedIdentityQualityEvaluator:
    """
    Ground-truth-aware evaluator for fused player identities.

    Supports:

    1. Track-level ground truth
    2. Frame-level ground truth
    3. Track accuracy
    4. GID purity
    5. False merges
    6. False splits
    7. Identity switches
    8. ID Precision / Recall / F1
    9. IDF1-style identity matching
    """

    def __init__(
        self,
        manager,
        ground_truth: dict[str, Any] | str | Path,
    ):
        self.manager = manager

        if isinstance(ground_truth, (str, Path)):
            with open(ground_truth, "r", encoding="utf-8") as f:
                ground_truth = json.load(f)

        self.ground_truth = ground_truth

        self.track_identity_gt = {
            int(track_id): str(identity)
            for track_id, identity in
            ground_truth.get("track_identity", {}).items()
        }

        self.frame_identity_gt = self._normalize_frame_gt(
            ground_truth.get("frame_identity", {})
        )

        self.result = IdentityQualityResult()

    # ------------------------------------------------------------------
    # Ground truth normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_frame_gt(
        frame_gt: dict[str, Any]
    ) -> dict[int, dict[int, str]]:

        normalized = {}

        for frame, tracks in frame_gt.items():

            frame_index = int(frame)

            normalized[frame_index] = {
                int(track_id): str(identity)
                for track_id, identity in tracks.items()
            }

        return normalized

    # ------------------------------------------------------------------
    # Manager access
    # ------------------------------------------------------------------

    def _get_track_identity_map(self) -> dict[int, str]:

        mapping = self.manager.get_track_identity_map()

        return {
            int(track_id): str(gid)
            for track_id, gid in mapping.items()
        }

    def _get_tracks(self) -> dict[int, Any]:

        """
        Retrieve tracks from the fused identity manager.

        The current manager exposes its tracking history through the
        underlying tracker. This method intentionally supports several
        common internal layouts so the evaluator remains decoupled
        from one exact manager implementation.
        """

        candidates = [
            "_tracks",
            "tracks",
            "_track_history",
            "track_history",
        ]

        for name in candidates:

            if hasattr(self.manager, name):

                value = getattr(self.manager, name)

                if isinstance(value, dict):
                    return {
                        int(k): v
                        for k, v in value.items()
                    }

        # Try tracker
        if hasattr(self.manager, "tracker"):

            tracker = self.manager.tracker

            if hasattr(tracker, "get_track_history"):

                history = tracker.get_track_history()

                return {
                    int(k): v
                    for k, v in history.items()
                }

        raise AttributeError(
            "Unable to locate track history on fused identity manager. "
            "Expose the tracking history through manager.tracks or "
            "manager.get_track_history()."
        )

    # ------------------------------------------------------------------
    # Track level evaluation
    # ------------------------------------------------------------------

    def evaluate_track_level(self):

        track_to_gid = self._get_track_identity_map()

        self.result.total_tracks = len(track_to_gid)

        evaluated = 0
        correct = 0

        gid_to_gt_votes = defaultdict(Counter)
        gt_to_gids = defaultdict(set)

        for track_id, gid in track_to_gid.items():

            gt_identity = self.track_identity_gt.get(track_id)

            if gt_identity is None:
                continue

            evaluated += 1

            gid_to_gt_votes[gid][gt_identity] += 1
            gt_to_gids[gt_identity].add(gid)

        self.result.evaluated_tracks = evaluated

        # --------------------------------------------------------------
        # Determine dominant GT identity for every predicted GID
        # --------------------------------------------------------------

        for gid, votes in gid_to_gt_votes.items():

            if not votes:
                continue

            dominant_gt, dominant_count = votes.most_common(1)[0]

            total = sum(votes.values())

            purity = (
                dominant_count / total
                if total > 0
                else 0.0
            )

            self.result.predicted_gid_to_gt[gid] = dominant_gt
            self.result.gid_purity[gid] = purity

            if purity < 1.0:
                self.result.suspicious_gids.append(gid)

            if len(votes) > 1:
                self.result.merged_gids.append(gid)

            correct += dominant_count

        self.result.correct_track_assignments = correct

        self.result.incorrect_track_assignments = (
            evaluated - correct
        )

        if evaluated:
            self.result.track_accuracy = (
                correct / evaluated
            )

        if self.result.gid_purity:
            self.result.track_purity = (
                sum(self.result.gid_purity.values())
                / len(self.result.gid_purity)
            )

        # --------------------------------------------------------------
        # Ground truth → predicted GIDs
        # --------------------------------------------------------------

        self.result.gt_to_predicted_gids = {
            identity: sorted(gids)
            for identity, gids in gt_to_gids.items()
        }

        self.result.ground_truth_identities = len(
            gt_to_gids
        )

        self.result.predicted_identities = len(
            self.result.predicted_gid_to_gt
        )

        # --------------------------------------------------------------
        # False splits
        # --------------------------------------------------------------

        for gt_identity, gids in gt_to_gids.items():

            if len(gids) > 1:

                self.result.fragmented_gt_identities[
                    gt_identity
                ] = sorted(gids)

        self.result.false_splits = len(
            self.result.fragmented_gt_identities
        )

        # --------------------------------------------------------------
        # False merges
        # --------------------------------------------------------------

        self.result.false_merges = len(
            self.result.merged_gids
        )

    # ------------------------------------------------------------------
    # Frame level evaluation
    # ------------------------------------------------------------------

    def evaluate_frame_level(self):

        if not self.frame_identity_gt:
            return

        track_to_gid = self._get_track_identity_map()

        correct = 0
        incorrect = 0
        evaluated = 0

        # frame -> predicted GID -> GT identities
        frame_predictions = defaultdict(dict)

        for frame_index, track_gt in self.frame_identity_gt.items():

            for track_id, gt_identity in track_gt.items():

                gid = track_to_gid.get(track_id)

                if gid is None:
                    continue

                evaluated += 1

                frame_predictions[frame_index][
                    track_id
                ] = (
                    gid,
                    gt_identity,
                )

        self.result.evaluated_frames = evaluated

        # --------------------------------------------------------------
        # Calculate frame accuracy
        # --------------------------------------------------------------

        for frame_index, predictions in frame_predictions.items():

            for track_id, (gid, gt_identity) in predictions.items():

                predicted_gt = (
                    self.result.predicted_gid_to_gt.get(gid)
                )

                if predicted_gt == gt_identity:
                    correct += 1
                else:
                    incorrect += 1

        self.result.correct_frame_assignments = correct
        self.result.incorrect_frame_assignments = incorrect

        if evaluated:

            self.result.frame_accuracy = (
                correct / evaluated
            )

        self.result.total_frames = len(
            self.frame_identity_gt
        )

        # --------------------------------------------------------------
        # Identity switches
        # --------------------------------------------------------------

        for track_id in self.track_identity_gt:

            previous_gid = None
            previous_frame = None

            for frame_index in sorted(
                self.frame_identity_gt.keys()
            ):

                tracks = self.frame_identity_gt[
                    frame_index
                ]

                if track_id not in tracks:
                    continue

                gid = track_to_gid.get(track_id)

                if gid is None:
                    continue

                if (
                    previous_gid is not None
                    and gid != previous_gid
                ):

                    self.result.identity_switches += 1

                    self.result.identity_switch_details.append(
                        {
                            "track_id": track_id,
                            "from_gid": previous_gid,
                            "to_gid": gid,
                            "from_frame": previous_frame,
                            "to_frame": frame_index,
                        }
                    )

                previous_gid = gid
                previous_frame = frame_index

    # ------------------------------------------------------------------
    # ID metrics
    # ------------------------------------------------------------------

    def evaluate_identity_metrics(self):

        """
        Calculate identity assignment precision/recall/F1.

        We compare GT identity pairs against predicted GID pairs.

        If two detections belong to the same real player they should
        receive the same predicted GID.

        If they belong to different players they should not share
        a GID.
        """

        track_to_gid = self._get_track_identity_map()

        evaluated_tracks = [
            track_id
            for track_id in self.track_identity_gt
            if track_id in track_to_gid
        ]

        if len(evaluated_tracks) < 2:
            return

        true_positive = 0
        false_positive = 0
        false_negative = 0

        for i in range(len(evaluated_tracks)):

            track_a = evaluated_tracks[i]

            gt_a = self.track_identity_gt[track_a]
            gid_a = track_to_gid[track_a]

            for j in range(i + 1, len(evaluated_tracks)):

                track_b = evaluated_tracks[j]

                gt_b = self.track_identity_gt[track_b]
                gid_b = track_to_gid[track_b]

                same_gt = gt_a == gt_b
                same_pred = gid_a == gid_b

                if same_gt and same_pred:
                    true_positive += 1

                elif not same_gt and same_pred:
                    false_positive += 1

                elif same_gt and not same_pred:
                    false_negative += 1

        precision_denominator = (
            true_positive + false_positive
        )

        recall_denominator = (
            true_positive + false_negative
        )

        if precision_denominator:

            self.result.id_precision = (
                true_positive
                / precision_denominator
            )

        if recall_denominator:

            self.result.id_recall = (
                true_positive
                / recall_denominator
            )

        if (
            self.result.id_precision
            + self.result.id_recall
        ):

            self.result.id_f1 = (
                2
                * self.result.id_precision
                * self.result.id_recall
                / (
                    self.result.id_precision
                    + self.result.id_recall
                )
            )

        # For the track-pair formulation, this is the same
        # identity-consistency F1.
        self.result.idf1 = self.result.id_f1

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> IdentityQualityResult:

        self.result = IdentityQualityResult()

        self.evaluate_track_level()

        self.evaluate_frame_level()

        self.evaluate_identity_metrics()

        return self.result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    def print_report(
        self,
        result: IdentityQualityResult | None = None,
    ):

        if result is None:
            result = self.result

        print()
        print("=" * 72)
        print("FUSED IDENTITY QUALITY EVALUATION")
        print("=" * 72)

        print()
        print("TRACK COVERAGE")
        print("-" * 72)

        print(
            f"Total tracks:              "
            f"{result.total_tracks}"
        )

        print(
            f"GT-evaluated tracks:       "
            f"{result.evaluated_tracks}"
        )

        print(
            f"Correct assignments:       "
            f"{result.correct_track_assignments}"
        )

        print(
            f"Incorrect assignments:     "
            f"{result.incorrect_track_assignments}"
        )

        print()
        print("IDENTITY STATISTICS")
        print("-" * 72)

        print(
            f"Predicted identities:      "
            f"{result.predicted_identities}"
        )

        print(
            f"Ground-truth identities:   "
            f"{result.ground_truth_identities}"
        )

        print(
            f"Track accuracy:            "
            f"{self._pct(result.track_accuracy)}"
        )

        print(
            f"Average GID purity:        "
            f"{self._pct(result.track_purity)}"
        )

        print()
        print("IDENTITY METRICS")
        print("-" * 72)

        print(
            f"ID Precision:              "
            f"{self._pct(result.id_precision)}"
        )

        print(
            f"ID Recall:                 "
            f"{self._pct(result.id_recall)}"
        )

        print(
            f"ID F1:                     "
            f"{self._pct(result.id_f1)}"
        )

        print(
            f"IDF1:                      "
            f"{self._pct(result.idf1)}"
        )

        print()
        print("ERROR ANALYSIS")
        print("-" * 72)

        print(
            f"False merges:              "
            f"{result.false_merges}"
        )

        print(
            f"False splits:              "
            f"{result.false_splits}"
        )

        print(
            f"Identity switches:        "
            f"{result.identity_switches}"
        )

        if result.evaluated_frames:

            print()
            print("FRAME-LEVEL METRICS")
            print("-" * 72)

            print(
                f"GT frames:                 "
                f"{result.total_frames}"
            )

            print(
                f"Evaluated frame labels:    "
                f"{result.evaluated_frames}"
            )

            print(
                f"Frame accuracy:            "
                f"{self._pct(result.frame_accuracy)}"
            )

    # ------------------------------------------------------------------
    # Detailed reports
    # ------------------------------------------------------------------

    def print_merged_identities(self):

        print()
        print("=" * 72)
        print("FALSE MERGE / MIXED IDENTITY REPORT")
        print("=" * 72)

        if not self.result.merged_gids:

            print()
            print("No merged identities detected.")

            return

        for gid in sorted(self.result.merged_gids):

            purity = self.result.gid_purity.get(
                gid,
                0.0,
            )

            gt_identity = self.result.predicted_gid_to_gt.get(
                gid
            )

            print()
            print(
                f"{gid}: "
                f"dominant={gt_identity}, "
                f"purity={purity:.3f}"
            )

    def print_suspicious_identities(self):

        print()
        print("=" * 72)
        print("SUSPICIOUS IDENTITIES")
        print("=" * 72)

        if not self.result.suspicious_gids:

            print()
            print("No suspicious identities found.")

            return

        for gid in sorted(
            self.result.suspicious_gids
        ):

            purity = self.result.gid_purity.get(
                gid,
                0.0,
            )

            print(
                f"{gid}: "
                f"purity={purity:.3f}"
            )

    def print_fragmentation(self):

        print()
        print("=" * 72)
        print("GROUND-TRUTH IDENTITY FRAGMENTATION")
        print("=" * 72)

        if not self.result.fragmented_gt_identities:

            print()
            print("No fragmented identities found.")

            return

        for (
            gt_identity,
            gids,
        ) in sorted(
            self.result.fragmented_gt_identities.items()
        ):

            print(
                f"{gt_identity}: "
                f"{len(gids)} predicted GIDs "
                f"-> {gids}"
            )

    def print_identity_switches(self):

        print()
        print("=" * 72)
        print("IDENTITY SWITCHES")
        print("=" * 72)

        if not self.result.identity_switch_details:

            print()
            print("No identity switches detected.")

            return

        for switch in self.result.identity_switch_details:

            print(
                f"Track {switch['track_id']}: "
                f"{switch['from_gid']} -> "
                f"{switch['to_gid']} "
                f"at frame "
                f"{switch['from_frame']} -> "
                f"{switch['to_frame']}"
            )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_report(
        self,
        output_path: str | Path,
        result: IdentityQualityResult | None = None,
    ):

        if result is None:
            result = self.result

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                result.to_dict(),
                f,
                indent=2,
            )