"""
Phase 7 - Global Identity Visual Validation.

Visual inspection tool for GlobalIdentityManager V2.

Purpose:

- inspect risky accepted identity matches
- inspect high-similarity rejected matches
- inspect appearance mismatches
- save annotated frames for manual verification

IMPORTANT:

This module NEVER changes identity matching decisions.

It only visualizes diagnostics produced by
GlobalIdentityManager.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import cv2

from ai_engine.schemas.track import Identity, Track


class IdentityVisualValidation:
    """
    Visual validation utility for GlobalIdentityManager V2.

    It receives:

        - original video
        - tracks
        - match diagnostics

    and generates annotated images showing suspicious matches.
    """

    def __init__(
        self,
        video_path: str,
        tracks: Iterable[Track],
        diagnostics: list[dict[str, Any]],
        output_dir: str = "media/identity_validation",
        identities: Optional[
            dict[str, Identity]
        ] = None,
    ) -> None:

        self.video_path = video_path

        self.tracks = list(tracks)

        self.diagnostics = list(
            diagnostics
        )

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.track_lookup: dict[
            int,
            Track,
        ] = {
            track.local_id: track
            for track in self.tracks
        }

        self.identities = (
            identities or {}
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def get_risky_matches(
        self,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:

        risky = []

        for item in self.diagnostics:

            if not item.get(
                "accepted",
                False,
            ):
                continue

            appearance = self._float(
                item.get(
                    "appearance_similarity",
                    -1.0,
                ),
                default=-1.0,
            )

            spatial = self._float(
                item.get(
                    "spatial_distance",
                    0.0,
                )
            )

            motion = self._float(
                item.get(
                    "motion_difference",
                    0.0,
                )
            )

            low_appearance = (
                appearance >= 0.0
                and appearance < 0.75
            )

            weak_motion_spatial = (
                spatial > 180.0
                and motion < 10.0
            )

            if (
                low_appearance
                or weak_motion_spatial
            ):
                risky.append(item)

        risky.sort(
            key=self._risk_sort_key,
            reverse=True,
        )

        if limit is not None:
            risky = risky[:limit]

        return risky

    # ========================================================

    def get_high_similarity_rejections(
        self,
        minimum_similarity: float = 0.85,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:

        results = []

        for item in self.diagnostics:

            if item.get("accepted", False):
                continue

            similarity = self._float(
                item.get(
                    "appearance_similarity",
                    -1.0,
                ),
                default=-1.0,
            )

            if similarity >= minimum_similarity:
                results.append(item)

        results.sort(
            key=lambda item: self._float(
                item.get(
                    "appearance_similarity",
                    0.0,
                )
            ),
            reverse=True,
        )

        if limit is not None:
            results = results[:limit]

        return results

    # ========================================================

    def get_appearance_mismatches(
        self,
        maximum_similarity: float = 0.70,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:

        results = []

        for item in self.diagnostics:

            if item.get("accepted", False):
                continue

            similarity = self._float(
                item.get(
                    "appearance_similarity",
                    -1.0,
                ),
                default=-1.0,
            )

            if (
                similarity >= 0.0
                and similarity < maximum_similarity
            ):
                results.append(item)

        results.sort(
            key=lambda item: self._float(
                item.get(
                    "appearance_similarity",
                    0.0,
                )
            )
        )

        if limit is not None:
            results = results[:limit]

        return results

    # ========================================================
    # FRAME EXTRACTION
    # ========================================================

    def extract_frame(
        self,
        frame_number: int,
    ):
        """
        Extract one zero-based frame from the source video.
        """

        if frame_number < 0:
            raise ValueError(
                "frame_number must be >= 0."
            )

        cap = cv2.VideoCapture(
            self.video_path
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: "
                f"{self.video_path}"
            )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number,
        )

        success, frame = cap.read()

        cap.release()

        if not success:
            raise RuntimeError(
                f"Could not read frame "
                f"{frame_number}"
            )

        return frame

    # ========================================================
    # DRAW TRACK
    # ========================================================

    def _draw_track(
        self,
        frame,
        track: Track,
        frame_number: int,
        label_prefix: str = "Track",
    ) -> None:

        observation = self._find_observation(
            track,
            frame_number,
        )

        if observation is None:
            return

        bbox = self._get_bbox(
            observation
        )

        if bbox is None:
            return

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            3,
        )

        label = (
            f"{label_prefix} "
            f"{track.local_id}"
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(25, y1 - 10),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # ========================================================
    # DRAW IDENTITY SOURCE
    # ========================================================

    def _draw_identity_source_tracks(
        self,
        frame,
        diagnostic: dict[str, Any],
        frame_number: int,
    ) -> None:

        source_ids = diagnostic.get(
            "identity_source_track_ids",
            [],
        )

        if not source_ids:

            identity_id = diagnostic.get(
                "identity_id"
            )

            identity = self.identities.get(
                identity_id
            )

            if identity is not None:
                source_ids = (
                    identity.source_track_ids
                )

        for source_id in source_ids:

            try:
                source_id = int(source_id)
            except (
                TypeError,
                ValueError,
            ):
                continue

            source_track = self.track_lookup.get(
                source_id
            )

            if source_track is None:
                continue

            self._draw_track(
                frame,
                source_track,
                frame_number,
                label_prefix="Identity source track",
            )

    # ========================================================
    # DRAW DIAGNOSTIC
    # ========================================================

    def _draw_diagnostic(
        self,
        frame,
        diagnostic: dict[str, Any],
        frame_number: int,
    ) -> None:

        accepted = bool(
            diagnostic.get(
                "accepted",
                False,
            )
        )

        appearance = self._float(
            diagnostic.get(
                "appearance_similarity",
                -1.0,
            ),
            default=-1.0,
        )

        score = self._float(
            diagnostic.get(
                "score",
                0.0,
            )
        )

        temporal = diagnostic.get(
            "temporal_gap",
            0,
        )

        spatial = self._float(
            diagnostic.get(
                "spatial_distance",
                0.0,
            )
        )

        motion = self._float(
            diagnostic.get(
                "motion_difference",
                0.0,
            )
        )

        track_id = diagnostic.get(
            "track_id"
        )

        identity_id = diagnostic.get(
            "identity_id"
        )

        reason = diagnostic.get(
            "reason",
            "",
        )

        team_match = diagnostic.get(
            "team_match"
        )

        jersey_match = diagnostic.get(
            "jersey_match"
        )

        status = (
            "ACCEPTED"
            if accepted
            else "REJECTED"
        )

        header = (
            f"Frame {frame_number} | "
            f"Track {track_id} -> "
            f"{identity_id} | "
            f"{status}"
        )

        overlay_height = 210

        cv2.rectangle(
            frame,
            (10, 10),
            (950, overlay_height),
            (20, 20, 20),
            -1,
        )

        cv2.putText(
            frame,
            header,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        appearance_text = (
            f"{appearance:.3f}"
            if appearance >= 0.0
            else "N/A"
        )

        lines = [
            f"Score: {score:.3f}",
            f"Appearance: {appearance_text}",
            f"Temporal gap: {temporal}",
            f"Spatial distance: {spatial:.2f}",
            f"Motion difference: {motion:.2f}",
            f"Team match: {team_match}",
            f"Jersey match: {jersey_match}",
            f"Reason: {reason}",
        ]

        y = 68

        for line in lines:

            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            y += 18

    # ========================================================
    # SAVE DIAGNOSTIC FRAME
    # ========================================================

    def save_diagnostic_frame(
        self,
        diagnostic: dict[str, Any],
        prefix: str = "match",
    ) -> Optional[str]:

        track_id = diagnostic.get(
            "track_id"
        )

        if track_id is None:
            return None

        try:
            track_id = int(track_id)
        except (
            TypeError,
            ValueError,
        ):
            return None

        track = self.track_lookup.get(
            track_id
        )

        if track is None:
            return None

        frame_number = (
            self._get_reference_frame(
                track,
                diagnostic,
            )
        )

        frame = self.extract_frame(
            frame_number
        )

        # Draw candidate track.
        self._draw_track(
            frame,
            track,
            frame_number,
            label_prefix="Candidate track",
        )

        # Draw existing identity source fragments.
        self._draw_identity_source_tracks(
            frame,
            diagnostic,
            frame_number,
        )

        # Draw diagnostic information.
        self._draw_diagnostic(
            frame,
            diagnostic,
            frame_number,
        )

        identity_id = diagnostic.get(
            "identity_id",
            "unknown",
        )

        filename = (
            f"{prefix}_"
            f"track_{track_id}_"
            f"{identity_id}_"
            f"frame_{frame_number}.jpg"
        )

        output_path = (
            self.output_dir
            / filename
        )

        success = cv2.imwrite(
            str(output_path),
            frame,
        )

        if not success:
            return None

        return str(output_path)

    # ========================================================
    # REPORTS
    # ========================================================

    def generate_risk_report(
        self,
        limit: Optional[int] = None,
    ) -> list[str]:

        risky = self.get_risky_matches(
            limit=limit
        )

        paths = []

        for diagnostic in risky:

            path = self.save_diagnostic_frame(
                diagnostic,
                prefix="RISK",
            )

            if path:
                paths.append(path)

        return paths

    def generate_high_similarity_report(
        self,
        minimum_similarity: float = 0.85,
        limit: Optional[int] = None,
    ) -> list[str]:

        rejected = (
            self.get_high_similarity_rejections(
                minimum_similarity=minimum_similarity,
                limit=limit,
            )
        )

        paths = []

        for diagnostic in rejected:

            path = self.save_diagnostic_frame(
                diagnostic,
                prefix="HIGH_SIM_REJECT",
            )

            if path:
                paths.append(path)

        return paths

    def generate_appearance_mismatch_report(
        self,
        maximum_similarity: float = 0.70,
        limit: Optional[int] = None,
    ) -> list[str]:

        mismatches = (
            self.get_appearance_mismatches(
                maximum_similarity=maximum_similarity,
                limit=limit,
            )
        )

        paths = []

        for diagnostic in mismatches:

            path = self.save_diagnostic_frame(
                diagnostic,
                prefix="APPEARANCE_MISMATCH",
            )

            if path:
                paths.append(path)

        return paths

    # ========================================================
    # SUMMARY
    # ========================================================

    def print_summary(self) -> None:

        risky = self.get_risky_matches()

        high_similarity = (
            self.get_high_similarity_rejections()
        )

        mismatches = (
            self.get_appearance_mismatches()
        )

        print()
        print("=" * 80)
        print("PHASE 7 VISUAL VALIDATION")
        print("=" * 80)

        print(
            "Diagnostics:",
            len(self.diagnostics),
        )

        print(
            "Risky accepted matches:",
            len(risky),
        )

        print(
            "High-similarity rejected:",
            len(high_similarity),
        )

        print(
            "Appearance mismatches:",
            len(mismatches),
        )

        print(
            "Output directory:",
            self.output_dir,
        )

        print("=" * 80)

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _risk_sort_key(
        item: dict[str, Any],
    ):

        appearance = IdentityVisualValidation._float(
            item.get(
                "appearance_similarity",
                1.0,
            ),
            default=1.0,
        )

        spatial = IdentityVisualValidation._float(
            item.get(
                "spatial_distance",
                0.0,
            )
        )

        return (
            1.0 - appearance,
            spatial,
        )

    @staticmethod
    def _find_observation(
        track: Track,
        frame_number: int,
    ):

        for observation in track.observations:

            observation_frame = getattr(
                observation,
                "frame_index",
                None,
            )

            if observation_frame == frame_number:
                return observation

        if not track.observations:
            return None

        valid = []

        for observation in track.observations:

            observation_frame = getattr(
                observation,
                "frame_index",
                None,
            )

            if observation_frame is None:
                continue

            valid.append(
                (
                    abs(
                        observation_frame
                        - frame_number
                    ),
                    observation,
                )
            )

        if not valid:
            return None

        valid.sort(
            key=lambda x: x[0]
        )

        return valid[0][1]

    @staticmethod
    def _get_bbox(
        observation,
    ):

        bbox = getattr(
            observation,
            "bbox",
            None,
        )

        if bbox is None:
            return None

        if hasattr(
            bbox,
            "x1",
        ):

            return (
                int(bbox.x1),
                int(bbox.y1),
                int(bbox.x2),
                int(bbox.y2),
            )

        if isinstance(
            bbox,
            (tuple, list),
        ) and len(bbox) >= 4:

            return (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
            )

        return None

    @staticmethod
    def _get_reference_frame(
        track: Track,
        diagnostic: dict[str, Any],
    ) -> int:
        """
        Prefer the beginning of the candidate fragment.

        If possible, use the frame immediately after the
        identity's last frame. This is the most useful frame
        for inspecting the transition between fragments.
        """

        identity_last_frame = diagnostic.get(
            "identity_last_frame"
        )

        if (
            identity_last_frame is not None
            and track.first_frame is not None
        ):

            candidate = (
                int(identity_last_frame)
                + 1
            )

            if (
                candidate
                >= track.first_frame
                and candidate
                <= track.last_frame
            ):
                return candidate

        if track.first_frame is not None:
            return int(
                track.first_frame
            )

        if track.observations:

            frame_number = getattr(
                track.observations[0],
                "frame_index",
                None,
            )

            if frame_number is not None:
                return int(frame_number)

        return 0

    @staticmethod
    def _float(
        value,
        default: float = 0.0,
    ) -> float:

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return default