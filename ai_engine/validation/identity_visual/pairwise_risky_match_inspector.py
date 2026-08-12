"""
Pairwise Risky Identity Match Visual Inspector.

Consumes the existing GlobalIdentityManager output and
IdentityVisualValidator results.

It does NOT perform identity matching.

For every HIGH/CRITICAL accepted identity match:

    previous fragment -> new fragment

is rendered side-by-side.

The inspector prefers the exact track pair recorded by the
validator. If the validator does not expose the previous
track ID, it falls back to finding the closest previous
fragment belonging to the same global identity.

Generated evidence contains:

- previous track ID
- previous frame
- new track ID
- new frame
- global identity
- appearance similarity
- temporal gap
- spatial distance
- motion difference
- match score
- risk level
- validation warnings
- matcher reason
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ai_engine.validation.identity_visual.identity_visual_validator import (
    IdentityVisualValidator,
)

logger = logging.getLogger(__name__)


class PairwiseRiskyMatchInspector:
    """
    Generate pairwise visual evidence for risky identity merges.

    This class is read-only with respect to:

        manager
        identities
        tracks
        observations
    """

    VERSION = "v2.0"

    def __init__(
        self,
        output_dir: str | Path = "media/identity_pairwise_inspection",
        crop_padding: int = 80,
        crop_height: int = 360,
        jpeg_quality: int = 92,
    ) -> None:

        self.output_dir = Path(output_dir)

        self.crop_padding = int(crop_padding)
        self.crop_height = int(crop_height)
        self.jpeg_quality = int(jpeg_quality)

        if self.crop_padding < 0:
            raise ValueError(
                "crop_padding must be >= 0."
            )

        if self.crop_height < 100:
            raise ValueError(
                "crop_height must be >= 100."
            )

        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(
                "jpeg_quality must be between 1 and 100."
            )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def generate(
        self,
        manager: Any,
        video_path: str | Path,
        *,
        max_matches: Optional[int] = None,
        create_contact_sheet: bool = True,
        contact_sheet_columns: int = 2,
    ) -> dict[str, Any]:

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        validator = IdentityVisualValidator()

        # --------------------------------------------------------
        # Validate existing manager output.
        # --------------------------------------------------------

        validation_report = validator.validate_manager(
            manager
        )

        risky_matches = validator.get_risky_matches()

        # --------------------------------------------------------
        # Only accepted matches should be visually inspected.
        # --------------------------------------------------------

        risky_matches = [
            item
            for item in risky_matches
            if self._is_accepted_match(item)
        ]

        # --------------------------------------------------------
        # Sort:
        #
        # CRITICAL
        # HIGH
        # MEDIUM
        # LOW
        #
        # For equal risk:
        # lower appearance similarity first.
        # --------------------------------------------------------

        risk_order = {
            "CRITICAL": 0,
            "HIGH": 1,
            "MEDIUM": 2,
            "LOW": 3,
        }

        risky_matches.sort(
            key=lambda item: (
                risk_order.get(
                    str(
                        item.get(
                            "risk_level",
                            "LOW",
                        )
                    ).upper(),
                    99,
                ),
                self._float(
                    item.get(
                        "appearance_similarity"
                    ),
                    -1.0,
                ),
                -self._float(
                    item.get(
                        "spatial_distance"
                    ),
                    0.0,
                ),
            )
        )

        total_risky_matches = len(
            risky_matches
        )

        if max_matches is not None:
            risky_matches = risky_matches[
                :max_matches
            ]

        tracks = (
            getattr(
                manager,
                "tracks",
                {},
            )
            or {}
        )

        identities = (
            getattr(
                manager,
                "identities",
                {},
            )
            or {}
        )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise ValueError(
                f"Unable to open video: {video_path}"
            )

        evidence: list[dict[str, Any]] = []

        try:

            for index, risky in enumerate(
                risky_matches,
                start=1,
            ):

                item = self._generate_pair(
                    risky=risky,
                    tracks=tracks,
                    identities=identities,
                    capture=capture,
                    index=index,
                )

                if item is not None:
                    evidence.append(item)

        finally:
            capture.release()

        # --------------------------------------------------------
        # Manifest.
        # --------------------------------------------------------

        manifest = {
            "version": self.VERSION,
            "video_path": str(video_path),

            "validation_status": validation_report.get(
                "status"
            ),

            "risky_matches_detected": total_risky_matches,

            "risky_matches_selected": len(
                risky_matches
            ),

            "evidence_generated": len(
                evidence
            ),

            "evidence": evidence,
        }

        manifest_path = (
            self.output_dir
            / "pairwise_risky_manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        manifest["manifest_path"] = str(
            manifest_path
        )

        # --------------------------------------------------------
        # Contact sheet.
        # --------------------------------------------------------

        if (
            create_contact_sheet
            and evidence
        ):

            contact_sheet_path = (
                self.output_dir
                / "identity_pairwise_contact_sheet.jpg"
            )

            self._create_contact_sheet(
                evidence,
                contact_sheet_path,
                columns=contact_sheet_columns,
            )

            manifest[
                "contact_sheet"
            ] = str(
                contact_sheet_path
            )

            manifest_path.write_text(
                json.dumps(
                    manifest,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        return manifest

    # ============================================================
    # PAIR GENERATION
    # ============================================================

    def _generate_pair(
        self,
        *,
        risky: dict[str, Any],
        tracks: dict[Any, Any],
        identities: dict[Any, Any],
        capture: cv2.VideoCapture,
        index: int,
    ) -> Optional[dict[str, Any]]:

        identity_id = self._get_identity_id(
            risky
        )

        if not identity_id:
            logger.warning(
                "Risky match has no identity ID: %s",
                risky,
            )
            return None

        identity = self._get_identity(
            identities,
            identity_id,
        )

        if identity is None:
            logger.warning(
                "Identity %s not found.",
                identity_id,
            )
            return None

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # Prefer the exact NEW track recorded by the validator.
        # --------------------------------------------------------

        new_track_id = self._get_new_track_id(
            risky
        )

        if new_track_id is None:
            logger.warning(
                "Risky match has no new track ID: %s",
                risky,
            )
            return None

        new_track = self._get_track(
            tracks,
            new_track_id,
        )

        if new_track is None:
            logger.warning(
                "New track %s not found.",
                new_track_id,
            )
            return None

        new_observation = (
            self._first_observation(
                new_track
            )
        )

        if new_observation is None:
            logger.warning(
                "New track %s has no observations.",
                new_track_id,
            )
            return None

        new_frame = int(
            new_observation.frame_index
        )

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # First try to use the EXACT previous track stored
        # by the validator.
        #
        # Only fall back to temporal inference if unavailable.
        # --------------------------------------------------------

        previous_track_id = (
            self._get_previous_track_id(
                risky
            )
        )

        previous_track = None

        if previous_track_id is not None:

            previous_track = self._get_track(
                tracks,
                previous_track_id,
            )

            if previous_track is None:
                logger.warning(
                    "Validator specified previous track %s "
                    "but it was not found. Falling back.",
                    previous_track_id,
                )

        # --------------------------------------------------------
        # Fallback:
        #
        # Find the closest previous fragment belonging to the
        # same global identity.
        # --------------------------------------------------------

        if previous_track is None:

            previous_track = (
                self._find_previous_fragment(
                    identity=identity,
                    tracks=tracks,
                    new_track=new_track,
                )
            )

        if previous_track is None:
            logger.warning(
                "No previous fragment found for "
                "%s -> track %s.",
                identity_id,
                new_track_id,
            )
            return None

        previous_observation = (
            self._last_observation(
                previous_track
            )
        )

        if previous_observation is None:
            logger.warning(
                "Previous track %s has no observations.",
                previous_track.local_id,
            )
            return None

        previous_frame = int(
            previous_observation.frame_index
        )

        # --------------------------------------------------------
        # Sanity check.
        # --------------------------------------------------------

        if previous_frame >= new_frame:

            logger.warning(
                "Invalid pair ordering: "
                "previous frame=%s, new frame=%s "
                "for identity=%s.",
                previous_frame,
                new_frame,
                identity_id,
            )

            return None

        # --------------------------------------------------------
        # Read video frames.
        # --------------------------------------------------------

        previous_image = self._read_frame(
            capture,
            previous_frame,
        )

        new_image = self._read_frame(
            capture,
            new_frame,
        )

        if (
            previous_image is None
            or new_image is None
        ):
            logger.warning(
                "Unable to read pair frames: "
                "%s / %s",
                previous_frame,
                new_frame,
            )
            return None

        # --------------------------------------------------------
        # Crop observations.
        # --------------------------------------------------------

        previous_crop = (
            self._crop_observation(
                previous_image,
                previous_observation,
            )
        )

        new_crop = (
            self._crop_observation(
                new_image,
                new_observation,
            )
        )

        if (
            previous_crop is None
            or new_crop is None
        ):
            logger.warning(
                "Unable to crop pair: "
                "%s -> %s",
                previous_track.local_id,
                new_track.local_id,
            )
            return None

        # --------------------------------------------------------
        # Normalize heights.
        # --------------------------------------------------------

        previous_crop = (
            self._resize_height(
                previous_crop,
                self.crop_height,
            )
        )

        new_crop = (
            self._resize_height(
                new_crop,
                self.crop_height,
            )
        )

        # --------------------------------------------------------
        # Add headers.
        # --------------------------------------------------------

        previous_panel = (
            self._add_panel_header(
                previous_crop,
                title="PREVIOUS FRAGMENT",
                track_id=int(
                    previous_track.local_id
                ),
                frame_index=previous_frame,
            )
        )

        new_panel = (
            self._add_panel_header(
                new_crop,
                title="NEW FRAGMENT",
                track_id=int(
                    new_track.local_id
                ),
                frame_index=new_frame,
            )
        )

        # --------------------------------------------------------
        # Build side-by-side image.
        # --------------------------------------------------------

        pair = self._side_by_side(
            previous_panel,
            new_panel,
        )

        # --------------------------------------------------------
        # Add diagnostic footer.
        # --------------------------------------------------------

        pair = self._add_footer(
            pair,
            risky=risky,
            identity=identity_id,
            previous_track_id=int(
                previous_track.local_id
            ),
            previous_frame=previous_frame,
            new_track_id=int(
                new_track.local_id
            ),
            new_frame=new_frame,
        )

        # --------------------------------------------------------
        # Save.
        # --------------------------------------------------------

        identity_dir = (
            self.output_dir
            / (
                "identity_"
                + self._safe_name(
                    identity_id
                )
            )
        )

        identity_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        risk_level = str(
            risky.get(
                "risk_level",
                "UNKNOWN",
            )
        ).lower()

        filename = (
            f"{index:03d}_"
            f"{self._safe_name(identity_id)}_"
            f"prev_{previous_track.local_id}_"
            f"new_{new_track.local_id}_"
            f"{risk_level}.jpg"
        )

        image_path = (
            identity_dir
            / filename
        )

        ok = cv2.imwrite(
            str(image_path),
            pair,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                self.jpeg_quality,
            ],
        )

        if not ok:
            logger.warning(
                "Failed to write %s",
                image_path,
            )
            return None

        # --------------------------------------------------------
        # Manifest entry.
        # --------------------------------------------------------

        return {
            "index": index,

            "identity_id": identity_id,

            "previous_track_id": int(
                previous_track.local_id
            ),

            "previous_frame": previous_frame,

            "new_track_id": int(
                new_track.local_id
            ),

            "new_frame": new_frame,

            "temporal_gap": self._int(
                risky.get(
                    "temporal_gap"
                ),
                new_frame
                - previous_frame
                - 1,
            ),

            "appearance_similarity": self._float(
                risky.get(
                    "appearance_similarity"
                ),
                -1.0,
            ),

            "spatial_distance": self._float(
                risky.get(
                    "spatial_distance"
                ),
                0.0,
            ),

            "motion_difference": self._float(
                risky.get(
                    "motion_difference"
                ),
                0.0,
            ),

            "score": self._float(
                risky.get(
                    "score"
                ),
                0.0,
            ),

            "risk_level": risky.get(
                "risk_level"
            ),

            "status": risky.get(
                "status"
            ),

            "matched": risky.get(
                "matched"
            ),

            "reason": risky.get(
                "reason"
            ),

            "validation_warnings": list(
                risky.get(
                    "validation_warnings",
                    [],
                )
                or []
            ),

            "matcher_reasons": list(
                risky.get(
                    "matcher_reasons",
                    [],
                )
                or []
            ),

            "image_path": str(
                image_path
            ),
        }

    # ============================================================
    # RISKY MATCH HELPERS
    # ============================================================

    @staticmethod
    def _is_accepted_match(
        risky: dict[str, Any],
    ) -> bool:

        matched = risky.get(
            "matched"
        )

        if matched is True:
            return True

        status = str(
            risky.get(
                "status",
                "",
            )
        ).upper()

        if status in {
            "ACCEPTED",
            "MATCHED",
        }:
            return True

        # If the validator has no explicit matched/status field,
        # the presence of a match score can be enough to identify
        # an accepted match in some versions.
        if (
            "score" in risky
            and risky.get("score") is not None
        ):
            return True

        return False

    @staticmethod
    def _get_identity_id(
        risky: dict[str, Any],
    ) -> Optional[str]:

        for key in (
            "identity_id",
            "global_identity_id",
            "global_id",
            "gid",
        ):

            value = risky.get(key)

            if value is not None:
                return str(value)

        return None

    @staticmethod
    def _get_new_track_id(
        risky: dict[str, Any],
    ) -> Optional[Any]:

        for key in (
            "new_track_id",
            "track_id",
            "matched_track_id",
            "source_track_id",
        ):

            value = risky.get(key)

            if value is not None:
                return value

        return None

    @staticmethod
    def _get_previous_track_id(
        risky: dict[str, Any],
    ) -> Optional[Any]:

        for key in (
            "previous_track_id",
            "prev_track_id",
            "source_previous_track_id",
            "previous_fragment_track_id",
            "existing_track_id",
            "identity_track_id",
        ):

            value = risky.get(key)

            if value is not None:
                return value

        return None

    # ============================================================
    # IDENTITY HELPERS
    # ============================================================

    @staticmethod
    def _get_identity(
        identities: dict[Any, Any],
        identity_id: str,
    ) -> Optional[Any]:

        identity = identities.get(
            identity_id
        )

        if identity is not None:
            return identity

        try:
            identity = identities.get(
                int(identity_id)
            )

            if identity is not None:
                return identity

        except (
            TypeError,
            ValueError,
        ):
            pass

        return None

    # ============================================================
    # PREVIOUS FRAGMENT FALLBACK
    # ============================================================

    @staticmethod
    def _find_previous_fragment(
        *,
        identity: Any,
        tracks: dict[Any, Any],
        new_track: Any,
    ) -> Optional[Any]:

        source_track_ids = list(
            getattr(
                identity,
                "source_track_ids",
                [],
            )
            or []
        )

        new_first_frame = getattr(
            new_track,
            "first_frame",
            None,
        )

        if new_first_frame is None:

            first_obs = (
                PairwiseRiskyMatchInspector
                ._first_observation(
                    new_track
                )
            )

            if first_obs is None:
                return None

            new_first_frame = (
                first_obs.frame_index
            )

        candidates = []

        for track_id in source_track_ids:

            if int(track_id) == int(
                new_track.local_id
            ):
                continue

            previous = (
                PairwiseRiskyMatchInspector
                ._get_track(
                    tracks,
                    track_id,
                )
            )

            if previous is None:
                continue

            last_frame = getattr(
                previous,
                "last_frame",
                None,
            )

            if last_frame is None:

                last_obs = (
                    PairwiseRiskyMatchInspector
                    ._last_observation(
                        previous
                    )
                )

                if last_obs is None:
                    continue

                last_frame = (
                    last_obs.frame_index
                )

            if int(last_frame) < int(
                new_first_frame
            ):

                candidates.append(
                    (
                        int(last_frame),
                        previous,
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return candidates[0][1]

    # ============================================================
    # TRACK HELPERS
    # ============================================================

    @staticmethod
    def _get_track(
        tracks: dict[Any, Any],
        track_id: Any,
    ) -> Optional[Any]:

        track = tracks.get(
            track_id
        )

        if track is not None:
            return track

        track = tracks.get(
            str(track_id)
        )

        if track is not None:
            return track

        try:
            return tracks.get(
                int(track_id)
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _first_observation(
        track: Any,
    ) -> Optional[Any]:

        observations = list(
            getattr(
                track,
                "observations",
                [],
            )
            or []
        )

        if not observations:
            return None

        return min(
            observations,
            key=lambda obs: int(
                obs.frame_index
            ),
        )

    @staticmethod
    def _last_observation(
        track: Any,
    ) -> Optional[Any]:

        observations = list(
            getattr(
                track,
                "observations",
                [],
            )
            or []
        )

        if not observations:
            return None

        return max(
            observations,
            key=lambda obs: int(
                obs.frame_index
            ),
        )

    # ============================================================
    # VIDEO
    # ============================================================

    @staticmethod
    def _read_frame(
        capture: cv2.VideoCapture,
        frame_index: int,
    ) -> Optional[Any]:

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index),
        )

        ok, frame = capture.read()

        if not ok:
            return None

        return frame

    # ============================================================
    # CROPPING
    # ============================================================

    def _crop_observation(
        self,
        frame: Any,
        observation: Any,
    ) -> Optional[Any]:

        if observation is None:
            return None

        bbox = getattr(
            observation,
            "bbox",
            None,
        )

        if bbox is None:
            return None

        x1 = int(
            max(
                0,
                bbox.x1
                - self.crop_padding,
            )
        )

        y1 = int(
            max(
                0,
                bbox.y1
                - self.crop_padding,
            )
        )

        x2 = int(
            min(
                frame.shape[1],
                bbox.x2
                + self.crop_padding,
            )
        )

        y2 = int(
            min(
                frame.shape[0],
                bbox.y2
                + self.crop_padding,
            )
        )

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[
            y1:y2,
            x1:x2,
        ].copy()

    @staticmethod
    def _resize_height(
        image: Any,
        height: int,
    ) -> Any:

        if image.shape[0] == height:
            return image

        scale = (
            height
            / image.shape[0]
        )

        width = max(
            1,
            int(
                image.shape[1]
                * scale
            ),
        )

        return cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )

    # ============================================================
    # DRAWING
    # ============================================================

    @staticmethod
    def _add_panel_header(
        image: Any,
        *,
        title: str,
        track_id: int,
        frame_index: int,
    ) -> Any:

        header_height = 65

        canvas = cv2.copyMakeBorder(
            image,
            header_height,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(20, 20, 20),
        )

        cv2.putText(
            canvas,
            title,
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            (
                f"Track {track_id} | "
                f"frame={frame_index}"
            ),
            (12, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return canvas

    def _add_footer(
        self,
        image: Any,
        *,
        risky: dict[str, Any],
        identity: str,
        previous_track_id: int,
        previous_frame: int,
        new_track_id: int,
        new_frame: int,
    ) -> Any:

        footer_height = 235

        canvas = cv2.copyMakeBorder(
            image,
            0,
            footer_height,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(10, 10, 10),
        )

        lines = [
            (
                f"{identity} | "
                f"risk={risky.get('risk_level', 'N/A')} | "
                f"status={risky.get('status', 'N/A')}"
            ),

            (
                f"Track {previous_track_id} "
                f"@ frame {previous_frame}"
                f"  ->  "
                f"Track {new_track_id} "
                f"@ frame {new_frame}"
            ),

            (
                f"appearance="
                f"{self._fmt(risky.get('appearance_similarity'))}"
                f" | score="
                f"{self._fmt(risky.get('score'))}"
            ),

            (
                f"temporal_gap="
                f"{risky.get('temporal_gap', 'N/A')}"
                f" | spatial="
                f"{self._fmt(risky.get('spatial_distance'))}"
                f" | motion="
                f"{self._fmt(risky.get('motion_difference'))}"
            ),

            (
                "warnings="
                + self._join_values(
                    risky.get(
                        "validation_warnings",
                        [],
                    )
                )
            ),

            (
                "reason="
                + str(
                    risky.get(
                        "reason",
                        "",
                    )
                )
            ),

            (
                "matcher="
                + self._join_values(
                    risky.get(
                        "matcher_reasons",
                        [],
                    )
                )
            ),
        ]

        y = image.shape[0] + 27

        for line in lines:

            # Prevent huge diagnostic strings from
            # destroying the visual evidence.
            line = str(line)[:220]

            cv2.putText(
                canvas,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            y += 30

        return canvas

    @staticmethod
    def _side_by_side(
        left: Any,
        right: Any,
    ) -> Any:

        height = max(
            left.shape[0],
            right.shape[0],
        )

        if left.shape[0] < height:

            left = cv2.copyMakeBorder(
                left,
                0,
                height
                - left.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        if right.shape[0] < height:

            right = cv2.copyMakeBorder(
                right,
                0,
                height
                - right.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        return cv2.hconcat(
            [
                left,
                right,
            ]
        )

    # ============================================================
    # CONTACT SHEET
    # ============================================================

    def _create_contact_sheet(
        self,
        evidence: list[dict[str, Any]],
        output_path: Path,
        *,
        columns: int = 2,
    ) -> None:

        if not evidence:
            return

        if columns < 1:
            raise ValueError(
                "contact_sheet_columns must be >= 1."
            )

        images = []

        for item in evidence:

            image = cv2.imread(
                item["image_path"]
            )

            if image is not None:
                images.append(
                    image
                )

        if not images:
            return

        cell_width = max(
            image.shape[1]
            for image in images
        )

        cell_height = max(
            image.shape[0]
            for image in images
        )

        rows = (
            len(images)
            + columns
            - 1
        ) // columns

        row_images = []

        for row in range(rows):

            cells = []

            for col in range(columns):

                index = (
                    row * columns
                    + col
                )

                if index < len(images):

                    cell = (
                        self._fit_to_cell(
                            images[index],
                            cell_width,
                            cell_height,
                        )
                    )

                else:

                    cell = (
                        self._blank_cell(
                            cell_width,
                            cell_height,
                        )
                    )

                cells.append(
                    cell
                )

            row_images.append(
                cv2.hconcat(
                    cells
                )
            )

        sheet = cv2.vconcat(
            row_images
        )

        sheet = cv2.copyMakeBorder(
            sheet,
            10,
            10,
            10,
            10,
            cv2.BORDER_CONSTANT,
            value=(30, 30, 30),
        )

        cv2.imwrite(
            str(output_path),
            sheet,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                self.jpeg_quality,
            ],
        )

    @staticmethod
    def _fit_to_cell(
        image: Any,
        width: int,
        height: int,
    ) -> Any:

        return cv2.resize(
            image,
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _blank_cell(
        width: int,
        height: int,
    ) -> Any:

        return np.full(
            (
                height,
                width,
                3,
            ),
            30,
            dtype=np.uint8,
        )

    # ============================================================
    # UTILITIES
    # ============================================================

    @staticmethod
    def _safe_name(
        value: Any,
    ) -> str:

        return "".join(
            char
            if (
                char.isalnum()
                or char in (
                    "_",
                    "-",
                )
            )
            else "_"
            for char in str(value)
        )

    @staticmethod
    def _fmt(
        value: Any,
    ) -> str:

        if value is None:
            return "N/A"

        try:
            return f"{float(value):.3f}"

        except (
            TypeError,
            ValueError,
        ):
            return str(value)

    @staticmethod
    def _float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        if value is None:
            return default

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _int(
        value: Any,
        default: int = 0,
    ) -> int:

        if value is None:
            return default

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _join_values(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return ", ".join(
                str(item)
                for item in value
            )

        return str(value)