"""
Visual Evidence Generator V5.3.1.

Read-only evidence-generation layer built on top of
IdentityVisualValidator V5.2.

V5.3.1 Step 1 changes:

- Adds max_observation_frame_gap configuration.
- Prevents identity observation lookup from using observations that
  are too far away from the requested frame.
- Prevents track observation lookup from using observations that
  are too far away from the requested frame.

V5.3.1 Step 2 changes:

- Records observation_frame_index.
- Records observation_frame_gap.
- Records observation_source.
- Displays observation provenance in rendered evidence.
- Applies provenance tracking to both match evidence and
  identity summary evidence.

V5.3.1 Step 3 changes:

- Makes evidence-frame selection diagnostic-risk aware.
- Prioritizes the diagnostic frame.
- Prioritizes observations closest to the diagnostic frame.
- Preserves temporal coverage as a fallback.
- CRITICAL/HIGH risk evidence receives the strongest diagnostic
  proximity priority.
- REVIEW evidence also prioritizes diagnostic proximity.
- Does not modify the Step 1 observation-gap rule.
- Does not modify the Step 2 observation-provenance behavior.

Responsibilities:

- Run V5.2 validation.
- Locate useful observations belonging to an identity/track.
- Select representative evidence frames.
- Read frames from the source video.
- Draw identity/track bounding boxes and evidence metadata.
- Write evidence images and a JSON manifest.
- Never modify Identity, Track, TrackObservation, or manager state.

V5.3 deliberately does NOT perform identity matching.
It consumes the already-produced GlobalIdentityManager output.

Dependencies:
OpenCV (cv2)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2

from ai_engine.validation.identity_visual.identity_visual_validator import (
    IdentityVisualValidator,
    VisualValidationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class EvidenceFrame:
    """Description of one generated evidence frame."""

    identity_id: str
    frame_index: int
    image_path: str
    evidence_type: str
    status: str
    risk_level: str

    track_id: Optional[int] = None
    source_track_ids: list[int] = field(default_factory=list)

    bbox: Optional[list[float]] = None

    # V5.3.1 Step 2: observation provenance
    observation_frame_index: Optional[int] = None
    observation_frame_gap: Optional[int] = None
    observation_source: Optional[str] = None

    appearance_similarity: float = -1.0
    temporal_gap: int = 0
    spatial_distance: float = 0.0
    motion_difference: float = 0.0

    team: Optional[str] = None
    jersey_number: Optional[int] = None

    warnings: list[str] = field(default_factory=list)
    matcher_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "track_id": self.track_id,
            "source_track_ids": list(self.source_track_ids),
            "frame_index": self.frame_index,
            "image_path": self.image_path,
            "evidence_type": self.evidence_type,
            "status": self.status,
            "risk_level": self.risk_level,
            "bbox": self.bbox,

            # V5.3.1 Step 2: observation provenance
            "observation_frame_index": self.observation_frame_index,
            "observation_frame_gap": self.observation_frame_gap,
            "observation_source": self.observation_source,

            "appearance_similarity": self.appearance_similarity,
            "temporal_gap": self.temporal_gap,
            "spatial_distance": self.spatial_distance,
            "motion_difference": self.motion_difference,
            "team": self.team,
            "jersey_number": self.jersey_number,
            "warnings": list(self.warnings),
            "matcher_reasons": list(self.matcher_reasons),
        }


class VisualEvidenceGenerator:
    """
    V5.3.1 visual evidence generator.

    The generator is intentionally read-only with respect to the manager
    and its Identity/Track objects. It creates only external evidence files.
    """

    VERSION = "v5.3.1"

    def __init__(
        self,
        output_dir: str | Path = "media/validation_evidence",
        validator: Optional[IdentityVisualValidator] = None,
        max_evidence_per_match: int = 2,
        max_evidence_per_identity: int = 4,
        jpeg_quality: int = 92,
        max_observation_frame_gap: int = 5,
    ) -> None:
        if max_evidence_per_match < 1:
            raise ValueError(
                "max_evidence_per_match must be >= 1."
            )

        if max_evidence_per_identity < 1:
            raise ValueError(
                "max_evidence_per_identity must be >= 1."
            )

        if not 1 <= jpeg_quality <= 100:
            raise ValueError(
                "jpeg_quality must be between 1 and 100."
            )

        if max_observation_frame_gap < 0:
            raise ValueError(
                "max_observation_frame_gap must be >= 0."
            )

        self.output_dir = Path(output_dir)
        self.validator = validator or IdentityVisualValidator()

        self.max_evidence_per_match = int(
            max_evidence_per_match
        )
        self.max_evidence_per_identity = int(
            max_evidence_per_identity
        )
        self.jpeg_quality = int(jpeg_quality)

        # V5.3.1 Step 1:
        # Maximum allowed distance between the requested frame and
        # the observation frame used for visual evidence.
        self.max_observation_frame_gap = int(
            max_observation_frame_gap
        )

        self.evidence: list[EvidenceFrame] = []

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def generate(
        self,
        manager: Any,
        video_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        include_pass: bool = False,
        include_review: bool = True,
        include_high_risk: bool = True,
        include_invalid: bool = True,
        write_manifest: bool = True,
    ) -> dict[str, Any]:
        """
        Generate visual evidence for validation results.

        By default:
            PASS      -> skipped
            REVIEW    -> included
            HIGH_RISK -> included
            INVALID   -> included

        Set include_pass=True to include healthy matches as evidence too.
        """
        self.evidence = []

        validation_report = self.validator.validate_manager(
            manager
        )

        destination = (
            Path(output_dir)
            if output_dir
            else self.output_dir
        )

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        capture = self._open_video(video_path)

        try:
            match_results = self.validator.match_results

            selected_matches = [
                result
                for result in match_results
                if self._status_selected(
                    result.status,
                    include_pass=include_pass,
                    include_review=include_review,
                    include_high_risk=include_high_risk,
                    include_invalid=include_invalid,
                )
            ]

            tracks = (
                getattr(manager, "tracks", {})
                or {}
            )

            identities = (
                getattr(manager, "identities", {})
                or {}
            )

            per_identity_count: dict[str, int] = {}

            for result in selected_matches:
                identity = identities.get(
                    result.identity_id
                )

                if identity is None:
                    logger.warning(
                        "Identity %s not found for evidence result.",
                        result.identity_id,
                    )
                    continue

                current_count = per_identity_count.get(
                    result.identity_id,
                    0,
                )

                if (
                    current_count
                    >= self.max_evidence_per_identity
                ):
                    continue

                frames = self._select_match_frames(
                    result,
                    identity,
                    tracks,
                    capture,
                )

                for frame_info in frames[
                    : self.max_evidence_per_match
                ]:
                    if (
                        current_count
                        >= self.max_evidence_per_identity
                    ):
                        break

                    evidence = self._render_match_evidence(
                        result=result,
                        identity=identity,
                        tracks=tracks,
                        capture=capture,
                        frame_index=frame_info,
                        output_dir=destination,
                    )

                    if evidence is not None:
                        self.evidence.append(evidence)
                        current_count += 1

                per_identity_count[
                    result.identity_id
                ] = current_count

            # Also produce identity-level summary evidence for identities
            # with no match diagnostic selected, when their V5.2 identity
            # result itself requires review.
            self._generate_identity_summary_evidence(
                manager=manager,
                identities=identities,
                tracks=tracks,
                capture=capture,
                output_dir=destination,
                per_identity_count=per_identity_count,
                include_pass=include_pass,
                include_review=include_review,
                include_high_risk=include_high_risk,
                include_invalid=include_invalid,
            )

        finally:
            capture.release()

        manifest = {
            "version": self.VERSION,
            "video_path": str(video_path),
            "validation": validation_report,
            "evidence_count": len(self.evidence),
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
        }

        if write_manifest:
            manifest_path = (
                destination
                / "evidence_manifest.json"
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

        return manifest

    # ------------------------------------------------------------------
    # FRAME SELECTION
    # ------------------------------------------------------------------

    def _select_match_frames(
        self,
        result: VisualValidationResult,
        identity: Any,
        tracks: dict[Any, Any],
        capture: cv2.VideoCapture,
    ) -> list[int]:
        """
        Select evidence frames using diagnostic-risk-aware ranking.

        V5.3.1 Step 3 priority:

        CRITICAL / HIGH:
            1. Diagnostic frame
            2. Observations closest to diagnostic frame
            3. Remaining valid observation frames
            4. Temporal coverage

        REVIEW:
            1. Diagnostic frame
            2. Observations closest to diagnostic frame
            3. Temporal coverage

        Other statuses:
            1. Diagnostic frame
            2. Observations closest to diagnostic frame
            3. Temporal coverage

        The diagnostic frame is always preserved as the first candidate
        when available.

        Importantly, this method does NOT change the observation-gap
        rule from Step 1. The actual observation validity check still
        happens inside _find_identity_observation() and
        _find_track_observation() during rendering.
        """

        candidate_frames: list[int] = []

        diagnostic_frame = getattr(
            result,
            "frame_index",
            None,
        )

        diagnostic_frame_int: Optional[int] = None

        if diagnostic_frame is not None:
            try:
                diagnostic_frame_int = int(
                    diagnostic_frame
                )
            except (
                TypeError,
                ValueError,
            ):
                diagnostic_frame_int = None

        # --------------------------------------------------------------
        # V5.3.1 Step 3:
        # 1. Diagnostic frame gets absolute priority.
        # --------------------------------------------------------------

        if diagnostic_frame_int is not None:
            candidate_frames.append(
                diagnostic_frame_int
            )

        # --------------------------------------------------------------
        # Collect observation frames.
        #
        # These are actual observation frames, so they are the strongest
        # candidates after the diagnostic frame itself.
        # --------------------------------------------------------------

        observation_frames: list[int] = []

        track = (
            tracks.get(result.track_id)
            if result.track_id is not None
            else None
        )

        if track is not None:
            track_observations = list(
                getattr(
                    track,
                    "observations",
                    [],
                )
                or []
            )

            observation_frames.extend(
                self._observation_frame_values(
                    track_observations
                )
            )

        identity_observations = list(
            getattr(
                identity,
                "observations",
                [],
            )
            or []
        )

        observation_frames.extend(
            self._observation_frame_values(
                identity_observations
            )
        )

        observation_frames = self._unique_ints(
            observation_frames
        )

        # --------------------------------------------------------------
        # V5.3.1 Step 3:
        # 2. Closest observation frames to the diagnostic frame.
        #
        # When there is a diagnostic frame, sort observations by:
        #
        #     distance from diagnostic frame
        #
        # This means a frame immediately surrounding a suspicious
        # diagnostic event is preferred over arbitrary first/middle/last
        # observations.
        # --------------------------------------------------------------

        if diagnostic_frame_int is not None:
            closest_observations = sorted(
                observation_frames,
                key=lambda frame: (
                    abs(
                        frame
                        - diagnostic_frame_int
                    ),
                    frame,
                ),
            )
        else:
            # No diagnostic frame means there is no meaningful anchor.
            # Preserve chronological ordering for temporal coverage.
            closest_observations = sorted(
                observation_frames
            )

        # --------------------------------------------------------------
        # Risk-aware weighting.
        #
        # CRITICAL/HIGH receive the strongest preference for observations
        # near the diagnostic event.
        #
        # REVIEW receives the same proximity-first behavior, but we also
        # preserve temporal coverage more aggressively.
        # --------------------------------------------------------------

        risk_level = str(
            getattr(
                result,
                "risk_level",
                "",
            )
            or ""
        ).upper()

        status = str(
            getattr(
                result,
                "status",
                "",
            )
            or ""
        ).upper()

        if risk_level in {
            "CRITICAL",
            "HIGH",
        }:
            # High-risk evidence should remain concentrated around the
            # diagnostic event.
            ranked_observations = self._rank_high_risk_observations(
                closest_observations,
                diagnostic_frame_int,
            )

        elif status == "REVIEW":
            # Review evidence also prioritizes the diagnostic event,
            # while retaining temporal coverage after nearby evidence.
            ranked_observations = self._rank_review_observations(
                closest_observations,
                diagnostic_frame_int,
            )

        else:
            # Normal/pass evidence still benefits from diagnostic
            # proximity, but chronological coverage is sufficient.
            ranked_observations = self._rank_default_observations(
                closest_observations,
                diagnostic_frame_int,
            )

        candidate_frames.extend(
            ranked_observations
        )

        # --------------------------------------------------------------
        # V5.3.1 Step 3:
        # Final temporal coverage fallback.
        #
        # This preserves the old first/middle/last behavior, but only
        # after diagnostic and proximity-aware candidates have been
        # exhausted.
        # --------------------------------------------------------------

        temporal_frames = (
            self._representative_observation_frames(
                observation_frames,
                limit=3,
            )
        )

        candidate_frames.extend(
            temporal_frames
        )

        return self._unique_ints(
            candidate_frames
        )

    @staticmethod
    def _observation_frame_values(
        observations: Iterable[Any],
    ) -> list[int]:
        """
        Extract valid observation frame indexes.

        This helper only extracts frame indexes. It does not apply the
        observation-gap rule.

        The actual max_observation_frame_gap enforcement remains inside
        _find_identity_observation() and _find_track_observation().
        """
        frame_values: list[int] = []

        for observation in observations:
            frame_index = getattr(
                observation,
                "frame_index",
                None,
            )

            if frame_index is None:
                continue

            try:
                frame_values.append(
                    int(frame_index)
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

        return frame_values

    @staticmethod
    def _rank_high_risk_observations(
        observations: list[int],
        diagnostic_frame: Optional[int],
    ) -> list[int]:
        """
        Rank observations for CRITICAL/HIGH risk diagnostics.

        The closest observations to the diagnostic event are preferred.
        Temporal coverage is deliberately secondary.
        """
        if diagnostic_frame is None:
            return observations

        return sorted(
            observations,
            key=lambda frame: (
                abs(
                    frame
                    - diagnostic_frame
                ),
                frame,
            ),
        )

    @staticmethod
    def _rank_review_observations(
        observations: list[int],
        diagnostic_frame: Optional[int],
    ) -> list[int]:
        """
        Rank observations for REVIEW diagnostics.

        Nearby observations come first. Since observations are already
        sorted by diagnostic distance, this method additionally ensures
        chronological stability for equal-distance candidates.
        """
        if diagnostic_frame is None:
            return sorted(observations)

        return sorted(
            observations,
            key=lambda frame: (
                abs(
                    frame
                    - diagnostic_frame
                ),
                frame,
            ),
        )

    @staticmethod
    def _rank_default_observations(
        observations: list[int],
        diagnostic_frame: Optional[int],
    ) -> list[int]:
        """
        Rank observations for statuses without special risk handling.

        Diagnostic proximity remains useful, but chronological ordering
        is used when there is no diagnostic anchor.
        """
        if diagnostic_frame is None:
            return sorted(observations)

        return sorted(
            observations,
            key=lambda frame: (
                abs(
                    frame
                    - diagnostic_frame
                ),
                frame,
            ),
        )

    @staticmethod
    def _representative_observation_frames(
        observations: Iterable[Any] | Iterable[int],
        limit: int = 3,
    ) -> list[int]:
        """
        Select first/middle/last observation frames.

        This remains the temporal-coverage fallback from the previous
        implementation.

        It is intentionally kept unchanged in behavior so Step 3 only
        changes the priority of candidates rather than removing temporal
        coverage.
        """
        observations = list(observations)

        if not observations:
            return []

        if all(
            isinstance(
                observation,
                int,
            )
            for observation in observations
        ):
            frame_values = [
                int(observation)
                for observation in observations
            ]
        else:
            frame_values = [
                int(obs.frame_index)
                for obs in observations
                if getattr(
                    obs,
                    "frame_index",
                    None,
                )
                is not None
            ]

        if not frame_values:
            return []

        if len(frame_values) <= limit:
            return frame_values

        indexes = [
            0,
            len(frame_values) // 2,
            len(frame_values) - 1,
        ]

        selected: list[int] = []

        for index in indexes:
            if len(selected) >= limit:
                break

            selected.append(
                frame_values[index]
            )

        return selected

    # ------------------------------------------------------------------
    # RENDERING
    # ------------------------------------------------------------------

    def _render_match_evidence(
        self,
        result: VisualValidationResult,
        identity: Any,
        tracks: dict[Any, Any],
        capture: cv2.VideoCapture,
        frame_index: int,
        output_dir: Path,
    ) -> Optional[EvidenceFrame]:
        frame = self._read_frame(
            capture,
            frame_index,
        )

        if frame is None:
            logger.warning(
                "Could not read frame %s for identity %s.",
                frame_index,
                result.identity_id,
            )
            return None

        # --------------------------------------------------------------
        # V5.3.1 Step 2:
        # Track exactly where the observation came from.
        # --------------------------------------------------------------

        observation = self._find_identity_observation(
            identity,
            frame_index,
            max_frame_gap=self.max_observation_frame_gap,
        )

        observation_source = None

        if observation is not None:
            observation_source = "identity"

        elif result.track_id is not None:
            track = tracks.get(
                result.track_id
            )

            observation = self._find_track_observation(
                track,
                frame_index,
                max_frame_gap=self.max_observation_frame_gap,
            )

            if observation is not None:
                observation_source = "track"

        # --------------------------------------------------------------
        # V5.3.1 Step 2:
        # Calculate observation provenance.
        # --------------------------------------------------------------

        observation_frame_index = None
        observation_frame_gap = None

        if observation is not None:
            observation_frame_index = getattr(
                observation,
                "frame_index",
                None,
            )

            if observation_frame_index is not None:
                observation_frame_index = int(
                    observation_frame_index
                )

                observation_frame_gap = abs(
                    observation_frame_index
                    - int(frame_index)
                )

        bbox = self._bbox_to_list(
            getattr(
                observation,
                "bbox",
                None,
            )
        )

        self._draw_observation(
            frame,
            observation,
            label=self._build_identity_label(
                result,
                identity,
            ),
        )

        self._draw_header(
            frame,
            title="IDENTITY VISUAL EVIDENCE V5.3.1",
            result=result,
        )

        self._draw_footer(
            frame,
            [
                f"identity={result.identity_id}",
                f"track={result.track_id}",
                f"frame={frame_index}",
                f"appearance={result.appearance_similarity:.4f}",
                f"temporal_gap={result.temporal_gap}",
                f"spatial={result.spatial_distance:.2f}",
                f"motion={result.motion_difference:.2f}",
                (
                    "observation_gap_limit="
                    f"{self.max_observation_frame_gap}"
                ),
                (
                    "observation="
                    f"{observation_frame_index}"
                ),
                (
                    "observation_gap="
                    f"{observation_frame_gap}"
                ),
                (
                    "observation_source="
                    f"{observation_source or 'none'}"
                ),
            ],
        )

        identity_dir = (
            output_dir
            / f"identity_{self._safe_name(result.identity_id)}"
        )

        identity_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            f"frame_{int(frame_index):06d}"
            f"_track_"
            f"{result.track_id if result.track_id is not None else 'na'}"
            f"_{result.status.lower()}.jpg"
        )

        image_path = identity_dir / filename

        ok = cv2.imwrite(
            str(image_path),
            frame,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                self.jpeg_quality,
            ],
        )

        if not ok:
            logger.warning(
                "Failed to write evidence image: %s",
                image_path,
            )
            return None

        evidence_type = self._evidence_type(
            result
        )

        return EvidenceFrame(
            identity_id=result.identity_id,
            track_id=result.track_id,
            source_track_ids=list(
                getattr(
                    identity,
                    "source_track_ids",
                    [],
                )
                or []
            ),
            frame_index=int(frame_index),
            image_path=str(image_path),
            evidence_type=evidence_type,
            status=result.status,
            risk_level=result.risk_level,
            bbox=bbox,

            # V5.3.1 Step 2: observation provenance
            observation_frame_index=observation_frame_index,
            observation_frame_gap=observation_frame_gap,
            observation_source=observation_source,

            appearance_similarity=result.appearance_similarity,
            temporal_gap=result.temporal_gap,
            spatial_distance=result.spatial_distance,
            motion_difference=result.motion_difference,
            team=getattr(
                identity,
                "team",
                None,
            ),
            jersey_number=getattr(
                identity,
                "jersey_number",
                None,
            ),
            warnings=list(
                result.validation_warnings
            ),
            matcher_reasons=list(
                result.matcher_reasons
            ),
        )

    def _generate_identity_summary_evidence(
        self,
        manager: Any,
        identities: dict[Any, Any],
        tracks: dict[Any, Any],
        capture: cv2.VideoCapture,
        output_dir: Path,
        per_identity_count: dict[str, int],
        *,
        include_pass: bool,
        include_review: bool,
        include_high_risk: bool,
        include_invalid: bool,
    ) -> None:
        """
        Generate one representative identity image when an identity itself
        needs evidence but no suitable diagnostic image was produced.
        """
        for (
            identity_id,
            identity_result,
        ) in self.validator.identity_results.items():

            if per_identity_count.get(
                identity_id,
                0,
            ) > 0:
                continue

            if not self._status_selected(
                identity_result.status,
                include_pass=include_pass,
                include_review=include_review,
                include_high_risk=include_high_risk,
                include_invalid=include_invalid,
            ):
                continue

            identity = identities.get(
                identity_id
            )

            if identity is None:
                continue

            observations = list(
                getattr(
                    identity,
                    "observations",
                    [],
                )
                or []
            )

            frame_indexes = (
                self._representative_observation_frames(
                    observations,
                    limit=1,
                )
            )

            if not frame_indexes:
                continue

            frame_index = frame_indexes[0]

            frame = self._read_frame(
                capture,
                frame_index,
            )

            if frame is None:
                continue

            # ----------------------------------------------------------
            # V5.3.1 Step 2:
            # Summary evidence explicitly uses identity observations.
            # ----------------------------------------------------------

            observation = self._find_identity_observation(
                identity,
                frame_index,
                max_frame_gap=self.max_observation_frame_gap,
            )

            observation_source = (
                "identity"
                if observation is not None
                else None
            )

            observation_frame_index = None
            observation_frame_gap = None

            if observation is not None:
                observation_frame_index = getattr(
                    observation,
                    "frame_index",
                    None,
                )

                if observation_frame_index is not None:
                    observation_frame_index = int(
                        observation_frame_index
                    )

                    observation_frame_gap = abs(
                        observation_frame_index
                        - int(frame_index)
                    )

            self._draw_observation(
                frame,
                observation,
                label=(
                    f"Identity {identity_id} | "
                    f"{getattr(identity, 'class_name', 'unknown')}"
                ),
            )

            self._draw_header(
                frame,
                title="IDENTITY SUMMARY EVIDENCE V5.3.1",
                result=None,
                status=identity_result.status,
                risk_level=identity_result.risk_level,
            )

            self._draw_footer(
                frame,
                [
                    f"identity={identity_id}",
                    f"frame={frame_index}",
                    (
                        "source_tracks="
                        f"{getattr(identity, 'source_track_ids', [])}"
                    ),
                    (
                        "observations="
                        f"{getattr(identity, 'frame_count', 0)}"
                    ),
                    f"team={getattr(identity, 'team', None)}",
                    (
                        "jersey="
                        f"{getattr(identity, 'jersey_number', None)}"
                    ),
                    (
                        "observation_gap_limit="
                        f"{self.max_observation_frame_gap}"
                    ),
                    (
                        "observation="
                        f"{observation_frame_index}"
                    ),
                    (
                        "observation_gap="
                        f"{observation_frame_gap}"
                    ),
                    (
                        "observation_source="
                        f"{observation_source or 'none'}"
                    ),
                ],
            )

            identity_dir = (
                output_dir
                / f"identity_{self._safe_name(identity_id)}"
            )

            identity_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            image_path = identity_dir / (
                f"frame_{int(frame_index):06d}"
                f"_identity_summary.jpg"
            )

            ok = cv2.imwrite(
                str(image_path),
                frame,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    self.jpeg_quality,
                ],
            )

            if not ok:
                continue

            self.evidence.append(
                EvidenceFrame(
                    identity_id=str(identity_id),
                    track_id=None,
                    source_track_ids=list(
                        getattr(
                            identity,
                            "source_track_ids",
                            [],
                        )
                        or []
                    ),
                    frame_index=int(frame_index),
                    image_path=str(image_path),
                    evidence_type="identity_summary",
                    status=identity_result.status,
                    risk_level=identity_result.risk_level,
                    bbox=self._bbox_to_list(
                        getattr(
                            observation,
                            "bbox",
                            None,
                        )
                    ),

                    # V5.3.1 Step 2: observation provenance
                    observation_frame_index=observation_frame_index,
                    observation_frame_gap=observation_frame_gap,
                    observation_source=observation_source,

                    team=getattr(
                        identity,
                        "team",
                        None,
                    ),
                    jersey_number=getattr(
                        identity,
                        "jersey_number",
                        None,
                    ),
                    warnings=list(
                        identity_result.reasons
                    ),
                )
            )

            per_identity_count[
                identity_id
            ] = 1

    # ------------------------------------------------------------------
    # VIDEO
    # ------------------------------------------------------------------

    @staticmethod
    def _open_video(
        video_path: str | Path,
    ) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise ValueError(
                f"Unable to open video: {video_path}"
            )

        return capture

    @staticmethod
    def _read_frame(
        capture: cv2.VideoCapture,
        frame_index: int,
    ) -> Optional[Any]:
        if frame_index < 0:
            return None

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index),
        )

        ok, frame = capture.read()

        if not ok:
            return None

        return frame

    # ------------------------------------------------------------------
    # OBSERVATIONS
    # ------------------------------------------------------------------

    @staticmethod
    def _find_identity_observation(
        identity: Any,
        frame_index: int,
        max_frame_gap: int = 5,
    ) -> Optional[Any]:
        """
        Find the identity observation nearest to frame_index.

        V5.3.1 Step 1:
        The nearest observation is only accepted when its frame distance
        is <= max_frame_gap.

        This prevents an evidence frame from being annotated with a
        bounding box belonging to a visually unrelated point in time.
        """
        observations = list(
            getattr(
                identity,
                "observations",
                [],
            )
            or []
        )

        if not observations:
            return None

        exact = [
            obs
            for obs in observations
            if getattr(
                obs,
                "frame_index",
                None,
            ) == frame_index
        ]

        if exact:
            return exact[0]

        valid_observations = [
            obs
            for obs in observations
            if getattr(
                obs,
                "frame_index",
                None,
            ) is not None
        ]

        if not valid_observations:
            return None

        nearest = min(
            valid_observations,
            key=lambda obs: abs(
                int(
                    getattr(
                        obs,
                        "frame_index",
                        0,
                    )
                )
                - int(frame_index)
            ),
        )

        nearest_frame = int(
            getattr(
                nearest,
                "frame_index",
                0,
            )
        )

        frame_gap = abs(
            nearest_frame
            - int(frame_index)
        )

        if frame_gap > max_frame_gap:
            return None

        return nearest

    @staticmethod
    def _find_track_observation(
        track: Any,
        frame_index: int,
        max_frame_gap: int = 5,
    ) -> Optional[Any]:
        """
        Find the track observation nearest to frame_index.

        V5.3.1 Step 1:
        The nearest observation is only accepted when its frame distance
        is <= max_frame_gap.
        """
        if track is None:
            return None

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

        exact = [
            obs
            for obs in observations
            if getattr(
                obs,
                "frame_index",
                None,
            ) == frame_index
        ]

        if exact:
            return exact[0]

        valid_observations = [
            obs
            for obs in observations
            if getattr(
                obs,
                "frame_index",
                None,
            ) is not None
        ]

        if not valid_observations:
            return None

        nearest = min(
            valid_observations,
            key=lambda obs: abs(
                int(
                    getattr(
                        obs,
                        "frame_index",
                        0,
                    )
                )
                - int(frame_index)
            ),
        )

        nearest_frame = int(
            getattr(
                nearest,
                "frame_index",
                0,
            )
        )

        frame_gap = abs(
            nearest_frame
            - int(frame_index)
        )

        if frame_gap > max_frame_gap:
            return None

        return nearest

    # ------------------------------------------------------------------
    # DRAWING
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_observation(
        frame: Any,
        observation: Any,
        *,
        label: str,
    ) -> None:
        if observation is None:
            return

        bbox = getattr(
            observation,
            "bbox",
            None,
        )

        if bbox is None:
            return

        x1 = int(
            max(
                0,
                bbox.x1,
            )
        )

        y1 = int(
            max(
                0,
                bbox.y1,
            )
        )

        x2 = int(
            max(
                0,
                bbox.x2,
            )
        )

        y2 = int(
            max(
                0,
                bbox.y2,
            )
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            3,
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(
                    25,
                    y1 - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_header(
        frame: Any,
        *,
        title: str,
        result: Optional[VisualValidationResult],
        status: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> None:
        if result is not None:
            status = result.status
            risk_level = result.risk_level

        text = (
            f"{title} | "
            f"status={status or 'N/A'} | "
            f"risk={risk_level or 'N/A'}"
        )

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 42),
            (20, 20, 20),
            -1,
        )

        cv2.putText(
            frame,
            text,
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_footer(
        frame: Any,
        lines: list[str],
    ) -> None:
        if not lines:
            return

        line_height = 26
        padding = 10

        height = (
            len(lines) * line_height
            + padding * 2
        )

        y_start = (
            frame.shape[0]
            - height
        )

        cv2.rectangle(
            frame,
            (0, y_start),
            (
                frame.shape[1],
                frame.shape[0],
            ),
            (20, 20, 20),
            -1,
        )

        for index, line in enumerate(lines):
            y = (
                y_start
                + padding
                + (index + 1) * line_height
                - 5
            )

            cv2.putText(
                frame,
                line[:180],
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _build_identity_label(
        result: VisualValidationResult,
        identity: Any,
    ) -> str:
        team = getattr(
            identity,
            "team",
            None,
        )

        jersey = getattr(
            identity,
            "jersey_number",
            None,
        )

        extras = []

        if team is not None:
            extras.append(
                f"team={team}"
            )

        if jersey is not None:
            extras.append(
                f"jersey={jersey}"
            )

        suffix = (
            " | "
            + " ".join(extras)
            if extras
            else ""
        )

        return (
            f"ID {result.identity_id}"
            f" | track={result.track_id}"
            f" | {result.risk_level}"
            f"{suffix}"
        )

    @staticmethod
    def _evidence_type(
        result: VisualValidationResult,
    ) -> str:
        if result.risk_level == "CRITICAL":
            return "critical_match"

        if result.risk_level == "HIGH":
            return "high_risk_match"

        if result.status == "REVIEW":
            return "review_match"

        return "pass_match"

    @staticmethod
    def _bbox_to_list(
        bbox: Any,
    ) -> Optional[list[float]]:
        if bbox is None:
            return None

        return [
            float(
                getattr(
                    bbox,
                    "x1",
                    0.0,
                )
            ),
            float(
                getattr(
                    bbox,
                    "y1",
                    0.0,
                )
            ),
            float(
                getattr(
                    bbox,
                    "x2",
                    0.0,
                )
            ),
            float(
                getattr(
                    bbox,
                    "y2",
                    0.0,
                )
            ),
        ]

    @staticmethod
    def _unique_ints(
        values: Iterable[int],
    ) -> list[int]:
        result: list[int] = []
        seen: set[int] = set()

        for value in values:
            try:
                value = int(value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if value not in seen:
                seen.add(value)
                result.append(value)

        return result

    @staticmethod
    def _safe_name(
        value: Any,
    ) -> str:
        text = str(value)

        return "".join(
            character
            if (
                character.isalnum()
                or character in "-_."
            )
            else "_"
            for character in text
        )

    @staticmethod
    def _status_selected(
        status: str,
        *,
        include_pass: bool,
        include_review: bool,
        include_high_risk: bool,
        include_invalid: bool,
    ) -> bool:
        return (
            (
                status == "PASS"
                and include_pass
            )
            or (
                status == "REVIEW"
                and include_review
            )
            or (
                status == "HIGH_RISK"
                and include_high_risk
            )
            or (
                status == "INVALID"
                and include_invalid
            )
        )

    def get_evidence(
        self,
    ) -> list[dict[str, Any]]:
        return [
            item.to_dict()
            for item in self.evidence
        ]


def generate_visual_evidence(
    manager: Any,
    video_path: str | Path,
    output_dir: str | Path = "media/validation_evidence",
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Convenience API for V5.3.1 evidence generation.
    """

    generator = VisualEvidenceGenerator(
        output_dir=output_dir,
        **{
            key: value
            for key, value in kwargs.items()
            if key in {
                "validator",
                "max_evidence_per_match",
                "max_evidence_per_identity",
                "jpeg_quality",
                "max_observation_frame_gap",
            }
        },
    )

    return generator.generate(
        manager,
        video_path,
        output_dir=output_dir,
        include_pass=kwargs.get(
            "include_pass",
            False,
        ),
        include_review=kwargs.get(
            "include_review",
            True,
        ),
        include_high_risk=kwargs.get(
            "include_high_risk",
            True,
        ),
        include_invalid=kwargs.get(
            "include_invalid",
            True,
        ),
        write_manifest=kwargs.get(
            "write_manifest",
            True,
        ),
    )