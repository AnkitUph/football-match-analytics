
"""
Global Identity Diagnostic Report.

Phase 7 diagnostic utility for Global Identity Manager V5.1.

Purpose:
    Inspect the final global identities produced by the identity
    pipeline without changing any matching decisions.

Reports:
    - Global identity ID
    - class
    - ByteTrack source IDs
    - frame range
    - observation count
    - team
    - jersey number
    - identity confidence
    - match history
    - accepted/rejected candidate matches
    - appearance similarity
    - temporal gap
    - spatial distance
    - motion difference
    - match reason

This module is diagnostic only.
It does NOT modify identities or matching behavior.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class GlobalIdentityDiagnosticReport:
    """
    Generate human-readable diagnostics for global identities.
    """

    def __init__(
        self,
        manager,
    ) -> None:

        self.manager = manager

    # ============================================================
    # PUBLIC API
    # ============================================================

    def generate(
        self,
        limit: int | None = None,
        include_rejected: bool = True,
    ) -> str:
        """
        Generate the complete diagnostic report.

        Args:
            limit:
                Maximum number of identities to display.
                None means all identities.

            include_rejected:
                Include rejected candidate matches in the
                per-identity match history.

        Returns:
            Formatted diagnostic report string.
        """

        identities = list(
            self.manager.identities.values()
        )

        identities.sort(
            key=lambda identity: (
                identity.class_name,
                identity.identity_id,
            )
        )

        if limit is not None:

            identities = identities[:limit]

        lines: list[str] = []

        lines.extend(
            self._header()
        )

        lines.extend(
            self._overall_summary()
        )

        lines.append("")

        lines.extend(
            self._class_summary()
        )

        lines.append("")

        lines.extend(
            self._identity_sections(
                identities=identities,
                include_rejected=include_rejected,
            )
        )

        lines.append("")

        lines.extend(
            self._suspicious_identities()
        )

        lines.append("")

        lines.extend(
            self._footer()
        )

        return "\n".join(lines)

    def print_report(
        self,
        limit: int | None = None,
        include_rejected: bool = True,
    ) -> None:
        """
        Print the diagnostic report.
        """

        print(
            self.generate(
                limit=limit,
                include_rejected=include_rejected,
            )
        )

    # ============================================================
    # HEADER
    # ============================================================

    @staticmethod
    def _header() -> list[str]:

        return [
            "",
            "=" * 78,
            "GLOBAL IDENTITY DIAGNOSTIC REPORT",
            "=" * 78,
        ]

    # ============================================================
    # OVERALL SUMMARY
    # ============================================================

    def _overall_summary(self) -> list[str]:

        manager = self.manager

        identities = list(
            manager.identities.values()
        )

        multi_fragment = [
            identity
            for identity in identities
            if len(
                identity.source_track_ids
            ) > 1
        ]

        single_fragment = [
            identity
            for identity in identities
            if len(
                identity.source_track_ids
            ) == 1
        ]

        total_observations = sum(
            len(identity.observations)
            for identity in identities
        )

        return [
            "",
            "OVERALL SUMMARY",
            "-" * 78,
            (
                f"Global identities: "
                f"{len(identities)}"
            ),
            (
                f"ByteTrack fragments processed: "
                f"{manager.total_tracks_processed}"
            ),
            (
                f"Tracks merged: "
                f"{manager.total_tracks_merged}"
            ),
            (
                f"New global identities: "
                f"{manager.total_new_identities}"
            ),
            (
                f"Multi-fragment identities: "
                f"{len(multi_fragment)}"
            ),
            (
                f"Single-fragment identities: "
                f"{len(single_fragment)}"
            ),
            (
                f"Total global observations: "
                f"{total_observations}"
            ),
            (
                f"Candidate comparisons: "
                f"{len(manager.match_diagnostics)}"
            ),
        ]

    # ============================================================
    # CLASS SUMMARY
    # ============================================================

    def _class_summary(self) -> list[str]:

        grouped: dict[str, list[Any]] = (
            defaultdict(list)
        )

        for identity in (
            self.manager.identities.values()
        ):

            grouped[
                identity.class_name
            ].append(identity)

        lines = [
            "IDENTITIES BY CLASS",
            "-" * 78,
        ]

        for class_name in sorted(
            grouped
        ):

            identities = grouped[
                class_name
            ]

            fragment_count = sum(
                len(
                    identity.source_track_ids
                )
                for identity in identities
            )

            multi_fragment = sum(
                1
                for identity in identities
                if len(
                    identity.source_track_ids
                ) > 1
            )

            observations = sum(
                len(identity.observations)
                for identity in identities
            )

            lines.append(
                f"{class_name}:"
            )

            lines.append(
                f"  Global identities: "
                f"{len(identities)}"
            )

            lines.append(
                f"  Source fragments: "
                f"{fragment_count}"
            )

            lines.append(
                f"  Multi-fragment identities: "
                f"{multi_fragment}"
            )

            lines.append(
                f"  Observations: "
                f"{observations}"
            )

            lines.append("")

        return lines

    # ============================================================
    # IDENTITY SECTIONS
    # ============================================================

    def _identity_sections(
        self,
        identities,
        include_rejected: bool,
    ) -> list[str]:

        lines: list[str] = []

        diagnostics = (
            self.manager.match_diagnostics
        )

        for index, identity in enumerate(
            identities,
            start=1,
        ):

            lines.extend(
                self._identity_section(
                    identity=identity,
                    diagnostics=diagnostics,
                    include_rejected=(
                        include_rejected
                    ),
                )
            )

            if index != len(identities):

                lines.extend(
                    [
                        "",
                        "-" * 78,
                    ]
                )

        return lines

    def _identity_section(
        self,
        identity,
        diagnostics: list[dict[str, Any]],
        include_rejected: bool,
    ) -> list[str]:

        source_ids = list(
            identity.source_track_ids
        )

        identity_diagnostics = [
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.get(
                "identity_id"
            )
            == identity.identity_id
        ]

        accepted = [
            diagnostic
            for diagnostic
            in identity_diagnostics
            if diagnostic.get("matched")
        ]

        rejected = [
            diagnostic
            for diagnostic
            in identity_diagnostics
            if not diagnostic.get("matched")
        ]

        lines = [
            "",
            (
                f"IDENTITY {identity.identity_id}"
            ),
            "",
            (
                f"Class: "
                f"{identity.class_name}"
            ),
            (
                f"Active: "
                f"{identity.active}"
            ),
            (
                f"Identity confidence: "
                f"{identity.identity_confidence:.3f}"
            ),
            (
                f"Team: "
                f"{identity.team}"
            ),
            (
                f"Jersey number: "
                f"{identity.jersey_number}"
            ),
            (
                f"First frame: "
                f"{identity.first_frame}"
            ),
            (
                f"Last frame: "
                f"{identity.last_frame}"
            ),
            (
                f"Observations: "
                f"{len(identity.observations)}"
            ),
            (
                f"ByteTrack fragments: "
                f"{len(source_ids)}"
            ),
            (
                f"Source track IDs: "
                f"{source_ids}"
            ),
            "",
            (
                f"Accepted matches involving identity: "
                f"{len(accepted)}"
            ),
            (
                f"Rejected candidates involving identity: "
                f"{len(rejected)}"
            ),
        ]

        if accepted:

            lines.extend(
                [
                    "",
                    "ACCEPTED MATCH HISTORY",
                    "-" * 50,
                ]
            )

            for diagnostic in accepted:

                lines.extend(
                    self._format_diagnostic(
                        diagnostic
                    )
                )

        if include_rejected and rejected:

            lines.extend(
                [
                    "",
                    "REJECTED CANDIDATE HISTORY",
                    "-" * 50,
                ]
            )

            for diagnostic in rejected:

                lines.extend(
                    self._format_diagnostic(
                        diagnostic
                    )
                )

        return lines

    # ============================================================
    # MATCH FORMATTER
    # ============================================================

    @staticmethod
    def _format_diagnostic(
        diagnostic: dict[str, Any],
    ) -> list[str]:

        appearance = diagnostic.get(
            "appearance_similarity",
            -1.0,
        )

        if appearance is None:
            appearance = -1.0

        appearance_text = (
            f"{appearance:.3f}"
            if appearance >= 0.0
            else "N/A"
        )

        score = diagnostic.get(
            "score"
        )

        score_text = (
            f"{score:.3f}"
            if score is not None
            else "N/A"
        )

        spatial = diagnostic.get(
            "spatial_distance"
        )

        spatial_text = (
            f"{spatial:.2f}"
            if spatial is not None
            else "N/A"
        )

        motion = diagnostic.get(
            "motion_difference"
        )

        motion_text = (
            f"{motion:.2f}"
            if motion is not None
            else "N/A"
        )

        return [
            (
                f"  Track ID: "
                f"{diagnostic.get('track_id')}"
            ),
            (
                f"    Score: "
                f"{score_text}"
            ),
            (
                f"    Temporal gap: "
                f"{diagnostic.get('temporal_gap')}"
            ),
            (
                f"    Spatial distance: "
                f"{spatial_text}"
            ),
            (
                f"    Motion difference: "
                f"{motion_text}"
            ),
            (
                f"    Appearance similarity: "
                f"{appearance_text}"
            ),
            (
                f"    Appearance evaluated: "
                f"{diagnostic.get('appearance_evaluated')}"
            ),
            (
                f"    Team match: "
                f"{diagnostic.get('team_match')}"
            ),
            (
                f"    Jersey match: "
                f"{diagnostic.get('jersey_match')}"
            ),
            (
                f"    Matched: "
                f"{diagnostic.get('matched')}"
            ),
            (
                f"    Reason: "
                f"{diagnostic.get('reason')}"
            ),
        ]

    # ============================================================
    # SUSPICIOUS IDENTITIES
    # ============================================================

    def _suspicious_identities(self) -> list[str]:

        identities = list(
            self.manager.identities.values()
        )

        diagnostics = (
            self.manager.match_diagnostics
        )

        lines = [
            "SUSPICIOUS / UNDER-MERGING CANDIDATES",
            "-" * 78,
        ]

        suspicious_found = False

        # ----------------------------------------------------
        # Case 1:
        # Identity consists of only one fragment while
        # rejected candidates exist for that identity.
        # ----------------------------------------------------

        for identity in identities:

            source_count = len(
                identity.source_track_ids
            )

            if source_count != 1:
                continue

            rejected = [
                diagnostic
                for diagnostic
                in diagnostics
                if diagnostic.get(
                    "identity_id"
                )
                == identity.identity_id
                and not diagnostic.get(
                    "matched"
                )
            ]

            if not rejected:
                continue

            suspicious = [
                diagnostic
                for diagnostic
                in rejected
                if self._looks_potentially_close(
                    diagnostic
                )
            ]

            if not suspicious:
                continue

            suspicious_found = True

            lines.extend(
                [
                    "",
                    (
                        f"Identity: "
                        f"{identity.identity_id}"
                    ),
                    (
                        f"  Class: "
                        f"{identity.class_name}"
                    ),
                    (
                        f"  Fragment: "
                        f"{identity.source_track_ids}"
                    ),
                    (
                        f"  Observations: "
                        f"{len(identity.observations)}"
                    ),
                    (
                        f"  Close rejected candidates: "
                        f"{len(suspicious)}"
                    ),
                ]
            )

            for diagnostic in suspicious[:5]:

                lines.extend(
                    [
                        (
                            f"    Track "
                            f"{diagnostic.get('track_id')}: "
                            f"score="
                            f"{diagnostic.get('score'):.3f}"
                        ),
                        (
                            f"      reason="
                            f"{diagnostic.get('reason')}"
                        ),
                        (
                            f"      appearance="
                            f"{diagnostic.get('appearance_similarity'):.3f}"
                            if diagnostic.get(
                                "appearance_similarity",
                                -1.0,
                            ) >= 0.0
                            else
                            "      appearance=N/A"
                        ),
                    ]
                )

        if not suspicious_found:

            lines.extend(
                [
                    "",
                    "No obvious under-merging candidates "
                    "were detected by the current heuristic.",
                ]
            )

        return lines

    # ============================================================
    # CLOSE-CANDIDATE HEURISTIC
    # ============================================================

    def _looks_potentially_close(
        self,
        diagnostic: dict[str, Any],
    ) -> bool:
        """
        Identify rejected candidates worth visual inspection.

        This does NOT change matching behavior.

        It deliberately uses broad diagnostic thresholds
        so that potentially interesting cases are surfaced.
        """

        score = diagnostic.get(
            "score"
        )

        appearance = diagnostic.get(
            "appearance_similarity",
            -1.0,
        )

        spatial = diagnostic.get(
            "spatial_distance"
        )

        motion = diagnostic.get(
            "motion_difference"
        )

        # ----------------------------------------------------
        # High score but rejected
        # ----------------------------------------------------

        if (
            score is not None
            and score >= 0.45
        ):
            return True

        # ----------------------------------------------------
        # Strong appearance but rejected
        # ----------------------------------------------------

        if (
            appearance is not None
            and appearance >= 0.80
        ):
            return True

        # ----------------------------------------------------
        # Good geometry but rejected
        # ----------------------------------------------------

        if (
            spatial is not None
            and spatial <= 150.0
            and (
                motion is None
                or motion <= 60.0
            )
        ):
            return True

        return False

    # ============================================================
    # FOOTER
    # ============================================================

    @staticmethod
    def _footer() -> list[str]:

        return [
            "=" * 78,
            "END GLOBAL IDENTITY DIAGNOSTIC REPORT",
            "=" * 78,
        ]

