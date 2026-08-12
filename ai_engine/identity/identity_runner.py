"""
Global Identity Pipeline Runner V5.1.

Pipeline:

    Video
      ↓
    YOLO Detector
      ↓
    ByteTrack
      ↓
    ByteTrack track history
      ↓
    Global Identity Matcher V5.1
      ↓
    Global Identity Manager
      ↓
    Persistent global identities
      ↓
    Global Identity Fragment Analysis
      ↓
    Optional Phase 7 Visual Validation

V5.1 runner goals:

1. Keep runner configuration aligned with IdentityMatcher V5.1.
2. Expose important matcher thresholds.
3. Avoid silently overriding matcher defaults.
4. Preserve conservative identity matching.
5. Provide complete diagnostics.
6. Analyze how ByteTrack fragments are distributed
   across global identities.
7. Detect heavily fragmented identities.
8. Detect large temporal gaps between fragments.
9. Detect possible under-merging.
10. Keep visual validation independent from matching.
"""

from __future__ import annotations

import logging
from statistics import mean, median

from ai_engine.identity.identity_manager import (
    GlobalIdentityManager,
)

from ai_engine.identity.identity_matcher import (
    IdentityMatcher,
)

from ai_engine.tracking.tracking_runner import (
    run_tracking,
)


logger = logging.getLogger(__name__)


# ============================================================
# GENERIC OBJECT HELPERS
# ============================================================


def _get_value(obj, *names, default=None):
    """
    Safely retrieve a value from either an object or dictionary.

    Supports:
        - dataclass/object attributes
        - dictionaries
    """

    for name in names:

        if isinstance(obj, dict):

            if name in obj:
                return obj[name]

        else:

            if hasattr(obj, name):
                return getattr(obj, name)

    return default


# ============================================================
# TRACK FIELD HELPERS
# ============================================================


def _extract_track_id(track):
    """
    Extract ByteTrack local ID.

    Current Track schema:
        Track.local_id
    """

    value = _get_value(
        track,
        "local_id",
        "track_id",
        "tracker_id",
        "id",
        default=None,
    )

    if value is None:
        return None

    try:
        return int(value)

    except (TypeError, ValueError):
        return value


def _extract_start_frame(track):
    """
    Extract first frame of a ByteTrack fragment.

    Current Track schema:
        Track.first_frame
    """

    return _get_value(
        track,
        "first_frame",
        "start_frame",
        "frame_start",
        "start",
        default=None,
    )


def _extract_end_frame(track):
    """
    Extract last frame of a ByteTrack fragment.

    Current Track schema:
        Track.last_frame
    """

    return _get_value(
        track,
        "last_frame",
        "end_frame",
        "frame_end",
        "end",
        default=None,
    )


def _extract_class(track):
    """
    Extract class name.
    """

    return _get_value(
        track,
        "class_name",
        "class_label",
        "label",
        "class",
        default="unknown",
    )


# ============================================================
# IDENTITY FRAGMENT HELPERS
# ============================================================


def _extract_identity_track_ids(identity):
    """
    Extract ByteTrack fragment IDs belonging to a Global Identity.

    IMPORTANT:
        The current Identity schema stores these in:

            source_track_ids

        Example:

            Identity(
                identity_id="GID_0001",
                class_name="player",
                source_track_ids=[21, 57, 91],
                ...
            )

    source_track_ids is therefore the primary field.

    Additional fallback fields are retained for compatibility
    with older identity implementations.
    """

    candidates = _get_value(
        identity,

        # ----------------------------------------------------
        # CURRENT PROJECT FIELD
        # ----------------------------------------------------

        "source_track_ids",

        # ----------------------------------------------------
        # LEGACY / COMPATIBILITY FIELDS
        # ----------------------------------------------------

        "track_ids",
        "fragment_ids",
        "track_history_ids",
        "track_fragments",
        "fragments",
        "tracks",
        "members",

        default=None,
    )

    if candidates is None:
        return []

    # --------------------------------------------------------
    # Single integer/string
    # --------------------------------------------------------

    if isinstance(candidates, (int, str)):

        try:
            return [int(candidates)]

        except (TypeError, ValueError):
            return [candidates]

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(candidates, dict):

        result = []

        for key in candidates.keys():

            try:
                result.append(int(key))

            except (TypeError, ValueError):
                result.append(key)

        return result

    # --------------------------------------------------------
    # List / tuple / set
    # --------------------------------------------------------

    if isinstance(candidates, (list, tuple, set)):

        result = []

        for item in candidates:

            if isinstance(item, (int, str)):

                try:
                    result.append(int(item))

                except (TypeError, ValueError):
                    result.append(item)

            else:

                track_id = _extract_track_id(item)

                if track_id is not None:
                    result.append(track_id)

        return result

    return []


# ============================================================
# TEMPORAL GAP
# ============================================================


def _calculate_fragment_gap(
    previous_track,
    current_track,
):
    """
    Calculate temporal gap between two ByteTrack fragments.

    Example:

        Fragment A:
            frames 0 -> 100

        Fragment B:
            frames 120 -> 200

        Gap:
            19 frames

    Formula:

        current_start - previous_end - 1
    """

    previous_end = _extract_end_frame(
        previous_track
    )

    current_start = _extract_start_frame(
        current_track
    )

    if (
        previous_end is None
        or current_start is None
    ):
        return None

    try:

        return max(
            0,
            int(current_start)
            - int(previous_end)
            - 1,
        )

    except (TypeError, ValueError):

        return None


# ============================================================
# GLOBAL IDENTITY FRAGMENT ANALYSIS
# ============================================================


def analyze_global_identity_fragments(
    manager: GlobalIdentityManager,
    track_history,
    long_gap_threshold: int = 75,
    heavy_fragment_threshold: int = 5,
):
    """
    Analyze how ByteTrack fragments are distributed across
    Global Identities.

    This is DIAGNOSTIC ONLY.

    It does NOT modify:

        - identities
        - matcher decisions
        - track assignments
        - thresholds

    Returns:
        Dictionary containing summary statistics and
        per-identity fragment reports.
    """

    # ========================================================
    # BUILD TRACK LOOKUP
    # ========================================================

    track_lookup = {}

    all_track_ids = set()

    for track in track_history:

        track_id = _extract_track_id(track)

        if track_id is None:
            continue

        track_lookup[track_id] = track
        all_track_ids.add(track_id)

    # ========================================================
    # IDENTITY COLLECTION
    # ========================================================

    identities = manager.identities

    fragment_reports = []

    total_fragments = 0

    identities_with_fragments = 0

    fragment_counts = []

    assigned_track_ids = set()

    # ========================================================
    # ANALYZE EVERY GLOBAL IDENTITY
    # ========================================================

    for global_id, identity in identities.items():

        # ----------------------------------------------------
        # Get ByteTrack IDs belonging to this identity
        # ----------------------------------------------------

        track_ids = _extract_identity_track_ids(
            identity
        )

        # Remove duplicates while preserving order.
        unique_track_ids = list(
            dict.fromkeys(track_ids)
        )

        # Track IDs that actually exist in the
        # original track history.
        valid_track_ids = []

        missing_track_ids = []

        for track_id in unique_track_ids:

            if track_id in track_lookup:

                valid_track_ids.append(
                    track_id
                )

                assigned_track_ids.add(
                    track_id
                )

            else:

                missing_track_ids.append(
                    track_id
                )

        # ----------------------------------------------------
        # Resolve actual Track objects
        # ----------------------------------------------------

        fragments = []

        for track_id in valid_track_ids:

            track = track_lookup.get(
                track_id
            )

            if track is not None:
                fragments.append(track)

        # ----------------------------------------------------
        # Sort fragments chronologically
        # ----------------------------------------------------

        fragments.sort(
            key=lambda track: (
                _extract_start_frame(track)
                if _extract_start_frame(track)
                is not None
                else -1
            )
        )

        fragment_count = len(
            fragments
        )

        total_fragments += (
            fragment_count
        )

        if fragment_count > 0:

            identities_with_fragments += 1

        fragment_counts.append(
            fragment_count
        )

        # ====================================================
        # FRAGMENT DURATIONS
        # ====================================================

        durations = []

        for track in fragments:

            start = _extract_start_frame(
                track
            )

            end = _extract_end_frame(
                track
            )

            if (
                start is None
                or end is None
            ):
                continue

            try:

                duration = (
                    int(end)
                    - int(start)
                    + 1
                )

                durations.append(
                    duration
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

        # ====================================================
        # TEMPORAL GAPS
        # ====================================================

        gaps = []

        gap_details = []

        for previous, current in zip(
            fragments,
            fragments[1:],
        ):

            gap = _calculate_fragment_gap(
                previous,
                current,
            )

            if gap is None:
                continue

            gaps.append(gap)

            gap_details.append(
                {
                    "previous_track_id": (
                        _extract_track_id(
                            previous
                        )
                    ),
                    "previous_end_frame": (
                        _extract_end_frame(
                            previous
                        )
                    ),
                    "current_track_id": (
                        _extract_track_id(
                            current
                        )
                    ),
                    "current_start_frame": (
                        _extract_start_frame(
                            current
                        )
                    ),
                    "gap_frames": gap,
                }
            )

        # ====================================================
        # IDENTITY METADATA
        # ====================================================

        class_name = _get_value(
            identity,
            "class_name",
            "class_label",
            "label",
            "class",
            default="unknown",
        )

        identity_first_frame = _get_value(
            identity,
            "first_frame",
            default=None,
        )

        identity_last_frame = _get_value(
            identity,
            "last_frame",
            default=None,
        )

        # ====================================================
        # REPORT
        # ====================================================

        report = {

            "global_id": global_id,

            "class_name": class_name,

            "fragment_count": fragment_count,

            "track_ids": valid_track_ids,

            "missing_track_ids": (
                missing_track_ids
            ),

            "durations": durations,

            "gaps": gaps,

            "gap_details": gap_details,

            "total_tracked_frames": (
                sum(durations)
            ),

            "max_fragment_duration": (
                max(durations)
                if durations
                else 0
            ),

            "min_fragment_duration": (
                min(durations)
                if durations
                else 0
            ),

            "average_fragment_duration": (
                mean(durations)
                if durations
                else 0.0
            ),

            "max_gap": (
                max(gaps)
                if gaps
                else 0
            ),

            "average_gap": (
                mean(gaps)
                if gaps
                else 0.0
            ),

            "identity_first_frame": (
                identity_first_frame
            ),

            "identity_last_frame": (
                identity_last_frame
            ),

            "has_long_gap": (
                max(gaps)
                >= long_gap_threshold
                if gaps
                else False
            ),

            "is_heavily_fragmented": (
                fragment_count
                >= heavy_fragment_threshold
            ),
        }

        fragment_reports.append(
            report
        )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    multi_fragment_identities = [
        report
        for report in fragment_reports
        if report["fragment_count"] > 1
    ]

    heavily_fragmented_identities = [
        report
        for report in fragment_reports
        if report["fragment_count"]
        >= heavy_fragment_threshold
    ]

    long_gap_identities = [
        report
        for report in fragment_reports
        if report["max_gap"]
        >= long_gap_threshold
    ]

    # ========================================================
    # UNASSIGNED TRACKS
    # ========================================================

    unassigned_track_ids = sorted(
        all_track_ids
        - assigned_track_ids
    )

    # ========================================================
    # FRAGMENT COUNT DISTRIBUTION
    # ========================================================

    zero_fragment_identities = [
        report
        for report in fragment_reports
        if report["fragment_count"] == 0
    ]

    single_fragment_identities = [
        report
        for report in fragment_reports
        if report["fragment_count"] == 1
    ]

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        # ----------------------------------------------------
        # Basic
        # ----------------------------------------------------

        "global_identities": len(
            identities
        ),

        "track_fragments_available": len(
            all_track_ids
        ),

        "identities_with_fragments": (
            identities_with_fragments
        ),

        "identities_without_fragments": len(
            zero_fragment_identities
        ),

        "total_fragments": (
            total_fragments
        ),

        "assigned_track_fragments": len(
            assigned_track_ids
        ),

        "unassigned_track_fragments": len(
            unassigned_track_ids
        ),

        "unassigned_track_ids": (
            unassigned_track_ids
        ),

        # ----------------------------------------------------
        # Fragment distribution
        # ----------------------------------------------------

        "average_fragments_per_identity": (
            total_fragments
            / len(identities)
            if identities
            else 0.0
        ),

        "median_fragments_per_identity": (
            median(fragment_counts)
            if fragment_counts
            else 0.0
        ),

        "max_fragments_for_identity": (
            max(fragment_counts)
            if fragment_counts
            else 0
        ),

        # ----------------------------------------------------
        # Fragment categories
        # ----------------------------------------------------

        "zero_fragment_identities": len(
            zero_fragment_identities
        ),

        "single_fragment_identities": len(
            single_fragment_identities
        ),

        "multi_fragment_identities": len(
            multi_fragment_identities
        ),

        "heavily_fragmented_identities": len(
            heavily_fragmented_identities
        ),

        "long_gap_identities": len(
            long_gap_identities
        ),

        # ----------------------------------------------------
        # Thresholds
        # ----------------------------------------------------

        "long_gap_threshold": (
            long_gap_threshold
        ),

        "heavy_fragment_threshold": (
            heavy_fragment_threshold
        ),

        # ----------------------------------------------------
        # Complete reports
        # ----------------------------------------------------

        "fragment_reports": (
            fragment_reports
        ),
    }

    return summary


# ============================================================
# PRINT FRAGMENT ANALYSIS
# ============================================================


def print_global_identity_fragment_analysis(
    fragment_analysis,
    limit: int = 20,
):
    """
    Print human-readable global identity fragment analysis.
    """

    print()
    print("=" * 70)
    print(
        "GLOBAL IDENTITY FRAGMENT ANALYSIS"
    )
    print("=" * 70)

    # ========================================================
    # BASIC COUNTS
    # ========================================================

    print(
        f"Global identities: "
        f"{fragment_analysis['global_identities']}"
    )

    print(
        f"ByteTrack fragments available: "
        f"{fragment_analysis['track_fragments_available']}"
    )

    print(
        f"Identities with fragments: "
        f"{fragment_analysis['identities_with_fragments']}"
    )

    print(
        f"Identities without fragments: "
        f"{fragment_analysis['identities_without_fragments']}"
    )

    print(
        f"Total ByteTrack fragments assigned: "
        f"{fragment_analysis['total_fragments']}"
    )

    print(
        f"Assigned track fragments: "
        f"{fragment_analysis['assigned_track_fragments']}"
    )

    print(
        f"Unassigned track fragments: "
        f"{fragment_analysis['unassigned_track_fragments']}"
    )

    # ========================================================
    # FRAGMENT DISTRIBUTION
    # ========================================================

    print()

    print(
        f"Average fragments / identity: "
        f"{fragment_analysis['average_fragments_per_identity']:.2f}"
    )

    print(
        f"Median fragments / identity: "
        f"{fragment_analysis['median_fragments_per_identity']:.2f}"
    )

    print(
        f"Maximum fragments for one identity: "
        f"{fragment_analysis['max_fragments_for_identity']}"
    )

    print(
        f"Single-fragment identities: "
        f"{fragment_analysis['single_fragment_identities']}"
    )

    print(
        f"Multi-fragment identities: "
        f"{fragment_analysis['multi_fragment_identities']}"
    )

    print(
        f"Heavily fragmented identities "
        f"(>= {fragment_analysis['heavy_fragment_threshold']}): "
        f"{fragment_analysis['heavily_fragmented_identities']}"
    )

    print(
        f"Identities with long gaps "
        f"(>= {fragment_analysis['long_gap_threshold']} frames): "
        f"{fragment_analysis['long_gap_identities']}"
    )

    # ========================================================
    # MOST FRAGMENTED IDENTITIES
    # ========================================================

    reports = sorted(
        fragment_analysis[
            "fragment_reports"
        ],
        key=lambda item: (
            item["fragment_count"],
            item["total_tracked_frames"],
        ),
        reverse=True,
    )

    print()
    print(
        "MOST FRAGMENTED GLOBAL IDENTITIES"
    )

    print("-" * 70)

    shown = 0

    for report in reports:

        if report["fragment_count"] <= 1:
            continue

        print(
            f"GID {report['global_id']} | "
            f"{report['class_name']} | "
            f"{report['fragment_count']} fragments | "
            f"{report['total_tracked_frames']} frames | "
            f"max gap {report['max_gap']} frames | "
            f"tracks {report['track_ids']}"
        )

        shown += 1

        if shown >= limit:
            break

    if shown == 0:

        print(
            "No multi-fragment identities found."
        )

    # ========================================================
    # LARGEST FRAGMENT GAPS
    # ========================================================

    reports_by_gap = sorted(
        fragment_analysis[
            "fragment_reports"
        ],
        key=lambda item: (
            item["max_gap"],
            item["fragment_count"],
        ),
        reverse=True,
    )

    print()
    print(
        "LARGEST FRAGMENT GAPS"
    )

    print("-" * 70)

    shown = 0

    for report in reports_by_gap:

        if report["max_gap"] <= 0:
            continue

        print(
            f"GID {report['global_id']} | "
            f"{report['class_name']} | "
            f"max gap {report['max_gap']} frames | "
            f"{report['fragment_count']} fragments"
        )

        for gap in report["gap_details"]:

            if (
                gap["gap_frames"]
                == report["max_gap"]
            ):

                print(
                    f"    "
                    f"Track {gap['previous_track_id']} "
                    f"ended at frame "
                    f"{gap['previous_end_frame']} "
                    f"→ "
                    f"Track {gap['current_track_id']} "
                    f"started at frame "
                    f"{gap['current_start_frame']} "
                    f""
                    f"(gap={gap['gap_frames']})"
                )

        shown += 1

        if shown >= limit:
            break

    if shown == 0:

        print(
            "No temporal gaps found."
        )

    # ========================================================
    # UNASSIGNED TRACKS
    # ========================================================

    unassigned = (
        fragment_analysis[
            "unassigned_track_ids"
        ]
    )

    print()

    print(
        "UNASSIGNED BYTE TRACK FRAGMENTS"
    )

    print("-" * 70)

    if unassigned:

        print(
            f"Count: {len(unassigned)}"
        )

        print(
            f"IDs: {unassigned}"
        )

    else:

        print(
            "All available ByteTrack fragments "
            "are assigned to global identities."
        )

    print("=" * 70)


# ============================================================
# PIPELINE
# ============================================================


def run_identity_pipeline(
    video_path: str,
    model_path: str,
    fps: float | None = None,
    imgsz: int = 1280,
    confidence: float = 0.25,

    # --------------------------------------------------------
    # TEMPORAL / GEOMETRY
    # --------------------------------------------------------

    max_temporal_gap: int = 75,
    max_spatial_distance: float = 300.0,
    max_motion_difference: float = 80.0,

    # --------------------------------------------------------
    # APPEARANCE
    # --------------------------------------------------------

    minimum_appearance_similarity: float = 0.72,
    appearance_normal_similarity: float = 0.78,
    appearance_strong_similarity: float = 0.85,

    appearance_recent_embeddings: int = 20,
    appearance_average_top_k: int = 5,

    # --------------------------------------------------------
    # MATCH WEIGHTS
    # --------------------------------------------------------

    temporal_weight: float = 0.10,
    spatial_weight: float = 0.25,
    motion_weight: float = 0.20,
    appearance_weight: float = 0.35,
    team_weight: float = 0.07,
    jersey_weight: float = 0.03,

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    minimum_match_score: float = 0.60,

    # --------------------------------------------------------
    # GEOMETRY QUALITY
    # --------------------------------------------------------

    strong_spatial_ratio: float = 0.35,
    strong_motion_ratio: float = 0.40,

    # --------------------------------------------------------
    # LONG TEMPORAL GAP
    # --------------------------------------------------------

    long_gap_ratio: float = 0.50,
    long_gap_min_appearance: float = 0.78,

    # --------------------------------------------------------
    # WEAK APPEARANCE
    # --------------------------------------------------------

    weak_appearance_max_spatial_ratio: float = 0.20,
    weak_appearance_max_motion_ratio: float = 0.25,
    weak_appearance_min_score: float = 0.72,

    # --------------------------------------------------------
    # STRONG APPEARANCE
    # --------------------------------------------------------

    strong_appearance_max_spatial_ratio: float = 0.85,
    strong_appearance_max_motion_ratio: float = 0.90,

    # --------------------------------------------------------
    # MOTION SAFETY
    # --------------------------------------------------------

    minimum_observations_for_motion: int = 2,

    # --------------------------------------------------------
    # GLOBAL IDENTITY FRAGMENT ANALYSIS
    # --------------------------------------------------------

    analyze_fragments: bool = True,

    fragment_analysis_limit: int = 20,

    fragment_long_gap_threshold: int = 75,

    fragment_heavy_threshold: int = 5,

    # --------------------------------------------------------
    # PHASE 7 VISUAL VALIDATION
    # --------------------------------------------------------

    generate_visual_validation: bool = False,

    visual_validation_output_dir: str = (
        "media/identity_validation"
    ),

    visual_validation_limit: int | None = 20,

) -> GlobalIdentityManager:
    """
    Run ByteTrack followed by Global Identity Manager V5.1.

    Returns:
        GlobalIdentityManager instance.

    Notes:
        Fragment analysis and visual validation are diagnostic
        operations only. They do not modify identity decisions.
    """

    # ========================================================
    # PIPELINE HEADER
    # ========================================================

    print()
    print("=" * 70)
    print(
        "IDENTITY PIPELINE V5.1"
    )
    print("=" * 70)

    # ========================================================
    # BYTE TRACK
    # ========================================================

    print()
    print(
        "Running ByteTrack..."
    )

    tracker = run_tracking(
        video_path=video_path,
        model_path=model_path,
        fps=fps,
        imgsz=imgsz,
        confidence=confidence,
    )

    track_history = (
        tracker.get_track_history()
    )

    logger.info(
        "ByteTrack produced %d track fragments",
        len(track_history),
    )

    print(
        f"ByteTrack fragments: "
        f"{len(track_history)}"
    )

    # ========================================================
    # IDENTITY MATCHER V5.1
    # ========================================================

    matcher = IdentityMatcher(

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        max_temporal_gap=(
            max_temporal_gap
        ),

        max_spatial_distance=(
            max_spatial_distance
        ),

        max_motion_difference=(
            max_motion_difference
        ),

        # ----------------------------------------------------
        # Appearance
        # ----------------------------------------------------

        minimum_appearance_similarity=(
            minimum_appearance_similarity
        ),

        appearance_normal_similarity=(
            appearance_normal_similarity
        ),

        appearance_strong_similarity=(
            appearance_strong_similarity
        ),

        appearance_recent_embeddings=(
            appearance_recent_embeddings
        ),

        appearance_average_top_k=(
            appearance_average_top_k
        ),

        # ----------------------------------------------------
        # Weights
        # ----------------------------------------------------

        temporal_weight=(
            temporal_weight
        ),

        spatial_weight=(
            spatial_weight
        ),

        motion_weight=(
            motion_weight
        ),

        appearance_weight=(
            appearance_weight
        ),

        team_weight=(
            team_weight
        ),

        jersey_weight=(
            jersey_weight
        ),

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        minimum_match_score=(
            minimum_match_score
        ),

        # ----------------------------------------------------
        # Geometry quality
        # ----------------------------------------------------

        strong_spatial_ratio=(
            strong_spatial_ratio
        ),

        strong_motion_ratio=(
            strong_motion_ratio
        ),

        # ----------------------------------------------------
        # Long gap
        # ----------------------------------------------------

        long_gap_ratio=(
            long_gap_ratio
        ),

        long_gap_min_appearance=(
            long_gap_min_appearance
        ),

        # ----------------------------------------------------
        # Weak appearance
        # ----------------------------------------------------

        weak_appearance_max_spatial_ratio=(
            weak_appearance_max_spatial_ratio
        ),

        weak_appearance_max_motion_ratio=(
            weak_appearance_max_motion_ratio
        ),

        weak_appearance_min_score=(
            weak_appearance_min_score
        ),

        # ----------------------------------------------------
        # Strong appearance
        # ----------------------------------------------------

        strong_appearance_max_spatial_ratio=(
            strong_appearance_max_spatial_ratio
        ),

        strong_appearance_max_motion_ratio=(
            strong_appearance_max_motion_ratio
        ),

        # ----------------------------------------------------
        # Motion
        # ----------------------------------------------------

        minimum_observations_for_motion=(
            minimum_observations_for_motion
        ),
    )

    # ========================================================
    # MATCHER CONFIGURATION
    # ========================================================

    print()
    print("=" * 70)
    print(
        "IDENTITY MATCHER CONFIGURATION"
    )
    print("=" * 70)

    print(
        f"Matcher version: "
        f"{matcher.VERSION}"
    )

    print(
        f"Max temporal gap: "
        f"{matcher.max_temporal_gap}"
    )

    print(
        f"Max spatial distance: "
        f"{matcher.max_spatial_distance}"
    )

    print(
        f"Max motion difference: "
        f"{matcher.max_motion_difference}"
    )

    print(
        f"Minimum appearance: "
        f"{matcher.minimum_appearance_similarity:.3f}"
    )

    print(
        f"Normal appearance: "
        f"{matcher.appearance_normal_similarity:.3f}"
    )

    print(
        f"Strong appearance: "
        f"{matcher.appearance_strong_similarity:.3f}"
    )

    print(
        f"Minimum match score: "
        f"{matcher.minimum_match_score:.3f}"
    )

    print(
        f"Recent appearance embeddings: "
        f"{matcher.appearance_recent_embeddings}"
    )

    print(
        f"Appearance top-k: "
        f"{matcher.appearance_average_top_k}"
    )

    # ========================================================
    # GLOBAL IDENTITY MANAGER
    # ========================================================

    manager = GlobalIdentityManager(
        matcher=matcher,
        minimum_match_score=(
            minimum_match_score
        ),
    )

    # ========================================================
    # BUILD IDENTITIES
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BUILDING GLOBAL IDENTITIES"
    )
    print("=" * 70)

    manager.build_identities(
        track_history.values()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    statistics = (
        manager.get_statistics()
    )

    print()
    print("=" * 70)
    print(
        "GLOBAL IDENTITY SUMMARY"
    )
    print("=" * 70)

    print(
        f"ByteTrack fragments: "
        f"{statistics['tracks_processed']}"
    )

    print(
        f"Tracks merged: "
        f"{statistics['tracks_merged']}"
    )

    print(
        f"New global identities: "
        f"{statistics['new_identities']}"
    )

    print(
        f"Global identities: "
        f"{statistics['global_identities']}"
    )

    print(
        f"Matcher version: "
        f"{statistics.get('matcher_version', 'unknown')}"
    )

    # ========================================================
    # IDENTITIES BY CLASS
    # ========================================================

    print()
    print(
        "Global identities by class:"
    )

    for class_name, count in sorted(
        statistics[
            "identities_by_class"
        ].items()
    ):

        print(
            f"  {class_name}: {count}"
        )

    # ========================================================
    # MATCHING DIAGNOSTICS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "MATCHING DIAGNOSTICS"
    )
    print("=" * 70)

    print(
        f"Candidates checked: "
        f"{statistics['candidates_checked']}"
    )

    print(
        f"Accepted matches: "
        f"{statistics['accepted_matches']}"
    )

    print(
        f"Rejected matches: "
        f"{statistics['rejected_matches']}"
    )

    # ========================================================
    # REJECTION REASONS
    # ========================================================

    rejection_reasons = (
        statistics.get(
            "rejection_reasons",
            {},
        )
    )

    if rejection_reasons:

        print()
        print(
            "Rejection reasons:"
        )

        for reason, count in sorted(
            rejection_reasons.items(),
            key=lambda item: (
                item[1],
                item[0],
            ),
            reverse=True,
        ):

            print(
                f"  {reason}: {count}"
            )

    # ========================================================
    # ACCEPTED SCORE
    # ========================================================

    accepted_score = (
        statistics.get(
            "accepted_score"
        )
    )

    if accepted_score:

        print()
        print(
            "Accepted match scores:"
        )

        print(
            f"  Min: "
            f"{accepted_score['min']:.3f}"
        )

        print(
            f"  Max: "
            f"{accepted_score['max']:.3f}"
        )

        print(
            f"  Avg: "
            f"{accepted_score['avg']:.3f}"
        )

    # ========================================================
    # APPEARANCE DIAGNOSTICS
    # ========================================================

    appearance = (
        statistics.get(
            "appearance_diagnostics",
            {},
        )
    )

    print()
    print("=" * 70)
    print(
        "APPEARANCE DIAGNOSTICS"
    )
    print("=" * 70)

    print(
        f"Candidates requiring appearance: "
        f"{appearance.get('candidates_requiring_appearance', 0)}"
    )

    print(
        f"Comparisons: "
        f"{appearance.get('comparisons', 0)}"
    )

    print(
        f"Valid comparisons: "
        f"{appearance.get('valid_comparisons', 0)}"
    )

    print(
        f"Missing embeddings: "
        f"{appearance.get('missing_embeddings', 0)}"
    )

    print(
        f"Strong matches: "
        f"{appearance.get('strong_matches', 0)}"
    )

    print(
        f"Weak matches: "
        f"{appearance.get('weak_matches', 0)}"
    )

    print(
        f"Accepted appearance matches: "
        f"{appearance.get('accepted_matches', 0)}"
    )

    print(
        f"Rejected appearance matches: "
        f"{appearance.get('rejected_matches', 0)}"
    )

    # ========================================================
    # APPEARANCE SIMILARITY
    # ========================================================

    similarity = (
        appearance.get(
            "similarity",
            {},
        )
    )

    if similarity:

        print()
        print(
            "Appearance similarity:"
        )

        print(
            f"  Min: "
            f"{similarity.get('min', 0.0):.3f}"
        )

        print(
            f"  Max: "
            f"{similarity.get('max', 0.0):.3f}"
        )

        print(
            f"  Avg: "
            f"{similarity.get('avg', 0.0):.3f}"
        )

    # ========================================================
    # EVALUATION
    # ========================================================

    evaluation = (
        manager.get_identity_matching_evaluation()
    )

    evaluation_summary = (
        manager.get_identity_matching_evaluation_summary()
    )

    print()
    print("=" * 70)
    print(
        "IDENTITY MATCHING EVALUATION"
    )
    print("=" * 70)

    print(
        f"Candidates: "
        f"{evaluation_summary.get('candidates', 0)}"
    )

    print(
        f"Appearance comparisons: "
        f"{evaluation_summary.get('appearance_comparisons', 0)}"
    )

    print(
        f"Appearance accepted: "
        f"{evaluation_summary.get('appearance_accepted', 0)}"
    )

    print(
        f"Appearance rejected: "
        f"{evaluation_summary.get('appearance_rejected', 0)}"
    )

    print(
        f"Strong matches: "
        f"{evaluation_summary.get('strong_matches', 0)}"
    )

    print(
        f"Average appearance similarity: "
        f"{evaluation_summary.get('average_similarity', 0.0):.3f}"
    )

    print(
        f"Accepted average similarity: "
        f"{evaluation_summary.get('accepted_average_similarity', 0.0):.3f}"
    )

    print(
        f"Risky matches: "
        f"{evaluation_summary.get('risky_matches', 0)}"
    )

    print(
        f"Recommendation: "
        f"{evaluation_summary.get('recommendation', 'N/A')}"
    )

    logger.debug(
        "Full identity matching evaluation: %s",
        evaluation,
    )

    # ========================================================
    # TEMPORAL GAP DIAGNOSTIC
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TEMPORAL GAP DIAGNOSTIC"
    )
    print("=" * 70)

    temporal_rejections = (
        rejection_reasons.get(
            "temporal_gap_too_large",
            0,
        )
    )

    print(
        f"Temporal-gap rejections: "
        f"{temporal_rejections}"
    )

    print(
        f"Configured maximum temporal gap: "
        f"{max_temporal_gap} frames"
    )

    if temporal_rejections > 0:

        print()
        print(
            "NOTE:"
        )

        print(
            "Tracks rejected by the temporal gate "
            "never reach spatial, motion, or appearance "
            "matching."
        )

        print(
            "This is intentional in IdentityMatcher V5.1."
        )

    # ========================================================
    # GLOBAL IDENTITY FRAGMENT ANALYSIS
    # ========================================================

    fragment_analysis = None

    if analyze_fragments:

        try:

            fragment_analysis = (
                analyze_global_identity_fragments(
                    manager=manager,
                    track_history=(
                        track_history.values()
                    ),
                    long_gap_threshold=(
                        fragment_long_gap_threshold
                    ),
                    heavy_fragment_threshold=(
                        fragment_heavy_threshold
                    ),
                )
            )

            print_global_identity_fragment_analysis(
                fragment_analysis,
                limit=(
                    fragment_analysis_limit
                ),
            )

            # ------------------------------------------------
            # Make analysis available to callers
            # ------------------------------------------------

            manager.fragment_analysis = (
                fragment_analysis
            )

            logger.info(
                "Global identity fragment analysis "
                "completed: identities=%d "
                "fragments=%d "
                "multi_fragment=%d",
                fragment_analysis[
                    "global_identities"
                ],
                fragment_analysis[
                    "total_fragments"
                ],
                fragment_analysis[
                    "multi_fragment_identities"
                ],
            )

        except Exception as exc:

            logger.exception(
                "Global identity fragment analysis failed"
            )

            print()
            print(
                "WARNING: Global identity "
                "fragment analysis failed."
            )

            print(
                f"Reason: {exc}"
            )

            manager.fragment_analysis = None

    else:

        manager.fragment_analysis = None

    # ========================================================
    # UNDER-MERGING DIAGNOSTIC
    # ========================================================

    print()
    print("=" * 70)
    print(
        "UNDER-MERGING DIAGNOSTIC"
    )
    print("=" * 70)

    print(
        "High appearance similarity with rejected "
        "geometry/score should be reviewed separately."
    )

    print(
        "Fragment analysis shows how many ByteTrack "
        "fragments form each global identity."
    )

    if fragment_analysis is not None:

        multi_fragment_count = (
            fragment_analysis[
                "multi_fragment_identities"
            ]
        )

        heavy_fragment_count = (
            fragment_analysis[
                "heavily_fragmented_identities"
            ]
        )

        long_gap_count = (
            fragment_analysis[
                "long_gap_identities"
            ]
        )

        print()

        print(
            f"Multi-fragment identities: "
            f"{multi_fragment_count}"
        )

        print(
            f"Heavily fragmented identities: "
            f"{heavy_fragment_count}"
        )

        print(
            f"Identities with long gaps: "
            f"{long_gap_count}"
        )

        if heavy_fragment_count > 0:

            print()

            print(
                "WARNING: Some identities contain "
                "many ByteTrack fragments."
            )

            print(
                "These identities should be reviewed "
                "for possible unstable tracking or "
                "over-fragmentation."
            )

    print(
        "Phase 7 visual validation can be used "
        "to inspect suspicious relationships."
    )

    print("=" * 70)

    # ========================================================
    # OPTIONAL PHASE 7 VISUAL VALIDATION
    # ========================================================

    if generate_visual_validation:

        from ai_engine.identity.identity_visual_validation import (
            IdentityVisualValidation,
        )

        validator = IdentityVisualValidation(
            video_path=video_path,
            tracks=track_history.values(),
            diagnostics=(
                manager.match_diagnostics
            ),
            output_dir=(
                visual_validation_output_dir
            ),
            identities=manager.identities,
        )

        print()
        print("=" * 70)
        print(
            "PHASE 7 VISUAL VALIDATION"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Risky matches
        # ----------------------------------------------------

        risky_paths = (
            validator.generate_risk_report(
                limit=(
                    visual_validation_limit
                )
            )
        )

        # ----------------------------------------------------
        # High appearance similarity but rejected
        # ----------------------------------------------------

        high_similarity_paths = (
            validator.generate_high_similarity_report(
                minimum_similarity=(
                    appearance_strong_similarity
                ),
                limit=(
                    visual_validation_limit
                ),
            )
        )

        # ----------------------------------------------------
        # Appearance mismatch
        # ----------------------------------------------------

        mismatch_paths = (
            validator.generate_appearance_mismatch_report(
                maximum_similarity=(
                    minimum_appearance_similarity
                ),
                limit=(
                    visual_validation_limit
                ),
            )
        )

        print(
            f"Risk images: "
            f"{len(risky_paths)}"
        )

        print(
            f"High-similarity rejection images: "
            f"{len(high_similarity_paths)}"
        )

        print(
            f"Appearance mismatch images: "
            f"{len(mismatch_paths)}"
        )

        print(
            f"Output directory: "
            f"{visual_validation_output_dir}"
        )

        print("=" * 70)

    # ========================================================
    # RETURN
    # ========================================================

    return manager