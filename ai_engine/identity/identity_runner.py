"""
Global Identity Pipeline Runner.

Pipeline:

    Video
      ↓
    YOLO Detector
      ↓
    ByteTrack
      ↓
    ByteTrack track history
      ↓
    Global Identity Manager
      ↓
    Persistent global identities

Optional Phase 7:

    Global Identity diagnostics
      ↓
    Identity Visual Validation
      ↓
    Annotated validation images
"""

from __future__ import annotations

import logging
from typing import Optional

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


def run_identity_pipeline(
    video_path: str,
    model_path: str,
    fps: float | None = None,
    imgsz: int = 1280,
    confidence: float = 0.25,
    max_temporal_gap: int = 50,
    max_spatial_distance: float = 250.0,
    max_motion_difference: float = 80.0,
    minimum_match_score: float = 0.55,
    minimum_appearance_similarity: float = 0.70,
    appearance_strong_similarity: float = 0.85,
    generate_visual_validation: bool = False,
    visual_validation_output_dir: str = (
        "media/identity_validation"
    ),
    visual_validation_limit: int | None = 20,
) -> GlobalIdentityManager:
    """
    Run ByteTrack followed by global identity merging.

    Returns:
        GlobalIdentityManager instance.

    Optional Phase 7 visual validation can be enabled with:

        generate_visual_validation=True

    Matching decisions are not modified by visual validation.
    """

    # ========================================================
    # BYTE TRACK
    # ========================================================

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

    # ========================================================
    # IDENTITY MATCHER
    # ========================================================

    matcher = IdentityMatcher(
        max_temporal_gap=max_temporal_gap,
        max_spatial_distance=max_spatial_distance,
        max_motion_difference=max_motion_difference,
        minimum_appearance_similarity=(
            minimum_appearance_similarity
        ),
        appearance_strong_similarity=(
            appearance_strong_similarity
        ),
    )

    # ========================================================
    # GLOBAL IDENTITY MANAGER
    # ========================================================

    manager = GlobalIdentityManager(
        matcher=matcher,
        minimum_match_score=minimum_match_score,
    )

    manager.build_identities(
        track_history.values()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    statistics = manager.get_statistics()

    print()
    print("=" * 70)
    print("GLOBAL IDENTITY SUMMARY")
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

    # ========================================================
    # IDENTITIES BY CLASS
    # ========================================================

    print()
    print("Global identities by class:")

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
    print("Matching diagnostics:")

    print(
        f"  Candidates checked: "
        f"{statistics['candidates_checked']}"
    )

    print(
        f"  Accepted matches: "
        f"{statistics['accepted_matches']}"
    )

    print(
        f"  Rejected matches: "
        f"{statistics['rejected_matches']}"
    )

    # ========================================================
    # ACCEPTED SCORE
    # ========================================================

    accepted_score = statistics.get(
        "accepted_score"
    )

    if accepted_score:

        print()
        print("Accepted match scores:")

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
    # APPEARANCE
    # ========================================================

    appearance = statistics.get(
        "appearance_diagnostics",
        {},
    )

    print()
    print("Appearance diagnostics:")

    print(
        f"  Comparisons: "
        f"{appearance.get('comparisons', 0)}"
    )

    print(
        f"  Valid comparisons: "
        f"{appearance.get('valid_comparisons', 0)}"
    )

    print(
        f"  Missing embeddings: "
        f"{appearance.get('missing_embeddings', 0)}"
    )

    print(
        f"  Strong matches: "
        f"{appearance.get('strong_matches', 0)}"
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
    print("Identity matching evaluation:")

    print(
        f"  Appearance comparisons: "
        f"{evaluation_summary['appearance_comparisons']}"
    )

    print(
        f"  Appearance accepted: "
        f"{evaluation_summary['appearance_accepted']}"
    )

    print(
        f"  Appearance rejected: "
        f"{evaluation_summary['appearance_rejected']}"
    )

    print(
        f"  Strong matches: "
        f"{evaluation_summary['strong_matches']}"
    )

    print(
        f"  Average appearance similarity: "
        f"{evaluation_summary['average_similarity']:.3f}"
    )

    print(
        f"  Accepted average similarity: "
        f"{evaluation_summary['accepted_average_similarity']:.3f}"
    )

    print(
        f"  Risky matches: "
        f"{evaluation_summary['risky_matches']}"
    )

    print(
        f"  Recommendation: "
        f"{evaluation_summary['recommendation']}"
    )

    # Keep the full report available for callers/debugging.
    logger.debug(
        "Full identity matching evaluation: %s",
        evaluation,
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
            diagnostics=manager.match_diagnostics,
            output_dir=visual_validation_output_dir,
            identities=manager.identities,
        )

        print()
        print("=" * 70)
        print("PHASE 7 VISUAL VALIDATION")
        print("=" * 70)

        risky_paths = (
            validator.generate_risk_report(
                limit=visual_validation_limit
            )
        )

        high_similarity_paths = (
            validator.generate_high_similarity_report(
                minimum_similarity=(
                    appearance_strong_similarity
                ),
                limit=visual_validation_limit,
            )
        )

        mismatch_paths = (
            validator.generate_appearance_mismatch_report(
                maximum_similarity=(
                    minimum_appearance_similarity
                ),
                limit=visual_validation_limit,
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

    return manager