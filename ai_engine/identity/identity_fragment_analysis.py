from collections import Counter


def analyze_global_identity_fragments(
    manager,
    tracks,
    limit: int | None = 30,
    long_gap_threshold: int = 75,
) -> dict:
    """
    Analyze how ByteTrack fragments are distributed
    across Global Identity Manager identities.

    This is diagnostic only.

    The authoritative identity -> fragment relationship
    is:

        Identity.source_track_ids

    Each source_track_id corresponds to:

        Track.local_id

    Temporal gaps use the same definition as
    IdentityMatcher V5.1:

        gap = next_track.first_frame
              - previous_track.last_frame
              - 1
    """

    tracks = list(tracks)

    # --------------------------------------------------------
    # TRACK LOOKUP
    # --------------------------------------------------------

    track_by_id = {
        track.local_id: track
        for track in tracks
    }

    # --------------------------------------------------------
    # IDENTITY ANALYSIS
    # --------------------------------------------------------

    identity_reports = []

    total_fragments_assigned = 0
    identities_with_fragments = 0
    identities_with_long_gaps = 0
    multi_fragment_identities = 0
    heavily_fragmented_identities = 0

    all_fragment_ids = set()

    for identity in manager.identities.values():

        fragment_ids = list(
            getattr(
                identity,
                "source_track_ids",
                [],
            )
        )

        fragment_ids = [
            fragment_id
            for fragment_id in fragment_ids
            if fragment_id in track_by_id
        ]

        fragment_tracks = [
            track_by_id[fragment_id]
            for fragment_id in fragment_ids
        ]

        fragment_tracks.sort(
            key=lambda track: (
                track.first_frame
                if track.first_frame is not None
                else float("inf")
            )
        )

        fragment_count = len(
            fragment_tracks
        )

        total_fragments_assigned += (
            fragment_count
        )

        all_fragment_ids.update(
            fragment_ids
        )

        if fragment_count > 0:
            identities_with_fragments += 1

        if fragment_count > 1:
            multi_fragment_identities += 1

        if fragment_count >= 5:
            heavily_fragmented_identities += 1

        # ----------------------------------------------------
        # TEMPORAL GAPS
        # ----------------------------------------------------

        gaps = []

        for previous, current in zip(
            fragment_tracks,
            fragment_tracks[1:],
        ):

            if (
                previous.last_frame is None
                or current.first_frame is None
            ):
                continue

            gap = (
                current.first_frame
                - previous.last_frame
                - 1
            )

            if gap < 0:
                continue

            gaps.append(gap)

        max_gap = (
            max(gaps)
            if gaps
            else 0
        )

        long_gaps = [
            gap
            for gap in gaps
            if gap >= long_gap_threshold
        ]

        if long_gaps:
            identities_with_long_gaps += 1

        # ----------------------------------------------------
        # FRAME RANGE
        # ----------------------------------------------------

        first_frame = (
            fragment_tracks[0].first_frame
            if fragment_tracks
            else None
        )

        last_frame = (
            fragment_tracks[-1].last_frame
            if fragment_tracks
            else None
        )

        total_frames = 0

        for track in fragment_tracks:

            if (
                track.first_frame is not None
                and track.last_frame is not None
            ):
                total_frames += (
                    track.last_frame
                    - track.first_frame
                    + 1
                )

        identity_reports.append(
            {
                "identity_id": (
                    identity.identity_id
                ),
                "class_name": (
                    identity.class_name
                ),
                "fragment_ids": (
                    fragment_ids
                ),
                "fragment_count": (
                    fragment_count
                ),
                "first_frame": (
                    first_frame
                ),
                "last_frame": (
                    last_frame
                ),
                "total_frames": (
                    total_frames
                ),
                "max_gap": (
                    max_gap
                ),
                "long_gaps": (
                    long_gaps
                ),
            }
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    identity_reports.sort(
        key=lambda report: (
            report["fragment_count"],
            report["total_frames"],
            report["max_gap"],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    global_identity_count = len(
        manager.identities
    )

    average_fragments = (
        total_fragments_assigned
        / global_identity_count
        if global_identity_count
        else 0.0
    )

    fragment_counts = [
        report["fragment_count"]
        for report in identity_reports
    ]

    median_fragments = (
        sorted(fragment_counts)[
            len(fragment_counts) // 2
        ]
        if fragment_counts
        else 0
    )

    # --------------------------------------------------------
    # CONSISTENCY CHECK
    # --------------------------------------------------------

    processed_tracks = (
        getattr(
            manager,
            "total_tracks_processed",
            0,
        )
    )

    merged_tracks = (
        getattr(
            manager,
            "total_tracks_merged",
            0,
        )
    )

    expected_assignments = (
        processed_tracks
    )

    assignment_consistent = (
        total_fragments_assigned
        == expected_assignments
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {
        "global_identities": (
            global_identity_count
        ),
        "identities_with_fragments": (
            identities_with_fragments
        ),
        "total_fragments_assigned": (
            total_fragments_assigned
        ),
        "average_fragments_per_identity": (
            average_fragments
        ),
        "median_fragments_per_identity": (
            median_fragments
        ),
        "maximum_fragments": (
            max(fragment_counts)
            if fragment_counts
            else 0
        ),
        "multi_fragment_identities": (
            multi_fragment_identities
        ),
        "heavily_fragmented_identities": (
            heavily_fragmented_identities
        ),
        "identities_with_long_gaps": (
            identities_with_long_gaps
        ),
        "processed_tracks": (
            processed_tracks
        ),
        "merged_tracks": (
            merged_tracks
        ),
        "expected_fragment_assignments": (
            expected_assignments
        ),
        "assignment_consistent": (
            assignment_consistent
        ),
        "unique_fragment_ids": (
            len(all_fragment_ids)
        ),
        "identities": (
            identity_reports[:limit]
            if limit is not None
            else identity_reports
        ),
    }

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GLOBAL IDENTITY FRAGMENT ANALYSIS")
    print("=" * 70)

    print(
        f"Global identities: "
        f"{global_identity_count}"
    )

    print(
        f"Identities with fragments: "
        f"{identities_with_fragments}"
    )

    print(
        f"Total ByteTrack fragments assigned: "
        f"{total_fragments_assigned}"
    )

    print(
        f"Average fragments / identity: "
        f"{average_fragments:.2f}"
    )

    print(
        f"Median fragments / identity: "
        f"{median_fragments}"
    )

    print(
        f"Maximum fragments for one identity: "
        f"{result['maximum_fragments']}"
    )

    print(
        f"Multi-fragment identities: "
        f"{multi_fragment_identities}"
    )

    print(
        f"Heavily fragmented identities (>=5): "
        f"{heavily_fragmented_identities}"
    )

    print(
        f"Identities with long gaps "
        f"(>={long_gap_threshold} frames): "
        f"{identities_with_long_gaps}"
    )

    print()
    print("Fragment assignment consistency:")

    print(
        f"  Processed tracks: "
        f"{processed_tracks}"
    )

    print(
        f"  Merged tracks: "
        f"{merged_tracks}"
    )

    print(
        f"  Assigned fragments: "
        f"{total_fragments_assigned}"
    )

    print(
        f"  Expected assignments: "
        f"{expected_assignments}"
    )

    if assignment_consistent:
        print(
            "  STATUS: PASS"
        )
    else:
        print(
            "  STATUS: FAIL"
        )

    # --------------------------------------------------------
    # TOP FRAGMENTED IDENTITIES
    # --------------------------------------------------------

    print()
    print("Most fragmented global identities:")

    for report in identity_reports[:limit]:

        print(
            f"{report['identity_id']} | "
            f"{report['class_name']} | "
            f"{report['fragment_count']} fragments | "
            f"{report['total_frames']} frames | "
            f"max gap {report['max_gap']} frames"
        )

    # --------------------------------------------------------
    # LARGEST GAPS
    # --------------------------------------------------------

    largest_gaps = []

    for report in identity_reports:

        if report["max_gap"] <= 0:
            continue

        largest_gaps.append(
            (
                report["max_gap"],
                report["identity_id"],
                report["class_name"],
                report["fragment_count"],
            )
        )

    largest_gaps.sort(
        reverse=True
    )

    print()
    print("Largest fragment gaps:")

    if not largest_gaps:

        print(
            "  None"
        )

    else:

        for (
            gap,
            identity_id,
            class_name,
            fragment_count,
        ) in largest_gaps[:limit]:

            print(
                f"{identity_id} | "
                f"{class_name} | "
                f"{gap} frames | "
                f"{fragment_count} fragments"
            )

    print("=" * 70)

    return result