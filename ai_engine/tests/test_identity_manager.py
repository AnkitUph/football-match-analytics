from ai_engine.identity.identity_manager import GlobalIdentityManager
from ai_engine.identity.identity_matcher import IdentityMatcher
from ai_engine.schemas.track import (
    BoundingBox,
    Track,
    TrackObservation,
)


def make_track(
    local_id,
    class_name,
    start_frame,
    end_frame,
    start_x,
    start_y,
    velocity_x=1.0,
    velocity_y=0.0,
):
    observations = []

    for frame in range(start_frame, end_frame + 1):
        elapsed = frame - start_frame

        x = start_x + velocity_x * elapsed
        y = start_y + velocity_y * elapsed

        bbox = BoundingBox(
            x1=x - 10,
            y1=y - 20,
            x2=x + 10,
            y2=y,
        )

        observations.append(
            TrackObservation(
                frame_index=frame,
                bbox=bbox,
                confidence=0.9,
                class_name=class_name,
            )
        )

    return Track(
        local_id=local_id,
        class_name=class_name,
        observations=observations,
        active=False,
        first_frame=start_frame,
        last_frame=end_frame,
    )


def test_compatible_player_tracks_merge():
    track_1 = make_track(
        local_id=1,
        class_name="player",
        start_frame=0,
        end_frame=100,
        start_x=500,
        start_y=500,
        velocity_x=1.0,
    )

    track_2 = make_track(
        local_id=2,
        class_name="player",
        start_frame=105,
        end_frame=200,
        start_x=605,
        start_y=500,
        velocity_x=1.0,
    )

    manager = GlobalIdentityManager(
        matcher=IdentityMatcher(
            max_temporal_gap=10,
            max_spatial_distance=100,
            max_motion_difference=20,
        ),
        minimum_match_score=0.50,
    )

    identities = manager.build_identities(
        [track_1, track_2]
    )

    assert len(identities) == 1

    identity = list(
        identities.values()
    )[0]

    assert identity.source_track_ids == [1, 2]
    assert identity.first_frame == 0
    assert identity.last_frame == 200


def test_far_apart_tracks_do_not_merge():
    track_1 = make_track(
        local_id=1,
        class_name="player",
        start_frame=0,
        end_frame=100,
        start_x=500,
        start_y=500,
        velocity_x=1.0,
    )

    track_2 = make_track(
        local_id=2,
        class_name="player",
        start_frame=105,
        end_frame=200,
        start_x=1200,
        start_y=500,
        velocity_x=1.0,
    )

    manager = GlobalIdentityManager(
        matcher=IdentityMatcher(
            max_temporal_gap=10,
            max_spatial_distance=100,
            max_motion_difference=20,
        ),
        minimum_match_score=0.50,
    )

    identities = manager.build_identities(
        [track_1, track_2]
    )

    assert len(identities) == 2


def test_different_classes_do_not_merge():
    player_track = make_track(
        local_id=1,
        class_name="player",
        start_frame=0,
        end_frame=100,
        start_x=500,
        start_y=500,
    )

    referee_track = make_track(
        local_id=2,
        class_name="referee",
        start_frame=105,
        end_frame=200,
        start_x=500,
        start_y=500,
    )

    manager = GlobalIdentityManager(
        minimum_match_score=0.50,
    )

    identities = manager.build_identities(
        [player_track, referee_track]
    )

    assert len(identities) == 2
