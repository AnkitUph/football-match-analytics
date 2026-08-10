
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Optional, Set

from .identity_record import IdentityRecord


class IdentityRegistry:
    """
    Registry/adaptor for fused global identities.

    Responsibilities
    ----------------
    - Read the ByteTrack -> Global Identity mapping produced by
      the fused identity pipeline.
    - Read ByteTrack trajectory/history data.
    - Convert tracks into IdentityRecord objects.
    - Provide lookup APIs for later analytics stages.

    IMPORTANT:
    IdentityRegistry does NOT perform identity matching.

    Matching remains entirely inside the fused identity pipeline.
    """

    def __init__(self, manager: Any):
        self.manager = manager

        # GID -> IdentityRecord
        self.identities: Dict[str, IdentityRecord] = {}

        # ByteTrack ID -> GID
        self.track_to_global: Dict[int, str] = {}

        # ByteTrack ID -> track history object
        self.track_history: Any = None

        self._build()

    # ==============================================================
    # PUBLIC API
    # ==============================================================

    def get_identity(
        self,
        global_id: str,
    ) -> Optional[IdentityRecord]:
        """Return one identity by global ID."""
        return self.identities.get(global_id)

    def get_all_identities(self) -> Dict[str, IdentityRecord]:
        """Return a copy of all registered identities."""
        return dict(self.identities)

    def get_track_identity(
        self,
        track_id: int,
    ) -> Optional[IdentityRecord]:
        """Return the global identity associated with a ByteTrack ID."""

        global_id = self.get_global_id(track_id)

        if global_id is None:
            return None

        return self.identities.get(global_id)

    def get_global_id(
        self,
        track_id: int,
    ) -> Optional[str]:
        """Return the GID associated with a ByteTrack ID."""
        return self.track_to_global.get(int(track_id))

    def identity_count(self) -> int:
        """Return the number of registered global identities."""
        return len(self.identities)

    def track_count(self) -> int:
        """Return the number of registered ByteTrack IDs."""
        return len(self.track_to_global)

    # ==============================================================
    # SUMMARY
    # ==============================================================

    def print_summary(self) -> None:
        """Print a concise registry summary."""

        print()
        print("=" * 70)
        print("IDENTITY REGISTRY")
        print("=" * 70)

        print(
            f"Global identities:       "
            f"{len(self.identities)}"
        )

        print(
            f"Registered track IDs:    "
            f"{len(self.track_to_global)}"
        )

        fragmented = sum(
            1
            for identity in self.identities.values()
            if getattr(identity, "is_fragmented", False)
        )

        print(
            f"Fragmented identities:   "
            f"{fragmented}"
        )

        if self.identities:
            average = (
                sum(
                    getattr(identity, "track_count", 0)
                    for identity in self.identities.values()
                )
                / len(self.identities)
            )
        else:
            average = 0.0

        print(
            f"Average tracks / GID:    "
            f"{average:.2f}"
        )

        print("=" * 70)

    def print_identities(self) -> None:
        """Print all registered global identities."""

        print()
        print("=" * 90)
        print("REGISTERED GLOBAL IDENTITIES")
        print("=" * 90)

        if not self.identities:
            print("No identities registered.")
            return

        for global_id, identity in self.identities.items():

            tracks = getattr(
                identity,
                "track_ids",
                [],
            )

            class_name = getattr(
                identity,
                "class_name",
                None,
            )

            total_frames = getattr(
                identity,
                "total_frames",
                0,
            )

            first_frame = getattr(
                identity,
                "first_frame",
                None,
            )

            last_frame = getattr(
                identity,
                "last_frame",
                None,
            )

            print(
                f"{global_id}: "
                f"tracks={tracks} | "
                f"class={class_name} | "
                f"frames={total_frames} | "
                f"range={first_frame}->{last_frame}"
            )

    def print_track_mapping(self) -> None:
        """Print ByteTrack -> GID mapping."""

        print()
        print("=" * 70)
        print("TRACK → GLOBAL IDENTITY")
        print("=" * 70)

        for track_id in sorted(self.track_to_global):
            print(
                f"Track {track_id:3d} "
                f"→ {self.track_to_global[track_id]}"
            )

    def to_dict(self) -> dict:
        """Serialize the complete registry."""

        return {
            "identities": {
                global_id: identity.to_dict()
                for global_id, identity in self.identities.items()
            },
            "track_to_global": {
                str(track_id): global_id
                for track_id, global_id
                in self.track_to_global.items()
            },
        }

    # ==============================================================
    # BUILD
    # ==============================================================

    def _build(self) -> None:
        """
        Build the registry from the fused identity manager.
        """

        # ----------------------------------------------------------
        # 1. Find ByteTrack -> GID mapping
        # ----------------------------------------------------------

        mapping = self._find_track_to_global_mapping()

        if mapping is None:
            raise AttributeError(
                "\n"
                "Could not find ByteTrack ID -> global identity "
                "mapping on the fused identity manager.\n\n"
                "The fused pipeline produced the mapping, but the "
                "registry could not locate it.\n"
                "Run the manager diagnostic before changing the "
                "fusion algorithm."
            )

        self.track_to_global = {
            int(track_id): str(global_id)
            for track_id, global_id in mapping.items()
        }

        # ----------------------------------------------------------
        # 2. Find ByteTrack trajectory history
        # ----------------------------------------------------------

        self.track_history = self._find_track_store()

        # ----------------------------------------------------------
        # 3. Build IdentityRecord objects
        # ----------------------------------------------------------

        for track_id, global_id in sorted(
            self.track_to_global.items()
        ):

            if global_id not in self.identities:
                self.identities[global_id] = IdentityRecord(
                    global_id=global_id
                )

            identity = self.identities[global_id]

            track_data = self._get_track_data(
                self.track_history,
                track_id,
            )

            self._update_identity_from_track(
                identity=identity,
                track_id=track_id,
                track_data=track_data,
            )

        # ----------------------------------------------------------
        # 4. Final normalization
        # ----------------------------------------------------------

        self._finalize_identities()

    # ==============================================================
    # MAPPING DISCOVERY
    # ==============================================================

    def _find_track_to_global_mapping(
        self,
    ) -> Optional[dict]:
        """
        Find the ByteTrack -> GID mapping.

        Supports:

            manager.track_to_global

            manager.track_to_global_id

            manager.track_global_mapping

            manager.track_to_gid

            manager.global_identity_mapping

            manager.identity_mapping

        It also searches nested manager objects.

        Additionally supports:

            GID -> [track IDs]

        mappings and automatically inverts them.
        """

        # ----------------------------------------------------------
        # Direct attribute names
        # ----------------------------------------------------------

        candidates = (
            "track_to_global",
            "track_to_global_id",
            "track_global_mapping",
            "track_to_gid",
            "global_identity_mapping",
            "identity_mapping",
            "track_identity_mapping",
            "global_id_mapping",
        )

        for name in candidates:

            value = getattr(
                self.manager,
                name,
                None,
            )

            normalized = self._normalize_track_to_global(
                value
            )

            if normalized is not None:
                return normalized

        # ----------------------------------------------------------
        # Search nested objects
        # ----------------------------------------------------------

        visited: Set[int] = set()

        result = self._search_object_for_mapping(
            self.manager,
            visited=visited,
            depth=0,
            max_depth=5,
        )

        return result

    def _search_object_for_mapping(
        self,
        obj: Any,
        visited: Set[int],
        depth: int,
        max_depth: int,
    ) -> Optional[dict]:
        """
        Recursively search an object for a mapping.

        This makes the registry independent from the exact internal
        structure of the fused identity manager.
        """

        if obj is None:
            return None

        if depth > max_depth:
            return None

        object_id = id(obj)

        if object_id in visited:
            return None

        visited.add(object_id)

        # ----------------------------------------------------------
        # Dictionary
        # ----------------------------------------------------------

        if isinstance(obj, dict):

            normalized = self._normalize_track_to_global(obj)

            if normalized is not None:
                return normalized

            inverted = self._normalize_global_to_tracks(obj)

            if inverted is not None:
                return inverted

            # Search nested values.
            for value in obj.values():

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                    ),
                ):
                    continue

                result = self._search_object_for_mapping(
                    value,
                    visited,
                    depth + 1,
                    max_depth,
                )

                if result is not None:
                    return result

            return None

        # ----------------------------------------------------------
        # Lists / tuples / sets
        # ----------------------------------------------------------

        if isinstance(
            obj,
            (list, tuple, set),
        ):

            for value in obj:

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                    ),
                ):
                    continue

                result = self._search_object_for_mapping(
                    value,
                    visited,
                    depth + 1,
                    max_depth,
                )

                if result is not None:
                    return result

            return None

        # ----------------------------------------------------------
        # Object attributes
        # ----------------------------------------------------------

        try:
            attributes = vars(obj)
        except TypeError:
            return None

        for name, value in attributes.items():

            # Mapping-like attribute.
            normalized = self._normalize_track_to_global(
                value
            )

            if normalized is not None:
                return normalized

            inverted = self._normalize_global_to_tracks(
                value
            )

            if inverted is not None:
                return inverted

            # Search nested object.
            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                    type(None),
                ),
            ):
                continue

            result = self._search_object_for_mapping(
                value,
                visited,
                depth + 1,
                max_depth,
            )

            if result is not None:
                return result

        return None

    def _normalize_track_to_global(
        self,
        value: Any,
    ) -> Optional[dict]:
        """
        Validate a possible:

            track_id -> GID

        dictionary.
        """

        if not isinstance(value, dict):
            return None

        if not value:
            return None

        result = {}

        valid = 0

        for track_id, global_id in value.items():

            try:
                track_id_int = int(track_id)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if not isinstance(
                global_id,
                str,
            ):
                continue

            if not global_id.startswith("GID_"):
                continue

            result[track_id_int] = global_id
            valid += 1

        # Require a meaningful number of valid entries.
        if valid == 0:
            return None

        return result

    def _normalize_global_to_tracks(
        self,
        value: Any,
    ) -> Optional[dict]:
        """
        Validate and invert:

            GID -> [track IDs]

        into:

            track ID -> GID
        """

        if not isinstance(value, dict):
            return None

        if not value:
            return None

        result = {}

        valid = 0

        for global_id, tracks in value.items():

            if not isinstance(
                global_id,
                str,
            ):
                continue

            if not global_id.startswith("GID_"):
                continue

            if not isinstance(
                tracks,
                (list, tuple, set),
            ):
                continue

            for track_id in tracks:

                try:
                    track_id_int = int(track_id)
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                result[track_id_int] = global_id
                valid += 1

        if valid == 0:
            return None

        return result

    # ==============================================================
    # TRACK HISTORY DISCOVERY
    # ==============================================================

    def _find_track_store(self) -> Any:
        """
        Locate ByteTrack trajectory history.

        Supports the actual structure currently used by the
        ByteTrackTracker:

            tracker.track_history

        as well as nested tracker objects.
        """

        candidates = (
            "track_history",
            "tracks",
            "trajectory_history",
            "trajectories",
            "track_histories",
        )

        # Direct manager attributes.
        for name in candidates:

            value = getattr(
                self.manager,
                name,
                None,
            )

            if self._looks_like_track_store(value):
                return value

        # Common nested tracker objects.
        nested_candidates = (
            "tracker",
            "tracking",
            "byte_tracker",
            "track_manager",
            "tracking_manager",
            "tracker_engine",
        )

        visited: Set[int] = set()

        for name in nested_candidates:

            value = getattr(
                self.manager,
                name,
                None,
            )

            if value is None:
                continue

            result = self._search_object_for_track_store(
                value,
                visited=visited,
                depth=0,
                max_depth=5,
            )

            if result is not None:
                return result

        # General recursive search.
        return self._search_object_for_track_store(
            self.manager,
            visited=set(),
            depth=0,
            max_depth=5,
        )

    def _search_object_for_track_store(
        self,
        obj: Any,
        visited: Set[int],
        depth: int,
        max_depth: int,
    ) -> Any:

        if obj is None:
            return None

        if depth > max_depth:
            return None

        object_id = id(obj)

        if object_id in visited:
            return None

        visited.add(object_id)

        if self._looks_like_track_store(obj):
            return obj

        # Dictionary.
        if isinstance(obj, dict):

            for key, value in obj.items():

                # Avoid recursively walking every TrackHistory
                # object. The dictionary itself would already have
                # been detected above.
                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                    ),
                ):
                    continue

                result = self._search_object_for_track_store(
                    value,
                    visited,
                    depth + 1,
                    max_depth,
                )

                if result is not None:
                    return result

            return None

        # Object.
        try:
            attributes = vars(obj)
        except TypeError:
            return None

        for name, value in attributes.items():

            if name in (
                "track_history",
                "tracks",
                "trajectory_history",
                "trajectories",
                "track_histories",
            ):

                if self._looks_like_track_store(value):
                    return value

            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                    type(None),
                ),
            ):
                continue

            result = self._search_object_for_track_store(
                value,
                visited,
                depth + 1,
                max_depth,
            )

            if result is not None:
                return result

        return None

    def _looks_like_track_store(
        self,
        value: Any,
    ) -> bool:
        """
        Check whether a value resembles:

            Dict[int, TrackHistory]
        """

        if not isinstance(value, dict):
            return False

        if not value:
            return False

        sample_keys = list(value.keys())[:10]

        numeric_keys = 0

        for key in sample_keys:

            try:
                int(key)
                numeric_keys += 1
            except (
                TypeError,
                ValueError,
            ):
                pass

        if numeric_keys == 0:
            return False

        return True

    # ==============================================================
    # TRACK DATA
    # ==============================================================

    def _get_track_data(
        self,
        track_store: Any,
        track_id: int,
    ) -> Any:
        """Retrieve one ByteTrack trajectory."""

        if track_store is None:
            return None

        if not isinstance(
            track_store,
            dict,
        ):
            return None

        # Normal integer key.
        value = track_store.get(track_id)

        if value is not None:
            return value

        # String key fallback.
        return track_store.get(str(track_id))

    # ==============================================================
    # IDENTITY UPDATE
    # ==============================================================

    def _update_identity_from_track(
        self,
        identity: IdentityRecord,
        track_id: int,
        track_data: Any,
    ) -> None:
        """
        Add one ByteTrack trajectory to an IdentityRecord.
        """

        observations = self._extract_observations(
            track_data
        )

        # Even if trajectory information is missing, the track
        # still belongs to the identity.
        if not observations:

            identity.add_track(
                track_id=track_id
            )

            return

        # Sort defensively by frame.
        observations = sorted(
            observations,
            key=lambda observation: (
                self._get_frame_index(observation)
                if self._get_frame_index(observation)
                is not None
                else -1
            ),
        )

        frame_indices = [
            self._get_frame_index(observation)
            for observation in observations
        ]

        frame_indices = [
            frame
            for frame in frame_indices
            if frame is not None
        ]

        first_frame = (
            min(frame_indices)
            if frame_indices
            else None
        )

        last_frame = (
            max(frame_indices)
            if frame_indices
            else None
        )

        frame_count = len(observations)

        identity.add_track(
            track_id=track_id,
            first_frame=first_frame,
            last_frame=last_frame,
            frame_count=frame_count,
        )

        self._update_class(
            identity,
            observations,
        )

        self._update_team(
            identity,
            observations,
        )

        self._update_jersey(
            identity,
            observations,
        )

        self._update_confidence(
            identity,
            observations,
        )

    # ==============================================================
    # OBSERVATIONS
    # ==============================================================

    def _extract_observations(
        self,
        track_data: Any,
    ) -> list:
        """
        Convert TrackHistory / trajectory containers into
        a list of TrackObservation objects.
        """

        if track_data is None:
            return []

        # Direct list/tuple.
        if isinstance(
            track_data,
            (list, tuple),
        ):
            return list(track_data)

        # Actual TrackHistory structure.
        observations = getattr(
            track_data,
            "observations",
            None,
        )

        if observations is not None:

            try:
                return list(observations)
            except TypeError:
                pass

        # Generic history fallback.
        history = getattr(
            track_data,
            "history",
            None,
        )

        if history is not None:

            try:
                return list(history)
            except TypeError:
                pass

        return []

    def _get_frame_index(
        self,
        observation: Any,
    ) -> Optional[int]:

        frame = getattr(
            observation,
            "frame_index",
            None,
        )

        if frame is None:
            return None

        try:
            return int(frame)
        except (
            TypeError,
            ValueError,
        ):
            return None

    # ==============================================================
    # IDENTITY ATTRIBUTES
    # ==============================================================

    def _update_class(
        self,
        identity: IdentityRecord,
        observations: Iterable[Any],
    ) -> None:

        classes = []

        for observation in observations:

            class_name = getattr(
                observation,
                "class_name",
                None,
            )

            if class_name:
                classes.append(
                    str(class_name)
                )

        if not classes:
            return

        identity.class_name = Counter(
            classes
        ).most_common(1)[0][0]

    def _update_team(
        self,
        identity: IdentityRecord,
        observations: Iterable[Any],
    ) -> None:

        teams = []

        for observation in observations:

            team = getattr(
                observation,
                "team",
                None,
            )

            if team is not None:
                teams.append(team)

        if not teams:
            return

        identity.set_team(
            Counter(
                teams
            ).most_common(1)[0][0]
        )

    def _update_jersey(
        self,
        identity: IdentityRecord,
        observations: Iterable[Any],
    ) -> None:

        numbers = []

        for observation in observations:

            number = getattr(
                observation,
                "jersey_number",
                None,
            )

            if number is not None:
                numbers.append(number)

        if not numbers:
            return

        identity.set_jersey_number(
            Counter(
                numbers
            ).most_common(1)[0][0]
        )

    def _update_confidence(
        self,
        identity: IdentityRecord,
        observations: Iterable[Any],
    ) -> None:

        confidences = []

        for observation in observations:

            confidence = getattr(
                observation,
                "confidence",
                None,
            )

            if confidence is None:
                continue

            try:
                confidences.append(
                    float(confidence)
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

        if not confidences:
            return

        identity.confidence = (
            sum(confidences)
            / len(confidences)
        )

    # ==============================================================
    # FINALIZATION
    # ==============================================================

    def _finalize_identities(self) -> None:
        """
        Final consistency pass.

        Recalculate fragmented state if IdentityRecord exposes
        the corresponding attributes.
        """

        for identity in self.identities.values():

            track_ids = getattr(
                identity,
                "track_ids",
                None,
            )

            if track_ids is None:
                continue

            # Remove accidental duplicates while preserving order.
            unique_track_ids = list(
                dict.fromkeys(track_ids)
            )

            try:
                identity.track_ids = unique_track_ids
            except Exception:
                pass

            # Some IdentityRecord implementations expose
            # track_count as a property, while others store it.
            # Do not overwrite read-only properties.
            try:
                if hasattr(
                    identity,
                    "__dict__",
                ):
                    if "track_count" in identity.__dict__:
                        identity.track_count = len(
                            unique_track_ids
                        )
            except Exception:
                pass

