

"""
ByteTrack wrapper for the football AI pipeline.

Responsibilities:

- Convert our Detection objects -> supervision Detections
- Run ByteTrack
- Convert ByteTrack results -> our Track / TrackObservation schemas
- Maintain current active tracks
- Maintain complete trajectory history
- Preserve class information
- Extract appearance embeddings for tracked people
- Propagate cached appearance embeddings between OSNet extractions
- Store appearance embeddings inside TrackObservation
- Collect tracking-quality and fragmentation statistics

This module deliberately does NOT perform:

- global identity merging
- team assignment
- jersey-number recognition
- homography
- heatmap generation
- event detection
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import supervision as sv

from ai_engine.identity.appearance_extractor import (
    AppearanceExtractor,
)

from ai_engine.schemas.track import (
    BoundingBox,
    Track,
    TrackObservation,
)

logger = logging.getLogger(__name__)


class ByteTrackTracker:
    """
    Short-term tracker based on supervision's ByteTrack.

    ByteTrack IDs are local/short-term IDs.

    Appearance embeddings are extracted periodically
    and propagated between extraction frames.

    Later pipeline stages will use:

        ByteTrack
            ↓
        Appearance embeddings
            ↓
        Global Identity Manager
            ↓
        Track merging
            ↓
        Team assignment
            ↓
        Jersey number
    """

    # =========================================================
    # CLASSES THAT RECEIVE APPEARANCE EMBEDDINGS
    # =========================================================

    APPEARANCE_CLASSES = {
        "player",
        "goalkeeper",
        "referee",
    }

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        fps: float = 25.0,

        # -----------------------------------------------------
        # ByteTrack parameters
        # -----------------------------------------------------

        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 60,
        minimum_matching_threshold: float = 0.80,
        minimum_consecutive_frames: int = 1,

        # -----------------------------------------------------
        # Appearance parameters
        # -----------------------------------------------------

        appearance_model_name: str = "osnet_x1_0",
        appearance_device: str = "cpu",
        appearance_interval: int = 10,
        appearance_refresh_on_new_track: bool = True,
        minimum_track_length: int = 5,
    ):
        """
        Initialize ByteTrack and the appearance extractor.

        Args:
            fps:
                Video FPS.

            track_activation_threshold:
                Minimum detection confidence used by ByteTrack
                when activating tracks.

            lost_track_buffer:
                Number of frames ByteTrack keeps a lost track
                alive.

                At 25 FPS:

                    60  ≈ 2.4 seconds
                    100 ≈ 4.0 seconds
                    125 ≈ 5.0 seconds

                We start with 100 for football.

            minimum_matching_threshold:
                IoU matching threshold.

                Lowering this from 0.80 to 0.70 allows
                moderately shifted bounding boxes to continue
                matching the same track.

            minimum_consecutive_frames:
                Number of consecutive frames required before
                a track is considered active.

            appearance_model_name:
                TorchReID appearance model.

            appearance_device:
                Appearance extraction device.

            appearance_interval:
                Number of frames between OSNet refreshes.

            appearance_refresh_on_new_track:
                Extract appearance immediately when a new
                ByteTrack ID appears.
        """

        self.fps = float(fps)

        # -----------------------------------------------------
        # Store configuration.
        #
        # IMPORTANT:
        # These values are stored so reset() can recreate
        # ByteTrack with the exact same configuration.
        # -----------------------------------------------------

        self.track_activation_threshold = float(
            track_activation_threshold
        )

        self.lost_track_buffer = int(
            lost_track_buffer
        )

        self.minimum_matching_threshold = float(
            minimum_matching_threshold
        )

        self.minimum_consecutive_frames = int(
            minimum_consecutive_frames
        )

        # -----------------------------------------------------
        # ByteTrack
        # -----------------------------------------------------

        self.tracker = self._create_bytetrack()

        # -----------------------------------------------------
        # Appearance extractor
        # -----------------------------------------------------

        self.appearance_extractor = AppearanceExtractor(
            model_name=appearance_model_name,
            device=appearance_device,
        )

        self.appearance_model_name = appearance_model_name
        self.appearance_device = appearance_device

        # -----------------------------------------------------
        # Appearance propagation configuration
        # -----------------------------------------------------

        self.appearance_interval = max(
            1,
            int(appearance_interval),
        )

        self.appearance_refresh_on_new_track = bool(
            appearance_refresh_on_new_track
        )

        # -----------------------------------------------------
        # Tracks currently active in latest frame.
        #
        # local_id -> Track
        # -----------------------------------------------------

        self.current_tracks: Dict[int, Track] = {}

        # -----------------------------------------------------
        # Complete trajectory history.
        #
        # local_id -> Track
        # -----------------------------------------------------

        self.track_history: Dict[int, Track] = {}

        # -----------------------------------------------------
        # Last known class for each ByteTrack ID.
        # -----------------------------------------------------

        self.track_classes: Dict[int, str] = {}

        # -----------------------------------------------------
        # Last frame where each track was observed.
        # -----------------------------------------------------

        self.last_seen: Dict[int, int] = {}

        # =====================================================
        # APPEARANCE CACHE
        # =====================================================

        self.last_appearance_embeddings: Dict[
            int,
            List[float],
        ] = {}

        self.last_appearance_frame: Dict[
            int,
            int,
        ] = {}

        # =====================================================
        # APPEARANCE STATISTICS
        # =====================================================

        self.appearance_extraction_attempts = 0
        self.appearance_embeddings_extracted = 0
        self.appearance_extraction_failures = 0
        self.appearance_skipped = 0
        self.appearance_extractions_performed = 0
        self.appearance_embeddings_propagated = 0
        self.appearance_tracks_without_embedding = 0

        # =====================================================
        # TRACKING STATISTICS
        # =====================================================

        # Total number of observations received by ByteTrack.
        self.total_track_observations = 0

        # Number of frames processed.
        self.frames_processed = 0

        # Number of active tracks per frame.
        self.active_track_counts: List[int] = []

        logger.info(
            "ByteTrackTracker initialized: "
            "fps=%.2f, activation=%.2f, "
            "lost_buffer=%d, matching=%.2f, "
            "min_consecutive=%d",
            self.fps,
            self.track_activation_threshold,
            self.lost_track_buffer,
            self.minimum_matching_threshold,
            self.minimum_consecutive_frames,
        )

        logger.info(
            "Appearance extractor initialized: "
            "model=%s, device=%s, interval=%d",
            self.appearance_model_name,
            self.appearance_device,
            self.appearance_interval,
        )

    # =========================================================
    # BYTE TRACK FACTORY
    # =========================================================

    def _create_bytetrack(self):
        """
        Create ByteTrack using the currently configured
        parameters.

        Keeping creation in one place prevents reset()
        from accidentally using different parameters.
        """

        return sv.ByteTrack(
            track_activation_threshold=(
                self.track_activation_threshold
            ),
            lost_track_buffer=(
                self.lost_track_buffer
            ),
            minimum_matching_threshold=(
                self.minimum_matching_threshold
            ),
            frame_rate=self.fps,
            minimum_consecutive_frames=(
                self.minimum_consecutive_frames
            ),
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def update(
        self,
        detections: list,
        frame_index: int,
        frame: np.ndarray,
    ) -> List[Track]:
        """
        Update ByteTrack with detections from one frame.
        """

        if frame is None:
            raise ValueError(
                "frame must not be None"
            )

        # -----------------------------------------------------
        # Convert our detections -> supervision detections
        # -----------------------------------------------------

        supervision_detections = (
            self._detections_to_supervision(
                detections
            )
        )

        # -----------------------------------------------------
        # Run ByteTrack
        # -----------------------------------------------------

        tracked_detections = (
            self.tracker.update_with_detections(
                supervision_detections
            )
        )

        # -----------------------------------------------------
        # Convert ByteTrack results -> Track objects
        # -----------------------------------------------------

        tracks = self._convert_tracks(
            tracked_detections=tracked_detections,
            original_detections=detections,
            frame_index=frame_index,
            frame=frame,
        )

        # -----------------------------------------------------
        # Current-frame state
        # -----------------------------------------------------

        self._update_current_tracks(
            tracks
        )

        # -----------------------------------------------------
        # Complete history
        # -----------------------------------------------------

        self._update_history(
            tracks
        )

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        self.frames_processed += 1
        self.total_track_observations += len(tracks)
        self.active_track_counts.append(
            len(tracks)
        )

        return tracks

    # =========================================================
    # RESET
    # =========================================================

    def reset(self) -> None:
        """
        Completely reset ByteTrack and stored state.

        Uses the original configuration instead of
        hard-coded parameters.
        """

        self.tracker = self._create_bytetrack()

        self.current_tracks.clear()
        self.track_history.clear()
        self.track_classes.clear()
        self.last_seen.clear()

        self.last_appearance_embeddings.clear()
        self.last_appearance_frame.clear()

        self.appearance_extraction_attempts = 0
        self.appearance_embeddings_extracted = 0
        self.appearance_extraction_failures = 0
        self.appearance_skipped = 0
        self.appearance_extractions_performed = 0
        self.appearance_embeddings_propagated = 0
        self.appearance_tracks_without_embedding = 0

        self.total_track_observations = 0
        self.frames_processed = 0
        self.active_track_counts.clear()

        logger.info(
            "ByteTrackTracker reset with configuration: "
            "activation=%.2f, lost_buffer=%d, "
            "matching=%.2f, min_consecutive=%d",
            self.track_activation_threshold,
            self.lost_track_buffer,
            self.minimum_matching_threshold,
            self.minimum_consecutive_frames,
        )

    # =========================================================
    # DETECTION -> SUPERVISION
    # =========================================================

    def _detections_to_supervision(
        self,
        detections: list,
    ) -> sv.Detections:

        if not detections:
            return sv.Detections.empty()

        xyxy = []
        confidence = []
        class_id = []

        for detection in detections:

            bbox = detection.bbox

            xyxy.append(
                [
                    float(bbox.x1),
                    float(bbox.y1),
                    float(bbox.x2),
                    float(bbox.y2),
                ]
            )

            confidence.append(
                float(detection.confidence)
            )

            class_id.append(
                int(detection.class_id)
            )

        return sv.Detections(
            xyxy=np.asarray(
                xyxy,
                dtype=np.float32,
            ),
            confidence=np.asarray(
                confidence,
                dtype=np.float32,
            ),
            class_id=np.asarray(
                class_id,
                dtype=np.int64,
            ),
        )

    # =========================================================
    # SUPERVISION -> OUR TRACK SCHEMA
    # =========================================================

    def _convert_tracks(
        self,
        tracked_detections: sv.Detections,
        original_detections: list,
        frame_index: int,
        frame: np.ndarray,
    ) -> List[Track]:

        if (
            tracked_detections is None
            or len(tracked_detections) == 0
            or tracked_detections.tracker_id is None
        ):
            self.current_tracks = {}
            return []

        tracks: List[Track] = []

        tracker_ids = (
            tracked_detections.tracker_id
        )

        for index, tracker_id in enumerate(
            tracker_ids
        ):

            if tracker_id is None:
                continue

            local_id = int(tracker_id)

            # -------------------------------------------------
            # BOUNDING BOX
            # -------------------------------------------------

            xyxy = (
                tracked_detections.xyxy[index]
            )

            bbox = BoundingBox(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
            )

            # -------------------------------------------------
            # CONFIDENCE
            # -------------------------------------------------

            confidence = 0.0

            if tracked_detections.confidence is not None:
                confidence = float(
                    tracked_detections.confidence[index]
                )

            # -------------------------------------------------
            # CLASS ID
            # -------------------------------------------------

            class_id = None

            if tracked_detections.class_id is not None:
                class_id = int(
                    tracked_detections.class_id[index]
                )

            # -------------------------------------------------
            # CLASS NAME
            # -------------------------------------------------

            class_name = self._find_class_name(
                bbox=bbox,
                class_id=class_id,
                original_detections=original_detections,
            )

            if class_name is None:
                class_name = self.track_classes.get(
                    local_id,
                    "unknown",
                )

            self.track_classes[
                local_id
            ] = class_name

            # -------------------------------------------------
            # APPEARANCE
            # -------------------------------------------------

            appearance_embedding = (
                self._extract_appearance_embedding(
                    frame=frame,
                    bbox=bbox,
                    class_name=class_name,
                    frame_index=frame_index,
                    local_id=local_id,
                )
            )

            # -------------------------------------------------
            # OBSERVATION
            # -------------------------------------------------

            observation = TrackObservation(
                frame_index=frame_index,
                bbox=bbox,
                confidence=confidence,
                class_name=class_name,
                appearance_embedding=(
                    appearance_embedding
                ),
            )

            # -------------------------------------------------
            # GET OR CREATE TRACK
            # -------------------------------------------------

            track = self.track_history.get(
                local_id
            )

            if track is None:

                track = Track(
                    local_id=local_id,
                    class_name=class_name,
                    observations=[],
                    active=True,
                    first_frame=frame_index,
                    last_frame=frame_index,
                )

            else:

                track.active = True

                if track.first_frame is None:
                    track.first_frame = frame_index

            # -------------------------------------------------
            # ADD OBSERVATION
            # -------------------------------------------------

            if (
                not track.observations
                or track.observations[-1].frame_index
                != frame_index
            ):
                track.observations.append(
                    observation
                )

            track.last_frame = frame_index

            tracks.append(track)

            self.last_seen[
                local_id
            ] = frame_index

        return tracks

    # =========================================================
    # APPEARANCE EXTRACTION + PROPAGATION
    # =========================================================

    def _extract_appearance_embedding(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        class_name: str,
        frame_index: int,
        local_id: int,
    ) -> Optional[List[float]]:

        if class_name not in self.APPEARANCE_CLASSES:

            self.appearance_skipped += 1

            return None

        cached_embedding = (
            self.last_appearance_embeddings.get(
                local_id
            )
        )

        last_extraction_frame = (
            self.last_appearance_frame.get(
                local_id
            )
        )

        is_new_track = (
            cached_embedding is None
            or last_extraction_frame is None
        )

        should_extract = False

        if is_new_track:

            should_extract = (
                self.appearance_refresh_on_new_track
            )

        else:

            frames_since_extraction = (
                frame_index
                - last_extraction_frame
            )

            should_extract = (
                frames_since_extraction
                >= self.appearance_interval
            )

        # -----------------------------------------------------
        # Propagation
        # -----------------------------------------------------

        if not should_extract:

            if cached_embedding is not None:

                self.appearance_embeddings_propagated += 1

                return list(
                    cached_embedding
                )

            self.appearance_tracks_without_embedding += 1

            return None

        # -----------------------------------------------------
        # OSNet extraction
        # -----------------------------------------------------

        self.appearance_extraction_attempts += 1
        self.appearance_extractions_performed += 1

        try:

            embedding = (
                self.appearance_extractor.extract(
                    frame,
                    bbox,
                )
            )

        except Exception:

            self.appearance_extraction_failures += 1

            logger.exception(
                "Appearance extraction failed "
                "for local_id=%d at frame=%d",
                local_id,
                frame_index,
            )

            if cached_embedding is not None:

                self.appearance_embeddings_propagated += 1

                return list(
                    cached_embedding
                )

            self.appearance_tracks_without_embedding += 1

            return None

        if embedding is None:

            self.appearance_extraction_failures += 1

            if cached_embedding is not None:

                self.appearance_embeddings_propagated += 1

                return list(
                    cached_embedding
                )

            self.appearance_tracks_without_embedding += 1

            return None

        # -----------------------------------------------------
        # Validate embedding
        # -----------------------------------------------------

        try:

            embedding = [
                float(value)
                for value in embedding
            ]

        except Exception:

            self.appearance_extraction_failures += 1

            logger.exception(
                "Invalid appearance embedding "
                "for local_id=%d at frame=%d",
                local_id,
                frame_index,
            )

            if cached_embedding is not None:
                return list(cached_embedding)

            return None

        if not embedding:

            self.appearance_extraction_failures += 1

            if cached_embedding is not None:
                return list(cached_embedding)

            return None

        embedding_array = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if not np.all(
            np.isfinite(
                embedding_array
            )
        ):

            self.appearance_extraction_failures += 1

            if cached_embedding is not None:
                return list(cached_embedding)

            return None

        # -----------------------------------------------------
        # Cache
        # -----------------------------------------------------

        self.last_appearance_embeddings[
            local_id
        ] = list(embedding)

        self.last_appearance_frame[
            local_id
        ] = frame_index

        self.appearance_embeddings_extracted += 1

        return list(embedding)

    # =========================================================
    # CURRENT TRACK STATE
    # =========================================================

    def _update_current_tracks(
        self,
        tracks: List[Track],
    ) -> None:

        self.current_tracks = {
            track.local_id: track
            for track in tracks
        }

    # =========================================================
    # COMPLETE HISTORY
    # =========================================================

    def _update_history(
        self,
        tracks: List[Track],
    ) -> None:

        active_ids = set()

        for track in tracks:

            local_id = track.local_id

            active_ids.add(
                local_id
            )

            self.track_history[
                local_id
            ] = track

        for local_id, track in (
            self.track_history.items()
        ):

            if local_id not in active_ids:

                track.active = False

    # =========================================================
    # CLASS MATCHING
    # =========================================================

    def _find_class_name(
        self,
        bbox: BoundingBox,
        class_id: Optional[int],
        original_detections: list,
    ) -> Optional[str]:

        if not original_detections:
            return None

        # -----------------------------------------------------
        # Same class + highest IoU
        # -----------------------------------------------------

        best_detection = None
        best_iou = 0.0

        for detection in original_detections:

            if (
                class_id is not None
                and int(detection.class_id)
                != class_id
            ):
                continue

            iou = self._calculate_iou(
                bbox,
                detection.bbox,
            )

            if iou > best_iou:

                best_iou = iou
                best_detection = detection

        if best_detection is not None:
            return best_detection.class_name

        # -----------------------------------------------------
        # Fallback: highest IoU
        # -----------------------------------------------------

        best_detection = None
        best_iou = 0.0

        for detection in original_detections:

            iou = self._calculate_iou(
                bbox,
                detection.bbox,
            )

            if iou > best_iou:

                best_iou = iou
                best_detection = detection

        if best_detection is not None:
            return best_detection.class_name

        return None

    # =========================================================
    # IOU
    # =========================================================

    @staticmethod
    def _calculate_iou(
        a: BoundingBox,
        b: BoundingBox,
    ) -> float:

        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)

        intersection_width = max(
            0.0,
            x2 - x1,
        )

        intersection_height = max(
            0.0,
            y2 - y1,
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        area_a = (
            max(0.0, a.x2 - a.x1)
            * max(0.0, a.y2 - a.y1)
        )

        area_b = (
            max(0.0, b.x2 - b.x1)
            * max(0.0, b.y2 - b.y1)
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    # =========================================================
    # APPEARANCE STATISTICS
    # =========================================================

    def get_appearance_statistics(
        self,
    ) -> dict:

        attempts = (
            self.appearance_extraction_attempts
        )

        successful = (
            self.appearance_embeddings_extracted
        )

        success_rate = (
            successful / attempts
            if attempts > 0
            else 0.0
        )

        return {
            "model": self.appearance_model_name,
            "device": self.appearance_device,
            "appearance_interval": (
                self.appearance_interval
            ),
            "extraction_attempts": attempts,
            "osnet_extractions_performed": (
                self.appearance_extractions_performed
            ),
            "embeddings_extracted": successful,
            "embeddings_propagated": (
                self.appearance_embeddings_propagated
            ),
            "extraction_failures": (
                self.appearance_extraction_failures
            ),
            "tracks_without_embedding": (
                self.appearance_tracks_without_embedding
            ),
            "skipped_non_person": (
                self.appearance_skipped
            ),
            "success_rate": success_rate,
        }

    # =========================================================
    # TRACKING STATISTICS
    # =========================================================

    def get_tracking_statistics(
        self,
    ) -> dict:
        """
        Return statistics useful for measuring
        ByteTrack fragmentation.

        These are NOT identity metrics.

        They tell us how fragmented the short-term
        tracker is before Global Identity Manager.
        """

        tracks = list(
            self.track_history.values()
        )

        lifetimes = [
            track.frame_count
            for track in tracks
            if track.frame_count > 0
        ]

        fps = max(
            self.fps,
            1.0,
        )

        one_second = int(
            round(fps)
        )

        if lifetimes:

            short_tracks = sum(
                lifetime < one_second
                for lifetime in lifetimes
            )

            medium_tracks = sum(
                one_second
                <= lifetime
                < 5 * one_second
                for lifetime in lifetimes
            )

            long_tracks = sum(
                lifetime >= 5 * one_second
                for lifetime in lifetimes
            )

            average_lifetime = (
                sum(lifetimes)
                / len(lifetimes)
            )

            sorted_lifetimes = sorted(
                lifetimes
            )

            middle = len(
                sorted_lifetimes
            ) // 2

            if len(sorted_lifetimes) % 2 == 0:

                median_lifetime = (
                    sorted_lifetimes[middle - 1]
                    + sorted_lifetimes[middle]
                ) / 2.0

                median_lifetime /= 2.0

            else:

                median_lifetime = float(
                    sorted_lifetimes[middle]
                )

        else:

            short_tracks = 0
            medium_tracks = 0
            long_tracks = 0
            average_lifetime = 0.0
            median_lifetime = 0.0

        return {
            "frames_processed": self.frames_processed,
            "unique_tracks": len(tracks),
            "total_track_observations": (
                self.total_track_observations
            ),
            "average_track_lifetime_frames": (
                average_lifetime
            ),
            "median_track_lifetime_frames": (
                median_lifetime
            ),
            "average_track_lifetime_seconds": (
                average_lifetime / fps
            ),
            "short_tracks_under_1_second": (
                short_tracks
            ),
            "medium_tracks_1_to_5_seconds": (
                medium_tracks
            ),
            "long_tracks_over_5_seconds": (
                long_tracks
            ),
            "configuration": {
                "track_activation_threshold": (
                    self.track_activation_threshold
                ),
                "lost_track_buffer": (
                    self.lost_track_buffer
                ),
                "minimum_matching_threshold": (
                    self.minimum_matching_threshold
                ),
                "minimum_consecutive_frames": (
                    self.minimum_consecutive_frames
                ),
            },
        }

    # =========================================================
    # QUERY HELPERS
    # =========================================================

    def get_current_tracks(
        self,
    ) -> List[Track]:

        return list(
            self.current_tracks.values()
        )

    def get_track_history(
        self,
    ) -> Dict[int, Track]:

        return self.track_history

    def get_track(
        self,
        local_id: int,
    ) -> Optional[Track]:

        return self.track_history.get(
            local_id
        )

    def get_active_track_ids(
        self,
    ) -> List[int]:

        return list(
            self.current_tracks.keys()
        )

    def get_all_track_ids(
        self,
    ) -> List[int]:

        return list(
            self.track_history.keys()
        )

    def get_track_count(
        self,
    ) -> int:

        return len(
            self.track_history
        )

    def get_active_track_count(
        self,
    ) -> int:

        return len(
            self.current_tracks
        )
