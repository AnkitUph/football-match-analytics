
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

Appearance stage:

    Detection
        ↓
    ByteTrack
        ↓
    BoundingBox
        ↓
    AppearanceExtractor / OSNet
        ↓
    Cached appearance embedding
        ↓
    Propagation
        ↓
    TrackObservation

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

    Appearance embeddings are extracted periodically and
    propagated between extraction frames.

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
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 60,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
        appearance_model_name: str = "osnet_x1_0",
        appearance_device: str = "cpu",
        appearance_interval: int = 10,
        appearance_refresh_on_new_track: bool = True,
    ):
        """
        Initialize ByteTrack and the appearance extractor.

        Args:
            fps:
                Video FPS.

            track_activation_threshold:
                ByteTrack detection activation threshold.

            lost_track_buffer:
                Number of frames a lost track is kept alive.

            minimum_matching_threshold:
                ByteTrack matching threshold.

            minimum_consecutive_frames:
                Minimum consecutive frames required to activate
                a track.

            appearance_model_name:
                TorchReID model used for appearance extraction.

            appearance_device:
                Device for appearance extraction.

                Examples:
                    "cpu"
                    "cuda"

            appearance_interval:
                Number of frames between OSNet appearance
                refreshes for an existing track.

                Example:

                    appearance_interval=10

                means OSNet runs once every 10 frames per
                tracked object and the last valid embedding
                is propagated between refreshes.

            appearance_refresh_on_new_track:
                Whether to immediately extract an appearance
                embedding when a new ByteTrack ID appears.
        """

        self.fps = float(fps)

        # -----------------------------------------------------
        # ByteTrack
        # -----------------------------------------------------

        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=self.fps,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

        # -----------------------------------------------------
        # Appearance extractor
        # -----------------------------------------------------

        self.appearance_extractor = AppearanceExtractor(
            model_name=appearance_model_name,
            device=appearance_device,
        )

        self.appearance_model_name = (
            appearance_model_name
        )

        self.appearance_device = (
            appearance_device
        )

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
        #
        # Never cleared during the video unless reset()
        # is called.
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

        # -----------------------------------------------------
        # Last valid appearance embedding for each local ID.
        #
        # local_id -> embedding
        # -----------------------------------------------------

        self.last_appearance_embeddings: Dict[
            int,
            List[float],
        ] = {}

        # -----------------------------------------------------
        # Last frame on which OSNet successfully generated
        # an appearance embedding.
        #
        # local_id -> frame_index
        # -----------------------------------------------------

        self.last_appearance_frame: Dict[
            int,
            int,
        ] = {}

        # =====================================================
        # APPEARANCE STATISTICS
        # =====================================================

        # Number of observations for which appearance
        # extraction was actually attempted.
        self.appearance_extraction_attempts = 0

        # Number of successful NEW embeddings.
        self.appearance_embeddings_extracted = 0

        # Number of failed OSNet extraction attempts.
        self.appearance_extraction_failures = 0

        # Number of observations skipped because the class
        # is not eligible for appearance extraction.
        self.appearance_skipped = 0

        # Number of actual OSNet calls.
        self.appearance_extractions_performed = 0

        # Number of times a cached embedding was propagated.
        self.appearance_embeddings_propagated = 0

        # Number of tracks that currently have no appearance
        # embedding available.
        self.appearance_tracks_without_embedding = 0

        logger.info(
            "ByteTrackTracker initialized: "
            "fps=%.2f, activation=%.2f, "
            "lost_buffer=%d, matching=%.2f, "
            "min_consecutive=%d",
            self.fps,
            track_activation_threshold,
            lost_track_buffer,
            minimum_matching_threshold,
            minimum_consecutive_frames,
        )

        logger.info(
            "Appearance extractor initialized: "
            "model=%s, device=%s, interval=%d",
            self.appearance_model_name,
            self.appearance_device,
            self.appearance_interval,
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

        Appearance embeddings are extracted periodically
        and propagated between extraction frames.

        Args:
            detections:
                Detection objects produced by YOLO.

            frame_index:
                Current video frame index.

            frame:
                Original BGR OpenCV frame.

        Returns:
            List[Track] containing tracks active in this frame.
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
        # Convert ByteTrack results -> our Track objects
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

        return tracks

    # =========================================================
    # RESET
    # =========================================================

    def reset(self) -> None:
        """
        Completely reset ByteTrack and stored trajectory,
        appearance, and propagation state.
        """

        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60,
            minimum_matching_threshold=0.8,
            frame_rate=self.fps,
            minimum_consecutive_frames=1,
        )

        self.current_tracks.clear()
        self.track_history.clear()
        self.track_classes.clear()
        self.last_seen.clear()

        # -----------------------------------------------------
        # Appearance propagation state
        # -----------------------------------------------------

        self.last_appearance_embeddings.clear()
        self.last_appearance_frame.clear()

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        self.appearance_extraction_attempts = 0
        self.appearance_embeddings_extracted = 0
        self.appearance_extraction_failures = 0
        self.appearance_skipped = 0

        self.appearance_extractions_performed = 0
        self.appearance_embeddings_propagated = 0
        self.appearance_tracks_without_embedding = 0

        logger.info(
            "ByteTrackTracker reset"
        )

    # =========================================================
    # DETECTION -> SUPERVISION
    # =========================================================

    def _detections_to_supervision(
        self,
        detections: list,
    ) -> sv.Detections:
        """
        Convert our Detection schema into supervision.Detections.
        """

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
        """
        Convert ByteTrack output into our Track schema.

        ByteTrack's tracker_id becomes:

            Track.local_id

        Appearance embeddings are extracted periodically
        after the Track bounding box and class name have
        been determined.
        """

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

            local_id = int(
                tracker_id
            )

            # =================================================
            # BOUNDING BOX
            # =================================================

            xyxy = (
                tracked_detections.xyxy[index]
            )

            bbox = BoundingBox(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
            )

            # =================================================
            # CONFIDENCE
            # =================================================

            confidence = 0.0

            if (
                tracked_detections.confidence
                is not None
            ):
                confidence = float(
                    tracked_detections.confidence[
                        index
                    ]
                )

            # =================================================
            # CLASS ID
            # =================================================

            class_id = None

            if (
                tracked_detections.class_id
                is not None
            ):
                class_id = int(
                    tracked_detections.class_id[
                        index
                    ]
                )

            # =================================================
            # CLASS NAME
            # =================================================

            class_name = (
                self._find_class_name(
                    bbox=bbox,
                    class_id=class_id,
                    original_detections=(
                        original_detections
                    ),
                )
            )

            # Fallback to previous known class.
            if class_name is None:
                class_name = (
                    self.track_classes.get(
                        local_id,
                        "unknown",
                    )
                )

            self.track_classes[
                local_id
            ] = class_name

            # =================================================
            # APPEARANCE
            # =================================================

            appearance_embedding = (
                self._extract_appearance_embedding(
                    frame=frame,
                    bbox=bbox,
                    class_name=class_name,
                    frame_index=frame_index,
                    local_id=local_id,
                )
            )

            # =================================================
            # CREATE OBSERVATION
            # =================================================

            observation = TrackObservation(
                frame_index=frame_index,
                bbox=bbox,
                confidence=confidence,
                class_name=class_name,
                appearance_embedding=(
                    appearance_embedding
                ),
            )

            # =================================================
            # GET OR CREATE TRACK
            # =================================================

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
                    track.first_frame = (
                        frame_index
                    )

            # =================================================
            # ADD OBSERVATION
            # =================================================

            if not track.observations:

                track.observations.append(
                    observation
                )

            elif (
                track.observations[-1].frame_index
                != frame_index
            ):

                track.observations.append(
                    observation
                )

            track.last_frame = frame_index

            tracks.append(
                track
            )

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
        """
        Extract or propagate an appearance embedding.

        Strategy:

            New track
                ↓
            OSNet extraction
                ↓
            Cache embedding
                ↓
            Propagate for N frames
                ↓
            Refresh OSNet
                ↓
            Cache new embedding

        This prevents expensive OSNet CPU inference from
        running for every tracked person on every frame.

        Returns:
            Appearance embedding or None.
        """

        # -----------------------------------------------------
        # Only people receive appearance embeddings.
        # -----------------------------------------------------

        if class_name not in self.APPEARANCE_CLASSES:

            self.appearance_skipped += 1

            return None

        # -----------------------------------------------------
        # Retrieve cached appearance.
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Determine whether this is a new appearance track.
        # -----------------------------------------------------

        is_new_track = (
            cached_embedding is None
            or last_extraction_frame is None
        )

        # -----------------------------------------------------
        # Determine whether OSNet should run.
        # -----------------------------------------------------

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

        # =====================================================
        # PROPAGATION PATH
        # =====================================================

        if not should_extract:

            if cached_embedding is not None:

                self.appearance_embeddings_propagated += 1

                return list(
                    cached_embedding
                )

            self.appearance_tracks_without_embedding += 1

            return None

        # =====================================================
        # OSNET EXTRACTION PATH
        # =====================================================

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

            # -------------------------------------------------
            # Preserve previous valid embedding.
            # -------------------------------------------------

            if cached_embedding is not None:

                self.appearance_embeddings_propagated += 1

                return list(
                    cached_embedding
                )

            self.appearance_tracks_without_embedding += 1

            return None

        # =====================================================
        # NONE RESULT
        # =====================================================

        if embedding is None:

            self.appearance_extraction_failures += 1

            logger.debug(
                "Appearance extractor returned None "
                "for local_id=%d at frame=%d",
                local_id,
                frame_index,
            )

            # -------------------------------------------------
            # Preserve previous valid embedding.
            # -------------------------------------------------

            if cached_embedding is not None:

                self.appearance_embeddings_propagated += 1

                return list(
                    cached_embedding
                )

            self.appearance_tracks_without_embedding += 1

            return None

        # =====================================================
        # DEFENSIVE VALIDATION
        # =====================================================

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

                self.appearance_embeddings_propagated += 1

                return list(
                    cached_embedding
                )

            return None

        if not embedding:

            self.appearance_extraction_failures += 1

            logger.debug(
                "Empty appearance embedding "
                "for local_id=%d at frame=%d",
                local_id,
                frame_index,
            )

            if cached_embedding is not None:

                self.appearance_embeddings_propagated += 1

                return list(
                    cached_embedding
                )

            return None

        # -----------------------------------------------------
        # Validate finite values.
        # -----------------------------------------------------

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

            logger.debug(
                "Non-finite appearance embedding "
                "for local_id=%d at frame=%d",
                local_id,
                frame_index,
            )

            if cached_embedding is not None:

                self.appearance_embeddings_propagated += 1

                return list(
                    cached_embedding
                )

            return None

        # =====================================================
        # CACHE NEW EMBEDDING
        # =====================================================

        self.last_appearance_embeddings[
            local_id
        ] = list(
            embedding
        )

        self.last_appearance_frame[
            local_id
        ] = frame_index

        self.appearance_embeddings_extracted += 1

        logger.debug(
            "New appearance embedding extracted "
            "for local_id=%d at frame=%d",
            local_id,
            frame_index,
        )

        return list(
            embedding
        )

    # =========================================================
    # CURRENT TRACK STATE
    # =========================================================

    def _update_current_tracks(
        self,
        tracks: List[Track],
    ) -> None:
        """
        Store only tracks visible in the current frame.
        """

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
        """
        Store/update complete track history.

        Tracks that disappear from ByteTrack are marked
        inactive but are never deleted.
        """

        active_ids = set()

        for track in tracks:

            local_id = track.local_id

            active_ids.add(
                local_id
            )

            self.track_history[
                local_id
            ] = track

        # -----------------------------------------------------
        # Mark disappeared tracks inactive.
        # -----------------------------------------------------

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
        """
        Recover class_name from the original Detection list.

        Matching strategy:

            1. same class_id
            2. highest IoU
            3. fallback to highest IoU regardless of class
        """

        if not original_detections:
            return None

        # -----------------------------------------------------
        # First pass:
        # same class_id + highest IoU
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

            candidate_bbox = (
                detection.bbox
            )

            iou = self._calculate_iou(
                bbox,
                candidate_bbox,
            )

            if iou > best_iou:

                best_iou = iou
                best_detection = detection

        if best_detection is not None:

            return best_detection.class_name

        # -----------------------------------------------------
        # Second pass:
        # highest IoU regardless of class
        # -----------------------------------------------------

        best_detection = None
        best_iou = 0.0

        for detection in original_detections:

            candidate_bbox = (
                detection.bbox
            )

            iou = self._calculate_iou(
                bbox,
                candidate_bbox,
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
        """
        Calculate intersection-over-union between two boxes.
        """

        x1 = max(
            a.x1,
            b.x1,
        )

        y1 = max(
            a.y1,
            b.y1,
        )

        x2 = min(
            a.x2,
            b.x2,
        )

        y2 = min(
            a.y2,
            b.y2,
        )

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
            max(
                0.0,
                a.x2 - a.x1,
            )
            * max(
                0.0,
                a.y2 - a.y1,
            )
        )

        area_b = (
            max(
                0.0,
                b.x2 - b.x1,
            )
            * max(
                0.0,
                b.y2 - b.y1,
            )
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:

            return 0.0

        return (
            intersection / union
        )

    # =========================================================
    # APPEARANCE STATISTICS
    # =========================================================

    def get_appearance_statistics(
        self,
    ) -> dict:
        """
        Return appearance extraction and propagation
        statistics.
        """

        attempts = (
            self.appearance_extraction_attempts
        )

        successful = (
            self.appearance_embeddings_extracted
        )

        failures = (
            self.appearance_extraction_failures
        )

        skipped = (
            self.appearance_skipped
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

            "extraction_attempts": (
                attempts
            ),

            "osnet_extractions_performed": (
                self.appearance_extractions_performed
            ),

            "embeddings_extracted": (
                successful
            ),

            "embeddings_propagated": (
                self.appearance_embeddings_propagated
            ),

            "extraction_failures": (
                failures
            ),

            "tracks_without_embedding": (
                self.appearance_tracks_without_embedding
            ),

            "skipped_non_person": (
                skipped
            ),

            "success_rate": (
                success_rate
            ),
        }

    # =========================================================
    # QUERY HELPERS
    # =========================================================

    def get_current_tracks(
        self,
    ) -> List[Track]:
        """
        Return tracks visible in the latest processed frame.
        """

        return list(
            self.current_tracks.values()
        )

    def get_track_history(
        self,
    ) -> Dict[int, Track]:
        """
        Return all ByteTrack trajectories.
        """

        return self.track_history

    def get_track(
        self,
        local_id: int,
    ) -> Optional[Track]:
        """
        Get one complete trajectory.
        """

        return self.track_history.get(
            local_id
        )

    def get_active_track_ids(
        self,
    ) -> List[int]:
        """
        Return IDs visible in the latest frame.
        """

        return list(
            self.current_tracks.keys()
        )

    def get_all_track_ids(
        self,
    ) -> List[int]:
        """
        Return every ByteTrack ID observed so far.
        """

        return list(
            self.track_history.keys()
        )

    def get_track_count(
        self,
    ) -> int:
        """
        Number of unique ByteTrack IDs created so far.
        """

        return len(
            self.track_history
        )

    def get_active_track_count(
        self,
    ) -> int:
        """
        Number of tracks visible in the latest frame.
        """

        return len(
            self.current_tracks
        )

