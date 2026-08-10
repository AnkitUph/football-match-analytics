
"""
Robust OSNet appearance feature extraction for football
player re-identification.

Responsibilities
----------------

- Extract appearance embeddings from person crops.
- Use pretrained TorchReID OSNet.
- Normalize embeddings for cosine similarity.
- Handle imperfect football detections.
- Preserve compatibility with ByteTrack.
- Provide extraction diagnostics.
- Provide embedding quality metadata.
- Support reliable appearance propagation.

Pipeline
--------

    Video frame
        ↓
    Bounding box
        ↓
    Sanitized / padded crop
        ↓
    Crop validation
        ↓
    Optional upscaling
        ↓
    Resize / normalization
        ↓
    OSNet
        ↓
    Feature tensor
        ↓
    L2 normalization
        ↓
    Appearance embedding
        ↓
    Track / Identity propagation

V5 appearance propagation foundation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class AppearanceExtractor:
    """
    Robust appearance extractor based on pretrained OSNet.

    Designed for football footage where detections can be:

    - very small
    - partially outside the frame
    - tightly cropped
    - noisy
    - slightly malformed
    - affected by camera motion
    - partially occluded

    Important:

    This class extracts embeddings.

    It does NOT decide whether two identities match.

    IdentityMatcher / GlobalIdentityManager remain responsible
    for matching decisions.
    """

    DEFAULT_IMAGE_SIZE = (256, 128)

    PERSON_CLASSES = {
        "player",
        "goalkeeper",
        "referee",
    }

    IMAGENET_MEAN = np.asarray(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    IMAGENET_STD = np.asarray(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        device: Optional[str] = None,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        min_crop_width: int = 12,
        min_crop_height: int = 24,
        bbox_padding: float = 0.10,
        min_quality_score: float = 0.01,
        upscale_small_crops: bool = True,
        warmup: bool = True,
    ) -> None:
        """
        Initialize the appearance extractor.

        Args:
            model_name:
                TorchReID model name.

            device:
                "cuda" or "cpu".
                Automatically selected when omitted.

            image_size:
                OSNet input size as (height, width).

            min_crop_width:
                Minimum crop width considered useful.

            min_crop_height:
                Minimum crop height considered useful.

            bbox_padding:
                Relative padding around the detection box.

            min_quality_score:
                Quality value below which a crop is marked
                low quality.

                IMPORTANT:
                Low quality does NOT automatically mean failure.

            upscale_small_crops:
                Upscale small player crops before OSNet.

            warmup:
                Run a dummy inference after model loading.
        """

        self.model_name = model_name

        self.device = self._resolve_device(
            device
        )

        self.image_size = (
            int(image_size[0]),
            int(image_size[1]),
        )

        self.min_crop_width = max(
            1,
            int(min_crop_width),
        )

        self.min_crop_height = max(
            1,
            int(min_crop_height),
        )

        self.bbox_padding = float(
            max(
                0.0,
                bbox_padding,
            )
        )

        self.min_quality_score = float(
            max(
                0.0,
                min_quality_score,
            )
        )

        self.upscale_small_crops = bool(
            upscale_small_crops
        )

        # ========================================================
        # DIAGNOSTICS
        # ========================================================

        self.extraction_attempts = 0

        self.embeddings_extracted = 0

        self.extraction_failures = 0

        self.invalid_crops = 0
        self.invalid_bboxes = 0

        self.small_crops = 0
        self.low_quality_crops = 0

        self.model_inference_failures = 0
        self.preprocessing_failures = 0
        self.invalid_embeddings = 0

        self.upscaled_crops = 0
        self.padded_crops = 0

        self.zero_quality_crops = 0

        self.embedding_dimension: Optional[int] = None

        self.warmup_successful = False

        # Last successful embedding is useful for diagnostics
        # and downstream propagation systems.
        self.last_embedding: Optional[list[float]] = None

        self.last_quality_score: Optional[float] = None

        # ========================================================
        # MODEL
        # ========================================================

        self.model = self._load_model()

        self.model.eval()

        if warmup:
            self._warmup()

        logger.info(
            "AppearanceExtractor initialized: "
            "model=%s device=%s image_size=%s "
            "padding=%.2f min_crop=%sx%s",
            self.model_name,
            self.device,
            self.image_size,
            self.bbox_padding,
            self.min_crop_width,
            self.min_crop_height,
        )

    # ============================================================
    # MODEL
    # ============================================================

    def _load_model(self):
        """
        Load pretrained TorchReID OSNet.
        """

        try:
            import torchreid
        except ImportError as exc:
            raise ImportError(
                "torchreid is required for appearance "
                "re-identification.\n"
                "Install with:\n"
                "pip install torchreid"
            ) from exc

        logger.info(
            "Loading ReID model: %s",
            self.model_name,
        )

        model = torchreid.models.build_model(
            name=self.model_name,
            num_classes=1,
            pretrained=True,
        )

        model.to(
            self.device
        )

        model.eval()

        return model

    # ============================================================
    # DEVICE
    # ============================================================

    @staticmethod
    def _resolve_device(
        device: Optional[str],
    ) -> str:
        """
        Resolve inference device.
        """

        if device is not None:
            requested = device.lower()

            if requested == "cuda":
                if torch.cuda.is_available():
                    return "cuda"

                logger.warning(
                    "CUDA requested but unavailable. "
                    "Falling back to CPU."
                )

                return "cpu"

            return requested

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    # ============================================================
    # WARMUP
    # ============================================================

    def _warmup(self) -> None:
        """
        Run one dummy inference.

        This verifies that preprocessing, model loading,
        device placement and feature extraction are working.
        """

        try:
            height, width = self.image_size

            dummy = np.zeros(
                (
                    height,
                    width,
                    3,
                ),
                dtype=np.uint8,
            )

            tensor = self._preprocess(
                dummy
            )

            with torch.inference_mode():
                output = self.model(
                    tensor
                )

            features = self._extract_feature_tensor(
                output
            )

            if features is None:
                raise TypeError(
                    "OSNet warmup produced no "
                    "feature tensor."
                )

            if features.ndim == 1:
                features = features.unsqueeze(
                    0
                )

            if features.ndim != 2:
                raise ValueError(
                    "Invalid warmup feature shape: "
                    f"{tuple(features.shape)}"
                )

            self.embedding_dimension = int(
                features.shape[1]
            )

            self.warmup_successful = True

            logger.info(
                "OSNet warmup successful: "
                "output_shape=%s embedding_dimension=%d",
                tuple(features.shape),
                self.embedding_dimension,
            )

        except Exception:
            self.warmup_successful = False

            logger.exception(
                "OSNet warmup failed."
            )

    # ============================================================
    # PUBLIC EXTRACTION API
    # ============================================================

    def extract(
        self,
        frame: np.ndarray,
        bbox: Any,
    ) -> Optional[list[float]]:
        """
        Extract a normalized OSNet embedding.

        Returns:
            L2-normalized embedding as list[float]
            or None if extraction fails.
        """

        result = self.extract_with_diagnostics(
            frame,
            bbox,
        )

        if result is None:
            return None

        return result["embedding"]

    def extract_with_diagnostics(
        self,
        frame: np.ndarray,
        bbox: Any,
    ) -> Optional[dict[str, Any]]:
        """
        Extract an embedding together with diagnostics.

        Returns:

            {
                "embedding": [...],
                "quality_score": 0.72,
                "crop_width": 52,
                "crop_height": 118,
                "upscaled": True,
                "embedding_dimension": 512,
            }

        or None when extraction fails.

        This API is useful when the tracking layer needs
        to decide whether an embedding should become the
        current Track appearance embedding.
        """

        self.extraction_attempts += 1

        if not self._validate_frame(
            frame
        ):
            self._register_failure(
                "invalid_crops"
            )
            return None

        crop = self.crop_bbox(
            frame,
            bbox,
        )

        if crop is None:
            self._register_failure(
                "invalid_bboxes"
            )
            return None

        original_height, original_width = (
            crop.shape[:2]
        )

        quality = self._quality_score(
            crop
        )

        self.last_quality_score = quality

        if quality <= 0.0:
            self.zero_quality_crops += 1

        elif quality < self.min_quality_score:
            self.low_quality_crops += 1

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT reject a crop merely because it is small.
        #
        # Small players are normal in football footage.
        #
        # Upscale instead.
        # --------------------------------------------------------

        upscaled = False

        if self.upscale_small_crops:

            resized_crop, upscaled = (
                self._upscale_if_needed(
                    crop
                )
            )

            crop = resized_crop

        try:
            tensor = self._preprocess(
                crop
            )

        except Exception:

            self.preprocessing_failures += 1
            self.extraction_failures += 1

            logger.debug(
                "Appearance preprocessing failed.",
                exc_info=True,
            )

            return None

        try:

            with torch.inference_mode():

                output = self.model(
                    tensor
                )

        except RuntimeError:

            self.model_inference_failures += 1
            self.extraction_failures += 1

            logger.debug(
                "OSNet inference RuntimeError.",
                exc_info=True,
            )

            return None

        except Exception:

            self.model_inference_failures += 1
            self.extraction_failures += 1

            logger.debug(
                "OSNet inference failed.",
                exc_info=True,
            )

            return None

        features = self._extract_feature_tensor(
            output
        )

        if features is None:

            self.invalid_embeddings += 1
            self.extraction_failures += 1

            logger.debug(
                "OSNet returned no feature tensor."
            )

            return None

        if features.ndim == 1:

            features = features.unsqueeze(
                0
            )

        if features.ndim != 2:

            self.invalid_embeddings += 1
            self.extraction_failures += 1

            logger.debug(
                "Invalid OSNet feature shape: %s",
                tuple(features.shape),
            )

            return None

        if features.shape[0] < 1:

            self.invalid_embeddings += 1
            self.extraction_failures += 1

            return None

        # --------------------------------------------------------
        # Track embedding dimension.
        # --------------------------------------------------------

        dimension = int(
            features.shape[1]
        )

        if self.embedding_dimension is None:

            self.embedding_dimension = dimension

        elif (
            self.embedding_dimension
            != dimension
        ):

            self.invalid_embeddings += 1
            self.extraction_failures += 1

            logger.error(
                "Embedding dimension changed: "
                "expected=%s actual=%s",
                self.embedding_dimension,
                dimension,
            )

            return None

        # --------------------------------------------------------
        # L2 normalization.
        # --------------------------------------------------------

        features = F.normalize(
            features,
            p=2,
            dim=1,
            eps=1e-12,
        )

        embedding_tensor = features[0]

        # --------------------------------------------------------
        # Torch validation.
        # --------------------------------------------------------

        if not torch.isfinite(
            embedding_tensor
        ).all():

            self.invalid_embeddings += 1
            self.extraction_failures += 1

            logger.debug(
                "OSNet produced NaN/Inf embedding."
            )

            return None

        embedding = (
            embedding_tensor
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        # --------------------------------------------------------
        # NumPy validation.
        # --------------------------------------------------------

        if embedding.ndim != 1:

            self.invalid_embeddings += 1
            self.extraction_failures += 1

            return None

        if embedding.size == 0:

            self.invalid_embeddings += 1
            self.extraction_failures += 1

            return None

        if not np.all(
            np.isfinite(
                embedding
            )
        ):

            self.invalid_embeddings += 1
            self.extraction_failures += 1

            return None

        norm = float(
            np.linalg.norm(
                embedding
            )
        )

        if norm <= 1e-8:

            self.invalid_embeddings += 1
            self.extraction_failures += 1

            return None

        # --------------------------------------------------------
        # Final normalization.
        # --------------------------------------------------------

        embedding = (
            embedding / norm
        ).astype(
            np.float32,
            copy=False,
        )

        embedding_list = (
            embedding.tolist()
        )

        # --------------------------------------------------------
        # Store last successful embedding.
        #
        # This is intentionally NOT used as an automatic
        # substitute for failed crops.
        #
        # The tracking layer should explicitly decide when
        # to propagate an embedding.
        # --------------------------------------------------------

        self.last_embedding = (
            embedding_list
        )

        self.embeddings_extracted += 1

        return {
            "embedding": embedding_list,

            "quality_score": float(
                quality
            ),

            "crop_width": int(
                original_width
            ),

            "crop_height": int(
                original_height
            ),

            "processed_width": int(
                crop.shape[1]
            ),

            "processed_height": int(
                crop.shape[0]
            ),

            "upscaled": bool(
                upscaled
            ),

            "embedding_dimension": int(
                dimension
            ),
        }

    # ============================================================
    # FRAME VALIDATION
    # ============================================================

    @staticmethod
    def _validate_frame(
        frame: Any,
    ) -> bool:
        """
        Validate an OpenCV frame.
        """

        if frame is None:
            return False

        if not isinstance(
            frame,
            np.ndarray,
        ):
            return False

        if frame.size == 0:
            return False

        if frame.ndim != 3:
            return False

        if frame.shape[2] != 3:
            return False

        return True

    # ============================================================
    # FEATURE OUTPUT HANDLING
    # ============================================================

    @staticmethod
    def _extract_feature_tensor(
        features: Any,
    ) -> Optional[torch.Tensor]:
        """
        Convert different TorchReID output formats
        into a feature tensor.

        TorchReID models may return:

            Tensor

        or:

            tuple/list containing Tensor
        """

        if isinstance(
            features,
            torch.Tensor,
        ):
            return features

        if isinstance(
            features,
            (tuple, list),
        ):

            # Prefer the first 2D tensor because OSNet
            # appearance features are expected to be
            # [batch, embedding_dimension].

            for item in features:

                if not isinstance(
                    item,
                    torch.Tensor,
                ):
                    continue

                if item.ndim in (
                    1,
                    2,
                ):
                    return item

            # Fallback to any tensor.
            for item in features:

                if isinstance(
                    item,
                    torch.Tensor,
                ):
                    return item

        return None

    # ============================================================
    # CROP
    # ============================================================

    def crop_bbox(
        self,
        frame: np.ndarray,
        bbox: Any,
    ) -> Optional[np.ndarray]:
        """
        Safely crop a bounding box.

        Features:

        - validates coordinates
        - handles reversed boxes
        - handles NaN / Inf
        - adds proportional padding
        - clips to frame
        - prevents zero-area crops
        """

        if not self._validate_frame(
            frame
        ):
            return None

        frame_height, frame_width = (
            frame.shape[:2]
        )

        try:

            x1 = float(
                bbox.x1
            )

            y1 = float(
                bbox.y1
            )

            x2 = float(
                bbox.x2
            )

            y2 = float(
                bbox.y2
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ):

            return None

        coordinates = np.asarray(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(
                coordinates
            )
        ):
            return None

        # --------------------------------------------------------
        # Correct reversed coordinates.
        # --------------------------------------------------------

        if x2 < x1:
            x1, x2 = x2, x1

        if y2 < y1:
            y1, y2 = y2, y1

        width = x2 - x1
        height = y2 - y1

        if width <= 0.0:
            return None

        if height <= 0.0:
            return None

        # --------------------------------------------------------
        # Padding.
        # --------------------------------------------------------

        pad_x = (
            width
            * self.bbox_padding
        )

        pad_y = (
            height
            * self.bbox_padding
        )

        padded_x1 = x1 - pad_x
        padded_y1 = y1 - pad_y

        padded_x2 = x2 + pad_x
        padded_y2 = y2 + pad_y

        if self.bbox_padding > 0.0:
            self.padded_crops += 1

        # --------------------------------------------------------
        # Clip floating point coordinates.
        # --------------------------------------------------------

        padded_x1 = max(
            0.0,
            min(
                padded_x1,
                float(
                    frame_width - 1
                ),
            ),
        )

        padded_y1 = max(
            0.0,
            min(
                padded_y1,
                float(
                    frame_height - 1
                ),
            ),
        )

        padded_x2 = max(
            1.0,
            min(
                padded_x2,
                float(
                    frame_width
                ),
            ),
        )

        padded_y2 = max(
            1.0,
            min(
                padded_y2,
                float(
                    frame_height
                ),
            ),
        )

        # --------------------------------------------------------
        # Integer coordinates.
        # --------------------------------------------------------

        ix1 = int(
            np.floor(
                padded_x1
            )
        )

        iy1 = int(
            np.floor(
                padded_y1
            )
        )

        ix2 = int(
            np.ceil(
                padded_x2
            )
        )

        iy2 = int(
            np.ceil(
                padded_y2
            )
        )

        # --------------------------------------------------------
        # Final clipping.
        # --------------------------------------------------------

        ix1 = max(
            0,
            min(
                ix1,
                frame_width - 1,
            ),
        )

        iy1 = max(
            0,
            min(
                iy1,
                frame_height - 1,
            ),
        )

        ix2 = max(
            ix1 + 1,
            min(
                ix2,
                frame_width,
            ),
        )

        iy2 = max(
            iy1 + 1,
            min(
                iy2,
                frame_height,
            ),
        )

        if ix2 <= ix1:
            return None

        if iy2 <= iy1:
            return None

        crop = frame[
            iy1:iy2,
            ix1:ix2,
        ]

        if crop.size == 0:
            return None

        return np.ascontiguousarray(
            crop
        )

    # ============================================================
    # UPSCALING
    # ============================================================

    def _upscale_if_needed(
        self,
        crop: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        Upscale small crops.

        Unlike the previous implementation, small crops
        are NOT rejected before this function.

        This is important for football because distant
        players can naturally have tiny bounding boxes.
        """

        height, width = (
            crop.shape[:2]
        )

        if (
            height >= self.min_crop_height
            and width >= self.min_crop_width
        ):
            return crop, False

        scale_h = (
            self.min_crop_height
            / max(
                height,
                1,
            )
        )

        scale_w = (
            self.min_crop_width
            / max(
                width,
                1,
            )
        )

        scale = max(
            scale_h,
            scale_w,
        )

        new_width = max(
            self.min_crop_width,
            int(
                round(
                    width
                    * scale
                )
            ),
        )

        new_height = max(
            self.min_crop_height,
            int(
                round(
                    height
                    * scale
                )
            ),
        )

        try:

            resized = cv2.resize(
                crop,
                (
                    new_width,
                    new_height,
                ),
                interpolation=cv2.INTER_CUBIC,
            )

            self.upscaled_crops += 1

            return resized, True

        except Exception:

            logger.debug(
                "Failed to upscale "
                "appearance crop.",
                exc_info=True,
            )

            return crop, False

    # ============================================================
    # QUALITY
    # ============================================================

    @staticmethod
    def _quality_score(
        crop: np.ndarray,
    ) -> float:
        """
        Calculate lightweight image quality score.

        Uses Laplacian variance.

        Returns approximately [0, 1].

        IMPORTANT:

        This score is diagnostic only.

        Low quality does NOT automatically cause
        extraction failure.
        """

        try:

            if crop is None:
                return 0.0

            if crop.size == 0:
                return 0.0

            gray = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2GRAY,
            )

            variance = float(
                cv2.Laplacian(
                    gray,
                    cv2.CV_64F,
                ).var()
            )

            score = (
                variance
                / (
                    variance
                    + 100.0
                )
            )

            return float(
                np.clip(
                    score,
                    0.0,
                    1.0,
                )
            )

        except Exception:

            return 0.0

    # ============================================================
    # PREPROCESSING
    # ============================================================

    def _preprocess(
        self,
        crop: np.ndarray,
    ) -> torch.Tensor:
        """
        Convert BGR crop into OSNet input tensor.

        Pipeline:

            BGR
             ↓
            RGB
             ↓
            resize
             ↓
            float32
             ↓
            [0,1]
             ↓
            ImageNet normalization
             ↓
            CHW
             ↓
            batch
             ↓
            device
        """

        if crop is None:
            raise ValueError(
                "Crop cannot be None."
            )

        if crop.size == 0:
            raise ValueError(
                "Crop cannot be empty."
            )

        if crop.ndim != 3:
            raise ValueError(
                "Crop must be HWC image."
            )

        if crop.shape[2] != 3:
            raise ValueError(
                "Crop must contain 3 channels."
            )

        crop = np.ascontiguousarray(
            crop
        )

        # --------------------------------------------------------
        # BGR → RGB
        # --------------------------------------------------------

        rgb = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB,
        )

        height, width = (
            self.image_size
        )

        # --------------------------------------------------------
        # Resize.
        # --------------------------------------------------------

        rgb = cv2.resize(
            rgb,
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        # --------------------------------------------------------
        # Float32 [0,1].
        # --------------------------------------------------------

        image = rgb.astype(
            np.float32,
            copy=False,
        )

        image /= 255.0

        # --------------------------------------------------------
        # ImageNet normalization.
        # --------------------------------------------------------

        image = (
            image
            - self.IMAGENET_MEAN
        ) / self.IMAGENET_STD

        # --------------------------------------------------------
        # HWC → CHW.
        # --------------------------------------------------------

        image = np.transpose(
            image,
            (
                2,
                0,
                1,
            ),
        )

        image = np.ascontiguousarray(
            image,
            dtype=np.float32,
        )

        tensor = torch.from_numpy(
            image
        )

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            self.device,
            non_blocking=(
                self.device != "cpu"
            ),
        )

        return tensor

    # ============================================================
    # FAILURE TRACKING
    # ============================================================

    def _register_failure(
        self,
        category: str,
    ) -> None:
        """
        Register extraction failure.
        """

        self.extraction_failures += 1

        if category == "invalid_crops":

            self.invalid_crops += 1

        elif category == "invalid_bboxes":

            self.invalid_bboxes += 1

    # ============================================================
    # COSINE SIMILARITY
    # ============================================================

    @staticmethod
    def cosine_similarity(
        embedding_a: list[float],
        embedding_b: list[float],
    ) -> float:
        """
        Calculate cosine similarity between embeddings.
        """

        if not embedding_a:
            return 0.0

        if not embedding_b:
            return 0.0

        try:

            a = np.asarray(
                embedding_a,
                dtype=np.float32,
            )

            b = np.asarray(
                embedding_b,
                dtype=np.float32,
            )

        except Exception:

            return 0.0

        if a.ndim != 1:
            return 0.0

        if b.ndim != 1:
            return 0.0

        if a.shape != b.shape:
            return 0.0

        if not np.all(
            np.isfinite(a)
        ):
            return 0.0

        if not np.all(
            np.isfinite(b)
        ):
            return 0.0

        norm_a = float(
            np.linalg.norm(a)
        )

        norm_b = float(
            np.linalg.norm(b)
        )

        if norm_a <= 1e-8:
            return 0.0

        if norm_b <= 1e-8:
            return 0.0

        similarity = float(
            np.dot(a, b)
            / (
                norm_a
                * norm_b
            )
        )

        return float(
            np.clip(
                similarity,
                -1.0,
                1.0,
            )
        )

    # ============================================================
    # EMBEDDING VALIDATION
    # ============================================================

    @staticmethod
    def is_valid_embedding(
        embedding: Any,
    ) -> bool:
        """
        Check whether an embedding is usable.
        """

        if embedding is None:
            return False

        try:

            array = np.asarray(
                embedding,
                dtype=np.float32,
            )

        except Exception:

            return False

        if array.ndim != 1:
            return False

        if array.size == 0:
            return False

        if not np.all(
            np.isfinite(
                array
            )
        ):
            return False

        norm = float(
            np.linalg.norm(
                array
            )
        )

        if norm <= 1e-8:
            return False

        return True

    # ============================================================
    # STATISTICS
    # ============================================================

    def get_statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return detailed extraction statistics.
        """

        attempts = (
            self.extraction_attempts
        )

        success_rate = (
            self.embeddings_extracted
            / attempts
            if attempts > 0
            else 0.0
        )

        failure_rate = (
            self.extraction_failures
            / attempts
            if attempts > 0
            else 0.0
        )

        return {
            "model": self.model_name,

            "device": self.device,

            "image_size": self.image_size,

            "embedding_dimension": (
                self.embedding_dimension
            ),

            "bbox_padding": (
                self.bbox_padding
            ),

            "min_crop_width": (
                self.min_crop_width
            ),

            "min_crop_height": (
                self.min_crop_height
            ),

            "extraction_attempts": (
                attempts
            ),

            "embeddings_extracted": (
                self.embeddings_extracted
            ),

            "extraction_failures": (
                self.extraction_failures
            ),

            "success_rate": (
                success_rate
            ),

            "failure_rate": (
                failure_rate
            ),

            "invalid_crops": (
                self.invalid_crops
            ),

            "invalid_bboxes": (
                self.invalid_bboxes
            ),

            "small_crops": (
                self.small_crops
            ),

            "low_quality_crops": (
                self.low_quality_crops
            ),

            "zero_quality_crops": (
                self.zero_quality_crops
            ),

            "model_inference_failures": (
                self.model_inference_failures
            ),

            "preprocessing_failures": (
                self.preprocessing_failures
            ),

            "invalid_embeddings": (
                self.invalid_embeddings
            ),

            "upscaled_crops": (
                self.upscaled_crops
            ),

            "padded_crops": (
                self.padded_crops
            ),

            "warmup_successful": (
                self.warmup_successful
            ),
        }

    # ============================================================
    # COMPATIBILITY ALIAS
    # ============================================================

    def get_appearance_statistics(
        self,
    ) -> dict[str, Any]:
        """
        Compatibility alias used by ByteTrackTracker.
        """

        return self.get_statistics()

    # ============================================================
    # RESET
    # ============================================================

    def reset_statistics(
        self,
    ) -> None:
        """
        Reset extraction counters without reloading OSNet.
        """

        self.extraction_attempts = 0

        self.embeddings_extracted = 0

        self.extraction_failures = 0

        self.invalid_crops = 0
        self.invalid_bboxes = 0

        self.small_crops = 0
        self.low_quality_crops = 0

        self.model_inference_failures = 0
        self.preprocessing_failures = 0
        self.invalid_embeddings = 0

        self.upscaled_crops = 0
        self.padded_crops = 0

        self.zero_quality_crops = 0

        self.last_embedding = None
        self.last_quality_score = None

    # ============================================================
    # CLASS VALIDATION
    # ============================================================

    @staticmethod
    def is_valid_class(
        class_name: str,
    ) -> bool:
        """
        Return whether appearance extraction is applicable
        to this object class.
        """

        if not class_name:
            return False

        return (
            class_name.lower()
            in AppearanceExtractor.PERSON_CLASSES
        )

