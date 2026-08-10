
"""
OSNet ReID appearance embedding extractor.

## Appearance/ReID V2

This module extracts person appearance embeddings using
OSNet x1.0 from Torchreid.

Responsibilities:

- Load pretrained OSNet x1.0
- Crop a bounding box from an OpenCV frame
- Convert BGR -> RGB
- Apply OSNet preprocessing
- Extract the feature vector
- L2-normalize the embedding

This module is intentionally independent from:

- ByteTrack
- GlobalIdentityManager
- IdentityMatcher
- AppearanceGallery
- Team assignment
- Jersey recognition

The output embedding can later be used by the appearance
gallery and global identity matching stages.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchreid

from ai_engine.schemas.track import BoundingBox


class OSNetReIDEmbedder:
    """
    Extract normalized person appearance embeddings using
    OSNet x1.0.

    OSNet is a person ReID architecture designed to produce
    discriminative appearance features.

    The classifier layer is not used. The network's feature
    representation is returned instead.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        input_size: tuple[int, int] = (256, 128),
    ) -> None:

        self.input_size = input_size

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        # --------------------------------------------------
        # Build pretrained OSNet
        # --------------------------------------------------

        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1,
            pretrained=True,
        )

        self.model.eval()
        self.model.to(self.device)

        # --------------------------------------------------
        # OSNet feature dimension
        #
        # OSNet x1.0 produces a 512-dimensional feature
        # representation before the classifier.
        # --------------------------------------------------

        self.embedding_dimension = 512

        # --------------------------------------------------
        # Standard ImageNet preprocessing used by Torchreid
        # --------------------------------------------------

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.input_size
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )

    # ======================================================
    # PUBLIC API
    # ======================================================

    def embed_crop(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
    ) -> Optional[list[float]]:
        """
        Extract an OSNet embedding from a bounding box.

        Args:
            frame:
                OpenCV BGR image.

            bbox:
                BoundingBox in image coordinates.

        Returns:
            L2-normalized embedding as a list of floats.

            Returns None if the crop is invalid.
        """

        crop = self._crop_frame(
            frame,
            bbox,
        )

        if crop is None:
            return None

        return self.embed_image(
            crop
        )

    def embed_image(
        self,
        image: np.ndarray,
    ) -> Optional[list[float]]:
        """
        Extract an OSNet embedding from an image crop.

        Input:
            OpenCV BGR image.

        Output:
            L2-normalized 512-dimensional embedding.
        """

        if image is None:
            return None

        if image.size == 0:
            return None

        # --------------------------------------------------
        # OpenCV BGR -> RGB
        # --------------------------------------------------

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # --------------------------------------------------
        # NumPy -> PIL
        # --------------------------------------------------

        pil_image = Image.fromarray(
            rgb
        )

        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------

        tensor = self.transform(
            pil_image
        )

        # Add batch dimension:
        #
        # [3, H, W]
        #      ↓
        # [1, 3, H, W]
        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            self.device
        )

        # --------------------------------------------------
        # Feature extraction
        # --------------------------------------------------

        with torch.no_grad():

            features = self.model(
                tensor
            )

        # --------------------------------------------------
        # Flatten if necessary
        #
        # Expected:
        #
        # [1, 512]
        #
        # Some model configurations may produce additional
        # dimensions, so flatten safely.
        # --------------------------------------------------

        if features.ndim > 2:

            features = torch.flatten(
                features,
                start_dim=1,
            )

        # --------------------------------------------------
        # L2 normalization
        # --------------------------------------------------

        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=1,
        )

        # --------------------------------------------------
        # Convert to NumPy
        # --------------------------------------------------

        embedding = (
            features[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return embedding.tolist()

    # ======================================================
    # CROPPING
    # ======================================================

    @staticmethod
    def _crop_frame(
        frame: np.ndarray,
        bbox: BoundingBox,
    ) -> Optional[np.ndarray]:
        """
        Safely crop a bounding box from an image.

        Coordinates are clipped to the image boundaries.
        """

        if frame is None:
            return None

        if frame.size == 0:
            return None

        height, width = frame.shape[:2]

        # --------------------------------------------------
        # Clip coordinates
        # --------------------------------------------------

        x1 = max(
            0,
            min(
                width,
                int(round(bbox.x1)),
            ),
        )

        y1 = max(
            0,
            min(
                height,
                int(round(bbox.y1)),
            ),
        )

        x2 = max(
            0,
            min(
                width,
                int(round(bbox.x2)),
            ),
        )

        y2 = max(
            0,
            min(
                height,
                int(round(bbox.y2)),
            ),
        )

        # --------------------------------------------------
        # Validate bounding box
        # --------------------------------------------------

        if x2 <= x1:
            return None

        if y2 <= y1:
            return None

        crop = frame[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            return None

        return crop

