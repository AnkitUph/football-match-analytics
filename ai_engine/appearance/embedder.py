"""
Appearance embedding extraction.

Appearance/ReID V1
-------------------

This module converts a person bounding box from an OpenCV frame
into a normalized appearance embedding.

It is intentionally independent from:

- ByteTrack
- GlobalIdentityManager
- IdentityMatcher
- Team assignment
- Jersey recognition

Those integrations will happen later.

V1 uses a pretrained torchvision ResNet18 as a baseline
appearance feature extractor.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from ai_engine.schemas.track import BoundingBox


class AppearanceEmbedder:
    """
    Extract appearance embeddings from image crops.

    The classifier head of ResNet18 is removed so that the
    network produces a feature representation instead of
    class probabilities.

    Output embeddings are L2-normalized and therefore suitable
    for cosine similarity.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        input_size: int = 224,
    ) -> None:

        self.input_size = int(input_size)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        # --------------------------------------------------
        # Load pretrained ResNet18
        # --------------------------------------------------

        weights = ResNet18_Weights.DEFAULT

        model = resnet18(weights=weights)

        # Remove the classification layer.
        #
        # Original:
        #
        #     512 features -> 1000 classes
        #
        # We keep the 512-dimensional feature vector.
        self.model = nn.Sequential(
            *list(model.children())[:-1]
        )

        self.model.eval()
        self.model.to(self.device)

        # --------------------------------------------------
        # Official preprocessing for the pretrained model.
        # --------------------------------------------------

        self.transform = weights.transforms()

        # ResNet18 produces 512-dimensional embeddings.
        self.embedding_dimension = 512

    # ======================================================
    # PUBLIC API
    # ======================================================

    def embed_crop(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
    ) -> Optional[list[float]]:
        """
        Extract an appearance embedding from a bounding box.

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
        Extract an embedding from an image.

        The input image is expected to be an OpenCV BGR image.
        """

        if image is None:
            return None

        if image.size == 0:
            return None

        # --------------------------------------------------
        # OpenCV uses BGR.
        # Torchvision pretrained models expect RGB.
        # --------------------------------------------------

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # Convert numpy image to PIL through the transform's
        # expected interface.
        from PIL import Image

        pil_image = Image.fromarray(
            rgb
        )

        tensor = self.transform(
            pil_image
        )

        tensor = tensor.unsqueeze(
            0
        ).to(self.device)

        # --------------------------------------------------
        # Inference
        # --------------------------------------------------

        with torch.no_grad():
            features = self.model(
                tensor
            )

        # ResNet output:
        #
        # [batch, 512, 1, 1]
        #
        # Convert to:
        #
        # [batch, 512]
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
        Crop a bounding box safely from an image.

        Coordinates are clipped to image boundaries.
        """

        if frame is None:
            return None

        if frame.size == 0:
            return None

        height, width = frame.shape[:2]

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

        # Invalid bounding box.
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