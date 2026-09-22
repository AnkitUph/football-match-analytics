"""
Stage 3b: Re-ID & Visual Embeddings.

Extracts rich visual feature embeddings for player crops using Hugging Face's
vision foundation model (facebook/dinov2-small).

DINOv2 produces 384-dimensional feature vectors that capture overall player
appearance, kit patterns, silhouette, and color distributions while remaining
invariant to lighting variations, player orientation, and background grass.

Used for:
1. Team Classification: Unsupervised clustering of player tracklet embeddings.
2. Tracklet Stitching: Merging fragmented tracklets across brief occlusions.
"""

import logging
import numpy as np

from ai_engine.config import TeamReidConfig

logger = logging.getLogger(__name__)

# Global singleton cache for processor & model to avoid reloading weights
_DINO_PROCESSOR = None
_DINO_MODEL = None
_DEVICE = None


def _get_dino_model(model_name: str = "facebook/dinov2-small"):
    global _DINO_PROCESSOR, _DINO_MODEL, _DEVICE
    if _DINO_MODEL is not None:
        return _DINO_PROCESSOR, _DINO_MODEL, _DEVICE

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _DINO_PROCESSOR = AutoImageProcessor.from_pretrained(model_name)
        _DINO_MODEL = AutoModel.from_pretrained(model_name).to(_DEVICE)
        _DINO_MODEL.eval()
        logger.info(f"Loaded DINOv2 model '{model_name}' on {_DEVICE}")
    except Exception as e:
        logger.warning(f"Could not load Hugging Face model '{model_name}': {e}. Falling back to color signatures.")
        _DINO_PROCESSOR, _DINO_MODEL = None, None

    return _DINO_PROCESSOR, _DINO_MODEL, _DEVICE


class ReidEmbedder:
    """
    Extracts L2-normalized 384-d appearance embeddings from player image crops.
    """
    def __init__(self, config: TeamReidConfig | None = None, model_name: str = "facebook/dinov2-small"):
        self.config = config
        self.model_name = model_name
        self.embedding_dim = 384

    def embed(self, crop: np.ndarray) -> np.ndarray | None:
        """
        Extracts embedding for a single BGR image crop (np.ndarray).
        Returns an L2-normalized float32 vector of shape (384,), or None if crop is invalid.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 5:
            return None
        batch_res = self.embed_batch([crop])
        return batch_res[0] if batch_res else None

    def embed_batch(self, crops: list[np.ndarray]) -> list[np.ndarray | None]:
        """
        Extracts embeddings for a list of BGR image crops in one batched forward pass.
        Returns a list of L2-normalized float32 vectors of shape (384,) or None for invalid crops.
        """
        if not crops:
            return []

        results: list[np.ndarray | None] = [None] * len(crops)
        valid_indices = []
        valid_pil_crops = []

        import cv2
        from PIL import Image
        import torch

        for idx, crop in enumerate(crops):
            if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 5:
                continue
            try:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                valid_pil_crops.append(Image.fromarray(rgb))
                valid_indices.append(idx)
            except Exception:
                continue

        if not valid_pil_crops:
            return results

        processor, model, device = _get_dino_model(self.model_name)

        if processor is None or model is None:
            # Color-histogram fallback embedding (384-dim)
            for idx, pil_img in zip(valid_indices, valid_pil_crops):
                arr = np.array(pil_img)
                # Compute 3-channel 8x8x6 color histogram (384 bins)
                hist, _ = np.histogramdd(
                    arr.reshape(-1, 3),
                    bins=(8, 8, 6),
                    range=[(0, 256), (0, 256), (0, 256)]
                )
                feat = hist.flatten().astype(np.float32)
                norm = np.linalg.norm(feat)
                results[idx] = (feat / norm) if norm > 1e-6 else feat
            return results

        try:
            inputs = processor(images=valid_pil_crops, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                # Take CLS token embedding
                cls_tokens = outputs.last_hidden_state[:, 0, :]
                norm_tokens = torch.nn.functional.normalize(cls_tokens, p=2, dim=1)
                feats = norm_tokens.cpu().numpy().astype(np.float32)

            for i, idx in enumerate(valid_indices):
                results[idx] = feats[i]
        except Exception as e:
            logger.warning(f"DINOv2 inference error: {e}")
            for idx, pil_img in zip(valid_indices, valid_pil_crops):
                arr = np.array(pil_img)
                hist, _ = np.histogramdd(arr.reshape(-1, 3), bins=(8, 8, 6), range=[(0, 256), (0, 256), (0, 256)])
                feat = hist.flatten().astype(np.float32)
                norm = np.linalg.norm(feat)
                results[idx] = (feat / norm) if norm > 1e-6 else feat

        return results

    def update_running_average(
        self, existing: np.ndarray | None, new_embedding: np.ndarray, alpha: float = 0.3
    ) -> np.ndarray:
        """
        Exponential moving average for a tracklet's embedding vector.
        """
        if existing is None:
            return new_embedding
        blended = alpha * new_embedding + (1.0 - alpha) * existing
        norm = np.linalg.norm(blended)
        return (blended / norm) if norm > 1e-6 else blended


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two 1D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)

