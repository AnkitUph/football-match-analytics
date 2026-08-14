"""
Stage 3b: Re-ID Embeddings.

This is the SECOND-PASS addition that lets Stage 5 re-identify a player
after a camera cut, when spatial continuity is gone and 11 similarly-
dressed players make color alone ambiguous. Build this only after the
core single-angle pipeline (Stages 1,2,3-color,4,5,6,7) works end to end.

Uses torchreid's pretrained OSNet — no training needed to start. Only
fine-tune on your own footage later if match quality is insufficient.
"""

import numpy as np

from ai_engine.config import TeamReidConfig
from ai_engine.utils.types import Detection


class ReidEmbedder:
    def __init__(self, config: TeamReidConfig):
        self.config = config
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        # TODO:
        # import torchreid
        # self._model = torchreid.utils.FeatureExtractor(
        #     model_name=self.config.reid_model_name,
        #     model_path="",  # empty = use pretrained ImageNet/ReID weights
        #     device="cuda",
        # )
        raise NotImplementedError(
            "ReidEmbedder._load_model is stubbed — install torchreid."
        )

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """
        Returns a fixed-size embedding vector (config.reid_embedding_dim)
        for one player crop.

        TODO:
            model = self._load_model()
            features = model(crop)  # torchreid handles resize/normalize
            return features.cpu().numpy().flatten()
        """
        raise NotImplementedError("ReidEmbedder.embed is stubbed.")

    def update_running_average(
        self, existing: np.ndarray | None, new_embedding: np.ndarray, alpha: float = 0.3
    ) -> np.ndarray:
        """
        Exponential moving average for a tracklet's/identity's embedding,
        so one bad crop (motion blur, partial occlusion) doesn't wreck the
        signature. Call this every time you get a new embedding for the
        same tracklet/identity, not just once.
        """
        if existing is None:
            return new_embedding
        return alpha * new_embedding + (1 - alpha) * existing


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
