"""Frozen image and dense patch representations at the encoder boundary."""

from dataclasses import dataclass

import torch
from torch import Tensor

from alseg.core.helpers import check_tensor


@dataclass(frozen=True)
class EncoderFeatures:
    embeddings: Tensor
    patch_features: Tensor

    def __post_init__(self) -> None:
        check_tensor(self.embeddings, "embeddings", torch.float32, 2)
        check_tensor(self.patch_features, "patch_features", torch.float32, 4)
        b, d = self.embeddings.shape
        if b == 0 or d == 0 or self.patch_features.shape[:2] != (b, d):
            raise ValueError("Expected embeddings [B,D] and patches [B,D,h,w]")
        if min(self.patch_features.shape[2:]) <= 0:
            raise ValueError("Patch grid must be nonempty")
        norms = torch.linalg.vector_norm(self.embeddings, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
            raise ValueError("Image embeddings must be L2-normalized")
        if self.embeddings.requires_grad or self.patch_features.requires_grad:
            raise ValueError("Frozen encoder features must not require gradients")
