"""A fresh lightweight segmentation head over frozen dense encoder features."""

import torch.nn.functional as F
from torch import Tensor, nn

from alseg.core.predictions import IGNORE_INDEX, NUM_CLASSES


class LinearSegmentationHead(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.classifier = nn.Conv2d(feature_dim, NUM_CLASSES, kernel_size=1)

    def forward(self, features: Tensor, size: tuple[int, int]) -> Tensor:
        return F.interpolate(
            self.classifier(features),
            size=size,
            mode="bilinear",
            align_corners=False,
        )


def segmentation_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean over labeled pixels; all-void batches produce a finite zero loss."""
    count = (targets != IGNORE_INDEX).sum()
    # CUDA nll_loss2d's internal sum is not deterministic; reduce pixel losses separately.
    pixels = F.cross_entropy(
        logits, targets, ignore_index=IGNORE_INDEX, reduction="none"
    )
    return pixels.sum() / count.clamp_min(1)
