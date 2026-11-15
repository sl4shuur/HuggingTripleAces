"""Raw segmentation predictions and geometry-only validity masks."""

from dataclasses import dataclass

import torch
from torch import Tensor

from alseg.core.helpers import check_ids, check_tensor

NUM_CLASSES = 21
IGNORE_INDEX = 255
CLASS_NAMES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


@dataclass(frozen=True)
class LogitBatch:
    """Raw logits and a geometry-only mask; never contains unlabeled targets."""

    ids: list[int]
    logits: Tensor
    valid_pixels: Tensor

    def __post_init__(self) -> None:
        check_ids(self.ids, "ids")
        check_tensor(self.logits, "logits", torch.float32, 4)
        check_tensor(self.valid_pixels, "valid_pixels", torch.bool, 3)
        b, c, h, w = self.logits.shape
        if b == 0 or h == 0 or w == 0 or c != NUM_CLASSES or len(self.ids) != b:
            raise ValueError(
                "logits must have shape [len(ids),21,H,W] with nonempty dimensions"
            )
        if self.valid_pixels.shape != (b, h, w):
            raise ValueError(
                "valid_pixels must match logits batch and spatial dimensions"
            )
        if not self.valid_pixels.flatten(1).any(dim=1).all():
            raise ValueError(
                "Each image must contain at least one valid geometry pixel"
            )
