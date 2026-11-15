"""Held-out queried labels for temperature calibration."""

from dataclasses import dataclass

import torch
from torch import Tensor

from alseg.core.helpers import check_ids, check_tensor
from alseg.core.predictions import IGNORE_INDEX, NUM_CLASSES, LogitBatch


def split_labeled_ids(
    labeled_ids: list[int], fraction: float, min_images: int, seed: int
) -> tuple[list[int], list[int]]:
    """Shared per-round holdout: canonical IDs, independent RNG, at least one fit image.

    A zero fraction disables the holdout. Tiny smoke budgets cap it at len(L)-1.
    """
    check_ids(labeled_ids, "labeled_ids")
    if not labeled_ids:
        raise ValueError("Cannot split an empty labeled set")
    if not 0 <= fraction < 1 or type(min_images) is not int or min_images < 0:
        raise ValueError("Invalid calibration fraction or minimum image count")
    ids = sorted(labeled_ids)
    count = (
        min(len(ids) - 1, max(min_images, int(fraction * len(ids)))) if fraction else 0
    )
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))
    calibration = {ids[i] for i in order[:count].tolist()}
    return [i for i in ids if i not in calibration], [
        i for i in ids if i in calibration
    ]


@dataclass(frozen=True)
class CalibrationBatch:
    """Only held-out queried labels may be provided here; never official val."""

    predictions: LogitBatch
    targets: Tensor

    def __post_init__(self) -> None:
        check_tensor(self.targets, "targets", torch.int64, 3)
        if self.targets.shape != self.predictions.valid_pixels.shape:
            raise ValueError("Calibration targets must match predictions")
        valid_labels = ((self.targets >= 0) & (self.targets < NUM_CLASSES)) | (
            self.targets == IGNORE_INDEX
        )
        if not valid_labels.all():
            raise ValueError("Targets must be classes 0..20 or ignore label 255")
        if (self.targets[~self.predictions.valid_pixels] != IGNORE_INDEX).any():
            raise ValueError("Padding targets must use ignore label 255")
