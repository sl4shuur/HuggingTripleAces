"""Held-out queried labels for temperature calibration."""

from dataclasses import dataclass

import torch
from torch import Tensor

from alseg.core.predictions import IGNORE_INDEX, NUM_CLASSES, LogitBatch
from alseg.core.validation import validate_tensor


@dataclass(frozen=True)
class CalibrationBatch:
    """Only held-out queried labels may be provided here; never official val."""

    predictions: LogitBatch
    targets: Tensor

    def __post_init__(self) -> None:
        validate_tensor(self.targets, "targets", torch.int64, 3)
        if self.targets.shape != self.predictions.valid_pixels.shape:
            raise ValueError("Calibration targets must match predictions")
        valid_labels = ((self.targets >= 0) & (self.targets < NUM_CLASSES)) | (
            self.targets == IGNORE_INDEX
        )
        if not valid_labels.all():
            raise ValueError("Targets must be classes 0..20 or ignore label 255")
        if (self.targets[~self.predictions.valid_pixels] != IGNORE_INDEX).any():
            raise ValueError("Padding targets must use ignore label 255")
