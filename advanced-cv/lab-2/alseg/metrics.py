"""Dataset-wide semantic segmentation metrics, including background, ignoring 255."""

import torch
from torch import Tensor

from alseg.core.predictions import IGNORE_INDEX, NUM_CLASSES


def confusion_matrix(predictions: Tensor, targets: Tensor) -> Tensor:
    """Rows are ground truth, columns are predictions; void pixels never count."""
    if predictions.shape != targets.shape or targets.ndim not in (2, 3):
        raise ValueError(
            "Predictions/targets must have matching [H,W] or [B,H,W] shapes"
        )
    if predictions.dtype != torch.int64 or targets.dtype != torch.int64:
        raise ValueError("Predictions/targets must use int64 class IDs")
    valid = targets != IGNORE_INDEX
    truth, predicted = targets[valid], predictions[valid]
    if (
        (truth < 0)
        | (truth >= NUM_CLASSES)
        | (predicted < 0)
        | (predicted >= NUM_CLASSES)
    ).any():
        raise ValueError("Invalid non-void class IDs")
    return torch.bincount(
        truth * NUM_CLASSES + predicted,
        minlength=NUM_CLASSES**2,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def segmentation_metrics(matrix: Tensor) -> dict:
    """Compute ratios after summing confusion matrices, never average image IoUs."""
    if matrix.shape != (NUM_CLASSES, NUM_CLASSES) or (matrix < 0).any():
        raise ValueError("Expected a nonnegative [21,21] confusion matrix")
    matrix = matrix.double()
    intersection = matrix.diag()
    support = matrix.sum(dim=1)
    union = support + matrix.sum(dim=0) - intersection
    iou = torch.where(union > 0, intersection / union, torch.nan)
    accuracy = torch.where(support > 0, intersection / support, torch.nan)
    return {
        "miou": iou.nanmean().item(),
        "pixel_accuracy": (intersection.sum() / matrix.sum()).item(),
        "mean_class_accuracy": accuracy.nanmean().item(),
        "classes_with_union": int((union > 0).sum()),
        "valid_pixels": int(matrix.sum()),
        "per_class_iou": [
            None if torch.isnan(value) else value.item() for value in iou
        ],
    }
