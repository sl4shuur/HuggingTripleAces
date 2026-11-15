"""Official-val reporting at native mask resolution."""

import torch

from alseg.core.predictions import IGNORE_INDEX, NUM_CLASSES
from alseg.interfaces.data import EvaluationSource
from alseg.interfaces.encoder import FeatureSource
from alseg.metrics import confusion_matrix, segmentation_metrics
from alseg.nn.extraction import iter_features
from alseg.nn.heads import segmentation_loss


@torch.no_grad()
def evaluate_head(
    head,
    encoder: FeatureSource,
    source: EvaluationSource,
    ids: list[int],
    batch_size: int,
    device: str,
):
    """Report at original mask resolution, summing confusion over the entire split."""
    head.to(device).eval()
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    loss_sum = 0.0
    previews = []
    for batch_ids, features in iter_features(source, encoder, ids, batch_size):
        for image_id, patches in zip(batch_ids, features.patch_features, strict=True):
            target = source.get_mask(image_id)
            logits = head(patches[None].to(device), target.shape)
            prediction = logits.argmax(dim=1)[0].cpu()
            matrix += confusion_matrix(prediction, target)
            loss_sum += segmentation_loss(logits, target[None].to(device)).item() * int(
                (target != IGNORE_INDEX).sum()
            )
            if len(previews) < 3:
                previews.append(
                    {
                        "stem": f"val-{image_id}",
                        "image": source.get_image(image_id),
                        "prediction": prediction,
                        "target": target,
                    }
                )
    metrics = segmentation_metrics(matrix)
    if metrics["valid_pixels"] == 0:
        raise ValueError("Evaluation contains no valid pixels")
    metrics["loss"] = loss_sum / metrics["valid_pixels"]
    head.cpu()
    return metrics, previews
