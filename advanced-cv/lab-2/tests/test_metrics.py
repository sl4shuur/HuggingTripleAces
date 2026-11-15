import math

import pytest
import torch

from alseg.data.preprocessing import resize_mask
from alseg.metrics import confusion_matrix, segmentation_metrics
from alseg.nn.heads import LinearSegmentationHead, segmentation_loss


def test_hand_computed_iou_includes_background_and_ignores_void():
    targets = torch.tensor([[0, 0], [1, 255]])
    predictions = torch.tensor([[0, 1], [1, 20]])
    metrics = segmentation_metrics(confusion_matrix(predictions, targets))
    assert metrics["miou"] == pytest.approx(0.5)
    assert metrics["pixel_accuracy"] == pytest.approx(2 / 3)
    assert metrics["mean_class_accuracy"] == pytest.approx(0.75)
    assert metrics["valid_pixels"] == 3
    assert metrics["classes_with_union"] == 2
    assert metrics["per_class_iou"][20] is None


def test_dataset_confusion_is_summed_before_computing_iou():
    one = confusion_matrix(
        torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long)
    )
    many = confusion_matrix(
        torch.ones(1, 9, dtype=torch.long), torch.zeros(1, 9, dtype=torch.long)
    )
    assert segmentation_metrics(one + many)["miou"] == pytest.approx(0.05)


def test_all_void_metrics_and_loss_are_not_misreported_as_perfect():
    targets = torch.full((1, 2, 2), 255, dtype=torch.long)
    matrix = confusion_matrix(torch.zeros_like(targets), targets)
    metrics = segmentation_metrics(matrix)
    assert metrics["valid_pixels"] == 0 and math.isnan(metrics["miou"])
    logits = torch.randn(1, 21, 2, 2, requires_grad=True)
    loss = segmentation_loss(logits, targets)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_loss_matches_mean_cross_entropy_over_nonvoid_pixels():
    logits = torch.randn(2, 21, 2, 3, requires_grad=True)
    targets = torch.tensor([[[0, 1, 255], [2, 3, 4]], [[20, 255, 0], [1, 1, 255]]])
    expected = torch.nn.functional.cross_entropy(logits, targets, ignore_index=255)
    torch.testing.assert_close(segmentation_loss(logits, targets), expected)


def test_mask_resize_preserves_class_ids_and_head_supports_native_shapes():
    mask = torch.tensor([[0, 1], [20, 255]])
    resized = resize_mask(mask, 8)
    assert resized.unique().tolist() == [0, 1, 20, 255]
    assert resized.dtype == torch.long and resized.shape == (8, 8)
    head = LinearSegmentationHead(3)
    assert head(torch.ones(2, 3, 2, 2), (7, 11)).shape == (2, 21, 7, 11)


def test_invalid_class_ids_are_rejected():
    with pytest.raises(ValueError, match="class IDs"):
        confusion_matrix(torch.tensor([[21]]), torch.tensor([[0]]))
