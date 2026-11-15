"""Small deterministic inputs for developing acquisition without a task model."""

from dataclasses import dataclass

import torch

from alseg.core.calibration import CalibrationBatch
from alseg.core.pool import PoolManifest
from alseg.core.predictions import LogitBatch
from alseg.core.selection import AcquisitionMethod, SelectionRequest


@dataclass(frozen=True)
class SyntheticFixture:
    manifest: PoolManifest
    request: SelectionRequest
    predictions: LogitBatch
    calibration: CalibrationBatch
    expected_top_uncertainty: list[int]


def make_fixture(embedding_dim: int = 2) -> SyntheticFixture:
    """D=2 and D=5 expose accidental encoder-specific dimension assumptions."""
    if embedding_dim not in (2, 5):
        raise ValueError("Synthetic fixture supports embedding dimensions 2 and 5")
    embeddings = torch.zeros(6, embedding_dim, dtype=torch.float32)
    embeddings[:, :2] = torch.tensor(
        [[1, 0], [0.6, 0.8], [-0.6, 0.8], [-1, 0], [-0.6, -0.8], [0.6, -0.8]]
    )
    request = SelectionRequest(
        method=AcquisitionMethod.UNCERTAINTY,
        embeddings=embeddings,
        labeled_ids=[4, 0],
        unlabeled_ids=[1, 2, 3, 5],
        n_select=2,
        seed=42,
        uncertainty=torch.tensor([0.1, 0.9, 0.5, 0.9, 0.2, 0.3]),
    )
    # Batch order deliberately differs from canonical pool order.
    logits = torch.zeros(2, 21, 2, 3, dtype=torch.float32)
    logits[0, 1] = 4.0
    valid_pixels = torch.ones(2, 2, 3, dtype=torch.bool)
    valid_pixels[:, :, -1] = False
    predictions = LogitBatch([3, 1], logits, valid_pixels)
    calibration = CalibrationBatch(
        predictions=LogitBatch([4, 0], logits.clone(), valid_pixels.clone()),
        targets=torch.tensor(
            [[[1, 1, 255], [1, 255, 255]], [[0, 2, 255], [2, 0, 255]]]
        ),
    )
    return SyntheticFixture(
        manifest=PoolManifest(
            [f"synthetic_train_{i}" for i in range(6)], ["synthetic_val_0"]
        ),
        request=request,
        predictions=predictions,
        calibration=calibration,
        expected_top_uncertainty=[1, 3],
    )
