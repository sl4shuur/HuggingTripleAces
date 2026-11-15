"""Acquisition requests and selected-ID invariants."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from alseg.core.validation import validate_ids, validate_tensor

Method = Literal["random", "uncertainty", "maxherding", "uherding"]


@dataclass(frozen=True)
class SelectionRequest:
    """Select n_select NEW IDs; uncertainty weights belong to all N target rows.

    The runner resolves cold-start fallback before creating this request.
    Selectors must not mutate tensors or lists. Ties use the smallest pool ID.
    """

    method: Method
    embeddings: Tensor
    labeled_ids: list[int]
    unlabeled_ids: list[int]
    n_select: int
    seed: int
    uncertainty: Tensor | None = None
    sigma: float | None = None

    def __post_init__(self) -> None:
        if self.method not in ("random", "uncertainty", "maxherding", "uherding"):
            raise ValueError(f"Unknown acquisition method: {self.method}")
        validate_tensor(self.embeddings, "embeddings", torch.float32, 2)
        n, d = self.embeddings.shape
        if n == 0 or d == 0:
            raise ValueError("embeddings must have nonempty shape [N,D]")
        norms = torch.linalg.vector_norm(self.embeddings, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
            raise ValueError("embeddings must be L2-normalized")
        validate_ids(self.labeled_ids, "labeled_ids")
        validate_ids(self.unlabeled_ids, "unlabeled_ids")
        labeled, unlabeled = set(self.labeled_ids), set(self.unlabeled_ids)
        if labeled & unlabeled or labeled | unlabeled != set(range(n)):
            raise ValueError("L and U must form a disjoint partition of the pool")
        if self.unlabeled_ids != sorted(self.unlabeled_ids):
            raise ValueError("unlabeled_ids must be in increasing canonical ID order")
        if type(self.n_select) is not int or not 0 <= self.n_select <= len(unlabeled):
            raise ValueError("n_select must be an integer between zero and len(U)")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if self.uncertainty is not None:
            validate_tensor(self.uncertainty, "uncertainty", torch.float32, 1)
            if self.uncertainty.shape != (n,) or (self.uncertainty < 0).any():
                raise ValueError(
                    "uncertainty must contain N nonnegative target weights"
                )
        if self.method in ("uncertainty", "uherding") and self.uncertainty is None:
            raise ValueError(
                "Model-based acquisition requires uncertainty; resolve cold start first"
            )
        if self.sigma is not None and (
            isinstance(self.sigma, bool)
            or not isinstance(self.sigma, (int, float))
            or not math.isfinite(self.sigma)
            or self.sigma <= 0
        ):
            raise ValueError("sigma must be finite and positive")
        if self.method in ("maxherding", "uherding") and self.sigma is None:
            raise ValueError("Herding requires an explicitly resolved sigma")


def validate_selection(request: SelectionRequest, selected_ids: list[int]) -> None:
    """Runner check to perform before making any oracle query."""
    validate_ids(selected_ids, "selected_ids")
    if len(selected_ids) != request.n_select:
        raise ValueError("Selector returned the wrong number of new IDs")
    if not set(selected_ids) <= set(request.unlabeled_ids):
        raise ValueError("Selector returned an already-labeled or out-of-pool ID")
