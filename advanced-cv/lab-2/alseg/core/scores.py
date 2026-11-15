"""Alignment of image uncertainty scores to permanent pool rows."""

import torch
from torch import Tensor

from alseg.core.validation import validate_ids, validate_tensor


def assemble_scores(n_pool: int, batches: list[tuple[list[int], Tensor]]) -> Tensor:
    """Map out-of-order inference batches back to every canonical target row."""
    if type(n_pool) is not int or n_pool <= 0:
        raise ValueError("n_pool must be a positive integer")
    result = torch.empty(n_pool, dtype=torch.float32)
    seen: set[int] = set()
    for ids, scores in batches:
        validate_ids(ids, "score batch IDs")
        validate_tensor(scores, "scores", torch.float32, 1)
        if scores.shape != (len(ids),) or (scores < 0).any():
            raise ValueError("Scores must match IDs and be nonnegative")
        if any(i >= n_pool for i in ids) or seen.intersection(ids):
            raise ValueError("Score batches contain repeated or out-of-pool IDs")
        result[ids] = scores
        seen.update(ids)
    if len(seen) != n_pool:
        raise ValueError("Scores must cover the entire pool, including labeled targets")
    return result
