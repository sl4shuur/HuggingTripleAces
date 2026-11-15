"""Shared checks for CPU tensors and canonical pool IDs."""

import torch
from torch import Tensor


def check_ids(ids: list[int], name: str) -> None:
    if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError(f"{name} must be a list of nonnegative Python integers")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} contains duplicate IDs")


def check_tensor(value: Tensor, name: str, dtype: torch.dtype, ndim: int) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.device.type != "cpu" or value.dtype != dtype or value.ndim != ndim:
        raise ValueError(f"{name} must be a CPU {dtype} tensor with {ndim} dimensions")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
