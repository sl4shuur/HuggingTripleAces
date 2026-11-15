"""Shared helpers for the project."""

import csv
import json
from pathlib import Path

import torch
from torch import Tensor


def check_ids(ids: list[int], name: str) -> None:
    if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError(f"{name} must be a list of nonnegative Python integers")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} contains duplicate IDs")


def check_tensor(value: Tensor, name: str, dtype: torch.dtype, ndim: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.device.type != "cpu" or value.dtype != dtype or value.ndim != ndim:
        raise ValueError(f"{name} must be a CPU {dtype} tensor with {ndim} dimensions")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, data: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
