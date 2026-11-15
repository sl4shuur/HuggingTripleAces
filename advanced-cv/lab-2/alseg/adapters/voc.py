"""Concrete image, evaluation, and oracle access to VOC files."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor

from alseg.core.helpers import check_ids
from alseg.core.pool import PoolManifest
from alseg.data import load_mask, load_rgb
from alseg.interfaces import EvaluationSource, ImageSource, Oracle


class VocImages(ImageSource):
    """Image-only train-pool access for feature extraction and selection."""

    def __init__(self, root: Path, manifest: PoolManifest):
        self._root = Path(root)
        self._ids = tuple(manifest.pool_ids)

    def __len__(self) -> int:
        return len(self._ids)

    def _stem(self, image_id: int) -> str:
        if type(image_id) is not int or not 0 <= image_id < len(self):
            raise ValueError(f"Invalid split-local image ID: {image_id}")
        return self._ids[image_id]

    def get_image(self, image_id: int) -> Tensor:
        return load_rgb(self._root / "JPEGImages" / f"{self._stem(image_id)}.jpg")


class VocEvaluation(VocImages, EvaluationSource):
    """Official val images and masks; keep separate from train-pool access."""

    def __init__(self, root: Path, manifest: PoolManifest):
        super().__init__(root, manifest)
        self._ids = tuple(manifest.val_ids)

    def get_mask(self, image_id: int) -> Tensor:
        stem = self._stem(image_id)
        with Image.open(self._root / "JPEGImages" / f"{stem}.jpg") as image:
            size = image.size
        return load_mask(self._root / "SegmentationClass" / f"{stem}.png", size)


class VocOracle(Oracle):
    """Single-run oracle with atomic batch validation and an append-only audit log.

    Log successful and rejected attempts. Restart/resume and concurrent writers
    are not supported yet; use a fresh log path for each run.
    """

    def __init__(self, root: Path, manifest: PoolManifest, log_path: Path):
        self._root = Path(root)
        self._ids = tuple(manifest.pool_ids)
        self._masks: dict[int, Tensor] = {}
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents overwriting a previous run's query history.
        with self._log_path.open("x"):
            pass

    @property
    def labeled_ids(self) -> list[int]:
        return list(self._masks)

    def _log(self, ids: list[int], status: str, reason: str | None = None) -> None:
        event = {"ids": ids, "status": status, "reason": reason}
        with self._log_path.open("a") as file:
            file.write(json.dumps(event, default=repr) + "\n")

    def query(self, ids: list[int]) -> dict[int, Tensor]:
        try:
            check_ids(ids, "query IDs")
            if any(i >= len(self._ids) or i in self._masks for i in ids):
                raise ValueError("Query contains out-of-pool or already-labeled IDs")
            masks = {}
            for image_id in ids:
                stem = self._ids[image_id]
                with Image.open(self._root / "JPEGImages" / f"{stem}.jpg") as image:
                    size = image.size
                masks[image_id] = load_mask(
                    self._root / "SegmentationClass" / f"{stem}.png", size
                )
        except (ValueError, OSError) as error:
            self._log(ids, "rejected", str(error))
            raise
        self._log(ids, "accepted")
        # Retain compact class indices; callers always receive fresh int64 masks.
        self._masks.update({i: mask.to(torch.uint8) for i, mask in masks.items()})
        return masks

    def get_mask(self, image_id: int) -> Tensor:
        if type(image_id) is not int or image_id not in self._masks:
            raise KeyError(f"Mask has not been queried: {image_id}")
        return self._masks[image_id].to(torch.int64)
