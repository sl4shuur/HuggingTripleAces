"""Filesystem integration for official VOC segmentation splits."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from alseg.core.pool import PoolManifest
from alseg.core.predictions import IGNORE_INDEX, NUM_CLASSES


def load_manifest(root: Path) -> PoolManifest:
    """Keep official split-file order; integer IDs remain permanent row indices."""
    splits = validate_voc_root(root)
    return PoolManifest(pool_ids=splits["train"], val_ids=splits["val"])


def load_rgb(path: Path) -> Tensor:
    """Decode raw RGB without resizing or model-specific normalization."""
    with Image.open(path) as image:
        pixels = np.array(image.convert("RGB"), copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def load_mask(path: Path, image_size: tuple[int, int]) -> Tensor:
    """Read PNG class indices, not their palette colors; preserve VOC void=255."""
    with Image.open(path) as mask:
        if mask.mode not in ("P", "L"):
            raise ValueError(f"Expected an indexed or grayscale mask: {path}")
        if mask.size != image_size:
            raise ValueError(f"Image/mask size mismatch: {path}")
        pixels = torch.from_numpy(np.array(mask, dtype=np.int64, copy=True))
    valid = ((pixels >= 0) & (pixels < NUM_CLASSES)) | (pixels == IGNORE_INDEX)
    if not valid.all():
        raise ValueError(f"Mask has class IDs outside 0..20 and 255: {path}")
    return pixels


def inspect_voc_samples(root: Path, manifest: PoolManifest) -> dict:
    """Decode the first train/val pairs for setup diagnostics, not acquisition."""
    report = {}
    for split, ids in (("train", manifest.pool_ids), ("val", manifest.val_ids)):
        stem = ids[0]
        image = load_rgb(root / "JPEGImages" / f"{stem}.jpg")
        mask = load_mask(
            root / "SegmentationClass" / f"{stem}.png",
            (image.shape[2], image.shape[1]),
        )
        report[split] = {
            "stem": stem,
            "image_shape": list(image.shape),
            "mask_shape": list(mask.shape),
            "class_ids": mask.unique().tolist(),
            "void_pixels": int((mask == IGNORE_INDEX).sum()),
        }
    return report


def validate_voc_root(root: Path) -> dict[str, list[str]]:
    """Require all official train/val images and semantic masks; never use test."""
    splits: dict[str, list[str]] = {}
    for split in ("train", "val"):
        split_file = root / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_file.is_file():
            raise ValueError(f"Missing official segmentation split: {split_file}")
        ids = split_file.read_text().split()
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicate IDs in {split_file}")
        for image_id in ids:
            if Path(image_id).name != image_id or image_id in (".", ".."):
                raise ValueError(f"Invalid VOC filename stem: {image_id}")
            for folder, extension in (
                ("JPEGImages", ".jpg"),
                ("SegmentationClass", ".png"),
            ):
                file = root / folder / f"{image_id}{extension}"
                if not file.is_file():
                    raise ValueError(f"Missing {split} file: {file}")
        splits[split] = ids
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("Official segmentation train and val IDs overlap")
    return splits


def find_voc_root(dataset_root: Path) -> Path:
    """Accept only one complete train/val root; ambiguous mounts need an override."""
    matches = []
    for split_file in sorted(dataset_root.rglob("ImageSets/Segmentation/train.txt")):
        candidate = split_file.parents[2]
        try:
            validate_voc_root(candidate)
        except ValueError:
            continue
        matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(
            "Expected one complete VOC segmentation train/val root under "
            f"{dataset_root}; "
            f"found {len(matches)}. Set data.voc_root explicitly to a root containing "
            "ImageSets/Segmentation/{train,val}.txt, JPEGImages and SegmentationClass. "
            "VOC2012_test alone is insufficient."
        )
    return matches[0]
