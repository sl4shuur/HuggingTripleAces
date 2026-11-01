"""Keep each complete photo, resize once to a shared estimated pixels/mm."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torchvision import transforms as T

from soilcv.artifacts import file_hash, identity_hash, write_json
from soilcv.curves import TARGET_COLUMNS

NORMALIZE = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def make_photo_manifest(photos, *, target_ppm=2.0, image_root=None):
    """Use one scale for all photos. Rounding can change x/y PPM by <1 pixel."""
    if not np.isfinite(target_ppm) or target_ppm <= 0:
        raise ValueError("target_ppm must be positive.")
    if photos.empty or photos.duplicated(["sample_id", "photo"]).any():
        raise ValueError("Use unique, non-empty photo rows.")
    rows = []
    for photo in photos.sort_values(["sample_id", "photo"]).itertuples(index=False):
        path = Path(photo.path)
        native_ppm = float(photo.ppm_est)
        if not np.isfinite(native_ppm) or native_ppm <= 0:
            raise ValueError("Native PPM must be positive.")
        if target_ppm > native_ppm:
            raise ValueError("Choose a common PPM below every photo's native PPM.")
        with Image.open(path) as raw:
            width, height = ImageOps.exif_transpose(raw).size
        output_width = max(1, round(width * target_ppm / native_ppm))
        output_height = max(1, round(height * target_ppm / native_ppm))
        rows.append(
            {
                "sample_id": str(photo.sample_id),
                "photo": str(photo.photo),
                "device": str(getattr(photo, "device", "unknown")),
                "image_path": str(path.relative_to(image_root))
                if image_root
                else str(path.resolve()),
                "source_hash": file_hash(path),
                "width": width,
                "height": height,
                "output_width": output_width,
                "output_height": output_height,
                "native_ppm": native_ppm,
                "target_ppm": float(target_ppm),
                "actual_ppm_x": output_width * native_ppm / width,
                "actual_ppm_y": output_height * native_ppm / height,
                "coverage": 1.0,
                "missing_native_pixels": 0,
                "interpolation": "LANCZOS",
                "schema": "full_photo_ppm_v1",
            }
        )
    manifest = pd.DataFrame(rows)
    cache_id = identity_hash(manifest.to_dict("records"))
    manifest["cache_id"] = cache_id
    manifest["frame_path"] = [
        f"{cache_id}/{index:04d}.png" for index in range(len(manifest))
    ]
    return manifest


def cache_photos(manifest, root, *, image_root=None):
    """Cache complete rectangles. No crop or square resize is used."""
    root = Path(root)
    cache = root / manifest.cache_id.iloc[0]
    cache.mkdir(parents=True, exist_ok=True)
    saved = cache / "checksums.json"
    previous = json.loads(saved.read_text()) if saved.exists() else {}
    checksums = {}
    for row in manifest.itertuples(index=False):
        source = (
            Path(image_root) / row.image_path if image_root else Path(row.image_path)
        )
        if file_hash(source) != row.source_hash:
            raise ValueError("Source image changed.")
        destination = root / row.frame_path
        if destination.exists() and row.frame_path in previous:
            if file_hash(destination) != previous[row.frame_path]:
                raise ValueError("Cached photo checksum changed.")
        else:
            with Image.open(source) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                if image.size != (row.width, row.height):
                    raise ValueError("Photo geometry changed.")
                image.resize(
                    (row.output_width, row.output_height), Image.Resampling.LANCZOS
                ).save(destination)
        checksums[row.frame_path] = file_hash(destination)
    write_json(saved, checksums)
    return checksums


class QuarterTurn:
    def __call__(self, image):
        # expand=True keeps all pixels of a rectangular photo.
        return image.rotate(random.choice([0, 90, 180, 270]), expand=True)


def photo_transform(recipe, train=False):
    steps = []
    if train and recipe.get("augmentation", True):
        steps = [
            QuarterTurn(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        ]
    return T.Compose([*steps, T.ToTensor(), T.Normalize(**NORMALIZE)])


class FullPhotoBags:
    """One item = all complete photos of one soil; one loss per soil."""

    def __init__(self, manifest, cache_root, transform, labels=None):
        self.manifest = manifest.reset_index(drop=True).copy()
        self.root = Path(cache_root)
        self.transform = transform
        self.samples = sorted(self.manifest.sample_id.unique())
        self.targets = None
        if labels is not None:
            self.targets = labels.set_index("sample_id")[list(TARGET_COLUMNS)]
            if not set(self.samples) <= set(self.targets.index):
                raise ValueError("Some sample bags have no label.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch

        sample = self.samples[index]
        rows = self.manifest[self.manifest.sample_id.eq(sample)]
        images = []
        for row in rows.itertuples(index=False):
            with Image.open(self.root / row.frame_path) as image:
                images.append(self.transform(image.convert("RGB")))
        result = {
            "image": images,
            "bag_weights": torch.full((len(images),), 1 / len(images)),
            "sample_id": sample,
            "photo": rows.photo.tolist(),
            "device": rows.device.tolist(),
        }
        if self.targets is not None:
            result["target"] = torch.tensor(
                self.targets.loc[sample].to_numpy(dtype=np.float32)
            )[None]
        return result
