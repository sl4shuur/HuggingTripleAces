"""Physical tiles shared by frozen probes, supervised tuning and SSL adaptation."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from soilcv.bags import SampleBags, one_bag
from soilcv.camera import camera_family
from soilcv.curves import TARGET_COLUMNS
from soilcv.data import TileDataset

NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
PREPROCESSING = {
    "tile_mm": 64,
    "pixels": 224,
    "interpolation": "LANCZOS",
    "normalization": NORMALIZE,
    "pooling": "overlap_area_then_equal_photos",
    "random_resized_crop": False,
}


class TileTransform:
    def __init__(self, *, train=False, photometric=True):
        self.train, self.photometric = train, photometric
        self.jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)

    def __call__(self, image):
        if self.train:
            # Exact quarter-turns avoid interpolation and preserve physical scale.
            image = image.rotate(90 * random.randrange(4))
            if random.random() < 0.5:
                image = TF.hflip(image)
            if self.photometric:
                image = self.jitter(image)
        return TF.normalize(TF.to_tensor(image), **NORMALIZE)


def bag_loader(manifest, cache, labels=None, *, train=False, photometric=True, seed=42):
    return DataLoader(
        SampleBags(
            manifest, cache, TileTransform(train=train, photometric=photometric), labels
        ),
        batch_size=1,
        shuffle=train,
        num_workers=0,
        collate_fn=one_bag,
        generator=torch.Generator().manual_seed(seed),
    )


def tile_loader(manifest, cache, *, batch_size=32):
    return DataLoader(
        # Feature extraction may batch labelled and test tiles together.
        # TileDataset omits missing fold values, which breaks default collation.
        TileDataset(
            manifest.drop(columns="fold", errors="ignore"),
            cache,
            transform=TileTransform(),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def pool_features(ids, features, manifest):
    keys = ["sample_id", "photo", "tile_id"]
    frame = ids[keys].copy()
    weights = manifest[keys + ["photo_weight"]].copy()
    for key in keys:
        frame[key], weights[key] = frame[key].astype(str), weights[key].astype(str)
    frame = frame.merge(weights, on=keys, validate="one_to_one", how="left", sort=False)
    if frame.photo_weight.isna().any() or len(frame) != len(features):
        raise ValueError("Features must match the physical grid exactly.")
    sample_ids, values = [], []
    for sample, rows in frame.groupby("sample_id", sort=True):
        photo_features = []
        for _, photo in rows.groupby("photo", sort=True):
            photo_features.append(
                np.average(features[photo.index], axis=0, weights=photo.photo_weight)
            )
        sample_ids.append(sample)
        values.append(np.mean(photo_features, axis=0))
    return pd.DataFrame({"sample_id": sample_ids}), np.asarray(values, dtype=np.float32)


def camera_features(ids, features, manifest, sample_ids):
    """Reuse frozen tile descriptors for the same paired-camera diagnostics."""
    keys = ["sample_id", "photo", "tile_id"]
    rows, geometry = ids[keys].astype(str).copy(), manifest.copy()
    geometry[keys] = geometry[keys].astype(str)
    rows["position"] = np.arange(len(rows))
    rows = rows.merge(geometry[keys + ["device"]], on=keys, validate="one_to_one")
    rows = rows[rows.sample_id.isin(sample_ids)].copy()
    rows["device"] = rows.device.map(lambda name: camera_family(name) or name)
    groups, values = [], []
    for device, group in rows.groupby("device", sort=True):
        positions = group.position.to_numpy()
        samples, pooled = pool_features(
            ids.iloc[positions].reset_index(drop=True), features[positions], geometry
        )
        samples["device"] = device
        groups.append(samples)
        values.append(pooled)
    return pd.concat(groups, ignore_index=True), np.concatenate(values)


class FeatureBags:
    """One deterministic cached descriptor and one loss per physical sample."""

    def __init__(self, ids, features, folds, labels):
        self.manifest = ids.merge(folds, on="sample_id", validate="one_to_one")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(
            labels.set_index("sample_id")
            .loc[ids.sample_id, list(TARGET_COLUMNS)]
            .to_numpy(np.float32)
        )

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        return {
            "image": self.features[index],
            "target": self.targets[index],
            "sample_id": self.manifest.iloc[index].sample_id,
        }


class TwoViews:
    def __init__(self, manifest, cache):
        self.manifest = manifest.reset_index(drop=True)
        self.cache = Path(cache)
        self.transform = TileTransform(train=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        with Image.open(self.cache / self.manifest.iloc[index].tile_path) as image:
            image = image.convert("RGB")
            return self.transform(image), self.transform(image)


def adaptation_loader(manifest, cache, *, fold, seed=42, batch_size=32):
    # Enforce the split here as well as in the study runner.
    if manifest.empty or manifest.fold.isna().any() or manifest.fold.eq(fold).any():
        raise ValueError(
            "SSL may use only this fold's training photos; no validation/test."
        )
    photos = manifest.groupby("sample_id").photo.transform("nunique")
    weights = manifest.photo_weight.to_numpy() / photos.to_numpy()
    sampler = WeightedRandomSampler(
        weights,
        num_samples=max(batch_size, len(manifest)),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        TwoViews(manifest, cache), batch_size=batch_size, sampler=sampler, num_workers=0
    )
