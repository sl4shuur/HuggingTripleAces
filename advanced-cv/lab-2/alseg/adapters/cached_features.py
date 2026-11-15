"""Read immutable train/val features without loading a backbone or hidden masks."""

import json
from pathlib import Path

import numpy as np
import torch

from alseg.core.features import EncoderFeatures
from alseg.core.helpers import check_ids
from alseg.core.pool import PoolManifest
from alseg.interfaces.data import ImageSource
from alseg.interfaces.encoder import FeatureSource


class CachedFeatures(FeatureSource):
    def __init__(self, directory: Path, expected_identity: dict):
        self.directory = Path(directory)
        metadata = json.loads((self.directory / "metadata.json").read_text())
        if metadata.get("format_version") != 1 or not metadata.get("complete"):
            raise ValueError("Unsupported or incomplete feature cache")
        self.identity = metadata["encoder_identity"]
        for key, value in expected_identity.items():
            if self.identity.get(key) != value:
                raise ValueError(f"Feature cache encoder mismatch: {key}")
        manifest_data = json.loads(
            (self.directory.parent / "manifest.json").read_text()
        )
        manifest = PoolManifest(manifest_data["pool_ids"], manifest_data["val_ids"])
        if (
            manifest.sha256 != metadata["manifest_sha256"]
            or manifest.sha256 != manifest_data["sha256"]
        ):
            raise ValueError("Feature cache manifest checksum mismatch")
        self._splits = {}
        dim = self.identity["feature_dim"]
        grid = self.identity["image_size"] // self.identity["patch_size"]
        for split, stems in (("train", manifest.pool_ids), ("val", manifest.val_ids)):
            embeddings = np.load(
                self.directory / split / "embeddings.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            patches = np.load(
                self.directory / split / "patches.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            if (
                embeddings.dtype != np.float32
                or patches.dtype != np.float32
                or embeddings.shape != (len(stems), dim)
                or patches.shape != (len(stems), dim, grid, grid)
            ):
                raise ValueError(f"Incomplete or incompatible {split} feature arrays")
            self._splits[tuple(stems)] = (embeddings, patches)

    def get_features(self, source: ImageSource, ids: list[int]) -> EncoderFeatures:
        check_ids(ids, "cached feature IDs")
        if source.image_ids not in self._splits:
            raise ValueError(
                "Image source order does not match the cached train/val manifest"
            )
        embeddings, patches = self._splits[source.image_ids]
        if not ids or any(i >= len(embeddings) for i in ids):
            raise ValueError("Cached feature IDs must be nonempty and within the split")
        # NumPy advanced indexing copies only these rows into writable CPU memory.
        return EncoderFeatures(
            torch.from_numpy(embeddings[ids]), torch.from_numpy(patches[ids])
        )

    def describe(self) -> dict:
        return dict(self.identity)
