"""Image-only batched extraction, preserving permanent pool IDs."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from alseg.core.features import EncoderFeatures
from alseg.core.helpers import check_ids
from alseg.interfaces.data import ImageSource
from alseg.interfaces.encoder import FeatureSource


def iter_features(
    source: ImageSource,
    encoder: FeatureSource,
    ids: list[int],
    batch_size: int,
) -> Iterator[tuple[list[int], EncoderFeatures]]:
    """Stream patch maps without retaining all pool logits/features in memory."""
    check_ids(ids, "feature IDs")
    if any(i >= len(source) for i in ids):
        raise ValueError("Feature IDs must belong to the image source")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        features = encoder.get_features(source, batch_ids)
        if features.embeddings.shape[0] != len(batch_ids):
            raise ValueError("Encoder output batch does not match requested IDs")
        yield batch_ids, features


def write_split_features(
    source: ImageSource,
    encoder: FeatureSource,
    output_dir: Path,
    batch_size: int,
    limit: int | None = None,
) -> dict:
    """Write float32 arrays incrementally; retain only the current batch in RAM."""
    count = len(source) if limit is None else min(limit, len(source))
    if count < 1:
        raise ValueError("Feature extraction requires a nonempty split")
    output_dir.mkdir()
    embeddings = patches = None
    for ids, features in iter_features(source, encoder, list(range(count)), batch_size):
        if embeddings is None:
            embeddings = np.lib.format.open_memmap(
                output_dir / "embeddings.npy",
                mode="w+",
                dtype=np.float32,
                shape=(count, features.embeddings.shape[1]),
            )
            patches = np.lib.format.open_memmap(
                output_dir / "patches.npy",
                mode="w+",
                dtype=np.float32,
                shape=(count, *features.patch_features.shape[1:]),
            )
        embeddings[ids] = features.embeddings.numpy()
        patches[ids] = features.patch_features.numpy()
    embeddings.flush()
    patches.flush()
    return {
        "count": count,
        "complete": count == len(source),
        "embeddings_shape": list(embeddings.shape),
        "patches_shape": list(patches.shape),
    }
