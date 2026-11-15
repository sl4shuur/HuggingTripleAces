"""Image-only batched extraction, preserving permanent pool IDs."""

from collections.abc import Iterator

from alseg.core.features import EncoderFeatures
from alseg.core.helpers import check_ids
from alseg.interfaces.data import ImageSource
from alseg.interfaces.encoder import Encoder


def iter_features(
    source: ImageSource,
    encoder: Encoder,
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
        features = encoder.encode([source.get_image(i) for i in batch_ids])
        if features.embeddings.shape[0] != len(batch_ids):
            raise ValueError("Encoder output batch does not match requested IDs")
        yield batch_ids, features
