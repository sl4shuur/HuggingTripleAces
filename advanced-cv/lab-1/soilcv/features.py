"""Frozen feature extraction and exact cache identities."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import identity_hash, model_state_hash, write_json

_REQUIRED_IDENTITY = {
    "encoder",
    "weights_revision",
    "encoder_state_hash",
    "preprocessing_hash",
    "tile_cache_id",
    "pooling",
}


def feature_identity(
    encoder,
    *,
    name,
    weights_revision,
    preprocessing,
    tile_cache_id,
    pooling,
    fold=None,
    train_ids=None,
):
    identity = {
        "encoder": name,
        "weights_revision": weights_revision,
        "encoder_state_hash": model_state_hash(encoder),
        "preprocessing_hash": identity_hash(preprocessing),
        "tile_cache_id": tile_cache_id,
        "pooling": pooling,
        "fold": fold,
        "train_ids_hash": identity_hash(sorted(train_ids))
        if train_ids is not None
        else None,
    }
    _check_identity(identity)
    return identity


def _check_identity(identity):
    if any(not identity.get(key) for key in _REQUIRED_IDENTITY):
        raise ValueError(
            "Feature identity needs encoder, weights revision, actual state, preprocessing, tile cache, and pooling."
        )
    if identity.get("fold") is not None and not identity.get("train_ids_hash"):
        raise ValueError(
            "A fold-trained encoder needs its train sample IDs in the identity."
        )


def extract_features(encoder, loader, *, device="cpu"):
    """The encoder must return [batch, features]. Pooling is the caller's choice."""
    import torch

    from .nn import evaluation_mode
    from .training import batch_ids

    encoder.to(device)
    ids, features = [], []
    with evaluation_mode(encoder), torch.inference_mode():
        for batch in loader:
            output = encoder(batch["image"].to(device))
            if (
                not isinstance(output, torch.Tensor)
                or output.ndim != 2
                or len(output) != len(batch["image"])
            ):
                raise ValueError(
                    "Encoder must return [batch, features]; define pooling in its adapter."
                )
            ids.extend(batch_ids(batch))
            features.append(output.float().cpu().numpy())
    if not features:
        raise ValueError("Feature loader is empty.")
    frame = pd.DataFrame(ids)
    if frame.duplicated(["sample_id", "photo", "tile_id"]).any():
        raise ValueError("Extract features once per tile; use a non-repeating loader.")
    return frame, np.concatenate(features)


def save_features(path, ids, features, identity):
    _check_identity(identity)
    features = np.asarray(features)
    if (
        features.ndim != 2
        or len(features) != len(ids)
        or not len(ids)
        or not np.isfinite(features).all()
    ):
        raise ValueError(
            "Features must be finite [rows, features] values aligned to non-empty IDs."
        )
    required = ["sample_id", "photo", "tile_id"]
    if ids[required].isna().any().any() or ids.duplicated(required).any():
        raise ValueError("Feature rows need unique, non-missing tile IDs.")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    cache_id = identity_hash(identity)
    existing = path / "metadata.json"
    if existing.exists() and json.loads(existing.read_text())["cache_id"] != cache_id:
        raise ValueError("Use a new feature cache folder for a new identity.")
    np.save(path / "features.npy", features, allow_pickle=False)
    write_json(path / "ids.json", ids.to_dict("records"))
    from .artifacts import file_hash

    write_json(
        existing,
        {
            "identity": identity,
            "cache_id": cache_id,
            "features_hash": file_hash(path / "features.npy"),
            "ids_hash": file_hash(path / "ids.json"),
        },
    )


def load_features(path, expected_identity):
    from .artifacts import file_hash

    _check_identity(expected_identity)
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if (
        metadata["cache_id"] != identity_hash(expected_identity)
        or identity_hash(metadata["identity"]) != metadata["cache_id"]
    ):
        raise ValueError(
            "Feature cache identity differs from the requested encoder/input."
        )
    if (
        file_hash(path / "features.npy") != metadata["features_hash"]
        or file_hash(path / "ids.json") != metadata["ids_hash"]
    ):
        raise ValueError("Feature cache files changed.")
    ids = pd.DataFrame(json.loads((path / "ids.json").read_text()))
    features = np.load(path / "features.npy", allow_pickle=False)
    if len(ids) != len(features):
        raise ValueError("Feature IDs and array have different lengths.")
    return ids, features
