"""Reload Part 3 numerical regressors / neural checkpoints without Hub access."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from soilcv.artifacts import file_hash, package_identity
from soilcv.bags import cache_grid, make_grid_manifest
from soilcv.cnn.predict import predict_bags
from soilcv.curves import TARGET_COLUMNS, repair_curves, validate_curves
from soilcv.predictions import mean_curves

from .data import PREPROCESSING, bag_loader, pool_features
from .evaluation import weighted_knn
from .model import SSLEncoder, SSLRegressor, make_head


def verify_folder(folder):
    folder = Path(folder)
    hashes = json.loads((folder / "hashes.json").read_text())
    for name, digest in hashes.items():
        if file_hash(folder / name) != digest:
            raise ValueError(f"Changed Part 3 artifact: {name}")
    config = json.loads((folder / "config.json").read_text())
    required = {
        "config.json",
        "oof.csv",
        "test_predictions.csv",
        "compute.json",
        "result.json",
    }
    required.add(
        "regressor.npz" if config["strategy"] in {"knn", "ridge"} else "weights.pt"
    )
    if not required <= set(hashes):
        raise ValueError("Incomplete Part 3 artifact manifest.")
    if config["identity"]["soilcv"] != package_identity():
        raise ValueError("Use the source package saved with the Part 3 study.")
    if config["preprocessing"] != PREPROCESSING:
        raise ValueError("Unsupported physical preprocessing.")
    return config


def predict_fold(folder, photos, cache_root, *, device="cpu", amp=False):
    from .experiments import extract

    folder = Path(folder)
    config = verify_folder(folder)
    encoder = SSLEncoder(
        config["encoder"], config=config["model_config"], pooling=config["pooling"]
    ).to(device)
    strategy = config["strategy"]
    tiles = make_grid_manifest(photos, tile_mm=64, output_px=224)
    cache_grid(tiles, cache_root)
    if strategy in {"knn", "ridge", "lp"}:
        encoder_path = folder.parent / "encoder.pt"
        if file_hash(encoder_path) != config["encoder_sha256"]:
            raise ValueError("Source encoder checkpoint changed.")
        encoder.load_state_dict(
            torch.load(encoder_path, map_location=device, weights_only=True)
        )
        ids, features = extract(encoder, tiles, cache_root, device)
        ids, values = pool_features(ids, features, tiles)
        if strategy == "lp":
            head = (
                make_head(encoder.output_dim, config["train_mean_prior"])
                .to(device)
                .eval()
            )
            head.load_state_dict(
                torch.load(
                    folder / "weights.pt", map_location=device, weights_only=True
                )
            )
            with torch.inference_mode():
                predictions = head(torch.as_tensor(values, device=device)).cpu().numpy()
        else:
            with np.load(folder / "regressor.npz", allow_pickle=False) as state:
                if strategy == "ridge":
                    predictions = repair_curves(
                        ((values - state["mean"]) / state["scale"]) @ state["coef"].T
                        + state["intercept"]
                    )
                else:
                    predictions = weighted_knn(state["train"], state["targets"], values)
        result = ids.copy()
        result[list(TARGET_COLUMNS)] = predictions
    else:
        model = SSLRegressor(
            encoder, make_head(encoder.output_dim, config["train_mean_prior"])
        ).to(device)
        model.load_state_dict(
            torch.load(folder / "weights.pt", map_location=device, weights_only=True)
        )
        result, _ = predict_bags(
            model, bag_loader(tiles, cache_root), device=device, amp=amp
        )
    validate_curves(result[list(TARGET_COLUMNS)])
    return result


def predict_ensemble(folders, photos, cache_root, *, device="cpu", amp=False):
    configs = [verify_folder(folder) for folder in folders]
    if len(configs) != 4 or {row["fold"] for row in configs} != set(range(4)):
        raise ValueError("A complete SSL ensemble needs folds 0, 1, 2, 3 exactly once.")
    keys = ["study_id", "experiment", "encoder", "source", "strategy", "pooling"]
    if any(any(row[key] != configs[0][key] for key in keys) for row in configs[1:]):
        raise ValueError("Do not mix SSL studies, encoders or tuning strategies.")
    return mean_curves(
        [
            predict_fold(folder, photos, cache_root, device=device, amp=amp)
            for folder in folders
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument(
        "--photos",
        required=True,
        type=Path,
        help="CSV: sample_id, photo, path, ppm_est, device",
    )
    parser.add_argument("--cache", type=Path, default=Path("/tmp/soilcv-ssl-inference"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    photos = pd.read_csv(args.photos)
    operation = predict_fold if len(args.folders) == 1 else predict_ensemble
    selected = args.folders[0] if len(args.folders) == 1 else args.folders
    predictions = operation(selected, photos, args.cache, device=args.device)
    predictions.to_csv(args.output, index=False)
