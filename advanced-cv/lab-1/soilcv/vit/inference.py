"""Reload the saved local/global inputs and selected epoch weights of a fold."""

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from soilcv.artifacts import package_identity
from soilcv.bags import cache_grid, make_grid_manifest, one_bag
from soilcv.checkpoints import candidate_paths, load_candidates
from soilcv.cnn.predict import predict_bags
from soilcv.curves import TARGET_COLUMNS, validate_curves
from soilcv.inference.runner import ensemble_predictions, validate_ensemble
from soilcv.predictions import mean_curves

from .images import (
    NORMALIZE,
    FullPhotoBags,
    cache_photos,
    make_photo_manifest,
    photo_transform,
)
from .model import FullFrameModel


def load_model(folder, device="cpu", *, weights_path=None):
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    if config["stage"] != "vit_fullframe":
        raise ValueError("Use a Transformer checkpoint.")
    if config["versions"]["soilcv"] != package_identity():
        raise ValueError("Use the source package saved with this checkpoint.")
    recipe = config["params"]
    if (
        recipe["normalization"] != NORMALIZE
        or recipe["image_schema"] != "full_photo_ppm_v2"
    ):
        raise ValueError("Unsupported image preprocessing.")
    pooling = {
        "multiscale": "local_global_equal_photo_features",
        "tiles": "overlap_area_then_equal_photos",
    }
    if (
        recipe["view_mode"] not in pooling
        or recipe["pooling"] != pooling[recipe["view_mode"]]
    ):
        raise ValueError("Unsupported photo pooling.")
    if weights_path is None:
        if (folder / "candidates.json").is_file():
            weights_path = (
                folder / load_candidates(folder, selected_only=False)[0]["weights"]
            )
        else:
            weights_path = folder / "weights.pt"
    model = FullFrameModel(recipe, recipe["train_mean_prior"], pretrained=False).to(
        device
    )
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    return model, recipe


def photo_loader(recipe, photos, cache_root):
    manifest = make_photo_manifest(photos, target_ppm=recipe["target_ppm"])
    tiles = make_grid_manifest(
        photos, tile_mm=recipe["tile_mm"], output_px=recipe["pixels"]
    )
    if recipe["view_mode"] == "multiscale":
        cache_photos(manifest, cache_root)
    cache_grid(tiles, cache_root)
    dataset = FullPhotoBags(
        manifest,
        cache_root,
        photo_transform(recipe),
        tiles=tiles,
        view_mode=recipe["view_mode"],
    )
    return DataLoader(dataset, batch_size=1, collate_fn=one_bag, num_workers=0)


def predict_from_photos(model, recipe, photos, cache_root, device="cpu", amp=True):
    loader = photo_loader(recipe, photos, cache_root)
    predictions, _ = predict_bags(model, loader, device=device, amp=amp)
    return predictions


def predict_fold(folder, photos, cache_root, device="cpu", amp=True):
    """Apply the saved top-two/optional-third epoch rule before four-fold averaging."""
    folder = Path(folder)
    paths = candidate_paths(folder)
    model, recipe = load_model(folder, device, weights_path=paths[0])
    loader = photo_loader(recipe, photos, cache_root)
    predictions = []
    for index, path in enumerate(paths):
        if index:
            model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
        curves, _ = predict_bags(model, loader, device=device, amp=amp)
        predictions.append(curves)
    result = mean_curves(predictions)
    del model, loader
    return result


def predict_sample(
    photos, checkpoint_dirs, *, sample_id="query", device="cpu", amp=True
):
    if not photos or len(checkpoint_dirs) not in {1, 4}:
        raise ValueError(
            "Provide photos and one fold or one complete four-fold ensemble."
        )
    if len(checkpoint_dirs) == 4:
        validate_ensemble(
            [
                json.loads((Path(folder) / "config.json").read_text())
                for folder in checkpoint_dirs
            ]
        )
    rows = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "photo": Path(photo["path"]).name,
                "path": str(Path(photo["path"]).resolve()),
                "ppm_est": float(photo["ppm_est"]),
                "device": photo.get("device", "unknown"),
            }
            for photo in photos
        ]
    )
    with tempfile.TemporaryDirectory(prefix="soil-multiscale-") as temporary:
        predictions = [
            predict_fold(folder, rows, temporary, device, amp)
            for folder in checkpoint_dirs
        ]
    result = (
        ensemble_predictions(predictions, predictions[0])
        if len(predictions) == 4
        else predictions[0]
    )
    validate_curves(result[list(TARGET_COLUMNS)])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo", nargs="+", required=True, type=Path)
    parser.add_argument("--ppm", nargs="+", required=True, type=float)
    parser.add_argument("--checkpoint-dir", nargs="+", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-id", default="query")
    parser.add_argument("--output", type=Path, default=Path("prediction.csv"))
    args = parser.parse_args()
    if len(args.photo) != len(args.ppm):
        parser.error("Supply one PPM value for each photo.")
    supplied = [
        {"path": path, "ppm_est": ppm} for path, ppm in zip(args.photo, args.ppm)
    ]
    result = predict_sample(
        supplied, args.checkpoint_dir, sample_id=args.sample_id, device=args.device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
