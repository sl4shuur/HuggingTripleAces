"""Reload a full-photo Transformer and predict curves from photos and scale."""

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from soilcv.artifacts import package_identity
from soilcv.bags import one_bag
from soilcv.cnn.predict import predict_bags
from soilcv.curves import TARGET_COLUMNS, validate_curves
from soilcv.inference.runner import ensemble_predictions, validate_ensemble

from .images import (
    NORMALIZE,
    FullPhotoBags,
    cache_photos,
    make_photo_manifest,
    photo_transform,
)
from .model import FullFrameModel


def load_model(folder, device="cpu"):
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    if config["stage"] != "vit_fullframe":
        raise ValueError("Use a full-photo Transformer checkpoint.")
    if config["versions"]["soilcv"] != package_identity():
        raise ValueError("Use the code dataset version saved with this checkpoint.")
    recipe = config["params"]
    if (
        recipe["normalization"] != NORMALIZE
        or recipe["image_schema"] != "full_photo_ppm_v1"
    ):
        raise ValueError("Unsupported photo preprocessing.")
    if recipe["pooling"] != "equal_photo_features":
        raise ValueError("Unsupported photo pooling.")
    model = FullFrameModel(recipe, recipe["train_mean_prior"], pretrained=False).to(
        device
    )
    model.load_state_dict(
        torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    )
    return model, recipe


def predict_from_photos(model, recipe, photos, cache_root, device="cpu", amp=True):
    manifest = make_photo_manifest(photos, target_ppm=recipe["target_ppm"])
    cache_photos(manifest, cache_root)
    dataset = FullPhotoBags(manifest, cache_root, photo_transform(recipe))
    loader = DataLoader(dataset, batch_size=1, collate_fn=one_bag, num_workers=0)
    predictions, _ = predict_bags(model, loader, device=device, amp=amp)
    return predictions


def predict_sample(
    photos, checkpoint_dirs, *, sample_id="query", device="cpu", amp=True
):
    if not photos or not checkpoint_dirs:
        raise ValueError("Provide photos with PPM and at least one saved model folder.")
    if len(checkpoint_dirs) not in {1, 4}:
        raise ValueError("Use one checkpoint or one complete four-fold ensemble.")
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
    predictions = []
    with tempfile.TemporaryDirectory(prefix="soil-full-photo-") as temporary:
        for folder in checkpoint_dirs:
            model, recipe = load_model(folder, device)
            predictions.append(
                predict_from_photos(model, recipe, rows, temporary, device, amp)
            )
            del model
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
