"""A candidate-only model artifact must reproduce the selected epoch ensemble."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from soilcv.artifacts import package_identity, write_json
from soilcv.checkpoints import candidate_paths, load_candidates, update_candidates
from soilcv.curves import TARGET_COLUMNS
from soilcv.vit.images import NORMALIZE
from soilcv.vit.inference import load_model, predict_fold, predict_from_photos
from soilcv.vit.model import FullFrameModel


def photos(tmp_path):
    rows = []
    for sample, shape, camera in [
        ("soil-b", (256, 272), "Motorola"),
        ("soil-a", (272, 256), "Samsung"),
    ]:
        height, width = shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        pixels = np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8)
        path = tmp_path / f"{sample}.png"
        Image.fromarray(pixels).save(path)
        rows.append(
            {
                "sample_id": sample,
                "photo": path.name,
                "path": str(path),
                "ppm_est": 4.0,
                "device": camera,
            }
        )
    return pd.DataFrame(rows)


def candidate_artifact(tmp_path, view_mode, third_camera):
    torch.manual_seed(7)
    supplied = photos(tmp_path)
    prior = np.linspace(5, 100, 11)
    recipe = {
        "arch": "scratch_vit",
        "view_mode": view_mode,
        "pooling": "local_global_equal_photo_features"
        if view_mode == "multiscale"
        else "overlap_area_then_equal_photos",
        "target_ppm": 2.0,
        "tile_mm": 64,
        "pixels": 16,
        "chunk_size": 2,
        "patch_size": 16,
        "dim": 24,
        "depth": 1,
        "heads": 3,
        "dropout": 0.0,
        "drop_path": 0.0,
        "max_tokens": 128,
        "augmentation": False,
        "normalization": NORMALIZE,
        "image_schema": "full_photo_ppm_v2",
        "train_mean_prior": prior.tolist(),
    }
    model = FullFrameModel(recipe, prior, pretrained=False)
    original_bias = model.head.projection.bias.detach().clone()
    folder = tmp_path / "artifact"
    cache = tmp_path / "cache"
    records = []
    predictions = {}
    for epoch, camera in enumerate([1.0, 5.0, third_camera]):
        with torch.no_grad():
            model.head.projection.bias.copy_(
                original_bias + epoch * torch.linspace(-0.8, 0.8, 11)
            )
        predictions[epoch] = predict_from_photos(
            model, recipe, supplied, cache, device="cpu", amp=False
        )
        records = update_candidates(
            folder,
            records,
            model,
            epoch,
            {"emd": 10.0 + epoch, "camera_disagreement": camera},
            predictions[epoch],
        )
    write_json(
        folder / "config.json",
        {
            "stage": "vit_fullframe",
            "versions": {"soilcv": package_identity()},
            "params": recipe,
        },
    )
    assert not (folder / "weights.pt").exists()
    assert not (folder / "best.pt").exists()
    return folder, supplied, cache, predictions


@pytest.mark.parametrize("view_mode", ["multiscale", "tiles"])
@pytest.mark.parametrize("third_camera, expected_count", [(2.0, 3), (8.0, 2)])
def test_candidate_only_fold_reload_matches_selected_epoch_mean(
    tmp_path, view_mode, third_camera, expected_count
):
    folder, supplied, cache, saved = candidate_artifact(
        tmp_path, view_mode, third_camera
    )
    selected = load_candidates(folder)
    assert len(selected) == expected_count
    assert [record["epoch"] for record in selected] == list(range(expected_count))
    model, recipe = load_model(folder)
    top_one = predict_from_photos(
        model, recipe, supplied, cache, device="cpu", amp=False
    )
    np.testing.assert_allclose(
        top_one[list(TARGET_COLUMNS)], saved[0][list(TARGET_COLUMNS)], atol=1e-6
    )
    actual = predict_fold(folder, supplied, cache, device="cpu", amp=False)
    expected = np.mean(
        [saved[row["epoch"]][list(TARGET_COLUMNS)].to_numpy() for row in selected],
        axis=0,
    )
    assert actual.sample_id.tolist() == ["soil-a", "soil-b"]
    np.testing.assert_allclose(actual[list(TARGET_COLUMNS)], expected, atol=1e-6)
    assert not np.allclose(actual[list(TARGET_COLUMNS)], saved[0][list(TARGET_COLUMNS)])
    assert len(candidate_paths(folder)) == expected_count


def test_corrupt_even_unselected_candidate_is_rejected(tmp_path):
    folder, supplied, cache, _ = candidate_artifact(tmp_path, "multiscale", 8.0)
    manifest = json.loads((folder / "candidates.json").read_text())
    last = folder / manifest["candidates"][-1]["weights"]
    assert last not in candidate_paths(folder)
    last.write_bytes(last.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed candidate"):
        load_model(folder)
    with pytest.raises(ValueError, match="changed candidate"):
        predict_fold(folder, supplied, cache, device="cpu", amp=False)
