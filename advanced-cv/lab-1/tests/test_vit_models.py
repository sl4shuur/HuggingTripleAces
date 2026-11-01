"""Full-photo geometry and small, offline Transformer contracts."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from soilcv.curves import TARGET_COLUMNS, validate_curves
from soilcv.vit.images import (
    FullPhotoBags,
    cache_photos,
    make_photo_manifest,
    photo_transform,
)
from soilcv.vit.model import (
    FullFrameModel,
    ScratchViT,
    SwinPhotoEncoder,
    pad_to_multiple,
)


def test_full_photo_resize_keeps_rectangle_and_scale(tmp_path):
    source = tmp_path / "source.png"
    # Four different borders make a lost edge visible after cache roundtrip.
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[0] = [255, 0, 0]
    image[-1] = [0, 255, 0]
    image[:, 0] = [0, 0, 255]
    image[:, -1] = [255, 255, 255]
    Image.fromarray(image).save(source)
    photos = pd.DataFrame(
        [
            {
                "sample_id": "A001",
                "photo": "source.png",
                "path": str(source),
                "ppm_est": 2.0,
                "device": "Motorola",
            }
        ]
    )
    manifest = make_photo_manifest(photos, target_ppm=2)
    row = manifest.iloc[0]
    assert (row.output_width, row.output_height) == (30, 20)
    assert row.coverage == 1
    assert row.missing_native_pixels == 0
    assert row.actual_ppm_x == row.actual_ppm_y == 2
    cache_photos(manifest, tmp_path / "cache")
    with Image.open(tmp_path / "cache" / row.frame_path) as restored:
        np.testing.assert_array_equal(np.asarray(restored), image)
    recipe = {"augmentation": False}
    labels = pd.DataFrame(
        [["A001", *np.linspace(1, 100, 11)]], columns=["sample_id", *TARGET_COLUMNS]
    )
    bag = FullPhotoBags(manifest, tmp_path / "cache", photo_transform(recipe), labels)[
        0
    ]
    assert isinstance(bag["image"], list)
    assert bag["image"][0].shape == (3, 20, 30)
    assert bag["target"].shape == (1, 11)
    assert bag["bag_weights"].sum() == 1


def test_padding_keeps_all_original_pixels():
    image = torch.randn(1, 3, 19, 37)
    padded = pad_to_multiple(image, 16)
    assert padded.shape == (1, 3, 32, 48)
    torch.testing.assert_close(padded[..., :19, :37], image)
    assert padded[..., 19:, :].count_nonzero() == 0
    assert padded[..., :, 37:].count_nonzero() == 0


def test_scratch_variable_rectangles_and_sample_gradient():
    torch.set_num_threads(2)
    recipe = {
        "arch": "scratch_vit",
        "patch_size": 8,
        "dim": 24,
        "depth": 2,
        "heads": 3,
        "dropout": 0,
        "drop_path": 0,
    }
    prior = np.linspace(1, 100, 11)
    model = FullFrameModel(recipe, prior)
    images = [torch.randn(3, 19, 37), torch.randn(3, 29, 17)]
    weights = torch.tensor([0.5, 0.5])
    curve = model(images, weights)
    assert curve.shape == (1, 11)
    validate_curves(curve.detach().numpy())
    curve[:, :-1].sum().backward()
    gradient = model.encoder.patch_embed.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0
    model.eval()
    with torch.inference_mode():
        direct = model(images, weights)
        features = model.encode(images)
        pooled = model.pool(features, weights)
        torch.testing.assert_close(direct, pooled)
        reversed_curve = model(list(reversed(images)), weights)
        torch.testing.assert_close(direct, reversed_curve)


def test_scratch_token_guard_does_not_change_scale():
    model = ScratchViT(patch_size=8, dim=24, depth=1, heads=3, max_tokens=4)
    with pytest.raises(ValueError, match="common PPM"):
        model(torch.randn(1, 3, 17, 17))


def test_swin_last_stage_policy_and_full_rectangle():
    torch.set_num_threads(2)
    model = SwinPhotoEncoder(pretrained=False, tuning="last_stage")
    model.train()
    assert not model.backbone.features[1].training
    assert model.backbone.features[-1].training
    assert not any(p.requires_grad for p in model.backbone.features[1].parameters())
    assert all(p.requires_grad for p in model.backbone.features[-1].parameters())
    model.eval()
    with torch.inference_mode():
        feature = model(torch.randn(1, 3, 33, 51))
    assert feature.shape == (1, 768)
    assert torch.isfinite(feature).all()
    ids = model.layer_ids()
    assert set(ids) == set(dict(model.named_parameters()))


def test_common_ppm_rounding_is_recorded_without_per_photo_caps(tmp_path):
    rows = []
    for name, size, native_ppm in [
        ("a.png", (101, 53), 4.0),
        ("b.png", (202, 106), 8.0),
    ]:
        path = tmp_path / name
        Image.new("RGB", size, color="white").save(path)
        rows.append(
            {"sample_id": name, "photo": name, "path": str(path), "ppm_est": native_ppm}
        )
    manifest = make_photo_manifest(pd.DataFrame(rows), target_ppm=2)
    assert manifest.output_width.tolist() == [50, 50]
    assert manifest.output_height.tolist() == [26, 26]
    assert manifest.actual_ppm_x.nunique() == 1
    assert manifest.actual_ppm_y.nunique() == 1
    assert np.all(abs(manifest.actual_ppm_x - 2) < 0.02)
    assert np.all(abs(manifest.actual_ppm_y - 2) < 0.04)
    with pytest.raises(ValueError, match="native PPM"):
        make_photo_manifest(pd.DataFrame(rows), target_ppm=5)


def test_domain_probe_keeps_heldout_soils_and_pairs(tmp_path):
    from soilcv.vit.experiments import _camera_subset

    train = pd.DataFrame(
        {
            "sample_id": ["train1", "train1", "train2"],
            "device": ["Motorola Edge", "Samsung A52", "Samsung A52"],
            "fold": [1, 1, 2],
        }
    )
    valid = pd.DataFrame(
        {
            "sample_id": ["valid1", "valid1", "valid2"],
            "device": ["Motorola Edge", "Samsung A52", "Motorola Edge"],
            "fold": [0, 0, 0],
        }
    )
    source, heldout = _camera_subset(train, valid, "Motorola")
    assert source.sample_id.tolist() == ["train1"]
    assert source.device.tolist() == ["Motorola Edge"]
    assert set(heldout.sample_id) == {"valid1"}
    assert len(heldout) == 2
    assert set(heldout.fold) == {0}
    assert not set(source.sample_id) & set(heldout.sample_id)
