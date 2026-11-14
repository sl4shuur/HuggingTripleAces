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
    assert row.spatial_coverage == 1
    assert row.cropped_source_area == 0
    assert row.linear_scale_ratio == 1
    assert row.pixel_count_ratio == 1
    assert "coverage" not in manifest and "missing_native_pixels" not in manifest
    assert row.actual_ppm_x == row.actual_ppm_y == 2
    cache_photos(manifest, tmp_path / "cache")
    with Image.open(tmp_path / "cache" / row.frame_path) as restored:
        np.testing.assert_array_equal(np.asarray(restored), image)
    recipe = {"augmentation": False}
    labels = pd.DataFrame(
        [["A001", *np.linspace(1, 100, 11)]], columns=["sample_id", *TARGET_COLUMNS]
    )
    bag = FullPhotoBags(
        manifest,
        tmp_path / "cache",
        photo_transform(recipe),
        labels,
        view_mode="global",
    )[0]
    assert isinstance(bag["image"], list)
    assert bag["image"][0]["global"].shape == (3, 20, 30)
    assert bag["target"].shape == (1, 11)
    assert bag["bag_weights"].sum() == 1


def test_padding_keeps_all_original_pixels():
    image = torch.randn(1, 3, 19, 37)
    padded = pad_to_multiple(image, 16)
    assert padded.shape == (1, 3, 32, 48)
    torch.testing.assert_close(padded[..., :19, :37], image)
    torch.testing.assert_close(
        padded[..., :19, 37:], image[..., :, -1:].expand(-1, -1, -1, 11)
    )
    torch.testing.assert_close(
        padded[..., 19:, :37], image[..., -1:, :].expand(-1, -1, 13, -1)
    )


def test_scratch_variable_rectangles_and_sample_gradient():
    torch.set_num_threads(2)
    recipe = {
        "arch": "scratch_vit",
        "view_mode": "global",
        "patch_size": 8,
        "dim": 24,
        "depth": 2,
        "heads": 3,
        "dropout": 0,
        "drop_path": 0,
    }
    prior = np.linspace(1, 100, 11)
    model = FullFrameModel(recipe, prior)
    images = [{"global": torch.randn(3, 19, 37)}, {"global": torch.randn(3, 29, 17)}]
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


def test_multiscale_cache_keeps_fine_texture_and_complete_tile_area(tmp_path):
    from soilcv.bags import cache_grid, make_grid_manifest

    source = tmp_path / "checker.png"
    grid = (np.indices((80, 120)).sum(0) % 2 * 255).astype(np.uint8)
    Image.fromarray(np.repeat(grid[..., None], 3, axis=2)).save(source)
    photos = pd.DataFrame(
        [
            {
                "sample_id": "soil",
                "photo": source.name,
                "path": str(source),
                "ppm_est": 4.5,
                "device": "Motorola",
            }
        ]
    )
    frames = make_photo_manifest(photos, target_ppm=2)
    tiles = make_grid_manifest(photos, tile_mm=16, output_px=72)
    cache = tmp_path / "cache"
    cache_photos(frames, cache)
    cache_grid(tiles, cache)
    bag = FullPhotoBags(
        frames,
        cache,
        photo_transform({"augmentation": False}),
        tiles=tiles,
        view_mode="multiscale",
    )[0]
    photo = bag["image"][0]
    assert photo["local"].shape == (len(tiles), 3, 72, 72)
    assert photo["local_weights"].sum() == pytest.approx(1)
    np.testing.assert_allclose(photo["local_weights"].numpy(), tiles.photo_weight)
    assert bag["bag_weights"].tolist() == [1]
    row = frames.iloc[0]
    assert row.linear_scale_ratio == pytest.approx(2 / 4.5)
    assert row.pixel_count_ratio == pytest.approx(
        row.output_width * row.output_height / 9600
    )
    with Image.open(cache / row.frame_path) as global_view:
        global_variance = np.asarray(global_view).var()
    with Image.open(cache / tiles.iloc[0].tile_path) as local_view:
        local_variance = np.asarray(local_view).var()
    assert local_variance > 100 * global_variance
    with pytest.raises(ValueError, match="same photos"):
        FullPhotoBags(
            frames, cache, photo_transform({}), tiles=tiles.assign(photo="wrong")
        )


def test_multiscale_features_receive_gradients_from_both_views():
    recipe = {
        "arch": "scratch_vit",
        "view_mode": "multiscale",
        "patch_size": 8,
        "dim": 24,
        "depth": 1,
        "heads": 3,
        "dropout": 0,
        "drop_path": 0,
        "chunk_size": 1,
    }
    model = FullFrameModel(recipe, np.linspace(1, 100, 11))
    global_view = torch.randn(3, 19, 27, requires_grad=True)
    local_views = torch.randn(2, 3, 16, 16, requires_grad=True)
    photo = {
        "global": global_view,
        "local": local_views,
        "local_weights": torch.tensor([0.25, 0.75]),
    }
    curve = model([photo], torch.ones(1))
    curve[:, :-1].sum().backward()
    assert global_view.grad.abs().sum() > 0
    assert local_views.grad[0].abs().sum() > 0
    assert local_views.grad[1].abs().sum() > 0
    model.eval()
    with torch.inference_mode():
        features = model.encode([photo])
    assert features.shape == (1, 48)
    assert model.head.projection.in_features == 48


def test_scratch_fully_padded_tokens_cannot_change_cls():
    torch.manual_seed(42)
    model = ScratchViT(
        patch_size=8, dim=24, depth=2, heads=3, dropout=0, drop_path=0
    ).eval()
    image = torch.randn(1, 3, 19, 27)
    original = image.clone()
    with torch.inference_mode():
        baseline = model(image)
        model.padding_extra = 8
        extra = model(image)
    torch.testing.assert_close(image, original)
    torch.testing.assert_close(baseline, extra, rtol=1e-5, atol=1e-6)


def test_padding_diagnostic_keeps_sources_and_restores_encoder():
    from soilcv.vit.diagnostics import padding_sensitivity

    recipe = {
        "arch": "scratch_vit",
        "view_mode": "global",
        "patch_size": 8,
        "dim": 24,
        "depth": 1,
        "heads": 3,
        "dropout": 0,
        "drop_path": 0,
    }
    model = FullFrameModel(recipe, np.linspace(1, 100, 11))
    source = torch.randn(3, 19, 27)
    original = source.clone()
    bag = {
        "sample_id": "soil",
        "image": [{"global": source}],
        "bag_weights": torch.ones(1),
        "device": ["Motorola"],
    }
    model.encoder.padding_extra = 3
    metrics, pairs = padding_sensitivity(model, [bag])
    assert metrics["padding_samples"] == 1
    assert metrics["padding_extra_pixels"] == 8
    assert metrics["padding_max_emd"] < 1e-3
    assert pairs.sample_id.tolist() == ["soil"]
    assert model.encoder.padding_extra == 3
    torch.testing.assert_close(source, original)


def test_local_area_weights_do_not_change_equal_photo_weighting():
    from torch import nn

    class MeanFeatures(nn.Module):
        def forward(self, values):
            return values.mean((1, 2, 3))[:, None].expand(-1, 24)

    recipe = {
        "arch": "scratch_vit",
        "view_mode": "multiscale",
        "patch_size": 8,
        "dim": 24,
        "depth": 1,
        "heads": 3,
    }
    model = FullFrameModel(recipe, np.linspace(1, 100, 11)).eval()
    model.encoder = MeanFeatures()
    model.head = nn.Identity()
    photos = [
        {
            "local": torch.stack([torch.ones(3, 8, 8), 3 * torch.ones(3, 8, 8)]),
            "local_weights": torch.tensor([0.25, 0.75]),
            "global": 10 * torch.ones(3, 8, 8),
        },
        {
            "local": 5 * torch.ones(1, 3, 8, 8),
            "local_weights": torch.ones(1),
            "global": 20 * torch.ones(3, 8, 8),
        },
    ]
    with torch.inference_mode():
        pooled = model(photos, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(pooled[:, :24], torch.full((1, 24), 3.75))
    torch.testing.assert_close(pooled[:, 24:], torch.full((1, 24), 15.0))


def test_tiles_only_mode_never_needs_global_tensor():
    recipe = {
        "arch": "scratch_vit",
        "view_mode": "tiles",
        "patch_size": 8,
        "dim": 24,
        "depth": 1,
        "heads": 3,
        "dropout": 0,
        "drop_path": 0,
    }
    model = FullFrameModel(recipe, np.linspace(1, 100, 11)).eval()
    with torch.inference_mode():
        curve = model(
            [
                {
                    "local": torch.randn(2, 3, 16, 16),
                    "local_weights": torch.tensor([0.5, 0.5]),
                }
            ],
            torch.ones(1),
        )
    assert model.head.projection.in_features == 24
    validate_curves(curve.numpy())
