import json
import threading

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from alseg.adapters.voc import VocEvaluation, VocImages
from alseg.config import EncoderConfig, EncoderName, ExperimentConfig
from alseg.core.features import EncoderFeatures
from alseg.core.pool import PoolManifest
from alseg.interfaces.encoder import Encoder
from alseg.nn.encoders import VIT16_WEIGHTS, load_encoder
from alseg.nn.extraction import iter_features
from script import extract_features

pytest_plugins = ["test_data"]


class TinyEncoder(Encoder):
    def encode(self, images):
        values = torch.tensor([float(image.float().mean()) for image in images])
        vectors = torch.stack((values, torch.ones_like(values)), dim=1)
        return EncoderFeatures(
            torch.nn.functional.normalize(vectors, dim=1),
            values[:, None, None, None].expand(-1, 2, 1, 1).contiguous(),
        )

    def describe(self):
        return {
            "feature_dim": 2,
            "patch_size": 16,
            "image_size": 16,
            "weights_sha256": VIT16_WEIGHTS[EncoderName.DINO_VITB16]["sha256"],
            "image_pooling": "l2_normalized_cls",
            "preprocessing": "full_frame_square_bicubic_antialias_clamp_0_1",
        }


def test_cache_round_trip_uses_ids_and_never_decodes_images(voc, tmp_path, monkeypatch):
    root, manifest = voc
    config = ExperimentConfig(data={"voc_root": str(root)}, encoder={"image_size": 16})
    config_path = config.save(tmp_path / "config.json")
    output = tmp_path / "features"
    monkeypatch.setattr(extract_features, "load_encoder", lambda *args: TinyEncoder())
    args = [
        "--config",
        str(config_path),
        "--device",
        "cpu",
        "--output-dir",
        str(output),
    ]
    result = CliRunner().invoke(extract_features.main, args)
    assert result.exit_code == 0, result.output
    assert json.loads((output / "extraction.json").read_text())["reusable"]
    config.encoder.features_dir = output / EncoderName.DINO_VITB16
    cached = load_encoder(config.encoder.model_dump(mode="json"), tmp_path, "cpu")
    monkeypatch.setattr(
        VocImages, "get_image", lambda *args: pytest.fail("Decoded RGB")
    )
    pool = VocImages(root, manifest)
    batch_ids, features = next(iter_features(pool, cached, [1, 0], 2))
    expected = np.load(config.encoder.features_dir / "train" / "patches.npy")
    assert batch_ids == [1, 0]
    torch.testing.assert_close(
        features.patch_features, torch.from_numpy(expected[[1, 0]])
    )
    assert next(iter_features(VocEvaluation(root, manifest), cached, [0], 1))[
        1
    ].patch_features.shape == (1, 2, 1, 1)
    reordered = VocImages(root, PoolManifest(manifest.pool_ids[::-1], manifest.val_ids))
    with pytest.raises(ValueError, match="order"):
        next(iter_features(reordered, cached, [0], 1))
    with pytest.raises(ValueError, match="image_size"):
        load_encoder(
            EncoderConfig(
                features_dir=config.encoder.features_dir, image_size=32
            ).model_dump(mode="json"),
            tmp_path,
            "cpu",
        )
    repeated = CliRunner().invoke(extract_features.main, args)
    assert repeated.exit_code != 0 and "never overwritten" in repeated.output


def test_two_gpu_dispatch_assigns_one_concurrent_job_to_each_gpu(
    voc, tmp_path, monkeypatch
):
    path = ExperimentConfig(data={"voc_root": str(voc[0])}).save(
        tmp_path / "config.json"
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    calls = []
    barrier = threading.Barrier(2)

    def worker(config, name, device, *args):
        calls.append((name, device))
        barrier.wait(timeout=5)
        return {
            "encoder_identity": {"name": name},
            "device": device,
            "extraction_seconds": 0,
            "splits": {"train": {"complete": True, "patches_shape": [2, 2, 1, 1]}},
        }

    monkeypatch.setattr(extract_features, "extract_encoder", worker)
    result = CliRunner().invoke(
        extract_features.main,
        [
            "--config",
            str(path),
            "--encoder",
            "dino_vitb16",
            "--encoder",
            "ibot_vitb16",
            "--gpus",
            "2",
            "--output-dir",
            str(tmp_path / "parallel"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert sorted(calls) == [("dino_vitb16", "cuda:0"), ("ibot_vitb16", "cuda:1")]
