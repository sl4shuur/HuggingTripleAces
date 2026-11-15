import json

import pytest
import torch
from click.testing import CliRunner

pytest.importorskip("transformers")
from transformers import Dinov2Config, Dinov2Model

from alseg.adapters.dinov2 import DinoV2Encoder
from alseg.core.pool import PoolManifest
from alseg.data.preprocessing import prepare_images
from alseg.interfaces.data import ImageSource
from alseg.nn.extraction import iter_features
from script import extract_features


class TinyImages(ImageSource):
    def __len__(self):
        return 3

    def get_image(self, image_id):
        return torch.full((3, 17, 24), image_id * 100, dtype=torch.uint8)


@pytest.fixture
def encoder():
    torch.manual_seed(42)
    config = Dinov2Config(
        image_size=28,
        patch_size=14,
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=3,
    )
    return DinoV2Encoder(Dinov2Model(config), image_size=28)


def test_frozen_features_have_correct_geometry_and_allow_head_training(encoder):
    images = [TinyImages().get_image(0), TinyImages().get_image(2)]
    first = encoder.encode(images)
    assert first.embeddings.shape == (2, 12)
    assert first.patch_features.shape == (2, 12, 2, 2)
    torch.testing.assert_close(first.embeddings.norm(dim=1), torch.ones(2))
    encoder.model.train()
    second = encoder.encode(images)
    torch.testing.assert_close(first.patch_features, second.patch_features)
    assert not encoder.model.training
    head = torch.nn.Conv2d(12, 21, 1)
    head(first.patch_features).sum().backward()
    assert head.weight.grad is not None
    assert all(
        p.grad is None and not p.requires_grad for p in encoder.model.parameters()
    )


def test_streaming_retains_requested_id_order(encoder):
    source = TinyImages()
    batches = list(iter_features(source, encoder, [2, 0, 1], batch_size=2))
    assert [ids for ids, _ in batches] == [[2, 0], [1]]
    together = torch.cat([features.embeddings for _, features in batches])
    direct = encoder.encode([source.get_image(i) for i in [2, 0, 1]])
    torch.testing.assert_close(together, direct.embeddings, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("ids,batch_size", [([0, 0], 2), ([3], 2), ([0], 0)])
def test_streaming_rejects_invalid_requests(encoder, ids, batch_size):
    with pytest.raises(ValueError):
        list(iter_features(TinyImages(), encoder, ids, batch_size))


def test_preprocessing_uses_full_frame_rgb_and_checkpoint_normalization():
    pixels = prepare_images([torch.full((3, 11, 23), 255, dtype=torch.uint8)], 28)
    expected = torch.tensor(
        [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]
    )
    torch.testing.assert_close(pixels[0, :, 0, 0], expected)
    assert pixels.shape == (1, 3, 28, 28)
    for images in (
        [],
        [torch.zeros(3, 2, 2)],
        [torch.zeros(1, 2, 2, dtype=torch.uint8)],
    ):
        with pytest.raises(ValueError):
            prepare_images(images, 28)


def test_encoder_rejects_fractional_patch_grids_and_mutable_checkpoint_revisions(
    encoder,
):
    with pytest.raises(ValueError, match="multiple"):
        DinoV2Encoder(encoder.model, image_size=29)
    with pytest.raises(ValueError, match="SHA"):
        DinoV2Encoder.from_pretrained("facebook/dinov2-small", "main", 28)


def test_cli_saves_subset_identity_and_refuses_overwrite(
    encoder, tmp_path, monkeypatch
):
    monkeypatch.setattr(extract_features, "find_voc_root", lambda _: tmp_path)
    manifest = PoolManifest(["a", "b", "c"], ["v"])
    monkeypatch.setattr(extract_features, "load_manifest", lambda _: manifest)
    monkeypatch.setattr(extract_features, "VocImages", lambda *args: TinyImages())
    monkeypatch.setattr(
        DinoV2Encoder, "from_pretrained", lambda *args, **kwargs: encoder
    )
    output = tmp_path / "features"
    args = ["--device", "cpu", "--limit", "2", "--output-dir", str(output)]
    result = CliRunner().invoke(extract_features.main, args)
    assert result.exit_code == 0, result.output
    saved = torch.load(output / "embeddings.pt", weights_only=True)
    metadata = json.loads((output / "metadata.json").read_text())
    assert saved["ids"] == [0, 1]
    assert saved["embeddings"].shape == (2, 12)
    assert metadata["complete_pool"] is False
    assert metadata["pool_ids"] == ["a", "b", "c"]
    assert metadata["frozen"] is True
    repeated = CliRunner().invoke(extract_features.main, args)
    assert repeated.exit_code == 1
    assert "never overwritten" in repeated.output
