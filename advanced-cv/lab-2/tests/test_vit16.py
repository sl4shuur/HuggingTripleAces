import pytest
import torch

pytest.importorskip("timm")
from timm.models.vision_transformer import VisionTransformer

from alseg.adapters.vit16 import ViT16Encoder
from alseg.config import EncoderConfig
from alseg.nn.encoders import load_encoder


def test_vit_backbone_exposes_frozen_dense_features():
    model = VisionTransformer(
        img_size=32,
        patch_size=16,
        embed_dim=12,
        depth=1,
        num_heads=3,
        num_classes=0,
    )
    encoder = ViT16Encoder(model, 32, {"name": "unit-test"}, "cpu")
    features = encoder.encode([torch.full((3, 23, 39), 120, dtype=torch.uint8)])
    assert features.patch_features.shape == (1, 12, 2, 2)
    torch.testing.assert_close(features.embeddings.norm(dim=1), torch.ones(1))
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("name", ["dino_vitb16", "ibot_vitb16"])
def test_local_checkpoint_requires_the_pinned_official_hash(name, tmp_path):
    path = tmp_path / "weights.pth"
    torch.save({"cls_token": torch.zeros(1, 1, 768)}, path)
    config = EncoderConfig(name=name, checkpoint_path=path).model_dump(mode="json")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_encoder(config, tmp_path, "cpu")
