"""DINOv2 backbone adapter; no pretrained segmentation or classification head."""

import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from alseg.core.features import EncoderFeatures
from alseg.features import IMAGE_MEAN, IMAGE_STD, prepare_images
from alseg.interfaces import Encoder


class DinoV2Encoder(Encoder):
    def __init__(self, model: torch.nn.Module, image_size: int, device: str = "cpu"):
        self.patch_size = model.config.patch_size
        if model.config.model_type != "dinov2":
            raise ValueError(
                "Expected a plain DINOv2 backbone, without register tokens"
            )
        if (
            type(image_size) is not int
            or image_size <= 0
            or image_size % self.patch_size
        ):
            raise ValueError("image_size must be a positive multiple of patch_size")
        self.device = torch.device(device)
        self.model = model.to(self.device).requires_grad_(False).eval()
        self.image_size = image_size
        self.model_id: str | None = None
        self.revision: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str,
        image_size: int,
        device: str = "cpu",
        cache_dir: Path | None = None,
    ) -> "DinoV2Encoder":
        from transformers import Dinov2Model

        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("Pin checkpoint revision to an immutable 40-character SHA")
        model = Dinov2Model.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            use_safetensors=True,
        )
        encoder = cls(model, image_size, device)
        encoder.model_id, encoder.revision = model_id, revision
        return encoder

    @torch.no_grad()
    def encode(self, images: list[Tensor]) -> EncoderFeatures:
        pixels = prepare_images(images, self.image_size).to(self.device)
        self.model.eval()
        tokens = self.model(pixel_values=pixels).last_hidden_state
        grid = self.image_size // self.patch_size
        if tokens.shape[1:] != (1 + grid * grid, self.model.config.hidden_size):
            raise ValueError("Unexpected DINOv2 CLS/patch token layout")
        embeddings = F.normalize(tokens[:, 0].float(), dim=1)
        patches = (
            tokens[:, 1:]
            .transpose(1, 2)
            .reshape(
                len(images),
                self.model.config.hidden_size,
                grid,
                grid,
            )
        )
        # no_grad produces ordinary tensors that a trainable head can reuse.
        return EncoderFeatures(embeddings.cpu(), patches.float().cpu().contiguous())

    def describe(self) -> dict:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "feature_dim": self.model.config.hidden_size,
            "patch_size": self.patch_size,
            "image_size": self.image_size,
            "image_pooling": "l2_normalized_cls",
            "preprocessing": "full_frame_square_bicubic_antialias_clamp_0_1",
            "image_mean": list(IMAGE_MEAN),
            "image_std": list(IMAGE_STD),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "frozen": all(not p.requires_grad for p in self.model.parameters()),
        }
