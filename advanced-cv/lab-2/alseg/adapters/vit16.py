"""Frozen DINO/iBOT ViT backbone, with no pretraining or downstream head."""

import torch
import torch.nn.functional as F
from torch import Tensor

from alseg.core.features import EncoderFeatures
from alseg.data.preprocessing import IMAGE_MEAN, IMAGE_STD, prepare_images
from alseg.interfaces.encoder import Encoder


class ViT16Encoder(Encoder):
    def __init__(
        self, model: torch.nn.Module, image_size: int, identity: dict, device: str
    ):
        if type(image_size) is not int or image_size <= 0 or image_size % 16:
            raise ValueError("image_size must be a positive multiple of patch_size=16")
        self.model = model.to(device).requires_grad_(False).eval()
        self.image_size = image_size
        self.device = torch.device(device)
        self.identity = identity

    @torch.no_grad()
    def encode(self, images: list[Tensor]) -> EncoderFeatures:
        self.model.eval()
        pixels = prepare_images(images, self.image_size).to(self.device)
        tokens = self.model(pixel_values=pixels).last_hidden_state
        grid, dim = self.image_size // 16, self.model.config.hidden_size
        if tokens.shape[1:] != (1 + grid * grid, dim):
            raise ValueError("Unexpected ViT CLS/patch token layout")
        patches = tokens[:, 1:].transpose(1, 2).reshape(len(images), dim, grid, grid)
        return EncoderFeatures(
            F.normalize(tokens[:, 0].float(), dim=1).cpu(),
            patches.float().cpu().contiguous(),
        )

    def describe(self) -> dict:
        return {
            **self.identity,
            "feature_dim": self.model.config.hidden_size,
            "patch_size": 16,
            "image_size": self.image_size,
            "image_pooling": "l2_normalized_cls",
            "preprocessing": "full_frame_square_bicubic_antialias_clamp_0_1",
            "position_interpolation": "official_bicubic_scale_factor_offset_0.1",
            "image_mean": list(IMAGE_MEAN),
            "image_std": list(IMAGE_STD),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "frozen": all(not p.requires_grad for p in self.model.parameters()),
        }
