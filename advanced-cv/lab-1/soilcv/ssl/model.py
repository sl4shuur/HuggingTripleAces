"""Official SSL encoders, deterministic descriptors and sample-level regression."""

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from soilcv.nn import MassCurveHead, set_trainable

ENCODERS = {
    "dinov3": {
        "repo": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "revision": "114c1379950215c8b35dfcd4e90a5c251dde0d32",
    },
    "mae": {
        "repo": "facebook/vit-mae-base",
        "revision": "25b184bea5538bf5c4c852c79d221195fdd2778d",
    },
}


class SSLEncoder(nn.Module):
    def __init__(self, name, *, source=None, config=None, pooling="cls_patch"):
        super().__init__()
        from transformers import (
            DINOv3ViTConfig,
            DINOv3ViTModel,
            ViTMAEConfig,
            ViTMAEModel,
        )

        if name not in ENCODERS or pooling not in {"cls", "cls_patch"}:
            raise ValueError("Choose dinov3/mae and cls/cls_patch pooling.")
        self.name, self.pooling = name, pooling
        model_class = DINOv3ViTModel if name == "dinov3" else ViTMAEModel
        config_class = DINOv3ViTConfig if name == "dinov3" else ViTMAEConfig
        if config is not None:
            self.backbone = model_class(config_class(**config))
        else:
            spec = ENCODERS[name]
            self.backbone = model_class.from_pretrained(
                source or spec["repo"],
                **({} if source else {"revision": spec["revision"]}),
                attn_implementation="sdpa",
            )
        if name == "mae":
            # MAE defaults to random 75% masking, even during evaluation.
            self.backbone.config.mask_ratio = 0.0
        else:
            # Do not add a hidden random coordinate-scale augmentation.
            self.backbone.config.pos_embed_rescale = None
        self.output_dim = self.backbone.config.hidden_size * (
            2 if pooling == "cls_patch" else 1
        )

    def forward(self, images):
        if self.name == "mae":
            size = self.backbone.config.patch_size
            patches = (images.shape[-2] // size) * (images.shape[-1] // size)
            # Keep raster patch order as well as all patches. No RNG consumption.
            noise = torch.arange(patches, device=images.device, dtype=torch.float32)
            output = self.backbone(
                pixel_values=images, noise=noise.expand(len(images), -1)
            )
            start = 1
        else:
            output = self.backbone(pixel_values=images)
            start = 1 + self.backbone.config.num_register_tokens
        tokens = output.last_hidden_state
        cls = tokens[:, 0]
        if self.pooling == "cls":
            return cls
        return torch.cat([cls, tokens[:, start:].mean(1)], dim=1)

    def layer_ids(self):
        depth = self.backbone.config.num_hidden_layers
        result = {}
        for name, _ in self.named_parameters():
            parts = name.split(".")
            if "layer" in parts:
                layer = int(parts[parts.index("layer") + 1]) + 1
            elif "embeddings" in parts:
                layer = 0
            else:
                layer = depth + 1
            result[name] = layer
        return result

    def no_weight_decay(self):
        return [
            name
            for name, _ in self.named_parameters()
            if any(
                token in name
                for token in [
                    "cls_token",
                    "mask_token",
                    "register_tokens",
                    "position_embeddings",
                ]
            )
        ]


def make_head(size, prior):
    head = MassCurveHead(size)
    mass = np.maximum(np.diff(np.r_[0, prior]), 0.001)
    if mass.shape != (11,) or not np.isfinite(mass).all():
        raise ValueError("Provide the training fold's 11-bin mean curve.")
    with torch.no_grad():
        nn.init.normal_(head.projection.weight, std=0.001)
        head.projection.bias.copy_(torch.as_tensor(np.log(mass), dtype=torch.float32))
    return head


class SSLRegressor(nn.Module):
    """Area-weight tiles within photos; give each photo equal sample weight."""

    def __init__(self, encoder, head, *, frozen=False, chunk_size=32):
        super().__init__()
        self.encoder, self.head = encoder, head
        self.frozen, self.chunk_size = frozen, chunk_size
        if frozen:
            set_trainable(encoder, False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def encode(self, images):
        features = []
        for chunk in images.split(self.chunk_size):
            if self.training and not self.frozen:
                value = checkpoint(self.encoder, chunk, use_reentrant=False)
            else:
                value = self.encoder(chunk)
            features.append(value)
        return torch.cat(features)

    def pool(self, features, weights):
        if (
            len(features) != len(weights)
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError("Use finite non-negative tile weights with positive sum.")
        pooled = (features.float() * weights[:, None]).sum(0, keepdim=True)
        return self.head(pooled / weights.sum())

    def forward(self, images, weights):
        return self.pool(self.encode(images), weights)
