"""Lecture-style ViT and pretrained Swin, with one curve per full photo bag."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import models
from torchvision.ops import StochasticDepth

from soilcv.nn import MassCurveHead, set_trainable


def pad_to_multiple(image, multiple):
    """Pad normalized pixels on the right and bottom. Never discard an edge."""
    height, width = image.shape[-2:]
    return F.pad(image, (0, (-width) % multiple, 0, (-height) % multiple))


def position_2d(height, width, dim, *, device, dtype):
    """2-D sine/cosine positions work with any rectangular patch grid."""
    if dim % 4:
        raise ValueError("The embedding dimension must be divisible by four.")
    frequency = torch.arange(dim // 4, device=device, dtype=torch.float32)
    frequency = 1 / (10000 ** (frequency / (dim // 4)))
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    x = x.flatten()[:, None].float() * frequency
    y = y.flatten()[:, None].float() * frequency
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1).to(dtype)


class Attention(nn.Module):
    """The Lecture 1 QKV attention, using memory-efficient PyTorch SDPA."""

    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError("The embedding dimension must divide by the head count.")
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, tokens):
        batch, length, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.heads, dim // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        values = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.projection(values.transpose(1, 2).reshape(batch, length, dim))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, *, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, tokens):
        tokens = tokens + self.drop_path(self.attention(self.norm1(tokens)))
        return tokens + self.drop_path(self.mlp(self.norm2(tokens)))


class ScratchViT(nn.Module):
    """Conv patch embedding -> CLS + positions -> pre-norm blocks -> CLS."""

    def __init__(
        self,
        *,
        patch_size=16,
        dim=192,
        depth=4,
        heads=3,
        dropout=0.1,
        drop_path=0.1,
        max_tokens=2048,
    ):
        super().__init__()
        if patch_size < 1 or depth < 1 or dim % 4 or dim % heads:
            raise ValueError("Use a valid patch size, depth and embedding dimension.")
        self.patch_size = patch_size
        self.max_tokens = max_tokens
        self.output_dim = dim
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    dropout=dropout,
                    drop_path=drop_path * index / max(1, depth - 1),
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, image):
        # Each image is padded alone, so every patch contains real image pixels.
        patches = self.patch_embed(pad_to_multiple(image, self.patch_size))
        height, width = patches.shape[-2:]
        if height * width > self.max_tokens:
            raise ValueError(
                "Too many tokens. Lower the common PPM for the whole study."
            )
        tokens = patches.flatten(2).transpose(1, 2)
        positions = position_2d(
            height, width, self.output_dim, device=tokens.device, dtype=tokens.dtype
        )
        tokens = tokens + positions[None]
        tokens = torch.cat([self.cls.expand(len(image), -1, -1), tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)[:, 0]

    def layer_ids(self):
        ids = {}
        for name, _ in self.named_parameters():
            if name.startswith("blocks."):
                layer = int(name.split(".")[1]) + 1
            elif name.startswith("norm."):
                layer = len(self.blocks) + 1
            else:
                layer = 0
            ids[name] = layer
        return ids


class SwinPhotoEncoder(nn.Module):
    """Window attention supports large, rectangular photos at modest cost."""

    def __init__(self, *, pretrained=True, tuning="frozen", drop_path=0.1):
        super().__init__()
        if tuning not in {"frozen", "last_stage", "full"}:
            raise ValueError("Choose frozen, last_stage or full Swin tuning.")
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.swin_t(weights=weights)
        paths = [m for m in self.backbone.modules() if isinstance(m, StochasticDepth)]
        for index, module in enumerate(paths):
            module.p = drop_path * index / max(1, len(paths) - 1)
        self.output_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.tuning = tuning
        if tuning != "full":
            set_trainable(self.backbone, False)
        if tuning == "last_stage":
            set_trainable(self.backbone.features[-1], True)
            set_trainable(self.backbone.norm, True)

    def train(self, mode=True):
        super().train(mode)
        if self.tuning == "frozen":
            self.backbone.eval()
        elif self.tuning == "last_stage":
            for module in self.backbone.features[:-1]:
                module.eval()
        return self

    def forward(self, image):
        height, width = image.shape[-2:]
        # The hierarchy downsamples by 32. Padding retains every source pixel.
        features = self.backbone.features(pad_to_multiple(image, 32))
        features = self.backbone.norm(features)
        rows = torch.arange(features.shape[1], device=image.device)
        columns = torch.arange(features.shape[2], device=image.device)
        valid_y = (height - 32 * rows).clamp(0, 32).float()
        valid_x = (width - 32 * columns).clamp(0, 32).float()
        area = valid_y[:, None] * valid_x[None, :]
        # Partial edge cells count by valid area, not padded area.
        return (features.float() * area[None, :, :, None]).sum((1, 2)) / area.sum()

    def layer_ids(self):
        ids = {}
        for name, _ in self.named_parameters():
            if name.startswith("backbone.features."):
                layer = int(name.split(".")[2])
            else:
                layer = 8
            ids[name] = layer
        return ids


class FullFrameModel(nn.Module):
    """Encode all complete photos, average features, predict one sample curve."""

    def __init__(self, recipe, train_mean, *, pretrained=True):
        super().__init__()
        if recipe["arch"] == "scratch_vit":
            self.encoder = ScratchViT(
                patch_size=recipe.get("patch_size", 16),
                dim=recipe.get("dim", 192),
                depth=recipe.get("depth", 4),
                heads=recipe.get("heads", 3),
                dropout=recipe.get("dropout", 0.1),
                drop_path=recipe.get("drop_path", 0.1),
                max_tokens=recipe.get("max_tokens", 2048),
            )
            self.frozen = False
        elif recipe["arch"] == "swin_t":
            tuning = recipe.get("tuning", "frozen")
            self.encoder = SwinPhotoEncoder(
                pretrained=pretrained,
                tuning=tuning,
                drop_path=recipe.get("drop_path", 0.1),
            )
            self.frozen = tuning == "frozen"
        else:
            raise ValueError("Unknown full-photo Transformer architecture.")
        self.head = MassCurveHead(self.encoder.output_dim)
        mass = np.maximum(np.diff(np.r_[0, train_mean]), 0.001)
        if mass.shape != (11,) or not np.isfinite(mass).all():
            raise ValueError("Provide an 11-bin training-fold mean curve.")
        with torch.no_grad():
            nn.init.normal_(self.head.projection.weight, std=0.001)
            self.head.projection.bias.copy_(
                torch.tensor(np.log(mass), dtype=torch.float32)
            )

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def encode(self, images):
        features = []
        for image in images:
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError("Each complete photo must be a CHW RGB tensor.")
            if self.training and not self.frozen:
                values = checkpoint(self.encoder, image[None], use_reentrant=False)
            else:
                values = self.encoder(image[None])
            features.append(values)
        if not features:
            raise ValueError("A sample needs at least one photo.")
        return torch.cat(features)

    def pool(self, features, weights):
        weights = weights.float()
        if (
            len(weights) != len(features)
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError("Use one finite positive weight per photo.")
        values = (features.float() * weights[:, None]).sum(
            0, keepdim=True
        ) / weights.sum()
        return self.head(values)

    def forward(self, images, weights):
        return self.pool(self.encode(images), weights)


def encoder_layer_ids(encoder):
    return encoder.layer_ids()
