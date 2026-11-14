"""Vision Transformer implemented from scratch in Lecture 1 (Module_1/Lecture_1/vit_from_scratch.ipynb).

Module names follow `timm`, so pretrained `timm` ViT checkpoints can be loaded with `load_state_dict(strict=True)`.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbed(nn.Module):
    """Image -> sequence of patch tokens. (B, C, H, W) -> (B, N, D)"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by the patch size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                        # (B, D, H/P, W/P)
        return x.flatten(2).transpose(1, 2)     # (B, N, D)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.store_attn = False
        self.attn_map = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                  # each (B, h, N, d)

        if self.store_attn:
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            self.attn_map = attn.detach()
            x = self.attn_drop(attn) @ v
        else:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
        return x * mask / keep_prob


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, temperature: float = 10000.0) -> torch.Tensor:
    """Returns (grid_size**2, embed_dim) fixed positional embeddings in raster order."""
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos embeddings"
    rows, cols = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing="ij")
    omega = 1.0 / temperature ** (torch.arange(embed_dim // 4) / (embed_dim // 4))
    r = rows.flatten()[:, None].float() * omega[None]
    c = cols.flatten()[:, None].float() * omega[None]
    return torch.cat([c.sin(), c.cos(), r.sin(), r.cos()], dim=1)


def resample_pos_embed(pos_embed: torch.Tensor, new_grid: int, num_prefix_tokens: int = 1) -> torch.Tensor:
    """(1, prefix + old_grid**2, D) -> (1, prefix + new_grid**2, D)"""
    prefix, grid_tokens = pos_embed[:, :num_prefix_tokens], pos_embed[:, num_prefix_tokens:]
    old_grid = int(math.sqrt(grid_tokens.shape[1]))
    grid_tokens = rearrange(grid_tokens, "1 (h w) d -> 1 d h w", h=old_grid, w=old_grid)
    grid_tokens = F.interpolate(grid_tokens, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    grid_tokens = rearrange(grid_tokens, "1 d h w -> 1 (h w) d")
    return torch.cat([prefix, grid_tokens], dim=1)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        global_pool="token",      # "token" ([CLS]) or "avg" (GAP over patch tokens)
        pos_embed_type="learn",   # "learn" or "sincos"
    ):
        super().__init__()
        assert global_pool in ("token", "avg")
        self.global_pool = global_pool
        self.num_prefix_tokens = 1 if global_pool == "token" else 0

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if global_pool == "token" else None
        if pos_embed_type == "learn":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_prefix_tokens + num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            pe = get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid_size)
            pe = torch.cat([torch.zeros(self.num_prefix_tokens, embed_dim), pe])[None]
            self.register_buffer("pos_embed", pe, persistent=False)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dpr[i]) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        return self.norm(x)

    def pool(self, x):
        return x[:, 0] if self.global_pool == "token" else x[:, self.num_prefix_tokens:].mean(dim=1)

    def forward(self, x):
        return self.head(self.pool(self.forward_features(x)))

    def set_store_attn(self, flag: bool = True):
        for blk in self.blocks:
            blk.attn.store_attn = flag
            blk.attn.attn_map = None

    def get_attn_maps(self):
        return [blk.attn.attn_map for blk in self.blocks]
