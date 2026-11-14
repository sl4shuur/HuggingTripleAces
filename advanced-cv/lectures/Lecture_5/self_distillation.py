"""Models, losses and masks implemented in Lecture 5 (Module_1/Lecture_5/self_distillation.ipynb).

The code is the same as in the notebook cells; it lives in a module so that `train.py` can import it.
Every method is an `nn.Module` with the same interface:

    loss = method(views)            # views: list of augmented versions of the same batch of images
    method.step_end(progress)       # after every optimizer step (EMA update of the teacher), progress in [0, 1]
    method.features(x)              # {"teacher": ..., "student": ...}: one vector per image, for evaluation
    method.stats                    # dict of diagnostics of the last step (teacher entropy, ...)
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lecture_utils import BatchAugment
from vision_transformer import Block, VisionTransformer, get_2d_sincos_pos_embed, resample_pos_embed


# ----------------------------------------------------------------------------------------------------------------------
# Backbone and building blocks
# ----------------------------------------------------------------------------------------------------------------------
class ViTEncoder(VisionTransformer):
    """The ViT of Lecture 1 (ViT-Ti with 8x8 patches by default) as a feature extractor.

    Returns all tokens (B, prefix + N, D), accepts images of any size (positional embeddings are interpolated,
    needed for the small local crops of DINO) and can process a subset of the patches (needed for I-JEPA).
    """

    def __init__(self, img_size=96, patch_size=8, embed_dim=192, depth=12, num_heads=3, cls_token=True):
        super().__init__(img_size=img_size, patch_size=patch_size, num_classes=0, embed_dim=embed_dim, depth=depth,
                         num_heads=num_heads, global_pool="token" if cls_token else "avg", pos_embed_type="sincos")
        self.embed_dim, self.patch_size = embed_dim, patch_size

    def forward(self, x, keep=None):
        """keep: (n,) indices of the patches to process (the same for the whole batch); None = all patches."""
        grid = x.shape[-1] // self.patch_size
        pos = self.pos_embed
        if grid != self.patch_embed.grid_size:
            pos = resample_pos_embed(pos, grid, self.num_prefix_tokens)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            x = self.patch_embed(x)
            if self.cls_token is not None:
                x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
            x = x + pos
            if keep is not None:
                if self.cls_token is not None:
                    keep = torch.cat([keep.new_zeros(1), keep + 1])
                x = x[:, keep]
            x = self.norm(self.blocks(x))
        return x.float()                                   # mixed precision inside, float32 outside

    def pool(self, tokens):
        return tokens[:, 0] if self.cls_token is not None else tokens.mean(dim=1)


@torch.no_grad()
def ema_update(target, online, tau):
    """target <- tau * target + (1 - tau) * online, for all parameters."""
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(tau).add_(po.detach(), alpha=1 - tau)


def frozen_copy(module):
    module = copy.deepcopy(module)
    for p in module.parameters():
        p.requires_grad = False
    return module


def cosine_momentum(base, progress):
    """EMA coefficient that goes from `base` to 1 along a cosine, progress in [0, 1]."""
    return 1 - (1 - base) * (math.cos(math.pi * progress) + 1) / 2


# ----------------------------------------------------------------------------------------------------------------------
# DINO
# ----------------------------------------------------------------------------------------------------------------------
class MultiCrop(nn.Module):
    """DINO views: 2 global crops (large part of the image, full resolution) + n_local small low-resolution crops."""

    def __init__(self, n_local=6, img_size=96, local_size=48, global_scale=(0.4, 1.0), local_scale=(0.05, 0.4)):
        super().__init__()
        self.n_local = n_local
        self.global1 = BatchAugment(img_size, crop_scale=global_scale, blur_p=1.0)
        self.global2 = BatchAugment(img_size, crop_scale=global_scale, blur_p=0.1, solarize_p=0.2)
        self.local = BatchAugment(local_size, crop_scale=local_scale, blur_p=0.5)

    def forward(self, x):
        return [self.global1(x), self.global2(x)] + [self.local(x) for _ in range(self.n_local)]


class DINOHead(nn.Module):
    """MLP -> L2-normalized bottleneck -> cosine similarity to K learnable prototypes. Output: (B, K) logits."""

    def __init__(self, in_dim, out_dim=4096, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, bottleneck_dim))
        self.prototypes = nn.Parameter(torch.randn(out_dim, bottleneck_dim) * 0.02)

    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1)
        return x @ F.normalize(self.prototypes, dim=-1).T      # the weight-normalized last layer of the paper


def dino_loss(student_logits, teacher_logits, center, student_temp=0.1, teacher_temp=0.04):
    """Cross-entropy between the teacher's and the student's distributions over the K prototypes.

    student_logits: list of (B, K), one per view, global views first; teacher_logits: list of (B, K), global views only.
    The teacher is centred (subtract the running mean of its logits) and sharpened (low temperature).
    A view is never compared with itself.
    """
    targets = [F.softmax((t - center) / teacher_temp, dim=-1) for t in teacher_logits]
    log_probs = [F.log_softmax(s / student_temp, dim=-1) for s in student_logits]
    terms = [-(q * log_p).sum(dim=-1).mean() for i, q in enumerate(targets) for j, log_p in enumerate(log_probs) if i != j]
    return torch.stack(terms).mean()


class DINO(nn.Module):
    def __init__(self, encoder, out_dim=4096, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9, ema=0.99,
                 ema_schedule=True, centering=True):
        super().__init__()
        self.student, self.student_head = encoder, DINOHead(encoder.embed_dim, out_dim)
        self.teacher, self.teacher_head = frozen_copy(self.student), frozen_copy(self.student_head)
        self.student_temp, self.teacher_temp, self.center_momentum = student_temp, teacher_temp, center_momentum
        self.ema, self.ema_schedule, self.centering = ema, ema_schedule, centering
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.stats = {}

    def forward(self, views, n_global=2):
        # teacher: global views only, no gradient
        with torch.no_grad():
            teacher_logits = self.teacher_head(self.teacher(torch.cat(views[:n_global]))[:, 0])
        # student: all views; one forward pass per resolution
        student_logits = self.student_head(self.student(torch.cat(views[:n_global]))[:, 0]).chunk(n_global)
        if len(views) > n_global:
            n_local = len(views) - n_global
            student_logits += self.student_head(self.student(torch.cat(views[n_global:]))[:, 0]).chunk(n_local)
        loss = dino_loss(student_logits, teacher_logits.chunk(n_global), self.center, self.student_temp, self.teacher_temp)

        with torch.no_grad():
            if self.centering:                                 # running mean of the teacher logits
                self.center.mul_(self.center_momentum).add_(teacher_logits.mean(dim=0, keepdim=True),
                                                            alpha=1 - self.center_momentum)
            # diagnostics (Fig. 7 of the paper): entropy of the teacher target and KL(teacher || student)
            p_t = F.softmax((teacher_logits.chunk(n_global)[0] - self.center) / self.teacher_temp, dim=-1)
            log_p_s = F.log_softmax(student_logits[1] / self.student_temp, dim=-1)
            entropy = -(p_t * p_t.clamp_min(1e-12).log()).sum(dim=-1).mean()
            self.stats = {"teacher_entropy": entropy.item(), "kl": (-(p_t * log_p_s).sum(dim=-1).mean() - entropy).item(),
                          "prototypes_used": p_t.argmax(dim=-1).unique().numel()}
        return loss

    def step_end(self, progress):
        tau = cosine_momentum(self.ema, progress) if self.ema_schedule else self.ema
        ema_update(self.teacher, self.student, tau)
        ema_update(self.teacher_head, self.student_head, tau)

    @torch.no_grad()
    def features(self, x):
        return {"teacher": self.teacher(x)[:, 0], "student": self.student(x)[:, 0]}


# ----------------------------------------------------------------------------------------------------------------------
# I-JEPA
# ----------------------------------------------------------------------------------------------------------------------
def sample_block_size(grid, scale, aspect, generator=None):
    """Height and width (in patches) of a block that covers `scale` of the grid with the given aspect ratio range."""
    u = torch.rand(2, generator=generator).tolist()
    area = (scale[0] + u[0] * (scale[1] - scale[0])) * grid * grid
    ratio = aspect[0] + u[1] * (aspect[1] - aspect[0])
    h, w = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
    return min(max(h, 1), grid), min(max(w, 1), grid)


def block_indices(grid, h, w, generator=None):
    """Flat indices of the patches of an h x w block at a random position of the grid."""
    top = torch.randint(0, grid - h + 1, (1,), generator=generator).item()
    left = torch.randint(0, grid - w + 1, (1,), generator=generator).item()
    rows, cols = torch.meshgrid(torch.arange(top, top + h), torch.arange(left, left + w), indexing="ij")
    return (rows * grid + cols).flatten()


def sample_ijepa_masks(grid=12, n_targets=4, target_scale=(0.15, 0.2), target_aspect=(0.75, 1.5),
                       context_scale=(0.85, 1.0), min_context=12, generator=None):
    """Multi-block masking of I-JEPA.

    Returns context (n_ctx,): the visible patches, and targets (n_targets, h * w): the blocks to predict.
    All target blocks have the same size and may overlap; the context is one large block minus all targets.
    """
    while True:
        h, w = sample_block_size(grid, target_scale, target_aspect, generator)
        targets = torch.stack([block_indices(grid, h, w, generator) for _ in range(n_targets)])
        ch, cw = sample_block_size(grid, context_scale, (1.0, 1.0), generator)
        context = block_indices(grid, ch, cw, generator)
        context = context[~torch.isin(context, targets.flatten())]
        if len(context) >= min_context:
            return context, targets


def sample_random_masks(grid=12, n_targets=4, target_scale=(0.15, 0.2), generator=None):
    """Ablation: the same number of target patches, but scattered at random instead of forming blocks."""
    n = round((target_scale[0] + torch.rand(1, generator=generator).item() * (target_scale[1] - target_scale[0])) * grid * grid)
    perm = torch.randperm(grid * grid, generator=generator)
    return perm[n_targets * n:].sort().values, perm[:n_targets * n].view(n_targets, n)


def patchify(x, patch_size):
    """(B, 3, H, W) -> (B, N, patch_size**2 * 3) pixel values of every patch, in raster order."""
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(b, (h // patch_size) * (w // patch_size), -1)


class Predictor(nn.Module):
    """A narrow ViT that predicts the representation of the target patches from the context tokens.

    The targets enter as a shared learnable mask token plus the positional embedding of the patch to predict:
    the predictor is told *where* to predict, the context tells it *what* is in the image.
    """

    def __init__(self, num_patches, embed_dim=192, out_dim=192, pred_dim=96, depth=4, num_heads=3):
        super().__init__()
        self.embed = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.register_buffer("pos_embed", get_2d_sincos_pos_embed(pred_dim, int(math.sqrt(num_patches))), persistent=False)
        self.blocks = nn.Sequential(*[Block(pred_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim, eps=1e-6)
        self.proj = nn.Linear(pred_dim, out_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(VisionTransformer._init_weights)

    def forward(self, context_tokens, context, targets):
        """context_tokens: (B, n_ctx, D); context: (n_ctx,); targets: (T, n). Returns (B * T, n, out_dim)."""
        b, (t, n) = context_tokens.shape[0], targets.shape
        with torch.autocast(context_tokens.device.type, dtype=torch.bfloat16, enabled=context_tokens.is_cuda):
            ctx = self.embed(context_tokens) + self.pos_embed[context]                 # (B, n_ctx, d)
            queries = self.mask_token + self.pos_embed[targets]                        # (T, n, d)
            x = torch.cat([ctx.repeat_interleave(t, dim=0), queries.repeat(b, 1, 1)], dim=1)    # one sequence per block
            x = self.proj(self.norm(self.blocks(x))[:, -n:])
        return x.float()


class IJEPA(nn.Module):
    def __init__(self, encoder, ema=0.996, ema_schedule=True, target="latent", mask="multiblock", pred_dim=96,
                 pred_depth=4):
        super().__init__()
        assert encoder.cls_token is None and target in ("latent", "pixels") and mask in ("multiblock", "random")
        self.student, self.teacher = encoder, frozen_copy(encoder)        # context encoder and target encoder
        self.grid = encoder.patch_embed.grid_size
        out_dim = encoder.embed_dim if target == "latent" else 3 * encoder.patch_size ** 2
        self.predictor = Predictor(self.grid ** 2, encoder.embed_dim, out_dim, pred_dim, pred_depth)
        self.ema, self.ema_schedule, self.target, self.mask = ema, ema_schedule, target, mask
        self.stats = {}

    def forward(self, views):
        x = views[0]                                                     # a single view: no second augmentation
        context, targets = sample_ijepa_masks(self.grid) if self.mask == "multiblock" else sample_random_masks(self.grid)
        context, targets = context.to(x.device), targets.to(x.device)    # the same masks for the whole batch
        with torch.no_grad():
            if self.target == "latent":                                  # the target encoder sees the whole image
                h = self.teacher(x)
                h = F.layer_norm(h, (h.shape[-1],))
            else:                                                        # ablation: normalized pixels, as in MAE
                h = patchify(x, self.student.patch_size)
                h = (h - h.mean(dim=-1, keepdim=True)) / (h.var(dim=-1, keepdim=True) + 1e-6).sqrt()
            target_tokens = h[:, targets].flatten(0, 1)                  # (B * T, n, D)
            self.stats = {"target_std": h.std(dim=0).mean().item(), "context_patches": len(context)}
        context_tokens = self.student(x, keep=context)                   # the context encoder sees only the context
        prediction = self.predictor(context_tokens, context, targets)
        return F.smooth_l1_loss(prediction, target_tokens)

    def step_end(self, progress):
        ema_update(self.teacher, self.student, cosine_momentum(self.ema, progress) if self.ema_schedule else self.ema)

    @torch.no_grad()
    def features(self, x):
        return {"teacher": self.teacher(x).mean(dim=1), "student": self.student(x).mean(dim=1)}


# ----------------------------------------------------------------------------------------------------------------------
# Supervised reference
# ----------------------------------------------------------------------------------------------------------------------
class Supervised(nn.Module):
    needs_labels = True

    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.student, self.head = encoder, nn.Linear(encoder.embed_dim, num_classes)
        self.stats = {}

    def forward(self, views, labels):
        return F.cross_entropy(self.head(self.student.pool(self.student(views[0]))), labels, label_smoothing=0.1)

    def step_end(self, progress):
        pass

    @torch.no_grad()
    def features(self, x):
        h = self.student.pool(self.student(x))
        return {"teacher": h, "student": h}


METHODS = {"dino": DINO, "ijepa": IJEPA, "supervised": Supervised}
