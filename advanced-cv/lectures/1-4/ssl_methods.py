"""Augmentations, losses and methods implemented in Lecture 4 (Module_1/Lecture_4/contrastive_and_non_contrastive_ssl.ipynb).

The code is the same as in the notebook cells; it lives in a module so that `ssl_train.py` can import it.
Every method is an `nn.Module` with the same interface:

    loss = method(x1, x2)          # x1, x2: two augmented views of the same batch of images
    method.step_end(progress)      # after every optimizer step (momentum updates), progress in [0, 1]
    h = method.backbone(x)         # the representation we keep for downstream tasks
    z = method.projector(h)        # the embedding the loss is computed on
"""
import copy
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_transformer import VisionTransformer


# ----------------------------------------------------------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------------------------------------------------------
class BatchAugment(nn.Module):
    """The SimCLR / MoCo v2 augmentation pipeline, applied to a whole batch on the GPU.

    Input: images (B, 3, H, W) with values in [0, 255]. Output: float images (B, 3, img_size, img_size) in [0, 1].
    Every image gets its own random parameters. Any step can be switched off (crop_scale=None, *_p=0).
    """

    def __init__(self, img_size=96, crop_scale=(0.2, 1.0), crop_ratio=(3 / 4, 4 / 3), flip_p=0.5,
                 jitter=(0.4, 0.4, 0.4, 0.1), jitter_p=0.8, gray_p=0.2, blur_p=0.5, blur_sigma=(0.1, 2.0),
                 solarize_p=0.0):
        super().__init__()
        self.img_size, self.crop_scale, self.crop_ratio, self.flip_p = img_size, crop_scale, crop_ratio, flip_p
        self.jitter, self.jitter_p, self.gray_p = jitter, jitter_p, gray_p
        self.blur_p, self.blur_sigma, self.solarize_p = blur_p, blur_sigma, solarize_p
        self.blur_kernel = 2 * int(0.05 * img_size) + 1                  # ~10% of the image size, odd
        self.register_buffer("gray_weights", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        rgb_to_yiq = torch.tensor([[0.299, 0.587, 0.114], [0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]])
        self.register_buffer("rgb_to_yiq", rgb_to_yiq)
        self.register_buffer("yiq_to_rgb", torch.linalg.inv(rgb_to_yiq))

    @staticmethod
    def uniform(n, low, high, device):
        return torch.rand(n, device=device) * (high - low) + low

    @staticmethod
    def where(mask, a, b):
        return torch.where(mask.view(-1, 1, 1, 1), a, b)

    def gray(self, x):
        return (x * self.gray_weights).sum(dim=1, keepdim=True)

    def crop_and_flip(self, x):
        """Random resized crop + horizontal flip as one affine resampling."""
        b, dev = x.shape[0], x.device
        if self.crop_scale is None:
            w = h = torch.ones(b, device=dev)
        else:
            area = self.uniform(b, *self.crop_scale, dev)               # fraction of the image area
            ratio = self.uniform(b, math.log(self.crop_ratio[0]), math.log(self.crop_ratio[1]), dev).exp()
            w, h = (area * ratio).sqrt().clamp(max=1), (area / ratio).sqrt().clamp(max=1)   # fractions of W and H
        cx, cy = self.uniform(b, -1, 1, dev) * (1 - w), self.uniform(b, -1, 1, dev) * (1 - h)   # centre of the crop
        flip = torch.where(torch.rand(b, device=dev) < self.flip_p, -1.0, 1.0)
        theta = torch.zeros(b, 2, 3, device=dev)
        theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = w * flip, cx, h, cy
        grid = F.affine_grid(theta, (b, 3, self.img_size, self.img_size), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    def color_jitter(self, x):
        b, dev = x.shape[0], x.device
        brightness, contrast, saturation, hue = self.jitter
        for op in torch.randperm(4).tolist():                           # random order, as in torchvision
            if op == 0 and brightness > 0:
                x = x * self.uniform(b, 1 - brightness, 1 + brightness, dev).view(-1, 1, 1, 1)
            elif op == 1 and contrast > 0:
                mean = self.gray(x).mean(dim=(1, 2, 3), keepdim=True)
                x = (x - mean) * self.uniform(b, 1 - contrast, 1 + contrast, dev).view(-1, 1, 1, 1) + mean
            elif op == 2 and saturation > 0:
                g = self.gray(x)
                x = (x - g) * self.uniform(b, 1 - saturation, 1 + saturation, dev).view(-1, 1, 1, 1) + g
            elif op == 3 and hue > 0:                                   # rotate the chroma plane of YIQ
                angle = self.uniform(b, -hue, hue, dev) * 2 * math.pi
                rot = torch.zeros(b, 3, 3, device=dev)
                rot[:, 0, 0] = 1
                rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = angle.cos(), -angle.sin(), angle.sin(), angle.cos()
                x = torch.einsum("bij,bjhw->bihw", self.yiq_to_rgb @ rot @ self.rgb_to_yiq, x)
            x = x.clamp(0, 1)
        return x

    def gaussian_blur(self, x):
        """Separable Gaussian blur with an individual sigma per image (one grouped convolution per axis)."""
        b, c, h, w = x.shape
        k = self.blur_kernel
        sigma = self.uniform(b, *self.blur_sigma, x.device).view(-1, 1)
        offsets = torch.arange(k, device=x.device) - k // 2
        kernel = torch.exp(-offsets.pow(2) / (2 * sigma.pow(2)))        # (B, k)
        kernel = (kernel / kernel.sum(dim=1, keepdim=True)).repeat_interleave(c, dim=0)   # (B * C, k)
        x = F.pad(x.reshape(1, b * c, h, w), (k // 2, k // 2, k // 2, k // 2), mode="reflect")
        x = F.conv2d(x, kernel.view(b * c, 1, 1, k), groups=b * c)
        x = F.conv2d(x, kernel.view(b * c, 1, k, 1), groups=b * c)
        return x.reshape(b, c, h, w)

    @torch.no_grad()
    def forward(self, x):
        b, dev = x.shape[0], x.device
        x = x.float() / 255
        if self.crop_scale is not None or self.flip_p > 0 or x.shape[-1] != self.img_size:
            x = self.crop_and_flip(x)
        if self.jitter_p > 0:
            x = self.where(torch.rand(b, device=dev) < self.jitter_p, self.color_jitter(x), x)
        if self.gray_p > 0:
            x = self.where(torch.rand(b, device=dev) < self.gray_p, self.gray(x).expand(-1, 3, -1, -1), x)
        if self.blur_p > 0:
            x = self.where(torch.rand(b, device=dev) < self.blur_p, self.gaussian_blur(x), x)
        if self.solarize_p > 0:
            x = self.where(torch.rand(b, device=dev) < self.solarize_p, torch.where(x >= 0.5, 1 - x, x), x)
        return x.clamp(0, 1)


# ----------------------------------------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------------------------------------
def resnet18_backbone():
    """ResNet-18 for 96x96 images: 3x3 stride-1 stem instead of 7x7 stride-2. Output: (B, 512)."""
    model = timm.create_model("resnet18", pretrained=False, num_classes=0)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    return model, 512


def vit_tiny_backbone(img_size=96, patch_size=8):
    """Our ViT from Lecture 1: ViT-Ti with 8x8 patches (144 tokens), fixed sin-cos positions. Output: (B, 192)."""
    model = VisionTransformer(img_size=img_size, patch_size=patch_size, num_classes=0, embed_dim=192, depth=12,
                              num_heads=3, global_pool="token", pos_embed_type="sincos")
    return model, 192


BACKBONES = {"resnet18": resnet18_backbone, "vit_tiny": vit_tiny_backbone}


def mlp(dims, batch_norm=True, last_bn=False):
    """dims = [in, hidden, ..., out]: Linear -> (BN) -> ReLU for hidden layers, Linear -> (BN) for the last one."""
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(dims[-2], dims[-1], bias=not last_bn))
    if last_bn:
        layers.append(nn.BatchNorm1d(dims[-1]))
    return nn.Sequential(*layers)


def off_diagonal(m):
    """All off-diagonal elements of a square matrix as a flat vector."""
    n = m.shape[0]
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


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


class SSLMethod(nn.Module):
    """Common interface of all methods in this lecture."""

    def embed(self, x):
        h = self.backbone(x)
        return h, self.projector(h)

    def step_end(self, progress):
        pass


# ----------------------------------------------------------------------------------------------------------------------
# Contrastive: SimCLR
# ----------------------------------------------------------------------------------------------------------------------
def nt_xent(z1, z2, temperature=0.2):
    """NT-Xent (SimCLR): for each of the 2N views the positive is the other view of the same image,
    the negatives are the remaining 2N - 2 views in the batch."""
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)                         # (2N, D), unit norm
    sim = z @ z.T / temperature                                         # (2N, 2N) cosine similarity / tau
    sim.fill_diagonal_(float("-inf"))                                   # a view is not its own negative
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)   # index of the positive
    return F.cross_entropy(sim, targets)


def dcl_loss(z1, z2, temperature=0.2):
    """Decoupled contrastive loss (Yeh et al., 2021): NT-Xent with the positive removed from the denominator."""
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / temperature
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
    pos = sim.gather(1, targets[:, None]).squeeze(1)
    neg = sim.clone()
    neg.fill_diagonal_(float("-inf"))
    neg.scatter_(1, targets[:, None], float("-inf"))                    # drop the positive from the denominator
    return (-pos + torch.logsumexp(neg, dim=1)).mean()


class SimCLR(SSLMethod):
    def __init__(self, backbone, feat_dim, proj_dim=128, hidden_dim=2048, temperature=0.2, projector="mlp",
                 loss="nt_xent"):
        super().__init__()
        self.backbone = backbone
        self.projector = {"mlp": lambda: mlp([feat_dim, hidden_dim, proj_dim]),
                          "linear": lambda: nn.Linear(feat_dim, proj_dim),
                          "none": lambda: nn.Identity()}[projector]()
        self.temperature = temperature
        self.loss = loss                       # "nt_xent", or "dcl" for the decoupled loss introduced later

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        loss_fn = dcl_loss if self.loss == "dcl" else nt_xent
        return loss_fn(z1, z2, self.temperature)


# ----------------------------------------------------------------------------------------------------------------------
# Contrastive: MoCo (v2 with a queue, v3 without)
# ----------------------------------------------------------------------------------------------------------------------
class MoCo(SSLMethod):
    """queue_size > 0: MoCo v2 (negatives come from a queue of past keys).
    queue_size = 0 and predictor=True: MoCo v3 (negatives are the other keys of the batch, extra predictor)."""

    def __init__(self, backbone, feat_dim, proj_dim=128, hidden_dim=2048, temperature=0.2, queue_size=16384,
                 momentum=0.99, predictor=False):
        super().__init__()
        self.backbone = backbone
        self.projector = mlp([feat_dim, hidden_dim, proj_dim])
        self.predictor = mlp([proj_dim, hidden_dim, proj_dim]) if predictor else nn.Identity()
        self.key_backbone, self.key_projector = frozen_copy(self.backbone), frozen_copy(self.projector)
        self.temperature, self.momentum = temperature, momentum
        if queue_size > 0:
            self.register_buffer("queue", F.normalize(torch.randn(queue_size, proj_dim), dim=1))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        else:
            self.queue = None

    def contrast(self, q, k):
        if self.queue is None:                                          # in-batch negatives
            logits = q @ k.T                                            # (N, N), positives on the diagonal
            targets = torch.arange(len(q), device=q.device)
        else:                                                           # negatives from the queue
            l_pos = (q * k).sum(dim=1, keepdim=True)                    # (N, 1)
            l_neg = q @ self.queue.clone().detach().T                   # (N, K)
            logits = torch.cat([l_pos, l_neg], dim=1)                   # the positive is class 0
            targets = torch.zeros(len(q), dtype=torch.long, device=q.device)
        return F.cross_entropy(logits / self.temperature, targets)

    @torch.no_grad()
    def enqueue(self, keys):
        idx = (self.queue_ptr + torch.arange(len(keys), device=keys.device)) % len(self.queue)
        self.queue[idx] = keys                                          # overwrite the oldest keys (FIFO)
        self.queue_ptr[0] = (self.queue_ptr[0] + len(keys)) % len(self.queue)

    def forward(self, x1, x2):
        q1 = F.normalize(self.predictor(self.projector(self.backbone(x1))), dim=1)     # queries: gradient
        q2 = F.normalize(self.predictor(self.projector(self.backbone(x2))), dim=1)
        with torch.no_grad():                                                           # keys: no gradient
            k1 = F.normalize(self.key_projector(self.key_backbone(x1)), dim=1)
            k2 = F.normalize(self.key_projector(self.key_backbone(x2)), dim=1)
        loss = 0.5 * (self.contrast(q1, k2) + self.contrast(q2, k1))
        if self.queue is not None:
            self.enqueue(torch.cat([k1, k2]))
        return loss

    def step_end(self, progress):
        ema_update(self.key_backbone, self.backbone, self.momentum)
        ema_update(self.key_projector, self.projector, self.momentum)


# ----------------------------------------------------------------------------------------------------------------------
# Contrastive with nearest-neighbour positives: NNCLR
# ----------------------------------------------------------------------------------------------------------------------
class NNCLR(SSLMethod):
    """The positive of view 2 is not view 1 itself but its nearest neighbour in a support set of past embeddings."""

    def __init__(self, backbone, feat_dim, proj_dim=256, hidden_dim=2048, temperature=0.2, queue_size=16384):
        super().__init__()
        self.backbone = backbone
        self.projector = mlp([feat_dim, hidden_dim, hidden_dim, proj_dim], last_bn=True)
        self.predictor = mlp([proj_dim, hidden_dim, proj_dim])
        self.temperature = temperature
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, proj_dim), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def nearest_neighbour(self, z):
        return self.queue[(z @ self.queue.T).argmax(dim=1)]

    @torch.no_grad()
    def enqueue(self, keys):
        idx = (self.queue_ptr + torch.arange(len(keys), device=keys.device)) % len(self.queue)
        self.queue[idx] = keys
        self.queue_ptr[0] = (self.queue_ptr[0] + len(keys)) % len(self.queue)

    def contrast(self, nn, p):
        logits = nn @ p.T / self.temperature                            # (N, N), positives on the diagonal
        return F.cross_entropy(logits, torch.arange(len(p), device=p.device))

    def forward(self, x1, x2):
        z1 = F.normalize(self.projector(self.backbone(x1)), dim=1)
        z2 = F.normalize(self.projector(self.backbone(x2)), dim=1)
        p1, p2 = F.normalize(self.predictor(z1), dim=1), F.normalize(self.predictor(z2), dim=1)
        nn1, nn2 = self.nearest_neighbour(z1.detach()), self.nearest_neighbour(z2.detach())
        loss = 0.5 * (self.contrast(nn1, p2) + self.contrast(nn2, p1))
        self.enqueue(z1.detach())
        return loss


# ----------------------------------------------------------------------------------------------------------------------
# Non-contrastive, asymmetric: BYOL and SimSiam
# ----------------------------------------------------------------------------------------------------------------------
def byol_loss(p, z):
    """Squared distance between unit vectors = 2 - 2 cos(p, z). No gradient through the target z."""
    return (2 - 2 * F.cosine_similarity(p, z.detach(), dim=1)).mean()


class BYOL(SSLMethod):
    def __init__(self, backbone, feat_dim, proj_dim=256, hidden_dim=2048, ema=0.99, ema_schedule=True,
                 predictor=True, batch_norm=True):
        super().__init__()
        self.backbone = backbone                                         # online network: backbone + projector
        self.projector = mlp([feat_dim, hidden_dim, proj_dim], batch_norm=batch_norm)
        self.predictor = mlp([proj_dim, hidden_dim, proj_dim], batch_norm=batch_norm) if predictor else nn.Identity()
        self.target_backbone, self.target_projector = frozen_copy(self.backbone), frozen_copy(self.projector)
        self.ema, self.ema_schedule = ema, ema_schedule

    def forward(self, x1, x2):
        p1 = self.predictor(self.projector(self.backbone(x1)))
        p2 = self.predictor(self.projector(self.backbone(x2)))
        with torch.no_grad():
            t1 = self.target_projector(self.target_backbone(x1))
            t2 = self.target_projector(self.target_backbone(x2))
        return 0.5 * (byol_loss(p1, t2) + byol_loss(p2, t1))

    def step_end(self, progress):
        # cosine schedule of the momentum from `ema` to 1 (the target moves slower and slower)
        tau = 1 - (1 - self.ema) * (math.cos(math.pi * progress) + 1) / 2 if self.ema_schedule else self.ema
        ema_update(self.target_backbone, self.backbone, tau)
        ema_update(self.target_projector, self.projector, tau)


def negative_cosine(p, z, stop_grad=True):
    return -F.cosine_similarity(p, z.detach() if stop_grad else z, dim=1).mean()


class SimSiam(SSLMethod):
    def __init__(self, backbone, feat_dim, proj_dim=2048, hidden_dim=2048, pred_hidden_dim=512, stop_grad=True,
                 predictor=True):
        super().__init__()
        self.backbone = backbone
        self.projector = mlp([feat_dim, hidden_dim, hidden_dim, proj_dim], last_bn=True)
        self.predictor = mlp([proj_dim, pred_hidden_dim, proj_dim]) if predictor else nn.Identity()
        self.stop_grad = stop_grad

    def forward(self, x1, x2):
        z1, z2 = self.projector(self.backbone(x1)), self.projector(self.backbone(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return 0.5 * (negative_cosine(p1, z2, self.stop_grad) + negative_cosine(p2, z1, self.stop_grad))


# ----------------------------------------------------------------------------------------------------------------------
# Non-contrastive, symmetric: redundancy reduction (Barlow Twins, VICReg)
# ----------------------------------------------------------------------------------------------------------------------
def barlow_twins_loss(z1, z2, lambd=5e-3):
    """Make the cross-correlation matrix of the two views' embeddings equal to the identity."""
    n, d = z1.shape
    z1 = (z1 - z1.mean(0)) / z1.std(0).clamp_min(1e-6)                  # standardize every dimension over the batch
    z2 = (z2 - z2.mean(0)) / z2.std(0).clamp_min(1e-6)
    c = z1.T @ z2 / n                                                   # (D, D) cross-correlation
    on_diag = (torch.diagonal(c) - 1).pow(2).sum()                      # invariance: c_ii -> 1
    off_diag = off_diagonal(c).pow(2).sum()                             # redundancy reduction: c_ij -> 0
    return on_diag + lambd * off_diag


class BarlowTwins(SSLMethod):
    def __init__(self, backbone, feat_dim, proj_dim=2048, hidden_dim=2048, lambd=5e-3):
        super().__init__()
        self.backbone = backbone
        self.projector = mlp([feat_dim, hidden_dim, hidden_dim, proj_dim])
        self.lambd = lambd

    def forward(self, x1, x2):
        return barlow_twins_loss(self.projector(self.backbone(x1)), self.projector(self.backbone(x2)), self.lambd)


def vicreg_loss(z1, z2, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0):
    n, d = z1.shape
    invariance = F.mse_loss(z1, z2)                                     # the two views should match
    z1, z2 = z1 - z1.mean(0), z2 - z2.mean(0)
    std1, std2 = (z1.var(0) + 1e-4).sqrt(), (z2.var(0) + 1e-4).sqrt()
    variance = 0.5 * (F.relu(1 - std1).mean() + F.relu(1 - std2).mean())    # every dimension keeps std >= 1
    cov1, cov2 = z1.T @ z1 / (n - 1), z2.T @ z2 / (n - 1)
    covariance = (off_diagonal(cov1).pow(2).sum() + off_diagonal(cov2).pow(2).sum()) / d   # decorrelate dimensions
    return sim_coeff * invariance + var_coeff * variance + cov_coeff * covariance


class VICReg(SSLMethod):
    def __init__(self, backbone, feat_dim, proj_dim=2048, hidden_dim=2048, sim_coeff=25.0, var_coeff=25.0,
                 cov_coeff=1.0):
        super().__init__()
        self.backbone = backbone
        self.projector = mlp([feat_dim, hidden_dim, hidden_dim, proj_dim])
        self.coeffs = (sim_coeff, var_coeff, cov_coeff)

    def forward(self, x1, x2):
        return vicreg_loss(self.projector(self.backbone(x1)), self.projector(self.backbone(x2)), *self.coeffs)


# ----------------------------------------------------------------------------------------------------------------------
# Supervised reference (uses the labels and one view)
# ----------------------------------------------------------------------------------------------------------------------
class Supervised(SSLMethod):
    needs_labels = True

    def __init__(self, backbone, feat_dim, num_classes=10, label_smoothing=0.1):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Linear(feat_dim, num_classes)               # the classifier plays the role of the head
        self.label_smoothing = label_smoothing

    def forward(self, x1, x2, y):
        return F.cross_entropy(self.projector(self.backbone(x1)), y, label_smoothing=self.label_smoothing)


METHODS = {"simclr": SimCLR, "moco": MoCo, "nnclr": NNCLR, "byol": BYOL, "simsiam": SimSiam,
           "barlow_twins": BarlowTwins, "vicreg": VICReg, "supervised": Supervised}
