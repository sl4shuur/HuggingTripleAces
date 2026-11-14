import io
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int = 95431):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------------------------------------------------
# STL-10: 100k unlabeled + 5k labeled train + 8k labeled test images, 96x96, 10 classes
# ----------------------------------------------------------------------------------------------------------------------
STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
STL10_MEAN, STL10_STD = (0.4404, 0.4269, 0.3855), (0.2687, 0.2614, 0.2690)
STL10_SPLITS = {"unlabeled": 100_000, "train": 5_000, "test": 8_000}
# HuggingFace Hub mirror with the official splits (the Stanford server is often very slow)
STL10_HF_REPO = "jxie/stl10"


def _stl10_from_hf(root, split):
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from PIL import Image

    local = snapshot_download(STL10_HF_REPO, repo_type="dataset", local_dir=os.path.join(root, "stl10_hf"),
                              allow_patterns=[f"data/{split}-*.parquet"])
    files = sorted(f for f in os.listdir(os.path.join(local, "data")) if f.startswith(f"{split}-"))
    images, labels = [], []
    for f in files:
        table = pq.read_table(os.path.join(local, "data", f)).to_pydict()
        images += [np.asarray(Image.open(io.BytesIO(row["bytes"])).convert("RGB")) for row in table["image"]]
        labels += table["label"]
    # The mirror's class names are the original labels "1".."10" sorted as strings ("1", "10", "2", ...);
    # map its indices back to the official order used by torchvision (STL10_CLASSES).
    to_official = np.array([0, 9, 1, 2, 3, 4, 5, 6, 7, 8])
    return np.stack(images).transpose(0, 3, 1, 2), to_official[np.asarray(labels, dtype=np.int64)]


def _stl10_from_torchvision(root, split):
    import torchvision

    ds = torchvision.datasets.STL10(root, split=split, download=True)
    return ds.data, np.asarray(ds.labels, dtype=np.int64)


def load_stl10(root, split, source="hf"):
    """Returns (images uint8 (N, 3, 96, 96) memory-mapped, labels int64 (N,)); labels are -1 for the unlabeled split.

    The first call downloads the split (`source="hf"`: HuggingFace mirror, `source="torchvision"`: the official
    Stanford archive) and caches it as .npy files in `<root>/stl10_npy`, so that later calls and DataLoader workers
    share the same memory-mapped arrays.
    """
    assert split in STL10_SPLITS
    cache = os.path.join(root, "stl10_npy")
    x_path, y_path = os.path.join(cache, f"{split}_x.npy"), os.path.join(cache, f"{split}_y.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        os.makedirs(cache, exist_ok=True)
        x, y = (_stl10_from_hf if source == "hf" else _stl10_from_torchvision)(root, split)
        assert x.shape == (STL10_SPLITS[split], 3, 96, 96) and x.dtype == np.uint8, (x.shape, x.dtype)
        if split == "unlabeled":
            y = np.full(len(x), -1, dtype=np.int64)
        np.save(x_path + ".tmp.npy", x)
        np.save(y_path, y)
        os.replace(x_path + ".tmp.npy", x_path)
    return np.load(x_path, mmap_mode="r"), np.load(y_path)


class Normalize(nn.Module):
    """Float images (B, 3, H, W) in [0, 1] -> standardized with the STL-10 channel statistics."""

    def __init__(self, mean=STL10_MEAN, std=STL10_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation of frozen features (the protocols of Lecture 3)
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def knn_classify(train_x, train_y, test_x, num_classes, k=20, T=0.07, batch_size=1024):
    """Weighted k-NN on cosine similarity (DINO protocol). Returns predicted labels for test_x."""
    train_x, test_x = F.normalize(train_x.float(), dim=1), F.normalize(test_x.float(), dim=1)
    k = min(k, len(train_x))
    preds = []
    for i in range(0, len(test_x), batch_size):
        sims = test_x[i:i + batch_size] @ train_x.T                    # (b, N_train) cosine similarities
        top_sims, top_idx = sims.topk(k, dim=1)
        weights = (top_sims / T).exp()
        votes = torch.zeros(len(sims), num_classes, device=sims.device)
        votes.scatter_add_(1, train_y[top_idx], weights)               # sum of weights per class
        preds.append(votes.argmax(dim=1))
    return torch.cat(preds)


def knn_accuracy(train_x, train_y, test_x, test_y, num_classes=10, **kwargs):
    return (knn_classify(train_x, train_y, test_x, num_classes, **kwargs) == test_y).float().mean().item()


def fit_logreg(x, y, num_classes, weight_decay, max_iter=300):
    """Multinomial logistic regression with L2 penalty, solved by full-batch L-BFGS."""
    torch.manual_seed(0)
    clf = nn.Linear(x.shape[1], num_classes).to(x.device)
    opt = torch.optim.LBFGS(clf.parameters(), lr=1.0, max_iter=max_iter, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(clf(x), y) + weight_decay * clf.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return clf


def standardize(train, *others):
    mean, std = train.mean(0, keepdim=True), train.std(0, keepdim=True).clamp_min(1e-6)
    return [(t - mean) / std for t in (train, *others)]


def linear_probe(train_x, train_y, test_x, test_y, num_classes=10, val_fraction=0.2,
                 weight_decays=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1), seed=95431):
    """Lecture 3 protocol: standardize, select weight decay on a held-out part of train, refit on all of train,
    report test accuracy once."""
    train_x, test_x = train_x.float(), test_x.float()
    perm = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(seed)).to(train_x.device)
    n_val = int(len(train_x) * val_fraction)
    va, tr = perm[:n_val], perm[n_val:]
    xs_tr, xs_va = standardize(train_x[tr], train_x[va])
    val_acc = {}
    for wd in weight_decays:
        clf = fit_logreg(xs_tr, train_y[tr], num_classes, wd)
        with torch.no_grad():
            val_acc[wd] = (clf(xs_va).argmax(1) == train_y[va]).float().mean().item()
    best_wd = max(val_acc, key=val_acc.get)
    xs_train, xs_test = standardize(train_x, test_x)
    clf = fit_logreg(xs_train, train_y, num_classes, best_wd)
    with torch.no_grad():
        test_acc = (clf(xs_test).argmax(1) == test_y).float().mean().item()
    return {"test_acc": test_acc, "best_wd": best_wd, "best_val_acc": val_acc[best_wd]}


# ----------------------------------------------------------------------------------------------------------------------
# Label-free diagnostics of an embedding matrix Z (N, D)
# ----------------------------------------------------------------------------------------------------------------------
def singular_values(z):
    """Singular values of the centred, L2-normalized embeddings, divided by sqrt(N - 1).

    Their squares are the eigenvalues of the covariance matrix of z / ||z||, so they sum to at most 1:
    1 / sqrt(D) each for an isotropic embedding, all ~0 for a complete collapse, a few large ones and the rest ~0
    for a dimensional collapse.
    """
    z = F.normalize(z.double(), dim=1)
    return torch.linalg.svdvals(z - z.mean(0, keepdim=True)) / (len(z) - 1) ** 0.5


def rankme(z, eps=1e-7):
    """Effective rank (RankMe, Garrido et al., 2023): exp of the entropy of the normalized singular values."""
    s = torch.linalg.svdvals(z.double())
    p = s / s.sum().clamp_min(1e-12) + eps
    return torch.exp(-(p * p.log()).sum()).item()


def embedding_std(z):
    """Mean per-dimension std of the L2-normalized embeddings, rescaled by sqrt(D).

    ~1 if the embeddings are spread isotropically over the sphere, 0 if they all coincide (complete collapse).
    This is the quantity plotted in Fig. 2 of the SimSiam paper.
    """
    z = F.normalize(z.float(), dim=1)
    return (z.std(dim=0).mean() * z.shape[1] ** 0.5).item()


def alignment(z1, z2, alpha=2):
    """Wang & Isola (2020): mean distance between the L2-normalized embeddings of positive pairs (lower = better)."""
    z1, z2 = F.normalize(z1.float(), dim=1), F.normalize(z2.float(), dim=1)
    return (z1 - z2).norm(dim=1).pow(alpha).mean().item()


def uniformity(z, t=2, max_points=4096):
    """Wang & Isola (2020): log of the mean Gaussian potential between all pairs (lower = more uniform)."""
    z = F.normalize(z.float(), dim=1)[:max_points]
    return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().log().item()


# ----------------------------------------------------------------------------------------------------------------------
# Views (the batched GPU augmentation pipeline of Lecture 4)
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
# Figures
# ----------------------------------------------------------------------------------------------------------------------
def draw_self_distillation(figsize=(16, 5.6)):
    """Schematic of the two methods of the lecture: DINO (left) and I-JEPA (right)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    blue, grey, orange, green, yellow = "#cfe2f3", "#e6e6e6", "#fde5cd", "#d9ead3", "#fff2cc"
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def box(ax, x, y, text, color, w=2.3, h=0.62):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04", fc=color, ec="#444444"))
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5)

    def arrow(ax, x0, y0, x1, y1, **kwargs):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#444444", **kwargs))

    def stop_grad(ax, x, y):
        ax.plot([x - 0.3, x + 0.3], [y - 0.02, y + 0.18], color="crimson", lw=2.5)
        ax.plot([x - 0.3, x + 0.3], [y - 0.17, y + 0.03], color="crimson", lw=2.5)
        ax.text(x + 0.42, y, "stop-grad", color="crimson", fontsize=8.5, va="center")

    def ema(ax, xl, xr, y):
        ax.text((xl + xr) / 2, y, "EMA\n" + r"$\longrightarrow$", ha="center", va="center", fontsize=8.5, color="#555555")

    # --- DINO ---
    ax, xl, xr = axes[0], 1.6, 5.4
    ax.text(xl, 0.1, "all crops: 2 global + several local", ha="center", fontsize=9.5)
    ax.text(xr, 0.1, "global crops only", ha="center", fontsize=9.5)
    for x, color, name in [(xl, blue, "student"), (xr, grey, "teacher")]:
        arrow(ax, x, 0.4, x, 0.95)
        box(ax, x, 1.3, f"{name} ViT  →  [CLS]", color)
        arrow(ax, x, 1.65, x, 2.15)
        box(ax, x, 2.5, "head: MLP + $K$ prototypes", color)
    ema(ax, xl, xr, 1.9)
    arrow(ax, xl, 2.85, xl, 4.15)
    box(ax, xl, 4.5, r"softmax$(\cdot\, / \tau_s)$", blue)
    arrow(ax, xr, 2.85, xr, 3.15)
    box(ax, xr, 3.5, "centering:  $- c$", yellow)
    arrow(ax, xr, 3.85, xr, 4.15)
    box(ax, xr, 4.5, r"softmax$(\cdot\, / \tau_t)$,  $\tau_t < \tau_s$", yellow, w=2.9)
    arrow(ax, xl, 4.85, xl + 1.2, 5.45)
    arrow(ax, xr, 4.85, xr - 1.2, 5.45)
    stop_grad(ax, xr - 0.55, 5.1)
    box(ax, (xl + xr) / 2, 5.8, r"cross-entropy  $-\sum_k P_t^{(k)} \log P_s^{(k)}$   (different crops)", green, w=5.6)
    ax.set_title("DINO: self-distillation of a distribution over prototypes", fontsize=12)

    # --- I-JEPA ---
    ax = axes[1]
    ax.text(xl, 0.1, "context: visible patches of the image", ha="center", fontsize=9.5)
    ax.text(xr, 0.1, "the whole image", ha="center", fontsize=9.5)
    arrow(ax, xl, 0.4, xl, 0.95)
    box(ax, xl, 1.3, "context encoder (ViT)", blue)
    arrow(ax, xr, 0.4, xr, 0.95)
    box(ax, xr, 1.3, "target encoder (ViT)", grey)
    ema(ax, xl, xr, 1.3)
    arrow(ax, xl, 1.65, xl, 2.65)
    box(ax, xl, 3.0, "predictor (narrow ViT)", orange)
    box(ax, xl - 0.2, 2.15, "mask tokens + positions of the target block", yellow, w=3.3, h=0.42)
    arrow(ax, xr, 1.65, xr, 2.65)
    box(ax, xr, 3.0, "keep the tokens of the\ntarget blocks", grey, h=0.75)
    arrow(ax, xl, 3.35, xl, 4.15)
    box(ax, xl, 4.5, "predicted patch\nrepresentations", blue, h=0.75)
    arrow(ax, xr, 3.4, xr, 4.15)
    box(ax, xr, 4.5, "target patch\nrepresentations", grey, h=0.75)
    arrow(ax, xl, 4.9, xl + 1.2, 5.45)
    arrow(ax, xr, 4.9, xr - 1.2, 5.45)
    stop_grad(ax, xr - 0.55, 5.12)
    box(ax, (xl + xr) / 2, 5.8, "regression in representation space (no pixels, no second view)", green, w=5.6)
    ax.set_title("I-JEPA: predict the representation of masked blocks", fontsize=12)

    for ax in axes:
        ax.set_xlim(-0.6, 7.6)
        ax.set_ylim(-0.2, 6.3)
        ax.axis("off")
    fig.text(0.5, 0.01, "blue: trained by gradient descent   |   grey: exponential moving average of the blue network"
             "   |   yellow / orange: what makes the two branches different", ha="center", fontsize=9, color="#555555")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    return fig
