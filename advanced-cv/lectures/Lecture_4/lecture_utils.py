import io
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
# Figures
# ----------------------------------------------------------------------------------------------------------------------
def draw_joint_embedding_methods(figsize=(17, 5.2)):
    """Schematic of the two-branch architectures discussed in the lecture (in the spirit of Fig. 3 of SimSiam)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    blue, grey, orange, green = "#cfe2f3", "#e6e6e6", "#fde5cd", "#d9ead3"
    # (title, has predictor, right branch: "shared" | "ema", stop-gradient on the right, extra box, loss text)
    methods = [
        ("SimCLR", False, "shared", False, None, "InfoNCE\nnegatives: rest of the batch"),
        ("MoCo", False, "ema", True, "queue of\npast keys", "InfoNCE\nnegatives: queue"),
        ("BYOL", True, "ema", True, None, r"$\|\bar p - \bar z\|^2$" + "\nno negatives"),
        ("SimSiam", True, "shared", True, None, r"$-\cos(p, z)$" + "\nno negatives"),
        ("Barlow Twins / VICReg", False, "shared", False, None, "statistics of the batch:\nvariance, covariance"),
    ]
    fig, axes = plt.subplots(1, len(methods), figsize=figsize)

    def box(ax, x, y, text, color, w=1.5, h=0.62):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04", fc=color, ec="#444444"))
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5)

    def arrow(ax, x, y0, y1, **kwargs):
        ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="->", color="#444444", **kwargs))

    for ax, (title, predictor, right, stop_grad, extra, loss) in zip(axes, methods):
        xl, xr = 1.0, 3.2
        right_color = grey if right == "ema" else blue
        for x, color, name in [(xl, blue, "$x_1$"), (xr, right_color, "$x_2$")]:
            ax.text(x, 0.15, name, ha="center", va="center", fontsize=11)
            arrow(ax, x, 0.4, 0.95)
            box(ax, x, 1.3, "backbone $f$", color)
            arrow(ax, x, 1.65, 2.15)
            box(ax, x, 2.5, "projector $g$", color)
        ax.text((xl + xr) / 2, 1.9, "shared\nweights" if right == "shared" else "EMA\n" + r"$\longrightarrow$",
                ha="center", va="center", fontsize=8.5, color="#555555")
        if predictor:
            arrow(ax, xl, 2.85, 3.35)
            box(ax, xl, 3.7, "predictor $q$", orange)
            arrow(ax, xl, 4.05, 4.75)
        else:
            arrow(ax, xl, 2.85, 4.75)
        arrow(ax, xr, 2.85, 4.75)
        if stop_grad:
            ax.plot([xr - 0.3, xr + 0.3], [3.45, 3.65], color="crimson", lw=2.5)
            ax.plot([xr - 0.3, xr + 0.3], [3.3, 3.5], color="crimson", lw=2.5)
            ax.text(xr + 0.4, 3.48, "stop-\ngrad", color="crimson", fontsize=8.5, va="center")
        if extra:
            box(ax, xr + 0.25, 4.2, extra, grey, w=1.35, h=0.5)
        box(ax, (xl + xr) / 2, 5.2, loss, green, w=3.7, h=0.8)
        ax.set_xlim(-0.2, 4.6)
        ax.set_ylim(-0.1, 5.8)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.text(0.5, 0.01, "blue: trained by gradient descent   |   grey: not trained by gradients (momentum copy or memory)"
             "   |   orange: only in one branch", ha="center", fontsize=9, color="#555555")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    return fig
