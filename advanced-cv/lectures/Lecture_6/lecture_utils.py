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


def few_shot_probe(train_x, train_y, test_x, test_y, shots, num_classes=10, weight_decay=1e-2, seed=0):
    """Logistic regression on `shots` random labeled images per class (no validation split at this size)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.cat([torch.nonzero(train_y.cpu() == c).flatten()[torch.randperm(int((train_y == c).sum()), generator=g)[:shots]]
                     for c in range(num_classes)]).to(train_x.device)
    xs_tr, xs_te = standardize(train_x[idx].float(), test_x.float())
    clf = fit_logreg(xs_tr, train_y[idx], num_classes, weight_decay)
    with torch.no_grad():
        return (clf(xs_te).argmax(1) == test_y).float().mean().item()


# ----------------------------------------------------------------------------------------------------------------------
# Flickr30k: 31,783 images with 5 captions each; Karpathy split 29,000 train / 1,014 val / 1,000 test
# ----------------------------------------------------------------------------------------------------------------------
FLICKR_REPO = "nlphuji/flickr30k"


def load_flickr30k(root, split):
    """Returns (list of image paths, list of lists of 5 captions) for a split of the Karpathy split.

    The first call downloads the images (4.4 GB zip) and the annotations into `<root>/flickr30k` and unpacks them.
    """
    import json
    import zipfile

    import pandas as pd

    assert split in ("train", "val", "test")
    folder = os.path.join(root, "flickr30k")
    images = os.path.join(folder, "flickr30k-images")
    if not os.path.isdir(images):
        from huggingface_hub import hf_hub_download

        for f in ("flickr_annotations_30k.csv", "flickr30k-images.zip"):
            if not os.path.exists(os.path.join(folder, f)):
                hf_hub_download(FLICKR_REPO, f, repo_type="dataset", local_dir=folder)
        with zipfile.ZipFile(os.path.join(folder, "flickr30k-images.zip")) as z:
            z.extractall(folder)
    df = pd.read_csv(os.path.join(folder, "flickr_annotations_30k.csv"))
    df = df[df.split == split].sort_values("img_id")
    return [os.path.join(images, f) for f in df.filename], [json.loads(r) for r in df.raw]


class ImageFiles(torch.utils.data.Dataset):
    """Image paths, uint8 arrays (N, 3, H, W) or HuggingFace image datasets -> transformed tensors;
    `views` random views per image."""

    def __init__(self, items, transform, views=1):
        self.items, self.transform, self.views = items, transform, views

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image

        item = self.items[i]
        if isinstance(item, str):                 # a path
            img = Image.open(item).convert("RGB")
        elif isinstance(item, dict):              # a row of a HuggingFace dataset
            img = item["image"].convert("RGB")
        else:                                     # a uint8 array (3, H, W)
            img = Image.fromarray(np.asarray(item).transpose(1, 2, 0))
        if self.views == 1:
            return self.transform(img)
        return torch.stack([self.transform(img) for _ in range(self.views)])


# ----------------------------------------------------------------------------------------------------------------------
# Fine-grained transfer datasets of Lecture 3 (HuggingFace Hub mirrors with the official splits)
# ----------------------------------------------------------------------------------------------------------------------
TRANSFER_SOURCES = {   # name: (repo, training splits, test split)
    "pets": ("timm/oxford-iiit-pet", ["train"], "test"),
    "flowers": ("dpdl-benchmark/oxford_flowers102", ["train", "validation"], "test"),
    "dtd": ("dpdl-benchmark/dtd", ["train", "validation"], "test"),
}


def load_transfer(name, root):
    """Returns (train dataset, train labels, test dataset, test labels, class names); train = train + val."""
    from datasets import concatenate_datasets, load_dataset

    repo, train_splits, test_split = TRANSFER_SOURCES[name]
    load = lambda s: load_dataset(repo, split=s, cache_dir=os.path.join(root, "hf_datasets"))
    train = concatenate_datasets([load(s) for s in train_splits])
    test = load(test_split)
    names = [c.replace("_", " ") for c in test.features["label"].names]
    return train, np.asarray(train["label"]), test, np.asarray(test["label"]), names
