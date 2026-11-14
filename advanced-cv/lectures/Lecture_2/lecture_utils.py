import glob
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


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


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


def count_params(model, verbose=True):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Total params:     {total / 1e6:.2f}M")
        print(f"Trainable params: {trainable / 1e6:.2f}M")
    return total


@torch.inference_mode()
def benchmark_fn(fn, device, n_warmup=3, n_iters=10):
    """Average wall-clock time of `fn()` in milliseconds."""
    for _ in range(n_warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / n_iters * 1000


def denormalize(img, mean, std):
    """(C, H, W) normalized tensor -> (H, W, C) numpy image in [0, 1]."""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (img.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def read_lightning_metrics(log_dir):
    """Read the `metrics.csv` written by Lightning's CSVLogger (latest version)."""
    paths = sorted(glob.glob(os.path.join(log_dir, "**", "metrics.csv"), recursive=True), key=os.path.getmtime)
    if not paths:
        raise FileNotFoundError(f"No metrics.csv under {log_dir}")
    return pd.read_csv(paths[-1])


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation of ImageNet-1k classifiers on an ImageNet subset (Imagewoof / Imagenette)
# ----------------------------------------------------------------------------------------------------------------------
def download_fastai_subset(data_dir, name="imagewoof2-320"):
    """Downloads and extracts a fast.ai ImageNet subset (e.g. imagewoof2-320, imagenette2-320) if missing."""
    import tarfile
    import urllib.request

    root = os.path.join(data_dir, name)
    if not os.path.exists(root):
        os.makedirs(data_dir, exist_ok=True)
        archive = os.path.join(data_dir, f"{name}.tgz")
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(f"https://s3.amazonaws.com/fast-ai-imageclas/{name}.tgz", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(data_dir)
        os.remove(archive)
    return root


def make_eval_loader(root, model, batch_size=64, num_workers=4):
    """ImageFolder loader with the preprocessing that `model`'s pretrained checkpoint expects."""
    import timm
    from torchvision.datasets import ImageFolder

    cfg = timm.data.resolve_data_config({}, model=model)
    dataset = ImageFolder(root, transform=timm.data.create_transform(**cfg))
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=False
    )


@torch.inference_mode()
def evaluate_on_imagenet_subset(model, loader, device, forward_fn=None):
    """Top-1 accuracy of an ImageNet-1k classifier on a fast.ai ImageNet subset (Imagewoof / Imagenette).

    The folders of fast.ai subsets are ImageNet WordNet ids (e.g. n02086240), so we map them to ImageNet-1k indices.
    Returns two numbers:
      - "top1_subset":   argmax over the logits of the subset's classes only (an easy 10-way problem);
      - "top1_imagenet": argmax over all 1000 ImageNet classes (much harder for Imagewoof: 120 dog breeds compete).
    """
    from timm.data import ImageNetInfo

    wnids = ImageNetInfo().label_names()
    class_idx = torch.tensor([wnids.index(c) for c in loader.dataset.classes], device=device)
    forward_fn = forward_fn or model
    correct_subset, correct_all, total = 0, 0, 0
    for x, y in loader:
        logits = forward_fn(x.to(device))
        y = y.to(device)
        correct_subset += (logits[:, class_idx].argmax(-1) == y).sum().item()
        correct_all += (logits.argmax(-1) == class_idx[y]).sum().item()
        total += len(y)
    return {"top1_subset": correct_subset / total, "top1_imagenet": correct_all / total}


def count_gmacs(model, img_size=224, device="cpu"):
    """GMACs of one forward pass (FlopCounterMode counts FLOPs = 2 * MACs).

    NOTE: fused attention kernels are not always visible to the FLOP counter, so disable them with
    `timm.layers.set_fused_attn(False)` before creating the model.
    """
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.randn(1, 3, img_size, img_size, device=device)
    with FlopCounterMode(display=False) as counter, torch.inference_mode():
        model(x)
    return counter.get_total_flops() / 2e9


@torch.inference_mode()
def measure_throughput(model, device, img_size=224, batch_size=64, n_warmup=3, n_iters=10):
    """Images per second for inference with a fixed batch of random images."""
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    ms = benchmark_fn(lambda: model(x), device, n_warmup=n_warmup, n_iters=n_iters)
    return batch_size / ms * 1000
