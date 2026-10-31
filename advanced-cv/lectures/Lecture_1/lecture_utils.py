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


def plot_runs(run_names, logdir_root="logdirs", metric="val_acc", ax=None, title=None):
    ax = ax or plt.gca()
    for name in run_names:
        df = read_lightning_metrics(os.path.join(logdir_root, name))
        df = df.dropna(subset=[metric])
        ax.plot(df["epoch"] + 1, df[metric], marker="o", markersize=3, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    return ax
