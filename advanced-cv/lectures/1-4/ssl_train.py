"""Training and evaluation loop for the self-supervised experiments of Lecture 4.

    python ssl_train.py --config '{"name": "simclr", "method": "simclr", "epochs": 100}'

A run writes to `<results_dir>/runs/`:
    <name>.csv         one row per epoch: loss, k-NN accuracy and collapse diagnostics of the backbone / projector
    <name>_steps.csv   loss, learning rate and embedding std every few optimizer steps
    <name>.json        the config and the final metrics (k-NN, linear probe, effective rank, ...)
and, for runs with `artifacts=True`, to `<results_dir>/artifacts/`: singular value spectra, a t-SNE of the test
set and nearest neighbours of a few query images. The backbone weights go to `<logdir>/<name>/backbone.pt`.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from lecture_utils import (
    Normalize, alignment, count_params, embedding_std, get_device, knn_accuracy, linear_probe, load_stl10, rankme,
    set_seed, singular_values, uniformity,
)
from ssl_methods import BACKBONES, METHODS, BatchAugment

DEFAULT_CONFIG = dict(
    name="debug",
    method="simclr",              # key of ssl_methods.METHODS
    method_kwargs={},             # passed to the method constructor
    backbone="resnet18",          # key of ssl_methods.BACKBONES
    aug={},                       # overrides of ssl_methods.BatchAugment
    train_split="unlabeled",      # "unlabeled" (100k images) or "train" (5k labeled images)
    subset=None,                  # use only this many training images
    epochs=100,
    batch_size=512,
    optimizer="sgd",              # "sgd", "lars" or "adamw"
    base_lr=0.06,                 # learning rate for batch size 256; scaled linearly with the batch size
    weight_decay=5e-4,
    warmup_epochs=10,
    freeze_patch_embed=False,     # MoCo v3 trick for ViTs
    eval_every=5,                 # epochs between evaluations
    log_every=20,                 # steps between rows of <name>_steps.csv
    artifacts=False,
    seed=95431,
)


class AMPBackbone(nn.Module):
    """Runs the backbone under mixed precision and returns float32 features, so heads and losses stay in float32."""

    def __init__(self, net, device):
        super().__init__()
        self.net = net
        self.device_type = device
        self.enabled = device == "cuda"

    def forward(self, x):
        with torch.autocast(self.device_type, dtype=torch.bfloat16, enabled=self.enabled):
            return self.net(x).float()


class LARS(torch.optim.Optimizer):
    """SGD with momentum and layer-wise adaptive learning rates (the optimizer of SimCLR, BYOL, Barlow Twins)."""

    def __init__(self, params, lr, weight_decay=0.0, momentum=0.9, trust_coefficient=0.001):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                                      trust_coefficient=trust_coefficient))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                update = p.grad
                if p.ndim > 1:      # no weight decay and no adaptation for biases and normalization parameters
                    update = update.add(p, alpha=g["weight_decay"])
                    p_norm, u_norm = torch.norm(p), torch.norm(update)
                    q = torch.where((p_norm > 0) & (u_norm > 0), g["trust_coefficient"] * p_norm / u_norm,
                                    torch.ones_like(p_norm))
                    update = update.mul(q)
                buf = self.state[p].setdefault("mu", torch.zeros_like(p))
                buf.mul_(g["momentum"]).add_(update)
                p.add_(buf, alpha=-g["lr"])


def build_optimizer(method, name, lr, weight_decay):
    params = [p for p in method.parameters() if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "lars":
        return LARS(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        decay, no_decay = [p for p in params if p.ndim > 1], [p for p in params if p.ndim <= 1]
        return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    raise ValueError(name)


def lr_at(step, total_steps, warmup_steps, lr):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    return lr * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))


class Evaluator:
    """Frozen-feature evaluation on the labeled part of STL-10 (5k train -> 8k test) plus label-free diagnostics."""

    def __init__(self, data_dir, device, n_pairs=2048, seed=0):
        self.device = device
        self.normalize = Normalize().to(device)
        train_x, train_y = load_stl10(data_dir, "train")
        test_x, test_y = load_stl10(data_dir, "test")
        self.train_x, self.test_x = torch.from_numpy(np.array(train_x)).to(device), torch.from_numpy(np.array(test_x)).to(device)
        self.train_y, self.test_y = torch.from_numpy(train_y).to(device), torch.from_numpy(test_y).to(device)
        # a fixed set of positive pairs (two augmented views of test images) for alignment / uniformity
        augment = BatchAugment().to(device)
        with torch.random.fork_rng(devices=[0] if device == "cuda" else []):
            torch.manual_seed(seed)
            self.view1, self.view2 = augment(self.test_x[:n_pairs]) * 255, augment(self.test_x[:n_pairs]) * 255

    @torch.no_grad()
    def embed(self, method, x, batch_size=1000):
        hs, zs = [], []
        for i in range(0, len(x), batch_size):
            h, z = method.embed(self.normalize(x[i:i + batch_size].float() / 255))
            hs.append(h.float())
            zs.append(z.float())
        return torch.cat(hs), torch.cat(zs)

    @torch.no_grad()
    def __call__(self, method):
        was_training = method.training
        method.eval()
        h_tr, z_tr = self.embed(method, self.train_x)
        h_te, z_te = self.embed(method, self.test_x)
        _, z1 = self.embed(method, self.view1)
        _, z2 = self.embed(method, self.view2)
        method.train(was_training)
        metrics = {
            "knn_acc": knn_accuracy(h_tr, self.train_y, h_te, self.test_y),
            "knn_acc_z": knn_accuracy(z_tr, self.train_y, z_te, self.test_y),
            "rankme_h": rankme(h_te), "rankme_z": rankme(z_te),
            "std_h": embedding_std(h_te), "std_z": embedding_std(z_te),
            "alignment": alignment(z1, z2), "uniformity": uniformity(z_te),
        }
        return metrics, (h_tr, h_te, z_te)


def save_artifacts(name, h_te, z_te, test_y, out_dir, n_tsne=3000, n_neighbours=6):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    os.makedirs(out_dir, exist_ok=True)
    spectra = [pd.DataFrame({"space": space, "index": np.arange(1, len(s) + 1), "sigma": s.cpu().numpy()})
               for space, s in (("backbone h", singular_values(h_te)), ("projector z", singular_values(z_te)))]
    pd.concat(spectra).to_csv(os.path.join(out_dir, f"{name}_spectrum.csv"), index=False)

    idx = np.random.default_rng(0).choice(len(h_te), n_tsne, replace=False)
    x = F.normalize(h_te[idx], dim=1).cpu().numpy()
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(
        PCA(n_components=min(50, x.shape[1]), random_state=0).fit_transform(x))
    pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "label": test_y[idx].cpu().numpy()}).round(3).to_csv(
        os.path.join(out_dir, f"{name}_tsne.csv"), index=False)

    # nearest neighbours (in the test set) of the first test image of every class
    hn = F.normalize(h_te, dim=1)
    queries = torch.stack([(test_y == c).nonzero()[0, 0] for c in range(int(test_y.max()) + 1)])
    sims = hn[queries] @ hn.T
    sims[torch.arange(len(queries)), queries] = -1
    top_sims, top_idx = sims.topk(n_neighbours, dim=1)
    rows = [{"query": int(q), "rank": r + 1, "neighbour": int(top_idx[i, r]), "similarity": float(top_sims[i, r])}
            for i, q in enumerate(queries) for r in range(n_neighbours)]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{name}_neighbours.csv"), index=False)


def train_ssl(config, data_dir="../../data", results_dir="results", logdir="logdirs", device=None):
    cfg = {**DEFAULT_CONFIG, **config}
    unknown = set(config) - set(DEFAULT_CONFIG)
    assert not unknown, f"unknown config keys: {unknown}"
    device = device or get_device()
    name = cfg["name"]
    runs_dir = os.path.join(results_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    # data: the whole split lives on the device as uint8, views are created there by BatchAugment (no DataLoader)
    images, labels = load_stl10(data_dir, cfg["train_split"])
    if cfg["subset"]:
        keep = np.sort(np.random.default_rng(cfg["seed"]).choice(len(images), cfg["subset"], replace=False))
        images, labels = images[keep], labels[keep]
    images, labels = torch.from_numpy(np.array(images)).to(device), torch.from_numpy(labels).to(device)
    augment, normalize = BatchAugment(**cfg["aug"]).to(device), Normalize().to(device)
    method_cls = METHODS[cfg["method"]]
    needs_labels = getattr(method_cls, "needs_labels", False)
    steps_per_epoch = len(images) // cfg["batch_size"]

    # model
    backbone, feat_dim = BACKBONES[cfg["backbone"]]()
    if cfg["freeze_patch_embed"]:
        backbone.patch_embed.requires_grad_(False)
    if device == "cuda":
        backbone = backbone.to(memory_format=torch.channels_last)
    method = method_cls(AMPBackbone(backbone, device), feat_dim, **cfg["method_kwargs"]).to(device)
    lr = cfg["base_lr"] * cfg["batch_size"] / 256
    optimizer = build_optimizer(method, cfg["optimizer"], lr, cfg["weight_decay"])
    total_steps, warmup_steps = cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch
    evaluator = Evaluator(data_dir, device)
    print(f"[{name}] {cfg['method']} / {cfg['backbone']} | {count_params(method.backbone) / 1e6:.1f}M backbone params | "
          f"{len(images)} images, {steps_per_epoch} steps/epoch, {cfg['epochs']} epochs | lr {lr:.4g}", flush=True)

    epoch_rows, step_rows = [], []
    metrics, feats = evaluator(method)                      # epoch 0: the randomly initialized network
    epoch_rows.append({"epoch": 0, "step": 0, "loss": float("nan"), "minutes": 0.0, **metrics})
    step, diverged, start = 0, False, time.time()
    method.train()
    for epoch in range(1, cfg["epochs"] + 1):
        losses = []
        perm = torch.randperm(len(images), device=device)
        for i in range(steps_per_epoch):
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup_steps, lr)
            idx = perm[i * cfg["batch_size"]:(i + 1) * cfg["batch_size"]]
            views = [normalize(augment(images[idx])) for _ in range(1 if needs_labels else 2)]
            if device == "cuda":
                views = [v.contiguous(memory_format=torch.channels_last) for v in views]
            loss = method(views[0], None, labels[idx]) if needs_labels else method(views[0], views[1])
            if not torch.isfinite(loss):
                diverged = True
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            method.step_end(step / total_steps)
            losses.append(loss.item())
            if step % cfg["log_every"] == 0:
                with torch.no_grad():
                    _, z = method.embed(views[0][:256])
                step_rows.append({"step": step, "epoch": step / steps_per_epoch, "loss": losses[-1],
                                  "lr": optimizer.param_groups[0]["lr"], "std_z": embedding_std(z)})
        if diverged:
            print(f"[{name}] loss is not finite at step {step}, stopping", flush=True)
            epoch_rows.append({"epoch": epoch, "step": step, "loss": float("nan"), "minutes": (time.time() - start) / 60})
            pd.DataFrame(epoch_rows).to_csv(os.path.join(runs_dir, f"{name}.csv"), index=False)
            pd.DataFrame(step_rows, columns=["step", "epoch", "loss", "lr", "std_z"]).to_csv(
                os.path.join(runs_dir, f"{name}_steps.csv"), index=False)
            break
        row = {"epoch": epoch, "step": step, "loss": float(np.mean(losses)), "minutes": (time.time() - start) / 60}
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            metrics, feats = evaluator(method)
            row.update(metrics)
            print(f"[{name}] epoch {epoch:3d} | loss {row['loss']:.4f} | kNN {metrics['knn_acc']:.3f} | "
                  f"rank h {metrics['rankme_h']:.0f} z {metrics['rankme_z']:.0f} | std z {metrics['std_z']:.2f} | "
                  f"{row['minutes']:.1f} min", flush=True)
        epoch_rows.append(row)
        pd.DataFrame(epoch_rows).to_csv(os.path.join(runs_dir, f"{name}.csv"), index=False)
        pd.DataFrame(step_rows).to_csv(os.path.join(runs_dir, f"{name}_steps.csv"), index=False)

    # final evaluation
    h_tr, h_te, z_te = feats
    probe = linear_probe(h_tr, evaluator.train_y, h_te, evaluator.test_y)
    summary = {
        "name": name, "method": cfg["method"], "backbone": cfg["backbone"],
        "backbone_params_m": round(count_params(method.backbone) / 1e6, 2),
        "train_images": len(images), "epochs": cfg["epochs"], "batch_size": cfg["batch_size"],
        "minutes": round((time.time() - start) / 60, 1), "diverged": diverged,
        "final_loss": epoch_rows[-1]["loss"], "linear_acc": probe["test_acc"], **metrics, "config": cfg,
    }
    if cfg["artifacts"]:
        save_artifacts(name, h_te, z_te, evaluator.test_y, os.path.join(results_dir, "artifacts"))
    os.makedirs(os.path.join(logdir, name), exist_ok=True)
    torch.save(method.backbone.net.state_dict(), os.path.join(logdir, name, "backbone.pt"))
    with open(os.path.join(runs_dir, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[{name}] done in {summary['minutes']} min | kNN {summary['knn_acc']:.4f} | "
          f"linear {summary['linear_acc']:.4f}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=json.loads, required=True, help="JSON overrides of DEFAULT_CONFIG")
    parser.add_argument("--data-dir", default="../../data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--logdir", default="logdirs")
    args = parser.parse_args()
    train_ssl(args.config, args.data_dir, args.results_dir, args.logdir)
