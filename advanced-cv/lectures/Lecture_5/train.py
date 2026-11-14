"""Training and evaluation loop for the experiments of Lecture 5 (DINO and I-JEPA with a ViT-Ti/8 on STL-10).

    python train.py --config '{"name": "dino", "method": "dino", "epochs": 100}'

A run writes to `<results_dir>/runs/`:
    <name>.csv         one row per epoch: loss, k-NN accuracy of the teacher and of the student, effective rank
    <name>_steps.csv   loss, learning rate and the method's diagnostics every few optimizer steps
    <name>.json        the config and the final metrics (k-NN, linear probe, ...)
and, for runs with `artifacts=True`, to `<results_dir>/artifacts/<name>.npz`: attention maps and a PCA of the patch
tokens for a few test images. The teacher weights go to `<logdir>/<name>/encoder.pt`.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from lecture_utils import (
    BatchAugment, Normalize, count_params, embedding_std, get_device, knn_accuracy, linear_probe, load_stl10, rankme,
    set_seed,
)
from self_distillation import METHODS, MultiCrop, ViTEncoder

DEFAULT_CONFIG = dict(
    name="debug",
    method="dino",                # key of self_distillation.METHODS
    method_kwargs={},             # passed to the method constructor
    views={},                     # dino: overrides of MultiCrop; other methods: overrides of BatchAugment
    train_split="unlabeled",      # "unlabeled" (100k images) or "train" (5k labeled images)
    epochs=100,
    batch_size=512,
    base_lr=5e-4,                 # AdamW learning rate for batch size 256; scaled linearly with the batch size
    weight_decay=0.05,
    warmup_epochs=10,
    grad_clip=3.0,
    eval_every=5,                 # epochs between evaluations
    log_every=20,                 # steps between rows of <name>_steps.csv
    artifacts=False,
    seed=95431,
)
ARTIFACT_IMAGES = 12              # test images (the first of every class and two more) used for the pictures
ARTIFACT_SIZE = 192               # they are upsampled 2x, so that the token grid is 24 x 24 instead of 12 x 12


def lr_at(step, total_steps, warmup_steps, lr):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    return lr * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))


class Evaluator:
    """Frozen-feature evaluation on the labeled part of STL-10 (5k train -> 8k test), as in Lectures 3 and 4."""

    def __init__(self, data_dir, device):
        self.normalize = Normalize().to(device)
        train_x, train_y = load_stl10(data_dir, "train")
        test_x, test_y = load_stl10(data_dir, "test")
        self.train_x, self.test_x = torch.from_numpy(np.array(train_x)).to(device), torch.from_numpy(np.array(test_x)).to(device)
        self.train_y, self.test_y = torch.from_numpy(train_y).to(device), torch.from_numpy(test_y).to(device)

    @torch.no_grad()
    def embed(self, method, x, batch_size=1000):
        feats = [method.features(self.normalize(x[i:i + batch_size].float() / 255)) for i in range(0, len(x), batch_size)]
        return {k: torch.cat([f[k] for f in feats]) for k in feats[0]}

    @torch.no_grad()
    def __call__(self, method):
        was_training = method.training
        method.eval()
        tr, te = self.embed(method, self.train_x), self.embed(method, self.test_x)
        method.train(was_training)
        metrics = {
            "knn_acc": knn_accuracy(tr["teacher"], self.train_y, te["teacher"], self.test_y),
            "knn_acc_student": knn_accuracy(tr["student"], self.train_y, te["student"], self.test_y),
            "rankme_h": rankme(te["teacher"]), "std_h": embedding_std(te["teacher"]),
        }
        return metrics, (tr["teacher"], te["teacher"])


def artifact_images(test_x, test_y, n=ARTIFACT_IMAGES):
    """Indices of the first test image of every class, then the second image of the first classes."""
    first = [(test_y == c).nonzero()[0, 0].item() for c in range(10)]
    second = [(test_y == c).nonzero()[1, 0].item() for c in range(10)]
    return (first + second)[:n]


@torch.no_grad()
def save_artifacts(name, encoder, evaluator, out_dir, size=ARTIFACT_SIZE):
    """Attention of [CLS] to the patches in the last block (per head) and a 3-component PCA of the patch tokens."""
    os.makedirs(out_dir, exist_ok=True)
    was_training = encoder.training
    encoder.eval()
    idx = artifact_images(evaluator.test_x, evaluator.test_y)
    x = F.interpolate(evaluator.test_x[idx].float() / 255, size=size, mode="bicubic", align_corners=False).clamp(0, 1)
    grid = size // encoder.patch_size
    encoder.set_store_attn(True)
    tokens = encoder(evaluator.normalize(x))
    attn = encoder.get_attn_maps()[-1].float()                                    # (B, heads, tokens, tokens)
    encoder.set_store_attn(False)
    encoder.train(was_training)
    prefix = encoder.num_prefix_tokens
    patches = tokens[:, prefix:]                                                  # (B, N, D)
    flat = patches.reshape(-1, patches.shape[-1])
    flat = flat - flat.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(flat, q=3, center=False)
    out = {"index": np.array(idx), "pca": (flat @ v).reshape(len(idx), grid, grid, 3).cpu().numpy().astype(np.float16)}
    if prefix:
        out["cls_attention"] = attn[:, :, 0, prefix:].reshape(len(idx), -1, grid, grid).cpu().numpy().astype(np.float16)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **out)


def train(config, data_dir="../../data", results_dir="results", logdir="logdirs", device=None):
    cfg = {**DEFAULT_CONFIG, **config}
    unknown = set(config) - set(DEFAULT_CONFIG)
    assert not unknown, f"unknown config keys: {unknown}"
    device = device or get_device()
    name = cfg["name"]
    runs_dir = os.path.join(results_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    set_seed(cfg["seed"])

    # data: the whole split lives on the device as uint8, views are created there (no DataLoader), as in Lecture 4
    images, labels = load_stl10(data_dir, cfg["train_split"])
    images, labels = torch.from_numpy(np.array(images)).to(device), torch.from_numpy(labels).to(device)
    normalize = Normalize().to(device)
    if cfg["method"] == "dino":
        make_views = MultiCrop(**cfg["views"]).to(device)
    else:
        augment = BatchAugment(**cfg["views"]).to(device)
        make_views = lambda x: [augment(x)]
    steps_per_epoch = len(images) // cfg["batch_size"]

    # model: I-JEPA has no [CLS] token (its image representation is the average of the patch tokens)
    method_cls = METHODS[cfg["method"]]
    needs_labels = getattr(method_cls, "needs_labels", False)
    method = method_cls(ViTEncoder(cls_token=cfg["method"] != "ijepa"), **cfg["method_kwargs"]).to(device)
    params = [p for p in method.parameters() if p.requires_grad]
    lr = cfg["base_lr"] * cfg["batch_size"] / 256
    optimizer = torch.optim.AdamW([{"params": [p for p in params if p.ndim > 1], "weight_decay": cfg["weight_decay"]},
                                   {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0}], lr=lr)
    total_steps, warmup_steps = cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch
    evaluator = Evaluator(data_dir, device)
    print(f"[{name}] {cfg['method']} | {count_params(method.student) / 1e6:.1f}M encoder params | {len(images)} images, "
          f"{steps_per_epoch} steps/epoch, {cfg['epochs']} epochs | lr {lr:.4g}", flush=True)

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
            views = [normalize(v) for v in make_views(images[idx])]
            loss = method(views, labels[idx]) if needs_labels else method(views)
            if not torch.isfinite(loss):
                diverged = True
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            optimizer.step()
            step += 1
            method.step_end(step / total_steps)
            losses.append(loss.item())
            if step % cfg["log_every"] == 0:
                step_rows.append({"step": step, "epoch": step / steps_per_epoch, "loss": losses[-1],
                                  "lr": optimizer.param_groups[0]["lr"], **method.stats})
        if diverged:
            print(f"[{name}] loss is not finite at step {step}, stopping", flush=True)
            epoch_rows.append({"epoch": epoch, "step": step, "loss": float("nan"), "minutes": (time.time() - start) / 60})
        else:
            row = {"epoch": epoch, "step": step, "loss": float(np.mean(losses)), "minutes": (time.time() - start) / 60}
            if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
                metrics, feats = evaluator(method)
                row.update(metrics)
                print(f"[{name}] epoch {epoch:3d} | loss {row['loss']:.4f} | kNN teacher {metrics['knn_acc']:.3f} "
                      f"student {metrics['knn_acc_student']:.3f} | rank {metrics['rankme_h']:.0f} | "
                      f"{' '.join(f'{k} {v:.3g}' for k, v in method.stats.items())} | {row['minutes']:.1f} min", flush=True)
            epoch_rows.append(row)
        pd.DataFrame(epoch_rows).to_csv(os.path.join(runs_dir, f"{name}.csv"), index=False)
        pd.DataFrame(step_rows).to_csv(os.path.join(runs_dir, f"{name}_steps.csv"), index=False)
        if diverged:
            break

    # final evaluation
    h_tr, h_te = feats
    probe = linear_probe(h_tr, evaluator.train_y, h_te, evaluator.test_y)
    encoder = getattr(method, "teacher", method.student)
    summary = {
        "name": name, "method": cfg["method"], "encoder_params_m": round(count_params(method.student) / 1e6, 2),
        "train_images": len(images), "epochs": cfg["epochs"], "batch_size": cfg["batch_size"],
        "minutes": round((time.time() - start) / 60, 1), "diverged": diverged,
        "final_loss": epoch_rows[-1]["loss"], "linear_acc": probe["test_acc"], **metrics, "config": cfg,
    }
    if cfg["artifacts"]:
        save_artifacts(name, encoder, evaluator, os.path.join(results_dir, "artifacts"))
    os.makedirs(os.path.join(logdir, name), exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(logdir, name, "encoder.pt"))
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
    train(args.config, args.data_dir, args.results_dir, args.logdir)
