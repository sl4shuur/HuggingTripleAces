"""All computations of Lecture 6. Run from this folder:

    python experiments.py --list                     # what will be run
    python experiments.py --gpus 0                   # everything (finished runs are skipped)
    python experiments.py --gpus 0 --groups loss     # one group
    python experiments.py --gpus 0 --smoke           # 1-epoch versions into results_smoke/ (code check)

Group "pretrained" evaluates the downloaded CLIP / SigLIP / SigLIP 2 / DINOv2 models; the other groups train dual
encoders on Flickr30k with a frozen image tower (see train.py). Image features are cached in `features/`.
"""
import argparse
import glob
import json
import os
import time
import traceback

import pandas as pd

DATA_DIR = os.path.join("..", "..", "data")
EPOCHS = 50

EXPERIMENTS = {}


def add(name, group, **config):
    assert name not in EXPERIMENTS, name
    EXPERIMENTS[name] = (group, {"name": name, "epochs": EPOCHS, **config})


# --- loss x batch size: CLIP (softmax) vs SigLIP (sigmoid), DINOv2 ViT-S/14 image tower, text tower from scratch ---
SEEDS = (95431, 1, 2)
for bs in (64, 256, 1024, 4096):
    for loss in ("clip", "siglip"):
        for i, seed in enumerate(SEEDS):
            add(f"{loss}_bs{bs}" + (f"_seed{i}" if i else ""), "loss", loss=loss, batch_size=bs, seed=seed)
# --- which frozen image tower (SigLIP loss, batch 1024) ---
for tower in ("sup_vits16_in21k", "random_vits16", "lecture5_dino_vitti8"):
    add(f"tower_{tower}", "towers", image_tower=tower)
# --- pretrained text tower (MiniLM, fine-tuned with a 10x smaller learning rate) ---
add("text_minilm", "towers", text_tower="minilm", text_lr=5e-5)
# --- how many image-text pairs (same number of optimizer steps as with all 29k images) ---
for n in (1000, 3000, 10000):
    add(f"data_{n // 1000}k", "data", train_images=n, batch_size=256, eval_every=5 * 29000 // n)   # vs siglip_bs256

GROUPS = ["pretrained", "loss", "towers", "data"]


def smoke_config(cfg):
    return {**cfg, "epochs": 1, "eval_every": 1}


def aggregate(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "runs", "*.json"))):
        with open(path) as f:
            row = json.load(f)
        row.pop("config")
        row["group"] = EXPERIMENTS[row["name"]][0] if row["name"] in EXPERIMENTS else "other"
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df.insert(1, "group", df.pop("group"))
        df.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0", help="the GPU to use (runs are sequential)")
    parser.add_argument("--groups", default=None, help=f"comma-separated subset of {GROUPS}")
    parser.add_argument("--names", default=None, help="comma-separated run names")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-run finished runs")
    parser.add_argument("--aggregate", action="store_true", help="only rebuild results/summary.csv")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.split(",")[0]
    results_dir = "results_smoke" if args.smoke else "results"
    if args.aggregate:
        aggregate(results_dir)
        return
    groups = args.groups.split(",") if args.groups else GROUPS
    names = args.names.split(",") if args.names else None
    todo = [(n, g, c) for n, (g, c) in EXPERIMENTS.items()
            if g in groups and (names is None or n in names)]
    if args.list:
        for n, g, c in todo:
            print(f"{g:8s} {n:28s} {c}")
        print(f"{len(todo)} runs" + (" + pretrained evaluation" if "pretrained" in groups else ""))
        return

    from train import train   # imports torch: after CUDA_VISIBLE_DEVICES is set

    if "pretrained" in groups and names is None and not args.smoke:
        if args.force or not os.path.exists(os.path.join(results_dir, "pretrained", "geometry.npz")):
            from pretrained import run
            t0 = time.time()
            try:
                run(DATA_DIR, results_dir)
                print(f"[gpu {args.gpus}] pretrained: ok ({(time.time() - t0) / 60:.1f} min)", flush=True)
            except Exception:
                traceback.print_exc()
                print(f"[gpu {args.gpus}] pretrained: FAILED", flush=True)
    for name, group, cfg in todo:
        if not args.force and os.path.exists(os.path.join(results_dir, "runs", f"{name}.json")):
            continue
        try:
            train(smoke_config(cfg) if args.smoke else cfg, DATA_DIR, results_dir, "features")
            print(f"[gpu {args.gpus}] {name}: ok", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"[gpu {args.gpus}] {name}: FAILED", flush=True)
    aggregate(results_dir)


if __name__ == "__main__":
    main()
