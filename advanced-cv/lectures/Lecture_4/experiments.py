"""All training runs of Lecture 4 and a launcher that spreads them over the available GPUs.

    python experiments.py --list                      # show all runs
    python experiments.py --gpus 0,1                  # run everything that is not finished yet
    python experiments.py --gpus 0 --groups main,aug  # only some groups
    python experiments.py --gpus 0 --names simclr     # only some runs
    python experiments.py --gpus 0 --smoke            # 2-epoch versions on 4k images into results_smoke/ (code check)
    python experiments.py --aggregate                 # rebuild results/summary.csv from the finished runs

Finished runs (those with `results/runs/<name>.json`) are skipped, so the launcher can be stopped and restarted.
"""
import argparse
import glob
import json
import os
import queue
import subprocess
import sys
import threading

import pandas as pd

EPOCHS_MAIN = 100       # the main comparison of methods
EPOCHS_ABLATION = 30    # every ablation run

# Method defaults (ResNet-18, SGD with momentum, cosine schedule; see ssl_train.DEFAULT_CONFIG for the rest).
# `base_lr` (the learning rate for a batch of 256) was chosen per method from a coarse grid {0.03 ... 0.48}
# in short pilot runs (20 epochs on 50k images); everything else follows the papers, scaled down.
METHOD_CONFIGS = {
    "simclr": dict(method="simclr", base_lr=0.24, method_kwargs=dict(temperature=0.2)),
    "moco": dict(method="moco", base_lr=0.24, method_kwargs=dict(temperature=0.2, queue_size=16384, momentum=0.99)),
    "nnclr": dict(method="nnclr", base_lr=0.12, method_kwargs=dict(temperature=0.2, queue_size=16384)),
    "byol": dict(method="byol", base_lr=0.24, method_kwargs=dict(ema=0.99)),
    "simsiam": dict(method="simsiam", base_lr=0.12, method_kwargs={}),
    "barlow_twins": dict(method="barlow_twins", base_lr=0.02, method_kwargs=dict(lambd=5e-3)),
    "vicreg": dict(method="vicreg", base_lr=0.12, method_kwargs={}),
    # MoCo v3: no queue, predictor, symmetric loss
    "mocov3": dict(method="moco", base_lr=0.12, method_kwargs=dict(temperature=0.2, queue_size=0, momentum=0.99,
                                                                   predictor=True, proj_dim=256)),
}

EXPERIMENTS = {}


def add(name, group, base, epochs=EPOCHS_ABLATION, method_kwargs=None, **overrides):
    cfg = {**METHOD_CONFIGS[base], "name": name, "epochs": epochs, **overrides}
    cfg["method_kwargs"] = {**METHOD_CONFIGS[base]["method_kwargs"], **(method_kwargs or {})}
    cfg.setdefault("eval_every", 5 if epochs > 50 else 1)
    assert name not in EXPERIMENTS, name
    EXPERIMENTS[name] = (group, cfg)


# --- main: all methods, same backbone, data and budget --------------------------------------------------------------
for m in ["simclr", "moco", "byol", "simsiam", "barlow_twins", "vicreg"]:
    add(m, "main", m, epochs=EPOCHS_MAIN, artifacts=True)
add("mocov3_resnet18", "main", "mocov3", epochs=EPOCHS_MAIN, artifacts=True)
add("mocov3_vit_tiny", "main", "mocov3", epochs=EPOCHS_MAIN, artifacts=True, backbone="vit_tiny", optimizer="adamw",
    base_lr=1e-3, weight_decay=0.1, freeze_patch_embed=True)
# SimSiam with the learning rate of the pilots (0.12) collapses dimensionally in the 100-epoch run (see the lecture),
# so the main comparison also has it with lower learning rates
for lr in [0.06, 0.03]:
    add(f"simsiam_lr{lr}", "main", "simsiam", epochs=EPOCHS_MAIN, artifacts=True, base_lr=lr)
# supervised reference: the same backbone trained with labels on the 5k labeled images (same number of steps as
# an ablation run: 39 steps/epoch * 150 epochs = 5850 steps)
EXPERIMENTS["supervised_5k"] = ("main", dict(
    name="supervised_5k", method="supervised", method_kwargs={}, train_split="train", epochs=150, batch_size=128,
    base_lr=0.1, eval_every=10, artifacts=True))

# --- ablation baselines (30 epochs) -----------------------------------------------------------------------------------
for m in ["simclr", "moco", "byol", "simsiam", "vicreg"]:
    add(f"abl_{m}", "ablation_base", m)

# --- augmentations ----------------------------------------------------------------------------------------------------
NO_COLOR = dict(jitter_p=0.0, gray_p=0.0)
AUGS = {
    "no_crop": dict(crop_scale=None),
    "no_color": NO_COLOR,
    "no_blur": dict(blur_p=0.0),
    "crop_only": dict(**NO_COLOR, blur_p=0.0),
    "crop_008": dict(crop_scale=(0.08, 1.0)),
    "crop_05": dict(crop_scale=(0.5, 1.0)),
}
for aug_name, aug in AUGS.items():
    add(f"aug_simclr_{aug_name}", "aug", "simclr", aug=aug)
for aug_name in ["no_color", "crop_only"]:
    add(f"aug_byol_{aug_name}", "aug", "byol", aug=AUGS[aug_name])

# --- SimCLR: temperature, projection head, loss -----------------------------------------------------------------------
for t in [0.05, 0.1, 0.5, 1.0]:
    add(f"temp_simclr_{t}", "temperature", "simclr", method_kwargs=dict(temperature=t))
for proj in ["none", "linear"]:
    add(f"proj_simclr_{proj}", "projector", "simclr", method_kwargs=dict(projector=proj))
add("loss_simclr_dcl", "loss", "simclr", method_kwargs=dict(loss="dcl"))
add("loss_nnclr", "loss", "nnclr")

# --- negatives: batch size and queue size -----------------------------------------------------------------------------
for bs in [64, 128, 256]:
    add(f"batch_simclr_{bs}", "batch", "simclr", batch_size=bs)
add("batch_simclr_dcl_64", "batch", "simclr", batch_size=64, method_kwargs=dict(loss="dcl"))
for m in ["moco", "byol", "simsiam"]:
    add(f"batch_{m}_64", "batch", m, batch_size=64)
for k in [256, 1024, 4096, 65536]:
    add(f"queue_moco_{k}", "queue", "moco", method_kwargs=dict(queue_size=k))

# --- collapse: remove the mechanisms that prevent it ------------------------------------------------------------------
add("col_simsiam_no_stopgrad", "collapse", "simsiam", method_kwargs=dict(stop_grad=False))
add("col_simsiam_no_predictor", "collapse", "simsiam", method_kwargs=dict(predictor=False))
add("col_simsiam_no_both", "collapse", "simsiam", method_kwargs=dict(stop_grad=False, predictor=False))
add("col_byol_no_predictor", "collapse", "byol", method_kwargs=dict(predictor=False))
add("col_byol_ema0", "collapse", "byol", method_kwargs=dict(ema=0.0, ema_schedule=False))    # target = online
add("col_byol_ema1", "collapse", "byol", method_kwargs=dict(ema=1.0, ema_schedule=False))    # fixed random target
add("col_vicreg_no_var", "collapse", "vicreg", method_kwargs=dict(var_coeff=0.0))
add("col_vicreg_no_cov", "collapse", "vicreg", method_kwargs=dict(cov_coeff=0.0))
add("col_vicreg_no_var_no_cov", "collapse", "vicreg", method_kwargs=dict(var_coeff=0.0, cov_coeff=0.0))

# --- learning rate of SimSiam (abl_simsiam uses 0.12) -----------------------------------------------------------------
for lr in [0.03, 0.06, 0.24, 0.48]:
    add(f"lr_simsiam_{lr}", "lr", "simsiam", base_lr=lr)

# --- amount of unlabeled data (same number of optimizer steps as abl_simclr) ------------------------------------------
add("data_simclr_10k", "data", "simclr", subset=10_000, epochs=10 * EPOCHS_ABLATION, eval_every=10, warmup_epochs=100)
add("data_simclr_30k", "data", "simclr", subset=30_000, epochs=100, eval_every=5, warmup_epochs=33)

GROUPS = list(dict.fromkeys(group for group, _ in EXPERIMENTS.values()))


def smoke_config(cfg):
    cfg = {**cfg, "epochs": 2, "eval_every": 1, "warmup_epochs": 1, "log_every": 5}
    cfg["subset"] = min(cfg.get("subset") or 4096, 4096)
    cfg["base_lr"] = 0.25 * cfg.get("base_lr", 0.06)      # 8 steps of warmup: keep the learning rate low
    return cfg


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
    parser.add_argument("--gpus", default="0", help="comma-separated GPU ids, one run at a time per GPU")
    parser.add_argument("--groups", default=None, help=f"comma-separated subset of {GROUPS}")
    parser.add_argument("--names", default=None, help="comma-separated run names")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun finished runs")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--data-dir", default="../../data")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    results_dir = args.results_dir or ("results_smoke" if args.smoke else "results")
    logdir = "logdirs_smoke" if args.smoke else "logdirs"

    selected = [(name, group, cfg) for name, (group, cfg) in EXPERIMENTS.items()
                if (args.groups is None or group in args.groups.split(","))
                and (args.names is None or name in args.names.split(","))]
    if args.list:
        for name, group, cfg in selected:
            done = os.path.exists(os.path.join(results_dir, "runs", f"{name}.json"))
            print(f"{'[done]' if done else '[    ]'} {group:14s} {name:28s} {cfg['epochs']:4d} epochs")
        print(f"{len(selected)} runs")
        return
    if args.aggregate:
        print(aggregate(results_dir))
        return

    todo = queue.Queue()
    for name, group, cfg in selected:
        if args.force or not os.path.exists(os.path.join(results_dir, "runs", f"{name}.json")):
            todo.put(smoke_config(cfg) if args.smoke else cfg)
    print(f"{todo.qsize()} runs to do on GPUs {args.gpus}", flush=True)

    def worker(gpu):
        while True:
            try:
                cfg = todo.get_nowait()
            except queue.Empty:
                return
            os.makedirs(os.path.join(logdir, cfg["name"]), exist_ok=True)
            with open(os.path.join(logdir, cfg["name"], "train.log"), "w") as log:
                code = subprocess.run(
                    [sys.executable, "ssl_train.py", "--config", json.dumps(cfg), "--data-dir", args.data_dir,
                     "--results-dir", results_dir, "--logdir", logdir],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, stdout=log, stderr=subprocess.STDOUT,
                ).returncode
            print(f"[gpu {gpu}] {cfg['name']}: {'ok' if code == 0 else f'FAILED (exit code {code})'}", flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    aggregate(results_dir)


if __name__ == "__main__":
    main()
