"""All training runs of Lecture 5 and a launcher that runs them one after another on the given GPUs.

    python experiments.py --list                      # show all runs
    python experiments.py --gpus 0                    # run everything that is not finished yet
    python experiments.py --gpus 0 --groups main      # only some groups
    python experiments.py --gpus 0 --names dino       # only some runs
    python experiments.py --gpus 0 --smoke            # 1-epoch versions into results_smoke/ (code check)
    python experiments.py --aggregate                 # rebuild results/summary.csv from the finished runs

Finished runs (those with `results/runs/<name>.json`) are skipped, so the launcher can be stopped and restarted.
The MoCo v3 run of Lecture 4 (the same ViT-Ti/8, the same data and budget) is imported as a reference if
`../Lecture_4/results` and `../Lecture_4/logdirs` are available.
"""
import argparse
import glob
import json
import os
import queue
import shutil
import subprocess
import sys
import threading

import pandas as pd

EPOCHS_MAIN = 100       # the comparison of methods
EPOCHS_ABLATION = 30    # every ablation run

# Method defaults (ViT-Ti/8, AdamW, cosine schedule; see train.DEFAULT_CONFIG for the rest). The learning rate and
# the teacher momentum were chosen in short pilot runs (20 epochs); everything else follows the papers, scaled down.
METHOD_CONFIGS = {
    "dino": dict(method="dino", base_lr=5e-4, method_kwargs=dict(ema=0.99, teacher_temp=0.04), views={}),
    # I-JEPA sees one view per image, made with a random crop and a flip only
    "ijepa": dict(method="ijepa", base_lr=5e-4, method_kwargs=dict(ema=0.99),
                  views=dict(crop_scale=(0.3, 1.0), jitter_p=0.0, gray_p=0.0, blur_p=0.0)),
}

EXPERIMENTS = {}


def add(name, group, base, epochs=EPOCHS_ABLATION, method_kwargs=None, views=None, **overrides):
    cfg = {**METHOD_CONFIGS[base], "name": name, "epochs": epochs, **overrides}
    cfg["method_kwargs"] = {**METHOD_CONFIGS[base]["method_kwargs"], **(method_kwargs or {})}
    cfg["views"] = {**METHOD_CONFIGS[base]["views"], **(views or {})}
    cfg.setdefault("eval_every", 5 if epochs > 50 else 1)
    assert name not in EXPERIMENTS, name
    EXPERIMENTS[name] = (group, cfg)


# --- main: both methods with the full budget ----------------------------------------------------------------------------
add("dino", "main", "dino", epochs=EPOCHS_MAIN, artifacts=True)
add("ijepa", "main", "ijepa", epochs=EPOCHS_MAIN, artifacts=True)
# supervised reference: the same ViT trained with labels on the 5k labeled images (300 epochs = 2700 steps)
EXPERIMENTS["supervised_5k"] = ("main", dict(
    name="supervised_5k", method="supervised", method_kwargs={}, views={}, train_split="train", epochs=300,
    warmup_epochs=30, eval_every=20, artifacts=True))

# --- DINO ablations ---------------------------------------------------------------------------------------------------
add("abl_dino", "dino", "dino")
add("dino_no_centering", "dino", "dino", method_kwargs=dict(centering=False))
add("dino_no_sharpening", "dino", "dino", method_kwargs=dict(teacher_temp=0.1))       # teacher as soft as the student
add("dino_no_centering_no_sharpening", "dino", "dino", method_kwargs=dict(centering=False, teacher_temp=0.1))
add("dino_ema0", "dino", "dino", method_kwargs=dict(ema=0.0, ema_schedule=False))       # teacher = student
add("dino_no_multicrop", "dino", "dino", views=dict(n_local=0))

# --- I-JEPA ablations -------------------------------------------------------------------------------------------------
add("abl_ijepa", "ijepa", "ijepa")
add("ijepa_ema0", "ijepa", "ijepa", method_kwargs=dict(ema=0.0, ema_schedule=False))    # target encoder = context encoder
add("ijepa_pixels", "ijepa", "ijepa", method_kwargs=dict(target="pixels"))              # predict pixels (MAE-style)
add("ijepa_random_mask", "ijepa", "ijepa", method_kwargs=dict(mask="random"))           # scattered targets

GROUPS = list(dict.fromkeys(group for group, _ in EXPERIMENTS.values()))
REFERENCE = "mocov3_vit_tiny"                           # run of Lecture 4
LECTURE_4 = os.path.join("..", "Lecture_4")


def smoke_config(cfg):
    return {**cfg, "epochs": 1, "eval_every": 1, "warmup_epochs": 1, "log_every": 5}


def import_reference(results_dir):
    """Copies the logs of the MoCo v3 ViT-Ti/8 run of Lecture 4 and computes the same pictures for its backbone."""
    src, weights = os.path.join(LECTURE_4, "results", "runs"), os.path.join(LECTURE_4, "logdirs", REFERENCE, "backbone.pt")
    if not os.path.exists(os.path.join(src, f"{REFERENCE}.json")):
        print(f"reference run {REFERENCE} not found in {src}, skipping")
        return
    os.makedirs(os.path.join(results_dir, "runs"), exist_ok=True)
    for suffix in (".json", ".csv"):
        shutil.copy(os.path.join(src, REFERENCE + suffix), os.path.join(results_dir, "runs", REFERENCE + suffix))
    if os.path.exists(weights) and not os.path.exists(os.path.join(results_dir, "artifacts", f"{REFERENCE}.npz")):
        import torch
        from lecture_utils import get_device
        from self_distillation import ViTEncoder
        from train import Evaluator, save_artifacts
        device = get_device()
        encoder = ViTEncoder().to(device)
        encoder.load_state_dict(torch.load(weights, map_location=device))
        save_artifacts(REFERENCE, encoder, Evaluator("../../data", device), os.path.join(results_dir, "artifacts"))


def aggregate(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "runs", "*.json"))):
        with open(path) as f:
            row = json.load(f)
        row.pop("config")
        row["group"] = EXPERIMENTS[row["name"]][0] if row["name"] in EXPERIMENTS else "reference"
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
            print(f"{'[done]' if done else '[    ]'} {group:8s} {name:34s} {cfg['epochs']:4d} epochs")
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
                    [sys.executable, "train.py", "--config", json.dumps(cfg), "--data-dir", args.data_dir,
                     "--results-dir", results_dir, "--logdir", logdir],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, stdout=log, stderr=subprocess.STDOUT,
                ).returncode
            print(f"[gpu {gpu}] {cfg['name']}: {'ok' if code == 0 else f'FAILED (exit code {code})'}", flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not args.smoke:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.split(",")[0]
        import_reference(results_dir)
    aggregate(results_dir)


if __name__ == "__main__":
    main()
