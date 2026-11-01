"""Report one complete Part 2 study with four-fold ensembles."""

import json
from pathlib import Path

import pandas as pd

from soilcv.artifacts import identity_hash, package_identity, write_json
from soilcv.camera import camera_metrics, domain_metrics, paired_camera_errors
from soilcv.curves import make_submission
from soilcv.inference.runner import ensemble_predictions
from soilcv.metrics import summarize_cv


def read_fold(root, metadata, experiment, fold, *, train_camera=None):
    from .experiments import verify_completed

    purpose = "standard" if train_camera is None else "domain"
    folder = root / "runs" / experiment / f"fold_{fold}"
    base = experiment if train_camera is None else experiment.rsplit("__", 1)[0]
    recipe = next(
        row
        for row in json.loads((root / "recipes.json").read_text())
        if row["id"] == base
    )
    recipe_hash = identity_hash(
        {
            "recipe": recipe,
            "identity": metadata["identity"],
            "fold": fold,
            "train_camera": train_camera,
        }
    )
    result = verify_completed(folder, recipe_hash)
    expected = {
        "experiment": base,
        "fold": fold,
        "run_id": metadata["run_id"],
        "purpose": purpose,
        "train_camera": train_camera,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Mixed study, fold or purpose: {folder}.")
    return folder, result


def check_camera_folds(camera, folds):
    if camera.empty or camera.duplicated(["sample_id", "device"]).any():
        raise ValueError("Camera OOF must visit each held-out sample/device once.")
    expected = folds.set_index("sample_id").fold
    if not set(camera.sample_id).issubset(expected.index):
        raise ValueError("Camera OOF contains unknown samples.")
    if "fold" not in camera or not camera.fold.eq(camera.sample_id.map(expected)).all():
        raise ValueError("Camera curves have the wrong held-out fold.")


def collect_results(root=None, recipes=None):
    from .experiments import DOMAIN_BASELINE, RECIPES, ROOT

    root = ROOT if root is None else Path(root)
    recipes = RECIPES if recipes is None else recipes
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata["identity"]["soilcv"] != package_identity():
        raise ValueError("Package changed after this study was prepared.")
    if json.loads((root / "recipes.json").read_text()) != recipes:
        raise ValueError("Study recipes changed.")
    status = json.loads((root / "status.json").read_text())
    expected_jobs = 4 * len(recipes) + 8
    if (
        status["failed"]
        or len(status["jobs"]) != expected_jobs
        or any(job["state"] != "complete" for job in status["jobs"])
    ):
        raise RuntimeError("Wait for every standard and camera diagnostic fold.")
    labels = pd.read_json(root / "labels.json")
    template = pd.read_json(root / "template.json")
    folds = pd.read_csv(root / "sample_folds.csv")
    output = root / "results"
    output.mkdir(exist_ok=True)
    rows = []
    checkpoints = []
    camera_rows = []
    pair_rows = []
    predictions = {}
    for recipe in recipes:
        name = recipe["id"]
        oof = []
        test = []
        cameras = []
        results = []
        for fold in range(4):
            folder, result = read_fold(root, metadata, name, fold)
            results.append(result)
            oof.append(pd.read_csv(folder / "oof.csv"))
            test.append(pd.read_csv(folder / "test_predictions.csv"))
            cameras.append(pd.read_csv(folder / "camera_predictions.csv"))
            checkpoints.append(
                {
                    "experiment": name,
                    "fold": fold,
                    "run_id": metadata["run_id"],
                    "wandb_run": result["run_url"],
                    "artifact": result["checkpoint_artifact"],
                }
            )
        combined = pd.concat(oof, ignore_index=True)
        predictions[name] = combined
        scores = summarize_cv(labels, combined, folds)
        combined.to_csv(output / f"{name}_oof.csv", index=False)
        submission = ensemble_predictions(test, template)
        make_submission(submission, template, output / f"{name}_submission.csv")
        camera = pd.concat(cameras, ignore_index=True)
        check_camera_folds(camera, folds)
        metrics = camera_metrics(labels, camera, require_pairs=True)
        pairs = paired_camera_errors(labels, camera)
        pairs["experiment"] = name
        pair_rows.append(pairs)
        camera["experiment"] = name
        camera_rows.append(camera)
        rows.append(
            {
                "experiment": name,
                "run_id": metadata["run_id"],
                "arch": recipe["arch"],
                "cv_emd": scores["emd"],
                "fold_mean": scores["fold_mean"],
                "fold_std": scores["fold_std"],
                **metrics,
                "gpu_hours": sum(result["gpu_hours"] for result in results),
                "peak_gpu_mb": max(result["peak_gpu_memory_mb"] for result in results),
                "epochs_per_fold": ",".join(
                    str(result["epochs_ran"]) for result in results
                ),
                "best_epochs": ",".join(
                    str(result["best_epoch"]) for result in results
                ),
                "public_lb": "not submitted",
                "private_lb": "not available",
            }
        )
    table = pd.DataFrame(rows).sort_values("cv_emd").reset_index(drop=True)
    table.to_csv(output / "vit_results.csv", index=False)
    pd.DataFrame(checkpoints).to_csv(output / "vit_checkpoints.csv", index=False)
    write_json(
        output / "inference_ensembles.json",
        [
            {
                "name": recipe["id"],
                "checkpoints": [
                    item["artifact"]
                    for item in checkpoints
                    if item["experiment"] == recipe["id"]
                ],
            }
            for recipe in recipes
        ],
    )
    pd.concat(camera_rows, ignore_index=True).to_csv(
        output / "camera_oof.csv", index=False
    )
    pd.concat(pair_rows, ignore_index=True).to_csv(
        output / "camera_pairs.csv", index=False
    )
    domain = collect_domain(root, metadata, DOMAIN_BASELINE, labels, folds, output)
    best = table.iloc[0]
    make_submission(
        pd.read_csv(output / f"{best.experiment}_submission.csv"),
        template,
        root / "submission.csv",
    )
    write_json(
        output / "selection.json",
        {
            "run_id": metadata["run_id"],
            "selected_experiment": best.experiment,
            "cv_emd": float(best.cv_emd),
            "camera_disagreement": float(best.camera_disagreement),
            "selection_rule": "lowest standard sample OOF EMD; camera metric reported separately",
            "test_prediction": "equal mean of four fold curves of one recipe",
            "competition_submission_made": False,
        },
    )
    write_report(root, metadata, table, domain)
    return table, predictions, domain


def collect_domain(root, metadata, baseline, labels, folds, output):
    rows = []
    pair_rows = []
    all_cameras = []
    for train_camera in ("Motorola", "Samsung"):
        name = f"{baseline}__{train_camera.lower()}"
        cameras = []
        results = []
        for fold in range(4):
            folder, result = read_fold(
                root, metadata, name, fold, train_camera=train_camera
            )
            results.append(result)
            cameras.append(pd.read_csv(folder / "camera_predictions.csv"))
        combined = pd.concat(cameras, ignore_index=True)
        check_camera_folds(combined, folds)
        metrics, pairs = domain_metrics(labels, combined, train_camera)
        pairs["train_camera"] = train_camera
        pairs["fold"] = pairs.sample_id.map(folds.set_index("sample_id").fold)
        pair_rows.append(pairs)
        combined["train_camera"] = train_camera
        all_cameras.append(combined)
        rows.append(
            {
                "experiment": baseline,
                "run_id": metadata["run_id"],
                "train_camera": train_camera,
                **metrics,
                "gpu_hours": sum(result["gpu_hours"] for result in results),
                "peak_gpu_mb": max(result["peak_gpu_memory_mb"] for result in results),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "domain_results.csv", index=False)
    pd.concat(pair_rows, ignore_index=True).to_csv(
        output / "domain_pairs.csv", index=False
    )
    pd.concat(all_cameras, ignore_index=True).to_csv(
        output / "domain_camera_oof.csv", index=False
    )
    return table


def write_report(root, metadata, table, domain):
    best = table.iloc[0]
    lines = [
        "# ViT results",
        "",
        f"Run ID: `{metadata['run_id']}`.",
        f"Code hash: `{metadata['identity']['soilcv']['source_hash']}`.",
        "",
        "The same four sample folds are used for every model. Lower EMD is better.",
        "Each test CSV is the mean of four fold models of the same recipe.",
        "",
        "| Recipe | OOF EMD | Fold std | Camera gap | Fit GPU hours | Peak MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.experiment} | {row.cv_emd:.3f} | {row.fold_std:.3f} | "
            f"{row.camera_disagreement:.3f} | {row.gpu_hours:.3f} | {row.peak_gpu_mb:.0f} |"
        )
    lines.extend(
        [
            "",
            f"The best standard OOF recipe is {best.experiment}.",
            "Camera gap means EMD between Motorola and Samsung predictions for the same held-out soil.",
            "It is not a label error. The raw pairs are in results/camera_pairs.csv.",
            "",
            "## Cross-camera check",
            "",
            "These models train on one camera only. The physical sample folds stay the same.",
            "Both errors use the same held-out soils with both cameras.",
            "",
            "| Training camera | Paired soils | Same EMD | Cross EMD | Cross minus same |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in domain.itertuples():
        lines.append(
            f"| {row.train_camera} | {row.domain_paired_samples} | {row.e_same:.3f} | "
            f"{row.e_cross:.3f} | {row.delta_domain:.3f} |"
        )
    lines.extend(
        [
            "",
            "A positive gap means the held-out camera has a larger error.",
            "Camera and view can both change. This does not measure iPhone accuracy.",
            "These checks are separate from the standard model ranking.",
            "",
            "## Input and limits",
            "",
            "Each full photo is resized once to one common estimated pixels-per-mm scale.",
            "No source area is cut out. Small grains can still be lost by downsampling.",
            "The physical scale still depends on the no-crop assumption in the source data.",
            "Photos keep their shape. Padding only fits the patch grid.",
            "Each photo has equal sample weight. One sample curve gets one loss.",
            "",
            "Best epochs use two separate metrics: sample EMD and camera disagreement.",
            "The first three epochs are a fixed burn-in. Selection starts at epoch four.",
            "A new checkpoint must not be worse on either metric and must improve one.",
            "This strict rule can keep an early model. It does not use a weighted score.",
            "Both validation cameras affect epoch selection in the camera stress test.",
            "The second camera is absent from training updates, but not from model selection.",
            "We still select epochs on few held-out soils. CV can be optimistic.",
            "Fold std is not a confidence interval. The leaderboard uses few test soils.",
            "Public LB: not submitted. Private LB: not available.",
            "AI helped write this code. Read it before the defense.",
        ]
    )
    path = root / "VIT_RESULTS.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def plot_scores(table):
    import matplotlib.pyplot as plt

    ordered = table.sort_values("cv_emd", ascending=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].barh(ordered.experiment, ordered.cv_emd, xerr=ordered.fold_std, capsize=3)
    axes[0].set(xlabel="Sample EMD", title="Four fixed folds")
    axes[1].barh(ordered.experiment, ordered.camera_disagreement)
    axes[1].set(xlabel="Camera disagreement EMD", title="Same held-out soils")
    figure.tight_layout()
    return figure


def plot_learning(root=None, recipe=None):
    import matplotlib.pyplot as plt

    from .experiments import RECIPES, ROOT

    root = ROOT if root is None else Path(root)
    recipe = RECIPES[0]["id"] if recipe is None else recipe
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for fold in range(4):
        history = json.loads(
            (root / "runs" / recipe / f"fold_{fold}" / "history.json").read_text()
        )
        epochs = [row["epoch"] + 1 for row in history]
        for axis, name in zip(axes, ("emd", "camera_disagreement")):
            axis.plot(
                epochs, [row["val"][name] for row in history], label=f"fold {fold}"
            )
            axis.set(xlabel="Epoch", ylabel=name, title=recipe)
            axis.legend()
    figure.tight_layout()
    return figure
