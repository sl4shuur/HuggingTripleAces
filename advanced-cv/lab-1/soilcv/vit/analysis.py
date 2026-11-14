"""Report one complete Part 2 study with four-fold ensembles."""

import json
import math
from pathlib import Path

import pandas as pd

from soilcv.artifacts import identity_hash, package_identity, write_json
from soilcv.camera import camera_metrics, domain_metrics, paired_camera_errors
from soilcv.checkpoints import load_candidates
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


def read_padding_check(folder, experiment, fold, folds):
    metrics = json.loads((folder / "padding_diagnostic.json").read_text())
    pairs = pd.read_csv(folder / "padding_pairs.csv")
    heldout = set(folds.loc[folds.fold.eq(fold), "sample_id"])
    if (
        pairs.empty
        or pairs.sample_id.duplicated().any()
        or not set(pairs.sample_id) <= heldout
    ):
        raise ValueError(
            "Padding checks must use unique held-out samples from this fold."
        )
    if metrics["padding_samples"] != len(pairs):
        raise ValueError("Padding sample count differs from its saved pair table.")
    values = pairs.padding_emd.to_numpy(dtype=float)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Padding EMD must be finite and non-negative.")
    if not math.isclose(
        metrics["padding_max_emd"], float(values.max()), rel_tol=1e-6, abs_tol=1e-9
    ):
        raise ValueError("Padding summary differs from its saved pair table.")
    return {
        "experiment": experiment,
        "fold": fold,
        "checked_samples": ",".join(pairs.sample_id.astype(str)),
        **metrics,
    }


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
    padding_checks = []
    for recipe in recipes:
        name = recipe["id"]
        oof = []
        best_only = []
        test = []
        cameras = []
        results = []
        padding_rows = []
        for fold in range(4):
            folder, result = read_fold(root, metadata, name, fold)
            results.append(result)
            padding = read_padding_check(folder, name, fold, folds)
            padding_rows.append(padding)
            padding_checks.append(padding)
            oof.append(pd.read_csv(folder / "oof.csv"))
            best_only.append(pd.read_csv(folder / "best_only_oof.csv"))
            test.append(pd.read_csv(folder / "test_predictions.csv"))
            cameras.append(pd.read_csv(folder / "camera_predictions.csv"))
            candidates = load_candidates(folder)
            checkpoints.append(
                {
                    "experiment": name,
                    "fold": fold,
                    "run_id": metadata["run_id"],
                    "wandb_run": result["run_url"],
                    "artifact": result["checkpoint_artifact"],
                    "selected_epochs": ",".join(
                        str(row["epoch"]) for row in candidates
                    ),
                    "selected_checkpoint_count": len(candidates),
                }
            )
        combined = pd.concat(oof, ignore_index=True)
        predictions[name] = combined
        scores = summarize_cv(labels, combined, folds)
        best_combined = pd.concat(best_only, ignore_index=True)
        best_scores = summarize_cv(labels, best_combined, folds)
        best_combined.to_csv(output / f"{name}_best_only_oof.csv", index=False)
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
                "view_mode": recipe.get("view_mode", "multiscale"),
                "local_ppm": recipe.get("pixels", 288) / recipe.get("tile_mm", 64),
                "context_ppm": recipe.get("target_ppm", 2.0),
                "cv_emd": scores["emd"],
                "fold_mean": scores["fold_mean"],
                "fold_std": scores["fold_std"],
                "best_only_cv_emd": best_scores["emd"],
                "best_only_fold_std": best_scores["fold_std"],
                "padding_samples": sum(row["padding_samples"] for row in padding_rows),
                "padding_max_emd": max(row["padding_max_emd"] for row in padding_rows),
                "selected_checkpoints_per_fold": ",".join(
                    str(result["selected_checkpoint_count"]) for result in results
                ),
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
    pd.DataFrame(padding_checks).to_csv(output / "padding_checks.csv", index=False)
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
            "test_prediction": "mean of four folds; each fold uses its same selected 2/3 EMD epochs",
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
        "Each test CSV is the mean of four folds of the same recipe.",
        "Each fold averages its top two validation EMD epochs. Rank three joins only if its camera disagreement is lower than rank two.",
        "The OOF curve uses the exact same selected epochs as test inference.",
        "",
        "| Recipe | Input | Epoch ensemble EMD | Best-only EMD | Fold std | Camera disagreement | GPU hours |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.experiment} | {row.view_mode} | {row.cv_emd:.3f} | {row.best_only_cv_emd:.3f} | "
            f"{row.fold_std:.3f} | {row.camera_disagreement:.3f} | {row.gpu_hours:.3f} |"
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
            "A positive cross-camera gap means the held-out camera has a larger error.",
            "Camera and view can both change. This does not measure iPhone accuracy.",
            "These checks are separate from the standard model ranking.",
            "",
            "## Padding check",
            "",
            "Each fold checks one held-out soil with the best EMD model.",
            "Only the padded canvas changes. The source image content and scale stay the same.",
            "This is a sensitivity check, not a label error or proof of camera robustness.",
            "Per-fold values and checked sample IDs are in results/padding_checks.csv.",
            "",
            "| Recipe | Checked soils | Largest EMD change |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in table.itertuples():
        lines.append(
            f"| {row.experiment} | {row.padding_samples} | {row.padding_max_emd:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Input and limits",
            "",
            "The global view uses full photos at 2 estimated pixels/mm for context.",
            "The local view uses all 64 mm tiles at 288 px, about 4.5 pixels/mm, for detail.",
            "Multiscale models join local and global features before the sample curve head.",
            "Tile-only Swin models use the same local physical grid as CNN MIL.",
            "Spatial coverage measures area, not retained image information.",
            "image_geometry.csv also reports the linear scale ratio and pixel count ratio.",
            "Both views still downsample some source images. No claim of lossless detail is made.",
            "The physical scale still depends on the no-crop assumption in the source data.",
            "Photos keep their shape. Edge padding repeats border pixels.",
            "Scratch attention weights partial patches by real area and masks empty tokens.",
            "Swin final pooling uses real area. Earlier Swin attention can still see padding.",
            "Each fold saves a padding sensitivity check on one held-out soil, using the best EMD model.",
            "Each photo has equal sample weight. One sample curve gets one loss.",
            "",
            "Best checkpoint and early stopping use validation EMD only, from epoch one.",
            "Camera disagreement is logged as a diagnostic. It cannot block a better EMD checkpoint.",
            "Three EMD-ranked weight files are kept. The optional third prediction uses the camera rule above.",
            "Best-only OOF is reported next to the selected epoch ensemble OOF.",
            "The camera rule is a validation-based choice too. These OOF scores can be optimistic.",
            "Large temporary optimizer checkpoints are removed only after a fold succeeds.",
            "All top-three inference weights, selection metadata and OOF curves are kept.",
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
