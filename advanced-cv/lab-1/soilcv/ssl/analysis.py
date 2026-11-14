"""Complete OOF comparisons, paired adaptation effects and publication plots."""

import json
import shutil
from pathlib import Path

import pandas as pd

from soilcv.artifacts import file_hash, write_json
from soilcv.curves import make_submission
from soilcv.metrics import summarize_cv
from soilcv.predictions import mean_curves

from .inference import verify_folder


def save_source(root):
    """Keep the exact code and lockfile beside the GPU results."""
    package = Path(__file__).parents[1]
    destination = Path(root) / "source"
    shutil.copytree(
        package,
        destination / "soilcv",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ["pyproject.toml", "uv.lock", "HYPOTHESES.md", "release.json"]:
        if (package.parent / name).is_file():
            shutil.copy2(package.parent / name, destination / name)


def expected_experiments():
    result = {}
    for source in ["dinov3", "mae", "dinov3-adapted"]:
        encoder = "dinov3" if source.startswith("dinov3") else "mae"
        for strategy in ["knn", "ridge", "lp", "lpft", "full"]:
            result[f"{source}-{strategy}"] = (encoder, source, strategy)
    for strategy in ["ridge-cls", "lpft-uniform-lr", "lpft-no-color"]:
        result[f"dinov3-{strategy}"] = ("dinov3", "dinov3", strategy)
    return result


def collect_results(root):
    root = Path(root)
    labels, template = (
        pd.read_json(root / "labels.json"),
        pd.read_json(root / "template.json"),
    )
    folds = pd.read_csv(root / "sample_folds.csv", dtype={"sample_id": str})
    output = root / "results"
    output.mkdir(exist_ok=True)
    rows, fold_rows, diagnostics, adaptations = [], [], [], []
    for experiment, (encoder, source, strategy) in expected_experiments().items():
        oof, test, costs, feature_costs, run_links = [], [], [], [], []
        for fold in range(4):
            parent = root / "folds" / encoder / f"fold_{fold}"
            receipt = json.loads((parent / "complete.json").read_text())
            folder = parent / source / strategy
            config = verify_folder(folder)
            if (
                config["experiment"] != experiment
                or config["fold"] != fold
                or config["study_id"] != receipt["study_id"]
            ):
                raise ValueError("Mixed studies or held-out folds.")
            oof.append(pd.read_csv(folder / "oof.csv"))
            test.append(pd.read_csv(folder / "test_predictions.csv"))
            costs.append(json.loads((folder / "compute.json").read_text()))
            feature_costs.append(
                json.loads((folder.parent / "feature_compute.json").read_text())
            )
            fold_result = json.loads((folder / "result.json").read_text())
            fold_rows.append(fold_result)
            run_receipt = folder / "wandb_run.json"
            run_links.append(
                json.loads(run_receipt.read_text())["run_url"]
                if run_receipt.exists()
                else receipt["run_url"]
            )
            path = folder / "representation.json"
            if path.exists():
                diagnostics.append(
                    {
                        "experiment": experiment,
                        "fold": fold,
                        **json.loads(path.read_text()),
                    }
                )
        all_oof = pd.concat(oof, ignore_index=True)
        scores = summarize_cv(labels, all_oof, folds)
        all_oof.to_csv(output / f"{experiment}_oof.csv", index=False)
        all_test = mean_curves(test)
        all_test.to_csv(output / f"{experiment}_test.csv", index=False)
        make_submission(
            all_test, template, path=output / f"{experiment}_submission.csv"
        )
        rows.append(
            {
                "experiment": experiment,
                "encoder": encoder,
                "source": source,
                "strategy": strategy,
                "cv_emd": scores["fold_mean"],
                "cv_std": scores["fold_std"],
                "gpu_hours": sum(cost["gpu_hours"] for cost in costs),
                "shared_feature_gpu_hours": sum(
                    cost["gpu_hours"] for cost in feature_costs
                ),
                "peak_gpu_memory_mb": max(
                    cost.get("peak_gpu_memory_mb", 0) for cost in costs
                ),
                "public_lb": None,
                "private_lb": None,
                "runs": ", ".join(run_links),
            }
        )
    table = pd.DataFrame(rows)
    for fold in range(4):
        folder = root / "folds" / "dinov3" / f"fold_{fold}" / "adaptation"
        adaptations.append(
            {
                "fold": fold,
                **json.loads((folder / "compute.json").read_text()),
                **json.loads((folder / "representation.json").read_text()),
            }
        )
    paired = []
    fold_table = pd.DataFrame(fold_rows)
    for strategy in ["knn", "ridge", "lp", "lpft", "full"]:
        base = (
            fold_table[fold_table.experiment.eq(f"dinov3-{strategy}")]
            .set_index("fold")
            .emd
        )
        adapted = (
            fold_table[fold_table.experiment.eq(f"dinov3-adapted-{strategy}")]
            .set_index("fold")
            .emd
        )
        delta = adapted - base
        paired.append(
            {
                "strategy": strategy,
                "adapted_minus_base_emd": float(delta.mean()),
                "paired_fold_std": float(delta.std(ddof=1)),
                "improved_folds": int((delta < 0).sum()),
            }
        )
    table.to_csv(output / "scores.csv", index=False)
    fold_table.to_csv(output / "fold_scores.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "representations.csv", index=False)
    pd.DataFrame(adaptations).to_csv(output / "adaptation_compute.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "adaptation_effects.csv", index=False)
    # Ablations explain a change; the main matrix is the model selection pool.
    main = table[table.strategy.isin(["knn", "ridge", "lp", "lpft", "full"])]
    best = main.sort_values("cv_emd", kind="stable").iloc[0].experiment
    submission = pd.read_csv(output / f"{best}_submission.csv")
    submission.to_csv(root / "submission.csv", index=False)
    report(root, table, pd.DataFrame(paired), adaptations, best)
    plot_results(root, table, pd.DataFrame(diagnostics))
    write_json(
        root / "manifest.json",
        {
            str(path.relative_to(root)): file_hash(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
    )
    return table


def report(root, table, effects, adaptations, best):
    lines = [
        "# Part 3 — DINOv3 + MAE",
        "",
        "Lower EMD is better. Same four physical-sample folds as Parts 1–2.",
        "",
        "| Encoder / source | Strategy | CV EMD ± fold std | Downstream GPU h | Shared feature GPU h | Peak GiB | Public LB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.source} | {row.strategy} | {row.cv_emd:.3f} ± {row.cv_std:.3f} | {row.gpu_hours:.3f} | {row.shared_feature_gpu_hours:.3f} | {row.peak_gpu_memory_mb / 1024:.2f} | not submitted |"
        )
    cost = sum(row["gpu_hours"] for row in adaptations)
    lines += [
        "",
        "## Domain adaptation",
        "",
        f"Four fold-local DINO-style continuations cost **{cost:.3f} GPU h** total. This is shared across adapted strategies, not charged five times.",
        "",
        "| Strategy | Adapted − original EMD | Improved folds / 4 |",
        "|---|---:|---:|",
    ]
    for row in effects.itertuples():
        lines.append(
            f"| {row.strategy} | {row.adapted_minus_base_emd:+.3f} | {row.improved_folds} |"
        )
    scores = table.set_index("experiment").cv_emd
    lines += ["", "## Observed comparisons", ""]
    lines.append(
        f"Frozen ridge: DINOv3 {scores['dinov3-ridge']:.3f}, MAE {scores['mae-ridge']:.3f} EMD."
    )
    for source in ["dinov3", "mae", "dinov3-adapted"]:
        lpft, full, ridge = (
            scores[f"{source}-{strategy}"] for strategy in ["lpft", "full", "ridge"]
        )
        lines.append(
            f"{source}: LP-FT {lpft:.3f}, full FT {full:.3f}, frozen ridge {ridge:.3f}. "
            f"Full FT minus ridge is {full - ridge:+.3f}; LP-FT minus full FT is {lpft - full:+.3f} EMD."
        )
    for variant, reference, hypothesis in [
        ("ridge-cls", "ridge", "H-SSL-04"),
        ("lpft-uniform-lr", "lpft", "H-SSL-05"),
        ("lpft-no-color", "lpft", "H-SSL-06"),
    ]:
        delta = scores[f"dinov3-{variant}"] - scores[f"dinov3-{reference}"]
        lines.append(
            f"{hypothesis}: {variant} minus {reference} = {delta:+.3f} EMD. "
            "Positive favours the reference; negative favours the ablation."
        )
    lines += [
        "",
        "Negative differences favour adaptation. With only four folds, treat differences as descriptive; no significance claim.",
        "",
        "## Protocol and interpretation",
        "",
        "All 64 mm grid tiles are resized directly from native crops to 224 px. Area weights correct edge overlap within each photo; photos have equal sample weight. PPM still assumes full-image resizing without unknown cropping. No foreground mask is available, so background remains a possible shortcut.",
        "",
        "DINOv3 uses CLS + mean patch tokens, excluding register tokens. MAE uses all patches with deterministic raster order and mask_ratio=0. Frozen ridge fits its scaler on training soils only (alpha=100); cosine k-NN uses k=3, temperature=.07. LP fits a linear mass-bin projection with EMD; LP-FT starts from its best LP head. Full FT starts with a fresh head. Both FT variants update the full encoder with AdamW, warmup/cosine and layer decay .75.",
        "",
        "Adaptation sees only the 18 training soils of its fold. Neither held-out nor test photos enter SSL. It uses two scale-preserving full-tile views, an EMA teacher, centering and sharpening, and a fixed final epoch. This is a small DINO-style continuation, not the original DINOv3 recipe. Inspect loss together with entropy and feature variation: low loss alone can mean collapse.",
        "",
        "Random zoom / resized crops without corrected scale change apparent grain size; strong blur removes fine grains. MixUp/CutMix targets are not justified for soil mass fractions. Augmentation therefore uses quarter-turns, flips and mild color changes only. The no-color ablation tests the latter.",
        "",
        "Fine-tuning can worsen validation when 18 independent soils are insufficient or the random head distorts transferable features. CKA is computed on the exact same deterministic held-out tiles before/after tuning, with tile IDs saved. Higher CKA measures representation preservation, not accuracy or causality. Multiple tiles from one soil are correlated; they are not independent evidence. Constant features give undefined CKA rather than a misleading zero or one.",
        "",
        f"Selected by main-matrix CV: **{best}**. Its four-fold test average is `submission.csv`. Public/private LB values are absent until actual competition receipts are available. No score is invented. Part 2 used an epoch ensemble; Part 3 uses a single best epoch, so comparisons do not isolate encoder changes alone.",
        "",
        "Shared feature extraction is reported separately; downstream costs do not include it or adaptation. LP-FT also requires the reported LP stage. Peak GPU memory is allocated memory, not reserved memory. Frozen fits run on CPU; their timer includes only that stage.",
        "",
        "## Reproducibility",
        "",
        "Exact official Hub revisions are recorded in metadata.json; settings.json, environment.json, per-fold config/weights, numerical regressors, histories, OOF, test curves and file hashes accompany the study. Checkpoints and run URLs are in each fold's complete.json. No additional labelled data or pseudo-labels were used.",
        "",
        "AI assistance: Codex helped implement and test Part 3. Keep this chat transcript with the submission and review the methods before the oral defence. The pre-run hypotheses were written before GPU runs; no git commit was created, at the project owner's request.",
        "",
    ]
    (root / "PART3_RESULTS.md").write_text("\n".join(lines))


def plot_results(root, table, diagnostics):
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"dinov3": "#2764a5", "mae": "#d28a27", "dinov3-adapted": "#178577"}
    strategies = ["knn", "ridge", "lp", "lpft", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for source, color in colors.items():
        rows = (
            table[table.source.eq(source) & table.strategy.isin(strategies)]
            .set_index("strategy")
            .loc[strategies]
        )
        axes[0].errorbar(
            strategies,
            rows.cv_emd,
            yerr=rows.cv_std,
            label=source,
            color=color,
            marker="o",
            capsize=3,
        )
    axes[0].set(
        xlabel="Tuning strategy",
        ylabel="Validation EMD (mean ± fold std)",
        title="Same soils, same physical scale",
    )
    axes[0].legend(frameon=False)
    for source, color in colors.items():
        names = set(table.loc[table.source.eq(source), "experiment"])
        rows = diagnostics[diagnostics.experiment.isin(names)]
        rows = rows[rows.experiment.str.endswith(("-lp", "-lpft", "-full"))]
        if not rows.empty:
            merged = rows.merge(table[["experiment", "cv_emd"]], on="experiment")
            axes[1].scatter(
                merged.cka, merged.cv_emd, color=color, alpha=0.7, label=source
            )
    axes[1].set(
        xlabel="Held-out tile CKA before / after",
        ylabel="Experiment CV EMD",
        title="Preservation does not imply quality",
        xlim=(0, 1.03),
    )
    fig.savefig(root / "results" / "transfer_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    for fold in range(4):
        history = pd.read_json(
            root / "folds" / "dinov3" / f"fold_{fold}" / "adaptation" / "history.json"
        )
        for axis, metric in zip(
            axes, ["loss", "teacher_entropy", "feature_std"], strict=True
        ):
            axis.plot(history.epoch, history[metric], label=f"fold {fold}")
            axis.set(xlabel="Adaptation epoch", ylabel=metric.replace("_", " "))
    axes[0].legend(frameon=False)
    fig.savefig(root / "results" / "adaptation_curves.png", dpi=180)
    plt.close(fig)
