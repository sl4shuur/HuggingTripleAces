"""Compose cumulative acquisition, fresh-head training and val reporting."""

import importlib.metadata
import os
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import torch

from alseg.adapters.voc import VocEvaluation, VocImages, VocOracle
from alseg.config import TrackingConfig, get_project_version
from alseg.core.helpers import check_ids, write_csv, write_json
from alseg.core.pool import PoolManifest
from alseg.core.predictions import IGNORE_INDEX, NUM_CLASSES
from alseg.core.selection import AcquisitionMethod
from alseg.data.preprocessing import resize_mask
from alseg.data.voc import find_voc_root, load_manifest
from alseg.interfaces.data import ImageSource, Oracle
from alseg.interfaces.encoder import FeatureSource
from alseg.nn.encoders import load_encoder
from alseg.nn.evaluation import evaluate_head
from alseg.nn.extraction import iter_features
from alseg.nn.training import fit_head
from alseg.runtime.tracking import (
    configure_tracking,
    log_experiment_history,
    log_run_artifacts,
    wandb_run,
)


def _validate_training_config(config: dict, device: str) -> None:
    task_config, data_config = config["task"], config["data"]
    if task_config["head"] != "linear_1x1" or not task_config["retrain_from_scratch"]:
        raise ValueError("This recipe requires a fresh linear_1x1 head")
    if device not in ("cpu", "cuda") or (
        device == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("Requested training device is unavailable")
    if (
        data_config["num_classes"] != NUM_CLASSES
        or data_config["ignore_index"] != IGNORE_INDEX
    ):
        raise ValueError("This task requires 21 classes and ignore_index=255")


def prepare_labeled_features(
    config: dict,
    pool: ImageSource,
    oracle: Oracle,
    encoder: FeatureSource,
    ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare patch features and previously queried masks in the supplied ID order."""
    encoder_config = config["encoder"]
    features = torch.cat(
        [
            batch.patch_features
            for _, batch in iter_features(
                pool, encoder, ids, encoder_config["batch_size"]
            )
        ]
    )
    targets = torch.stack(
        [
            resize_mask(oracle.get_mask(image_id), encoder_config["image_size"])
            for image_id in ids
        ]
    )
    return features, targets


def _build_run_config(
    config: dict,
    tracking: TrackingConfig,
    manifest: PoolManifest,
    encoder: FeatureSource,
    selection: dict,
    device: str,
    smoke: bool,
) -> dict:
    return {
        **config,
        "tracking": tracking.model_dump(mode="json"),
        "encoder_identity": encoder.describe(),
        "budget": selection["budget"],
        "round": selection["round"],
        "n_new": len(selection["new_ids"]),
        "n_fit": len(selection["fit_ids"]),
        "n_calibration": len(selection["calibration_ids"]),
        "selected_ids": selection["ids"],
        "new_ids": selection["new_ids"],
        "fit_ids": selection["fit_ids"],
        "calibration_ids": selection["calibration_ids"],
        "acquisition_metadata": selection["acquisition"],
        "pool_ids": manifest.pool_ids,
        "val_ids": manifest.val_ids,
        **{
            f"{stream}_seed": config["seed"]
            + config["protocol"][f"{stream}_seed_offset"]
            + (selection["round"] if stream == "selection" else 0)
            for stream in ("selection", "training", "calibration")
        },
        "val_images": min(8, len(manifest.val_ids)) if smoke else len(manifest.val_ids),
        "smoke": smoke,
        "project_version": get_project_version(),
        "runtime": {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch": torch.__version__,
            **{
                name: importlib.metadata.version(name)
                for name in ("lightning", "wandb", "transformers", "timm")
            },
            "precision": "32-true",
        },
    }


def _save_results(
    run_dir: Path, run_config: dict, metrics: dict, timings: dict
) -> dict:
    write_json(run_dir / "metrics.json", metrics)
    acquisition = run_config["acquisition_metadata"]
    result = {
        "method": acquisition["method"],
        "selection_method": acquisition["selection_method"],
        "acquisition_budget": acquisition["acquisition_budget"],
        "acquisition_n_fit": len(acquisition.get("calibration_fit_ids", [])),
        "acquisition_n_calibration": len(acquisition.get("calibration_ids", [])),
        "sigma_used": acquisition.get("sigma"),
        "tau_used": acquisition.get("tau", 1.0),
        "fallback": acquisition.get("fallback"),
        "encoder": run_config["encoder_identity"].get(
            "name", run_config["encoder_identity"].get("model_id")
        ),
        **{
            name: run_config[name]
            for name in (
                "budget",
                "round",
                "n_new",
                "seed",
                "n_fit",
                "n_calibration",
                "smoke",
                "val_images",
            )
        },
        "epochs": run_config["task"]["epochs"],
        **{
            name: metrics[name]
            for name in ("miou", "pixel_accuracy", "mean_class_accuracy")
        },
        "val_loss": metrics["loss"],
        **timings,
    }
    write_csv(run_dir / "results.csv", [result])
    return result


def _fit_and_save_head(
    features: torch.Tensor,
    targets: torch.Tensor,
    run_config: dict,
    run_dir: Path,
    logger,
) -> torch.nn.Module:
    task = fit_head(
        features,
        targets,
        run_config["task"],
        run_config["training_seed"],
        run_config["runtime"]["device"],
        logger,
    )
    write_csv(run_dir / "training.csv", task.history)
    torch.save(
        {
            "state_dict": task.head.state_dict(),
            "feature_dim": features.shape[1],
            "encoder": run_config["encoder_identity"],
        },
        run_dir / "head.pt",
    )
    return task.head


def run_training(
    config: dict,
    run_dir: Path,
    budget: int,
    device: str,
    tracking: TrackingConfig,
    smoke: bool = False,
) -> dict:
    """Legacy single-budget recipe: fit all L, without a calibration holdout."""
    return _run_experiment(
        config, run_dir, [budget], device, tracking, smoke, single=True
    )[0]


def run_experiment(
    config: dict,
    run_dir: Path,
    device: str,
    tracking: TrackingConfig,
    smoke: bool = False,
    *,
    acquisition: Callable[..., tuple[list[int], dict]] | None = None,
) -> list[dict]:
    """Train every reporting head on all queried L, regardless of acquisition.

    The optional acquisition callback receives keyword arguments config, pool,
    oracle, encoder, round_index and the cumulative target budget. It returns
    (new_ids, metadata), without calling Oracle.query or mutating its inputs.
    It may read only previously queried masks through oracle.get_mask.
    """
    return _run_experiment(
        config,
        run_dir,
        config["budgets"],
        device,
        tracking,
        smoke,
        acquisition=acquisition,
    )


def _query_round(
    config: dict,
    oracle: Oracle,
    unlabeled_ids: list[int],
    round_index: int,
    budget: int,
    *,
    acquisition_result: tuple[list[int], dict] | None = None,
) -> dict:
    """Validate the entire selection before querying; reporting always fits all L."""
    previous_budget = len(oracle.labeled_ids)
    n_new = budget - previous_budget
    if not 0 < n_new <= len(unlabeled_ids):
        raise ValueError("budget must add labels without exceeding the unlabeled pool")
    if acquisition_result is None:
        protocol = config["protocol"]
        seed = config["seed"] + protocol["selection_seed_offset"] + round_index
        order = torch.randperm(
            len(unlabeled_ids), generator=torch.Generator().manual_seed(seed)
        )
        new_ids = [unlabeled_ids[i] for i in order[:n_new].tolist()]
        metadata = {"method": AcquisitionMethod.RANDOM.value}
    else:
        new_ids, metadata = acquisition_result
    check_ids(new_ids, "selected_ids")
    if len(new_ids) != n_new:
        raise ValueError("Acquisition returned the wrong number of new IDs")
    if not set(new_ids) <= set(unlabeled_ids):
        raise ValueError("Acquisition returned an already-labeled or out-of-pool ID")
    if not isinstance(metadata, dict):
        raise ValueError("Acquisition metadata must be a dictionary")
    method = AcquisitionMethod(
        metadata.get("method", config["acquisition"]["method"])
    ).value
    metadata = {
        "method": method,
        "selection_method": method,
        "acquisition_budget": previous_budget,
        **metadata,
    }
    oracle.query(new_ids)
    labeled_ids = oracle.labeled_ids
    return {
        "round": round_index,
        "budget": budget,
        "new_ids": new_ids,
        "ids": labeled_ids,
        "unlabeled_ids": sorted(set(unlabeled_ids) - set(new_ids)),
        "fit_ids": labeled_ids,
        "calibration_ids": [],
        "acquisition": metadata,
    }


def _run_round(
    config: dict,
    tracking: TrackingConfig,
    pool: ImageSource,
    oracle: Oracle,
    encoder: FeatureSource,
    evaluation: VocEvaluation,
    run_dir: Path,
    selection_seconds: float,
) -> dict:
    """Extract fit features, train a fresh head, evaluate val and save round artifacts."""
    start = time.perf_counter()
    features, targets = prepare_labeled_features(
        config, pool, oracle, encoder, config["fit_ids"]
    )
    timings = {
        "selection_seconds": selection_seconds,
        "feature_seconds": time.perf_counter() - start,
    }
    name = f"{config['experiment_name']}-b{config['budget']}"
    with wandb_run(tracking, run_dir, name, config) as logger:
        start = time.perf_counter()
        head = _fit_and_save_head(features, targets, config, run_dir, logger)
        timings["train_seconds"] = time.perf_counter() - start
        del features, targets
        start = time.perf_counter()
        metrics, previews = evaluate_head(
            head,
            encoder,
            evaluation,
            list(range(config["val_images"])),
            config["encoder"]["batch_size"],
            config["runtime"]["device"],
        )
        timings["eval_seconds"] = time.perf_counter() - start
        result = _save_results(run_dir, config, metrics, timings)
        if logger:
            logger.log_metrics({f"perf/{key}": value for key, value in timings.items()})
        log_run_artifacts(logger, run_dir, metrics, previews)
    return result


def _save_round_inputs(
    run_dir: Path,
    round_dir: Path,
    config: dict,
    selection: dict,
    selections: list[dict],
) -> None:
    """Persist queried IDs before any training starts."""
    round_dir.mkdir(exist_ok=round_dir == run_dir)
    write_json(round_dir / "config.json", config)
    write_json(round_dir / "selection.json", selection)
    write_json(run_dir / "selections.json", selections)
    write_json(run_dir / "state.json", {**selection, "status": "queried"})


def _run_experiment(
    config: dict,
    run_dir: Path,
    budgets: list[int],
    device: str,
    tracking: TrackingConfig,
    smoke: bool,
    *,
    single: bool = False,
    acquisition: Callable[..., tuple[list[int], dict]] | None = None,
) -> list[dict]:
    config = {**config, "device": device}
    if (
        not single
        and acquisition is None
        and AcquisitionMethod(config["acquisition"]["method"])
        != AcquisitionMethod.RANDOM
    ):
        raise ValueError("Non-random acquisition requires an acquisition callback")
    tracking = configure_tracking(tracking)
    _validate_training_config(config, device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    data_config = config["data"]
    root = (
        Path(data_config["voc_root"])
        if data_config["voc_root"]
        else find_voc_root(Path(data_config["dataset_root"]))
    )
    manifest = load_manifest(root)
    pool = VocImages(root, manifest)
    if (
        not budgets
        or any(type(b) is not int or not 0 < b <= len(pool) for b in budgets)
        or any(a >= b for a, b in pairwise(budgets))
    ):
        raise ValueError(
            "budgets must be strictly increasing and within the train pool size"
        )
    if run_dir.exists():
        raise ValueError(
            "Use a new output directory; existing artifacts are never overwritten"
        )
    encoder = load_encoder(
        config["encoder"], Path(config["paths"]["cache_dir"]), device
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    oracle = VocOracle(root, manifest, run_dir / "oracle_queries.jsonl")
    manifest_data = {"pool_ids": manifest.pool_ids, "val_ids": manifest.val_ids}
    manifest_hash = manifest.sha256
    write_json(run_dir / "manifest.json", {**manifest_data, "sha256": manifest_hash})
    if not single:
        write_json(
            run_dir / "config.json",
            {**config, "tracking": tracking.model_dump(mode="json")},
        )
    evaluation = VocEvaluation(root, manifest)
    unlabeled_ids = list(range(len(pool)))
    results, selections = [], []
    for round_index, budget in enumerate(budgets):
        start = time.perf_counter()
        acquisition_result = (
            acquisition(
                config=config,
                pool=pool,
                oracle=oracle,
                encoder=encoder,
                round_index=round_index,
                budget=budget,
            )
            if acquisition is not None
            else None
        )
        if acquisition is not None and acquisition_result is None:
            raise ValueError("Acquisition must return (selected_ids, metadata)")
        selection = _query_round(
            config,
            oracle,
            unlabeled_ids,
            round_index,
            budget,
            acquisition_result=acquisition_result,
        )
        unlabeled_ids = selection["unlabeled_ids"]
        selection["stems"] = [manifest.pool_ids[i] for i in selection["ids"]]
        selection_seconds = time.perf_counter() - start
        round_dir = (
            run_dir if single else run_dir / f"round-{round_index:02d}-budget-{budget}"
        )
        run_config = _build_run_config(
            config, tracking, manifest, encoder, selection, device, smoke
        )
        run_config["manifest_sha256"] = manifest_hash
        run_config["experiment_name"] = run_dir.name
        selections.append(selection)
        _save_round_inputs(run_dir, round_dir, run_config, selection, selections)
        result = _run_round(
            run_config,
            tracking,
            pool,
            oracle,
            encoder,
            evaluation,
            round_dir,
            selection_seconds,
        )
        results.append(result)
        write_csv(run_dir / "results.csv", results)
        write_json(run_dir / "state.json", {**selection, "status": "completed"})
    if not single:
        log_experiment_history(tracking, run_dir, run_config, results)
    return results
