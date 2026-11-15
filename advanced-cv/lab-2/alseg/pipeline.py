"""Compose one Random-budget run from data, frozen features, training and reporting."""

import importlib.metadata
import json
import os
import time
from pathlib import Path

import torch

from alseg.adapters.dinov2 import DinoV2Encoder
from alseg.adapters.voc import VocEvaluation, VocImages, VocOracle
from alseg.config import PROJECT_ROOT, TrackingConfig, load_config
from alseg.core.pool import PoolManifest
from alseg.core.predictions import IGNORE_INDEX, NUM_CLASSES
from alseg.core.selection import AcquisitionMethod
from alseg.data.preprocessing import resize_mask
from alseg.data.voc import find_voc_root, load_manifest
from alseg.helpers.io import write_csv
from alseg.interfaces.data import ImageSource, Oracle
from alseg.interfaces.encoder import Encoder
from alseg.nn.evaluation import evaluate_head
from alseg.nn.extraction import iter_features
from alseg.nn.training import fit_head
from alseg.runtime.tracking import configure_tracking, log_run_artifacts, wandb_run


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


def _load_encoder(config: dict, device: str) -> DinoV2Encoder:
    encoder_config = config["encoder"]
    return DinoV2Encoder.from_pretrained(
        encoder_config["model_id"],
        encoder_config["revision"],
        encoder_config["image_size"],
        device=device,
        cache_dir=Path(config["paths"]["cache_dir"]) / "huggingface",
    )


def _prepare_labeled_features(
    config: dict, pool: ImageSource, oracle: Oracle, encoder: Encoder, ids: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
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
    encoder: Encoder,
    ids: list[int],
    device: str,
    smoke: bool,
) -> dict:
    return {
        **config,
        "tracking": tracking.model_dump(mode="json"),
        "encoder_identity": encoder.describe(),
        "budget": len(ids),
        "n_fit": len(ids),
        "n_calibration": 0,
        "selected_ids": ids,
        "pool_ids": manifest.pool_ids,
        "val_ids": manifest.val_ids,
        "selection_seed": config["seed"] + config["protocol"]["selection_seed_offset"],
        "training_seed": config["seed"] + config["protocol"]["training_seed_offset"],
        "val_images": min(8, len(manifest.val_ids)) if smoke else len(manifest.val_ids),
        "smoke": smoke,
        "project_version": load_config(PROJECT_ROOT / "pyproject.toml")["project"][
            "version"
        ],
        "runtime": {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch": torch.__version__,
            **{
                name: importlib.metadata.version(name)
                for name in ("lightning", "wandb", "transformers")
            },
            "precision": "32-true",
        },
    }


def _save_results(
    run_dir: Path, run_config: dict, metrics: dict, timings: dict
) -> dict:
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    result = {
        "method": AcquisitionMethod.RANDOM.value,
        **{
            name: run_config[name]
            for name in (
                "budget",
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
    """Fit all queried L. Temperature calibration and multi-round AL are later steps."""
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
    if type(budget) is not int or not 0 < budget <= len(pool):
        raise ValueError("budget must be between one and the train pool size")
    seed = config["seed"] + config["protocol"]["selection_seed_offset"]
    ids = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed))[
        :budget
    ].tolist()
    run_dir.mkdir(parents=True, exist_ok=False)
    oracle = VocOracle(root, manifest, run_dir / "oracle_queries.jsonl")
    oracle.query(ids)
    encoder = _load_encoder(config, device)
    run_config = _build_run_config(
        config, tracking, manifest, encoder, ids, device, smoke
    )
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    (run_dir / "selection.json").write_text(
        json.dumps(
            {"ids": ids, "stems": [manifest.pool_ids[image_id] for image_id in ids]},
            indent=2,
        )
        + "\n"
    )

    start = time.perf_counter()
    features, targets = _prepare_labeled_features(config, pool, oracle, encoder, ids)
    timings = {"feature_seconds": time.perf_counter() - start}
    with wandb_run(tracking, run_dir, run_dir.name, run_config) as logger:
        start = time.perf_counter()
        head = _fit_and_save_head(features, targets, run_config, run_dir, logger)
        timings["train_seconds"] = time.perf_counter() - start

        start = time.perf_counter()
        metrics, previews = evaluate_head(
            head,
            encoder,
            VocEvaluation(root, manifest),
            list(range(run_config["val_images"])),
            config["encoder"]["batch_size"],
            device,
        )
        timings["eval_seconds"] = time.perf_counter() - start
        result = _save_results(run_dir, run_config, metrics, timings)
        if logger:
            logger.log_metrics(
                {f"perf/{name}": value for name, value in timings.items()}
            )
        log_run_artifacts(logger, run_dir, metrics, previews)
    return result
