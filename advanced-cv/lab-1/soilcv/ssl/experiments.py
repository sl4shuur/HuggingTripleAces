"""Run the Part 3 matrix on the existing four physical-sample folds."""

import argparse
import copy
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from soilcv.artifacts import (
    ComputeMeter,
    file_hash,
    identity_hash,
    package_identity,
    save_environment,
    write_json,
)
from soilcv.bags import cache_grid, coverage_table, make_grid_manifest
from soilcv.camera import camera_metrics
from soilcv.cnn.predict import predict_bags
from soilcv.config import ExperimentConfig, seed_everything
from soilcv.curves import TARGET_COLUMNS, repair_curves
from soilcv.data import DEFAULT_FOLDS_PATH, load_folds, load_metadata
from soilcv.features import feature_identity, load_features, save_features
from soilcv.metrics import evaluate_samples
from soilcv.nn import EMDLoss, make_param_groups
from soilcv.tracking import log_epoch, log_evaluation, start_run
from soilcv.training import train_one_epoch

from .adaptation import ADAPTATION, adapt_encoder
from .data import (
    PREPROCESSING,
    FeatureBags,
    adaptation_loader,
    bag_loader,
    camera_features,
    pool_features,
    tile_loader,
)
from .evaluation import frozen_predict, representation_metrics, weighted_knn
from .model import ENCODERS, SSLEncoder, SSLRegressor, make_head

ROOT = Path(os.environ.get("SOILCV_SSL_OUTPUT_DIR", "/kaggle/working/part3_ssl_v2"))
CACHE = Path(os.environ.get("SOILCV_SSL_CACHE_DIR", "/tmp/soilcv-ssl-tiles"))
SETTINGS = {
    "preprocessing": PREPROCESSING,
    "encoders": ENCODERS,
    "strategies": ["knn", "ridge", "lp", "lpft", "full"],
    "lp_epochs": 100,
    "lp_patience": 15,
    "ft_epochs": 16,
    "ft_patience": 5,
    "head_lr": 0.001,
    "backbone_lr": 1e-5,
    "weight_decay": 0.05,
    "layer_decay": 0.75,
    "warmup": 2,
    "adaptation": ADAPTATION,
    "selection": "single best validation EMD; no camera gate",
    "seed": 42,
}


def tracked_run(config, tracking_root, run=None):
    """One recipe/fold per run, using the same grouping as CNN and ViT studies."""
    if tracking_root is None:
        return nullcontext(run)
    params = {key: value for key, value in config.items() if key != "identity"}
    if config.get("stage") != "ssl-adaptation" and config["strategy"] not in {
        "knn",
        "ridge",
    }:
        params.update(
            {
                key: SETTINGS[key]
                for key in ["head_lr", "backbone_lr", "weight_decay", "warmup"]
            }
        )
        prefix = "lp" if config["strategy"] == "lp" else "ft"
        params.update(
            epochs=SETTINGS[f"{prefix}_epochs"], patience=SETTINGS[f"{prefix}_patience"]
        )
    experiment = ExperimentConfig(
        config["experiment"],
        stage=config.get("stage", "ssl-transfer"),
        model=ENCODERS[config["encoder"]]["repo"],
        fold=config["fold"],
        seed=42 + config["fold"],
        hypothesis_id="H-SSL-01:06",
        output_root=str(tracking_root),
        versions={"soilcv": config["identity"]["soilcv"], "run_id": config["study_id"]},
        params=params,
    )
    return start_run(experiment, name=f"{config['experiment']}-fold-{config['fold']}")


def prepare(data=None, *, root=ROOT, cache=CACHE):
    root.mkdir(parents=True, exist_ok=True)
    data = (
        Path(data)
        if data
        else next(Path("/kaggle/input").rglob("Training_labels_updated.csv")).parent
    )
    labels, photos, template = load_metadata(data)
    folds = load_folds(DEFAULT_FOLDS_PATH, labels.sample_id)
    tiles = make_grid_manifest(photos, tile_mm=64, output_px=224, image_root=data)
    identity = {
        "soilcv": package_identity(),
        "settings": SETTINGS,
        "tiles": tiles.cache_id.iloc[0],
        "data": {
            name: file_hash(data / name)
            for name in [
                "Training_labels_updated.csv",
                "ppm_updated.csv",
                "sample_submission.csv",
            ]
        },
        "hypotheses_hash": file_hash(Path(__file__).parents[2] / "HYPOTHESES.md"),
    }
    if (root / "metadata.json").exists() and json.loads(
        (root / "metadata.json").read_text()
    )["identity"] != identity:
        raise ValueError("Code/data/settings changed: use a fresh output folder.")
    model_root = os.environ.get("SOILCV_SSL_MODELS")
    sources = {}
    for name, spec in ENCODERS.items():
        if model_root:
            source = Path(model_root) / name
        else:
            from huggingface_hub import snapshot_download

            source = Path(
                snapshot_download(
                    spec["repo"],
                    revision=spec["revision"],
                    allow_patterns=["config.json", "model.safetensors"],
                )
            )
        if not (source / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing official weights: {source}")
        sources[name] = {
            "path": str(source),
            "weights_sha256": file_hash(source / "model.safetensors"),
            **spec,
        }
    write_json(
        root / "metadata.json",
        {"identity": identity, "study_id": identity_hash(identity), "models": sources},
    )
    cache_grid(tiles, cache, image_root=data)
    tiles = tiles.merge(folds, on="sample_id", how="left", validate="many_to_one")
    coverage_table(tiles).to_csv(root / "coverage.csv", index=False)
    for name, frame in [
        ("tiles", tiles),
        ("labels", labels),
        ("photos", photos),
        ("template", template),
    ]:
        frame.to_json(root / f"{name}.json", orient="records", indent=2)
    folds.to_csv(root / "sample_folds.csv", index=False)
    write_json(root / "settings.json", SETTINGS)
    save_environment(root / "environment.json")
    print(
        "Prepared",
        len(tiles),
        "physical tiles;",
        len(labels),
        "labelled soils",
        flush=True,
    )


def extract(encoder, manifest, cache, device):
    ids, values = [], []
    encoder.eval()
    with torch.inference_mode():
        for batch in tile_loader(manifest, cache):
            with torch.autocast(
                device_type=torch.device(device).type,
                enabled=str(device).startswith("cuda"),
            ):
                value = encoder(batch["image"].to(device))
            values.append(value.float().cpu().numpy())
            ids.extend(
                {key: str(batch[key][i]) for key in ["sample_id", "photo", "tile_id"]}
                for i in range(len(value))
            )
    return pd.DataFrame(ids), np.concatenate(values)


def cached_features(encoder, manifest, folder, name, fold, train_ids, device):
    identity = feature_identity(
        encoder,
        name=name,
        weights_revision=ENCODERS[name]["revision"],
        preprocessing=PREPROCESSING,
        tile_cache_id=manifest.cache_id.iloc[0],
        pooling="cls_patch",
        fold=fold,
        train_ids=train_ids,
    )
    destination = folder / "features"
    if (destination / "metadata.json").exists():
        return load_features(destination, identity)
    with ComputeMeter(device, output_path=folder / "feature_compute.json"):
        ids, values = extract(encoder, manifest, CACHE, device)
        save_features(destination, ids, values, identity)
    return ids, values


def frame(ids, predictions):
    result = ids.copy()
    result[list(TARGET_COLUMNS)] = predictions
    return result


def fit_stage(
    model,
    train_loader,
    valid_loader,
    labels,
    folder,
    *,
    device,
    frozen,
    config,
    run,
    is_probe=False,
    probe_cameras=None,
):
    """Keep one inference state per stage; early stopping uses only sample EMD."""
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "config.json", config)
    epochs = SETTINGS["lp_epochs"] if is_probe else SETTINGS["ft_epochs"]
    patience = SETTINGS["lp_patience"] if is_probe else SETTINGS["ft_patience"]
    groups = make_param_groups(
        model if is_probe else model.head,
        lr=SETTINGS["head_lr"],
        weight_decay=SETTINGS["weight_decay"],
    )
    if not frozen:
        groups += make_param_groups(
            model.encoder,
            lr=SETTINGS["backbone_lr"],
            weight_decay=SETTINGS["weight_decay"],
            layer_ids=model.encoder.layer_ids(),
            layer_decay=config["layer_decay"],
            no_weight_decay=model.encoder.no_weight_decay(),
        )
    optimizer = torch.optim.AdamW(groups)

    def factor(epoch):
        if epoch < SETTINGS["warmup"]:
            return (epoch + 1) / SETTINGS["warmup"]
        progress = (epoch - SETTINGS["warmup"]) / max(
            1, epochs - SETTINGS["warmup"] - 1
        )
        return 0.05 + 0.95 * (1 + math.cos(math.pi * min(1, progress))) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    # Percentage/log10 EMD gradients need a smaller initial scale than CE losses.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=str(device).startswith("cuda"), init_scale=128
    )
    loss_fn = EMDLoss().to(device)
    best, bad, history, global_step = math.inf, 0, [], 0
    with ComputeMeter(device, output_path=folder / "compute.json"):
        for epoch in range(epochs):
            metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device=device,
                amp=True,
                accumulation_steps=1 if is_probe else 2,
                clip_grad=3,
                scaler=scaler,
                global_step=global_step,
            )
            global_step = metrics["global_step"]
            if is_probe:
                model.eval()
                predictions = []
                with torch.inference_mode():
                    for batch in valid_loader:
                        values = model(batch["image"].to(device)).cpu().numpy()
                        predictions.append(
                            frame(
                                pd.DataFrame({"sample_id": batch["sample_id"]}), values
                            )
                        )
                oof = pd.concat(predictions, ignore_index=True)
                cameras = None
                if probe_cameras is not None:
                    camera_ids, camera_features = probe_cameras
                    with torch.inference_mode():
                        values = (
                            model(torch.as_tensor(camera_features, device=device))
                            .cpu()
                            .numpy()
                        )
                    cameras = frame(camera_ids, values)
            else:
                oof, cameras = predict_bags(
                    model, valid_loader, device=device, amp=True, cameras=True
                )
            validation = evaluate_samples(labels, oof)
            if cameras is not None and not cameras.empty:
                validation.update(camera_metrics(labels, cameras))
            score = validation["emd"]
            history.append({"epoch": epoch, "train": metrics, "val": validation})
            if score < best and metrics["optimizer_steps"] > 0:
                best, bad = score, 0
                torch.save(model.state_dict(), folder / "weights.pt")
                oof.to_csv(folder / "oof.csv", index=False)
                write_json(folder / "selection.json", {"epoch": epoch, "emd": best})
                if run is not None:
                    run.summary.update(
                        {
                            "selected/epoch": epoch,
                            "selected/emd": best,
                            "selected/selection": "emd",
                        }
                    )
                    if "camera_disagreement" in validation:
                        run.summary["selected/camera_disagreement"] = validation[
                            "camera_disagreement"
                        ]
            elif math.isfinite(best):
                bad += 1
            if metrics["optimizer_steps"] > 0:
                scheduler.step()
            write_json(folder / "history.json", history)
            if run is not None:
                log_epoch(run, epoch, metrics, validation)
            print(
                config["experiment"],
                "fold",
                config["fold"],
                {
                    "epoch": epoch,
                    "train/loss": metrics["loss"],
                    "val/emd": score,
                    "optimizer_steps": metrics["optimizer_steps"],
                    "skipped_steps": metrics["skipped_steps"],
                },
                flush=True,
            )
            if bad >= patience:
                break
    if sum(row["train"]["optimizer_steps"] for row in history) == 0:
        raise FloatingPointError(
            "All AMP updates were skipped; no trained result can be reported."
        )
    model.load_state_dict(
        torch.load(folder / "weights.pt", weights_only=True, map_location=device)
    )
    return model


def finish(folder, config, oof, test, labels, *, diagnostics=None, run=None):
    oof = oof.copy()
    oof["fold"] = config["fold"]
    oof.to_csv(folder / "oof.csv", index=False)
    test.to_csv(folder / "test_predictions.csv", index=False)
    write_json(folder / "config.json", config)
    if diagnostics is not None:
        write_json(folder / "representation.json", diagnostics)
    validation = evaluate_samples(labels, oof)
    result = {
        "experiment": config["experiment"],
        "fold": config["fold"],
        **validation,
    }
    if run is not None:
        camera_path = folder / "camera_predictions.csv"
        if camera_path.exists():
            validation.update(camera_metrics(labels, pd.read_csv(camera_path)))
        log_evaluation(
            run,
            {**validation, **json.loads((folder / "compute.json").read_text())},
            prefix="best",
        )
        if config["strategy"] in {"knn", "ridge"}:
            log_epoch(run, 0, {}, validation)
        if diagnostics is not None:
            log_evaluation(run, diagnostics, prefix="representation")
        write_json(folder / "wandb_run.json", {"run_id": run.id, "run_url": run.url})
    write_json(folder / "result.json", result)
    write_json(
        folder / "hashes.json",
        {
            str(path.relative_to(folder)): file_hash(path)
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.name != "hashes.json"
        },
    )
    return result


def run_source(
    name,
    source_name,
    encoder,
    manifest,
    labels,
    folds,
    fold,
    folder,
    metadata,
    device,
    run,
    *,
    tracking_root=None,
):
    folder.mkdir(parents=True, exist_ok=True)
    train_ids = set(folds.loc[folds.fold.ne(fold), "sample_id"])
    valid_ids = set(folds.loc[folds.fold.eq(fold), "sample_id"])
    if train_ids & valid_ids:
        raise ValueError("Sample leakage.")
    source_config = {
        "encoder": name,
        "source": source_name,
        "fold": fold,
        "model_config": encoder.backbone.config.to_dict(),
        "pooling": encoder.pooling,
        "study_id": metadata["study_id"],
        "preprocessing": PREPROCESSING,
    }
    torch.save(encoder.state_dict(), folder / "encoder.pt")
    source_config["encoder_sha256"] = file_hash(folder / "encoder.pt")
    write_json(folder / "source.json", source_config)
    ids, tile_features = cached_features(
        encoder, manifest, folder, name, fold, train_ids, device
    )
    sample_ids, features = pool_features(ids, tile_features, manifest)
    train_mask = sample_ids.sample_id.isin(train_ids).to_numpy()
    valid_mask = sample_ids.sample_id.isin(valid_ids).to_numpy()
    test_mask = ~(train_mask | valid_mask)
    train_labels = (
        labels.set_index("sample_id")
        .loc[sample_ids.loc[train_mask, "sample_id"]]
        .reset_index()
    )
    valid_labels = labels[labels.sample_id.isin(valid_ids)]
    targets = train_labels[list(TARGET_COLUMNS)].to_numpy()
    prior = targets.mean(0)
    camera_ids, camera_values = camera_features(ids, tile_features, manifest, valid_ids)
    base_config = {
        **source_config,
        "identity": metadata["identity"],
        "train_ids": sorted(train_ids),
        "valid_ids": sorted(valid_ids),
        "layer_decay": SETTINGS["layer_decay"],
        "photometric": True,
        "train_mean_prior": prior.tolist(),
    }

    for strategy in ["knn", "ridge"]:
        config = {
            **base_config,
            "strategy": strategy,
            "experiment": f"{source_name}-{strategy}",
        }
        destination = folder / strategy
        destination.mkdir(exist_ok=True)
        with tracked_run(config, tracking_root, run) as stage_run:
            with ComputeMeter("cpu", output_path=destination / "compute.json"):
                query = features[~train_mask]
                predicted, fitted = frozen_predict(
                    features[train_mask], targets, query, strategy
                )
            # Store numerical parameters, not an executable sklearn pickle.
            if fitted is not None:
                scaler, ridge = fitted.steps[0][1], fitted.steps[1][1]
                np.savez(
                    destination / "regressor.npz",
                    mean=scaler.mean_,
                    scale=scaler.scale_,
                    coef=ridge.coef_,
                    intercept=ridge.intercept_,
                )
            else:
                np.savez(
                    destination / "regressor.npz",
                    train=features[train_mask],
                    targets=targets,
                )
            predictions = frame(
                sample_ids.loc[~train_mask].reset_index(drop=True), predicted
            )
            camera_predictions = (
                repair_curves(fitted.predict(camera_values))
                if fitted is not None
                else weighted_knn(features[train_mask], targets, camera_values)
            )
            frame(camera_ids, camera_predictions).to_csv(
                destination / "camera_predictions.csv", index=False
            )
            finish(
                destination,
                config,
                predictions[predictions.sample_id.isin(valid_ids)],
                predictions[~predictions.sample_id.isin(valid_ids)],
                valid_labels,
                run=stage_run,
            )

    # Matched deterministic held-out tiles, capped evenly per photo before CKA.
    valid_tiles = manifest[manifest.sample_id.isin(valid_ids)]
    diagnostic_tiles = (
        valid_tiles.groupby(["sample_id", "photo"], sort=True).head(4).head(256)
    )
    diagnostic_ids, before = extract(encoder, diagnostic_tiles, CACHE, device)
    diagnostic_ids.to_csv(folder / "diagnostic_tiles.csv", index=False)
    seed_everything(42 + fold)
    head = make_head(encoder.output_dim, prior).to(device)
    config = {**base_config, "strategy": "lp", "experiment": f"{source_name}-lp"}
    destination = folder / "lp"
    with tracked_run(config, tracking_root, run) as stage_run:
        train_data = FeatureBags(
            sample_ids.loc[train_mask].reset_index(drop=True),
            features[train_mask],
            folds,
            labels,
        )
        valid_data = FeatureBags(
            sample_ids.loc[valid_mask].reset_index(drop=True),
            features[valid_mask],
            folds,
            labels,
        )
        head = fit_stage(
            head,
            DataLoader(
                train_data,
                batch_size=6,
                shuffle=True,
                generator=torch.Generator().manual_seed(42 + fold),
            ),
            DataLoader(valid_data, batch_size=6),
            valid_labels,
            destination,
            device=device,
            frozen=True,
            config=config,
            run=stage_run,
            is_probe=True,
            probe_cameras=(camera_ids, camera_values),
        )
        lp_head = copy.deepcopy(head.state_dict())
        with torch.inference_mode():
            prediction = (
                head(torch.as_tensor(features[valid_mask | test_mask], device=device))
                .cpu()
                .numpy()
            )
        curves = frame(
            sample_ids.loc[valid_mask | test_mask].reset_index(drop=True), prediction
        )
        with torch.inference_mode():
            camera_predictions = (
                head(torch.as_tensor(camera_values, device=device)).cpu().numpy()
            )
        frame(camera_ids, camera_predictions).to_csv(
            destination / "camera_predictions.csv", index=False
        )
        finish(
            destination,
            config,
            curves[curves.sample_id.isin(valid_ids)],
            curves[~curves.sample_id.isin(valid_ids)],
            valid_labels,
            diagnostics=representation_metrics(before, before),
            run=stage_run,
        )

    variants = [("lpft", 0.75, True), ("full", 0.75, True)]
    if source_name == "dinov3":
        variants += [("lpft-uniform-lr", 1.0, True), ("lpft-no-color", 0.75, False)]
        cls_size = encoder.backbone.config.hidden_size
        destination = folder / "ridge-cls"
        destination.mkdir(exist_ok=True)
        config = {
            **base_config,
            "strategy": "ridge",
            "pooling": "cls",
            "experiment": "dinov3-ridge-cls",
        }
        with tracked_run(config, tracking_root, run) as stage_run:
            with ComputeMeter("cpu", output_path=destination / "compute.json"):
                predicted, fitted = frozen_predict(
                    features[train_mask, :cls_size],
                    targets,
                    features[~train_mask, :cls_size],
                    "ridge",
                )
            scaler, ridge = fitted.steps[0][1], fitted.steps[1][1]
            np.savez(
                destination / "regressor.npz",
                mean=scaler.mean_,
                scale=scaler.scale_,
                coef=ridge.coef_,
                intercept=ridge.intercept_,
            )
            curves = frame(
                sample_ids.loc[~train_mask].reset_index(drop=True), predicted
            )
            frame(
                camera_ids, repair_curves(fitted.predict(camera_values[:, :cls_size]))
            ).to_csv(destination / "camera_predictions.csv", index=False)
            finish(
                destination,
                config,
                curves[curves.sample_id.isin(valid_ids)],
                curves[~curves.sample_id.isin(valid_ids)],
                valid_labels,
                run=stage_run,
            )

    for strategy, decay, photometric in variants:
        seed_everything(42 + fold)
        encoder.load_state_dict(
            torch.load(folder / "encoder.pt", weights_only=True, map_location=device)
        )
        head = make_head(encoder.output_dim, prior).to(device)
        if strategy.startswith("lpft"):
            head.load_state_dict(lp_head)
        model = SSLRegressor(encoder, head).to(device)
        config = {
            **base_config,
            "strategy": strategy,
            "layer_decay": decay,
            "photometric": photometric,
            "experiment": f"{source_name}-{strategy}",
            "train_mean_prior": prior.tolist(),
        }
        destination = folder / strategy
        with tracked_run(config, tracking_root, run) as stage_run:
            train_loader = bag_loader(
                manifest[manifest.sample_id.isin(train_ids)],
                CACHE,
                labels,
                train=True,
                photometric=photometric,
                seed=42 + fold,
            )
            valid_loader = bag_loader(valid_tiles, CACHE, labels)
            fit_stage(
                model,
                train_loader,
                valid_loader,
                valid_labels,
                destination,
                device=device,
                frozen=False,
                config=config,
                run=stage_run,
            )
            oof, cameras = predict_bags(
                model, valid_loader, device=device, amp=True, cameras=True
            )
            cameras.to_csv(destination / "camera_predictions.csv", index=False)
            test, _ = predict_bags(
                model,
                bag_loader(
                    manifest[
                        manifest.sample_id.isin(sample_ids.loc[test_mask, "sample_id"])
                    ],
                    CACHE,
                ),
                device=device,
                amp=True,
            )
            _, after = extract(encoder, diagnostic_tiles, CACHE, device)
            finish(
                destination,
                config,
                oof,
                test,
                valid_labels,
                diagnostics=representation_metrics(before, after),
                run=stage_run,
            )
            if (
                tracking_root is not None
                and source_name in {"mae", "dinov3-adapted"}
                and strategy == "full"
            ):
                publish_fold(stage_run, folder.parent, name, fold, metadata)
    encoder.load_state_dict(
        torch.load(folder / "encoder.pt", weights_only=True, map_location=device)
    )
    return before, diagnostic_tiles


def publish_fold(run, folder, name, fold, metadata):
    import wandb

    artifact = wandb.Artifact(
        f"ssl-{name}-fold-{fold}",
        type="model",
        metadata={"study_id": metadata["study_id"], "fold": fold},
    )
    for path in sorted(folder.rglob("*")):
        if path.is_file() and "features" not in path.parts:
            artifact.add_file(str(path), name=str(path.relative_to(folder)))
    logged = run.log_artifact(artifact)
    logged.wait()
    write_json(
        folder / "complete.json",
        {
            "run_url": run.url,
            "artifact": logged.qualified_name,
            "study_id": metadata["study_id"],
        },
    )


def run_fold(name, fold, *, root=ROOT, device="cuda:0"):
    torch.set_num_threads(2)
    seed_everything(42 + fold)
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata["identity"]["soilcv"] != package_identity():
        raise ValueError("Source changed after preparation.")
    labels, manifest = (
        pd.read_json(root / "labels.json"),
        pd.read_json(root / "tiles.json"),
    )
    folds = pd.read_csv(root / "sample_folds.csv", dtype={"sample_id": str})
    encoder = SSLEncoder(name, source=metadata["models"][name]["path"]).to(device)
    folder = root / "folds" / name / f"fold_{fold}"
    before, diagnostic_tiles = run_source(
        name,
        name,
        encoder,
        manifest,
        labels,
        folds,
        fold,
        folder / name,
        metadata,
        device,
        None,
        tracking_root=root / "tracking",
    )
    if name == "dinov3":
        train = manifest[manifest.fold.notna() & manifest.fold.ne(fold)]
        adaptation_config = {
            "experiment": "dinov3-adaptation",
            "stage": "ssl-adaptation",
            "encoder": name,
            "fold": fold,
            "identity": metadata["identity"],
            "study_id": metadata["study_id"],
            **ADAPTATION,
        }
        with tracked_run(adaptation_config, root / "tracking") as run:
            adapt_encoder(
                encoder,
                adaptation_loader(train, CACHE, fold=fold, seed=142 + fold),
                folder / "adaptation",
                fold=fold,
                device=device,
                run=run,
            )
            _, adapted = extract(encoder, diagnostic_tiles, CACHE, device)
            diagnostics = representation_metrics(before, adapted)
            write_json(folder / "adaptation" / "representation.json", diagnostics)
            log_evaluation(run, diagnostics, prefix="representation")
            log_evaluation(
                run,
                json.loads((folder / "adaptation" / "compute.json").read_text()),
                prefix="adaptation",
            )
        run_source(
            name,
            "dinov3-adapted",
            encoder,
            manifest,
            labels,
            folds,
            fold,
            folder / "dinov3-adapted",
            metadata,
            device,
            None,
            tracking_root=root / "tracking",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "launch"])
    parser.add_argument("--experiment", choices=list(ENCODERS))
    parser.add_argument("--fold", type=int, choices=range(4))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "train":
        run_fold(args.experiment, args.fold)
    else:
        from soilcv.jobs import install_signal_handler, run_jobs

        install_signal_handler()
        run_jobs(
            [
                {"experiment": name, "fold": fold}
                for fold in range(4)
                for name in ENCODERS
            ],
            ROOT,
            "soilcv.ssl.experiments",
            gpu_count=min(2, torch.cuda.device_count()),
        )
