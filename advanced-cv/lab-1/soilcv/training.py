"""A small supervised loop. SSL pretraining can reuse the other package modules."""

from pathlib import Path
import shutil
import math

import numpy as np
import pandas as pd
import torch

from .curves import TARGET_COLUMNS, aggregate_predictions, repair_curves
from .metrics import evaluate_samples
from .nn import evaluation_mode, freeze_bn_stats
from .artifacts import ComputeMeter, identity_hash, save_checkpoint, load_checkpoint, write_json, save_predictions, save_environment, package_identity
from .config import save_config


def batch_ids(batch):
    required = ["sample_id", "photo", "tile_id"]
    if any(key not in batch for key in required):
        raise ValueError("Every batch needs sample_id, photo, and tile_id.")
    rows = []
    for i in range(len(batch["image"])):
        row = {key: str(batch[key][i].item() if isinstance(batch[key][i], torch.Tensor) else batch[key][i]) for key in required}
        if "fold" in batch:
            row["fold"] = int(batch["fold"][i])
        rows.append(row)
    return rows


def train_one_epoch(model, loader, optimizer, loss_fn, *, device="cpu", amp=False, accumulation_steps=1,
                    clip_grad=None, scaler=None, scheduler=None, freeze_batchnorm=False, frozen_modules=(), global_step=0):
    """Mean loss required. Normalize accumulated gradients by actual window size."""
    if not isinstance(accumulation_steps, int) or accumulation_steps < 1 or len(loader) == 0:
        raise ValueError("Use a non-empty loader and positive integer accumulation_steps.")
    if clip_grad is not None and clip_grad <= 0:
        raise ValueError("clip_grad must be positive.")
    if getattr(loss_fn, "reduction", "mean") != "mean":
        raise ValueError("Use a mean-reduced loss with this training loop.")
    device = torch.device(device)
    use_amp = bool(amp and device.type == "cuda")
    scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=use_amp)
    if hasattr(loss_fn, "to"):
        loss_fn.to(device)
    model.train()
    if freeze_batchnorm:
        freeze_bn_stats(model)
    for module in frozen_modules:
        if any(p.requires_grad for p in module.parameters()):
            raise ValueError("Frozen modules must have requires_grad=False.")
        module.eval()
    optimizer.zero_grad(set_to_none=True)
    total_loss, seen, window_size, updates, skipped = 0.0, 0, 0, 0, 0
    for batch_index, batch in enumerate(loader):
        image, target = batch["image"].to(device), batch["target"].to(device)
        size = len(target)
        if size == 0:
            raise ValueError("Empty training batch.")
        with torch.autocast(device_type=device.type, enabled=use_amp):
            prediction = model(image, batch['bag_weights'].to(device)) if 'bag_weights' in batch else model(image)
            loss = loss_fn(prediction, target)
        if loss.numel() != 1 or not torch.isfinite(loss):
            raise FloatingPointError("Expected one finite mean loss.")
        scaler.scale(loss * size).backward()
        total_loss += float(loss.detach()) * size
        seen += size
        window_size += size
        if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window_size)
            if not scaler.is_enabled() and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite gradients.")
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, error_if_nonfinite=not scaler.is_enabled())
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                updates += 1
                global_step += 1
                if scheduler is not None:
                    scheduler.step()
            else:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
            window_size = 0
    metrics = {"loss": total_loss / seen, "optimizer_steps": updates, "skipped_steps": skipped, "global_step": global_step}
    metrics.update({f"lr_{i}": group["lr"] for i, group in enumerate(optimizer.param_groups)})
    return metrics


def predict_tiles(model, loader, *, device="cpu", amp=False):
    device = torch.device(device)
    frames = []
    with evaluation_mode(model), torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
                values = model(batch["image"].to(device))
            if not isinstance(values, torch.Tensor) or values.shape != (len(batch["image"]), 11):
                raise ValueError("Model must return [batch, 11] curves.")
            values = values.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("Non-finite predictions.")
            frame = pd.DataFrame(batch_ids(batch))
            frame[list(TARGET_COLUMNS)] = values
            frames.append(frame)
    if not frames:
        raise ValueError("Prediction loader is empty.")
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["sample_id", "photo", "tile_id"]).any():
        raise ValueError("Prediction loaders must visit each tile once. Do not use a repeated training sampler.")
    return result


def validate_epoch(model, loader, labels, *, device="cpu", amp=False, repair=False):
    tiles = predict_tiles(model, loader, device=device, amp=amp)
    predictions = aggregate_predictions(tiles, check=False)
    if repair:
        predictions[list(TARGET_COLUMNS)] = repair_curves(predictions[list(TARGET_COLUMNS)])
    return evaluate_samples(labels, predictions), predictions


def _loader_state(loader):
    return {name: obj.generator.get_state() for name, obj in [("loader", loader), ("sampler", loader.sampler)]
            if getattr(obj, "generator", None) is not None}


def _restore_loader_state(loader, states):
    for name, obj in [("loader", loader), ("sampler", loader.sampler)]:
        if name in states:
            if getattr(obj, "generator", None) is None:
                raise ValueError("Resume needs the same loader/sampler generator setup.")
            obj.generator.set_state(states[name])


def fit_supervised_fold(model, train_loader, valid_loader, valid_labels, optimizer, loss_fn, config, *, epochs=10,
                        device="cpu", amp=False, scheduler=None, scheduler_step="epoch", patience=None,
                        accumulation_steps=1, clip_grad=None, freeze_batchnorm=False, frozen_modules=(),
                        repair=False, run=None, resume_from=None, validation_fn=None):
    """Fit one saved fold. Return best sample predictions and history; load best model weights."""
    if config.fold is None or epochs < 1 or scheduler_step not in {"epoch", "batch", "metric"}:
        raise ValueError("Set a fold, positive epochs, and scheduler_step=epoch/batch/metric.")
    if patience is not None and patience < 1:
        raise ValueError("patience must be positive or None.")
    train_manifest, valid_manifest = train_loader.dataset.manifest, valid_loader.dataset.manifest
    train_ids, valid_ids = set(train_manifest.sample_id), set(valid_manifest.sample_id)
    if not train_ids or not valid_ids or train_ids & valid_ids:
        raise ValueError("Train and validation must contain different, non-empty sample sets.")
    if set(valid_labels.sample_id) != valid_ids:
        raise ValueError("Validation labels must match the validation sample IDs.")
    if "fold" not in train_manifest or "fold" not in valid_manifest:
        raise ValueError("Join the saved folds into both manifests before fitting.")
    if train_manifest.fold.eq(config.fold).any() or not valid_manifest.fold.eq(config.fold).all():
        raise ValueError("Loader samples do not match the requested held-out fold.")
    if resume_from and (train_loader.persistent_workers or valid_loader.persistent_workers):
        raise ValueError("Use persistent_workers=False for epoch-boundary resume.")
    model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp and torch.device(device).type == "cuda"))
    settings = {"epochs": epochs, "amp": bool(amp), "scheduler_step": scheduler_step, "patience": patience,
                "accumulation_steps": accumulation_steps, "clip_grad": clip_grad, "freeze_batchnorm": freeze_batchnorm,
                "repair_curves": repair, "loss": type(loss_fn).__name__, "loss_scale": getattr(loss_fn, "scale", None),
                "loss_reduction": getattr(loss_fn, "reduction", None)}
    identity_settings = {key: value for key, value in settings.items() if key != "epochs"}
    identity = identity_hash({"code": package_identity(), "config": {k: v for k, v in config.to_dict().items() if k != "output_root"},
                              "train_ids": sorted(train_ids), "valid_ids": sorted(valid_ids), "training": identity_settings})
    root = config.output_dir
    start_epoch, global_step, best_emd, bad_epochs, history, best_records = 0, 0, math.inf, 0, [], None
    if resume_from:
        state = load_checkpoint(resume_from, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                                expected_metadata={"identity": identity})
        metadata = state["metadata"]
        start_epoch, global_step = state["epoch"] + 1, state["global_step"]
        best_emd, bad_epochs = metadata["best_emd"], metadata["bad_epochs"]
        history, best_records = metadata["history"], metadata["best_predictions"]
        if run is not None and state["run_id"] != run.id:
            raise ValueError("Resume the checkpoint's W&B run ID, not a different run.")
        _restore_loader_state(train_loader, metadata["train_loader_rng"])
        _restore_loader_state(valid_loader, metadata["valid_loader_rng"])
        source_best = Path(resume_from).parent / "best.pt"
        if not source_best.exists():
            raise ValueError("Resume needs both last.pt and its best.pt sidecar.")
        if source_best.resolve() != (root / "best.pt").resolve():
            shutil.copyfile(source_best, root / "best.pt")
        if start_epoch >= epochs:
            raise ValueError("epochs must be larger than the saved epoch count.")
    root.mkdir(parents=True, exist_ok=True)
    save_config(config, root / "config.json")
    save_environment(root / "environment.json")
    write_json(root / "training_settings.json", settings)
    if run is not None:
        run.config.update({"training": settings}, allow_val_change=True)
    with ComputeMeter(device, output_path=root / "compute.json") as meter:
        for epoch in range(start_epoch, epochs):
            if patience is not None and bad_epochs >= patience:
                break
            train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device=device, amp=amp,
                accumulation_steps=accumulation_steps, clip_grad=clip_grad, scaler=scaler,
                scheduler=scheduler if scheduler_step == "batch" else None, freeze_batchnorm=freeze_batchnorm,
                frozen_modules=frozen_modules, global_step=global_step)
            global_step = train_metrics["global_step"]
            validation, predictions = (validation_fn or validate_epoch)(model, valid_loader, valid_labels, device=device, amp=amp, repair=repair)
            if scheduler is not None:
                if scheduler_step == "epoch":
                    scheduler.step()
                elif scheduler_step == "metric":
                    scheduler.step(validation["emd"])
            # TODO: compare EMA/fixed epochs and repeat finalists on more splits when time allows.
            improved = validation["emd"] < best_emd
            if improved:
                best_emd, bad_epochs = validation["emd"], 0
                best_records = predictions.to_dict("records")
            else:
                bad_epochs += 1
            history.append({"epoch": epoch, "train": train_metrics, "val": validation})
            metadata = {"identity": identity, "best_emd": best_emd, "bad_epochs": bad_epochs, "history": history,
                        "best_predictions": best_records, "train_loader_rng": _loader_state(train_loader),
                        "valid_loader_rng": _loader_state(valid_loader)}
            checkpoint_args = dict(optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, global_step=global_step,
                                   config=config, run_id=run.id if run is not None else None, metadata=metadata)
            if improved:
                save_checkpoint(root / "best.pt", model, **checkpoint_args)
            save_checkpoint(root / "last.pt", model, **checkpoint_args)
            write_json(root / "history.json", history)
            if run is not None:
                from .tracking import log_epoch
                log_epoch(run, epoch, train_metrics, validation)
    load_checkpoint(root / "best.pt", model, expected_metadata={"identity": identity}, restore_rng=False)
    best_predictions = pd.DataFrame(best_records)
    best_predictions["fold"] = config.fold
    save_predictions(best_predictions, root / "oof.csv")
    if run is not None:
        from .tracking import log_evaluation
        log_evaluation(run, {**evaluate_samples(valid_labels, best_predictions), **meter.result}, prefix="best")
    return best_predictions, history
