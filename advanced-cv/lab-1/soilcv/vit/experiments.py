"""Local-detail plus full-photo Transformer studies on the saved soil folds."""

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import models

from soilcv.artifacts import file_hash, identity_hash, package_identity, write_json
from soilcv.bags import cache_grid, coverage_table, make_grid_manifest, one_bag
from soilcv.camera import camera_family, camera_metrics, domain_metrics
from soilcv.checkpoints import candidate_paths, candidate_predictions, load_candidates
from soilcv.cnn.predict import predict_bags, validate_bags_camera
from soilcv.config import ExperimentConfig, seed_everything
from soilcv.curves import TARGET_COLUMNS
from soilcv.data import DEFAULT_FOLDS_PATH, load_folds, load_metadata
from soilcv.metrics import evaluate_samples
from soilcv.nn import EMDLoss, make_param_groups
from soilcv.predictions import mean_curves
from soilcv.tracking import log_artifact, start_run
from soilcv.training import fit_supervised_fold

from .images import (
    NORMALIZE,
    FullPhotoBags,
    cache_photos,
    make_photo_manifest,
    photo_transform,
)
from .model import FullFrameModel, encoder_layer_ids

FAMILY = os.environ.get("SOILCV_VIT_FAMILY", "scratch")
if FAMILY not in {"scratch", "pretrained"}:
    raise ValueError("SOILCV_VIT_FAMILY must be scratch or pretrained.")
ROOT = Path(
    os.environ.get(
        "SOILCV_VIT_OUTPUT_DIR", f"/kaggle/working/part2_{FAMILY}_multiscale"
    )
)
CACHE = Path(os.environ.get("SOILCV_VIT_CACHE_DIR", "/tmp/soilcv-multiscale"))
HYPOTHESES_COMMIT = "a39b8a5"
BASE = {
    "target_ppm": 2.0,
    "augmentation": True,
    "weight_decay": 0.05,
    "schedule": "warmup_cosine",
    "warmup": 3,
    "accumulation_steps": 2,
    "deterministic": True,
    "drop_path": 0.1,
    "image_schema": "full_photo_ppm_v2",
    "view_mode": "multiscale",
    "tile_mm": 64,
    "pixels": 288,
    "chunk_size": 8,
    "pooling": "local_global_equal_photo_features",
    "selection": "emd",
    "keep_top_k": 3,
}
SCRATCH_BASE = dict(
    BASE,
    arch="scratch_vit",
    patch_size=16,
    dim=192,
    depth=4,
    heads=3,
    dropout=0.1,
    max_tokens=2048,
    backbone_lr=3e-4,
    head_lr=3e-4,
    layer_decay=1.0,
    epochs=32,
    patience=7,
)
PRETRAINED_BASE = dict(
    BASE,
    arch="swin_t",
    backbone_lr=1e-4,
    head_lr=0.001,
    layer_decay=0.75,
    epochs=24,
    patience=6,
)
SCRATCH_RECIPES = [
    dict(SCRATCH_BASE, id="vit-scratch-p16", hypothesis="H-VIT2-01"),
    dict(SCRATCH_BASE, id="vit-scratch-p32", hypothesis="H-VIT2-02", patch_size=32),
    dict(SCRATCH_BASE, id="vit-scratch-no-warmup", hypothesis="H-VIT2-03", warmup=0),
]
PRETRAINED_RECIPES = [
    dict(
        PRETRAINED_BASE,
        id="swin-frozen",
        hypothesis="H-VIT2-04",
        tuning="frozen",
        head_lr=0.003,
    ),
    dict(
        PRETRAINED_BASE,
        id="swin-last-stage",
        hypothesis="H-VIT2-04",
        tuning="last_stage",
    ),
    dict(
        PRETRAINED_BASE,
        id="swin-full",
        hypothesis="H-VIT2-04",
        tuning="full",
        backbone_lr=3e-5,
    ),
]
TILE_BASE = dict(
    PRETRAINED_BASE,
    view_mode="tiles",
    pooling="overlap_area_then_equal_photos",
    grid="full_grid_v1",
    tuning="full",
    backbone_lr=1e-4,
    head_lr=0.001,
    layer_decay=1.0,
    weight_decay=0.01,
    epochs=16,
    patience=5,
    schedule="constant",
    warmup=0,
)
TILE_RECIPES = [
    dict(
        TILE_BASE,
        id="swin-tiles-frozen",
        hypothesis="H-VIT2-05",
        tuning="frozen",
        head_lr=0.003,
    ),
    dict(TILE_BASE, id="swin-tiles-constant", hypothesis="H-VIT2-05"),
    dict(
        TILE_BASE,
        id="swin-tiles-cosine",
        hypothesis="H-VIT2-05",
        schedule="warmup_cosine",
        warmup=3,
    ),
]
PRETRAINED_RECIPES += TILE_RECIPES
RECIPES = SCRATCH_RECIPES if FAMILY == "scratch" else PRETRAINED_RECIPES
DOMAIN_BASELINE = "vit-scratch-p16" if FAMILY == "scratch" else "swin-last-stage"


def jobs():
    standard = [(recipe["id"], fold, None) for recipe in RECIPES for fold in range(4)]
    domain = [
        (DOMAIN_BASELINE, fold, camera)
        for camera in ("Motorola", "Samsung")
        for fold in range(4)
    ]
    return standard + domain


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    data = next(Path("/kaggle/input").rglob("Training_labels_updated.csv")).parent
    labels, photos, template = load_metadata(data)
    folds = load_folds(DEFAULT_FOLDS_PATH, labels.sample_id)
    manifest = make_photo_manifest(
        photos, target_ppm=BASE["target_ppm"], image_root=data
    )
    tiles = make_grid_manifest(
        photos, tile_mm=BASE["tile_mm"], output_px=BASE["pixels"], image_root=data
    )
    # The 2 PPM view is context only. Every native 64 mm region also has a 288 px local view.
    # Fixed PPM is never changed per photo to meet a memory limit.
    tokens = np.ceil(manifest.output_height / 16) * np.ceil(manifest.output_width / 16)
    if FAMILY == "scratch" and tokens.max() > SCRATCH_BASE["max_tokens"]:
        raise ValueError(
            "The shared PPM produces too many tokens. Lower it for all photos."
        )
    identity = {
        "soilcv": package_identity(),
        "data": {
            name: file_hash(data / name)
            for name in (
                "Training_labels_updated.csv",
                "ppm_updated.csv",
                "sample_submission.csv",
            )
        },
        "frames": manifest.cache_id.iloc[0],
        "tiles": tiles.cache_id.iloc[0],
        "folds_hash": file_hash(DEFAULT_FOLDS_PATH),
        "recipes": identity_hash(RECIPES),
        "hypotheses_commit": HYPOTHESES_COMMIT,
    }
    path = ROOT / "metadata.json"
    if path.exists():
        metadata = json.loads(path.read_text())
        if metadata["identity"] != identity:
            raise ValueError("Code or data changed. Use a fresh output folder.")
    else:
        pretrained = None
        if FAMILY == "pretrained":
            weights = models.Swin_T_Weights.IMAGENET1K_V1
            weights.get_state_dict(progress=False, check_hash=True)
            weight_path = (
                Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
            )
            pretrained = {
                "enum": str(weights),
                "url": weights.url,
                "sha256": file_hash(weight_path),
            }
        metadata = {
            "identity": identity,
            "run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            + "-"
            + identity_hash(identity)[:10],
            "data_root": str(data),
            "family": FAMILY,
            "pretrained": pretrained,
        }
        write_json(path, metadata)
    cache_photos(manifest, CACHE, image_root=data)
    cache_grid(tiles, CACHE, image_root=data)
    manifest = manifest.merge(folds, on="sample_id", how="left", validate="many_to_one")
    manifest.to_json(ROOT / "frames.json", orient="records", indent=2)
    spatial = (
        coverage_table(tiles)
        .rename(columns={"coverage": "spatial_coverage"})
        .drop(columns="missing_native_pixels")
    )
    local_geometry = tiles.drop_duplicates(["sample_id", "photo"])[
        ["sample_id", "photo", "crop_px", "tile_mm", "output_px"]
    ].copy()
    local_geometry["local_ppm"] = local_geometry.output_px / local_geometry.tile_mm
    local_geometry["linear_scale_ratio"] = (
        local_geometry.output_px / local_geometry.crop_px
    )
    local_geometry["pixel_count_ratio"] = local_geometry.linear_scale_ratio**2
    spatial.merge(
        local_geometry, on=["sample_id", "photo"], validate="one_to_one"
    ).to_csv(ROOT / "tile_spatial_coverage.csv", index=False)
    tiles = tiles.merge(folds, on="sample_id", how="left", validate="many_to_one")
    tiles.to_json(ROOT / "tiles.json", orient="records", indent=2)
    coverage_columns = [
        "sample_id",
        "photo",
        "device",
        "width",
        "height",
        "output_width",
        "output_height",
        "native_ppm",
        "target_ppm",
        "actual_ppm_x",
        "actual_ppm_y",
        "spatial_coverage",
        "cropped_source_area",
        "linear_scale_ratio",
        "pixel_count_ratio",
    ]
    manifest[coverage_columns].to_csv(ROOT / "image_geometry.csv", index=False)
    for name, frame in (("labels", labels), ("photos", photos), ("template", template)):
        frame.to_json(ROOT / f"{name}.json", orient="records", indent=2)
    folds.to_csv(ROOT / "sample_folds.csv", index=False)
    write_json(ROOT / "recipes.json", RECIPES)
    print(
        "Study:",
        FAMILY,
        "| photos:",
        len(manifest),
        "| max patch-16 tokens:",
        int(tokens.max()),
        flush=True,
    )
    return metadata


def make_loader(manifest, recipe, labels=None, *, tiles=None, train=False, seed=242):
    if tiles is None:
        tiles = pd.read_json(ROOT / "tiles.json")
    keys = manifest[["sample_id", "photo"]].drop_duplicates()
    tiles = tiles.merge(
        keys, on=["sample_id", "photo"], how="inner", validate="many_to_one"
    )
    dataset = FullPhotoBags(
        manifest,
        CACHE,
        photo_transform(recipe, train=train),
        labels,
        tiles=tiles,
        view_mode=recipe["view_mode"],
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=train,
        num_workers=0,
        collate_fn=one_bag,
        generator=torch.Generator().manual_seed(seed),
    )


def verify_completed(folder, recipe_hash=None):
    folder = Path(folder)
    result = json.loads((folder / "result.json").read_text())
    if recipe_hash is not None and result["recipe_hash"] != recipe_hash:
        raise ValueError("Existing run has different settings.")
    hashes = json.loads((folder / "artifact_hashes.json").read_text())
    required = {
        "oof.csv",
        "test_predictions.csv",
        "config.json",
        "history.json",
        "compute.json",
        "camera_predictions.csv",
        "inference_check.json",
        "candidates.json",
        "best_only_oof.csv",
        "padding_diagnostic.json",
        "padding_pairs.csv",
        "checkpoint_summary.json",
    }
    if not required <= set(hashes):
        raise ValueError("Incomplete run manifest.")
    candidates = load_candidates(folder, selected_only=False, verify_predictions=True)
    candidate_files = {
        row[key] for row in candidates for key in ("weights", "predictions")
    }
    if not candidate_files <= set(hashes):
        raise ValueError(
            "Candidate files are missing from the completed fold manifest."
        )
    selected = load_candidates(folder)
    if result["selected_epochs"] != [row["epoch"] for row in selected]:
        raise ValueError("Completed result uses a different epoch selection.")
    if result["selected_checkpoint_count"] != len(selected):
        raise ValueError("Completed result has a different checkpoint count.")
    for name, digest in hashes.items():
        if not (folder / name).is_file() or file_hash(folder / name) != digest:
            raise ValueError("Missing or changed run artifact: " + name)
    return result


def _camera_subset(train, valid, train_camera):
    if train_camera is None:
        return train, valid
    train = train[
        train.device.map(camera_family).eq(camera_family(train_camera))
    ].copy()
    families = valid.assign(family=valid.device.map(camera_family))
    paired = families.groupby("sample_id").family.agg(set)
    paired_ids = paired[
        paired.map(lambda names: {"motorola", "samsung"} <= names)
    ].index
    valid = valid[valid.sample_id.isin(paired_ids)].copy()
    if train.empty or valid.empty:
        raise ValueError(
            "A cross-camera fold needs training photos and paired held-out soils."
        )
    return train, valid


def _optimizer(model, recipe):
    groups = make_param_groups(
        model.head, lr=recipe["head_lr"], weight_decay=recipe["weight_decay"]
    )
    if not model.frozen:
        no_decay = [
            name
            for name, _ in model.encoder.named_parameters()
            if name == "cls" or "relative_position_bias_table" in name
        ]
        groups += make_param_groups(
            model.encoder,
            lr=recipe["backbone_lr"],
            weight_decay=recipe["weight_decay"],
            layer_ids=encoder_layer_ids(model.encoder),
            layer_decay=recipe["layer_decay"],
            no_weight_decay=no_decay,
        )
    return torch.optim.AdamW(groups)


def _scheduler(optimizer, recipe):
    def factor(epoch):
        if epoch < recipe["warmup"]:
            return (epoch + 1) / recipe["warmup"]
        progress = (epoch - recipe["warmup"]) / max(
            1, recipe["epochs"] - recipe["warmup"] - 1
        )
        return 0.05 + 0.95 * (1 + math.cos(math.pi * min(1, progress))) / 2

    if recipe["schedule"] == "constant":
        return None
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def run_fold(recipe, fold, train_camera=None):
    torch.set_num_threads(2)
    seed_everything(42 + fold, deterministic=True)
    metadata = json.loads((ROOT / "metadata.json").read_text())
    if metadata["identity"]["soilcv"] != package_identity():
        raise ValueError("Package changed after preparation.")
    labels = pd.read_json(ROOT / "labels.json")
    manifest = pd.read_json(ROOT / "frames.json")
    train = manifest[manifest.fold.notna() & manifest.fold.ne(fold)].copy()
    valid = manifest[manifest.fold.eq(fold)].copy()
    test = manifest[manifest.fold.isna()].copy()
    train, valid = _camera_subset(train, valid, train_camera)
    train_labels = labels[labels.sample_id.isin(train.sample_id)]
    valid_labels = labels[labels.sample_id.isin(valid.sample_id)]
    run_name = (
        recipe["id"]
        if train_camera is None
        else recipe["id"] + "__" + train_camera.lower()
    )
    recipe_hash = identity_hash(
        {
            "recipe": recipe,
            "identity": metadata["identity"],
            "fold": fold,
            "train_camera": train_camera,
        }
    )
    folder = ROOT / "runs" / run_name / f"fold_{fold}"
    if (folder / "result.json").exists():
        return verify_completed(folder, recipe_hash)
    if (folder / "candidates.json").exists():
        raise ValueError(
            "This fold has a partial training run. Resume it explicitly or use a fresh output folder."
        )
    prior = train_labels[list(TARGET_COLUMNS)].to_numpy().mean(0)
    train_loader = make_loader(train, recipe, train_labels, train=True, seed=242 + fold)
    valid_loader = make_loader(valid, recipe, valid_labels)
    model = FullFrameModel(recipe, prior).to("cuda:0")
    optimizer = _optimizer(model, recipe)
    purpose = "standard" if train_camera is None else "domain"
    config = ExperimentConfig(
        run_name,
        stage="vit_fullframe",
        model=recipe["arch"],
        fold=fold,
        seed=42 + fold,
        hypothesis_id=recipe["hypothesis"] if train_camera is None else "H-VIT-08",
        output_root=str(ROOT / "runs"),
        versions={
            "soilcv": metadata["identity"]["soilcv"],
            "run_id": metadata["run_id"],
            "identity": metadata["identity"],
        },
        params={
            **recipe,
            "normalization": NORMALIZE,
            "train_mean_prior": prior.tolist(),
            "train_camera": train_camera,
            "purpose": purpose,
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "loss_units": "percent-log10",
        },
    )
    with start_run(config) as run:
        best_oof, history = fit_supervised_fold(
            model,
            train_loader,
            valid_loader,
            valid_labels,
            optimizer,
            EMDLoss(),
            config,
            epochs=recipe["epochs"],
            device="cuda:0",
            amp=True,
            scheduler=_scheduler(optimizer, recipe),
            patience=recipe["patience"],
            accumulation_steps=recipe["accumulation_steps"],
            clip_grad=5,
            frozen_modules=(model.encoder,) if model.frozen else (),
            run=run,
            validation_fn=validate_bags_camera,
            selection="emd",
            selection_start_epoch=0,
            keep_top_k=recipe["keep_top_k"],
        )
        best_oof.to_csv(folder / "best_only_oof.csv", index=False)
        oof = candidate_predictions(folder)
        oof["fold"] = fold
        oof.to_csv(folder / "oof.csv", index=False)
        selected = load_candidates(folder)
        camera_curves = []
        test_curves = []
        for weights in candidate_paths(folder):
            model.load_state_dict(
                torch.load(weights, map_location="cpu", weights_only=True)
            )
            _, camera = predict_bags(
                model, valid_loader, device="cuda:0", amp=True, cameras=True
            )
            camera_curves.append(camera)
            predictions, _ = predict_bags(
                model, make_loader(test, recipe), device="cuda:0", amp=True
            )
            test_curves.append(predictions)
        camera = mean_curves(camera_curves, keys=["sample_id", "device"])
        camera["fold"] = fold
        camera.to_csv(folder / "camera_predictions.csv", index=False)
        predictions = mean_curves(test_curves)
        predictions.to_csv(folder / "test_predictions.csv", index=False)
        ensemble_metrics = {
            **evaluate_samples(valid_labels, oof),
            **camera_metrics(valid_labels, camera, require_pairs=True),
            "checkpoint_count": len(selected),
        }
        run.summary.update(
            {f"ensemble/{name}": value for name, value in ensemble_metrics.items()}
        )
        # Check all test soils through the same selected epoch ensemble used in public inference.
        from .inference import predict_fold

        photos = pd.read_json(ROOT / "photos.json")
        check = predict_fold(
            folder, photos[photos.split.eq("test")], CACHE, device="cuda:0", amp=True
        )
        before = predictions.set_index("sample_id")[list(TARGET_COLUMNS)].sort_index()
        after = check.set_index("sample_id")[list(TARGET_COLUMNS)].sort_index()
        if not before.index.equals(after.index):
            raise ValueError("Reloaded inference changed test IDs.")
        difference = float(np.abs(before.to_numpy() - after.to_numpy()).max())
        if difference > 1e-3:
            raise ValueError("Reloaded inference differs from saved test curves.")
        write_json(
            folder / "inference_check.json",
            {
                "samples": len(check),
                "max_absolute_difference": difference,
                "tolerance": 1e-3,
                "amp": True,
                "selected_epochs": [row["epoch"] for row in selected],
            },
        )
        # Only a held-out sample is used; this is a diagnostic, never a selection metric.
        best_weights = (
            folder / load_candidates(folder, selected_only=False)[0]["weights"]
        )
        model.load_state_dict(
            torch.load(best_weights, map_location="cpu", weights_only=True)
        )
        from .diagnostics import padding_sensitivity

        padding, padding_pairs = padding_sensitivity(
            model, [next(iter(valid_loader))], device="cuda:0", amp=True
        )
        padding_pairs.to_csv(folder / "padding_pairs.csv", index=False)
        write_json(folder / "padding_diagnostic.json", padding)
        run.summary.update(
            {
                f"padding/{name}": value
                for name, value in padding.items()
                if isinstance(value, (int, float))
            }
        )
        import wandb

        model_artifact = wandb.Artifact(
            f"{run_name}-fold-{fold}",
            type="model",
            metadata={"run_id": metadata["run_id"], "recipe_hash": recipe_hash},
        )
        artifact_files = [folder / "config.json", folder / "candidates.json"]
        for candidate in load_candidates(folder, selected_only=False):
            artifact_files += [
                folder / candidate["weights"],
                folder / candidate["predictions"],
            ]
        for path in artifact_files:
            name = str(path.relative_to(folder))
            model_artifact.add_file(str(path), name=name)
        artifact = run.log_artifact(model_artifact)
        artifact.wait()
        log_artifact(
            run,
            folder / "oof.csv",
            name=f"{run_name}-fold-{fold}-oof",
            artifact_type="predictions",
        )
        best_epoch = load_candidates(folder, selected_only=False)[0]["epoch"]
        checkpoint_summary = {
            "best_epoch": best_epoch,
            "best_emd": evaluate_samples(valid_labels, best_oof)["emd"],
            "early_stopping_metric": "validation EMD only",
            "selected_epochs": [row["epoch"] for row in selected],
            "selection_rule": "top2_emd_plus_third_if_camera_better_than_second",
            "optimizer_checkpoints_kept": False,
            "reason": "Finished fold. Keep all top-three weight files within Kaggle Outputs budget.",
        }
        write_json(folder / "checkpoint_summary.json", checkpoint_summary)
        result = {
            "experiment": recipe["id"],
            "fold": fold,
            "purpose": purpose,
            "train_camera": train_camera,
            "run_id": metadata["run_id"],
            "recipe_hash": recipe_hash,
            **evaluate_samples(valid_labels, oof),
            **camera_metrics(valid_labels, camera, require_pairs=True),
            **json.loads((folder / "compute.json").read_text()),
            "epochs_ran": len(history),
            "best_epoch": best_epoch,
            "best_emd": evaluate_samples(valid_labels, best_oof)["emd"],
            "selected_epochs": [row["epoch"] for row in selected],
            "selected_checkpoint_count": len(selected),
            "padding_max_emd": padding["padding_max_emd"],
            "prediction_rule": "top2_emd_plus_third_if_camera_better_than_second",
            "run_url": run.url,
            "checkpoint_artifact": artifact.qualified_name,
        }
        if train_camera is not None:
            domain, pairs = domain_metrics(valid_labels, camera, train_camera)
            result.update(domain)
            run.summary.update(
                {f"domain/{name}": value for name, value in domain.items()}
            )
            pairs.to_csv(folder / "domain_pairs.csv", index=False)
    # Only this finished fold's large temporary optimizer states are removed.
    # Earlier studies use other output roots and are never changed here.
    for temporary in ("best.pt", "last.pt"):
        (folder / temporary).unlink(missing_ok=True)
    names = [
        "oof.csv",
        "test_predictions.csv",
        "config.json",
        "history.json",
        "compute.json",
        "camera_predictions.csv",
        "inference_check.json",
        "candidates.json",
        "best_only_oof.csv",
        "padding_diagnostic.json",
        "padding_pairs.csv",
        "checkpoint_summary.json",
    ]
    for candidate in load_candidates(folder, selected_only=False):
        names += [candidate["weights"], candidate["predictions"]]
    if train_camera is not None:
        names.append("domain_pairs.csv")
    write_json(
        folder / "artifact_hashes.json",
        {name: file_hash(folder / name) for name in names},
    )
    write_json(folder / "result.json", result)
    print("Complete:", run_name, fold, "EMD:", result["emd"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train"])
    parser.add_argument("--experiment", default=RECIPES[0]["id"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train-camera", choices=["Motorola", "Samsung"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        run_fold(
            next(dict(r) for r in RECIPES if r["id"] == args.experiment),
            args.fold,
            args.train_camera,
        )
