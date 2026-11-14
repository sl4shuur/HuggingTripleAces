"""One CSV per checkpoint or complete four-fold ensemble. No training or W&B run."""

import gc
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from soilcv.artifacts import identity_hash, write_json
from soilcv.curves import (
    align_samples,
    make_submission,
)
from soilcv.predictions import mean_curves

DEFAULT_PROJECT = "public-team/soil-grain-size"


def run_path(reference):
    """Accept a run ID, entity/project/run, or a W&B/Forge run URL."""
    if "://" in reference:
        parts = urlparse(reference).path.strip("/").split("/")
        if "runs" not in parts:
            raise ValueError("Use a W&B run URL.")
        index = parts.index("runs")
        if index < 2 or index + 1 >= len(parts):
            raise ValueError("Incomplete run URL.")
        return "/".join([parts[index - 2], parts[index - 1], parts[index + 1]])
    if "/" not in reference:
        return DEFAULT_PROJECT + "/" + reference
    if len(reference.split("/")) != 3:
        raise ValueError("Use entity/project/run_id or an exact model artifact.")
    return reference


def resolve_model(api, reference):
    """Find only a committed model artifact. An unfinished run returns no model."""
    if ":" in reference and "://" not in reference:
        artifact = api.artifact(reference, type="model")
        run = artifact.logged_by()
        if run is None:
            raise ValueError("Model artifact has no source run.")
        if artifact.state != "COMMITTED":
            return None, run
        return artifact, run
    run = api.run(run_path(reference))
    models = [
        item
        for item in run.logged_artifacts()
        if item.type == "model" and item.state == "COMMITTED"
    ]
    if not models:
        return None, run
    if len(models) != 1:
        raise ValueError(
            "This run has several model artifacts. Use an exact artifact version."
        )
    return models[0], run


def ensemble_signature(config):
    """Only fold, train prior, and worker placement may differ within a recipe."""
    params = {
        key: value
        for key, value in config["params"].items()
        if key not in {"train_mean_prior", "physical_gpu"}
    }
    versions = config["versions"]
    if not versions.get("run_id"):
        raise ValueError("Four-fold ensembles need the same saved study run_id.")
    return identity_hash(
        {
            "experiment_id": config["experiment_id"],
            "stage": config["stage"],
            "model": config["model"],
            "params": params,
            "study": versions["run_id"],
            "soilcv": versions["soilcv"],
        }
    )


def validate_ensemble(configs):
    if len(configs) != 4:
        raise ValueError("An ensemble needs exactly four ready fold checkpoints.")
    if {config["fold"] for config in configs} != {0, 1, 2, 3}:
        raise ValueError("Use each fold exactly once: 0, 1, 2, 3.")
    if len({ensemble_signature(config) for config in configs}) != 1:
        raise ValueError("All four folds must use the same recipe, source, and study.")


def ensemble_predictions(predictions, template):
    """Align before averaging. A wrong sample ID must never become a silent NaN."""
    if len(predictions) != 4:
        raise ValueError("Expected four fold predictions.")
    aligned = [align_samples(template, frame) for frame in predictions]
    return align_samples(template[["sample_id"]], mean_curves(aligned))


def predict_checkpoint(folder, config, photos, cache_root, device):
    """Use the model's saved preprocessing. New architectures have one dispatch here."""
    if config["stage"] == "cnn_mil":
        from torch.utils.data import DataLoader

        from soilcv.bags import SampleBags, cache_grid, make_grid_manifest, one_bag
        from soilcv.cnn.inference import load_model
        from soilcv.cnn.model import image_transform
        from soilcv.cnn.predict import predict_bags

        model, recipe = load_model(folder, device)
        manifest = make_grid_manifest(
            photos, tile_mm=recipe["tile_mm"], output_px=recipe["pixels"]
        )
        cache_grid(manifest, cache_root)
        dataset = SampleBags(manifest, cache_root, image_transform(recipe))
        loader = DataLoader(dataset, batch_size=1, collate_fn=one_bag, num_workers=0)
        predictions, _ = predict_bags(model, loader, device=device, amp=True)
    elif config["stage"] == "vit_fullframe":
        from soilcv.vit.inference import predict_fold

        predictions = predict_fold(folder, photos, cache_root, device=device, amp=True)
        model = None
    else:
        raise ValueError("Unsupported model stage: " + config["stage"])
    del model
    gc.collect()
    import torch

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return predictions


def _selection(item):
    if isinstance(item, str) and item.strip():
        return None, [item.strip()]
    if isinstance(item, dict):
        references = item.get("checkpoints", [])
        if len(references) != 4 or any(
            not isinstance(ref, str) or not ref.strip() for ref in references
        ):
            raise ValueError("Fill all four checkpoints in each ensemble group.")
        if len(set(references)) != 4:
            raise ValueError("Repeated checkpoint in an ensemble group.")
        return item.get("name", "ensemble"), [ref.strip() for ref in references]
    if item is not None:
        raise ValueError("Use a checkpoint string, a four-fold group, or None.")
    return None


def run_checkpoints(checkpoints, output_dir, *, data_root=None, device="auto"):
    """Strings make one-model CSVs. Dicts with four checkpoints make ensemble CSVs."""
    import torch
    import wandb

    from soilcv.data import load_metadata

    selected = [(index, _selection(item)) for index, item in enumerate(checkpoints, 1)]
    selected = [
        (index, selection) for index, selection in selected if selection is not None
    ]
    if not selected:
        raise ValueError("Fill at least one checkpoint or four-fold group.")
    torch.set_num_threads(2)
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if data_root is None:
        data_root = next(Path("/kaggle/input").rglob("sample_submission.csv")).parent
    _, photos, template = load_metadata(data_root)
    photos = photos[photos.split.eq("test")].copy()
    api = wandb.Api()
    results = []
    skipped = []
    manifest_path = output_dir.parent / "inference_manifest.json"
    with tempfile.TemporaryDirectory(prefix="soil-inference-") as temporary:
        temporary = Path(temporary)
        for slot, (group_name, references) in selected:
            print(f"[{slot}] Resolve {len(references)} checkpoint(s)", flush=True)
            resolved = []
            for reference in references:
                artifact, run = resolve_model(api, reference)
                if artifact is None:
                    skipped.append(
                        {
                            "slot": slot,
                            "reference": reference,
                            "run": run.url,
                            "reason": "No committed model artifact yet",
                        }
                    )
                else:
                    resolved.append((artifact, run, dict(run.config)))
            if len(resolved) != len(references):
                # Never silently call a three-fold prediction a four-fold ensemble.
                print("Skip group: not all checkpoints are ready.", flush=True)
                write_json(manifest_path, {"predictions": results, "skipped": skipped})
                continue
            if group_name is not None:
                validate_ensemble([config for _, _, config in resolved])
            curves = []
            sources = []
            for number, (artifact, run, config) in enumerate(resolved):
                folder = temporary / f"model_{slot:02d}_{number}"
                artifact.download(root=str(folder))
                write_json(folder / "config.json", config)
                curves.append(
                    predict_checkpoint(
                        folder, config, photos, temporary / "cache", device
                    )
                )
                sources.append(
                    {
                        "artifact": artifact.qualified_name,
                        "run": run.url,
                        "fold": config["fold"],
                        "experiment": config["experiment_id"],
                    }
                )
            prediction = (
                ensemble_predictions(curves, template)
                if group_name is not None
                else curves[0]
            )
            name = group_name or sources[0]["artifact"].split("/")[-1]
            name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
            output = output_dir / f"submission_{slot:02d}_{name}.csv"
            pending = output.with_suffix(".tmp.csv")
            frame = make_submission(prediction, template, pending)
            pending.replace(output)
            results.append(
                {
                    "slot": slot,
                    "name": name,
                    "ensemble": group_name is not None,
                    "sources": sources,
                    "output": str(output),
                    "device": str(device),
                    "samples": len(frame),
                }
            )
            write_json(manifest_path, {"predictions": results, "skipped": skipped})
            print("Saved:", output.name, frame.shape, flush=True)
    write_json(manifest_path, {"predictions": results, "skipped": skipped})
    if not results:
        raise RuntimeError("No complete checkpoint selection is ready yet.")
    return results
