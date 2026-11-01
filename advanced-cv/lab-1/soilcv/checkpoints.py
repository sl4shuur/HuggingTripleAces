"""Keep the best EMD epochs and choose a small prediction ensemble."""

import json
import math
import re
import shutil
from pathlib import Path

import pandas as pd
import torch

from .artifacts import file_hash, save_predictions, write_json
from .predictions import mean_curves

SCHEMA = "top_emd_v1"
SELECTION = "top_two_plus_third_if_camera_better_than_second"


def selected_candidates(records):
    """Take EMD ranks 1 and 2; take rank 3 only for lower camera disagreement."""
    ranked = sorted(records, key=lambda item: (item["emd"], item["epoch"]))
    if len({item["epoch"] for item in ranked}) != len(ranked):
        raise ValueError("Candidate epochs must be unique.")
    if any(not math.isfinite(item["emd"]) for item in ranked):
        raise ValueError("Candidate EMD must be finite.")
    chosen = ranked[:2]
    if len(ranked) >= 3:
        second = ranked[1].get("camera_disagreement")
        third = ranked[2].get("camera_disagreement")
        if (
            second is not None
            and third is not None
            and math.isfinite(second)
            and math.isfinite(third)
            and third < second
        ):
            chosen.append(ranked[2])
    return chosen


def _candidate_path(root, relative):
    path = Path(relative)
    if path.parent != Path("candidates") or not re.fullmatch(
        r"epoch_\d+(_oof\.csv|\.pt)", path.name
    ):
        raise ValueError("Unexpected candidate sidecar path.")
    return Path(root) / path


def write_candidate_index(root, records, top_k):
    records = sorted(records, key=lambda item: (item["emd"], item["epoch"]))
    selected = selected_candidates(records)
    write_json(
        Path(root) / "candidates.json",
        {
            "schema": SCHEMA,
            "top_k": top_k,
            "selection": SELECTION,
            "candidates": records,
            "selected_epochs": [item["epoch"] for item in selected],
        },
    )


def prune_candidate_files(root, records):
    """Delete only this pool's known file names, after last.pt is safely saved."""
    root = Path(root)
    keep = {item[key] for item in records for key in ("weights", "predictions")}
    for path in (root / "candidates").glob("epoch_*"):
        relative = str(path.relative_to(root))
        if (
            re.fullmatch(r"epoch_\d+(_oof\.csv|\.pt)", path.name)
            and relative not in keep
        ):
            path.unlink()


def update_candidates(
    root, records, model, epoch, metrics, predictions, *, top_k=3, prune=True
):
    """Write one weight state and its OOF curves only if it enters the EMD top-k."""
    if top_k not in {1, 2, 3}:
        raise ValueError("Keep one, two or three EMD checkpoints.")
    emd = float(metrics["emd"])
    if not math.isfinite(emd):
        raise ValueError("Candidate EMD must be finite.")
    if any(item["epoch"] == epoch for item in records):
        raise ValueError("An epoch can enter the candidate pool only once.")
    if len(records) >= top_k:
        worst = max(records, key=lambda item: (item["emd"], item["epoch"]))
        if (emd, epoch) >= (worst["emd"], worst["epoch"]):
            return records
    root = Path(root)
    directory = root / "candidates"
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / f"epoch_{epoch:04d}.pt"
    temporary = weights.with_suffix(".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(weights)
    prediction_path = directory / f"epoch_{epoch:04d}_oof.csv"
    save_predictions(predictions, prediction_path)
    camera = metrics.get("camera_disagreement")
    camera = float(camera) if camera is not None and math.isfinite(camera) else None
    record = {
        "epoch": int(epoch),
        "emd": emd,
        "camera_disagreement": camera,
        "weights": str(weights.relative_to(root)),
        "weights_sha256": file_hash(weights),
        "predictions": str(prediction_path.relative_to(root)),
        "predictions_sha256": file_hash(prediction_path),
    }
    records = sorted([*records, record], key=lambda item: (item["emd"], item["epoch"]))[
        :top_k
    ]
    write_candidate_index(root, records, top_k)
    if prune:
        prune_candidate_files(root, records)
    return records


def load_candidates(root, *, selected_only=True, verify_predictions=False):
    root = Path(root)
    manifest = json.loads((root / "candidates.json").read_text())
    if manifest["schema"] != SCHEMA or manifest["selection"] != SELECTION:
        raise ValueError("Unsupported checkpoint candidate selection.")
    records = manifest["candidates"]
    if (
        not records
        or len(records) > manifest["top_k"]
        or manifest["top_k"] not in {1, 2, 3}
    ):
        raise ValueError("Invalid checkpoint candidate count.")
    if records != sorted(records, key=lambda item: (item["emd"], item["epoch"])):
        raise ValueError("Checkpoint candidates must be sorted by validation EMD.")
    chosen = selected_candidates(records)
    if manifest["selected_epochs"] != [item["epoch"] for item in chosen]:
        raise ValueError("Saved candidate selection differs from the rule.")
    for item in records:
        keys = ("weights", "predictions") if verify_predictions else ("weights",)
        for key in keys:
            path = _candidate_path(root, item[key])
            if not path.is_file() or file_hash(path) != item[key + "_sha256"]:
                raise ValueError("Missing or changed candidate file: " + item[key])
    return chosen if selected_only else records


def candidate_paths(root):
    return [_candidate_path(root, item["weights"]) for item in load_candidates(root)]


def candidate_predictions(root):
    """Use the same selected epochs for OOF and test prediction."""
    records = load_candidates(root, verify_predictions=True)
    frames = [
        pd.read_csv(_candidate_path(root, item["predictions"])) for item in records
    ]
    return mean_curves(frames)


def restore_candidates(source, destination, records, top_k):
    """Resume from last.pt metadata, even if a newer index was written before a crash."""
    source, destination = Path(source), Path(destination)
    for item in records:
        for key in ("weights", "predictions"):
            original = _candidate_path(source, item[key])
            if not original.is_file() or file_hash(original) != item[key + "_sha256"]:
                raise ValueError(
                    "Resume needs the saved candidate sidecars: " + item[key]
                )
            target = _candidate_path(destination, item[key])
            target.parent.mkdir(parents=True, exist_ok=True)
            if original.resolve() != target.resolve():
                shutil.copy2(original, target)
    if records:
        write_candidate_index(destination, records, top_k)
    prune_candidate_files(destination, records)
