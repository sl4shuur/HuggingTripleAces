"""Paired camera checks on the same held-out physical samples."""

import numpy as np
import pandas as pd

from .curves import TARGET_COLUMNS, check_ids, validate_curves
from .metrics import emd_per_sample


def camera_family(name):
    name = str(name).lower()
    for family in ("motorola", "samsung"):
        if family in name:
            return family
    return None


def paired_camera_errors(labels, camera_predictions):
    """Return one row per sample with both Motorola and Samsung predictions.

    A missing camera is excluded from both errors, so the compared sample sets
    are identical. Multiple devices in one family must be pooled before this
    function, rather than silently changing their relative weight.
    """
    check_ids(labels)
    required = ["sample_id", "device", *TARGET_COLUMNS]
    if any(name not in camera_predictions for name in required):
        raise ValueError("Camera predictions need sample_id, device and curves.")
    if camera_predictions[required].isna().any().any():
        raise ValueError("Camera predictions contain missing values.")
    if not set(camera_predictions.sample_id).issubset(set(labels.sample_id)):
        raise ValueError("Camera prediction IDs must have held-out labels.")
    validate_curves(camera_predictions[list(TARGET_COLUMNS)])
    frame = camera_predictions.copy()
    frame["family"] = frame.device.map(camera_family)
    selected = frame[frame.family.notna()]
    if selected.duplicated(["sample_id", "family"]).any():
        raise ValueError("Use one pooled prediction per sample and camera family.")
    targets = labels.set_index("sample_id")[list(TARGET_COLUMNS)]
    rows = []
    for sample, group in selected.groupby("sample_id", sort=True):
        group = group.set_index("family")
        if not {"motorola", "samsung"}.issubset(group.index):
            continue
        motorola = group.loc[["motorola"], list(TARGET_COLUMNS)].to_numpy()
        samsung = group.loc[["samsung"], list(TARGET_COLUMNS)].to_numpy()
        truth = targets.loc[[sample]].to_numpy()
        rows.append(
            {
                "sample_id": sample,
                "motorola_emd": float(emd_per_sample(truth, motorola)[0]),
                "samsung_emd": float(emd_per_sample(truth, samsung)[0]),
                "camera_disagreement": float(emd_per_sample(motorola, samsung)[0]),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["sample_id", "motorola_emd", "samsung_emd", "camera_disagreement"],
    )


def camera_metrics(labels, camera_predictions, *, require_pairs=False):
    pairs = paired_camera_errors(labels, camera_predictions)
    if require_pairs and pairs.empty:
        raise ValueError("Camera selection needs at least one paired held-out sample.")
    metrics = {"camera_paired_samples": len(pairs)}
    for name in ("motorola_emd", "samsung_emd", "camera_disagreement"):
        metrics[name] = None if pairs.empty else float(pairs[name].mean())
    return metrics


def domain_metrics(labels, camera_predictions, train_camera):
    """Same vs cross errors for a model trained on ONE camera family only.

    The caller must restrict training photos. Validation uses the unchanged
    sample folds, with both errors measured on the exact same paired samples.
    This is an Android camera diagnostic, not an estimate of iPhone accuracy.
    """
    family = camera_family(train_camera)
    if family is None:
        raise ValueError("train_camera must be Motorola or Samsung.")
    pairs = paired_camera_errors(labels, camera_predictions)
    if pairs.empty:
        raise ValueError("Cross-camera evaluation needs paired held-out samples.")
    cross = "samsung" if family == "motorola" else "motorola"
    same_error = float(pairs[f"{family}_emd"].mean())
    cross_error = float(pairs[f"{cross}_emd"].mean())
    return {
        "domain_paired_samples": len(pairs),
        "e_same": same_error,
        "e_cross": cross_error,
        "delta_domain": cross_error - same_error,
        "camera_disagreement": float(pairs.camera_disagreement.mean()),
    }, pairs


def improves_checkpoint(metrics, best, *, selection="emd", min_delta=0.0):
    """Strict Pareto selection keeps EMD and camera disagreement separate.

    A camera-aware checkpoint can replace the selected one only when neither
    metric grows and at least one falls by more than min_delta. This can keep
    an early conservative model; there is no hidden weighted score or drift.
    """
    if selection not in {"emd", "emd_camera"}:
        raise ValueError("selection must be emd or emd_camera.")
    if not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and non-negative.")
    names = ("emd", "camera_disagreement") if selection == "emd_camera" else ("emd",)
    values = []
    for name in names:
        value = metrics.get(name)
        if value is None or not np.isfinite(value):
            raise ValueError(f"Checkpoint selection needs finite {name}.")
        values.append(float(value))
    if best is None:
        return True
    old = [float(best[name]) for name in names]
    return all(a <= b for a, b in zip(values, old)) and any(
        a < b - min_delta for a, b in zip(values, old)
    )
