"""Official percent/log10 EMD and separate diagnostics. No silent curve repair."""

import numpy as np
import pandas as pd

from .curves import (
    LOG_WIDTHS,
    SIEVE_MM,
    TARGET_COLUMNS,
    align_samples,
    curve_array,
    validate_curves,
)


def emd_per_sample(true, pred):
    true, pred = curve_array(true), curve_array(pred)
    if true.shape != pred.shape:
        raise ValueError("True and predicted curve shapes must match.")
    return (np.abs(true - pred)[..., :-1] * LOG_WIDTHS).sum(axis=-1)


def grain_size(curve, percent=50):
    curve = validate_curves(curve)
    if curve.ndim != 1 or not 0 <= percent <= 100:
        raise ValueError("Use one curve and a percent from 0 to 100.")
    if percent < curve[0]:
        return np.nan
    right = np.searchsorted(curve, percent, side="left")
    if right == 0:
        return float(SIEVE_MM[0])
    left = right - 1
    fraction = (percent - curve[left]) / (curve[right] - curve[left])
    log_size = np.log10(SIEVE_MM)
    return float(10 ** (log_size[left] + fraction * (log_size[right] - log_size[left])))


def _aligned_arrays(labels, predictions):
    predictions = align_samples(labels, predictions)
    true = validate_curves(labels[list(TARGET_COLUMNS)])
    pred = validate_curves(predictions[list(TARGET_COLUMNS)])
    return true, pred


def evaluate_samples(labels, predictions):
    true, pred = _aligned_arrays(labels, predictions)
    absolute = np.abs(true - pred)
    result = {
        "samples": len(true),
        "emd": float(emd_per_sample(true, pred).mean()),
        "mae_pp": float(absolute.mean()),
        "worst_point_pp": float(absolute.max(axis=1).mean()),
    }
    for percent in [10, 50, 60]:
        a = np.array([grain_size(row, percent) for row in true])
        b = np.array([grain_size(row, percent) for row in pred])
        valid = np.isfinite(a) & np.isfinite(b)
        result[f"d{percent}_valid_pairs"] = int(valid.sum())
        result[f"d{percent}_skipped_pairs"] = int((~valid).sum())
        result[f"d{percent}_abs_log10"] = (
            float(np.abs(np.log10(a[valid] / b[valid])).mean()) if valid.any() else None
        )
    return result


def range_errors(labels, predictions):
    true, pred = _aligned_arrays(labels, predictions)
    errors = np.abs(true - pred)[:, :-1] * LOG_WIDTHS
    ranges = {
        "fine": SIEVE_MM[:-1] < 0.063,
        "sand": (SIEVE_MM[:-1] >= 0.063) & (SIEVE_MM[:-1] < 2),
        "coarse": SIEVE_MM[:-1] >= 2,
    }
    return pd.DataFrame(
        [
            {
                "range": name,
                "emd_part": float(errors[:, mask].sum(axis=1).mean()),
                "normalized_error_pp": float(
                    errors[:, mask].sum(axis=1).mean() / LOG_WIDTHS[mask].sum()
                ),
            }
            for name, mask in ranges.items()
        ]
    )


def group_errors(labels, predictions, group="soil_group"):
    true, pred = _aligned_arrays(labels, predictions)
    if group not in labels or labels[group].isna().any():
        raise ValueError("Every sample needs a group value.")
    frame = pd.DataFrame(
        {group: labels[group].to_numpy(), "emd": emd_per_sample(true, pred)}
    )
    return (
        frame.groupby(group)
        .emd.agg(samples="size", mean_emd="mean", median_emd="median")
        .reset_index()
    )


def summarize_cv(labels, predictions, folds, *, expected_folds=(0, 1, 2, 3)):
    """Require complete OOF IDs and the recorded validation fold for each row."""
    from .data import validate_folds

    validate_folds(folds, labels.sample_id, expected_folds=expected_folds)
    aligned = align_samples(labels, predictions)
    expected = folds.set_index("sample_id").loc[labels.sample_id, "fold"].to_numpy()
    if "fold" not in aligned or not np.array_equal(aligned.fold.to_numpy(), expected):
        raise ValueError("Each OOF row must carry its expected validation fold.")
    true, pred = _aligned_arrays(labels, aligned)
    frame = pd.DataFrame({"fold": expected, "emd": emd_per_sample(true, pred)})
    scores = frame.groupby("fold").emd.mean()
    return {
        "emd": float(frame.emd.mean()),
        "fold_mean": float(scores.mean()),
        "fold_std": float(scores.std(ddof=1)),
        "fold_scores": {str(k): float(v) for k, v in scores.items()},
        "samples": len(frame),
        "complete": True,
    }
