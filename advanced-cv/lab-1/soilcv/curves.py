"""Curve units, sample IDs, and equal photo/sample aggregation."""

import numpy as np
import pandas as pd

TARGET_COLUMNS = (
    "0.002",
    "0.0063",
    "0.02",
    "0.063",
    "0.2",
    "0.63",
    "2",
    "6.3",
    "20",
    "63",
    "200",
)
SIEVE_MM = np.array(TARGET_COLUMNS, dtype=float)
SIEVE_MM.setflags(write=False)
LOG_WIDTHS = np.diff(np.log10(SIEVE_MM))
LOG_WIDTHS.setflags(write=False)


def curve_array(values):
    array = np.asarray(values, dtype=float)
    if (
        array.ndim not in (1, 2)
        or array.shape[-1] != len(TARGET_COLUMNS)
        or array.size == 0
    ):
        raise ValueError("Expected one curve or a non-empty [samples, 11] array.")
    if not np.isfinite(array).all():
        raise ValueError("Curve values must be finite.")
    return array


def validate_curves(values, *, atol=1e-4):
    array = curve_array(values)
    if (array < -atol).any() or (array > 100 + atol).any():
        raise ValueError("Curves must use percentages from 0 to 100.")
    if (np.diff(array, axis=-1) < -atol).any():
        raise ValueError("Curves must not go down.")
    if not np.isclose(array[..., -1], 100, atol=atol, rtol=0).all():
        raise ValueError("Each curve must end at 100.")
    return array


def repair_curves(values):
    """Explicit clip + cumulative-max repair, as used in EDA. Not isotonic fitting."""
    array = np.maximum.accumulate(np.clip(curve_array(values), 0, 100), axis=-1)
    array[..., -1] = 100
    return array


def check_ids(frame, *, unique=True):
    if "sample_id" not in frame or frame.empty or frame.sample_id.isna().any():
        raise ValueError("A non-empty sample_id column is required.")
    if frame.sample_id.astype(str).str.strip().eq("").any():
        raise ValueError("Sample IDs cannot be empty.")
    if unique and frame.sample_id.duplicated().any():
        raise ValueError("Expected exactly one row per sample_id.")


def align_samples(reference, predictions):
    """Join by IDs; reject extra, missing, or repeated predictions."""
    check_ids(reference)
    check_ids(predictions)
    if set(reference.sample_id) != set(predictions.sample_id):
        raise ValueError("Prediction IDs must exactly match reference IDs.")
    return predictions.set_index("sample_id").loc[reference.sample_id].reset_index()


def _aggregate(frame, columns):
    check_ids(frame, unique=False)
    required = ["sample_id", "photo", "tile_id", *columns]
    if (
        any(column not in frame for column in required)
        or frame[required].isna().any().any()
    ):
        raise ValueError("Tiles need sample_id, photo, tile_id, and all value columns.")
    if frame.duplicated(["sample_id", "photo", "tile_id"]).any():
        raise ValueError("Repeated tile IDs would change aggregation weights.")
    if not np.isfinite(frame[list(columns)].to_numpy(float)).all():
        raise ValueError("Values must be finite.")
    photo_values = frame.groupby(["sample_id", "photo"], sort=False)[
        list(columns)
    ].mean()
    return photo_values.groupby("sample_id", sort=False).mean().reset_index()


def aggregate_features(frame, columns):
    """Mean tiles per photo, then mean photos per sample."""
    if not columns:
        raise ValueError("Set feature column names.")
    return _aggregate(frame, columns)


def aggregate_predictions(frame, *, check=True):
    if check:
        validate_curves(frame[list(TARGET_COLUMNS)])
    result = _aggregate(frame, TARGET_COLUMNS)
    if check:
        validate_curves(result[list(TARGET_COLUMNS)])
    return result


def make_submission(predictions, template, path=None):
    result = align_samples(template, predictions)[["sample_id", *TARGET_COLUMNS]]
    validate_curves(result[list(TARGET_COLUMNS)], atol=0)
    if path is not None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
    return result
