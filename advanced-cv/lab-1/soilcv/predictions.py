"""Average valid curves after matching their sample or camera IDs."""

import numpy as np

from .curves import TARGET_COLUMNS, validate_curves


def mean_curves(frames, keys=None):
    keys = ["sample_id"] if keys is None else keys
    if not frames:
        raise ValueError("Provide at least one prediction frame.")
    reference = frames[0].set_index(keys).sort_index()
    if not reference.index.is_unique:
        raise ValueError("Prediction keys must be unique.")
    arrays = []
    for frame in frames:
        aligned = frame.set_index(keys).sort_index()
        if not aligned.index.equals(reference.index):
            raise ValueError("Epoch predictions have different sample or camera IDs.")
        arrays.append(validate_curves(aligned[list(TARGET_COLUMNS)]))
    result = reference[[]].copy()
    result[list(TARGET_COLUMNS)] = np.mean(arrays, axis=0)
    return result.reset_index()
