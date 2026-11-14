"""Measure prediction changes when only the padded canvas changes."""

import numpy as np
import pandas as pd

from soilcv.cnn.predict import predict_bags
from soilcv.curves import TARGET_COLUMNS, align_samples
from soilcv.metrics import emd_per_sample


def padding_sensitivity(model, loader, *, device="cpu", amp=False):
    """Add one right/bottom alignment strip. Soil pixels and scale do not change.

    Scratch ViT masks fully padded keys and weights partial keys by valid area.
    Swin keeps its native window attention; this check measures its remaining
    sensitivity to the padded canvas. A small gap is not proof of no shortcuts.
    """
    encoder = model.encoder
    original = encoder.padding_extra
    extra = encoder.padding_multiple
    try:
        encoder.padding_extra = 0
        baseline, _ = predict_bags(model, loader, device=device, amp=amp)
        encoder.padding_extra = extra
        changed, _ = predict_bags(model, loader, device=device, amp=amp)
    finally:
        encoder.padding_extra = original
    changed = align_samples(baseline, changed)
    errors = emd_per_sample(
        baseline[list(TARGET_COLUMNS)].to_numpy(),
        changed[list(TARGET_COLUMNS)].to_numpy(),
    )
    pairs = pd.DataFrame({"sample_id": baseline.sample_id, "padding_emd": errors})
    return {
        "padding_mean_emd": float(np.mean(errors)),
        "padding_max_emd": float(np.max(errors)),
        "padding_samples": len(pairs),
        "padding_extra_pixels": extra,
    }, pairs
