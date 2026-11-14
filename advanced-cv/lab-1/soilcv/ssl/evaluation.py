"""Train-only frozen baselines and representation diagnostics."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from soilcv.curves import repair_curves


def frozen_predict(train, targets, query, strategy):
    if strategy == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=100, solver="svd"))
        model.fit(train, targets)
        return repair_curves(model.predict(query)), model
    if strategy != "knn":
        raise ValueError("Choose ridge or knn.")
    return weighted_knn(train, targets, query), None


def weighted_knn(train, targets, query, *, k=3, temperature=0.07):
    train, query = np.asarray(train, dtype=float), np.asarray(query, dtype=float)
    if not 1 <= k <= len(train) or temperature <= 0:
        raise ValueError("Set a valid neighbour count and positive temperature.")
    train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    similarity = query @ train.T
    nearest = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    logits = np.take_along_axis(similarity, nearest, axis=1) / temperature
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return repair_curves(np.einsum("nk,nkd->nd", weights, np.asarray(targets)[nearest]))


def linear_cka(before, after):
    """Centered linear CKA using N x N Gram matrices. Collapse is undefined."""
    x, y = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 3:
        raise ValueError("CKA needs at least three matched feature rows.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("CKA inputs must be finite.")
    x, y = x - x.mean(0), y - y.mean(0)
    k, l = x @ x.T, y @ y.T
    denominator = np.linalg.norm(k) * np.linalg.norm(l)
    return (
        None
        if denominator <= 1e-20
        else float(np.clip((k * l).sum() / denominator, 0, 1))
    )


def rankme(features):
    """Effective rank of centered descriptors, using singular-value entropy."""
    x = np.asarray(features, dtype=np.float64)
    singular = np.linalg.svd(x - x.mean(0), compute_uv=False)
    if singular.sum() <= 1e-12:
        return 0.0
    p = singular[singular > 0] / singular.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def representation_metrics(before, after):
    return {
        "cka": linear_cka(before, after),
        "rankme_before": rankme(before),
        "rankme_after": rankme(after),
        "feature_std_before": float(np.std(before, axis=0).mean()),
        "feature_std_after": float(np.std(after, axis=0).mean()),
        "rows": len(before),
        "unit": "matched held-out physical tiles; descriptive, not a selection metric",
    }
