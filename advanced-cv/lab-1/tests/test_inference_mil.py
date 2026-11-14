"""Four-fold ensembles must preserve samples, recipes, and complete fold coverage."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from soilcv.curves import TARGET_COLUMNS
from soilcv.inference import identity
from soilcv.inference.runner import (
    ensemble_predictions,
    resolve_model,
    run_path,
    validate_ensemble,
)


@pytest.mark.parametrize(
    "reference",
    [
        "uvi5xre4",
        "public-team/soil-grain-size/uvi5xre4",
        "https://forge.coreweave.com/wandb/public-team/soil-grain-size/runs/uvi5xre4/files?nw=demo",
        "https://wandb.ai/public-team/soil-grain-size/runs/uvi5xre4",
    ],
)
def test_run_references(reference):
    assert run_path(reference) == "public-team/soil-grain-size/uvi5xre4"


def test_ready_model_is_selected_among_other_artifacts():
    model = SimpleNamespace(type="model", state="COMMITTED")
    run = SimpleNamespace(
        logged_artifacts=lambda: [
            SimpleNamespace(type="predictions", state="COMMITTED"),
            model,
        ]
    )
    api = SimpleNamespace(run=lambda path: run)
    artifact, source = resolve_model(api, "uvi5xre4")
    assert artifact is model and source is run


def test_incomplete_model_does_not_block_other_runs():
    run = SimpleNamespace(
        logged_artifacts=lambda: [SimpleNamespace(type="model", state="PENDING")]
    )
    api = SimpleNamespace(run=lambda path: run)
    assert resolve_model(api, "uvi5xre4") == (None, run)


def test_multiple_models_require_an_exact_version():
    model = SimpleNamespace(type="model", state="COMMITTED")
    run = SimpleNamespace(logged_artifacts=lambda: [model, model])
    api = SimpleNamespace(run=lambda path: run)
    with pytest.raises(ValueError, match="several model artifacts"):
        resolve_model(api, "uvi5xre4")


def fold_configs():
    return [
        {
            "fold": fold,
            "experiment_id": "mil-r18-cosine",
            "stage": "cnn_mil",
            "model": "resnet18",
            "params": {
                "schedule": "cosine",
                "train_mean_prior": [fold],
                "physical_gpu": str(fold % 2),
            },
            "versions": {
                "run_id": "one-study",
                "soilcv": {"source_hash": "one-source"},
            },
        }
        for fold in range(4)
    ]


def test_ensemble_requires_all_four_distinct_folds():
    configs = fold_configs()
    validate_ensemble(configs)
    with pytest.raises(ValueError, match="exactly four"):
        validate_ensemble(configs[:3])
    configs[3]["fold"] = 2
    with pytest.raises(ValueError, match="each fold exactly once"):
        validate_ensemble(configs)


@pytest.mark.parametrize("field", ["study", "recipe", "source"])
def test_ensemble_rejects_mixed_studies_and_recipes(field):
    configs = fold_configs()
    if field == "study":
        configs[3]["versions"]["run_id"] = "other-study"
    elif field == "recipe":
        configs[3]["params"]["schedule"] = "constant"
    else:
        configs[3]["versions"]["soilcv"]["source_hash"] = "other-source"
    with pytest.raises(ValueError, match="same recipe, source, and study"):
        validate_ensemble(configs)


def test_ensemble_aligns_ids_before_averaging_and_rejects_missing_sample():
    template = pd.DataFrame({"sample_id": ["H1", "H2"]})
    predictions = []
    for fold in range(4):
        values = np.stack([np.linspace(fold, 100, 11), np.linspace(10 + fold, 100, 11)])
        frame = pd.DataFrame(values, columns=TARGET_COLUMNS)
        frame.insert(0, "sample_id", ["H1", "H2"])
        predictions.append(frame.iloc[::-1] if fold % 2 else frame)
    result = ensemble_predictions(predictions, template)
    assert result.sample_id.tolist() == ["H1", "H2"]
    assert result["0.002"].tolist() == [1.5, 11.5]
    assert result["200"].eq(100).all()
    predictions[-1] = predictions[-1].iloc[:1]
    with pytest.raises(ValueError, match="exactly match"):
        ensemble_predictions(predictions, template)


def test_finished_mil_source_is_verified_after_package_extension(monkeypatch):
    current = {**identity.MIL_STUDY_SOURCE, "source_hash": "new-inference-module"}
    monkeypatch.setattr(identity, "package_identity", lambda: current)
    config = {
        "stage": "cnn_mil",
        "versions": {"soilcv": deepcopy(identity.MIL_STUDY_SOURCE)},
    }
    identity.check_model_source(config)
    monkeypatch.setattr(
        identity, "semantic_hash", lambda path: "changed-executable-code"
    )
    with pytest.raises(ValueError, match="dependency changed"):
        identity.check_model_source(config)


def test_unknown_source_is_not_accepted(monkeypatch):
    monkeypatch.setattr(identity, "package_identity", lambda: {"source_hash": "new"})
    config = {"stage": "cnn_mil", "versions": {"soilcv": {"source_hash": "unknown"}}}
    with pytest.raises(ValueError, match="source package"):
        identity.check_model_source(config)
