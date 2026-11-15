import csv
import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lightning")
pytest.importorskip("transformers")
from transformers import Dinov2Config, Dinov2Model

from alseg.adapters.dinov2 import DinoV2Encoder
from alseg.adapters.voc import VocOracle
from alseg.config import EncoderName, ExperimentConfig, TrackingConfig
from alseg.pipeline import run_experiment, run_training

pytest_plugins = ["test_data"]


def test_random_budget_trains_and_reports_without_exposing_val_to_fit(
    voc, tmp_path, monkeypatch
):
    root, _ = voc
    encoder = DinoV2Encoder(
        Dinov2Model(
            Dinov2Config(
                image_size=28,
                patch_size=14,
                hidden_size=12,
                num_hidden_layers=1,
                num_attention_heads=3,
            )
        ),
        image_size=28,
    )
    original = {
        name: value.clone() for name, value in encoder.model.state_dict().items()
    }
    monkeypatch.setattr(
        DinoV2Encoder, "from_pretrained", lambda *args, **kwargs: encoder
    )
    config = ExperimentConfig().model_dump(mode="json")
    config["data"]["voc_root"] = str(root)
    config["paths"]["cache_dir"] = str(tmp_path / "cache")
    config["encoder"]["name"] = EncoderName.DINOV2
    config["encoder"]["image_size"] = 28
    config["task"]["epochs"] = 2
    run_dir = tmp_path / "run"
    result = run_training(
        config, run_dir, 2, "cpu", TrackingConfig(mode="disabled"), smoke=True
    )
    assert result["budget"] == result["n_fit"] == 2
    assert result["n_calibration"] == 0 and result["val_images"] == 1
    assert 0 <= result["miou"] <= 1
    assert result["smoke"] is True
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["valid_pixels"] == 5  # Original 2x3 mask, one void pixel.
    query = json.loads((run_dir / "oracle_queries.jsonl").read_text())
    assert len(query["ids"]) == 2 and query["status"] == "accepted"
    checkpoint = torch.load(run_dir / "head.pt", weights_only=True)
    assert checkpoint["state_dict"]["classifier.weight"].shape == (21, 12, 1, 1)
    for name, value in encoder.model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    saved_config = json.loads((run_dir / "config.json").read_text())
    assert saved_config["runtime"]["precision"] == "32-true"
    assert saved_config["tracking"]["wandb_entity"] == "public-team"
    assert (run_dir / "results.csv").is_file()
    assert len((run_dir / "training.csv").read_text().splitlines()) == 3


def test_random_rounds_query_only_new_ids(voc, tmp_path):
    from alseg.pipeline import _query_round

    root, manifest = voc
    oracle = VocOracle(root, manifest, tmp_path / "queries.jsonl")
    config = ExperimentConfig(protocol={"calibration_fraction": 0}).model_dump(
        mode="json"
    )
    first = _query_round(config, oracle, [0, 1], 0, 1, single=False)
    second = _query_round(config, oracle, first["unlabeled_ids"], 1, 2, single=False)
    assert len(first["new_ids"]) == len(second["new_ids"]) == 1
    assert set(first["new_ids"]).isdisjoint(second["new_ids"])
    assert sorted(second["ids"]) == [0, 1]
    assert second["unlabeled_ids"] == []
    assert second["fit_ids"] == [0, 1] and second["calibration_ids"] == []


@pytest.mark.parametrize("budgets", [[], [2, 1], [1, 1], [1, 3], [True]])
def test_invalid_budgets_fail_before_loading_encoder_or_querying(
    voc, tmp_path, monkeypatch, budgets
):
    from alseg import pipeline

    config = ExperimentConfig(data={"voc_root": str(voc[0])}).model_dump(mode="json")
    config["budgets"] = budgets
    monkeypatch.setattr(
        pipeline, "load_encoder", lambda *args: pytest.fail("Loaded encoder")
    )
    run_dir = tmp_path / "bad"
    with pytest.raises(ValueError, match="budgets"):
        run_experiment(config, run_dir, "cpu", TrackingConfig(mode="disabled"))
    assert not run_dir.exists()


def test_run_training_composes_stages_without_fitting_or_inference(
    voc, tmp_path, monkeypatch
):
    from alseg import pipeline

    root, _ = voc
    config = ExperimentConfig().model_dump(mode="json")
    config["data"]["voc_root"] = str(root)
    encoder = SimpleNamespace(describe=lambda: {"model_id": "unit-test-encoder"})
    monkeypatch.setattr(pipeline, "load_encoder", lambda *args: encoder)

    def prepare(config, pool, oracle, encoder, ids):
        assert oracle.labeled_ids == ids and len(ids) == 2
        assert not hasattr(pool, "get_mask")
        return torch.ones(2, 3, 2, 2), torch.zeros(2, 4, 4, dtype=torch.long)

    monkeypatch.setattr(pipeline, "_prepare_labeled_features", prepare)
    head = torch.nn.Conv2d(3, 21, 1)
    monkeypatch.setattr(
        pipeline,
        "fit_head",
        lambda *args: SimpleNamespace(
            head=head, history=[{"epoch": 0, "train/loss": 1.0}]
        ),
    )
    metrics = {
        "miou": 0.5,
        "pixel_accuracy": 0.75,
        "mean_class_accuracy": 0.6,
        "loss": 1.0,
    }

    def evaluate(received_head, received_encoder, source, ids, batch_size, device):
        assert received_head is head and received_encoder is encoder
        assert ids == [0] and source.get_mask(0).shape == (2, 3)
        return metrics, []

    monkeypatch.setattr(pipeline, "evaluate_head", evaluate)
    run_dir = tmp_path / "unit-run"
    result = run_training(config, run_dir, 2, "cpu", TrackingConfig(mode="disabled"))
    assert result["miou"] == 0.5 and result["n_fit"] == 2
    assert result["val_images"] == 1 and not result["smoke"]
    saved = json.loads((run_dir / "config.json").read_text())
    assert saved["val_ids"] == ["validation"]
    assert "api_key" not in saved["tracking"]
    assert (run_dir / "head.pt").is_file()
    assert (run_dir / "results.csv").is_file()


def test_failed_round_preserves_completed_metrics_and_query_state(
    voc, tmp_path, monkeypatch
):
    from alseg import pipeline

    config = ExperimentConfig(
        budgets=[1, 2],
        data={"voc_root": str(voc[0])},
        protocol={"calibration_fraction": 0},
    ).model_dump(mode="json")
    monkeypatch.setattr(
        pipeline,
        "load_encoder",
        lambda *args: SimpleNamespace(describe=lambda: {"name": "test"}),
    )
    monkeypatch.setattr(
        pipeline,
        "_prepare_labeled_features",
        lambda *args: (
            torch.ones(len(args[-1]), 3, 2, 2),
            torch.zeros(len(args[-1]), 4, 4, dtype=torch.long),
        ),
    )

    def fit(features, *args):
        if len(features) == 2:
            raise RuntimeError("training interrupted")
        return SimpleNamespace(head=torch.nn.Conv2d(3, 21, 1), history=[{"loss": 1.0}])

    monkeypatch.setattr(pipeline, "fit_head", fit)
    monkeypatch.setattr(
        pipeline,
        "evaluate_head",
        lambda *args: (
            {
                "miou": 0.5,
                "pixel_accuracy": 0.5,
                "mean_class_accuracy": 0.5,
                "loss": 1.0,
            },
            [],
        ),
    )
    run_dir = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="interrupted"):
        run_experiment(config, run_dir, "cpu", TrackingConfig(mode="disabled"))
    with (run_dir / "results.csv").open() as file:
        assert [int(r["budget"]) for r in csv.DictReader(file)] == [1]
    state = json.loads((run_dir / "state.json").read_text())
    assert state["budget"] == 2 and state["status"] == "queried"
    assert len((run_dir / "oracle_queries.jsonl").read_text().splitlines()) == 2
