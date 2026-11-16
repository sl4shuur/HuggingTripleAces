import csv
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("lightning")
pytest.importorskip("transformers")

from alseg import pipeline
from alseg.adapters.voc import VocOracle
from alseg.config import ExperimentConfig, TrackingConfig
from alseg.core.selection import AcquisitionMethod

pytest_plugins = ["test_data"]


@pytest.mark.parametrize("selected", [[], [1, 1], [0], [2], [True], [1.0], (1,)])
def test_invalid_selection_is_rejected_before_query(voc, tmp_path, selected):
    root, manifest = voc
    log = tmp_path / "queries.jsonl"
    oracle = VocOracle(root, manifest, log)
    oracle.query([0])
    previous_log = log.read_text()
    config = ExperimentConfig().model_dump(mode="json")
    with pytest.raises(ValueError):
        pipeline._query_round(
            config,
            oracle,
            [1],
            1,
            2,
            acquisition_result=(selected, {}),
        )
    assert oracle.labeled_ids == [0]
    assert log.read_text() == previous_log


def test_metadata_is_checked_before_query(voc, tmp_path):
    root, manifest = voc
    log = tmp_path / "queries.jsonl"
    oracle = VocOracle(root, manifest, log)
    config = ExperimentConfig().model_dump(mode="json")
    with pytest.raises(ValueError, match="metadata"):
        pipeline._query_round(
            config,
            oracle,
            [0, 1],
            0,
            1,
            acquisition_result=([1], None),
        )
    assert oracle.labeled_ids == []
    assert log.read_text() == ""


def test_nonrandom_without_callback_fails_before_encoder_loading(tmp_path, monkeypatch):
    config = ExperimentConfig(acquisition={"method": "uherding"}).model_dump(
        mode="json"
    )
    monkeypatch.setattr(
        pipeline, "load_encoder", lambda *args: pytest.fail("Loaded encoder")
    )
    with pytest.raises(ValueError, match="requires an acquisition callback"):
        pipeline.run_experiment(
            config, tmp_path / "run", "cpu", TrackingConfig(mode="disabled")
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("method", list(AcquisitionMethod))
def test_callback_uses_shared_reporting_loop_without_training(
    voc, tmp_path, monkeypatch, method
):
    config = ExperimentConfig(
        budgets=[1, 2], data={"voc_root": str(voc[0])}, acquisition={"method": method}
    ).model_dump(mode="json")
    encoder = SimpleNamespace(describe=lambda: {"name": "unit-test"})
    monkeypatch.setattr(pipeline, "load_encoder", lambda *args: encoder)
    calls = []

    def acquire(*, config, pool, oracle, encoder, round_index, budget):
        assert config["device"] == "cpu"
        assert len(pool) == 2 and not hasattr(pool, "get_mask")
        assert oracle.labeled_ids == ([] if round_index == 0 else [1])
        if round_index:
            assert oracle.get_mask(1).shape == (2, 3)
            with pytest.raises(KeyError):
                oracle.get_mask(0)
        calls.append((round_index, budget, encoder))
        return [1 - round_index], {
            "method": method.value,
            "selection_method": "random" if round_index == 0 else method.value,
            "acquisition_budget": len(oracle.labeled_ids),
            "calibration_fit_ids": [],
            "calibration_ids": oracle.labeled_ids,
            "sigma": 0.5,
            "tau": 2.0,
            "fallback": "cold_start" if round_index == 0 else None,
            "ece_by_temperature": {"2.0": 0.1},
        }

    def report(config, tracking, pool, oracle, encoder, evaluation, run_dir, elapsed):
        assert config["fit_ids"] == oracle.labeled_ids
        assert config["n_fit"] == config["budget"]
        assert config["calibration_ids"] == [] and config["n_calibration"] == 0
        metrics = {
            "miou": 0.5,
            "pixel_accuracy": 0.75,
            "mean_class_accuracy": 0.6,
            "loss": 1.0,
        }
        return pipeline._save_results(
            run_dir, config, metrics, {"selection_seconds": elapsed}
        )

    monkeypatch.setattr(pipeline, "_run_round", report)
    run_dir = tmp_path / method.value
    results = pipeline.run_experiment(
        config, run_dir, "cpu", TrackingConfig(mode="disabled"), acquisition=acquire
    )
    assert config["device"] == "cuda"
    assert calls == [(0, 1, encoder), (1, 2, encoder)]
    assert [row["n_fit"] for row in results] == [1, 2]
    assert [row["acquisition_budget"] for row in results] == [0, 1]
    assert [row["n_new"] for row in results] == [1, 1]
    assert all(row["method"] == method.value for row in results)
    assert results[0]["selection_method"] == "random"
    assert results[1]["selection_method"] == method.value
    assert results[1]["acquisition_n_fit"] == 0
    assert results[1]["acquisition_n_calibration"] == 1
    assert results[1]["sigma_used"] == 0.5 and results[1]["tau_used"] == 2.0
    selections = json.loads((run_dir / "selections.json").read_text())
    assert selections[1]["ids"] == selections[1]["fit_ids"] == [1, 0]
    assert selections[1]["acquisition"]["calibration_ids"] == [1]
    saved = json.loads((run_dir / "round-01-budget-2/config.json").read_text())
    assert saved["device"] == saved["runtime"]["device"] == "cpu"
    assert saved["acquisition_metadata"] == selections[1]["acquisition"]
    assert saved["acquisition_metadata"]["ece_by_temperature"] == {"2.0": 0.1}
    queries = [
        json.loads(line)
        for line in (run_dir / "oracle_queries.jsonl").read_text().splitlines()
    ]
    assert [query["ids"] for query in queries] == [[1], [0]]
    with (run_dir / "results.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [int(row["n_fit"]) for row in rows] == [1, 2]
    assert rows[1]["acquisition_n_calibration"] == "1"
