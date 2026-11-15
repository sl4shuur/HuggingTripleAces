import json

import pytest
from click import ClickException
from click.testing import CliRunner

from alseg.config import ExperimentConfig
from script.setup_project import main


def test_setup_reports_runtime_without_requiring_dataset():
    result = CliRunner().invoke(main, [])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["data_checked"] is False
    assert report["python"]
    assert report["torch"]


def test_setup_help_lists_supported_options():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--check-data" in result.output


def test_missing_voc_data_is_a_cli_error(tmp_path):
    config = ExperimentConfig(data={"dataset_root": str(tmp_path)}).save(
        tmp_path / "config.json"
    )
    result = CliRunner().invoke(
        main, ["--config", str(config), "--check-data"], standalone_mode=False
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ClickException)
    assert "VOC2012_test alone is insufficient" in str(result.exception)


def test_training_cli_loads_private_config_without_starting_training(
    tmp_path, monkeypatch
):
    pytest.importorskip("lightning")
    from alseg import pipeline
    from script.train_segmentation import main as train

    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_API_KEY=private-cli-test-key\n")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    calls = []

    def record_run(config, run_dir, budget, device, tracking, **kwargs):
        calls.append((config, budget, tracking))
        return {"budget": budget}

    monkeypatch.setattr(pipeline, "run_training", record_run)
    result = CliRunner().invoke(
        train, ["--env-file", str(env_file), "--wandb-mode", "offline"]
    )
    assert result.exit_code == 0, result.output
    config, budget, tracking = calls[0]
    assert budget == 21 and config["task"]["epochs"] == 20
    assert tracking.mode == "offline"
    assert tracking.api_key.get_secret_value() == "private-cli-test-key"
    assert "private-cli-test-key" not in result.output


def test_training_cli_honors_file_values_and_explicit_overrides(tmp_path, monkeypatch):
    pytest.importorskip("lightning")
    from alseg import pipeline
    from script.train_segmentation import main as train

    config_path = ExperimentConfig(budget=7, device="cpu", task={"epochs": 3}).save(
        tmp_path / "experiment.json"
    )
    calls = []

    def record_run(config, run_dir, budget, device, tracking, **kwargs):
        calls.append((budget, device, config["task"]["epochs"]))
        return {"budget": budget}

    monkeypatch.setattr(pipeline, "run_training", record_run)
    args = ["--config", str(config_path), "--wandb-mode", "disabled"]
    result = CliRunner().invoke(train, args)
    assert result.exit_code == 0, result.output
    assert calls[-1] == (7, "cpu", 3)
    result = CliRunner().invoke(train, args + ["--budget", "9"])
    assert result.exit_code == 0, result.output
    assert calls[-1] == (9, "cpu", 3)


def test_experiment_cli_runs_configured_sequence_and_overrides(tmp_path, monkeypatch):
    pytest.importorskip("lightning")
    from alseg import pipeline
    from script.run_experiment import main as experiment

    path = ExperimentConfig(
        budgets=[21, 42],
        encoder={"name": "ibot_vitb16"},
        device="cpu",
        task={"epochs": 3},
    ).save(tmp_path / "experiment.json")
    calls = []

    def record(config, run_dir, device, tracking, **kwargs):
        calls.append((config, run_dir, device, tracking, kwargs))
        return [{"budget": b} for b in config["budgets"]]

    monkeypatch.setattr(pipeline, "run_experiment", record)
    result = CliRunner().invoke(
        experiment,
        [
            "--config",
            str(path),
            "--wandb-mode",
            "disabled",
            "--smoke",
            "--output-dir",
            str(tmp_path / "run"),
        ],
    )
    assert result.exit_code == 0, result.output
    config, run_dir, device, tracking, kwargs = calls[0]
    assert config["budgets"] == [21, 42] and config["task"]["epochs"] == 2
    assert config["encoder"]["name"] == "ibot_vitb16"
    assert device == "cpu" and tracking.mode == "disabled" and kwargs["smoke"]
    assert run_dir == tmp_path / "run"
