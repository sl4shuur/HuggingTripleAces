import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from script.setup_project import main


def test_setup_reports_runtime_without_requiring_dataset():
    config = Path(__file__).parents[1] / "configs" / "base.toml"
    result = CliRunner().invoke(main, ["--config", str(config)])
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
    config = tmp_path / "config.toml"
    config.write_text(
        f'[data]\ndataset_root = "{tmp_path.as_posix()}"\nvoc_root = ""\n'
    )
    result = CliRunner().invoke(main, ["--config", str(config), "--check-data"])
    assert result.exit_code == 1
    assert "VOC2012_test alone is insufficient" in result.output
    assert "Traceback" not in result.output


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
