import os
from io import StringIO

import pytest
from rich.console import Console

from alseg import config


def test_project_root_finds_markers_above_nested_package(tmp_path, monkeypatch):
    root = tmp_path / "dataset" / "source"
    package = root / "alseg"
    nested = package / "nested"
    nested.mkdir(parents=True)
    (root / "pyproject.toml").touch()
    # A nearer pyproject alone is not a complete ALSeg project root.
    (package / "pyproject.toml").touch()
    monkeypatch.setattr(config, "__file__", str(nested / "config.py"))
    monkeypatch.chdir(tmp_path)
    assert config._project_root() == root


def test_project_root_falls_back_to_cwd_without_project_markers(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "__file__", str(tmp_path / "installed" / "config.py"))
    monkeypatch.chdir(tmp_path)
    assert config._project_root() == tmp_path


def test_project_version_is_independent_of_cwd(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert config.get_project_version(tmp_path) == "1.2.3"


def test_json_snapshot_round_trip_and_validation(tmp_path):
    experiment = config.ExperimentConfig(task={"epochs": 7}, run_dir=tmp_path / "run")
    path = experiment.save(tmp_path / "launch" / "config.json")
    restored = config.ExperimentConfig.load(path)
    assert restored.model_dump() == experiment.model_dump()
    path.write_text('{"task": {"epochs": 0}}')
    with pytest.raises(ValueError):
        config.ExperimentConfig.load(path)
    path.write_text("not json")
    with pytest.raises(ValueError):
        config.ExperimentConfig.load(path)


def test_tracking_reads_unprefixed_dotenv_and_environment_overrides_it(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_API_KEY=file-key\nUNRELATED_VALUE=allowed\n")
    tracking = config.TrackingConfig(_env_file=env_file)
    assert tracking.api_key.get_secret_value() == "file-key"
    assert "api_key" not in tracking.model_dump()
    assert "file-key" not in repr(tracking)
    assert "WANDB_API_KEY" not in os.environ
    monkeypatch.setenv("WANDB_API_KEY", "environment-key")
    assert (
        config.TrackingConfig(_env_file=env_file).api_key.get_secret_value()
        == "environment-key"
    )


def test_tracking_can_use_a_notebook_secret_without_serializing_it(monkeypatch):
    from alseg.runtime.tracking import configure_tracking

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(
        "alseg.runtime.tracking.get_secret", lambda name: "notebook-key"
    )
    tracking = configure_tracking(config.TrackingConfig(_env_file=None))
    assert tracking.api_key.get_secret_value() == "notebook-key"
    assert "notebook-key" not in tracking.model_dump_json()


def test_offline_tracking_does_not_request_notebook_secrets(monkeypatch):
    from alseg.runtime.tracking import configure_tracking

    def unexpected_read(name):
        pytest.fail("Offline tracking must not read notebook secrets")

    monkeypatch.setattr("alseg.runtime.tracking.get_secret", unexpected_read)
    tracking = config.TrackingConfig(_env_file=None, mode="offline")
    assert configure_tracking(tracking) is tracking


def test_notebook_config_merges_sections_without_changing_shared_defaults():
    experiment = config.ExperimentConfig(task={"epochs": 3}, encoder={"batch_size": 1})
    original = config.ExperimentConfig()
    assert experiment.task.epochs == 3
    assert experiment.task.batch_size == original.task.batch_size
    assert experiment.encoder.batch_size == 1
    assert experiment.encoder.revision == original.encoder.revision
    assert config.ExperimentConfig().task.epochs == original.task.epochs


@pytest.mark.parametrize(
    "overrides",
    [
        {"task": {"epochs": 0}},
        {"task": {"learning_rate": -0.1}},
        {"task": {"batchsize": 2}},
        {"device": "gpu"},
        {"budgets": []},
        {"budgets": [21, 21]},
        {"budgets": [42, 21]},
        {"budgets": [0, 21]},
        {"protocol": {"calibration_fraction": 1}},
        {"protocol": {"calibration_min_images": -1}},
        {"acquisition": {"method": "unsupported"}},
        {"acquisition": {"uncertainty_method": "unsupported"}},
        {"acquisition": {"temperatures": []}},
        {"acquisition": {"temperatures": [0]}},
        {"acquisition": {"temperatures": [float("inf")]}},
        {"acquisition": {"temperatures": [float("nan")]}},
        {"acquisition": {"ece_bins": 0}},
        {"encoder": {"name": "ibot_segmentation"}},
        {"encoder": {"name": "dino_vitb16", "image_size": 225}},
        {"encoder": {"name": "dinov2", "checkpoint_path": "/tmp/unsupported.pth"}},
        {"seed": 2**63},
    ],
)
def test_notebook_config_rejects_invalid_values_and_misspelled_fields(overrides):
    with pytest.raises(ValueError):
        config.ExperimentConfig(**overrides)


def test_config_show_and_serialization_exclude_private_key(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    secret = "private-display-test-key"
    env_file.write_text(f"WANDB_API_KEY={secret}\n")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    experiment = config.ExperimentConfig.load(env_file=env_file)
    output = StringIO()
    console = Console(file=output, force_jupyter=False, width=120)
    monkeypatch.setattr(config, "Console", lambda: console)
    experiment.show()
    assert "epochs" in output.getvalue() and "public-team" in output.getvalue()
    assert secret not in output.getvalue()
    assert "api_key" not in experiment.model_dump()["tracking"]
    assert secret not in experiment.model_dump_json()
    snapshot = experiment.save(tmp_path / "launch.json")
    assert secret not in snapshot.read_text()
    restored = config.ExperimentConfig.load(snapshot, env_file=env_file)
    assert restored.tracking.api_key.get_secret_value() == secret


def test_edited_config_is_validated_and_smoke_recipe_is_resolved():
    experiment = config.ExperimentConfig(smoke=True)
    assert experiment.task.epochs == 2
    experiment.task.epochs = 4
    assert experiment.task.epochs == 4
    with pytest.raises(ValueError):
        experiment.task.epochs = 0


def test_experiment_instances_do_not_share_mutable_defaults():
    first = config.ExperimentConfig()
    first.task.epochs = 9
    first.budgets.append(1000)
    first.data.voc_root = "/custom/voc"
    second = config.ExperimentConfig()
    assert second.task.epochs != first.task.epochs
    assert 1000 not in second.budgets
    assert second.data.voc_root == ""


def test_acquisition_method_defaults_to_random_and_round_trips():
    from alseg.core.selection import AcquisitionMethod

    assert config.AcquisitionConfig().method is AcquisitionMethod.RANDOM
    experiment = config.ExperimentConfig(acquisition={"method": "uherding"})
    assert experiment.model_dump(mode="json")["acquisition"]["method"] == "uherding"
    restored = config.ExperimentConfig.model_validate_json(experiment.model_dump_json())
    assert restored.acquisition.method is AcquisitionMethod.UHERDING
    assert restored.acquisition.uncertainty_method == "margin"
    assert restored.acquisition.temperatures == (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    assert restored.acquisition.ece_bins == 5


def test_encoder_enum_round_trips_as_a_json_string():
    experiment = config.ExperimentConfig(
        encoder=config.EncoderConfig(name=config.EncoderName.DINO_VITB16)
    )
    assert experiment.model_dump(mode="json")["encoder"]["name"] == "dino_vitb16"
    restored = config.ExperimentConfig.model_validate_json(experiment.model_dump_json())
    assert restored.encoder.name is config.EncoderName.DINO_VITB16
