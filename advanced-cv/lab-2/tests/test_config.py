import os

import pytest

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


def test_load_config_is_independent_of_cwd_and_reports_invalid_toml(
    tmp_path, monkeypatch
):
    file = tmp_path / "config.toml"
    file.write_text("[encoder]\nimage_size = 448\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert config.load_config(file) == {"encoder": {"image_size": 448}}
    file.write_text("[encoder\n")
    with pytest.raises(ValueError):
        config.load_config(file)


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
