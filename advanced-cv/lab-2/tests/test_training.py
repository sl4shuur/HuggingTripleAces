import pytest
import torch

pytest.importorskip("lightning")
pytest.importorskip("wandb")
from alseg.config import TrackingConfig
from alseg.nn.training import fit_head
from alseg.runtime.tracking import configure_tracking, log_run_artifacts, wandb_run


def recipe(epochs=3):
    return {
        "epochs": epochs,
        "batch_size": 2,
        "learning_rate": 0.1,
        "weight_decay": 0.0,
    }


def test_head_learning_and_fresh_run_reproducibility():
    features = torch.ones(4, 3, 2, 2)
    targets = torch.zeros(4, 4, 4, dtype=torch.long)
    first = fit_head(features, targets, recipe(), 42, "cpu")
    assert first.history[-1]["train/loss"] < first.history[0]["train/loss"]
    second = fit_head(features, targets, recipe(), 42, "cpu")
    for key, value in first.head.state_dict().items():
        torch.testing.assert_close(value, second.head.state_dict()[key])


def test_all_void_training_is_rejected():
    with pytest.raises(ValueError, match="no labeled pixels"):
        fit_head(
            torch.ones(2, 3, 2, 2), torch.full((2, 4, 4), 255), recipe(), 42, "cpu"
        )


def test_tracking_loads_private_env_without_putting_key_in_config(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    secret = "test-only-key-not-a-real-credential"
    env_file.write_text("WANDB_API_KEY=" + secret + "\n")
    config = TrackingConfig(_env_file=env_file)
    assert config.wandb_entity == "public-team"
    assert configure_tracking(config) is config
    import os

    assert "WANDB_API_KEY" not in os.environ
    assert config.api_key.get_secret_value() == secret
    assert secret not in config.model_dump_json()
    assert "api_key" not in config.model_dump()
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        configure_tracking(TrackingConfig(_env_file=tmp_path / "missing.env"))


def test_offline_wandb_records_training_media_and_artifacts(tmp_path, monkeypatch):
    import wandb

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "cache"))
    config = TrackingConfig(mode="offline")
    with wandb_run(
        config, tmp_path, "offline-test", {"tracking": config.model_dump()}
    ) as logger:
        assert logger.experiment.entity == "public-team"
        task = fit_head(
            torch.ones(2, 3, 2, 2),
            torch.zeros(2, 4, 4, dtype=torch.long),
            recipe(1),
            42,
            "cpu",
            logger,
        )
        torch.save(task.head.state_dict(), tmp_path / "head.pt")
        for name in ("config.json", "metrics.json", "selection.json"):
            (tmp_path / name).write_text("{}")
        for name in ("results.csv", "training.csv"):
            (tmp_path / name).write_text("metric\n1\n")
        metrics = {"miou": 0.5, "per_class_iou": [0.5] + [None] * 20}
        preview = {
            "stem": "synthetic",
            "image": torch.zeros(3, 4, 4, dtype=torch.uint8),
            "prediction": torch.zeros(4, 4, dtype=torch.long),
            "target": torch.zeros(4, 4, dtype=torch.long),
        }
        log_run_artifacts(logger, tmp_path, metrics, [preview])
    assert wandb.run is None
    assert list(tmp_path.rglob("*.wandb"))
    assert list(tmp_path.rglob("*.png"))
