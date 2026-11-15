"""W&B lifecycle and artifacts; authentication stays out of run configuration dumps."""

import os
from contextlib import contextmanager
from pathlib import Path

from pydantic import SecretStr

from alseg.config import TrackingConfig
from alseg.core.predictions import CLASS_NAMES
from alseg.runtime.environment import get_secret


def configure_tracking(config: TrackingConfig) -> TrackingConfig:
    """Fail before expensive work if online tracking has no API key."""
    if config.mode != "online" or config.api_key:
        return config
    api_key = get_secret("WANDB_API_KEY")
    if not api_key:
        raise ValueError(
            "Set WANDB_API_KEY in .env or Kaggle Add-ons > Secrets, "
            "or use --wandb-mode offline"
        )
    return config.model_copy(update={"api_key": SecretStr(api_key)})


@contextmanager
def wandb_run(config: TrackingConfig, run_dir: Path, name: str, run_config: dict):
    if config.mode == "disabled":
        yield False
        return
    import wandb
    from lightning.pytorch.loggers import WandbLogger

    if wandb.run is not None:
        raise ValueError(
            "A W&B run is already active; finish it before starting a new training run"
        )
    previous_cache = os.environ.get("WANDB_CACHE_DIR")
    os.environ.setdefault("WANDB_CACHE_DIR", str(run_dir / "wandb-cache"))
    run = None
    exit_code = 0
    try:
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=name,
            dir=str(run_dir),
            mode=config.mode,
            config=run_config,
            settings=wandb.Settings(
                disable_git=True,
                disable_code=True,
                api_key=config.api_key.get_secret_value() if config.api_key else None,
            ),
        )
        yield WandbLogger(experiment=run, log_model=False)
    except BaseException:
        exit_code = 1
        raise
    finally:
        if run is not None:
            run.finish(exit_code=exit_code)
        if previous_cache is None:
            os.environ.pop("WANDB_CACHE_DIR", None)


def log_run_artifacts(
    logger, run_dir: Path, metrics: dict, previews: list[dict]
) -> None:
    if not logger:
        return
    import wandb

    run = logger.experiment
    scalars = {
        f"val/{key}": value for key, value in metrics.items() if key != "per_class_iou"
    }
    logger.log_metrics(scalars)
    run.summary.update(scalars)
    run.log(
        {
            "val/per_class_iou": wandb.Table(
                columns=["class_id", "class_name", "iou"],
                data=[
                    [i, CLASS_NAMES[i], value]
                    for i, value in enumerate(metrics["per_class_iou"])
                ],
            )
        }
    )
    images = []
    for preview in previews:
        labels = dict(enumerate(CLASS_NAMES)) | {255: "void"}
        images.append(
            wandb.Image(
                preview["image"].permute(1, 2, 0).numpy(),
                caption=preview["stem"],
                masks={
                    "predictions": {
                        "mask_data": preview["prediction"].numpy(),
                        "class_labels": labels,
                    },
                    "ground_truth": {
                        "mask_data": preview["target"].numpy(),
                        "class_labels": labels,
                    },
                },
            )
        )
    if images:
        run.log({"val/predictions": images})
    artifact = wandb.Artifact(f"head-{run.id}", type="model")
    for name in (
        "head.pt",
        "config.json",
        "results.csv",
        "metrics.json",
        "selection.json",
        "training.csv",
    ):
        artifact.add_file(str(run_dir / name), name=name)
    run.log_artifact(artifact)
