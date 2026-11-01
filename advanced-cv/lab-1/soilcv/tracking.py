"""Online W&B. Default entity is public-team. Nothing starts on import."""

import os
from pathlib import Path

from .artifacts import save_environment
from .config import ExperimentConfig, save_config


def start_run(
    config,
    *,
    name=None,
    run_id=None,
    resume=False,
    api_key=None,
    secret_name="WANDB_API_KEY",
    job_type=None,
):
    """Read an attached Kaggle secret (or env key) only when a run is requested."""
    import wandb

    if not isinstance(config, ExperimentConfig):
        raise TypeError("Use ExperimentConfig for a tracked run.")
    data = config.to_dict()
    if wandb.run is not None:
        raise RuntimeError("Finish the active W&B run before starting another fold.")
    if resume and not run_id:
        raise ValueError("Resume needs the saved W&B run ID and a training checkpoint.")
    key = api_key or os.environ.get(secret_name)
    if not key:
        try:
            from kaggle_secrets import UserSecretsClient

            key = UserSecretsClient().get_secret(secret_name)
        except Exception as error:
            raise RuntimeError(
                f"Attach Kaggle Secret {secret_name}, or set that environment variable."
            ) from error
    if not wandb.login(key=key, verify=True):
        raise RuntimeError(
            "W&B online login failed. Check the key and Internet setting."
        )
    save_config(config, config.output_dir / "config.json")
    data["runtime"] = save_environment(config.output_dir / "environment.json")
    run = wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        group=config.experiment_id,
        job_type=job_type or config.stage,
        name=name
        or (
            f"{config.stage}-fold-{config.fold}"
            if config.fold is not None
            else "cv-summary"
        ),
        id=run_id,
        resume="must" if resume else "never",
        mode="online",
        config=data,
        dir=str(config.output_dir),
    )
    if run.settings.mode != "online":
        run.finish(exit_code=1)
        raise RuntimeError("Expected an online W&B run. Offline mode is not used here.")
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("val/emd", summary="min")
    run.define_metric("val/camera_disagreement", summary="min")
    return run


def log_epoch(run, epoch, train_metrics, validation_metrics=None):
    data = {"epoch": int(epoch)}
    data.update({f"train/{key}": value for key, value in train_metrics.items()})
    data.update(
        {f"val/{key}": value for key, value in (validation_metrics or {}).items()}
    )
    run.log(data)


def log_evaluation(run, metrics, *, prefix="cv", tables=None):
    import wandb

    for key, value in metrics.items():
        run.summary[f"{prefix}/{key}"] = value
    if tables:
        run.log(
            {
                f"{prefix}/{name}": wandb.Table(dataframe=table)
                for name, table in tables.items()
            }
        )


def log_artifact(run, path, *, name, artifact_type="results", metadata=None):
    import wandb

    path = Path(path)
    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
    if path.is_dir():
        artifact.add_dir(str(path))
    else:
        artifact.add_file(str(path))
    return run.log_artifact(artifact)


def finish_run(run, *, success=True):
    run.finish(exit_code=0 if success else 1)
