"""Train a fresh segmentation head and report official-val metrics through W&B."""

import json
import uuid
from pathlib import Path

import rich_click as click

from alseg.cli_common import CONTEXT_SETTINGS, config_option
from alseg.config import ExperimentConfig


@click.command(context_settings=CONTEXT_SETTINGS)
@config_option()
@click.option(
    "--budget",
    type=click.IntRange(min=1),
    help="Override the configured annotation budget.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    help="Override the configured training device.",
)
@click.option(
    "--wandb-mode",
    type=click.Choice(["online", "offline", "disabled"]),
    help="Override the tracking mode, keeping entity=public-team.",
)
@click.option(
    "--env-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Private .env outside the source dataset; default is PROJECT_ROOT/.env.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="New run directory; defaults to a unique configured runs subdirectory.",
)
@click.option(
    "--smoke",
    is_flag=True,
    default=None,
    help="Use two epochs/eight val images; excluded from benchmark results.",
)
def main(
    config_path: Path | None,
    budget: int | None,
    device: str | None,
    wandb_mode: str | None,
    env_file: Path | None,
    output_dir: Path | None,
    smoke: bool | None,
) -> None:
    """Train one Random-selected budget with fixed epochs and final evaluation."""
    from alseg.pipeline import run_training

    try:
        overrides = {
            name: value
            for name, value in {
                "budget": budget,
                "device": device,
                "run_dir": output_dir,
                "smoke": smoke,
            }.items()
            if value is not None
        }
        config = ExperimentConfig.load(config_path, env_file=env_file)
        for name, value in overrides.items():
            setattr(config, name, value)
        if wandb_mode:
            config.tracking = config.tracking.model_copy(update={"mode": wandb_mode})
        run_dir = config.run_dir or (
            config.paths.runs_dir / f"random-{config.budget}-{uuid.uuid4().hex[:8]}"
        )
        resolved = config.model_dump(mode="json")
        resolved["run_dir"] = str(run_dir)
        result = run_training(
            resolved,
            run_dir,
            config.budget,
            config.device,
            config.tracking,
            smoke=config.smoke,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result, indent=2))
    click.echo(f"Saved training run: {run_dir}")


if __name__ == "__main__":
    main()
