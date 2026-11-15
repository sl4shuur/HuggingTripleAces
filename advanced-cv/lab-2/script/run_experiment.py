"""Run cumulative Random budgets from one JSON experiment configuration."""

import json
import uuid
from pathlib import Path

import rich_click as click

from alseg.cli_common import CONTEXT_SETTINGS, config_option
from alseg.config import ExperimentConfig


@click.command(context_settings=CONTEXT_SETTINGS)
@config_option()
@click.option("--device", type=click.Choice(["cpu", "cuda"]))
@click.option("--wandb-mode", type=click.Choice(["online", "offline", "disabled"]))
@click.option(
    "--env-file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--smoke",
    is_flag=True,
    default=None,
    help="Two epochs/eight val images; budgets stay configured.",
)
def main(
    config_path: Path | None,
    device: str | None,
    wandb_mode: str | None,
    env_file: Path | None,
    output_dir: Path | None,
    smoke: bool | None,
) -> None:
    """Acquire only new IDs, hold out calibration labels and fit a fresh head per budget."""
    from alseg.pipeline import run_experiment

    try:
        config = ExperimentConfig.load(config_path, env_file=env_file)
        for name, value in {
            "device": device,
            "run_dir": output_dir,
            "smoke": smoke,
        }.items():
            if value is not None:
                setattr(config, name, value)
        if wandb_mode:
            config.tracking = config.tracking.model_copy(update={"mode": wandb_mode})
        run_dir = (
            config.run_dir
            or config.paths.runs_dir
            / f"{config.encoder.name}-random-{uuid.uuid4().hex[:8]}"
        )
        resolved = config.model_dump(mode="json")
        resolved["run_dir"] = str(run_dir)
        results = run_experiment(
            resolved, run_dir, config.device, config.tracking, smoke=config.smoke
        )
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(results, indent=2))
    click.echo(f"Saved experiment: {run_dir}")


if __name__ == "__main__":
    main()
