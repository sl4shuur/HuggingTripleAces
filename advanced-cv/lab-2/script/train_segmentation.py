"""Train a fresh segmentation head and report official-val metrics through W&B."""

import json
import uuid
from pathlib import Path

import rich_click as click

from alseg.cli_common import CONTEXT_SETTINGS, config_option
from alseg.config import PROJECT_ROOT, TrackingConfig, load_config


@click.command(context_settings=CONTEXT_SETTINGS)
@config_option()
@click.option("--budget", type=click.IntRange(min=1), default=21, show_default=True)
@click.option(
    "--device", type=click.Choice(["cpu", "cuda"]), default="cuda", show_default=True
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
    help="Use two epochs/eight val images; excluded from benchmark results.",
)
def main(
    config_path: Path,
    budget: int,
    device: str,
    wandb_mode: str | None,
    env_file: Path | None,
    output_dir: Path | None,
    smoke: bool,
) -> None:
    """Train one Random-selected budget with fixed epochs and final evaluation."""
    from alseg.pipeline import run_training

    try:
        config = load_config(config_path)
        tracking = TrackingConfig(
            _env_file=env_file or PROJECT_ROOT / ".env", **config["tracking"]
        )
        if wandb_mode:
            tracking = tracking.model_copy(update={"mode": wandb_mode})
        if smoke:
            config["task"]["epochs"] = 2
        output_dir = (
            output_dir
            or Path(config["paths"]["runs_dir"])
            / f"random-{budget}-{uuid.uuid4().hex[:8]}"
        )
        result = run_training(config, output_dir, budget, device, tracking, smoke=smoke)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result, indent=2))
    click.echo(f"Saved training run: {output_dir}")


if __name__ == "__main__":
    main()
