"""Read-only environment and official segmentation split checks."""

import json
import platform
import tomllib
from pathlib import Path

import rich_click as click
import torch

from alseg.data import find_voc_root, validate_voc_root
from cli_common import CONTEXT_SETTINGS, config_option


@click.command(context_settings=CONTEXT_SETTINGS)
@config_option()
@click.option("--check-data", is_flag=True, help="Validate official train/val files.")
def main(config_path: Path, check_data: bool) -> None:
    """Report the runtime and optionally check the VOC segmentation mount."""
    try:
        with config_path.open("rb") as file:
            settings = tomllib.load(file)
        report = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "dataset_root": settings["data"]["dataset_root"],
            "data_checked": False,
        }
        if check_data:
            explicit = settings["data"]["voc_root"]
            root = (
                Path(explicit)
                if explicit
                else find_voc_root(Path(settings["data"]["dataset_root"]))
            )
            splits = validate_voc_root(root)
            report.update(
                voc_root=str(root),
                split_sizes={name: len(ids) for name, ids in splits.items()},
                data_checked=True,
            )
    except (OSError, ValueError, KeyError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
