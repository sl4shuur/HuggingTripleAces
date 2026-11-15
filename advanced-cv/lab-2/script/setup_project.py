"""Read-only environment and official segmentation split checks."""

import json
import platform
from pathlib import Path

import rich_click as click
import torch

from alseg.cli_common import CONTEXT_SETTINGS, config_option
from alseg.config import load_config
from alseg.data import find_voc_root, inspect_voc_samples, load_manifest


@click.command(context_settings=CONTEXT_SETTINGS)
@config_option()
@click.option("--check-data", is_flag=True, help="Validate official train/val files.")
@click.option(
    "--inspect-data",
    is_flag=True,
    help="Also decode the first train/val RGB and mask pairs; implies --check-data.",
)
def main(config_path: Path, check_data: bool, inspect_data: bool) -> None:
    """Report the runtime and optionally check the VOC segmentation mount."""
    try:
        config = load_config(config_path)
        report = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "dataset_root": config["data"]["dataset_root"],
            "data_checked": False,
        }
        if check_data or inspect_data:
            explicit = config["data"]["voc_root"]
            root = (
                Path(explicit)
                if explicit
                else find_voc_root(Path(config["data"]["dataset_root"]))
            )
            manifest = load_manifest(root)
            report.update(
                voc_root=str(root),
                split_sizes={
                    "train": len(manifest.pool_ids),
                    "val": len(manifest.val_ids),
                },
                data_checked=True,
            )
            if inspect_data:
                report["samples"] = inspect_voc_samples(root, manifest)
    except (OSError, ValueError, KeyError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
