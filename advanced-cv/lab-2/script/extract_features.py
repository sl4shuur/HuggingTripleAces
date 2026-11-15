"""Cache image embeddings and dense train/val features, optionally on two GPUs."""

import importlib.metadata
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import rich_click as click
import torch

from alseg.adapters.voc import VocImages
from alseg.cli_common import CONTEXT_SETTINGS, config_option
from alseg.config import EncoderName, ExperimentConfig, get_project_version
from alseg.core.helpers import write_json
from alseg.core.pool import PoolManifest
from alseg.data.voc import find_voc_root, load_manifest
from alseg.nn.encoders import load_encoder
from alseg.nn.extraction import write_split_features


def extract_encoder(
    config: dict,
    name: str,
    device: str,
    voc_root: Path,
    manifest: PoolManifest,
    output_dir: Path,
    limit: int | None,
) -> dict:
    """Each worker owns one encoder, one device and its own output directory."""
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    encoder_config = {**config["encoder"], "name": name}
    encoder = load_encoder(encoder_config, Path(config["paths"]["cache_dir"]), device)
    directory = output_dir / name
    directory.mkdir()
    pool = VocImages(voc_root, manifest)
    encoder.encode([pool.get_image(0)])
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    splits = {}
    for split in ("train", "val"):
        click.echo(f"[{name} / {device}] Extracting {split}")
        splits[split] = write_split_features(
            VocImages(voc_root, manifest, split=split),
            encoder,
            directory / split,
            encoder_config["batch_size"],
            limit,
        )
        click.echo(f"[{name} / {device}] Saved {splits[split]['count']} {split} images")
    report = {
        "format_version": 1,
        "complete": True,
        "encoder_identity": {**encoder.describe(), "name": name},
        "encoder_config": encoder_config,
        "manifest_sha256": manifest.sha256,
        "splits": splits,
        "dtype": "float32",
        "project_version": get_project_version(),
        "extraction_seconds": time.perf_counter() - start,
        "device": device,
        "gpu": torch.cuda.get_device_name(device)
        if device.startswith("cuda")
        else None,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)
        if device.startswith("cuda")
        else None,
        "torch": str(torch.__version__),
        **{
            package: importlib.metadata.version(package)
            for package in ("transformers", "timm")
        },
    }
    # This is the completion marker: publish it only after both arrays are flushed.
    write_json(directory / "metadata.json", report)
    return report


@click.command(context_settings=CONTEXT_SETTINGS)
@config_option()
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    help="Override the configured device.",
)
@click.option(
    "--encoder",
    "encoder_names",
    multiple=True,
    type=click.Choice([name.value for name in EncoderName]),
    help="Repeat for multiple encoders; omitted means config.encoder.name.",
)
@click.option(
    "--gpus",
    type=click.IntRange(min=1, max=2),
    default=1,
    show_default=True,
    help="Two GPUs run two encoders concurrently, one per GPU.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    help="Diagnostic limit per split; omit for a reusable full cache.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="New cache directory; existing outputs are never overwritten.",
)
def main(
    config_path: Path | None,
    device: str | None,
    encoder_names: tuple[str, ...],
    gpus: int,
    limit: int | None,
    output_dir: Path | None,
) -> None:
    """Write canonical float32 image/patch arrays for official train and val."""
    try:
        experiment = ExperimentConfig.load(config_path)
        device = device or experiment.device
        config = experiment.model_dump(mode="json")
        names = list(encoder_names) or [experiment.encoder.name.value]
        if len(names) != len(set(names)):
            raise ValueError("Encoder names must be unique")
        if experiment.encoder.features_dir is not None:
            raise ValueError(
                "Extraction requires a live encoder; remove encoder.features_dir"
            )
        if len(names) > 1 and experiment.encoder.checkpoint_path is not None:
            raise ValueError(
                "A local checkpoint_path can only be used with one encoder"
            )
        if device == "cuda" and (
            not torch.cuda.is_available() or torch.cuda.device_count() < gpus
        ):
            raise ValueError(
                f"Requested {gpus} CUDA GPU(s); enable the matching Kaggle accelerator"
            )
        if gpus == 2 and (device != "cuda" or len(names) != 2):
            raise ValueError("--gpus 2 requires CUDA and exactly two --encoder choices")
        explicit = config["data"]["voc_root"]
        root = (
            Path(explicit)
            if explicit
            else find_voc_root(Path(config["data"]["dataset_root"]))
        )
        manifest = load_manifest(root)
        output_dir = (
            output_dir or experiment.paths.cache_dir / "features" / uuid.uuid4().hex
        )
        if output_dir.exists():
            raise ValueError(
                "Use a new output directory; existing artifacts are never overwritten"
            )
        output_dir.mkdir(parents=True)
        write_json(
            output_dir / "manifest.json",
            {**asdict(manifest), "sha256": manifest.sha256},
        )
        snapshot = {
            "complete": False,
            "config": config,
            "encoders": names,
            "gpus": gpus,
            "device": device,
            "limit": limit,
        }
        write_json(output_dir / "extraction.json", snapshot)
        start = time.perf_counter()
        if gpus == 2:
            # Independent models need neither DDP nor notebook-incompatible process spawning.
            with ThreadPoolExecutor(max_workers=2) as workers:
                jobs = [
                    workers.submit(
                        extract_encoder,
                        config,
                        name,
                        f"cuda:{i}",
                        root,
                        manifest,
                        output_dir,
                        limit,
                    )
                    for i, name in enumerate(names)
                ]
                reports = [job.result() for job in jobs]
        else:
            target = "cuda:0" if device == "cuda" else "cpu"
            reports = [
                extract_encoder(config, name, target, root, manifest, output_dir, limit)
                for name in names
            ]
        write_json(
            output_dir / "extraction.json",
            {
                **snapshot,
                "complete": True,
                "wall_seconds": time.perf_counter() - start,
                "reusable": all(
                    s["complete"]
                    for report in reports
                    for s in report["splits"].values()
                ),
            },
        )
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Saved feature cache: {output_dir}")
    for report in reports:
        click.echo(
            f"{report['encoder_identity']['name']}: {report['device']}, {report['extraction_seconds']:.1f}s, "
            f"patches {report['splits']['train']['patches_shape']}"
        )


if __name__ == "__main__":
    main()
