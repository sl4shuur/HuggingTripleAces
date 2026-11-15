"""Extract frozen DINOv2 image embeddings without accessing pool labels."""

import importlib.metadata
import json
import time
import tomllib
import uuid
from pathlib import Path

import rich_click as click
import torch

from alseg.adapters.dinov2 import DinoV2Encoder
from alseg.adapters.voc import VocImages
from alseg.cli_common import CONTEXT_SETTINGS, PROJECT_ROOT, config_option
from alseg.data import find_voc_root, load_manifest
from alseg.features import iter_features


@click.command(context_settings=CONTEXT_SETTINGS)
@config_option()
@click.option(
    "--device", type=click.Choice(["cpu", "cuda"]), default="cuda", show_default=True
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    help="Smoke test: first N pool images; omitted means full pool.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="New artifact directory; defaults to a unique cache subdirectory.",
)
def main(
    config_path: Path, device: str, limit: int | None, output_dir: Path | None
) -> None:
    """Save ID-aligned embeddings, preprocessing identity and runtime profiling."""
    try:
        with config_path.open("rb") as file:
            config = tomllib.load(file)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable; enable a Kaggle GPU or use --device cpu"
            )
        explicit = config["data"]["voc_root"]
        root = (
            Path(explicit)
            if explicit
            else find_voc_root(Path(config["data"]["dataset_root"]))
        )
        manifest = load_manifest(root)
        pool = VocImages(root, manifest)
        count = len(pool) if limit is None else limit
        if count > len(pool):
            raise ValueError("limit exceeds the train pool size")
        ids = list(range(count))
        output_dir = (
            output_dir
            or Path(config["paths"]["cache_dir"]) / "features" / uuid.uuid4().hex
        )
        if output_dir.exists():
            raise ValueError(
                "Use a new output directory; existing artifacts are never overwritten"
            )
        encoder_config = config["encoder"]
        encoder = DinoV2Encoder.from_pretrained(
            encoder_config["model_id"],
            encoder_config["revision"],
            encoder_config["image_size"],
            device=device,
            cache_dir=Path(config["paths"]["cache_dir"]) / "huggingface",
        )
        warmup = encoder.encode([pool.get_image(ids[0])])
        patch_shape = list(warmup.patch_features.shape[1:])
        del warmup
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        chunks = [
            batch.embeddings
            for _, batch in iter_features(
                pool, encoder, ids, encoder_config["batch_size"]
            )
        ]
        if device == "cuda":
            torch.cuda.synchronize()
        embeddings = torch.cat(chunks)
        report = {
            **encoder.describe(),
            "ids": ids,
            "pool_ids": manifest.pool_ids,
            "complete_pool": count == len(pool),
            "embeddings_shape": list(embeddings.shape),
            "patch_shape": patch_shape,
            "batch_size": encoder_config["batch_size"],
            "extraction_seconds": time.perf_counter() - start,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated()
            if device == "cuda"
            else None,
            "device": device,
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
        }
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
            report["project_version"] = tomllib.load(file)["project"]["version"]
        output_dir.mkdir(parents=True)
        torch.save({"ids": ids, "embeddings": embeddings}, output_dir / "embeddings.pt")
        (output_dir / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    except (OSError, ValueError, KeyError, ImportError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Saved {count} embeddings to {output_dir}")
    click.echo(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("ids", "pool_ids")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
