"""Build a small ZIP to upload through Kaggle's New Dataset UI."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from soilcv import __version__
from soilcv.artifacts import package_identity
from soilcv.config import ExperimentConfig


def build_bundle(destination):
    project = Path(__file__).resolve().parent
    root = project / "soilcv"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    identity = package_identity()
    release = {**identity, "default_wandb_entity": ExperimentConfig("release").wandb_entity,
               "default_wandb_project": ExperimentConfig("release").wandb_project}
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".md", ".csv", ".toml", ".lock"}:
                archive.write(path, "soilcv/" + str(path.relative_to(root)))
        archive.write(project / "pyproject.toml", "pyproject.toml")
        archive.write(project / "HYPOTHESES.md", "HYPOTHESES.md")
        archive.writestr("release.json", json.dumps(release, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / f"soil-cv-utils.zip",
        help="Output ZIP path (default: next to this script)",
    )
    print(build_bundle(parser.parse_args().destination))
