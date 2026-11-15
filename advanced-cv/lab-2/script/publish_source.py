"""Create or version a Kaggle source dataset using the official Python API."""

import hashlib
import json
import re
import uuid
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import rich_click as click
from packaging.version import Version

from alseg.cli_common import CONTEXT_SETTINGS
from alseg.config import PROJECT_ROOT, load_config


def stage_source(project_dir: Path, output_dir: Path, dataset: str, title: str) -> dict:
    """Build a reviewable source-only release; never copy a whole checkout."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[a-z0-9][a-z0-9-]{4,48}[a-z0-9]", dataset):
        raise ValueError("dataset must be owner/slug, with a 6..50 character slug")
    if not 6 <= len(title) <= 50:
        raise ValueError("Dataset title must contain 6..50 characters")
    project_dir = project_dir.resolve()
    config = load_config(project_dir / "pyproject.toml")
    version = config["project"]["version"]
    Version(version)
    paths = set()
    for pattern in (
        "alseg/**/*.py",
        "script/**/*.py",
        "tests/**/*.py",
        "configs/**/*.toml",
    ):
        paths.update(project_dir.glob(pattern))
    for name in (
        "pyproject.toml",
        "README.md",
        "SETUP_PLAN.md",
        "AGENTS.md",
        "kaggle/setup.ipynb",
    ):
        if (project_dir / name).is_file():
            paths.add(project_dir / name)
    contents = {}
    for path in sorted(paths):
        relative = path.relative_to(project_dir)
        if any(
            part.startswith(".") or part == "__pycache__" for part in relative.parts
        ):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(project_dir):
            raise ValueError(f"Source symlinks are not published: {relative}")
        if path.stem.lower() in ("credentials", "secrets", "tokens", "kaggle"):
            continue
        content = path.read_bytes()
        if path.suffix == ".ipynb":
            notebook = json.loads(content)
            notebook["metadata"] = {
                key: value
                for key, value in notebook.get("metadata", {}).items()
                if key in ("kernelspec", "language_info")
            }
            for cell in notebook["cells"]:
                cell["metadata"] = {}
                cell.pop("attachments", None)
                if cell["cell_type"] == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
            content = (json.dumps(notebook, indent=1) + "\n").encode()
        if b"jupyter-proxy." + b"kaggle.net/k/" in content:
            raise ValueError(f"Remove the Kaggle server access URL from {relative}")
        contents[relative.as_posix()] = content
    files = {
        name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    archive = output_dir / "source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, content in contents.items():
            bundle.writestr(name, content)
    release = {
        "dataset": dataset,
        "project_version": version,
        "source_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": files,
    }
    (output_dir / "source-release.json").write_text(
        json.dumps(release, indent=2) + "\n"
    )
    metadata = {
        "id": dataset,
        "title": title,
        "licenses": [{"name": "other"}],
        "description": f"ALSeg source release {version}. "
        "See source-release.json for file hashes.",
    }
    (output_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return release


def publish_release(api, folder: Path, release: dict, public: bool = False) -> str:
    """Auto-detect create/update; reject unchanged or decreasing project versions."""
    from requests.exceptions import HTTPError

    dataset = release["dataset"]
    try:
        status = api.dataset_status(dataset)
    except HTTPError as error:
        if error.response is None or error.response.status_code != 404:
            raise
        status = None
    if status is None:
        response = api.dataset_create_new(
            str(folder),
            public=public,
            quiet=True,
            convert_to_csv=False,
        )
        action = "created"
    else:
        if status != "ready":
            raise ValueError(
                f"Dataset is {status}; wait until it is ready before updating"
            )
        if public:
            raise ValueError(
                "--public is for creation only; updates preserve existing visibility"
            )
        with TemporaryDirectory(prefix="alseg-previous-release-") as directory:
            try:
                api.dataset_download_file(
                    dataset,
                    "source-release.json",
                    path=directory,
                    quiet=True,
                    unzip=True,
                )
            except HTTPError as error:
                if error.response is not None and error.response.status_code == 404:
                    raise ValueError(
                        "Existing dataset has no source-release.json; "
                        "refusing to overwrite it"
                    ) from error
                raise
            previous = json.loads((Path(directory) / "source-release.json").read_text())
        if previous["dataset"] != dataset:
            raise ValueError("Existing release belongs to another source dataset")
        if Version(release["project_version"]) <= Version(previous["project_version"]):
            raise ValueError(
                f"Increase pyproject.toml version above {previous['project_version']} "
                "before updating Kaggle"
            )
        response = api.dataset_create_version(
            str(folder),
            version_notes=f"ALSeg {release['project_version']}",
            quiet=True,
            convert_to_csv=False,
            delete_old_versions=False,
        )
        action = "updated"
    if response.error or response.status.lower() != "ok":
        raise ValueError(
            f"Kaggle rejected the release: {response.error or response.status}"
        )
    return action


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--dataset",
    required=True,
    help="Exact destination, e.g. denysmykhailov/alseg-source.",
)
@click.option(
    "--title", default="Active Learning Segmentation Source Code", show_default=True
)
@click.option(
    "--project-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=PROJECT_ROOT,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="New staging directory; defaults to a unique dist subdirectory.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Build and inspect the bundle without authentication or network calls.",
)
@click.option(
    "--public",
    is_flag=True,
    help="Create a public dataset; new datasets are private by default.",
)
def main(
    dataset: str,
    title: str,
    project_dir: Path,
    output_dir: Path | None,
    dry_run: bool,
    public: bool,
) -> None:
    """Publish a source release or update it after increasing the project version."""
    try:
        staging_root = (
            Path("/kaggle/working/alseg/source-releases")
            if Path("/kaggle/working").is_dir()
            else project_dir / "dist" / "kaggle-source"
        )
        output_dir = output_dir or staging_root / uuid.uuid4().hex[:8]
        release = stage_source(project_dir, output_dir, dataset, title)
        click.echo(
            f"Prepared {len(release['files'])} files, "
            f"project {release['project_version']}: {output_dir}"
        )
        if dry_run:
            for name in release["files"]:
                click.echo(name)
            return
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        action = publish_release(api, output_dir, release, public)
        click.echo(f"Release {action}: https://www.kaggle.com/datasets/{dataset}")
        click.echo("Upload accepted; Kaggle may still be processing the new version.")
    except (OSError, ValueError, KeyError, ImportError) as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    main()
