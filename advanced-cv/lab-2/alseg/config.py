"""Project location and TOML loading shared by entry points."""

import tomllib
from pathlib import Path


def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "alseg").is_dir():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _project_root()


def load_config(path: Path) -> dict:
    """Read a TOML configuration without depending on CLI presentation."""
    with path.open("rb") as file:
        return tomllib.load(file)
