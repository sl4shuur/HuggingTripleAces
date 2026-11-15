"""Project location and TOML loading shared by entry points."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "alseg").is_dir():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _project_root()


class TrackingConfig(BaseSettings):
    """Tracking settings from .env/environment, with no project-specific prefix."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="",
        env_ignore_empty=True,
        extra="ignore",
        populate_by_name=True,
        frozen=True,
    )

    wandb_entity: str = "public-team"
    wandb_project: str = "active-learning-segmentation"
    mode: Literal["online", "offline", "disabled"] = "online"
    api_key: SecretStr | None = Field(
        default=None, validation_alias="WANDB_API_KEY", exclude=True, repr=False
    )


def load_config(path: Path) -> dict:
    """Read a TOML configuration without depending on CLI presentation."""
    with path.open("rb") as file:
        return tomllib.load(file)
