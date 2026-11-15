"""Experiment defaults, validation and JSON snapshots shared by all entry points."""

import tomllib
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table


def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "alseg").is_dir():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _project_root()


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def show(self) -> None:
        """Show resolved values; secret fields are excluded by their models."""
        table = Table("Setting", "Value", title=type(self).__name__)
        for name, value in self.model_dump(mode="json").items():
            table.add_row(name, Pretty(value, expand_all=True))
        Console().print(table)


class TrackingConfig(BaseSettings, ConfigModel):
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


class DataConfig(ConfigModel):
    dataset_root: str = "/kaggle/input/datasets/gopalbhattrai/pascal-voc-2012-dataset"
    voc_root: str = ""  # Empty means discover official train/val under dataset_root.
    ignore_index: int = 255
    num_classes: int = 21


class PathsConfig(ConfigModel):
    cache_dir: Path = Path("/kaggle/working/alseg/cache")
    runs_dir: Path = Path("/kaggle/working/alseg/runs")


class ProtocolConfig(ConfigModel):
    cold_start: str = "random"
    selection_seed_offset: int = Field(default=0, ge=0)
    training_seed_offset: int = Field(default=10000, ge=0)
    calibration_seed_offset: int = Field(default=20000, ge=0)
    calibration_fraction: float = Field(default=0.2, ge=0, lt=1, allow_inf_nan=False)
    calibration_min_images: int = Field(default=4, ge=0)


class AcquisitionConfig(ConfigModel):
    tie_break: str = "smallest_pool_id"
    maxherding_sigma_rule: str = "median_pairwise_distance"
    uherding_sigma_rule: str = "minimum_labeled_pairwise_distance"


class TaskConfig(ConfigModel):
    head: Literal["linear_1x1"] = "linear_1x1"
    retrain_from_scratch: Literal[True] = True
    epochs: int = Field(default=20, gt=0)
    batch_size: int = Field(default=4, gt=0)
    learning_rate: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class EncoderName(StrEnum):
    DINOV2 = "dinov2"
    DINO_VITB16 = "dino_vitb16"
    IBOT_VITB16 = "ibot_vitb16"


class EncoderConfig(ConfigModel):
    name: EncoderName = EncoderName.DINO_VITB16
    model_id: str = "facebook/dinov2-small"
    revision: str = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    image_size: int = Field(default=448, gt=0)
    batch_size: int = Field(default=2, gt=0)
    checkpoint_path: Path | None = None

    @model_validator(mode="after")
    def validate_checkpoint(self) -> Self:
        if self.name != EncoderName.DINOV2 and self.image_size % 16:
            raise ValueError("DINO/iBOT image_size must be a multiple of patch_size=16")
        if self.name == EncoderName.DINOV2 and self.checkpoint_path is not None:
            raise ValueError("checkpoint_path is supported only for DINO/iBOT ViT-B/16")
        return self


class ExperimentConfig(ConfigModel):
    seed: int = Field(default=42, ge=0)
    budgets: list[int] = Field(default_factory=lambda: [21, 42, 84, 168, 336, 672])
    data: DataConfig = Field(default_factory=DataConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    budget: int = Field(default=21, gt=0)
    device: Literal["cpu", "cuda"] = "cuda"
    run_dir: Path | None = None
    smoke: bool = False

    @model_validator(mode="after")
    def validate_budgets(self) -> Self:
        if (
            not self.budgets
            or any(b <= 0 for b in self.budgets)
            or any(a >= b for a, b in pairwise(self.budgets))
        ):
            raise ValueError(
                "budgets must be nonempty, positive and strictly increasing"
            )
        if any(
            self.seed + offset + len(self.budgets) - 1 >= 2**63
            for offset in (
                self.protocol.selection_seed_offset,
                self.protocol.training_seed_offset,
                self.protocol.calibration_seed_offset,
            )
        ):
            raise ValueError("Round seeds must be below 2**63")
        return self

    @model_validator(mode="after")
    def resolve_smoke_recipe(self) -> Self:
        if self.smoke:
            self.task.epochs = 2
        return self

    def save(self, path: Path | str) -> Path:
        """Save resolved values for a script invocation, excluding credentials."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(
        cls, path: Path | str | None = None, *, env_file: Path | None = None
    ) -> Self:
        config = (
            cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
            if path
            else cls()
        )
        if env_file is not None:
            config.tracking = TrackingConfig(
                _env_file=env_file, **config.tracking.model_dump()
            )
        return config


def get_project_version(project_dir: Path = PROJECT_ROOT) -> str:
    """Read package metadata; experiment defaults live in the classes above."""
    with (project_dir / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]["version"]
