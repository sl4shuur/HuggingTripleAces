"""Small experiment configs and repeatable seeds."""

from dataclasses import asdict, dataclass, field
import json
import random
from pathlib import Path

import numpy as np


@dataclass
class ExperimentConfig:
    experiment_id: str
    stage: str = "supervised"
    model: str = ""
    seed: int = 42
    fold: int | None = None
    wandb_entity: str = "public-team"
    wandb_project: str = "soil-grain-size"
    output_root: str = "/kaggle/working/runs"
    hypothesis_id: str = ""
    versions: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.experiment_id or Path(self.experiment_id).name != self.experiment_id or self.experiment_id in {".", ".."}:
            raise ValueError("experiment_id must be a simple folder name.")
        if self.fold is not None and (not isinstance(self.fold, int) or self.fold < 0):
            raise ValueError("fold must be a non-negative integer or None.")
        if not self.wandb_entity or not self.wandb_project:
            raise ValueError("Set a W&B entity and project.")

    @property
    def output_dir(self):
        name = "summary" if self.fold is None else f"fold_{self.fold}"
        return Path(self.output_root) / self.experiment_id / name

    def to_dict(self):
        data = asdict(self)
        _check_config(data)
        return data


def _check_config(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).lower() in {"api_key", "wandb_api_key", "password", "access_token"}:
                raise ValueError("Keep secrets outside the experiment config.")
            _check_config(value)
    elif isinstance(data, (list, tuple)):
        for value in data:
            _check_config(value)


def save_config(config, path):
    path = Path(path)
    data = config.to_dict() if isinstance(config, ExperimentConfig) else config
    _check_config(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load_config(path):
    return ExperimentConfig(**json.loads(Path(path).read_text()))


def seed_everything(seed=42, *, deterministic=False):
    """Call before building models/loaders. Torch is optional for NumPy tools."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id):
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
