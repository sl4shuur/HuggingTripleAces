"""CPU execution without W&B uploads; all other values use shared defaults."""

from alseg.config import ExperimentConfig

config = ExperimentConfig(device="cpu", tracking={"mode": "offline"})
