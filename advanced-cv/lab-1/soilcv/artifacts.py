"""Run files, exact identities, resumable checkpoints, and compute records."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import resource
import sys
import time

import numpy as np


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot save {type(value).__name__} as JSON.")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=_json_default) + "\n")


def identity_hash(data):
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_state_hash(model):
    """Hash actual parameter/buffer bytes, including bfloat16 and scalar buffers."""
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def package_identity():
    from . import __version__
    root = Path(__file__).parent
    sources = [(str(path.relative_to(root)), file_hash(path)) for path in sorted(root.rglob("*.py"))]
    return {"version": __version__, "source_hash": identity_hash(sources),
            "folds_hash": file_hash(root / "assets" / "sample_folds.csv")}


def save_environment(path):
    from . import __version__
    libraries = {}
    for name in ["numpy", "pandas", "Pillow", "torch", "torchvision", "timm", "transformers", "scikit-learn", "wandb"]:
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    data = {"soilcv": package_identity(), "python": platform.python_version(), "platform": platform.platform(), "libraries": libraries}
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        data["cuda_runtime"] = torch.version.cuda
    write_json(path, data)
    return data


def save_predictions(frame, path):
    from .curves import TARGET_COLUMNS, check_ids, validate_curves
    check_ids(frame)
    validate_curves(frame[list(TARGET_COLUMNS)])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_checkpoint(path, model, *, optimizer=None, scheduler=None, scaler=None, epoch=0, global_step=0,
                    config=None, run_id=None, metadata=None):
    import torch
    state = {"model": model.state_dict(), "epoch": epoch, "global_step": global_step,
             "config": config.to_dict() if hasattr(config, "to_dict") else config,
             "run_id": run_id, "metadata": dict(metadata or {}),
             "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    if config is not None:
        from .config import _check_config
        _check_config(state["config"])
    for name, obj in [("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)]:
        state[name] = obj.state_dict() if obj is not None else None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, *, optimizer=None, scheduler=None, scaler=None, expected_metadata=None, restore_rng=True):
    """Load this package's trusted training checkpoint, not an unknown pickle."""
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in (expected_metadata or {}).items():
        if state["metadata"].get(key) != value:
            raise ValueError(f"Checkpoint identity differs: {key}.")
    for name, obj in [("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)]:
        if obj is not None and state.get(name) is None:
            raise ValueError(f"Checkpoint has no {name} state.")
    model.load_state_dict(state["model"])
    for name, obj in [("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)]:
        if obj is not None:
            obj.load_state_dict(state[name])
    if restore_rng:
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return state


class ComputeMeter:
    """Time a stage. CPU RSS is the process lifetime peak, not a resettable run peak."""
    def __init__(self, device="cpu", *, output_path=None):
        self.output_path = output_path
        self.device = str(device)
        self.result = {}

    def __enter__(self):
        self.is_cuda = self.device.startswith("cuda")
        if self.is_cuda:
            import torch
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_cuda:
            import torch
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - self.started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.result = {"wall_seconds": seconds, "gpu_hours": seconds / 3600 if self.is_cuda else 0.0,
                       "cpu_process_peak_mb": rss / (1024**2 if sys.platform == "darwin" else 1024),
                       "device": self.device, "completed": exc_type is None}
        if self.is_cuda:
            self.result.update(gpu_name=torch.cuda.get_device_name(self.device),
                               peak_gpu_memory_mb=torch.cuda.max_memory_allocated(self.device) / 1024**2)
        if self.output_path is not None:
            write_json(self.output_path, self.result)
