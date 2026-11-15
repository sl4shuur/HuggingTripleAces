"""Notebook platform detection and secret access."""

import os
import sys
from pathlib import Path
from typing import Literal

Environment = Literal["local", "kaggle", "colab"]


def detect_environment(verbose: bool = False) -> Environment:
    environment: Environment = "local"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").exists():
        environment = "kaggle"
    elif "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules:
        environment = "colab"
    if verbose:
        print(f"Detected environment: {environment}")
    return environment


def get_secret(name: str, *, environment: Environment | None = None) -> str | None:
    """Read notebook secrets, or a process environment variable locally."""
    environment = environment or detect_environment()
    match environment:
        case "kaggle":
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret(name)
        case "colab":
            from google.colab import userdata

            return userdata.get(name)
        case "local":
            return os.environ.get(name)
    raise ValueError(f"Unknown environment: {environment}")
