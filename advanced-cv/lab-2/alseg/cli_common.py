"""Shared CLI presentation and options; model commands add their own decorators."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import rich_click as click

from alseg.config import PROJECT_ROOT

F = TypeVar("F", bound=Callable[..., Any])

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "rich_help_config": {
        "text_markup": "rich",
        "show_arguments": True,
        "group_arguments_options": True,
        "style_option": "bold cyan",
        "style_command": "bold cyan",
        "style_usage": "bold yellow",
        "style_errors_panel_border": "red",
        "style_errors_suggestion": "magenta italic",
        "errors_suggestion": (
            "\nTry running the [bold cyan]'--help'[/] flag for more information."
        ),
        "errors_epilogue": "To find out more, visit our documentation.",
    },
}


def config_option(*, required: bool = False) -> Callable[[F], F]:
    """Share the TOML config option across project entry points."""
    return click.option(
        "--config",
        "config_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        required=required,
        default=None if required else PROJECT_ROOT / "configs" / "base.toml",
        show_default=True,
        help="Experiment configuration TOML.",
    )


def fit_options(function: F) -> F:
    """Apply below @click.command, alongside any model-specific @click.option."""

    # TODO: add dataset and other options

    return config_option(required=True)(function)


def predict_options(function: F) -> F:
    function = click.option(
        "--accelerator",
        type=click.Choice(["auto", "cpu", "gpu"]),
        default="auto",
        show_default=True,
    )(function)

    # TODO: split and other

    return click.option(
        "--output-dir",
        type=click.Path(path_type=Path, file_okay=False),
        help="Result folder; defaults to <run>/<split>.",
    )(function)
