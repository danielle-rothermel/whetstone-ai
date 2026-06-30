"""Legacy-adjacent script runtime setup.

These helpers are used by the v0 Typer CLIs and are intentionally forbidden
from pure graph/platform modules. New platform entrypoints should keep runtime
setup at their CLI boundary.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import platform
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

__all__ = ["configure_multiprocessing", "load_env_file", "run_typer_app"]


class TyperApp(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def load_env_file(env_file: str | Path = DEFAULT_ENV_FILE) -> Path | None:
    """Load package-local environment variables if the file exists."""
    env_path = Path(env_file)
    if not env_path.exists():
        return None
    load_dotenv(env_path, override=False)
    return env_path


def configure_multiprocessing() -> None:
    """Configure multiprocessing consistently for script entrypoints."""
    start_method = "fork" if platform.system() == "Linux" else "spawn"
    with contextlib.suppress(RuntimeError):
        mp.set_start_method(start_method, force=True)


def run_typer_app(app: TyperApp) -> None:
    configure_multiprocessing()
    logging.getLogger("dspy").setLevel(logging.WARNING)
    app()
