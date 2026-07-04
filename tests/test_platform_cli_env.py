from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from whetstone.platform import cli_env


def test_load_env_file_returns_none_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    def record_load(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(cli_env, "load_dotenv", record_load)

    result = cli_env.load_env_file(tmp_path / "missing.env")

    assert result is None
    assert calls == []


def test_load_env_file_loads_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("KEY=value\n", encoding="utf-8")
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    monkeypatch.setattr(
        cli_env,
        "load_dotenv",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = cli_env.load_env_file(env_path)

    assert result == env_path
    assert calls == [((env_path,), {"override": False})]


def test_configure_multiprocessing_uses_fork_on_non_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    monkeypatch.setattr(cli_env.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        cli_env.mp,
        "set_start_method",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    cli_env.configure_multiprocessing()

    assert calls == [(("fork",), {"force": True})]


def test_configure_multiprocessing_uses_spawn_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    monkeypatch.setattr(cli_env.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        cli_env.mp,
        "set_start_method",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    cli_env.configure_multiprocessing()

    assert calls == [(("spawn",), {"force": True})]


def test_configure_multiprocessing_suppresses_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_env.platform, "system", lambda: "Darwin")

    def raise_runtime_error(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("already set")

    monkeypatch.setattr(cli_env.mp, "set_start_method", raise_runtime_error)

    cli_env.configure_multiprocessing()


def test_run_typer_app_wires_multiprocessing_and_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    levels: list[int] = []
    real_get_logger = logging.getLogger

    def record_multiprocessing() -> None:
        order.append("multiprocessing")

    class FakeLogger:
        def setLevel(self, level: int) -> None:
            levels.append(level)

    def fake_get_logger(
        name: str | None = None,
    ) -> logging.Logger | FakeLogger:
        if name == "dspy":
            order.append("logger:dspy")
            return FakeLogger()
        return real_get_logger(name)

    def fake_app() -> None:
        order.append("app")

    monkeypatch.setattr(
        cli_env,
        "configure_multiprocessing",
        record_multiprocessing,
    )
    monkeypatch.setattr(cli_env.logging, "getLogger", fake_get_logger)

    cli_env.run_typer_app(fake_app)

    assert order == ["multiprocessing", "logger:dspy", "app"]
    assert levels == [logging.WARNING]
