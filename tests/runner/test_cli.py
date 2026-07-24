from __future__ import annotations

from typing import cast

from typer.testing import CliRunner

import whetstone.runner.cli as cli
from whetstone.runner.cell import CellConfig, CellOutcome


class _Record:
    def model_dump_json(self, **kwargs) -> str:
        del kwargs
        return '{"status":"ok"}'


class _Outcome:
    record = _Record()


def test_cell_command_selects_dry_factory_seam(monkeypatch) -> None:
    config = cast(CellConfig, object())
    calls: list[CellConfig] = []
    monkeypatch.setattr(cli, "_load_factory", lambda path: lambda: config)
    monkeypatch.setattr(
        cli,
        "run_dry_cell",
        lambda value: calls.append(value) or cast(CellOutcome, _Outcome()),
    )
    monkeypatch.setattr(
        cli,
        "run_cell",
        lambda value: (_ for _ in ()).throw(
            AssertionError(f"live cell selected for {value!r}")
        ),
    )

    result = CliRunner().invoke(
        cli.app,
        ["cell", "--factory", "tests.factory:cell", "--dry"],
    )

    assert result.exit_code == 0
    assert calls == [config]
    assert '"status":"ok"' in result.stdout


def test_status_validates_empty_ledger(tmp_path) -> None:
    result = CliRunner().invoke(
        cli.app,
        ["status", "--root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "[]"


def test_factory_path_is_explicit() -> None:
    result = CliRunner().invoke(
        cli.app,
        ["cell", "--factory", "not-an-import-path"],
    )

    assert result.exit_code == 2
    assert "module:callable" in result.output
