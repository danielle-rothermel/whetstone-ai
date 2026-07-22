"""The fake-transport dry-run cell path, driven as a program (no live call).

``--dry-run-fake`` runs a full cell against scripted fake transports so the CLI
plumbing (build_env_experiment -> baseline/ceiling/best official evals ->
optimizer internal search -> delta + CI -> ledger append) can be proven as a
program, not only under pytest. No test here passes ``--live``; every transport
is a scripted fake, so no live paid LLM call is ever made.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whetstone.runner.cli import main
from whetstone.runner.dryrun import run_dry_cell
from whetstone.runner.execution_mode import ExecutionMode
from whetstone.runner.ledger import Ledger


def test_dry_cell_eval_exact_match_env_baseline_equals_best(
    tmp_path: Path,
) -> None:
    outcome = run_dry_cell(
        env="c11",
        optimizer="eval",
        root=tmp_path,
        execution_mode=ExecutionMode.IN_PROCESS,
    )
    r = outcome.record
    # eval is the identity optimizer: no proposal steps, naive == best.
    assert r.optimizer_steps == 0
    assert r.baseline_official == pytest.approx(1.0)
    assert r.best_official == pytest.approx(1.0)
    assert r.delta == pytest.approx(0.0)
    assert r.canonical is False
    # No real spend: identical before/after credits snapshot.
    assert r.spend_usd == pytest.approx(0.0)
    # The ledger line landed on disk.
    assert len(Ledger(root=tmp_path).load()) == 1


def test_dry_cell_copro_exact_match_env_improves(tmp_path: Path) -> None:
    outcome = run_dry_cell(
        env="c11",
        optimizer="copro",
        root=tmp_path,
        execution_mode=ExecutionMode.IN_PROCESS,
    )
    r = outcome.record
    assert r.status == "improved"
    assert r.baseline_official == pytest.approx(0.0)
    assert r.best_official == pytest.approx(1.0)
    assert r.delta == pytest.approx(1.0)
    assert r.ci95 is not None
    assert r.internal_evals_count >= 1
    assert r.optimizer_steps >= 1


def test_dry_cell_runs_every_env_without_crashing(tmp_path: Path) -> None:
    # The plumbing must execute end-to-end for all five envs (c22 is a
    # constraint env where echoing gold cannot satisfy constraints, so it is
    # an honest no-improvement -- but it must still produce a ledger line).
    for env in ("c22", "c11", "c19", "c18", "c23"):
        outcome = run_dry_cell(
            env=env,
            optimizer="copro",
            root=tmp_path / env,
            execution_mode=ExecutionMode.IN_PROCESS,
        )
        assert outcome.record.status in ("improved", "no-improvement")
        assert outcome.record.baseline_official is not None
        assert outcome.record.best_official is not None


def test_cli_dry_run_fake_needs_no_live_flag(tmp_path: Path) -> None:
    # Crucially: this does NOT pass --live and does NOT raise SystemExit.
    code = main(
        [
            "--root",
            str(tmp_path),
            "--execution-mode",
            "in-process",
            "cell",
            "--optimizer",
            "eval",
            "--env",
            "c11",
            "--dry-run-fake",
        ]
    )
    assert code == 0
    assert len(Ledger(root=tmp_path).load()) == 1


def test_cli_dry_run_fake_resumes_completed_cell(tmp_path: Path) -> None:
    argv = [
        "--root",
        str(tmp_path),
        "--execution-mode",
        "in-process",
        "cell",
        "--optimizer",
        "copro",
        "--env",
        "c11",
        "--dry-run-fake",
    ]
    assert main(argv) == 0
    # A second run over the same ledger skips the completed cell (resumable).
    outcome = run_dry_cell(
        env="c11", optimizer="copro", root=tmp_path,
        execution_mode=ExecutionMode.IN_PROCESS,
    )
    assert outcome.skipped is True
    # Still exactly one ledger line (the completed cell was not duplicated).
    assert len(Ledger(root=tmp_path).load()) == 1
