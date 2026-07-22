"""Resumability: a completed cell is skipped; a killed cell restarts+resumes.

``cells.jsonl`` is the authoritative ledger. A completed ``(optimizer, env,
attempt)`` cell is skipped on resume. A cell interrupted mid-run (nothing
appended) is re-run on the next invocation; because this reduction's optimizer
state is not resumable, the cell is restarted and the outcome records that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whetstone.envs.registry import env_spec
from whetstone.runner.cell import CellConfig, run_cell
from whetstone.runner.execution_mode import ExecutionMode
from whetstone.runner.ledger import Ledger

from .support import (
    PROPOSER_MODEL,
    SPLIT,
    TASK_MODEL,
    FakeTransport,
    ScriptedProposer,
    _split_fits,
    credits_fetcher,
    improvement_reply,
    proposer_config,
    runner_execution_policy,
    tiny_experiment,
)

WIN = "WIN_TEMPLATE {input}"


def _pool_n(env: str) -> int:
    n = 1
    while not _split_fits(env_spec(env), n):
        n += 1
    return n


def _config(env: str, exp, *, attempt: int = 0) -> CellConfig:
    return CellConfig(
        optimizer="copro",
        env=env,
        lane="openrouter",
        attempt=attempt,
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        canonical=True,
        proposer_config=proposer_config(),
        proposer_transport=ScriptedProposer((WIN,)),
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        execution_policy=runner_execution_policy(),
        repeats=3,
        pool_n_per_stratum=_pool_n(env),
        split_sizes=SPLIT,
        execution_mode=ExecutionMode.IN_PROCESS,
    )


def test_completed_cell_is_skipped_on_resume(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(env, exp)
    ledger = Ledger(root=tmp_path)
    first = run_cell(
        cfg,
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    assert not first.skipped
    assert first.record.status == "improved"

    # A fresh Ledger over the SAME cells.jsonl (a real process restart) skips
    # the completed cell -- no re-run, no second ledger line.
    fresh_ledger = Ledger(root=tmp_path)
    fresh_ledger.load()
    second = run_cell(
        _config(env, exp),
        ledger=fresh_ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.5), (710.0, 617.0)]),
    )
    assert second.skipped
    assert second.reason == "already completed"
    # Still exactly one persisted cell line.
    assert len(Ledger(root=tmp_path).load()) == 1


def test_killed_mid_cell_restarts_and_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    ledger = Ledger(root=tmp_path)

    # Simulate a kill mid-cell: run_cell raises before appending (patch the
    # optimize step to blow up). Nothing is appended to cells.jsonl.
    cfg = _config(env, exp)
    import whetstone.runner.cell as cell_mod

    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("killed mid-optimization")

    monkeypatch.setattr(cell_mod, "run_optimize", _boom)
    with pytest.raises(RuntimeError, match="killed"):
        run_cell(
            cfg,
            ledger=ledger,
            credits_fetcher=credits_fetcher(
                [(710.0, 616.0), (710.0, 616.5)]
            ),
        )
    monkeypatch.undo()

    # The interrupted cell left NO completed cells.jsonl line.
    assert Ledger(root=tmp_path).load() == []

    # Resume: a fresh run completes the cell and records it as restarted
    # (this reduction's optimization state is not resumable).
    fresh = Ledger(root=tmp_path)
    fresh.load()
    resumed = run_cell(
        _config(env, exp),
        ledger=fresh,
        credits_fetcher=credits_fetcher([(710.0, 616.5), (710.0, 617.0)]),
    )
    assert not resumed.skipped
    assert resumed.record.status == "improved"
    assert len(Ledger(root=tmp_path).load()) == 1


def test_second_attempt_is_a_distinct_key(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    ledger = Ledger(root=tmp_path)
    run_cell(
        _config(env, exp, attempt=0),
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    # attempt=1 is a different resumability key -> a new cell is run.
    out = run_cell(
        _config(env, exp, attempt=1),
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.5), (710.0, 617.0)]),
    )
    assert not out.skipped
    assert out.record.attempt == 1
    assert out.resumed  # a prior (optimizer, env) attempt exists
    assert len(ledger.cells()) == 2
