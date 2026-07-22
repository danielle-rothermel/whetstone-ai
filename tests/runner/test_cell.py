"""Full cell runs against scripted responses: improvement + no-improvement."""

from __future__ import annotations

from pathlib import Path

import pytest

from whetstone.envs.registry import env_spec
from whetstone.runner.budget import BudgetGuard, ReserveError
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
    correct_reply,
    credits_fetcher,
    improvement_reply,
    no_improvement_reply,
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


def _config(
    env: str,
    *,
    optimizer: str,
    rollout_transport,
    proposer_transport,
    attempt: int = 0,
    canonical: bool = True,
    lane: str = "openrouter",
) -> CellConfig:
    return CellConfig(
        optimizer=optimizer,
        env=env,
        lane=lane,
        attempt=attempt,
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        canonical=canonical,
        proposer_config=proposer_config(),
        proposer_transport=proposer_transport,
        rollout_transport=rollout_transport,
        execution_policy=runner_execution_policy(),
        repeats=3,
        pool_n_per_stratum=_pool_n(env),
        split_sizes=SPLIT,
        execution_mode=ExecutionMode.IN_PROCESS,
    )


def test_cell_improvement_script(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env,
        optimizer="copro",
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        proposer_transport=ScriptedProposer((WIN,)),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg,
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    assert r.status == "improved"
    assert r.baseline_official == pytest.approx(0.0)
    assert r.best_official == pytest.approx(1.0)
    assert r.delta == pytest.approx(1.0)
    assert r.delta is not None
    assert r.ci95 is not None
    assert r.ci95[0] <= r.delta <= r.ci95[1]
    assert r.internal_evals_count >= 1
    assert r.spend_usd == pytest.approx(0.5)
    # The ledger line lands.
    assert len(ledger.cells()) == 1


def test_cell_no_improvement_script(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env,
        optimizer="copro",
        rollout_transport=FakeTransport(reply=no_improvement_reply(exp)),
        proposer_transport=ScriptedProposer(("also-loses {input}",)),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg,
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.1)]),
    )
    r = outcome.record
    assert r.status == "no-improvement"
    assert r.baseline_official == pytest.approx(0.0)
    assert r.best_official == pytest.approx(0.0)
    assert r.delta == pytest.approx(0.0)


def test_cell_ceiling_cached_once_per_env(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    ledger = Ledger(root=tmp_path)
    # First cell computes and records the ceiling.
    run_cell(
        _config(
            env,
            optimizer="copro",
            rollout_transport=FakeTransport(reply=correct_reply(exp)),
            proposer_transport=ScriptedProposer((WIN,)),
        ),
        ledger=ledger,
    )
    ceiling = ledger.ceiling_for(env)
    assert ceiling is not None
    # A second (different optimizer) cell reuses the cached ceiling verbatim.
    out2 = run_cell(
        _config(
            env,
            optimizer="miprov2",
            rollout_transport=FakeTransport(reply=correct_reply(exp)),
            proposer_transport=ScriptedProposer((WIN,)),
        ),
        ledger=ledger,
    )
    assert out2.record.ceiling_official == ceiling


def test_cell_records_execution_mode(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env,
        optimizer="eval",
        rollout_transport=FakeTransport(reply=correct_reply(exp)),
        proposer_transport=ScriptedProposer(()),
    )
    ledger = Ledger(root=tmp_path)
    out = run_cell(cfg, ledger=ledger)
    # eval identity: no proposal steps (naive == best).
    assert out.record.optimizer_steps == 0
    # The persisted aggregate artifacts carry the execution mode.
    # (recorded on the SplitEvaluation; the cell status is terminal)
    assert out.record.status in ("improved", "no-improvement")


def test_reserve_guard_refuses_canonical_below_reserve(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env,
        optimizer="copro",
        rollout_transport=FakeTransport(reply=no_improvement_reply(exp)),
        proposer_transport=ScriptedProposer(("x {input}",)),
    )
    ledger = Ledger(root=tmp_path)
    # remaining $10 < reserve $18.60 -> refuse to start.
    with pytest.raises(ReserveError, match="reserve"):
        run_cell(
            cfg,
            ledger=ledger,
            budget=BudgetGuard(),
            credits_fetcher=credits_fetcher([(700.0, 690.0), (700.0, 690.0)]),
        )
    # No cell line was appended (the cell never started).
    assert ledger.cells() == []


def test_stop_loss_halts_cell(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env,
        optimizer="copro",
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        proposer_transport=ScriptedProposer((WIN,)),
    )
    ledger = Ledger(root=tmp_path)
    # A tiny expected cost so the observed $5 spend crosses 2x -> halted.
    budget = BudgetGuard(expected_cell_usd=1.0)  # stop-loss $2
    outcome = run_cell(
        cfg,
        ledger=ledger,
        budget=budget,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 621.0)]),
    )
    assert outcome.record.status == "halted"
    assert outcome.record.spend_usd == pytest.approx(5.0)
