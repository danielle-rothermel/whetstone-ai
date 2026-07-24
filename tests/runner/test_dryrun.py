"""The fake-transport dry-run cell path, driven as a program (no live call).

``--dry-run-fake`` runs a full cell against scripted fake transports so the CLI
plumbing (build_env_experiment -> baseline/ceiling/best official evals ->
optimizer internal search -> delta + CI -> ledger append) can be proven as a
program, not only under pytest. No test here passes ``--live``; every transport
is a scripted fake, so no live paid LLM call is ever made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whetstone.envs.factory import build_env_experiment
from whetstone.envs.sampling import SamplingOverrides
from whetstone.runner.cli import main
from whetstone.runner.dryrun import (
    DRYRUN_TASK_MODEL,
    _pool_n_per_stratum,
    run_dry_cell,
)
from whetstone.runner.execution_mode import ExecutionMode
from whetstone.runner.ledger import Ledger
from whetstone.runner.optimizers import OPTIMIZATION_TRACE_SCHEMA


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
    # Eval row establishes the headroom gate: naive == ceiling == 1.0 in the
    # exact-match dry script -> no demonstrable headroom (CI includes 0).
    assert r.headroom_delta == pytest.approx(0.0)
    assert r.no_demonstrable_headroom is True
    assert r.official_repeats_used == 5
    # The per-env official cache was written for reuse by later cells.
    assert Ledger(root=tmp_path).env_cache_for("c11") is not None


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
    # The statistical-confidence fields land end-to-end via the dry-run seam.
    assert r.delta_ci95 == r.ci95
    assert r.naive_ci95 is not None
    assert r.ceiling_ci95 is not None
    assert r.headroom_delta is not None
    assert r.headroom_ci95 is not None
    # copro is not the Eval row, so the headroom gate flag stays unset.
    assert r.no_demonstrable_headroom is None
    assert r.official_repeats_used == 5
    assert r.escalated is False
    assert r.pooled_observation_counts["naive"] > 0
    assert r.pooled_observation_counts["best"] > 0


def test_dry_cell_writes_optimization_trace_with_per_task_and_text(
    tmp_path: Path,
) -> None:
    # The optimizer-search trace is persisted per cell (the highest-value
    # logging addition from the internal-signal analysis): a completed cell's
    # trace file carries the accepted-candidate PROMPT TEXT and each step's
    # per-task per-repeat scoring evidence, so internal-signal reliability is
    # analyzable from disk.
    outcome = run_dry_cell(
        env="c11",
        optimizer="copro",
        root=tmp_path,
        execution_mode=ExecutionMode.IN_PROCESS,
    )
    ledger = Ledger(root=tmp_path)
    cell_id = outcome.record.cell_id
    trace_path = ledger.optimization_trace_path(cell_id)
    assert trace_path.exists(), "the per-cell trace artifact must be written"
    # cells.jsonl points optimization_result_ref at the trace (relative path);
    # the bare accepted-candidate id is preserved on best_candidate_id.
    ref = outcome.record.artifacts.optimization_result_ref
    assert ref is not None and ref.endswith(".json")
    assert (tmp_path / ref) == trace_path
    assert outcome.record.artifacts.best_candidate_id  # bare id kept

    trace = json.loads(trace_path.read_text())
    assert trace["schema"] == OPTIMIZATION_TRACE_SCHEMA
    assert trace["cell_id"] == cell_id
    # The accepted-candidate prompt text is trivially extractable (reports).
    assert isinstance(trace["best_candidate_template"], str)
    assert trace["best_candidate_template"]  # non-empty
    # The as-run internal repeat count is recorded (the r=3 the runner drives,
    # not the briefs' documented r=1).
    assert trace["internal_repeat_count_as_run"] == 3
    assert trace["baseline_internal_score"] is not None
    assert trace["best_internal_score"] is not None
    # Every scored step carries its per-task evidence + full prompt text.
    scored = [s for s in trace["steps"] if s["accepted"]]
    assert scored, "at least one candidate must have scored"
    step = scored[0]
    assert isinstance(step["template"], str) and step["template"]
    assert step["evaluation"] is not None
    assert isinstance(step["evaluation"]["per_task_scores"], list)
    assert step["evaluation"]["per_task_scores"]  # one entry per internal task
    assert isinstance(step["evaluation"]["per_task_counts"], list)


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


def test_dry_cell_with_sampling_overrides_end_to_end(tmp_path: Path) -> None:
    # A reduced-sampling dry cell runs end-to-end: the overrides are recorded
    # on the ledger line, only the first-N official tasks are evaluated, and
    # the env cache is keyed by the REDUCED official eval_config_hash (so it
    # would MISS a full-config cache). The dry official split is (2,2,2), so
    # official-n=1 is a strict reduction.
    outcome = run_dry_cell(
        env="c11",
        optimizer="eval",
        root=tmp_path,
        execution_mode=ExecutionMode.IN_PROCESS,
        overrides=SamplingOverrides(official_n=1, official_repeats=2),
    )
    r = outcome.record
    assert r.sampling_overrides.official_n == 1
    assert r.sampling_overrides.official_repeats == 2
    # The official arms were driven at the overridden repeat count.
    assert r.official_repeats_used == 2
    assert r.status in ("improved", "no-improvement", "inconclusive")
    ledger = Ledger(root=tmp_path)
    # The reduced cache line is keyed by the reduced official eval_config_hash,
    # so a full-config read (the full hash) MISSES it and vice versa -- the
    # reduced cell is a distinct Eval Config identity.
    reduced = build_env_experiment(
        "c11", model=DRYRUN_TASK_MODEL,
        pool_n_per_stratum=_pool_n_per_stratum("c11"),
        split_sizes=(2, 2, 2),
        overrides=SamplingOverrides(official_n=1, official_repeats=2),
    )
    reduced_hash = (
        reduced.eval_configs.official.eval_config.config_identity_hash
    )
    assert ledger.env_cache_for(
        "c11", task_model=DRYRUN_TASK_MODEL, eval_config_hash=reduced_hash,
    ) is not None
    full = build_env_experiment(
        "c11", model=DRYRUN_TASK_MODEL,
        pool_n_per_stratum=_pool_n_per_stratum("c11"),
        split_sizes=(2, 2, 2),
    )
    full_hash = full.eval_configs.official.eval_config.config_identity_hash
    assert full_hash != reduced_hash
    assert ledger.env_cache_for(
        "c11", task_model=DRYRUN_TASK_MODEL, eval_config_hash=full_hash,
    ) is None


def test_cli_dry_run_fake_with_sampling_override_flags(tmp_path: Path) -> None:
    # The --official-n / --official-repeats flags flow through the dry-run CLI
    # path into the recorded sampling_overrides (no --live needed).
    code = main(
        [
            "--root", str(tmp_path),
            "--execution-mode", "in-process",
            "cell",
            "--optimizer", "eval",
            "--env", "c11",
            "--official-n", "1",
            "--official-repeats", "2",
            "--dry-run-fake",
        ]
    )
    assert code == 0
    loaded = Ledger(root=tmp_path).load()
    assert len(loaded) == 1
    assert loaded[0].sampling_overrides.official_n == 1
    assert loaded[0].sampling_overrides.official_repeats == 2


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


def test_dry_cell_ed1_encdec_end_to_end(tmp_path: Path) -> None:
    # ed1 --dry-run-fake drives the 3-node enc-dec graph offline (no network,
    # no
    # Docker): both scores land + the pinned dataset revision is recorded.
    from whetstone.envs.ed1 import ED1_DATASET_REVISION, ED1_ENV_NAME

    outcome = run_dry_cell(
        env=ED1_ENV_NAME, optimizer="eval", root=tmp_path,
        execution_mode=ExecutionMode.IN_PROCESS, budget_ratio=0.5,
    )
    r = outcome.record
    assert r.env == ED1_ENV_NAME
    assert r.best_official is not None  # pass rate
    assert r.dual_scores is not None
    assert r.dual_scores.best_compression is not None
    assert r.dual_scores.budget_ratio == 0.5
    assert r.dual_scores.dataset_revision == ED1_DATASET_REVISION
    assert len(Ledger(root=tmp_path).load()) == 1


def test_dry_cell_ed1_budget_ratio_flows(tmp_path: Path) -> None:
    from whetstone.envs.ed1 import ED1_ENV_NAME

    outcome = run_dry_cell(
        env=ED1_ENV_NAME, optimizer="eval", root=tmp_path,
        execution_mode=ExecutionMode.IN_PROCESS, budget_ratio=0.75,
    )
    assert outcome.record.dual_scores is not None
    assert outcome.record.dual_scores.budget_ratio == 0.75
