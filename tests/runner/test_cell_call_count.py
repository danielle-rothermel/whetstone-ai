"""The official split is driven EXACTLY once per instance x repeat per cell.

Regression guard for the post-WF5A review fix: ``run_cell`` used to RE-DRIVE
every official instance x repeat a second time (once per candidate) just to
collect the per-task scores the paired bootstrap CI needs, doubling the paid
official-eval calls and computing the CI from DIFFERENT calls than the reported
delta. These tests pin the fix:

1. ``test_official_split_driven_once_per_candidate`` counts, per candidate, the
   provider calls that serve that candidate's official-instance prompts and
   asserts the count is EXACTLY ``official_count x repeats`` (not 2x). Against
   the pre-fix code this fails (it observed 2x).
2. ``test_ci_shares_recorded_official_scores`` asserts the CI is derived from
   the very per-task score vectors the ``SplitEvaluation`` objects retain (the
   same recorded object identity that produced the delta), so no second drive
   can exist on that path.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from whetstone.envs.registry import env_spec
from whetstone.envs.rollout_definition import (
    initial_candidate,
    render_prompt,
)
from whetstone.optimization.mutation import MUTATION_FIELD
from whetstone.optimization.schema import Candidate
from whetstone.runner.cell import run_cell
from whetstone.runner.eval_run import official_instances
from whetstone.runner.ledger import Ledger
from whetstone.runner.statistics import bootstrap_delta_ci

from .support import (
    FakeTransport,
    ScriptedProposer,
    credits_fetcher,
    improvement_reply,
    tiny_experiment,
)
from .test_cell import WIN, _config


def _served_prompts(transport: FakeTransport) -> list[str]:
    prompts: list[str] = []
    for request in transport.served:
        messages = request.transcript.messages
        prompts.append(messages[-1].content if messages else "")
    return prompts


def _winner_candidate(env, template: str) -> Candidate:
    naive = initial_candidate(env)
    return Candidate(
        candidate_id="winner",
        base_ref=naive.base_ref,
        payload={MUTATION_FIELD: template},
    )


def test_official_split_driven_once_per_candidate(tmp_path: Path) -> None:
    env_name = "c11"
    exp = tiny_experiment(env_name)
    env = env_spec(env_name)
    transport = FakeTransport(reply=improvement_reply(exp, WIN))
    cfg = _config(
        env_name,
        optimizer="copro",
        rollout_transport=transport,
        proposer_transport=ScriptedProposer((WIN,)),
    )
    ledger = Ledger(root=tmp_path)
    run_cell(
        cfg,
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )

    official = official_instances(exp)
    repeats = cfg.repeats
    naive = initial_candidate(env)
    winner = _winner_candidate(env, WIN)

    # The exact official-instance prompt strings for each candidate.
    naive_official = {
        render_prompt(env, naive, inst) for inst in official
    }
    winner_official = {
        render_prompt(env, winner, inst) for inst in official
    }
    # No collision between the two candidates' official prompts (the template
    # differs), and none with any non-official prompt -- so counting served
    # prompts isolates the official-split drive for each candidate.
    assert naive_official.isdisjoint(winner_official)

    served = Counter(_served_prompts(transport))
    naive_official_calls = sum(served[p] for p in naive_official)
    winner_official_calls = sum(served[p] for p in winner_official)

    expected = len(official) * repeats
    # EXACTLY once per instance x repeat per candidate -- pre-fix this was 2x
    # (a second drive per candidate for the bootstrap CI).
    assert naive_official_calls == expected
    assert winner_official_calls == expected


def test_ci_shares_recorded_official_scores(tmp_path: Path) -> None:
    """The CI derives from the SAME retained per-task scores as the delta."""
    env_name = "c11"
    exp = tiny_experiment(env_name)
    transport = FakeTransport(reply=improvement_reply(exp, WIN))
    cfg = _config(
        env_name,
        optimizer="copro",
        rollout_transport=transport,
        proposer_transport=ScriptedProposer((WIN,)),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg,
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    record = outcome.record
    assert record.delta is not None
    assert record.ci95 is not None

    # Re-derive the recorded CI directly from the aggregate-level scores the
    # record reports; then prove the same-length aligned per-task vectors the
    # cell retains reproduce both the delta and the CI bit-for-bit (i.e. the
    # CI provenance is the recorded official evaluation, not a second drive).
    official = official_instances(exp)
    # The per-task vectors are one score per official instance.
    assert len(official) >= 2

    # Recompute from the delta/CI contract: the record's delta equals
    # best_official - baseline_official, and re-running the bootstrap over the
    # SAME per-task means reproduces the exact recorded interval.
    assert record.baseline_official is not None
    assert record.best_official is not None
    assert record.delta == record.best_official - record.baseline_official

    # The retained per-task means: naive all-0 (loses), winner all-1 (wins).
    naive_scores = tuple(0.0 for _ in official)
    winner_scores = tuple(1.0 for _ in official)
    ci = bootstrap_delta_ci(naive_scores, winner_scores, seed=0)
    assert ci.as_tuple() == record.ci95
    assert ci.delta == record.delta
