"""Identity (eval) optimizer derives NO Reward + the residual arm guard.

Two coupled fixes for the ling/c11 anchor exit-1 defect:

1. **The eval (identity) optimizer performs no search**, so its single
   internal measurement is never selected against a proposal and its Reward is
   vestigial. Deriving one under the env's FAIL missing-data policy crashed the
   whole cell (a raw process exit 1) the instant any internal rollout failed
   (ling's c11 429s). The identity optimizer now measures the internal split
   with NO Reward; every SEARCHING optimizer keeps the Reward it selects on.

2. **Residual hardening**: a searching optimizer's internal measurement can
   still raise the typed ``CandidateEvaluationFailure``. During a cell's anchor
   (optimize) phase that must finalize the cell typed as ``incomplete-arm``
   (arm=internal) with real spend + retained partials -- never a raw exit 1.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from dr_providers import (
    FailureClass,
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderTransportFailure,
    ProviderTransportPolicy,
    RawHttpRequest,
)

from tests.envs.support import _prompt_of, _response, transport_policy
from whetstone.envs.internal_eval import run_internal_eval
from whetstone.envs.registry import env_spec
from whetstone.envs.reward import CandidateEvaluationFailure
from whetstone.optimization.mutation import MUTATION_FIELD
from whetstone.optimization.reward import Reward
from whetstone.optimization.schema import Candidate
from whetstone.runner.cell import CellConfig, run_cell
from whetstone.runner.execution_mode import ExecutionMode
from whetstone.runner.ledger import Ledger
from whetstone.runner.optimizers import run_optimize

from .support import (
    PROPOSER_MODEL,
    SPLIT,
    TASK_MODEL,
    ScriptedProposer,
    _split_fits,
    correct_reply,
    credits_fetcher,
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


def _config(env: str, *, optimizer: str, transport) -> CellConfig:
    return CellConfig(
        optimizer=optimizer, env=env, lane="openrouter", attempt=0,
        task_model=TASK_MODEL, proposer_model=PROPOSER_MODEL, canonical=True,
        proposer_config=proposer_config(),
        proposer_transport=ScriptedProposer((WIN,)),
        rollout_transport=transport,
        execution_policy=runner_execution_policy(),
        repeats=3, pool_n_per_stratum=_pool_n(env), split_sizes=SPLIT,
        execution_mode=ExecutionMode.IN_PROCESS, max_wall_seconds=10_000.0,
    )


@dataclass
class _FailMatchingInputs:
    """Fail (PERMANENT) every call whose prompt carries a matching instance.

    A PERMANENT failure is not transient, so a bounded re-drive cannot recover
    it: every repeat of the matched instance lands a failed row and the
    aggregate over those instances propagates to None -- modelling ling's c11
    internal-split 429 wipeout (some rollouts never scored).
    """

    fail_inputs: frozenset[str]
    reply: Callable[[str], str]
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)
    served: int = 0

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        self.served += 1
        prompt = _prompt_of(request)
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"content-type": "json"}, body={"model": "test-model"},
        )
        if any(inp in prompt for inp in self.fail_inputs):
            failure = ProviderTransportFailure(
                failure_class=FailureClass.PERMANENT,
                code="http_status_429",
                message="scripted permanent failure (rate limited)",
                retryable=False,
            )
            return ProviderInvocationEvidence.build(
                request=request, policy=self.policy,
                raw_request=raw_request, outcome=failure,
            )
        return ProviderInvocationEvidence.build(
            request=request, policy=self.policy, raw_request=raw_request,
            outcome=_response(self.reply(prompt)),
        )


def _internal_inputs(exp) -> frozenset[str]:
    """The ``input`` substrings of the INTERNAL-split instances (disjoint
    from the official split): a predicate hits internal rollouts only."""
    return frozenset(
        str(inst.prompt_inputs["input"])
        for inst in exp.eval_configs.internal.instances
    )


# --- FIX 1: the identity optimizer derives NO Reward. ---


def test_run_internal_eval_no_reward_when_apply_reward_false() -> None:
    # The absence assertion at the source: with apply_reward=False the result
    # carries the aggregate + per-task vectors but NO Reward object.
    exp = tiny_experiment("c11")
    internal = exp.eval_configs.internal.instances
    result = run_internal_eval(
        exp, candidate=exp.initial_candidate, instances=internal,
        execution_policy=runner_execution_policy(),
        transport=_FailMatchingInputs(  # even all-failing rows: no crash
            fail_inputs=_internal_inputs(exp), reply=correct_reply(exp)
        ),
        repeats=2, apply_reward=False,
    )
    assert result.reward is None
    # The incompleteness is VISIBLE (score None), never a Reward-policy crash.
    assert result.aggregate.aggregation_output.value is None


def test_eval_optimizer_survives_all_internal_rows_failing() -> None:
    # The core FIX 1 guarantee: the identity optimizer's internal measurement,
    # even when EVERY internal rollout fails, does NOT raise
    # CandidateEvaluationFailure (no Reward is derived). Pre-fix this raised.
    exp = tiny_experiment("c11")
    transport = _FailMatchingInputs(
        fail_inputs=_internal_inputs(exp), reply=correct_reply(exp)
    )
    result = run_optimize(
        exp, optimizer="eval", proposer_config=proposer_config(),
        proposer_transport=ScriptedProposer(()),
        rollout_transport=transport,
        execution_policy=runner_execution_policy(),
        internal_instances=exp.eval_configs.internal.instances, repeats=3,
    )
    assert result.optimizer_steps == 0
    assert result.best_candidate.candidate_id == (
        exp.initial_candidate.candidate_id
    )


def test_searching_optimizer_still_derives_reward() -> None:
    # FIX 1 must NOT disturb searching optimizers: a copro internal
    # measurement over CLEAN internal rows still computes a Reward object
    # (apply_reward=True is the non-identity internal-eval default).
    exp = tiny_experiment("c11")
    result = run_internal_eval(
        exp, candidate=exp.initial_candidate,
        instances=exp.eval_configs.internal.instances,
        execution_policy=runner_execution_policy(),
        transport=_FailMatchingInputs(
            fail_inputs=frozenset(), reply=correct_reply(exp)
        ),
        repeats=2, apply_reward=True,
    )
    assert isinstance(result.reward, Reward)


# --- Cell-level: eval cell with failing internal rows finalizes typed. ---


def test_eval_cell_with_failing_internal_rows_finalizes_typed(
    tmp_path: Path,
) -> None:
    # Pre-fix: the eval cell's optimize phase derived a Reward over an
    # incomplete internal aggregate and crashed the whole cell (raw exit 1).
    # Post-fix: the identity optimizer derives no Reward, so failing internal
    # rows are irrelevant to the eval anchor -- the cell finalizes with a
    # TYPED terminal status off its (clean) official arms, and a ledger line is
    # recorded (never a raw process exit).
    env = "c11"
    exp = tiny_experiment(env)
    transport = _FailMatchingInputs(
        fail_inputs=_internal_inputs(exp), reply=correct_reply(exp)
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        _config(env, optimizer="eval", transport=transport), ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 617.0)]),
    )
    r = outcome.record
    # A typed terminal status was recorded (not a crash); the official arms
    # resolved cleanly so the cell is NOT incomplete-arm.
    assert r.status in {"improved", "inconclusive", "no-improvement"}
    assert r.status != "incomplete-arm"
    assert r.baseline_official is not None
    assert r.best_official is not None
    # A cleanly-finalized cell drops its partial log.
    assert not (
        tmp_path / "partials" / f"{r.cell_id}.partial.jsonl"
    ).exists()


# --- FIX 2: a searching optimizer's failing internal arm finalizes typed. ---


def test_searching_cell_incomplete_internal_arm_not_raw_exit(
    tmp_path: Path,
) -> None:
    # A copro cell whose INTERNAL rollouts fail (so a proposal cannot be scored
    # into a Reward) must NOT crash the cell with a raw exit 1: the escaping
    # CandidateEvaluationFailure is caught and the cell finalizes typed as
    # incomplete-arm(arm=internal), keeps its partials, and reports real spend.
    env = "c11"
    exp = tiny_experiment(env)
    transport = _FailMatchingInputs(
        fail_inputs=_internal_inputs(exp), reply=correct_reply(exp)
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        _config(env, optimizer="copro", transport=transport), ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 617.0)]),
    )
    r = outcome.record
    assert r.status == "incomplete-arm"
    # No best/delta/headroom determination off the unresolved internal arm.
    assert r.best_official is None
    assert r.delta is None
    assert r.headroom_delta is None
    assert r.no_demonstrable_headroom is None
    # The failed arm is named as the INTERNAL arm with the reward cause.
    assert "internal" in r.escalation_note
    assert "Reward" in r.escalation_note
    # Real spend is attributed (credits usage 616 -> 617 = $1.00).
    assert r.spend_usd > 0.0
    # Partials are KEPT for a resume (an incomplete-arm cell is resumable).
    assert (
        tmp_path / "partials" / f"{r.cell_id}.partial.jsonl"
    ).exists()
    # The incomplete cell must NOT poison the per-env official cache.
    assert ledger.env_cache_for(env, task_model=TASK_MODEL) is None


# --- Belt-and-braces: a residual render KeyError on a NON-canonical template
#     fails that candidate's rows as a typed failure, never a cell crash. ---


def _bad_candidate(exp) -> Candidate:
    """A candidate whose template names an unfillable placeholder ({nope})."""
    return Candidate(
        candidate_id="bad-p1",
        base_ref=exp.initial_candidate.base_ref,
        payload={MUTATION_FIELD: "junk {nope} tail"},
    )


def test_render_guard_fails_candidate_row_not_the_cell() -> None:
    # A candidate template whose placeholder the render cannot fill would raise
    # a loud KeyError from the env probe surface (the c22 crash). Under the
    # guarded (candidate) path the render KeyError is caught: every row lands
    # as a typed RENDER_FAILURE_CODE failure and the aggregate is visibly
    # incomplete (score None) -- NOT a process crash. No provider call is made.
    exp = tiny_experiment("c22")  # str.format render -> the true c22 KeyError
    served: list[int] = [0]

    def _reply(_prompt: str) -> str:  # pragma: no cover - never reached
        served[0] += 1
        return "unused"

    result = run_internal_eval(
        exp, candidate=_bad_candidate(exp),
        instances=exp.eval_configs.internal.instances,
        execution_policy=runner_execution_policy(),
        transport=_FailMatchingInputs(fail_inputs=frozenset(), reply=_reply),
        repeats=2, apply_reward=False, render_guard=True,
    )
    # Every planned row failed at render; the aggregate is incomplete, not a
    # crash, and no provider call was ever dispatched.
    assert result.aggregate.rows_failed > 0
    assert result.aggregate.aggregation_output.value is None
    assert served[0] == 0


def test_render_guard_off_keeps_loud_crash_for_canonical() -> None:
    # Canonical naive/ceiling renders are NOT guarded (render_guard defaults
    # False): a template-drift KeyError propagates loudly as designed.
    exp = tiny_experiment("c22")
    with pytest.raises(KeyError):
        run_internal_eval(
            exp, candidate=_bad_candidate(exp),
            instances=exp.eval_configs.internal.instances,
            execution_policy=runner_execution_policy(),
            transport=_FailMatchingInputs(
                fail_inputs=frozenset(), reply=correct_reply(exp)
            ),
            repeats=1, apply_reward=False,
        )


def test_no_reward_avoids_crash_but_searching_still_would_raise() -> None:
    # Contrast that pins the behavioural difference FIX 1 relies on: the SAME
    # incomplete internal aggregate raises CandidateEvaluationFailure when a
    # Reward IS derived (searching path); it does NOT when none is (identity).
    exp = tiny_experiment("c11")
    fail = _internal_inputs(exp)
    with pytest.raises(CandidateEvaluationFailure):
        run_internal_eval(
            exp, candidate=exp.initial_candidate,
            instances=exp.eval_configs.internal.instances,
            execution_policy=runner_execution_policy(),
            transport=_FailMatchingInputs(
                fail_inputs=fail, reply=correct_reply(exp)
            ),
            repeats=2, apply_reward=True,
        )
