"""Incomplete-official-arm emission guard (the c18:a1 defect).

When an official arm (naive or best) never resolves -- its aggregate scalar
propagates to ``None`` because some rollouts failed after FIX 2's bounded
re-drive -- the cell MUST NOT emit a headroom / no-demonstrable-headroom
determination or a terminal statistical status. It finalizes as
``incomplete-arm`` naming the failed arm/rows, keeps its partials for a resume,
does NOT poison the per-env official cache, and reports real spend.

This is the exact c18:a1 pathology: naive=None yet the ledger line carried
headroom_delta / no_demonstrable_headroom=True and status='no-improvement'
(a certified-looking verdict from a partial vector) with spend_usd=0.0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dr_providers import (
    FailureClass,
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderTransportFailure,
    ProviderTransportPolicy,
    RawHttpRequest,
)

from tests.envs.support import _prompt_of, _response, transport_policy
from whetstone.envs.registry import env_spec
from whetstone.envs.rollout_definition import initial_candidate, render_prompt
from whetstone.runner.cell import CellConfig, run_cell
from whetstone.runner.execution_mode import ExecutionMode
from whetstone.runner.ledger import Ledger

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


def _config(env: str, exp, *, transport) -> CellConfig:
    return CellConfig(
        optimizer="eval", env=env, lane="openrouter", attempt=0,
        task_model=TASK_MODEL, proposer_model=PROPOSER_MODEL, canonical=True,
        proposer_config=proposer_config(),
        proposer_transport=ScriptedProposer((WIN,)),
        rollout_transport=transport,
        execution_policy=runner_execution_policy(),
        repeats=3, pool_n_per_stratum=_pool_n(env), split_sizes=SPLIT,
        execution_mode=ExecutionMode.IN_PROCESS, max_wall_seconds=10_000.0,
    )


@dataclass
class _FailMatchingPrompts:
    """Fail (PERMANENT) prompts matching a predicate; succeed everything else.

    A PERMANENT transport failure is not transient, so FIX 2's bounded re-drive
    does NOT recover it -- the matching arm lands failed rows and its aggregate
    scalar propagates to None (an incomplete official arm), modelling the
    deepseek call failures behind c18:a1.
    """

    should_fail: Callable[[str], bool]
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
        if self.should_fail(prompt):
            failure = ProviderTransportFailure(
                failure_class=FailureClass.PERMANENT,
                code="empty_completion",
                message="scripted permanent failure (empty completion)",
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


def _naive_prompts_for(exp, instances) -> list[str]:
    env = env_spec(exp.env_name)
    naive = initial_candidate(env)
    return [render_prompt(env, naive, inst) for inst in instances]


def test_incomplete_naive_arm_finalizes_incomplete_not_certified(
    tmp_path: Path,
) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    official = exp.eval_configs.official.instances
    # Fail the naive-arm prompts of the FIRST official instance -> that task's
    # repeats all fail -> the naive aggregate scalar propagates to None.
    fail = frozenset(_naive_prompts_for(exp, official[:1]))
    transport = _FailMatchingPrompts(
        should_fail=lambda p: p in fail, reply=correct_reply(exp)
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        _config(env, exp, transport=transport), ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 617.0)]),
    )
    r = outcome.record
    # --- The core guarantee: NO certified-looking output. ---
    assert r.status == "incomplete-arm"
    assert r.baseline_official is None
    assert r.headroom_delta is None
    assert r.headroom_ci95 is None
    assert r.no_demonstrable_headroom is None
    assert r.delta is None
    assert r.delta_ci95 is None
    assert r.naive_ci95 is None
    # The failed arm + its row accounting are named in the note.
    assert "incomplete official arm" in r.escalation_note
    assert "naive" in r.escalation_note
    assert "rows_failed=" in r.escalation_note
    # Real spend is attributed (credits dropped 616 -> 617 usage = $1.00).
    assert r.spend_usd > 0.0
    # The partial log is KEPT for a resume (not deleted like a clean cell).
    assert (tmp_path / "partials" / f"{r.cell_id}.partial.jsonl").exists()
    # The eval-row official cache is NOT poisoned with the incomplete arm.
    assert ledger.env_cache_for(env, task_model=TASK_MODEL) is None


def test_incomplete_best_arm_also_incomplete(tmp_path: Path) -> None:
    # Symmetric: the BEST arm (a copro optimizer winner) failing on the
    # official split also yields incomplete-arm. Failing every prompt of the
    # first official instance (input substring) fails whichever candidate the
    # best arm drives on that task, so best_official propagates to None.
    env = "c11"
    exp = tiny_experiment(env)
    official = exp.eval_configs.official.instances
    first_input = official[0].prompt_inputs["input"]

    def _should_fail(prompt: str) -> bool:
        return str(first_input) in prompt

    transport = _FailMatchingPrompts(
        should_fail=_should_fail, reply=correct_reply(exp)
    )
    ledger = Ledger(root=tmp_path)
    cfg = _config(env, exp, transport=transport)
    outcome = run_cell(
        cfg, ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 617.0)]),
    )
    r = outcome.record
    # An arm covering the failed task is incomplete -> the guard fires and the
    # cell is NOT certified (no headroom, no terminal statistical status).
    assert r.status == "incomplete-arm"
    assert r.best_official is None
    assert r.headroom_delta is None
    assert r.no_demonstrable_headroom is None
    assert "incomplete official arm" in r.escalation_note
