"""Cell-level robustness: whole-cell wall deadline halt + rate-limit note.

A cell that breaches its ``max_wall_seconds`` finishes in-flight calls,
persists partials, and records ``status=halted`` with a halt reason. A
rate-limit typed failure on any call halves the shared effective concurrency
and the cell records that it fired.
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

from tests.envs.support import (
    _prompt_of,
    _response,
    execution_policy,
    transport_policy,
)
from whetstone.envs.internal_eval import run_internal_eval
from whetstone.envs.registry import env_spec
from whetstone.execution.fanout import FanoutConfig
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


def _config(env: str, exp, *, transport, max_wall: float) -> CellConfig:
    return CellConfig(
        optimizer="copro", env=env, lane="openrouter", attempt=0,
        task_model=TASK_MODEL, proposer_model=PROPOSER_MODEL, canonical=True,
        proposer_config=proposer_config(),
        proposer_transport=ScriptedProposer((WIN,)),
        rollout_transport=transport,
        execution_policy=runner_execution_policy(),
        repeats=3, pool_n_per_stratum=_pool_n(env), split_sizes=SPLIT,
        execution_mode=ExecutionMode.IN_PROCESS, max_wall_seconds=max_wall,
    )


def test_whole_cell_deadline_halts_and_persists(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    ledger = Ledger(root=tmp_path)
    # An injected clock lets the baseline + ceiling phases run within budget,
    # then JUMPS past the 100s deadline so the optimize/best phases see a
    # zero remaining budget -> dispatch stops -> the cell halts.
    ticks = iter(
        [0.0, 0.0, 10.0, 20.0]  # cell_start, start, baseline, ceiling
        + [5000.0] * 20         # every later clock read is past the deadline
    )

    def _clock() -> float:
        return next(ticks)

    cfg = _config(
        env, exp, transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        max_wall=100.0,
    )
    outcome = run_cell(
        cfg, ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.0)]),
        clock=_clock,
    )
    assert outcome.record.status == "halted"
    assert "wall deadline" in outcome.record.escalation_note
    # A halted cell KEEPS its partial log for a later resume.
    assert (tmp_path / "partials").exists()


@dataclass
class _RateLimitOnceTransport:
    """A transport that rate-limits ONE call, then serves the reply normally.

    The rate-limited call retries under a max_attempts=2 policy and the retry
    succeeds, so the aggregate still computes -- but the rate-limit typed
    failure is observed, so the shared effective concurrency halves.
    """

    reply: Callable[[str], str]
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)
    served: int = 0

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        self.served += 1
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"content-type": "json"}, body={"model": "test-model"},
        )
        if self.served == 1:
            failure = ProviderTransportFailure(
                failure_class=FailureClass.RATE_LIMITED,
                code="rate_limited",
                message="429 rate limited", retryable=True,
            )
            return ProviderInvocationEvidence.build(
                request=request, policy=self.policy,
                raw_request=raw_request, outcome=failure,
            )
        return ProviderInvocationEvidence.build(
            request=request, policy=self.policy, raw_request=raw_request,
            outcome=_response(self.reply(_prompt_of(request))),
        )


def test_rate_limit_halves_concurrency_in_internal_eval() -> None:
    # A rate-limit typed failure (that retries to success) is observed during a
    # fan-out phase -> the shared effective concurrency halves and the pass
    # records it. Driven at the internal-eval level so the aggregate is clean.
    env = "c11"
    exp = tiny_experiment(env)
    transport = _RateLimitOnceTransport(reply=correct_reply(exp))
    result = run_internal_eval(
        exp, candidate=exp.initial_candidate,
        instances=exp.eval_configs.official.instances,
        execution_policy=execution_policy(max_attempts=2),
        transport=transport, repeats=3,
        fanout=FanoutConfig(concurrency=4),
    )
    assert result.concurrency_halved
