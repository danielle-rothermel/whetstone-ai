"""RECORDED-artifact determinism under concurrent, out-of-order completion.

The bounded-concurrency fan-out must produce the SAME recorded aggregate and
per-task score vectors no matter which provider call finishes first. This test
drives ``evaluate_split`` through a fake transport that sleeps a per-prompt
amount so completion order is shuffled relative to input order, and asserts the
concurrent drive matches a sequential (concurrency=1) drive byte-for-byte.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from dr_providers import (
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    RawHttpRequest,
)

from tests.envs.support import transport_policy
from whetstone.execution.fanout import FanoutConfig
from whetstone.runner.eval_run import evaluate_split

from .support import (
    FakeTransport,
    correct_reply,
    runner_execution_policy,
    tiny_experiment,
)


@dataclass
class ShufflingTransport:
    """A fake transport that sleeps a per-prompt amount before replying.

    Completion order is deliberately decoupled from submission order: the sleep
    is a deterministic function of the prompt hash, so different calls finish
    at different times, out of input order. Records requests. No network.
    """

    reply: Callable[[str], str]
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)
    served: list[ProviderCallRequest] = field(default_factory=list)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        self.served.append(request)
        messages = request.transcript.messages
        prompt = messages[-1].content if messages else ""
        # Deterministic, prompt-dependent, order-scrambling delay.
        time.sleep((hash(prompt) % 5) * 0.004)
        text = self.reply(prompt)
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"Authorization": "Bearer k", "content-type": "json"},
            body={"model": "test-model"},
        )
        return ProviderInvocationEvidence.build(
            request=request,
            policy=self.policy,
            raw_request=raw_request,
            outcome=ProviderTransportResponse(
                text=text,
                raw_body={"choices": [{"message": {"content": text}}]},
                response_id="resp-1",
                model="test-model",
                finish_reason="stop",
            ),
        )


def test_concurrent_drive_matches_sequential_drive() -> None:
    env = "c11"
    exp = tiny_experiment(env)
    reply = correct_reply(exp)
    candidate = exp.initial_candidate
    official = exp.eval_configs.official.instances

    # Sequential (concurrency=1) reference.
    seq = evaluate_split(
        exp, candidate=candidate, instances=official, split_role="official",
        transport=FakeTransport(reply=reply),
        execution_policy=runner_execution_policy(), repeats=3,
        fanout=FanoutConfig(concurrency=1),
    )
    # Concurrent (5-way) drive with shuffled completion order.
    conc = evaluate_split(
        exp, candidate=candidate, instances=official, split_role="official",
        transport=ShufflingTransport(reply=reply),
        execution_policy=runner_execution_policy(), repeats=3,
        fanout=FanoutConfig(concurrency=5),
    )

    # The RECORDED artifact is identical: score, per-task vectors, counts, and
    # the persisted aggregate content hash.
    assert conc.score == seq.score
    assert conc.per_task_scores == seq.per_task_scores
    assert conc.per_task_counts == seq.per_task_counts
    assert conc.artifact_ref.content_hash == seq.artifact_ref.content_hash


def test_repeated_concurrent_drives_are_identical() -> None:
    env = "c11"
    exp = tiny_experiment(env)
    reply = correct_reply(exp)
    official = exp.eval_configs.official.instances

    def _drive() -> tuple[object, ...]:
        out = evaluate_split(
            exp, candidate=exp.initial_candidate, instances=official,
            split_role="official", transport=ShufflingTransport(reply=reply),
            execution_policy=runner_execution_policy(), repeats=3,
            fanout=FanoutConfig(concurrency=5),
        )
        return (
            out.score, out.per_task_scores, out.per_task_counts,
            out.artifact_ref.content_hash,
        )

    assert _drive() == _drive()
