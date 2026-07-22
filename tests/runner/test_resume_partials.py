"""Resume: a killed mid-drive phase skips already-recorded calls on resume.

The ``.partial.jsonl`` per-call log makes an evaluation drive resumable. A
crash mid-drive leaves completed calls on disk; a resume restores those
observations and re-drives ONLY the missing ones -- never re-calling a recorded
``(instance, candidate, repeat)``. These tests kill a drive with a fake
transport that raises after N calls, then resume and assert (a) the phase
completes and (b) the recorded calls are not re-driven.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from dr_providers import (
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    RawHttpRequest,
)

from tests.envs.support import transport_policy
from whetstone.execution.fanout import FanoutConfig
from whetstone.execution.partials import PartialLog
from whetstone.runner.eval_run import evaluate_split

from .support import (
    correct_reply,
    runner_execution_policy,
    tiny_experiment,
)


class _Killed(RuntimeError):
    """Simulated mid-drive crash."""


@dataclass
class CountingTransport:
    """A fake transport that counts calls and can crash after ``crash_after``.

    Replies the gold for every prompt (so a completed call scores 1.0). When
    ``crash_after`` is set, the (crash_after+1)-th call raises to model a kill
    mid-drive AFTER earlier calls have already been recorded. ``calls`` counts
    physical invocations so a resume can prove it did not re-drive.
    """

    reply: Callable[[str], str]
    crash_after: int | None = None
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)
    calls: int = 0

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        if self.crash_after is not None and self.calls >= self.crash_after:
            raise _Killed("killed mid-drive")
        self.calls += 1
        messages = request.transcript.messages
        prompt = messages[-1].content if messages else ""
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


@dataclass
class _UsageTransport:
    """A transport whose responses carry a token-usage block (like OpenRouter).

    Used to prove FIX 6: the cell-path partial records retain the measured
    token counts (they were null before) for spend reconciliation.
    """

    reply: Callable[[str], str]
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        from dr_providers import TokenUsage

        messages = request.transcript.messages
        prompt = messages[-1].content if messages else ""
        text = self.reply(prompt)
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"Authorization": "Bearer k", "content-type": "json"},
            body={"model": "test-model"},
        )
        return ProviderInvocationEvidence.build(
            request=request, policy=self.policy, raw_request=raw_request,
            outcome=ProviderTransportResponse(
                text=text,
                raw_body={"choices": [{"message": {"content": text}}]},
                response_id="resp-1", model="test-model",
                finish_reason="stop",
                usage=TokenUsage(
                    prompt_tokens=11, completion_tokens=7, total_tokens=18
                ),
            ),
        )


def _evaluate(exp, transport, partial_log, *, concurrency: int = 1):
    return evaluate_split(
        exp,
        candidate=exp.initial_candidate,
        instances=exp.eval_configs.official.instances,
        split_role="official",
        transport=transport,
        execution_policy=runner_execution_policy(),
        repeats=3,
        partial_log=partial_log,
        fanout=FanoutConfig(concurrency=concurrency),
    )


def test_cell_partials_retain_token_counts(tmp_path: Path) -> None:
    # FIX 6: cell-path partial records carry the measured token counts (were
    # null before) so spend reconciliation can sum them. The raw_response is
    # intentionally NOT persisted on the cell path (Rollout Results hold that
    # evidence), keeping the cell partial small.
    exp = tiny_experiment("c11")
    partial_log = PartialLog(path=tmp_path / "cell.partial.jsonl")
    _evaluate(exp, _UsageTransport(reply=correct_reply(exp)), partial_log)
    records = partial_log.load()
    assert records, "the drive should have recorded per-call partials"
    scored = [r for r in records if not r.failed]
    assert scored, "expected at least one scored call"
    for rec in scored:
        assert rec.prompt_tokens == 11
        assert rec.completion_tokens == 7
        assert rec.total_tokens == 18
        # raw_response is elided on the cell path (kept empty).
        assert rec.raw_response == ""


def test_resume_skips_already_recorded_calls(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    reply = correct_reply(exp)
    partial_log = PartialLog(path=tmp_path / "cell.partial.jsonl")

    total_planned = len(exp.eval_configs.official.instances) * 3

    # --- Drive 1: crash after 2 successful calls (sequential so the crash is
    # deterministic; the 2 completed calls are already on disk). ---
    crashing = CountingTransport(reply=reply, crash_after=2)
    with pytest.raises(_Killed):
        _evaluate(exp, crashing, partial_log, concurrency=1)
    recorded_after_crash = len(partial_log.load())
    assert recorded_after_crash == 2

    # --- Drive 2 (resume): a fresh transport completes the phase. It must
    # re-drive ONLY the missing calls -- never the 2 already recorded. ---
    resume = CountingTransport(reply=reply)
    result = _evaluate(exp, resume, partial_log, concurrency=1)
    assert result.score == 1.0  # every task scored 1.0
    # Exactly the not-yet-recorded calls were re-driven.
    assert resume.calls == total_planned - recorded_after_crash
    # Every planned observation is now on disk.
    assert len(partial_log.load()) == total_planned


def test_resume_from_complete_partial_drives_nothing(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    reply = correct_reply(exp)
    partial_log = PartialLog(path=tmp_path / "cell.partial.jsonl")

    # Drive 1: complete the whole phase, recording every call.
    first = CountingTransport(reply=reply)
    _evaluate(exp, first, partial_log)
    total = first.calls

    # Drive 2: everything is recorded, so a resume drives ZERO calls and still
    # reconstructs the same score from the restored observations.
    resume = CountingTransport(reply=reply)
    result = _evaluate(exp, resume, partial_log)
    assert resume.calls == 0
    assert result.score == 1.0
    assert total > 0
