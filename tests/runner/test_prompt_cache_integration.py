"""The prompt cache wired through the shared split-eval seam (task 31).

Proves the opt-in cache reaches the QA cell path via ``evaluate_split`` at its
single choke point (``execute_call``): a second byte-identical evaluation over
the same split + repeats reuses the stored provider Results (no new wire calls)
and marks every reused row cache_hit with the ORIGINAL entry's provenance and a
null (never fabricated) latency -- while the reduced aggregate is unchanged. It
also pins the opt-out default: with no cache the drive is byte-identical and no
store is created.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dr_providers import (
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderTransportPolicy,
    RawHttpRequest,
)

from tests.envs.support import _prompt_of, _response, transport_policy
from whetstone.execution.fanout import FanoutConfig
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.runner.eval_run import evaluate_split

from .support import correct_reply, runner_execution_policy, tiny_experiment


@dataclass
class _CountingTransport:
    """A prompt-keyed fake transport that COUNTS every wire call it serves."""

    reply: Callable[[str], str]
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        with self._lock:
            self.calls += 1
        text = self.reply(_prompt_of(request))
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"content-type": "json"},
            body={"model": "test-model"},
        )
        return ProviderInvocationEvidence.build(
            request=request, policy=self.policy,
            raw_request=raw_request, outcome=_response(text),
        )


def _evaluate(exp, transport, partial_log, cache):
    return evaluate_split(
        exp,
        candidate=exp.initial_candidate,
        instances=exp.eval_configs.official.instances,
        split_role="official",
        transport=transport,
        execution_policy=runner_execution_policy(),
        repeats=3,
        partial_log=partial_log,
        fanout=FanoutConfig(concurrency=1),
        cache=cache,
    )


def test_second_eval_reuses_cache_and_marks_rows(tmp_path: Path) -> None:
    exp = tiny_experiment("c11")
    reply = correct_reply(exp)
    cache = PromptResultCache(root=tmp_path)

    # Drive 1 (misses): populate the store. Every planned row drives once.
    t1 = _CountingTransport(reply=reply)
    log1 = PartialLog(path=tmp_path / "cellA.partial.jsonl")
    eval1 = _evaluate(exp, t1, log1, cache)
    planned = len(exp.eval_configs.official.instances) * 3
    assert t1.calls == planned
    counters_after_1 = cache.counters()
    assert counters_after_1["misses"] == planned
    assert counters_after_1["stores"] == planned
    assert counters_after_1["hits"] == 0

    # Drive 2 (a SEPARATE cell, no partial-log restore): every row is a HIT.
    # A fresh transport that would RAISE proves no wire call is made on a hit.
    def _boom(_prompt: str) -> str:  # pragma: no cover - must never be called
        raise AssertionError("cache hit must not touch the transport")

    t2 = _CountingTransport(reply=_boom)
    log2 = PartialLog(path=tmp_path / "cellB.partial.jsonl")
    eval2 = _evaluate(exp, t2, log2, cache)
    assert t2.calls == 0  # every row served from cache
    assert cache.counters()["hits"] == planned

    # The reduced aggregate is byte-identical across the two drives.
    assert eval2.score == eval1.score
    assert eval2.per_task_scores == eval1.per_task_scores

    # Every reused row is marked cache_hit with null latency + provenance ref
    # to the ORIGINAL entry (its source phase/unit + original store timestamp).
    rows2 = [r for r in log2.load()]
    assert rows2, "the second drive should have recorded per-call partials"
    assert all(r.cache_hit for r in rows2)
    for r in rows2:
        assert r.latency_s is None  # never a fabricated 0
        assert r.cache_source_phase == "cell"  # the drive-1 partial phase
        assert r.cache_source_at is not None
        assert r.cache_source_call_id is not None


def test_opt_out_default_is_byte_identical_no_store(tmp_path: Path) -> None:
    exp = tiny_experiment("c11")
    reply = correct_reply(exp)

    # No cache: two drives each fully drive the transport, no store is made,
    # and no row is marked cache_hit.
    t = _CountingTransport(reply=reply)
    log = PartialLog(path=tmp_path / "cell.partial.jsonl")
    evaluate_split(
        exp, candidate=exp.initial_candidate,
        instances=exp.eval_configs.official.instances,
        split_role="official", transport=t,
        execution_policy=runner_execution_policy(), repeats=3,
        partial_log=log, fanout=FanoutConfig(concurrency=1),
    )
    planned = len(exp.eval_configs.official.instances) * 3
    assert t.calls == planned
    assert not (tmp_path / "prompt_cache").exists()
    assert all(not r.cache_hit for r in log.load())
