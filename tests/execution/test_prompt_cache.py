"""Prompt-cache identity, provenance, concurrency, and restart behavior."""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Callable
from pathlib import Path

from dr_providers import (
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ReasoningEffort,
    Transcript,
    openrouter_chat_config,
)

from tests.provider import support as s
from whetstone.execution.prompt_cache import (
    PromptCacheError,
    PromptResultCache,
    execute_call,
    prompt_cache_key,
)
from whetstone.provider.attempt import ProviderCallResult
from whetstone.provider.driver import run_provider_call


def _request(
    *,
    model: str = "x/y",
    prompt: str = "hello",
    temperature: float | None = 0.0,
    reasoning: ReasoningEffort | None = None,
    top_p: float | None = None,
    token_limit: int | None = None,
) -> ProviderCallRequest:
    controls = GenerationControls(
        temperature=temperature,
        reasoning=reasoning,
        top_p=top_p,
        token_limit=token_limit,
    )
    return ProviderCallRequest(
        config=openrouter_chat_config(model=model, controls=controls),
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content=prompt),)
        ),
    )


def _successful_result(
    *,
    text: str,
    logical_call_id: str,
) -> ProviderCallResult:
    request = _request()
    transport_policy = s.build_transport_policy()
    transport = s.RecordingTransport(
        request=request,
        transport_policy=transport_policy,
        outcomes=[s.response_outcome(text=text)],
    )
    return run_provider_call(
        request=request,
        policy=s.build_execution_policy(
            transport_policy=transport_policy,
            max_attempts=1,
        ),
        transport=transport,
        logical_call_id=logical_call_id,
        clock=s.FakeClock(),
        sleep=s.SleepRecorder(),
    )


def _execute(
    cache: PromptResultCache | None,
    *,
    request: ProviderCallRequest,
    transport: Callable,
    logical_call_id: str,
    repeat_index: int = 0,
):
    return execute_call(
        request=request,
        policy=s.build_execution_policy(max_attempts=1),
        transport=transport,
        logical_call_id=logical_call_id,
        repeat_index=repeat_index,
        cache=cache,
        phase="internal",
        unit="candidate-1",
        clock=s.FakeClock(),
        sleep=s.SleepRecorder(),
    )


def _concurrent_put_worker(
    root: str,
    key: str,
    worker_id: int,
    barrier,
    output,
) -> None:
    call_id = f"call-{worker_id}"
    cache = PromptResultCache(root=Path(root))
    result = _successful_result(
        text=f"result-{worker_id}",
        logical_call_id=call_id,
    )
    barrier.wait()
    provenance = cache.put(
        key,
        result=result,
        phase="worker",
        unit=f"unit-{worker_id}",
        logical_call_id=call_id,
    )
    output.put(
        {
            "provenance": provenance.model_dump(mode="json"),
            "stores": cache.counters()["stores"],
        }
    )


def test_key_covers_resolved_call_and_repeat_identity() -> None:
    base = prompt_cache_key(request=_request(), repeat_index=0)
    variants = [
        _request(model="different/model"),
        _request(prompt="different"),
        _request(temperature=1.0),
        _request(reasoning=ReasoningEffort.HIGH),
        _request(top_p=0.5),
        _request(token_limit=256),
    ]
    assert all(
        prompt_cache_key(request=variant, repeat_index=0) != base
        for variant in variants
    )
    assert prompt_cache_key(request=_request(), repeat_index=1) != base
    assert prompt_cache_key(request=_request(), repeat_index=0) == base
    assert len(base) == 64


def test_hit_preserves_original_entry_provenance_and_nulls_latency(
    tmp_path: Path,
) -> None:
    cache = PromptResultCache(root=tmp_path)
    request = _request()
    policy = s.build_transport_policy()
    original_transport = s.RecordingTransport(
        request=request,
        transport_policy=policy,
        outcomes=[s.response_outcome(text="stored")],
    )
    miss = _execute(
        cache,
        request=request,
        transport=original_transport,
        logical_call_id="original-call",
    )
    assert not miss.cache_hit
    assert miss.telemetry().latency_s == 0.5

    unused_transport = s.RecordingTransport(
        request=request,
        transport_policy=policy,
        outcomes=[s.response_outcome(text="must-not-run")],
    )
    hit = _execute(
        cache,
        request=request,
        transport=unused_transport,
        logical_call_id="reuse-call",
    )
    assert hit.cache_hit
    assert unused_transport.served == []
    assert hit.result.logical_call_id == "original-call"
    assert hit.provenance is not None
    assert hit.provenance.source_phase == "internal"
    assert hit.provenance.source_unit == "candidate-1"
    assert hit.provenance.source_logical_call_id == "original-call"
    assert hit.cache_marks().cache_source_call_id == "original-call"
    assert hit.telemetry().latency_s is None


def test_cache_disabled_is_byte_identical_and_creates_no_bytes(
    tmp_path: Path,
) -> None:
    request = _request()
    transport_policy = s.build_transport_policy()
    policy = s.build_execution_policy(
        transport_policy=transport_policy,
        max_attempts=1,
    )
    direct_transport = s.RecordingTransport(
        request=request,
        transport_policy=transport_policy,
        outcomes=[s.response_outcome(text="same")],
    )
    direct = run_provider_call(
        request=request,
        policy=policy,
        transport=direct_transport,
        logical_call_id="same-call",
        clock=s.FakeClock(),
        sleep=s.SleepRecorder(),
    )
    wrapped_transport = s.RecordingTransport(
        request=request,
        transport_policy=transport_policy,
        outcomes=[s.response_outcome(text="same")],
    )
    wrapped = execute_call(
        request=request,
        policy=policy,
        transport=wrapped_transport,
        logical_call_id="same-call",
        repeat_index=0,
        cache=None,
        phase="internal",
        unit="candidate-1",
        clock=s.FakeClock(),
        sleep=s.SleepRecorder(),
    )
    assert wrapped.result.to_stable_dict() == direct.to_stable_dict()
    assert not wrapped.cache_hit
    assert wrapped.provenance is None
    assert list(tmp_path.iterdir()) == []


def test_multiprocess_writers_preserve_one_original_and_restart(
    tmp_path: Path,
) -> None:
    """Peer processes contend on one key; exactly one original survives."""
    context = multiprocessing.get_context("fork")
    worker_count = 6
    barrier = context.Barrier(worker_count)
    output = context.Queue()
    key = prompt_cache_key(request=_request(), repeat_index=0)
    processes = [
        context.Process(
            target=_concurrent_put_worker,
            args=(str(tmp_path), key, worker_id, barrier, output),
        )
        for worker_id in range(worker_count)
    ]
    for process in processes:
        process.start()
    reports = [output.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    provenances = [report["provenance"] for report in reports]
    assert all(provenance == provenances[0] for provenance in provenances)
    assert sum(report["stores"] for report in reports) == 1

    restarted = PromptResultCache(root=tmp_path)
    found = restarted.get_result(key)
    assert found is not None
    result, provenance = found
    assert provenance.model_dump(mode="json") == provenances[0]
    assert result.logical_call_id == provenance.source_logical_call_id
    assert result.generation is not None
    winner_id = provenance.source_logical_call_id.removeprefix("call-")
    assert result.generation.text == f"result-{winner_id}"


def test_invalid_entry_is_a_loud_miss_and_atomic_repair(
    tmp_path: Path,
) -> None:
    logs: list[str] = []
    cache = PromptResultCache(root=tmp_path, log=logs.append)
    request = _request()
    key = prompt_cache_key(request=request, repeat_index=0)
    path = cache.store_dir / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": "foreign"}))
    try:
        cache.get_result(key)
        raise AssertionError("expected PromptCacheError")
    except PromptCacheError:
        pass

    transport_policy = s.build_transport_policy()
    transport = s.RecordingTransport(
        request=request,
        transport_policy=transport_policy,
        outcomes=[s.response_outcome(text="repaired")],
    )
    repaired = _execute(
        cache,
        request=request,
        transport=transport,
        logical_call_id="repair-call",
    )
    assert not repaired.cache_hit
    assert any("LOUD MISS" in message for message in logs)
    restarted = PromptResultCache(root=tmp_path)
    found = restarted.get_result(key)
    assert found is not None
    assert found[0].generation is not None
    assert found[0].generation.text == "repaired"
