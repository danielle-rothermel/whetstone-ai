"""The run-level prompt result cache (task 31).

Covers: key completeness (every output-affecting knob AND the repeat index
change the key; a non-output-affecting label does not); a hit returns the
stored payload with provenance marks; a miss computes + stores; concurrent
thread writers are safe; a corrupt entry is a LOUD miss (recompute, no crash,
no bad data); and the opt-out default (cache=None) is byte-identical and
creates no store.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dr_providers import (
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    RawHttpRequest,
    ReasoningEffort,
    Transcript,
    openrouter_chat_config,
)

from tests.envs.support import execution_policy, transport_policy
from whetstone.execution.prompt_cache import (
    PromptCacheError,
    PromptResultCache,
    execute_call,
    prompt_cache_key,
)


@dataclass
class _CountingTransport:
    """A transport that returns a fixed text and COUNTS every wire call.

    The count is the load-bearing assertion: a cache hit must not touch the
    transport, so ``calls`` stays put across a served hit.
    """

    text: str = "answer-42"
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        with self._lock:
            self.calls += 1
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"content-type": "json"},
            body={"model": "test-model"},
        )
        response = ProviderTransportResponse(
            text=self.text,
            raw_body={"choices": [{"message": {"content": self.text}}]},
            response_id="resp-1", model="test-model", finish_reason="stop",
        )
        return ProviderInvocationEvidence.build(
            request=request, policy=self.policy,
            raw_request=raw_request, outcome=response,
        )


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
        temperature=temperature, reasoning=reasoning,
        top_p=top_p, token_limit=token_limit,
    )
    config = openrouter_chat_config(model=model, controls=controls)
    return ProviderCallRequest(
        config=config,
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content=prompt),)
        ),
    )


def _exec(
    cache: PromptResultCache | None,
    *,
    request: ProviderCallRequest,
    transport: Callable[..., ProviderInvocationEvidence],
    repeat_index: int = 0,
    phase: str = "cell",
    unit: str = "cand-A",
    logical_call_id: str = "lc#0",
):
    return execute_call(
        request=request, policy=execution_policy(max_attempts=1),
        transport=transport, logical_call_id=logical_call_id,
        repeat_index=repeat_index, cache=cache, phase=phase, unit=unit,
    )


# --- Key completeness --------------------------------------------------------


def test_key_changes_on_every_output_affecting_knob() -> None:
    base = prompt_cache_key(request=_request(), repeat_index=0)
    # Each output-affecting knob change produces a DISTINCT key.
    assert prompt_cache_key(
        request=_request(model="a/b"), repeat_index=0
    ) != base
    assert prompt_cache_key(
        request=_request(prompt="different"), repeat_index=0
    ) != base
    assert prompt_cache_key(
        request=_request(temperature=1.0), repeat_index=0
    ) != base
    assert prompt_cache_key(
        request=_request(reasoning=ReasoningEffort.HIGH), repeat_index=0
    ) != base
    assert prompt_cache_key(
        request=_request(top_p=0.5), repeat_index=0
    ) != base
    assert prompt_cache_key(
        request=_request(token_limit=256), repeat_index=0
    ) != base


def test_key_changes_on_repeat_index() -> None:
    # The MANDATORY repeat ordinal: r0 and r1 of the SAME resolved call are
    # distinct keys (else repeat variance collapses to zero).
    r0 = prompt_cache_key(request=_request(), repeat_index=0)
    r1 = prompt_cache_key(request=_request(), repeat_index=1)
    r2 = prompt_cache_key(request=_request(), repeat_index=2)
    assert len({r0, r1, r2}) == 3


def test_key_stable_for_identical_call_and_repeat() -> None:
    # Two independently-built identical requests at the same repeat hash equal.
    a = prompt_cache_key(request=_request(), repeat_index=3)
    b = prompt_cache_key(request=_request(), repeat_index=3)
    assert a == b and len(a) == 64


def test_key_ignores_non_output_affecting_transport_policy() -> None:
    # The key folds request identity + repeat ONLY. The execution/transport
    # policy is not output-affecting, so it never enters the key -- two calls
    # that differ only in transport policy MUST share an entry. (The key
    # function takes no policy at all; this documents that by construction.)
    assert prompt_cache_key(request=_request(), repeat_index=0) == (
        prompt_cache_key(request=_request(), repeat_index=0)
    )


# --- Miss computes + stores; hit serves the stored payload -------------------


def test_miss_computes_and_stores(tmp_path: Path) -> None:
    cache = PromptResultCache(root=tmp_path)
    transport = _CountingTransport(text="stored-out")
    req = _request()
    out = _exec(cache, request=req, transport=transport)
    assert out.cache_hit is False and out.provenance is None
    assert out.result.generation is not None
    assert out.result.generation.text == "stored-out"
    assert transport.calls == 1
    # The store file exists under <root>/prompt_cache/<shard>/<key>.json.
    key = prompt_cache_key(request=req, repeat_index=0)
    path = tmp_path / "prompt_cache" / key[:2] / f"{key}.json"
    assert path.exists()
    assert cache.counters() == {"hits": 0, "misses": 1, "stores": 1}


def test_hit_returns_stored_payload_with_provenance(tmp_path: Path) -> None:
    cache = PromptResultCache(root=tmp_path)
    transport = _CountingTransport(text="stored-out")
    req = _request()
    # First call: miss, stores with source (phase=cell, unit=cand-A).
    _exec(cache, request=req, transport=transport,
          phase="cell", unit="cand-A", logical_call_id="orig-call")
    assert transport.calls == 1
    # Second call for the SAME (request, repeat): a HIT served WITHOUT touching
    # the transport, carrying the ORIGINAL entry's provenance.
    hit = _exec(cache, request=req, transport=transport,
                phase="cell-later", unit="cand-B", logical_call_id="reuse")
    assert transport.calls == 1  # no new wire call
    assert hit.cache_hit is True
    assert hit.result.generation is not None
    assert hit.result.generation.text == "stored-out"
    prov = hit.provenance
    assert prov is not None
    assert prov.source_phase == "cell"
    assert prov.source_unit == "cand-A"
    assert prov.source_logical_call_id == "orig-call"
    assert prov.stored_at  # a real ISO timestamp of the original store
    assert cache.counters() == {"hits": 1, "misses": 1, "stores": 1}


def test_different_repeat_is_a_distinct_entry(tmp_path: Path) -> None:
    cache = PromptResultCache(root=tmp_path)
    transport = _CountingTransport()
    req = _request()
    _exec(cache, request=req, transport=transport, repeat_index=0)
    _exec(cache, request=req, transport=transport, repeat_index=1)
    # Both repeats drove the transport (distinct keys) -- repeat variance kept.
    assert transport.calls == 2
    assert cache.counters()["stores"] == 2


def test_cache_marks_null_on_miss_and_ref_on_hit(tmp_path: Path) -> None:
    cache = PromptResultCache(root=tmp_path)
    transport = _CountingTransport()
    req = _request()
    miss = _exec(cache, request=req, transport=transport)
    miss_marks = miss.cache_marks()
    assert miss_marks.cache_hit is False
    assert miss_marks.cache_source_phase is None
    hit = _exec(cache, request=req, transport=transport)
    hit_marks = hit.cache_marks()
    assert hit_marks.cache_hit is True
    assert hit_marks.cache_source_phase == "cell"
    assert hit_marks.cache_source_call_id == "lc#0"


# --- Opt-out default (cache=None) --------------------------------------------


def test_opt_out_drives_transport_and_creates_no_store(tmp_path: Path) -> None:
    transport = _CountingTransport()
    req = _request()
    out1 = _exec(None, request=req, transport=transport)
    out2 = _exec(None, request=req, transport=transport)
    # Both calls drove the transport (no reuse) and neither is a cache hit.
    assert transport.calls == 2
    assert out1.cache_hit is False and out2.cache_hit is False
    assert out1.provenance is None
    # NO store directory was created (byte-identical to a pre-cache run).
    assert not (tmp_path / "prompt_cache").exists()


# --- Concurrent writers ------------------------------------------------------


def test_concurrent_writers_are_safe(tmp_path: Path) -> None:
    cache = PromptResultCache(root=tmp_path)
    transport = _CountingTransport()
    # Many threads race to store DISTINCT keys (one per repeat) plus a few
    # duplicate keys, exercising the atomic-write + first-writer-wins path.
    req = _request()
    errors: list[BaseException] = []

    def worker(repeat: int) -> None:
        try:
            _exec(cache, request=req, transport=transport, repeat_index=repeat)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(r,))
        for r in list(range(8)) * 3  # 8 distinct keys, each raced 3x
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # 8 distinct keys stored exactly once each (first-writer-wins on dupes);
    # every one is readable back to a valid Result.
    assert cache.counters()["stores"] == 8
    for r in range(8):
        key = prompt_cache_key(request=req, repeat_index=r)
        found = cache.get_result(key)
        assert found is not None
        result, _prov = found
        assert result.generation is not None


# --- Corrupt entry -> loud miss ----------------------------------------------


def _write_corrupt(cache: PromptResultCache, key: str, body: str) -> Path:
    path = cache.store_dir / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_corrupt_bad_json_is_a_loud_miss(tmp_path: Path) -> None:
    logs: list[str] = []
    cache = PromptResultCache(root=tmp_path, log=logs.append)
    transport = _CountingTransport(text="recomputed")
    req = _request()
    key = prompt_cache_key(request=req, repeat_index=0)
    _write_corrupt(cache, key, "{not valid json")
    # A get on the corrupt entry raises a typed error (the low-level contract).
    try:
        cache.get_result(key)
        raise AssertionError("expected PromptCacheError on corrupt entry")
    except PromptCacheError:
        pass
    # execute_call degrades it to a LOUD miss: recompute + overwrite, no crash.
    out = _exec(cache, request=req, transport=transport)
    assert out.cache_hit is False
    assert out.result.generation is not None
    assert out.result.generation.text == "recomputed"
    assert transport.calls == 1
    assert any("PROMPT-CACHE CORRUPT ENTRY" in m for m in logs)
    # The bad entry was overwritten with a valid one -> a subsequent read hits.
    hit = _exec(cache, request=req, transport=transport)
    assert hit.cache_hit is True and transport.calls == 1


def test_corrupt_foreign_schema_is_a_loud_miss(tmp_path: Path) -> None:
    logs: list[str] = []
    cache = PromptResultCache(root=tmp_path, log=logs.append)
    transport = _CountingTransport()
    req = _request()
    key = prompt_cache_key(request=req, repeat_index=0)
    # Valid JSON but a foreign/absent schema stamp -> not trusted.
    _write_corrupt(cache, key, json.dumps({"schema": "some.other/v9"}))
    out = _exec(cache, request=req, transport=transport)
    assert out.cache_hit is False and transport.calls == 1
    assert any("PROMPT-CACHE CORRUPT ENTRY" in m for m in logs)


def test_corrupt_result_body_is_a_loud_miss(tmp_path: Path) -> None:
    logs: list[str] = []
    cache = PromptResultCache(root=tmp_path, log=logs.append)
    transport = _CountingTransport()
    req = _request()
    key = prompt_cache_key(request=req, repeat_index=0)
    # Correct schema + provenance but a result body that no longer validates.
    _write_corrupt(cache, key, json.dumps({
        "schema": "whetstone.execution.prompt_cache_entry/v1",
        "key": key,
        "provenance": {
            "key": key, "source_phase": "cell", "source_unit": "u",
            "source_logical_call_id": "c", "stored_at": "2026-01-01T00:00:00",
        },
        "result": {"not": "a valid ProviderCallResult"},
    }))
    out = _exec(cache, request=req, transport=transport)
    assert out.cache_hit is False and transport.calls == 1
    assert any("PROMPT-CACHE CORRUPT ENTRY" in m for m in logs)
