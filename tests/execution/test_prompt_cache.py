"""Prompt-cache identity, durability, and cross-process single-flight."""

from __future__ import annotations

import json
import multiprocessing
import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest
from dr_providers import (
    FailureClass,
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ReasoningEffort,
    Transcript,
    openrouter_chat_config,
)

from tests.execution.storage_workers import (
    cache_request,
    execute_cache_worker,
)
from tests.provider import support as s
from whetstone.execution._file_lock import FileLock
from whetstone.execution.prompt_cache import (
    PromptCacheError,
    PromptResultCache,
    execute_call,
    prompt_cache_key,
)
from whetstone.provider.driver import run_provider_call
from whetstone.provider.policy import ProviderExecutionPolicy


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


def _execute(
    cache: PromptResultCache | None,
    *,
    request: ProviderCallRequest,
    transport: Callable,
    logical_call_id: str,
    repeat_index: int = 0,
    drive_ordinal: int = 0,
    policy: ProviderExecutionPolicy | None = None,
):
    return execute_call(
        request=request,
        policy=policy or s.build_execution_policy(max_attempts=1),
        transport=transport,
        logical_call_id=logical_call_id,
        repeat_index=repeat_index,
        drive_ordinal=drive_ordinal,
        cache=cache,
        phase="internal",
        unit="candidate-1",
        clock=s.FakeClock(),
        sleep=s.SleepRecorder(),
    )


def _start_cache_contenders(
    *,
    context,
    root: Path,
    invocation_path: Path,
    worker_count: int,
):
    barrier = context.Barrier(worker_count)
    output = context.Queue()
    processes = [
        context.Process(
            target=execute_cache_worker,
            args=(
                str(root),
                str(invocation_path),
                worker_id,
                barrier,
                output,
            ),
        )
        for worker_id in range(worker_count)
    ]
    for process in processes:
        process.start()
    reports = [output.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    return reports


def test_v2_key_pins_all_semantic_identity_components() -> None:
    request = _request()
    policy = s.build_execution_policy(max_attempts=1)
    base = prompt_cache_key(request, policy, 0, 0)
    variants = [
        _request(model="different/model"),
        _request(prompt="different"),
        _request(temperature=1.0),
        _request(reasoning=ReasoningEffort.HIGH),
        _request(top_p=0.5),
        _request(token_limit=256),
    ]
    assert all(
        prompt_cache_key(variant, policy, 0, 0) != base for variant in variants
    )
    assert (
        prompt_cache_key(
            request,
            s.build_execution_policy(max_attempts=2),
            0,
            0,
        )
        != base
    )
    assert prompt_cache_key(request, policy, 1, 0) != base
    assert prompt_cache_key(request, policy, 0, 1) != base
    assert prompt_cache_key(request, policy, 0, 0) == base
    assert (
        base
        == "45a8ae33314d1fb81405fd4df3aa629ecab49e3fe91167e713efb425d1d34928"
    )


@pytest.mark.parametrize(
    ("repeat_index", "drive_ordinal", "error"),
    [
        (-1, 0, ValueError),
        (0, -1, ValueError),
        (True, 0, TypeError),
        (0, True, TypeError),
    ],
)
def test_key_ordinals_are_explicit_non_negative_integers(
    repeat_index: int,
    drive_ordinal: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        prompt_cache_key(
            _request(),
            s.build_execution_policy(max_attempts=1),
            repeat_index,
            drive_ordinal,
        )


@pytest.mark.parametrize(
    "invalid_key",
    [
        "aa/../../../../escaped",
        "A" * 64,
        "a" * 63,
        "a" * 65,
        "a" * 31 + "/" + "a" * 32,
        "a" * 31 + "\\" + "a" * 32,
    ],
)
@pytest.mark.parametrize("operation", ["get_result", "_path_for"])
def test_public_cache_key_boundaries_reject_noncanonical_keys_before_io(
    tmp_path: Path,
    invalid_key: str,
    operation: str,
) -> None:
    cache = PromptResultCache(root=tmp_path)
    with pytest.raises(
        ValueError,
        match="exactly 64 lowercase hexadecimal characters",
    ):
        getattr(cache, operation)(invalid_key)
    assert not cache.store_dir.exists()


def test_hit_preserves_original_entry_provenance_and_nulls_latency(
    tmp_path: Path,
) -> None:
    cache = PromptResultCache(root=tmp_path)
    request = _request()
    transport_policy = s.build_transport_policy()
    original_transport = s.RecordingTransport(
        request=request,
        transport_policy=transport_policy,
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
        transport_policy=transport_policy,
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

    key = prompt_cache_key(
        request,
        s.build_execution_policy(max_attempts=1),
        0,
        0,
    )
    stored = json.loads(cache._path_for(key).read_text())
    assert set(stored) == {
        "schema",
        "key",
        "request_identity",
        "execution_policy_hash",
        "repeat_index",
        "drive_ordinal",
        "result_policy_hash",
        "publication_id",
        "provenance",
        "result",
    }
    assert stored["schema"] == "whetstone.execution.prompt_cache_entry/v2"
    assert stored["key"] == key
    assert stored["request_identity"] == request.identity_payload()
    assert stored["execution_policy_hash"] == stored["result_policy_hash"]
    assert stored["repeat_index"] == 0
    assert stored["drive_ordinal"] == 0


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
        drive_ordinal=0,
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


def test_six_processes_execute_one_paid_call_and_restart_stats(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    invocation_path = tmp_path / "invocations"
    reports = _start_cache_contenders(
        context=context,
        root=tmp_path,
        invocation_path=invocation_path,
        worker_count=6,
    )

    assert len(invocation_path.read_text().splitlines()) == 1
    assert all(report["result"] == reports[0]["result"] for report in reports)
    assert sum(not report["cache_hit"] for report in reports) == 1
    assert sum(report["cache_hit"] for report in reports) == 5
    assert PromptResultCache(root=tmp_path).counters() == {
        "hits": 5,
        "misses": 1,
        "stores": 1,
    }
    assert json.loads(
        PromptResultCache(root=tmp_path)._stats_path.read_text()
    ) == {
        "schema": "whetstone.execution.prompt_cache_stats/v1",
        "hits": 5,
        "misses": 1,
        "stores": 1,
        "inflight_publication_ids": [],
    }

    policy = s.build_execution_policy(max_attempts=1)
    key = prompt_cache_key(cache_request(), policy, 0, 0)
    found = PromptResultCache(root=tmp_path).get_result(key)
    assert found is not None
    result, provenance = found
    assert result.logical_call_id == provenance.source_logical_call_id
    assert result.model_dump(mode="json") == reports[0]["result"]


def test_policy_and_drive_ordinal_partition_cached_results(
    tmp_path: Path,
) -> None:
    cache = PromptResultCache(root=tmp_path)
    request = _request()
    transport_policy = s.build_transport_policy()
    base_policy = s.build_execution_policy(
        transport_policy=transport_policy,
        max_attempts=1,
    )
    changed_policy = s.build_execution_policy(
        transport_policy=transport_policy,
        max_attempts=2,
    )
    first = s.RecordingTransport(
        request=request,
        transport_policy=transport_policy,
        outcomes=[
            s.failure_outcome(
                failure_class=FailureClass.RATE_LIMITED,
                status_code=429,
            )
        ],
    )
    second = s.RecordingTransport(
        request=request,
        transport_policy=transport_policy,
        outcomes=[s.response_outcome(text="redrive")],
    )
    changed = s.RecordingTransport(
        request=request,
        transport_policy=transport_policy,
        outcomes=[s.response_outcome(text="changed-policy")],
    )

    drive_zero = _execute(
        cache,
        request=request,
        transport=first,
        logical_call_id="drive-zero",
        policy=base_policy,
        drive_ordinal=0,
    )
    drive_one = _execute(
        cache,
        request=request,
        transport=second,
        logical_call_id="drive-one",
        policy=base_policy,
        drive_ordinal=1,
    )
    policy_variant = _execute(
        cache,
        request=request,
        transport=changed,
        logical_call_id="changed-policy",
        policy=changed_policy,
        drive_ordinal=1,
    )

    assert not drive_zero.result.succeeded
    assert drive_one.result.succeeded
    assert policy_variant.result.succeeded
    assert len(first.served) == len(second.served) == len(changed.served) == 1
    assert cache.counters() == {"hits": 0, "misses": 3, "stores": 3}


def test_killed_single_flight_owner_releases_lock_for_waiter(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    invocation_path = tmp_path / "invocations"
    owner_output = context.Queue()
    owner_started = context.Event()
    owner = context.Process(
        target=execute_cache_worker,
        args=(
            str(tmp_path),
            str(invocation_path),
            0,
            None,
            owner_output,
        ),
        kwargs={
            "block": True,
            "started": owner_started,
        },
    )
    owner.start()
    assert owner_started.wait(timeout=10)

    waiter_output = context.Queue()
    waiter_attempted = context.Event()
    waiter_acquired = context.Event()
    waiter = context.Process(
        target=execute_cache_worker,
        args=(
            str(tmp_path),
            str(invocation_path),
            1,
            None,
            waiter_output,
        ),
        kwargs={
            "lock_attempted": waiter_attempted,
            "lock_acquired": waiter_acquired,
        },
    )
    waiter.start()
    try:
        assert waiter_attempted.wait(timeout=10)
        assert not waiter_acquired.is_set()
        owner.terminate()
        owner.join(timeout=10)
        assert waiter_acquired.wait(timeout=10)
        waiter.join(timeout=10)
    finally:
        for process in (owner, waiter):
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
    assert owner.exitcode is not None
    assert waiter.exitcode == 0
    report = waiter_output.get(timeout=5)
    assert not report["cache_hit"]
    assert PromptResultCache(root=tmp_path).counters() == {
        "hits": 0,
        "misses": 1,
        "stores": 1,
    }


def test_corrupt_entry_repair_is_single_flight_across_processes(
    tmp_path: Path,
) -> None:
    policy = s.build_execution_policy(max_attempts=1)
    key = prompt_cache_key(cache_request(), policy, 0, 0)
    cache = PromptResultCache(root=tmp_path)
    path = cache.store_dir / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": "foreign"}))

    reports = _start_cache_contenders(
        context=multiprocessing.get_context("spawn"),
        root=tmp_path,
        invocation_path=tmp_path / "invocations",
        worker_count=6,
    )
    assert all(report["result"] == reports[0]["result"] for report in reports)
    assert len((tmp_path / "invocations").read_text().splitlines()) == 1
    assert cache.counters() == {"hits": 5, "misses": 1, "stores": 1}
    assert cache.get_result(key) is not None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("request_identity", {"tampered": True}, "request identities"),
        ("execution_policy_hash", "0" * 64, "policy hashes"),
        ("key", "0" * 64, "provenance keys"),
    ],
)
def test_stored_entry_identity_mismatch_fails_loudly(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    cache = PromptResultCache(root=tmp_path)
    request = _request()
    policy = s.build_execution_policy(max_attempts=1)
    transport = s.RecordingTransport(
        request=request,
        transport_policy=policy.transport_policy,
        outcomes=[s.response_outcome(text="stored")],
    )
    _execute(
        cache,
        request=request,
        transport=transport,
        logical_call_id="stored-call",
        policy=policy,
    )
    key = prompt_cache_key(request, policy, 0, 0)
    path = cache.store_dir / key[:2] / f"{key}.json"
    body = json.loads(path.read_text())
    body[field] = value
    path.write_text(json.dumps(body))

    with pytest.raises(PromptCacheError, match=message):
        cache.get_result(key)


def test_atomic_publication_fsyncs_entry_and_stats_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []
    replaced_source_modes: list[int] = []
    real_replace = os.replace

    def replace_and_record_mode(source: Path, destination: Path) -> None:
        replaced_source_modes.append(stat.S_IMODE(source.stat().st_mode))
        real_replace(source, destination)

    monkeypatch.setattr(
        "whetstone.execution.prompt_cache.fsync_parent_directory",
        observed.append,
    )
    monkeypatch.setattr(
        "whetstone.execution.prompt_cache.os.replace",
        replace_and_record_mode,
    )
    request = _request()
    policy = s.build_execution_policy(max_attempts=1)
    transport = s.RecordingTransport(
        request=request,
        transport_policy=policy.transport_policy,
        outcomes=[s.response_outcome(text="stored")],
    )
    _execute(
        PromptResultCache(root=tmp_path),
        request=request,
        transport=transport,
        logical_call_id="stored-call",
        policy=policy,
    )
    key = prompt_cache_key(request, policy, 0, 0)
    assert PromptResultCache(root=tmp_path)._path_for(key) in observed
    assert PromptResultCache(root=tmp_path)._stats_path in observed
    assert replaced_source_modes
    assert set(replaced_source_modes) == {0o600}


def test_restart_reconciles_kill_after_durable_entry_publication(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=execute_cache_worker,
        args=(
            str(tmp_path),
            str(tmp_path / "invocations"),
            0,
            None,
            output,
        ),
        kwargs={"crash_after_publication": True},
    )
    process.start()
    process.join(timeout=20)
    assert process.exitcode == 86

    cache = PromptResultCache(root=tmp_path)
    key = prompt_cache_key(
        cache_request(),
        s.build_execution_policy(max_attempts=1),
        0,
        0,
    )
    entry_path = cache._path_for(key)
    pending_path = cache._pending_accounting_path_for(key)
    assert entry_path.exists()
    assert pending_path.exists()
    assert not cache._stats_path.exists()

    restarted = PromptResultCache(root=tmp_path)
    assert restarted.counters() == {"hits": 0, "misses": 1, "stores": 1}
    assert restarted.get_result(key) is not None
    assert not pending_path.exists()
    assert not cache._applied_accounting_path_for(key).exists()


def test_reconciliation_preserves_journal_and_reports_corrupt_entry(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=execute_cache_worker,
        args=(
            str(tmp_path),
            str(tmp_path / "invocations"),
            0,
            None,
            output,
        ),
        kwargs={"crash_after_publication": True},
    )
    process.start()
    process.join(timeout=20)
    assert process.exitcode == 86

    cache = PromptResultCache(root=tmp_path)
    key = prompt_cache_key(
        cache_request(),
        s.build_execution_policy(max_attempts=1),
        0,
        0,
    )
    cache._path_for(key).write_text('{"schema":"corrupt"}')
    pending_path = cache._pending_accounting_path_for(key)

    with pytest.raises(PromptCacheError, match="entry invalid"):
        cache.counters()
    assert pending_path.exists()


def test_corrupt_entry_is_quarantined_before_pending_publication(
    tmp_path: Path,
) -> None:
    cache = PromptResultCache(root=tmp_path)
    policy = s.build_execution_policy(max_attempts=1)
    key = prompt_cache_key(cache_request(), policy, 0, 0)
    entry_path = cache._path_for(key)
    corrupt_body = b'{"schema":"corrupt"}'
    entry_path.write_bytes(corrupt_body)

    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=execute_cache_worker,
        args=(
            str(tmp_path),
            str(tmp_path / "invocations"),
            0,
            None,
            output,
        ),
        kwargs={"crash_after_pending": True},
    )
    process.start()
    process.join(timeout=20)
    assert process.exitcode == 87

    pending_path = cache._pending_accounting_path_for(key)
    quarantined = list(entry_path.parent.glob(f".{entry_path.name}.corrupt.*"))
    assert pending_path.exists()
    assert not entry_path.exists()
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == corrupt_body

    restarted = PromptResultCache(root=tmp_path)
    assert restarted.counters() == {"hits": 0, "misses": 0, "stores": 0}
    assert not pending_path.exists()
    transport = s.RecordingTransport(
        request=cache_request(),
        transport_policy=policy.transport_policy,
        outcomes=[s.response_outcome(text="repaired")],
    )
    repaired = _execute(
        restarted,
        request=cache_request(),
        transport=transport,
        logical_call_id="repair-call",
        policy=policy,
    )
    assert repaired.result.generation is not None
    assert repaired.result.generation.text == "repaired"
    assert restarted.counters() == {"hits": 0, "misses": 1, "stores": 1}
    assert quarantined[0].read_bytes() == corrupt_body


def _assert_unchanged(path: Path, *, body: bytes, mode: int) -> None:
    assert path.read_bytes() == body
    assert stat.S_IMODE(path.stat().st_mode) == mode


def test_prompt_cache_directory_symlinks_do_not_escape(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir(mode=0o755)
    external.chmod(0o755)
    cache_root = tmp_path / "cache-root"
    cache_root.mkdir()
    (cache_root / "prompt_cache").symlink_to(
        external, target_is_directory=True
    )

    cache = PromptResultCache(root=cache_root)
    key = prompt_cache_key(
        cache_request(),
        s.build_execution_policy(max_attempts=1),
        0,
        0,
    )
    with pytest.raises(OSError):
        cache._path_for(key)
    assert stat.S_IMODE(external.stat().st_mode) == 0o755
    assert list(external.iterdir()) == []

    (cache_root / "prompt_cache").unlink()
    cache.store_dir.mkdir()
    shard = cache.store_dir / key[:2]
    shard.symlink_to(external, target_is_directory=True)
    with pytest.raises(OSError):
        cache._path_for(key)
    assert stat.S_IMODE(external.stat().st_mode) == 0o755
    assert list(external.iterdir()) == []


@pytest.mark.parametrize("storage_kind", ["lock", "entry", "stats", "journal"])
def test_prompt_cache_file_symlinks_do_not_touch_external_target(
    tmp_path: Path,
    storage_kind: str,
) -> None:
    cache = PromptResultCache(root=tmp_path / "cache-root")
    key = prompt_cache_key(
        cache_request(),
        s.build_execution_policy(max_attempts=1),
        0,
        0,
    )
    entry_path = cache._path_for(key)
    paths = {
        "lock": cache._lock_path_for(key),
        "entry": entry_path,
        "stats": cache._stats_path,
        "journal": cache._pending_accounting_path_for(key),
    }
    victim = tmp_path / f"{storage_kind}-victim"
    body = f"{storage_kind}-external".encode()
    victim.write_bytes(body)
    victim.chmod(0o644)
    paths[storage_kind].symlink_to(victim)

    with pytest.raises((OSError, PromptCacheError)):
        if storage_kind == "stats":
            cache.counters()
        elif storage_kind == "journal":
            cache.counters()
        else:
            cache.get_result(key)
    _assert_unchanged(victim, body=body, mode=0o644)


def test_file_lock_rejects_symlink_without_chmodding_target(
    tmp_path: Path,
) -> None:
    victim = tmp_path / "lock-victim"
    body = b"external-lock"
    victim.write_bytes(body)
    victim.chmod(0o644)
    lock_path = tmp_path / "owned.lock"
    lock_path.symlink_to(victim)

    with pytest.raises(OSError), FileLock(lock_path):
        pass
    _assert_unchanged(victim, body=body, mode=0o644)


def test_cache_files_and_directories_are_private_under_permissive_umask(
    tmp_path: Path,
) -> None:
    def assert_private_modes(cache: PromptResultCache) -> None:
        paths = [cache.store_dir, *cache.store_dir.rglob("*")]
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o700
            for path in paths
            if path.is_dir()
        )
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o600
            for path in paths
            if path.is_file()
        )

    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=execute_cache_worker,
        args=(
            str(tmp_path),
            str(tmp_path / "invocations"),
            0,
            None,
            output,
        ),
        kwargs={"umask_value": 0},
    )
    process.start()
    output.get(timeout=20)
    process.join(timeout=20)
    assert process.exitcode == 0

    cache = PromptResultCache(root=tmp_path)
    key = prompt_cache_key(
        cache_request(),
        s.build_execution_policy(max_attempts=1),
        0,
        0,
    )
    assert_private_modes(cache)

    for path in cache.store_dir.rglob("*"):
        path.chmod(0o777)
    cache.store_dir.chmod(0o777)

    assert cache.get_result(key) is not None
    assert cache.counters() == {"hits": 0, "misses": 1, "stores": 1}
    assert_private_modes(cache)
