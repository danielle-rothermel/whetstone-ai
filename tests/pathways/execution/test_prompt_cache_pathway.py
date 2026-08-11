"""Multiprocess prompt-cache pathway tests."""

from __future__ import annotations

import json
import multiprocessing
import stat
from collections.abc import Callable
from pathlib import Path

import pytest
from dr_providers import (
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
    recover_cache_worker,
)
from tests.optimization.processes import join_processes, terminate_processes
from tests.provider import support as s
from whetstone.execution.prompt_cache import (
    PromptCacheError,
    PromptResultCache,
    execute_call,
    prompt_cache_key,
)
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
    sample_index: int = 0,
    drive_ordinal: int = 0,
    policy: ProviderExecutionPolicy | None = None,
):
    return execute_call(
        request=request,
        policy=policy or s.build_execution_policy(max_attempts=1),
        transport=transport,
        logical_call_id=logical_call_id,
        sample_index=sample_index,
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
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        reports = [output.get(timeout=20) for _ in processes]
        join_processes(started, timeout=20)
        return reports
    finally:
        barrier.abort()
        terminate_processes(started, timeout=20)


def _run_expected_cache_crash(
    *,
    root: Path,
    expected_exitcode: int,
    **crash_window: bool,
) -> None:
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=execute_cache_worker,
        args=(
            str(root),
            str(root / "invocations"),
            0,
            None,
            output,
        ),
        kwargs=crash_window,
    )
    started = []
    try:
        process.start()
        started.append(process)
        process.join(timeout=20)
        assert process.exitcode == expected_exitcode
    finally:
        terminate_processes(started, timeout=20)


def _recover_cache_in_fresh_process(
    *,
    root: Path,
    key: str,
) -> dict[str, object]:
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=recover_cache_worker,
        args=(str(root), key, output),
    )
    started = []
    try:
        process.start()
        started.append(process)
        report = output.get(timeout=20)
        join_processes(started, timeout=20)
        return report
    finally:
        terminate_processes(started, timeout=20)


@pytest.mark.process_integration
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


@pytest.mark.process_integration
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
    started = []
    try:
        owner.start()
        started.append(owner)
        assert owner_started.wait(timeout=10)
        waiter.start()
        started.append(waiter)
        assert waiter_attempted.wait(timeout=10)
        assert not waiter_acquired.is_set()
        owner.terminate()
        owner.join(timeout=10)
        assert waiter_acquired.wait(timeout=10)
        waiter.join(timeout=10)
        assert owner.exitcode is not None
        assert waiter.exitcode == 0
        report = waiter_output.get(timeout=5)
        assert not report["cache_hit"]
    finally:
        terminate_processes(started, timeout=10)
    assert PromptResultCache(root=tmp_path).counters() == {
        "hits": 0,
        "misses": 1,
        "stores": 1,
    }


@pytest.mark.process_integration
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


@pytest.mark.process_integration
@pytest.mark.parametrize(
    "stage",
    (
        "after_publication",
        "after_stats_write",
        "after_applied_rename",
    ),
)
def test_restart_reconciles_kill_after_crash_window(
    tmp_path: Path,
    stage: str,
) -> None:
    policy = s.build_execution_policy(max_attempts=1)
    key = prompt_cache_key(cache_request(), policy, 0, 0)

    if stage == "after_publication":
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
        started = []
        try:
            process.start()
            started.append(process)
            process.join(timeout=20)
            assert process.exitcode == 86
        finally:
            terminate_processes(started, timeout=20)

        cache = PromptResultCache(root=tmp_path)
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
        return

    crash_kwargs = (
        {"crash_after_stats_write": True}
        if stage == "after_stats_write"
        else {"crash_after_applied_rename": True}
    )
    expected_exitcode = 88 if stage == "after_stats_write" else 89
    _run_expected_cache_crash(
        root=tmp_path,
        expected_exitcode=expected_exitcode,
        **crash_kwargs,
    )

    cache = PromptResultCache(root=tmp_path)
    pending_path = cache._pending_accounting_path_for(key)
    applied_path = cache._applied_accounting_path_for(key)
    stats = json.loads(cache._stats_path.read_text())
    assert cache._path_for(key).exists()
    assert stats["hits"] == 0
    assert stats["misses"] == 1
    assert stats["stores"] == 1
    assert len(stats["inflight_publication_ids"]) == 1

    if stage == "after_stats_write":
        assert pending_path.exists()
        assert not applied_path.exists()
        expected_recovery = {
            "entry_readable": True,
            "counters": {"hits": 0, "misses": 1, "stores": 1},
            "inflight_publication_ids": [],
            "pending_exists": False,
            "applied_exists": False,
        }
    else:
        assert not pending_path.exists()
        assert applied_path.exists()
        expected_recovery = {
            "entry_readable": True,
            "counters": {"hits": 0, "misses": 1, "stores": 1},
            "inflight_publication_ids": [],
            "pending_exists": False,
            "applied_exists": False,
        }

    assert _recover_cache_in_fresh_process(root=tmp_path, key=key) == (
        expected_recovery
    )


@pytest.mark.process_integration
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
    started = []
    try:
        process.start()
        started.append(process)
        process.join(timeout=20)
        assert process.exitcode == 86
    finally:
        terminate_processes(started, timeout=20)

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


@pytest.mark.process_integration
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
    started = []
    try:
        process.start()
        started.append(process)
        process.join(timeout=20)
        assert process.exitcode == 87
    finally:
        terminate_processes(started, timeout=20)

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
    assert repaired.result.provider_generation is not None
    assert repaired.result.provider_generation.text == "repaired"
    assert restarted.counters() == {"hits": 0, "misses": 1, "stores": 1}
    assert quarantined[0].read_bytes() == corrupt_body


@pytest.mark.process_integration
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
    started = []
    try:
        process.start()
        started.append(process)
        output.get(timeout=20)
        join_processes(started, timeout=20)
    finally:
        terminate_processes(started, timeout=20)

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
