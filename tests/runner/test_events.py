"""Push-based run telemetry: the ``logs/events.jsonl`` event stream (task 24).

Two layers of coverage:

* the ``events`` MODULE in isolation -- the typed envelope (schema tag, real
  timestamp, structured identity components), the loud markers, null-not-zero
  truthfulness, thread-safe concurrent appends, and the traceback boundary;
* the WIRING through ``run_cell`` -- a finalized cell pushes ``cell_finalized``
  with REALIZED (not estimated) spend; a re-run of a completed attempt pushes
  the previously-SILENT ``attempt_skipped`` (the c18 collision class); an
  incomplete arm pushes ``arm_incomplete``; a zero-success baseline pushes a
  TYPED ``cell_failed``; a rate-limited window pushes ``rate_limit_pressure`` +
  a ``latency_snapshot``.

Every drive is against injected fake transports -- no live call.
"""

from __future__ import annotations

import threading
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
from whetstone.envs.registry import env_spec
from whetstone.envs.rollout_definition import initial_candidate, render_prompt
from whetstone.runner.cell import CellBaselineFailure, CellConfig, run_cell
from whetstone.runner.events import (
    ARM_INCOMPLETE,
    ATTEMPT_SKIPPED,
    CELL_FAILED,
    CELL_FINALIZED,
    EVENT_MARKERS,
    EVENTS_SCHEMA,
    LATENCY_SNAPSHOT,
    RATE_LIMIT_PRESSURE,
    SCREEN_KEY_LOCKED,
    TRACEBACK,
    EventStream,
    EventUnit,
    cell_finalized_event,
    emit_traceback_on_unhandled,
    latency_snapshot_event,
    rate_limit_pressure_event,
    screen_key_locked_event,
)
from whetstone.runner.execution_mode import ExecutionMode
from whetstone.runner.ledger import Ledger

from .support import (
    PROPOSER_MODEL,
    SPLIT,
    TASK_MODEL,
    FailingTransport,
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


def _config(
    env: str,
    *,
    optimizer: str,
    transport,
    proposer_transport,
    lane: str = "openrouter",
) -> CellConfig:
    return CellConfig(
        optimizer=optimizer, env=env, lane=lane, attempt=0,
        task_model=TASK_MODEL, proposer_model=PROPOSER_MODEL, canonical=True,
        proposer_config=proposer_config(),
        proposer_transport=proposer_transport,
        rollout_transport=transport,
        execution_policy=runner_execution_policy(),
        repeats=3, pool_n_per_stratum=_pool_n(env), split_sizes=SPLIT,
        execution_mode=ExecutionMode.IN_PROCESS, max_wall_seconds=10_000.0,
    )


def _unit() -> EventUnit:
    return EventUnit.for_cell(
        cell_id="eval:c11:a0", env="c11", optimizer="eval", attempt=0,
        lane="openrouter", model=TASK_MODEL,
    )


# --------------------------------------------------------------------------
# Module-level: the typed envelope, markers, truthfulness, concurrency.
# --------------------------------------------------------------------------


def test_event_envelope_carries_schema_timestamp_and_components() -> None:
    e = rate_limit_pressure_event(
        unit=_unit(), rate_limit_rows=3, concurrency_halved=True,
        guard_timeouts=1, window_label="official_naive",
        at="2026-07-23T00:00:00+00:00",
    )
    # The schema tag is stamped and serializes as the ``schema`` wire key.
    assert e.schema_ == EVENTS_SCHEMA
    line = e.to_line()
    assert '"schema": "whetstone.runner.events/v1"' in line
    # A real ISO-8601 timestamp -- never the empty-string precedent to avoid.
    assert e.at == "2026-07-23T00:00:00+00:00"
    assert e.at != ""
    # Structured identity components are SEPARATE fields (not only a composite
    # id) so a reader never has to parse ``opt:env:aN``.
    assert e.unit.env == "c11"
    assert e.unit.optimizer == "eval"
    assert e.unit.attempt == 0
    assert e.unit.lane == "openrouter"
    assert e.unit.model == TASK_MODEL
    assert e.unit.cell_id == "eval:c11:a0"


def test_event_roundtrips_through_the_wire_form() -> None:
    from whetstone.runner.events import RunEvent

    e = cell_finalized_event(
        unit=_unit(), status="improved", delta=1.0, delta_ci95=(0.5, 1.0),
        realized_spend_usd=0.5, duration_s=12.0,
        at="2026-07-23T00:00:00+00:00",
    )
    assert RunEvent.from_line(e.to_line()) == e


def test_markers_cover_every_event_and_preserve_grep_signatures() -> None:
    # Every event type has a loud, unique marker; the set preserves the
    # 429|rate.limit|halved|RATE-LIMIT|Traceback watcher grammar + extends it.
    assert set(EVENT_MARKERS) == {
        RATE_LIMIT_PRESSURE, ATTEMPT_SKIPPED, CELL_FINALIZED, CELL_FAILED,
        ARM_INCOMPLETE, SCREEN_KEY_LOCKED, LATENCY_SNAPSHOT, TRACEBACK,
    }
    assert len(set(EVENT_MARKERS.values())) == len(EVENT_MARKERS)
    assert EVENT_MARKERS[RATE_LIMIT_PRESSURE] == "RATE-LIMIT PRESSURE"
    assert EVENT_MARKERS[TRACEBACK] == "TRACEBACK"
    assert EVENT_MARKERS[SCREEN_KEY_LOCKED] == "SCREEN-KEY-LOCKED"
    # The marker line is a single greppable token + id + sorted key=val fields.
    e = rate_limit_pressure_event(
        unit=_unit(), rate_limit_rows=3, concurrency_halved=True,
        guard_timeouts=0, window_label="official_naive",
    )
    marker = e.marker_line()
    assert marker.startswith("RATE-LIMIT PRESSURE eval:c11:a0 ")
    assert "rate_limit_rows=3" in marker


def test_screen_key_locked_event_names_key_and_lock() -> None:
    # The refuse-to-start event carries the contested key + lock path and fires
    # the loud SCREEN-KEY-LOCKED grep marker.
    e = screen_key_locked_event(
        unit=EventUnit(screen_id="screen:m_enone", model="m"),
        screen_key="screen:m_enone",
        lock_path="/root/task_screen/ed1_m_enone.lock",
    )
    assert e.event == SCREEN_KEY_LOCKED
    assert e.marker == "SCREEN-KEY-LOCKED"
    assert e.fields["screen_key"] == "screen:m_enone"
    assert e.fields["lock_path"] == "/root/task_screen/ed1_m_enone.lock"
    assert e.marker_line().startswith("SCREEN-KEY-LOCKED screen:m_enone ")


def test_latency_snapshot_is_null_not_zero_when_unknown() -> None:
    e = latency_snapshot_event(
        unit=_unit(), median_latency_s=None, coverage=0,
        window_label="official_naive",
    )
    # An unknown median is None (never a fake 0.0), with coverage 0.
    assert e.fields["median_latency_s"] is None
    assert e.fields["coverage"] == 0


def test_emit_writes_jsonl_and_mirrors_loud_marker(tmp_path: Path) -> None:
    markers: list[str] = []
    stream = EventStream(root=tmp_path, marker_sink=markers.append)
    e = rate_limit_pressure_event(
        unit=_unit(), rate_limit_rows=2, concurrency_halved=False,
        guard_timeouts=0, window_label="w",
    )
    stream.emit(e)
    # The JSONL line lands at <root>/logs/events.jsonl and reloads.
    assert stream.path == tmp_path / "logs" / "events.jsonl"
    loaded = stream.load()
    assert len(loaded) == 1 and loaded[0].event == RATE_LIMIT_PRESSURE
    # The loud marker fired alongside.
    assert any(m.startswith("RATE-LIMIT PRESSURE") for m in markers)


def test_emit_is_concurrent_writer_safe(tmp_path: Path) -> None:
    # Many threads append single lines; none interleaves a half-written line,
    # so every line reloads cleanly and the count is exact.
    stream = EventStream(root=tmp_path, marker_sink=lambda _m: None)

    def _work(i: int) -> None:
        stream.emit(
            latency_snapshot_event(
                unit=_unit(), median_latency_s=float(i), coverage=1,
                window_label=f"w{i}",
            )
        )

    threads = [threading.Thread(target=_work, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(stream.load()) == 50


def test_emit_never_raises_on_a_broken_stream(tmp_path: Path) -> None:
    # A write failure (the logs path is a FILE, so mkdir of its parent-as-dir
    # fails) is swallowed loudly -- a broken stream can NEVER break a run.
    (tmp_path / "logs").write_text("i am a file, not a dir")
    markers: list[str] = []
    stream = EventStream(root=tmp_path, marker_sink=markers.append)
    # Must not raise.
    stream.emit(
        rate_limit_pressure_event(
            unit=_unit(), rate_limit_rows=1, concurrency_halved=False,
            guard_timeouts=0, window_label="w",
        )
    )
    # The loud marker still fired (independent of the JSONL write) and the
    # write failure is reported.
    assert any(m.startswith("RATE-LIMIT PRESSURE") for m in markers)
    assert any("EVENT-STREAM-WRITE-FAILED" in m for m in markers)


def test_traceback_boundary_emits_then_reraises(tmp_path: Path) -> None:
    stream = EventStream(root=tmp_path, marker_sink=lambda _m: None)
    with pytest.raises(ValueError, match="boom"):
        with emit_traceback_on_unhandled(stream, unit=_unit()):
            raise ValueError("boom")
    events = stream.load()
    assert len(events) == 1
    ev = events[0]
    assert ev.event == TRACEBACK
    assert ev.fields["exc_type"] == "ValueError"
    # The full formatted traceback is carried (preserving the Traceback grep
    # signature in the structured field).
    assert "Traceback" in ev.fields["traceback"]


def test_traceback_boundary_passes_named_types_through_silently(
    tmp_path: Path,
) -> None:
    stream = EventStream(root=tmp_path, marker_sink=lambda _m: None)
    # A named (already-handled) failure re-raises WITHOUT emitting a traceback.
    with pytest.raises(KeyError):
        with emit_traceback_on_unhandled(
            stream, unit=_unit(), reraise=(KeyError,)
        ):
            raise KeyError("handled elsewhere")
    assert stream.load() == []


# --------------------------------------------------------------------------
# Wiring through run_cell.
# --------------------------------------------------------------------------


def test_cell_finalized_event_carries_realized_spend(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    stream = EventStream(root=tmp_path, marker_sink=lambda _m: None)
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        _config(
            env, optimizer="copro",
            transport=FakeTransport(reply=improvement_reply(exp, WIN)),
            proposer_transport=ScriptedProposer((WIN,)),
        ),
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
        events=stream,
    )
    finalized = [e for e in stream.load() if e.event == CELL_FINALIZED]
    assert len(finalized) == 1
    ev = finalized[0]
    assert ev.fields["status"] == outcome.record.status == "improved"
    assert ev.fields["delta"] == pytest.approx(1.0)
    # REALIZED spend (the cell's own credits-delta), NOT a heartbeat estimate.
    assert ev.fields["realized_spend_usd"] == pytest.approx(
        outcome.record.spend_usd
    )
    assert "realized_spend_usd" in ev.fields
    assert "spend_estimate_usd" not in ev.fields
    # Latency snapshots were pushed per official window (FakeTransport logs a
    # real driver-clock latency, so coverage > 0 -- not a null-latency stream).
    lat = [e for e in stream.load() if e.event == LATENCY_SNAPSHOT]
    assert lat and any(e.fields["coverage"] > 0 for e in lat)


def test_attempt_skipped_event_is_emitted_on_a_completed_rerun(
    tmp_path: Path,
) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    ledger = Ledger(root=tmp_path)
    cfg = _config(
        env, optimizer="copro",
        transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        proposer_transport=ScriptedProposer((WIN,)),
    )
    # First run completes the cell (no events stream needed).
    run_cell(
        cfg, ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    # A SECOND run of the same completed (opt, env, attempt) SKIPS -- was
    # silent; now it pushes a loud attempt_skipped naming the prior status.
    markers: list[str] = []
    stream = EventStream(root=tmp_path, marker_sink=markers.append)
    outcome = run_cell(cfg, ledger=ledger, events=stream)
    assert outcome.skipped is True
    skipped = [e for e in stream.load() if e.event == ATTEMPT_SKIPPED]
    assert len(skipped) == 1
    assert skipped[0].fields["prior_status"] == "improved"
    assert skipped[0].unit.env == "c11"
    assert any(m.startswith("ATTEMPT-SKIPPED") for m in markers)


def test_cell_failed_event_is_typed(tmp_path: Path) -> None:
    # A zero-successful-rollout baseline raises CellBaselineFailure; a TYPED
    # cell_failed event is pushed BEFORE the raise (reason_class, not a bare
    # string dump).
    env = "c11"
    stream = EventStream(root=tmp_path, marker_sink=lambda _m: None)
    ledger = Ledger(root=tmp_path)
    with pytest.raises(CellBaselineFailure):
        run_cell(
            _config(
                env, optimizer="copro", transport=FailingTransport(),
                proposer_transport=ScriptedProposer((WIN,)),
            ),
            ledger=ledger,
            credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.0)]),
            events=stream,
        )
    failed = [e for e in stream.load() if e.event == CELL_FAILED]
    assert len(failed) == 1
    assert failed[0].fields["reason_class"] == "CellBaselineFailure"
    # No cell line was recorded (baseline failure is a hard plumbing block).
    assert ledger.cells() == []


def test_rate_limit_pressure_and_arm_incomplete_events(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    official = exp.eval_configs.official.instances
    naive = initial_candidate(env_spec(env))
    fail_prompts = frozenset(
        render_prompt(env_spec(env), naive, inst) for inst in official[:1]
    )
    transport = _RateLimitMatchingPrompts(
        should_fail=lambda p: p in fail_prompts, reply=correct_reply(exp)
    )
    stream = EventStream(root=tmp_path, marker_sink=lambda _m: None)
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        _config(
            env, optimizer="eval", transport=transport,
            proposer_transport=ScriptedProposer((WIN,)),
        ),
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 617.0)]),
        events=stream,
    )
    # The naive arm's failed task leaves that arm incomplete -> arm_incomplete
    # (NOT a certified cell_finalized).
    assert outcome.record.status == "incomplete-arm"
    events = stream.load()
    assert [e for e in events if e.event == ARM_INCOMPLETE]
    assert not [e for e in events if e.event == CELL_FINALIZED]
    # The rate-limited rows were surfaced as a rate_limit_pressure event with a
    # nonzero row count (the original task-24 core).
    pressure = [e for e in events if e.event == RATE_LIMIT_PRESSURE]
    assert pressure
    assert any(e.fields["rate_limit_rows"] > 0 for e in pressure)


def test_pilot_pushes_latency_snapshot(tmp_path: Path) -> None:
    # The pilot finalization path pushes a screen-keyed latency_snapshot. Pilot
    # call records carry no per-call latency, so the snapshot is honestly
    # null-coverage (median None, coverage 0) -- null-not-zero.
    from whetstone.runner.pilot import run_pilot

    env = "c11"
    exp = tiny_experiment(env)
    stream = EventStream(root=tmp_path, marker_sink=lambda _m: None)
    run_pilot(
        env=env, lane="openrouter", model=TASK_MODEL,
        transport=FakeTransport(reply=correct_reply(exp)),
        execution_policy=runner_execution_policy(),
        instance_count=2, pool_n_per_stratum=_pool_n(env), split_sizes=SPLIT,
        spec_estimate_tokens=100, events=stream,
    )
    lat = [e for e in stream.load() if e.event == LATENCY_SNAPSHOT]
    assert len(lat) == 1
    ev = lat[0]
    assert ev.unit.screen_id == f"pilot:{env}:{TASK_MODEL}"
    assert ev.unit.env == env
    assert ev.fields["median_latency_s"] is None
    assert ev.fields["coverage"] == 0
    # A clean pilot saw no rate limiting -> no rate_limit_pressure event.
    assert not [e for e in stream.load() if e.event == RATE_LIMIT_PRESSURE]


def test_screen_pushes_latency_snapshot(tmp_path: Path) -> None:
    # The screen finalization path pushes a latency_snapshot keyed by a
    # screen-level id; the screen rows carry per-call latency, so coverage > 0.
    from tests.envs.support import execution_policy
    from whetstone.envs.ed1 import load_ed1_tasks
    from whetstone.runner.task_screen import run_task_screen

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=2)
    reply_by_prompt = _all_pass_screen_reply(tasks)
    stream = EventStream(root=tmp_path, marker_sink=lambda _m: None)
    run_task_screen(
        model="qwen/qwen3-coder-flash",
        transport=FakeTransport(reply=reply_by_prompt),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=1, concurrency=4, events=stream,
    )
    lat = [e for e in stream.load() if e.event == LATENCY_SNAPSHOT]
    assert lat
    ev = lat[0]
    assert ev.unit.screen_id is not None
    assert ev.unit.screen_id.startswith("screen:qwen/qwen3-coder-flash")
    assert ev.unit.model == "qwen/qwen3-coder-flash"
    assert ev.fields["window"] == "screen"
    assert ev.fields["coverage"] > 0


def _all_pass_screen_reply(tasks) -> Callable[[str], str]:
    """A reply that returns each task's reference solution for any prompt.

    The screen prompts wrap the task; returning the canonical solution makes
    the code oracle pass, so the screen drives clean rows carrying latency.
    """
    solutions = [t.humaneval_task.canonical_solution for t in tasks]

    def reply(_prompt: str) -> str:
        # Any solution renders as a code block the extractor accepts; the
        # first is fine for a latency-coverage assertion (not a pass-rate one).
        return f"```python\n{solutions[0]}\n```"

    return reply


@dataclass
class _RateLimitMatchingPrompts:
    """Fail matching prompts with a RATE-LIMIT (429) typed failure.

    A ``RATE_LIMITED`` transport failure classifies to the semantic RATE_LIMIT
    class, so the recorded row ``failure_code`` reads as a rate limit -- the
    signal the runner counts for a ``rate_limit_pressure`` window event. Non-
    matching prompts succeed with the scripted reply.
    """

    should_fail: Callable[[str], bool]
    reply: Callable[[str], str]
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        prompt = _prompt_of(request)
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"content-type": "json"}, body={"model": "test-model"},
        )
        if self.should_fail(prompt):
            failure = ProviderTransportFailure(
                failure_class=FailureClass.RATE_LIMITED,
                code="http_status_429",
                message="scripted rate-limit (429)",
                retryable=True,
            )
            return ProviderInvocationEvidence.build(
                request=request, policy=self.policy,
                raw_request=raw_request, outcome=failure,
            )
        return ProviderInvocationEvidence.build(
            request=request, policy=self.policy, raw_request=raw_request,
            outcome=_response(self.reply(prompt)),
        )
