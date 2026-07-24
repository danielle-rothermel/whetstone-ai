"""The checklist-B pilot: token sanity, agreement, direction, extraction.

Per ``reports/validation-plan.md`` "Spend order" step 1, the pilot runs ~10
instances x both probes (naive + ceiling) x repeats ``{0, 1, 2}`` and records:

* **token counts vs spec estimate** -- observed prompt/completion tokens per
  call against a caller-supplied spec estimate;
* **temp-0 agreement rate** -- how often the three temp-0 repeats of one
  (instance, probe) agree on the 0/1 oracle score (a temp-0 sanity check);
* **naive-vs-ceiling direction** -- whether the ceiling probe's mean score is
  ``>=`` the naive probe's (the headroom line points the right way);
* **per-call extraction spot-record** -- for each call the raw response text,
  the extracted answer the oracle saw, and the resulting 0/1 score;
* **spend** -- the OpenRouter credits before/after (when lane=openrouter).

The transport is injected (a scripted fake in tests, dr-providers' real client
in a live run), so importing/running the pilot logic makes no live paid call by
itself. Output is written to ``<root>/pilots/<env>.json``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dr_providers import (
    MessageRole,
    PromptMessage,
    ProviderCallConfig,
    ProviderCallRequest,
    Transcript,
)
from whetstone_envs.core import Instance

from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.envs.oracle_operator import env_exact_match_score
from whetstone.envs.registry import EnvSpec, env_spec
from whetstone.envs.rollout_definition import render_prompt
from whetstone.envs.task import EnvTask
from whetstone.execution.call_support import (
    failure_code_of,
    guard_deadline_seconds,
    is_rate_limit_failure,
)
from whetstone.execution.fanout import (
    DEFAULT_CONCURRENCY,
    RUNNER_TIMEOUT_CODE,
    CallSpec,
    FanoutConfig,
    run_call_pool,
)
from whetstone.execution.partials import PartialCallRecord, PartialLog
from whetstone.execution.prompt_cache import (
    CallExecution,
    PromptResultCache,
    execute_call,
)
from whetstone.optimization.schema import Candidate
from whetstone.provider.attempt import ProviderCallResult
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy
from whetstone.runner.budget import CreditsSnapshot
from whetstone.runner.events import (
    EventStream,
    EventUnit,
    is_rate_limit_code,
    latency_snapshot_event,
    rate_limit_pressure_event,
)

__all__ = [
    "PILOT_REPEATS",
    "PilotCallRecord",
    "PilotProbeSummary",
    "PilotReport",
    "run_pilot",
]

#: The temp-0 repeat ids the pilot runs per (instance, probe).
PILOT_REPEATS: tuple[int, ...] = (0, 1, 2)

#: The default number of instances the pilot draws from the internal split.
DEFAULT_PILOT_INSTANCES = 10

#: The default whole-run wall deadline for a pilot (seconds; ``--max-wall-
#: seconds``). On breach the pilot finishes in-flight calls, persists partials,
#: and the CLI exits non-zero with a status summary.
DEFAULT_PILOT_MAX_WALL_SECONDS = 1200.0


@dataclass(frozen=True, slots=True)
class PilotCallRecord:
    """One per-call extraction spot-record (raw + extracted + score)."""

    instance_id: str
    probe: str
    repeat_id: int
    raw_response: str
    extracted: str
    score: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    failed: bool
    #: The failure taxonomy code for a failed call (e.g. transport ``code``
    #: like ``"missing_base_url"``, else the semantic class name); ``""`` for a
    #: successful call. Aggregated into the pilot's loud failure summary.
    failure_code: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "probe": self.probe,
            "repeat_id": self.repeat_id,
            "raw_response": self.raw_response,
            "extracted": self.extracted,
            "score": self.score,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "failed": self.failed,
            "failure_code": self.failure_code,
        }


@dataclass(frozen=True, slots=True)
class PilotProbeSummary:
    """The aggregate for one probe (naive or ceiling)."""

    probe: str
    mean_score: float | None
    agreement_rate: float
    call_count: int
    failed_count: int
    token_mean_total: float | None = None
    spec_estimate_tokens: int | None = None
    token_vs_spec: float | None = None
    #: Provenance of ``spec_estimate_tokens``: ``live-measured`` vs
    #: ``scaled-pending-measurement`` (from the env registry); ``override``
    #: when ``--spec-estimate-tokens`` supplied the estimate. The measured
    #: ``token_mean_total`` is the value a report shows against this estimate.
    estimate_source: str = ""

    @property
    def success_count(self) -> int:
        return self.call_count - self.failed_count

    def as_dict(self) -> dict[str, object]:
        return {
            "probe": self.probe,
            "mean_score": self.mean_score,
            "agreement_rate": self.agreement_rate,
            "call_count": self.call_count,
            "failed_count": self.failed_count,
            "token_mean_total": self.token_mean_total,
            "spec_estimate_tokens": self.spec_estimate_tokens,
            "token_vs_spec": self.token_vs_spec,
            "estimate_source": self.estimate_source,
        }


@dataclass(slots=True)
class PilotReport:
    """The full pilot report written to ``<root>/pilots/<env>.json``."""

    env: str
    lane: str
    instances: int
    spec_estimate_tokens: int | None
    naive: PilotProbeSummary
    ceiling: PilotProbeSummary
    direction_ok: bool
    token_mean_total: float | None
    token_vs_spec: float | None
    calls: list[PilotCallRecord] = field(default_factory=list)
    spend_before: CreditsSnapshot | None = None
    spend_after: CreditsSnapshot | None = None
    #: Whether a rate-limit failure halved the shared effective concurrency.
    concurrency_halved: bool = False
    #: Whether the whole-run wall deadline stopped dispatch (a partial run).
    deadline_reached: bool = False
    #: Whether this report was assembled from a resumed/partial run (crash).
    partial: bool = False
    #: A human-readable halt/partial reason for the report + CLI summary.
    status_note: str = ""

    @property
    def spend_usd(self) -> float | None:
        if self.spend_before is None or self.spend_after is None:
            return None
        before = self.spend_before.remaining_usd
        after = self.spend_after.remaining_usd
        if before is None or after is None:
            return None
        return before - after

    @property
    def call_count(self) -> int:
        return self.naive.call_count + self.ceiling.call_count

    @property
    def failed_count(self) -> int:
        return self.naive.failed_count + self.ceiling.failed_count

    @property
    def success_count(self) -> int:
        return self.call_count - self.failed_count

    @property
    def success_rate(self) -> float:
        """Fraction of calls that succeeded (0.0 when every call failed)."""
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count

    def failure_summary(self) -> dict[str, int]:
        """Count the failed calls by their recorded failure code.

        A zero-success pilot is a hard plumbing failure; this feeds the loud
        summary the CLI prints (and exits non-zero on) so a 100%-failed run is
        never mistaken for a clean ``naive=None ceiling=None`` result.
        """
        counts: dict[str, int] = {}
        for call in self.calls:
            if call.failed:
                counts[call.failure_code] = (
                    counts.get(call.failure_code, 0) + 1
                )
        return counts

    def as_dict(self) -> dict[str, object]:
        return {
            "env": self.env,
            "lane": self.lane,
            "instances": self.instances,
            "spec_estimate_tokens": self.spec_estimate_tokens,
            "naive": self.naive.as_dict(),
            "ceiling": self.ceiling.as_dict(),
            "direction_ok": self.direction_ok,
            "token_mean_total": self.token_mean_total,
            "token_vs_spec": self.token_vs_spec,
            "call_count": self.call_count,
            "failed_count": self.failed_count,
            "success_rate": self.success_rate,
            "failure_summary": self.failure_summary(),
            "concurrency_halved": self.concurrency_halved,
            "deadline_reached": self.deadline_reached,
            "partial": self.partial,
            "status_note": self.status_note,
            "spend_usd": self.spend_usd,
            "spend_before": (
                None if self.spend_before is None
                else {
                    "total_credits": self.spend_before.total_credits,
                    "total_usage": self.spend_before.total_usage,
                    "remaining_usd": self.spend_before.remaining_usd,
                }
            ),
            "spend_after": (
                None if self.spend_after is None
                else {
                    "total_credits": self.spend_after.total_credits,
                    "total_usage": self.spend_after.total_usage,
                    "remaining_usd": self.spend_after.remaining_usd,
                }
            ),
            "calls": [c.as_dict() for c in self.calls],
        }

    def write(self, root: Path) -> Path:
        """Write ``<root>/pilots/<env>.json``; return the path.

        ``root`` is used exactly as given (it already points into the
        validation dir): pilots land at ``<root>/pilots/`` alongside the
        ledger's ``<root>/cells.jsonl`` -- no extra ``validation`` segment.
        """
        out_dir = root / "pilots"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.env}.json"
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))
        return path


def _request(config: ProviderCallConfig, prompt: str) -> ProviderCallRequest:
    return ProviderCallRequest(
        config=config,
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content=prompt),)
        ),
    )


def _agreement_rate(
    scores_by_instance: dict[str, list[float | None]],
) -> float:
    """Fraction of instances whose temp-0 repeats all agree on the score."""
    if not scores_by_instance:
        return 0.0
    agreeing = 0
    for scores in scores_by_instance.values():
        present = [s for s in scores if s is not None]
        if present and len(set(present)) == 1:
            agreeing += 1
    return agreeing / len(scores_by_instance)


@dataclass(frozen=True, slots=True)
class _ProbeCall:
    """One driven pilot call: its spot-record plus the terminal call Result.

    ``result`` is ``None`` for a call RESTORED from the partial log (pilot
    resume) -- a restored call is never re-driven.
    """

    record: PilotCallRecord
    result: ProviderCallResult | None


def _probe_thunk(
    env: EnvSpec,
    *,
    probe: str,
    instance: Instance,
    request: ProviderCallRequest,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    procedure_hash: str,
    logical_call_id: str,
    partial_log: PartialLog | None,
    cache: PromptResultCache | None = None,
) -> Callable[[], _ProbeCall]:
    """A zero-arg thunk running one pilot call (the fan-out unit).

    On completion it appends its :class:`PartialCallRecord` to the partial log
    (when one is given) BEFORE returning, so a crash mid-run keeps every
    already-finished call durably on disk (incremental persistence).
    """

    def _build() -> tuple[_ProbeCall, CallExecution]:
        repeat_id = _repeat_of(logical_call_id)
        execution = execute_call(
            request=request,
            policy=execution_policy,
            transport=transport,
            logical_call_id=logical_call_id,
            repeat_index=repeat_id,
            cache=cache,
            phase="pilot",
            unit=probe,
        )
        result = execution.result
        if not result.succeeded or result.generation is None:
            return _ProbeCall(
                record=PilotCallRecord(
                    instance_id=str(instance.id),
                    probe=probe,
                    repeat_id=repeat_id,
                    raw_response="",
                    extracted="",
                    score=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    failed=True,
                    failure_code=failure_code_of(result),
                ),
                result=result,
            ), execution
        text = result.generation.text
        score = env_exact_match_score(
            env=env,
            generation=text,
            gold=instance.gold,
            evaluation_procedure_config_hash=procedure_hash,
        )
        usage = result.generation.response.usage
        return _ProbeCall(
            record=PilotCallRecord(
                instance_id=str(instance.id),
                probe=probe,
                repeat_id=repeat_id,
                raw_response=text,
                extracted=text,
                score=float(score.value),
                prompt_tokens=(
                    usage.prompt_tokens if usage is not None else None
                ),
                completion_tokens=(
                    usage.completion_tokens if usage is not None else None
                ),
                total_tokens=(
                    usage.total_tokens if usage is not None else None
                ),
                failed=False,
            ),
            result=result,
        ), execution

    def _run() -> _ProbeCall:
        call, execution = _build()
        if partial_log is not None:
            r = call.record
            marks = execution.cache_marks()
            partial_log.append(
                PartialCallRecord(
                    phase="pilot", instance_id=r.instance_id, unit=probe,
                    repeat_id=r.repeat_id, score=r.score, failed=r.failed,
                    failure_code=r.failure_code,
                    prompt_tokens=r.prompt_tokens,
                    completion_tokens=r.completion_tokens,
                    total_tokens=r.total_tokens, raw_response=r.raw_response,
                    cache_hit=marks.cache_hit,
                    cache_source_phase=marks.cache_source_phase,
                    cache_source_unit=marks.cache_source_unit,
                    cache_source_call_id=marks.cache_source_call_id,
                    cache_source_at=marks.cache_source_at,
                )
            )
        return call

    return _run


def _repeat_of(logical_call_id: str) -> int:
    """Recover the repeat id from a ``pilot::<task>::<probe>::<repeat>`` id."""
    return int(logical_call_id.rsplit("::", 1)[1])


def _restored_probe_calls(
    partial_log: PartialLog | None, probe: str
) -> dict[tuple[str, int], PilotCallRecord]:
    """Pilot spot-records already on disk for this probe (resume skip)."""
    if partial_log is None:
        return {}
    restored: dict[tuple[str, int], PilotCallRecord] = {}
    for rec in partial_log.load():
        if rec.phase != "pilot" or rec.unit != probe:
            continue
        restored[(rec.instance_id, rec.repeat_id)] = PilotCallRecord(
            instance_id=rec.instance_id,
            probe=probe,
            repeat_id=rec.repeat_id,
            raw_response=rec.raw_response,
            extracted=rec.raw_response,
            score=rec.score,
            prompt_tokens=rec.prompt_tokens,
            completion_tokens=rec.completion_tokens,
            total_tokens=rec.total_tokens,
            failed=rec.failed,
            failure_code=rec.failure_code,
        )
    return restored


@dataclass(slots=True)
class _ProbeRun:
    """A probe summary plus the fan-out flags the pilot report needs."""

    summary: PilotProbeSummary
    concurrency_halved: bool = False
    deadline_reached: bool = False
    guard_timeouts: int = 0


def _run_probe(
    experiment: EnvExperiment,
    *,
    probe: str,
    candidate: Candidate,
    instances: tuple[Instance, ...],
    transport: TransportCall,
    execution_policy: ProviderExecutionPolicy,
    procedure_hash: str,
    calls: list[PilotCallRecord],
    spec_estimate_tokens: int | None = None,
    estimate_source: str = "",
    fanout: FanoutConfig | None = None,
    partial_log: PartialLog | None = None,
    cache: PromptResultCache | None = None,
) -> _ProbeRun:
    env = env_spec(experiment.env_name)
    config = experiment.rollout_definition.provider_call_config
    fanout = fanout or FanoutConfig()
    restored = _restored_probe_calls(partial_log, probe)

    ordered_keys: list[tuple[str, int]] = []
    specs: list[CallSpec[tuple[str, int], _ProbeCall]] = []
    for instance in instances:
        task = EnvTask.from_instance(env.name, instance)
        prompt = render_prompt(env, candidate, instance)
        for repeat_id in PILOT_REPEATS:
            key = (str(instance.id), repeat_id)
            ordered_keys.append(key)
            if key in restored:
                continue
            specs.append(
                CallSpec(
                    key=key,
                    run=_probe_thunk(
                        env,
                        probe=probe,
                        instance=instance,
                        request=_request(config, prompt),
                        execution_policy=execution_policy,
                        transport=transport,
                        procedure_hash=procedure_hash,
                        logical_call_id=f"pilot::{task.task_identity()}::"
                        f"{probe}::{repeat_id}",
                        # Each thunk persists its own partial record on
                        # completion (incremental persistence for resume).
                        partial_log=partial_log,
                        cache=cache,
                    ),
                    deadline_seconds=guard_deadline_seconds(execution_policy),
                )
            )

    outcome = run_call_pool(
        specs,
        concurrency=fanout.concurrency,
        is_rate_limited=lambda c: c.result is not None
        and is_rate_limit_failure(c.result),
        max_wall_seconds=fanout.max_wall_seconds,
    )

    driven: dict[tuple[str, int], PilotCallRecord] = {}
    for res in outcome.results:
        if res.timed_out:
            instance_id, repeat_id = res.key
            timeout_record = PilotCallRecord(
                instance_id=instance_id, probe=probe, repeat_id=repeat_id,
                raw_response="", extracted="", score=None,
                prompt_tokens=None, completion_tokens=None, total_tokens=None,
                failed=True, failure_code=RUNNER_TIMEOUT_CODE,
            )
            driven[res.key] = timeout_record
            # A guard timeout is a real (failed) observation: record it so a
            # resume does not re-drive the call that already blew the deadline.
            if partial_log is not None:
                partial_log.append(
                    PartialCallRecord(
                        phase="pilot", instance_id=instance_id, unit=probe,
                        repeat_id=repeat_id, score=None, failed=True,
                        failure_code=RUNNER_TIMEOUT_CODE,
                    )
                )
        elif res.value is not None:
            driven[res.key] = res.value.record

    # Assemble spot-records + per-instance score vectors in input order.
    scores_by_instance: dict[str, list[float | None]] = {}
    all_scores: list[float] = []
    probe_totals: list[int] = []
    failed = 0
    for key in ordered_keys:
        instance_id, _repeat = key
        record = restored[key] if key in restored else driven.get(key)
        if record is None:
            # The whole-run deadline stopped dispatch before this call.
            scores_by_instance.setdefault(instance_id, []).append(None)
            continue
        calls.append(record)
        scores_by_instance.setdefault(instance_id, []).append(record.score)
        if record.failed:
            failed += 1
        elif record.score is not None:
            all_scores.append(record.score)
            if record.total_tokens is not None:
                probe_totals.append(record.total_tokens)

    mean_score = (
        sum(all_scores) / len(all_scores) if all_scores else None
    )
    token_mean = (
        sum(probe_totals) / len(probe_totals) if probe_totals else None
    )
    token_vs_spec = (
        token_mean / spec_estimate_tokens
        if token_mean is not None and spec_estimate_tokens
        else None
    )
    summary = PilotProbeSummary(
        probe=probe,
        mean_score=mean_score,
        agreement_rate=_agreement_rate(scores_by_instance),
        call_count=len(instances) * len(PILOT_REPEATS),
        failed_count=failed,
        token_mean_total=token_mean,
        spec_estimate_tokens=spec_estimate_tokens,
        token_vs_spec=token_vs_spec,
        estimate_source=estimate_source,
    )
    return _ProbeRun(
        summary=summary,
        concurrency_halved=outcome.concurrency_halved,
        deadline_reached=outcome.deadline_reached,
        guard_timeouts=outcome.guard_timeouts,
    )


def run_pilot(
    *,
    env: str,
    lane: str,
    model: str,
    transport: TransportCall,
    execution_policy: ProviderExecutionPolicy,
    instance_count: int = DEFAULT_PILOT_INSTANCES,
    pool_n_per_stratum: int | None = None,
    split_sizes: tuple[int, int, int] | None = None,
    spec_estimate_tokens: int | None = None,
    spend_before: CreditsSnapshot | None = None,
    spend_after: CreditsSnapshot | None = None,
    credits_fetcher: Callable[[], CreditsSnapshot | None] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_wall_seconds: float = DEFAULT_PILOT_MAX_WALL_SECONDS,
    partial_log: PartialLog | None = None,
    events: EventStream | None = None,
    cache: PromptResultCache | None = None,
) -> PilotReport:
    """Run the checklist-B pilot for one env and return its report.

    Builds the env experiment, draws ``instance_count`` instances from the
    internal split, and runs both probes x 3 temp-0 repeats through the
    injected transport, collecting token counts, agreement, direction, and
    per-call extraction spot-records.

    Calls fan out through a bounded worker pool (``concurrency``, default 5,
    halved once on a rate-limit failure), each under a runner-level guard, with
    a whole-run wall deadline (``max_wall_seconds``, default 1200). When a
    ``partial_log`` is given, each call is appended as it completes so a
    crashed run leaves a resumable/partial-report record.

    Spend is recorded when a ``credits_fetcher`` is injected (round-3 pilots
    self-report spend): the fetcher is snapshotted before and after the run,
    exactly as the cell path does -- closing the gap where only the cell path
    fetched credits. An explicit ``spend_before``/``spend_after`` still wins.

    Token sanity runs against the env's committed per-probe estimates
    (:class:`whetstone.envs.registry.TokenEstimate`, sourced from each
    baseline-spec §5): naive and ceiling are checked separately. A non-None
    ``spec_estimate_tokens`` is an explicit override that applies to BOTH
    probes (the CLI ``--spec-estimate-tokens`` flag).
    """
    experiment = build_env_experiment(
        env,
        model=model,
        pool_n_per_stratum=pool_n_per_stratum,
        split_sizes=split_sizes,
    )
    procedure_hash = experiment.eval_configs.procedure_config_hash
    internal = experiment.eval_configs.internal.instances
    instances = internal[:instance_count]

    estimate = env_spec(env).token_estimate
    naive_estimate = (
        spec_estimate_tokens
        if spec_estimate_tokens is not None
        else estimate.naive
    )
    ceiling_estimate = (
        spec_estimate_tokens
        if spec_estimate_tokens is not None
        else estimate.ceiling
    )
    estimate_source = (
        "override"
        if spec_estimate_tokens is not None
        else estimate.estimate_source
    )

    # --- Spend snapshot BEFORE (credits fetcher injection; round-3 fix). ---
    is_openrouter = lane == "openrouter"
    if spend_before is None and is_openrouter and credits_fetcher is not None:
        spend_before = credits_fetcher()

    # The two probes share one whole-run deadline: track remaining per probe.
    run_start = time.monotonic()

    def _probe_fanout() -> FanoutConfig:
        remaining = max_wall_seconds - (time.monotonic() - run_start)
        return FanoutConfig(
            concurrency=concurrency, max_wall_seconds=max(0.0, remaining)
        )

    calls: list[PilotCallRecord] = []
    naive_run = _run_probe(
        experiment,
        probe="naive",
        candidate=experiment.initial_candidate,
        instances=instances,
        transport=transport,
        execution_policy=execution_policy,
        procedure_hash=procedure_hash,
        calls=calls,
        spec_estimate_tokens=naive_estimate,
        estimate_source=estimate_source,
        fanout=_probe_fanout(),
        partial_log=partial_log,
        cache=cache,
    )
    ceiling_run = _run_probe(
        experiment,
        probe="ceiling",
        candidate=experiment.ceiling_candidate,
        instances=instances,
        transport=transport,
        execution_policy=execution_policy,
        procedure_hash=procedure_hash,
        calls=calls,
        spec_estimate_tokens=ceiling_estimate,
        estimate_source=estimate_source,
        fanout=_probe_fanout(),
        partial_log=partial_log,
        cache=cache,
    )
    naive = naive_run.summary
    ceiling = ceiling_run.summary

    # --- Spend snapshot AFTER (credits fetcher injection; round-3 fix). ---
    if spend_after is None and is_openrouter and credits_fetcher is not None:
        spend_after = credits_fetcher()

    direction_ok = (
        naive.mean_score is not None
        and ceiling.mean_score is not None
        and ceiling.mean_score >= naive.mean_score
    )
    totals = [c.total_tokens for c in calls if c.total_tokens is not None]
    token_mean_total = sum(totals) / len(totals) if totals else None
    # The blended token-vs-spec ratio uses the mean of the two per-probe
    # estimates when both probes share the same committed estimate is not the
    # case; a blended spec is the average of the naive + ceiling estimates.
    blended_spec = (naive_estimate + ceiling_estimate) / 2
    token_vs_spec = (
        token_mean_total / blended_spec
        if token_mean_total is not None and blended_spec
        else None
    )
    concurrency_halved = (
        naive_run.concurrency_halved or ceiling_run.concurrency_halved
    )
    deadline_reached = (
        naive_run.deadline_reached or ceiling_run.deadline_reached
    )
    status_note = ""
    if deadline_reached:
        status_note = (
            f"whole-run wall deadline {max_wall_seconds:.0f}s reached; "
            "dispatch stopped, in-flight calls finished"
        )
    # Push run telemetry (task 24): a rate_limit_pressure event when the run
    # saw any 429/rate-limit rows or a halving, plus a latency_snapshot. Keyed
    # by a screen-level id (a pilot has no attempt). The pilot call records
    # carry no per-call latency, so the snapshot is honestly null-coverage.
    if events is not None:
        unit = EventUnit(
            screen_id=f"pilot:{env}:{model}", env=env, lane=lane, model=model
        )
        rate_limit_rows = sum(
            1 for c in calls if is_rate_limit_code(c.failure_code)
        )
        if rate_limit_rows or concurrency_halved:
            events.emit(
                rate_limit_pressure_event(
                    unit=unit,
                    rate_limit_rows=rate_limit_rows,
                    concurrency_halved=concurrency_halved,
                    guard_timeouts=0,
                    window_label="pilot",
                )
            )
        events.emit(
            latency_snapshot_event(
                unit=unit, median_latency_s=None, coverage=0,
                window_label="pilot",
            )
        )
    return PilotReport(
        env=env,
        lane=lane,
        instances=len(instances),
        spec_estimate_tokens=spec_estimate_tokens,
        naive=naive,
        ceiling=ceiling,
        direction_ok=direction_ok,
        token_mean_total=token_mean_total,
        token_vs_spec=token_vs_spec,
        calls=calls,
        spend_before=spend_before,
        spend_after=spend_after,
        concurrency_halved=concurrency_halved,
        deadline_reached=deadline_reached,
        status_note=status_note,
    )
