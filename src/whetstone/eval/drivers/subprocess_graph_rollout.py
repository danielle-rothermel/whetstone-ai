from __future__ import annotations

import math
import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass
from uuid import uuid4

from dr_exec import (
    BudgetAxis,
    BudgetExceededOutcome,
    Budgets,
    CancelledOutcome,
    CompletedExecution,
    ExecutionJob,
    ExecutionPoolConfig,
    ExitedOutcome,
    FiniteDurationLimit,
    FixedPoolCapacity,
    ImportableEntryPoint,
    ImportableJsonResultError,
    JobId,
    SchedulerBroken,
    WorkerPoolImportableJsonExecutor,
    build_in_process_importable_json_job,
    parse_importable_json_result,
)
from pydantic import JsonValue

from whetstone.eval.drivers.graph_row_request import (
    GraphRowRequest,
    RowDispatchStatus,
    decode_graph_row_output,
    import_path_for_callable,
    import_path_for_type,
    worker_request_identities,
)
from whetstone.eval.drivers.graph_rollout import (
    GraphRolloutEvalDriver,
    _task_id,
    _task_prompt_inputs,
)
from whetstone.eval.drivers.rollout_aggregate import aggregate_rollout_outputs
from whetstone.eval.drivers.row_common import (
    RolloutRowOutput,
    validated_phase_wall_seconds,
)
from whetstone.eval.drivers.eval_result import InternalEvalResult
from whetstone.eval.protocol import EvalRequest, EvalTaskView
from whetstone.experiment.candidate import TemplateRenderContract
from whetstone.experiment.env import Experiment
from whetstone.experiment.sampling import EvalSplit
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.provider.driver import TransportCall
from whetstone.provider.language_model import (
    PlainPromptAdapter,
    StructuredPromptAdapter,
)
from whetstone.provider.llm_call import resolve_eval_rng_seed
from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = ["RowWorkerError", "SubprocessGraphRolloutEvalDriver"]


class RowWorkerError(RuntimeError):
    """A row worker failed for a reason that is not a row-level outcome."""


@dataclass(frozen=True, slots=True)
class _ScheduledRow:
    task_index: int
    seed_index: int
    task: EvalTaskView
    task_id: str
    task_hash: str


def _entry_point(row_job_entrypoint: str) -> ImportableEntryPoint:
    module_name, separator, attribute_name = row_job_entrypoint.partition(":")
    if not separator:
        raise ValueError(
            "row_job_entrypoint must be 'importable.module:top_level_callable'"
        )
    return ImportableEntryPoint(
        module_name=module_name, attribute_name=attribute_name
    )


def _build_graph_row_request(
    *,
    experiment: Experiment,
    request: EvalRequest,
    row: _ScheduledRow,
    sampling: EvalSplit,
    execution_policy: ProviderExecutionPolicy,
    render_contract: TemplateRenderContract,
    mutation_field: str,
    graph_external_input_field: str,
    transport_api_key_env: str,
    transport_factory: str,
    eval_runner: str,
    prompt_adapter_type: str,
    prompt_adapter: JsonValue,
    partial_log: PartialLog | None,
    prompt_cache: PromptResultCache | None,
) -> GraphRowRequest:
    rollout_graph = experiment.rollout_graph
    rendered = render_contract.render(
        request.candidate.payload[mutation_field],
        _task_prompt_inputs(row.task),
    )
    gold = getattr(row.task, "gold", "")
    return GraphRowRequest(
        candidate_id=request.candidate.candidate_id,
        task_id=row.task_id,
        task_index=row.task_index,
        seed_index=row.seed_index,
        split_role=sampling.split_role,
        rendered_prompt=rendered,
        graph_config=rollout_graph.graph_config.model_dump(mode="json"),
        rollout_graph_hash=rollout_graph.graph_hash,
        provider_call_config=rollout_graph.provider_call_config.model_dump(
            mode="json"
        ),
        rng_seed=resolve_eval_rng_seed(
            candidate_id=request.candidate.candidate_id,
            task_id=row.task_id,
            task_hash=row.task_hash,
            seed_index=row.seed_index,
            seed_plan=sampling.seed_plan,
        ),
        mutation_field=mutation_field,
        graph_external_input_field=graph_external_input_field,
        eval_procedure_config_hash=rollout_graph.procedure_config_hash,
        execution_policy=execution_policy.model_dump(mode="json"),
        execution_policy_hash=execution_policy.identity_hash,
        prompt_inputs={
            str(key): str(value)
            for key, value in _task_prompt_inputs(row.task).items()
        },
        gold=gold if isinstance(gold, str) else "",
        transport_api_key_env=transport_api_key_env,
        transport_factory=transport_factory,
        eval_runner=eval_runner,
        prompt_adapter_type=prompt_adapter_type,
        prompt_adapter=prompt_adapter,
        partial_log_path=(
            str(partial_log.path.resolve()) if partial_log is not None else None
        ),
        prompt_cache_path=(
            str(prompt_cache.root.resolve()) if prompt_cache is not None else None
        ),
    )


class SubprocessGraphRolloutEvalDriver(GraphRolloutEvalDriver):
    """Graph rollout driver running rows on a dr-exec worker pool.

    The driver owns one worker pool for its lifetime. Workers import the row
    entry point once at startup, so the whetstone import is paid per worker
    rather than per row. Close the driver (or use it as a context manager) to
    stop its workers.

    Closing is terminal: a closed driver refuses further runs rather than
    resurrecting workers, so the pool's fixed width holds for the whole
    driver rather than only until the next close.

    A driver handed to a :class:`~whetstone.eval.runtime_engine.RuntimeEvalEngine`
    belongs to that engine: closing the engine (or using it as a context
    manager) is what stops these workers, and is the supported path for
    every driver built by ``ReferenceEvalRuntimeConfig.build_engine``. Close
    the driver directly only when nothing else owns it.

    ``run`` is safe to call from several threads; the pool is created once
    under a lock. The pool's width is still fixed by whichever run reaches
    it first.

    Known limitation: when a row's entry point raises, dr-exec's worker
    discards the payload exception and reports the fixed detail "the
    importable JSON entry point raised". This driver therefore surfaces such
    a row as a ``RowWorkerError`` attributed to the payload, but without the
    exception's type, message, or traceback. A dr-exec fix will carry that
    detail on the frame; until it lands, reproduce a raising row on the
    in-process :class:`GraphRolloutEvalDriver`, which propagates the original
    exception.
    """

    def __init__(
        self,
        *,
        row_job_entrypoint: str = "whetstone.eval.drivers.graph_worker:run_row",
        transport_api_key_env: str = "WHETSTONE_TOY_API_KEY",
        unit_deadline_seconds: float = 86_400.0,
        eval_runner: object,
        mutation_field: str,
        render_contract: TemplateRenderContract,
        transport_factory: Callable[[ProviderExecutionPolicy], TransportCall],
        resolve_provider_call_config: object | None = None,
        graph_external_input_field: str = "prompt",
        aggregate_name: str = "score",
        prompt_adapter: PlainPromptAdapter | StructuredPromptAdapter | None = None,
    ) -> None:
        super().__init__(
            eval_runner=eval_runner,  # type: ignore[arg-type]
            mutation_field=mutation_field,
            render_contract=render_contract,
            transport_factory=transport_factory,
            resolve_provider_call_config=resolve_provider_call_config,  # type: ignore[arg-type]
            graph_external_input_field=graph_external_input_field,
            aggregate_name=aggregate_name,
            prompt_adapter=prompt_adapter,
        )
        self._row_job_entrypoint = row_job_entrypoint
        self._entry_point = _entry_point(row_job_entrypoint)
        self._transport_api_key_env = transport_api_key_env
        self._unit_deadline_seconds = _validated_unit_deadline(
            unit_deadline_seconds
        )
        self._transport_factory_path = import_path_for_callable(transport_factory)
        self._eval_runner_path = import_path_for_type(type(self._eval_runner))
        self._prompt_adapter_type_path = import_path_for_type(
            type(self._prompt_adapter)
        )
        self._prompt_adapter_payload = self._prompt_adapter.model_dump(
            mode="json"
        )
        self._executor: WorkerPoolImportableJsonExecutor | None = None
        self._finalizer: weakref.finalize | None = None
        self._pool_lock = threading.Lock()
        self._closed = False

    def _pool(self, concurrency: int) -> WorkerPoolImportableJsonExecutor:
        """Return this driver's pool, fixing its width on first use.

        A pool is bound to one entry point and one width for its lifetime,
        so the first run decides how wide this driver ever runs. A later run
        asking for more parallelism than the pool has is a caller mistake
        worth surfacing rather than quietly under-running.

        The lock covers the whole create-and-register step so concurrent
        first runs cannot each build a pool and leave one of them orphaned
        with no finalizer to reap its workers.
        """
        width = max(1, concurrency)
        with self._pool_lock:
            if self._closed:
                raise RowWorkerError(
                    "this driver is closed and cannot run rows; build a new "
                    "driver rather than reusing a closed one"
                )
            if self._executor is None:
                executor = WorkerPoolImportableJsonExecutor(
                    entry_point=self._entry_point,
                    worker_count=width,
                )
                # A caller that never closes the driver still releases its
                # workers when the driver becomes unreachable. close()
                # remains the explicit, prompt path.
                self._finalizer = weakref.finalize(
                    self, executor.close_blocking
                )
                self._executor = executor
            elif width > self._executor.width:
                raise RowWorkerError(
                    f"this driver's worker pool is {self._executor.width} "
                    f"wide and cannot widen to {width}; build a driver per "
                    "width"
                )
            return self._executor

    def close(self) -> None:
        """Stop every worker process this driver owns, for good."""
        with self._pool_lock:
            self._closed = True
            finalizer = self._finalizer
            self._finalizer = None
            self._executor = None
        if finalizer is not None:
            finalizer()

    def __enter__(self) -> SubprocessGraphRolloutEvalDriver:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def run(
        self,
        *,
        experiment: Experiment,
        sampling: EvalSplit,
        request: EvalRequest,
        eval_config_hash: str,
        execution_policy: ProviderExecutionPolicy,
        concurrency: int,
        max_wall_seconds: float | None,
        partial_log: PartialLog | None,
        prompt_cache: PromptResultCache | None,
    ) -> InternalEvalResult:
        _ = eval_config_hash
        self.preflight(request.candidate)
        num_seeds = sampling.seed_plan.num_seeds
        task_hashes = sampling.task_set.task_hashes
        scheduled_rows = tuple(
            _ScheduledRow(
                task_index=task_index,
                seed_index=seed_index,
                task=task,
                task_id=_task_id(task),
                task_hash=task_hash,
            )
            for task_index, (task, task_hash) in enumerate(
                zip(sampling.tasks, task_hashes, strict=True)
            )
            for seed_index in range(num_seeds)
        )
        row_requests = {
            (row.task_index, row.seed_index): _build_graph_row_request(
                experiment=experiment,
                request=request,
                row=row,
                sampling=sampling,
                execution_policy=execution_policy,
                render_contract=self._render_contract,
                mutation_field=self._mutation_field,
                graph_external_input_field=self._graph_external_input_field,
                transport_api_key_env=self._transport_api_key_env,
                transport_factory=self._transport_factory_path,
                eval_runner=self._eval_runner_path,
                prompt_adapter_type=self._prompt_adapter_type_path,
                prompt_adapter=self._prompt_adapter_payload,
                partial_log=partial_log,
                prompt_cache=prompt_cache,
            )
            for row in scheduled_rows
        }
        row_budgets = Budgets(
            wall_time=FiniteDurationLimit.from_seconds(
                self._unit_deadline_seconds
            )
        )
        keys_by_job: dict[JobId, tuple[int, int]] = {}
        job_ids: list[JobId] = []
        for row in scheduled_rows:
            key = (row.task_index, row.seed_index)
            job_id = JobId(uuid4())
            keys_by_job[job_id] = key
            job_ids.append(job_id)

        collected_identities: set[str] = set()
        outputs_by_key: dict[tuple[int, int], RolloutRowOutput] = {}
        deadline_reached = False
        pool = self._pool(concurrency)
        batch_wall = _batch_wall_time(max_wall_seconds)

        def _pending_jobs() -> Iterator[ExecutionJob]:
            for pending_id in job_ids:
                yield build_in_process_importable_json_job(
                    pending_id,
                    self._entry_point,
                    row_requests[keys_by_job[pending_id]].model_dump(
                        mode="json"
                    ),
                    budgets=row_budgets,
                )

        # Closing the stream is what drains the scheduler, so a row that
        # raises still leaves the pool quiescent instead of abandoned.
        stream = _scheduled_completions(
            pool,
            _pending_jobs(),
            width=max(1, concurrency),
            batch_wall=batch_wall,
        )
        with closing(stream) as completions:
            for completion in _translating_scheduler_breaks(completions):
                job_id = completion.result.execution_id.job_id
                key = keys_by_job[job_id]
                row_request = row_requests[key]
                payload, status = _interpret_completion(completion, key=key)
                if status in _DEADLINE_STATUSES:
                    deadline_reached = True
                if status is RowDispatchStatus.COMPLETED:
                    collected_identities.update(
                        worker_request_identities(payload)
                    )
                    outputs_by_key[key] = decode_graph_row_output(
                        payload, request=row_request
                    )
                else:
                    outputs_by_key[key] = decode_graph_row_output(
                        {}, request=row_request, dispatch_status=status
                    )

        missing_keys = [
            (row.task_index, row.seed_index)
            for row in scheduled_rows
            if (row.task_index, row.seed_index) not in outputs_by_key
        ]
        if missing_keys:
            raise RowWorkerError(
                "the row worker pool returned no outcome for rows "
                f"{sorted(missing_keys)!r}"
            )
        outputs = tuple(
            outputs_by_key[(row.task_index, row.seed_index)]
            for row in scheduled_rows
        )
        return aggregate_rollout_outputs(
            outputs=outputs,
            task_hashes=task_hashes,
            num_seeds=num_seeds,
            graph_hash=experiment.rollout_graph.graph_hash,
            matrix_plan=sampling.evaluation_matrix_plan,
            aggregate_name=self._aggregate_name,
            request_identities=frozenset(collected_identities),
            deadline_reached=deadline_reached,
        )


#: Statuses that mean the operation deadline stopped this row.
_DEADLINE_STATUSES = frozenset(
    {
        RowDispatchStatus.NOT_DISPATCHED,
        RowDispatchStatus.OPERATION_DEADLINE,
    }
)

#: dr-exec rejects a wall-time limit that does not convert to at least one
#: nanosecond, so anything below this is treated as already elapsed.
_MIN_REPRESENTABLE_WALL_SECONDS = 1e-9

#: Span above which a cancelled row is read as having occupied a worker.
#:
#: dr-exec publishes no started flag on ``CancelledOutcome``, so the measured
#: span is the evidence it does publish. A row the expired scheduler births
#: already cancelled never leases a worker: it returns from the pre-lease
#: cancel check having done nothing but stamp two clocks, which is tens of
#: microseconds. A row that reached a worker survives until the batch watcher
#: expires the scheduler, so it measures a large fraction of the batch wall.
#:
#: One millisecond sits two orders of magnitude above the bookkeeping span
#: and two orders below the smallest batch wall a real rollout declares, so
#: no observed span falls near it. Spawning a worker alone costs far more
#: than this floor, so a row that got as far as leasing is never read as
#: undispatched.
_STARTED_ROW_FLOOR_NS = 1_000_000


def _validated_unit_deadline(unit_deadline_seconds: float, /) -> float:
    """Reject a per-row budget dr-exec could not turn into a limit.

    The value only reaches ``FiniteDurationLimit`` when a run starts, so an
    unusable one would otherwise surface as a bare ``ValueError`` from deep
    inside a rollout rather than at the caller's construction site.
    """
    if (
        not math.isfinite(unit_deadline_seconds)
        or unit_deadline_seconds < _MIN_REPRESENTABLE_WALL_SECONDS
    ):
        raise ValueError(
            "unit_deadline_seconds must be a finite positive number of "
            f"seconds, not {unit_deadline_seconds!r}"
        )
    try:
        FiniteDurationLimit.from_seconds(unit_deadline_seconds)
    except ValueError as error:
        raise ValueError(
            f"unit_deadline_seconds={unit_deadline_seconds!r} is not a "
            f"usable per-row wall budget: {error}"
        ) from error
    return unit_deadline_seconds


def _scheduled_completions(
    pool: WorkerPoolImportableJsonExecutor,
    jobs: Iterator[ExecutionJob],
    /,
    *,
    width: int,
    batch_wall: FiniteDurationLimit | None,
) -> Iterator[CompletedExecution]:
    """Stream the batch at this run's requested parallelism.

    The pool's width is a ceiling on workers, not on this run: passing the
    run's own capacity is what lets a later, narrower run actually narrow
    instead of silently scheduling at the pool's original width.
    """
    return pool.run_many(
        jobs,
        config=ExecutionPoolConfig(
            capacity=FixedPoolCapacity(
                max_active_jobs=min(width, pool.width)
            )
        ),
        wall_time=batch_wall,
    )


def _translating_scheduler_breaks(
    completions: Iterator[CompletedExecution], /
) -> Iterator[CompletedExecution]:
    """Present a broken scheduler as this driver's own failure type.

    ``RowWorkerError`` is the driver's single failure surface, so a caller
    does not need to import dr-exec to catch a pool that broke mid-batch.
    """
    try:
        yield from completions
    except SchedulerBroken as error:
        raise RowWorkerError(
            f"the row worker pool's scheduler broke mid-batch: {error}"
        ) from error


def _batch_wall_time(
    max_wall_seconds: float | None, /
) -> FiniteDurationLimit | None:
    """Convert this operation's deadline into a batch wall-time ceiling.

    A negative or NaN wall is rejected by
    :func:`validated_phase_wall_seconds` before reaching here. Positive
    infinity and any wall too long for dr-exec to express arrive as ``None``
    — unbounded, exactly as the in-process driver reads them, so a generous
    wall is never mistaken for an expired one.

    What remains is a valid nonnegative wall dr-exec can express. One that
    has already elapsed — or is too small to round up to a whole nanosecond —
    is declared as the smallest limit dr-exec accepts: the batch expires
    immediately and no row is dispatched.
    """
    seconds = validated_phase_wall_seconds(max_wall_seconds)
    if seconds is None:
        return None
    if seconds < _MIN_REPRESENTABLE_WALL_SECONDS:
        return FiniteDurationLimit(max_ns=1)
    try:
        return FiniteDurationLimit.from_seconds(seconds)
    except ValueError:
        return FiniteDurationLimit(max_ns=1)


def _cancelled_row_started(completion: CompletedExecution, /) -> bool:
    """Read dr-exec's own evidence of whether a cancelled row ran.

    ``CancelledOutcome`` carries no started flag, and dr-exec's attribution
    is identical for both cancel paths, so the execution's measured span is
    the authoritative signal it does publish.

    dr-exec stamps ``started_at``/``duration_ns`` inside the executor's
    ``run_blocking``. A job the expired scheduler births already cancelled
    returns from the pre-lease cancel check without touching a worker, so it
    measures a bookkeeping span of microseconds. A job that reached a worker
    is only cancelled when the batch watcher expires the scheduler, so its
    span runs to the batch wall — orders of magnitude longer. Comparing
    against a floor far above bookkeeping cost and far below any usable
    batch wall is what separates the two without a second clock.

    This driver caps the scheduler's capacity at the pool's width, so an
    admitted job never waits for a worker slot: measured time means worker
    time.
    """
    return (
        completion.result.measurements.duration_ns
        >= _STARTED_ROW_FLOOR_NS
    )


def _interpret_completion(
    completion: CompletedExecution,
    /,
    *,
    key: tuple[int, int],
) -> tuple[dict[str, object], RowDispatchStatus]:
    """Map one dr-exec completion into this driver's row vocabulary.

    The batch deadline cancels rows already running in a worker alongside
    rows that never started, and both arrive as ``CancelledOutcome``. Only
    a row that was never handed to a worker is honestly "not dispatched";
    one that ran and was killed is the operation deadline.
    """

    result = completion.result
    outcome = result.outcome
    if isinstance(outcome, CancelledOutcome):
        if _cancelled_row_started(completion):
            return {}, RowDispatchStatus.OPERATION_DEADLINE
        return {}, RowDispatchStatus.NOT_DISPATCHED
    if isinstance(outcome, BudgetExceededOutcome):
        if outcome.axis is BudgetAxis.WALL_TIME:
            return {}, RowDispatchStatus.UNIT_TIMEOUT
        raise RowWorkerError(
            f"row {key!r} exceeded the unexpected {outcome.axis.value} budget"
        )
    if isinstance(outcome, ExitedOutcome) and outcome.exit_code == 0:
        return _completed_payload(completion, key=key), RowDispatchStatus.COMPLETED
    raise RowWorkerError(
        f"row {key!r} failed in its worker: {outcome.kind.value} "
        f"(owner={result.attribution.owner.value}, "
        f"detail={result.attribution.detail!r})"
    )


def _completed_payload(
    completion: CompletedExecution,
    /,
    *,
    key: tuple[int, int],
) -> dict[str, object]:
    try:
        payload = parse_importable_json_result(completion)
    except ImportableJsonResultError as error:
        raise RowWorkerError(
            f"row {key!r} did not return one worker result: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RowWorkerError(
            f"row {key!r} returned a worker result that is not a JSON object"
        )
    return dict(payload)
