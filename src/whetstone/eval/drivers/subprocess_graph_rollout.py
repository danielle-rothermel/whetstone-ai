from __future__ import annotations

import weakref
from collections.abc import Callable
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
    ExitedOutcome,
    FiniteDurationLimit,
    ImportableEntryPoint,
    ImportableJsonResultError,
    JobId,
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
from whetstone.eval.drivers.row_common import RolloutRowOutput
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
        self._unit_deadline_seconds = unit_deadline_seconds
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

    def _pool(self, concurrency: int) -> WorkerPoolImportableJsonExecutor:
        """Return this driver's pool, fixing its width on first use.

        A pool is bound to one entry point and one width for its lifetime,
        so the first run decides how wide this driver ever runs. A later run
        asking for more parallelism than the pool has is a caller mistake
        worth surfacing rather than quietly under-running.
        """
        width = max(1, concurrency)
        if self._executor is None:
            self._executor = WorkerPoolImportableJsonExecutor(
                entry_point=self._entry_point,
                worker_count=width,
            )
            # A caller that never closes the driver still releases its
            # workers when the driver becomes unreachable. close() remains
            # the explicit, prompt path.
            self._finalizer = weakref.finalize(
                self, self._executor.close_blocking
            )
        elif width > self._executor.width:
            raise RowWorkerError(
                f"this driver's worker pool is {self._executor.width} wide "
                f"and cannot widen to {width}; build a driver per width"
            )
        return self._executor

    def close(self) -> None:
        """Stop every worker process this driver owns."""
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
        jobs: list[ExecutionJob] = []
        for row in scheduled_rows:
            key = (row.task_index, row.seed_index)
            job_id = JobId(uuid4())
            keys_by_job[job_id] = key
            jobs.append(
                build_in_process_importable_json_job(
                    job_id,
                    self._entry_point,
                    row_requests[key].model_dump(mode="json"),
                    budgets=row_budgets,
                )
            )

        collected_identities: set[str] = set()
        outputs_by_key: dict[tuple[int, int], RolloutRowOutput] = {}
        deadline_reached = False
        pool = self._pool(concurrency)
        batch_wall = _batch_wall_time(max_wall_seconds)
        # Closing the stream is what drains the scheduler, so a row that
        # raises still leaves the pool quiescent instead of abandoned.
        with closing(pool.run_many(jobs, wall_time=batch_wall)) as completions:
            for completion in completions:
                key = keys_by_job[completion.result.execution_id.job_id]
                row_request = row_requests[key]
                payload, status = _interpret_completion(completion, key=key)
                if status is RowDispatchStatus.NOT_DISPATCHED:
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


def _batch_wall_time(
    max_wall_seconds: float | None, /
) -> FiniteDurationLimit | None:
    """Convert this operation's deadline into a batch wall-time ceiling.

    A deadline that has already elapsed cannot be declared as a positive
    limit, so it is expressed as the smallest one dr-exec accepts: the batch
    expires immediately and every row reports as not dispatched.
    """
    if max_wall_seconds is None:
        return None
    if max_wall_seconds <= 0:
        return FiniteDurationLimit(max_ns=1)
    return FiniteDurationLimit.from_seconds(max_wall_seconds)


def _interpret_completion(
    completion: CompletedExecution,
    /,
    *,
    key: tuple[int, int],
) -> tuple[dict[str, object], RowDispatchStatus]:
    """Map one dr-exec completion into this driver's row vocabulary."""

    result = completion.result
    outcome = result.outcome
    if isinstance(outcome, CancelledOutcome):
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
