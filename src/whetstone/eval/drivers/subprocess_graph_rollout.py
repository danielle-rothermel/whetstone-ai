from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from whetstone.eval.drivers.graph_row_request import (
    GraphRowRequest,
    decode_graph_row_output,
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
from whetstone.eval.schema import SubmissionResultRecord
from whetstone.experiment.candidate import Candidate, TemplateRenderContract
from whetstone.experiment.env import Experiment
from whetstone.experiment.sampling import EvalSplit
from whetstone.execution.fanout import CallSpec, FanoutStatus, ProcessJob, run_call_pool
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.provider.driver import TransportCall
from whetstone.provider.language_model import (
    PlainPromptAdapter,
    StructuredPromptAdapter,
)
from whetstone.provider.llm_call import derive_rng_seed
from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = ["SubprocessGraphRolloutEvalDriver"]


@dataclass(frozen=True, slots=True)
class _ScheduledRow:
    task_index: int
    seed_index: int
    task: EvalTaskView
    task_id: str
    task_hash: str


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
        rng_seed=derive_rng_seed(
            request.candidate.candidate_id,
            row.task_id,
            row.seed_index,
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
    )


class SubprocessGraphRolloutEvalDriver(GraphRolloutEvalDriver):
    """Process-pool graph rollout driver using ``run_call_pool``."""

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
        self._transport_api_key_env = transport_api_key_env
        self._unit_deadline_seconds = unit_deadline_seconds

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
        _ = (eval_config_hash, partial_log, prompt_cache)
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
            )
            for row in scheduled_rows
        }
        specs = [
            CallSpec(
                key=(row.task_index, row.seed_index),
                job=ProcessJob(
                    entrypoint=self._row_job_entrypoint,
                    payload=row_requests[
                        (row.task_index, row.seed_index)
                    ].model_dump(mode="json"),
                ),
                decode=lambda payload, req=row_requests[
                    (row.task_index, row.seed_index)
                ]: decode_graph_row_output(payload, request=req),
                deadline_seconds=self._unit_deadline_seconds,
            )
            for row in scheduled_rows
        ]
        pool_outcome = run_call_pool(
            specs,
            concurrency=max(1, concurrency),
            is_rate_limited=lambda _output: False,
            max_wall_seconds=max_wall_seconds,
        )
        outputs_by_key: dict[tuple[int, int], RolloutRowOutput] = {}
        for fanout_result in pool_outcome.results:
            key = fanout_result.key
            row_request = row_requests[key]
            if fanout_result.completed and fanout_result.value is not None:
                outputs_by_key[key] = fanout_result.value
            else:
                outputs_by_key[key] = decode_graph_row_output(
                    {},
                    request=row_request,
                    fanout_status=fanout_result.status,
                )
        outputs = tuple(outputs_by_key[(row.task_index, row.seed_index)] for row in scheduled_rows)
        return aggregate_rollout_outputs(
            outputs=outputs,
            task_hashes=task_hashes,
            num_seeds=num_seeds,
            graph_hash=experiment.rollout_graph.graph_hash,
            matrix_plan=sampling.evaluation_matrix_plan,
            aggregate_name=self._aggregate_name,
            concurrency_halved=pool_outcome.concurrency_halved,
            deadline_reached=pool_outcome.deadline_reached,
            guard_timeouts=pool_outcome.guard_timeouts,
        )

    def submission_result_record(
        self, submission_result: object | None
    ) -> SubmissionResultRecord | None:
        return None
