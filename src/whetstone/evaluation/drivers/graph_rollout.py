from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from dr_providers import ProviderCallConfig

from whetstone.core.identity import IdentityRef
from whetstone.evaluation.aggregate import TaskRows, unweighted_task_mean
from whetstone.evaluation.attribution import attribute_generated_row
from whetstone.evaluation.drivers.eval_result import (
    InternalEvalResult,
    per_task_count,
    per_task_score,
)
from whetstone.evaluation.drivers.graph_row import (
    graph_result_to_row_fields,
    run_rollout_row,
)
from whetstone.evaluation.drivers.row_common import (
    GenerationRowOutput,
    remaining_phase_wall_seconds,
    start_phase_deadline,
)
from whetstone.evaluation.eval_procedure import EvalProcedureRunner
from whetstone.evaluation.protocol import EvaluationRequest, EvaluationTaskView
from whetstone.evaluation.schema import SubmissionResultRecord
from whetstone.evaluation.traces import ExecutedRowState
from whetstone.experiment.candidate import Candidate, TemplateRenderContract
from whetstone.experiment.env import Experiment
from whetstone.experiment.graph.llm_call_run_node import (
    EvalRunNodeDeps,
    LlmCallRunNodeDeps,
    ProviderCallConfigResolver,
)
from whetstone.experiment.graph.run_node_registry import build_run_node
from whetstone.experiment.sampling import SplitSampling
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.provider.driver import TransportCall
from whetstone.provider.language_model import (
    PlainPromptAdapter,
    StructuredPromptAdapter,
)
from whetstone.provider.llm_call import LlmCallContext, derive_rng_seed
from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = ["GraphRolloutEvaluationDriver", "run_graph_evaluation_row"]


def _task_id(task: object) -> str:
    task_id = getattr(task, "task_id", None)
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task must expose a non-empty task_id")
    return task_id


def _task_prompt_inputs(task: object) -> dict[str, str]:
    prompt_inputs = getattr(task, "prompt_inputs", None)
    if not isinstance(prompt_inputs, dict):
        raise ValueError("task must expose prompt_inputs as a dict")
    return {str(key): str(value) for key, value in prompt_inputs.items()}


def _default_provider_config_resolver(
    experiment: Experiment,
) -> ProviderCallConfigResolver:
    provider_config = experiment.generation_graph.provider_call_config

    def resolve(_ref: IdentityRef) -> ProviderCallConfig:
        return provider_config

    return resolve


def run_graph_evaluation_row(
    *,
    experiment: Experiment,
    candidate: Candidate,
    task: EvaluationTaskView,
    task_index: int,
    sample_index: int,
    split_role: str,
    llm_context: LlmCallContext,
    eval_runner: EvalProcedureRunner,
    render_contract: TemplateRenderContract,
    mutation_field: str,
    resolve_provider_call_config: ProviderCallConfigResolver,
    graph_external_input_field: str = "prompt",
) -> GenerationRowOutput:
    template = candidate.payload[mutation_field]
    rendered = render_contract.render(template, _task_prompt_inputs(task))
    generation_graph = experiment.generation_graph
    run_node = build_run_node(
        llm_deps=LlmCallRunNodeDeps(
            context=llm_context,
            resolve_provider_call_config=resolve_provider_call_config,
            graph_hash=generation_graph.graph_hash,
            rng_seed=derive_rng_seed(
                candidate.candidate_id,
                _task_id(task),
                sample_index,
            ),
            sample_index=sample_index,
            drive_ordinal=0,
            phase=split_role,
            unit=candidate.candidate_id,
        ),
        eval_deps=EvalRunNodeDeps(runner=eval_runner, task=task),
    )
    result = run_rollout_row(
        graph=generation_graph.graph_config,
        inputs={graph_external_input_field: rendered},
        run_node=run_node,
    )
    return graph_result_to_row_fields(
        result,
        candidate_id=candidate.candidate_id,
        task_id=_task_id(task),
        task_index=task_index,
        sample_index=sample_index,
    )


def _deadline_missing_row(
    *,
    candidate_id: str,
    task_id: str,
    task_index: int,
    sample_index: int,
) -> GenerationRowOutput:
    return GenerationRowOutput(
        candidate_id=candidate_id,
        task_id=task_id,
        task_index=task_index,
        sample_index=sample_index,
        row_state=ExecutedRowState.MISSING,
        executed_component_steps=(),
        output_text=None,
        score=None,
        failure_code="deadline",
    )


@dataclass(frozen=True, slots=True)
class _ScheduledRow:
    task_index: int
    sample_index: int
    task: EvaluationTaskView
    task_id: str
    task_hash: str


class GraphRolloutEvaluationDriver:
    """Parallel in-process graph rollout driver for evaluation splits."""

    def __init__(
        self,
        *,
        eval_runner: EvalProcedureRunner,
        mutation_field: str,
        render_contract: TemplateRenderContract,
        transport_factory: Callable[[ProviderExecutionPolicy], TransportCall],
        resolve_provider_call_config: ProviderCallConfigResolver | None = None,
        graph_external_input_field: str = "prompt",
        aggregate_name: str = "score",
        prompt_adapter: PlainPromptAdapter | StructuredPromptAdapter | None = None,
    ) -> None:
        self._eval_runner = eval_runner
        self._mutation_field = mutation_field
        self._render_contract = render_contract
        self._transport_factory = transport_factory
        self._resolve_provider_call_config = resolve_provider_call_config
        self._graph_external_input_field = graph_external_input_field
        self._aggregate_name = aggregate_name
        self._prompt_adapter = (
            PlainPromptAdapter()
            if prompt_adapter is None
            else prompt_adapter
        )

    def preflight(self, candidate: Candidate) -> None:
        template = candidate.payload.get(self._mutation_field)
        self._render_contract.validate_template(template)

    def rendered_prompt(
        self,
        candidate: Candidate,
        task: object,
        *,
        max_budget: int | None,
    ) -> str:
        _ = max_budget
        template = candidate.payload[self._mutation_field]
        return self._render_contract.render(
            template,
            _task_prompt_inputs(task),
        )

    def submission_result_record(
        self, submission_result: object | None
    ) -> SubmissionResultRecord | None:
        return None

    def task_model_identity_hash(self, experiment: Experiment) -> str:
        provider = experiment.generation_graph.provider_call_config
        return str(provider.identity_hash)

    def expected_model_route(self, experiment: Experiment) -> str:
        route = experiment.generation_graph.provider_call_config.route
        return (
            f"{route.provider.value}/{route.protocol.value}/{route.model}"
        )

    def run(
        self,
        *,
        experiment: Experiment,
        sampling: SplitSampling,
        request: EvaluationRequest,
        execution_policy: ProviderExecutionPolicy,
        concurrency: int,
        max_wall_seconds: float | None,
        partial_log: PartialLog | None,
        prompt_cache: PromptResultCache | None,
    ) -> InternalEvalResult:
        _ = partial_log
        self.preflight(request.candidate)
        resolve_provider_call_config = (
            self._resolve_provider_call_config
            or _default_provider_config_resolver(experiment)
        )
        llm_context = LlmCallContext(
            execution_policy=execution_policy,
            transport=self._transport_factory(execution_policy),
            prompt_adapter=self._prompt_adapter,
            prompt_cache=prompt_cache,
        )
        num_samples = sampling.sample_plan.num_samples
        task_hashes = sampling.task_set.task_hashes
        scheduled_rows = tuple(
            _ScheduledRow(
                task_index=task_index,
                sample_index=sample_index,
                task=task,
                task_id=_task_id(task),
                task_hash=task_hash,
            )
            for task_index, (task, task_hash) in enumerate(
                zip(sampling.tasks, task_hashes, strict=True)
            )
            for sample_index in range(num_samples)
        )
        deadline = start_phase_deadline(max_wall_seconds)
        outputs_by_key: dict[tuple[int, int], GenerationRowOutput] = {}
        deadline_reached = False
        max_workers = max(1, concurrency)

        def _execute_row(row: _ScheduledRow) -> GenerationRowOutput:
            return run_graph_evaluation_row(
                experiment=experiment,
                candidate=request.candidate,
                task=row.task,
                task_index=row.task_index,
                sample_index=row.sample_index,
                split_role=sampling.split_role,
                llm_context=llm_context,
                eval_runner=self._eval_runner,
                render_contract=self._render_contract,
                mutation_field=self._mutation_field,
                resolve_provider_call_config=resolve_provider_call_config,
                graph_external_input_field=self._graph_external_input_field,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending: dict[object, _ScheduledRow] = {}
            for row in scheduled_rows:
                remaining = remaining_phase_wall_seconds(deadline)
                if remaining is not None and remaining <= 0:
                    deadline_reached = True
                    outputs_by_key[(row.task_index, row.sample_index)] = (
                        _deadline_missing_row(
                            candidate_id=request.candidate.candidate_id,
                            task_id=row.task_id,
                            task_index=row.task_index,
                            sample_index=row.sample_index,
                        )
                    )
                    continue
                pending[executor.submit(_execute_row, row)] = row

            for future in as_completed(pending):
                scheduled = pending[future]
                outputs_by_key[(scheduled.task_index, scheduled.sample_index)] = (
                    future.result()
                )

        outputs = tuple(
            outputs_by_key[(row.task_index, row.sample_index)]
            for row in scheduled_rows
        )
        task_rows: list[TaskRows] = []
        for task_index, task_hash in enumerate(task_hashes):
            row_values = tuple(
                attribute_generated_row(
                    row_state=output.row_state,
                    score=output.score,
                    failure_code=output.failure_code or None,
                )
                for output in outputs
                if output.task_index == task_index
            )
            task_rows.append(TaskRows(task_hash=task_hash, rows=row_values))

        matrix_plan = sampling.evaluation_matrix_plan
        aggregate = unweighted_task_mean(
            aggregate_name=self._aggregate_name,
            graph_hash=experiment.generation_graph.graph_hash,
            evaluation_binding_hash=request.evaluation_binding.identity_hash(),
            task_rows=tuple(task_rows),
            plan=matrix_plan,
        )
        per_task_scores = tuple(
            per_task_score(task_row, num_samples) for task_row in task_rows
        )
        per_task_counts = tuple(
            per_task_count(task_row, num_samples) for task_row in task_rows
        )
        return InternalEvalResult(
            aggregate=aggregate,
            reward=None,
            per_task_scores=per_task_scores,
            per_task_counts=per_task_counts,
            outputs=outputs,
            deadline_reached=deadline_reached,
        )
