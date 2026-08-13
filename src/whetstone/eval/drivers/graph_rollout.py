from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

from dr_providers import ProviderCallConfig

from whetstone.core.identity import IdentityRef
from whetstone.eval.drivers.graph_row import (
    execute_rollout_graph,
    graph_result_to_row_fields,
)
from whetstone.eval.drivers.rollout_aggregate import aggregate_rollout_outputs
from whetstone.eval.drivers.row_common import (
    RolloutRowOutput,
    remaining_phase_wall_seconds,
    start_phase_deadline,
)
from whetstone.eval.drivers.eval_result import InternalEvalResult
from whetstone.eval.eval_procedure import EvalProcedureRunner
from whetstone.eval.protocol import EvalRequest, EvalTaskView
from whetstone.eval.schema import SubmissionResultRecord
from whetstone.eval.traces import ExecutedRowState
from whetstone.experiment.candidate import Candidate, TemplateRenderContract
from whetstone.experiment.env import Experiment
from whetstone.experiment.graph.llm_call_run_node import (
    EvalRunNodeDeps,
    LlmCallRunNodeDeps,
    ProviderCallConfigResolver,
)
from whetstone.experiment.graph.run_node_registry import build_run_node
from whetstone.experiment.sampling import EvalSplit
from whetstone.execution._file_lock import ensure_private_directory
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.provider.driver import TransportCall
from whetstone.provider.language_model import (
    PlainPromptAdapter,
    StructuredPromptAdapter,
)
from whetstone.provider.llm_call import LlmCallContext, derive_rng_seed
from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = ["GraphRolloutEvalDriver", "run_rollout_row"]


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
    provider_config = experiment.rollout_graph.provider_call_config

    def resolve(_ref: IdentityRef) -> ProviderCallConfig:
        return provider_config

    return resolve


def run_rollout_row(
    *,
    experiment: Experiment,
    candidate: Candidate,
    task: EvalTaskView,
    task_index: int,
    seed_index: int,
    split_role: str,
    llm_context: LlmCallContext,
    eval_runner: EvalProcedureRunner,
    render_contract: TemplateRenderContract,
    mutation_field: str,
    resolve_provider_call_config: ProviderCallConfigResolver,
    graph_external_input_field: str = "prompt",
    request_identity_sink: list[str] | None = None,
) -> RolloutRowOutput:
    task_id = _task_id(task)
    template = candidate.payload[mutation_field]
    rendered = render_contract.render(template, _task_prompt_inputs(task))
    rollout_graph = experiment.rollout_graph
    run_node = build_run_node(
        llm_deps=LlmCallRunNodeDeps(
            context=llm_context,
            resolve_provider_call_config=resolve_provider_call_config,
            graph_hash=rollout_graph.graph_hash,
            rng_seed=derive_rng_seed(
                candidate.candidate_id,
                task_id,
                seed_index,
            ),
            task_id=task_id,
            seed_index=seed_index,
            drive_ordinal=0,
            phase=split_role,
            unit=candidate.candidate_id,
            split_role=split_role,
            request_identity_sink=request_identity_sink,
        ),
        eval_deps=EvalRunNodeDeps(runner=eval_runner, task=task),
    )
    result = execute_rollout_graph(
        graph=rollout_graph.graph_config,
        inputs={graph_external_input_field: rendered},
        run_node=run_node,
    )
    return graph_result_to_row_fields(
        result,
        candidate_id=candidate.candidate_id,
        task_id=task_id,
        task_index=task_index,
        seed_index=seed_index,
    )


def _deadline_missing_row(
    *,
    candidate_id: str,
    task_id: str,
    task_index: int,
    seed_index: int,
) -> RolloutRowOutput:
    return RolloutRowOutput(
        candidate_id=candidate_id,
        task_id=task_id,
        task_index=task_index,
        seed_index=seed_index,
        row_state=ExecutedRowState.MISSING,
        trace_steps=(),
        output_text=None,
        score=None,
        failure_code="deadline",
    )


@dataclass(frozen=True, slots=True)
class _ScheduledRow:
    task_index: int
    seed_index: int
    task: EvalTaskView
    task_id: str
    task_hash: str


class GraphRolloutEvalDriver:
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
        provider = experiment.rollout_graph.provider_call_config
        return str(provider.identity_hash)

    def expected_model_route(self, experiment: Experiment) -> str:
        route = experiment.rollout_graph.provider_call_config.route
        return (
            f"{route.provider.value}/{route.protocol.value}/{route.model}"
        )

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
        if partial_log is not None:
            ensure_private_directory(partial_log.path.parent)
        resolve_provider_call_config = (
            self._resolve_provider_call_config
            or _default_provider_config_resolver(experiment)
        )
        llm_context = LlmCallContext(
            execution_policy=execution_policy,
            transport=self._transport_factory(execution_policy),
            prompt_adapter=self._prompt_adapter,
            prompt_cache=prompt_cache,
            partial_log=partial_log,
        )
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
        deadline = start_phase_deadline(max_wall_seconds)
        outputs_by_key: dict[tuple[int, int], RolloutRowOutput] = {}
        deadline_reached = False
        request_identities: set[str] = set()
        max_workers = max(1, concurrency)

        def _execute_row(row: _ScheduledRow) -> tuple[RolloutRowOutput, tuple[str, ...]]:
            row_identities: list[str] = []
            output = run_rollout_row(
                experiment=experiment,
                candidate=request.candidate,
                task=row.task,
                task_index=row.task_index,
                seed_index=row.seed_index,
                split_role=sampling.split_role,
                llm_context=llm_context,
                eval_runner=self._eval_runner,
                render_contract=self._render_contract,
                mutation_field=self._mutation_field,
                resolve_provider_call_config=resolve_provider_call_config,
                graph_external_input_field=self._graph_external_input_field,
                request_identity_sink=row_identities,
            )
            return output, tuple(row_identities)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending: dict[object, _ScheduledRow] = {}
            for row in scheduled_rows:
                remaining = remaining_phase_wall_seconds(deadline)
                if remaining is not None and remaining <= 0:
                    deadline_reached = True
                    outputs_by_key[(row.task_index, row.seed_index)] = (
                        _deadline_missing_row(
                            candidate_id=request.candidate.candidate_id,
                            task_id=row.task_id,
                            task_index=row.task_index,
                            seed_index=row.seed_index,
                        )
                    )
                    continue
                pending[executor.submit(_execute_row, row)] = row

            unfinished = set(pending.keys())
            while unfinished:
                remaining = remaining_phase_wall_seconds(deadline)
                if remaining is not None and remaining <= 0:
                    deadline_reached = True
                    for future in unfinished:
                        scheduled = pending[future]
                        outputs_by_key[
                            (scheduled.task_index, scheduled.seed_index)
                        ] = _deadline_missing_row(
                            candidate_id=request.candidate.candidate_id,
                            task_id=scheduled.task_id,
                            task_index=scheduled.task_index,
                            seed_index=scheduled.seed_index,
                        )
                        future.cancel()
                    break
                wait_timeout = remaining
                done, not_done = wait(
                    unfinished,
                    timeout=wait_timeout,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    scheduled = pending[future]
                    try:
                        output, row_identities = future.result()
                    except Exception:
                        raise
                    request_identities.update(row_identities)
                    outputs_by_key[
                        (scheduled.task_index, scheduled.seed_index)
                    ] = output
                unfinished = not_done

            for row in scheduled_rows:
                key = (row.task_index, row.seed_index)
                if key not in outputs_by_key:
                    outputs_by_key[key] = _deadline_missing_row(
                        candidate_id=request.candidate.candidate_id,
                        task_id=row.task_id,
                        task_index=row.task_index,
                        seed_index=row.seed_index,
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
            request_identities=frozenset(request_identities),
            deadline_reached=deadline_reached,
        )
