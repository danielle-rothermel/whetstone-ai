from __future__ import annotations

from whetstone.evaluation.drivers.eval_result import (
    InternalEvalResult,
    per_task_count,
    per_task_score,
)
from whetstone.evaluation.drivers.row_common import GenerationRowOutput
from whetstone.evaluation.protocol import EvalRequest
from whetstone.evaluation.schema import SubmissionResultRecord
from whetstone.evaluation.traces import ExecutedRowState
from whetstone.evaluation.aggregate import RowValue, TaskRows, unweighted_task_mean
from whetstone.evaluation.driver import EvaluationDriver
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.candidate import Candidate, TemplateRenderContract
from whetstone.experiment.env import Experiment
from whetstone.experiment.sampling import SplitSampling
from whetstone.provider.policy import ProviderExecutionPolicy
from whetstone.testing.toy.experiment import TOY_MUTATION_FIELD
from whetstone.testing.toy.scoring import score_generation

__all__ = ["FakeEvaluationDriver"]


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


def _task_gold(task: object) -> str:
    gold = getattr(task, "gold", "")
    return gold if isinstance(gold, str) else ""


class FakeEvaluationDriver:
    """Deterministic evaluation driver for toy experiments (no network)."""

    def __init__(
        self,
        *,
        mutation_field: str = TOY_MUTATION_FIELD,
        render_contract: TemplateRenderContract | None = None,
        aggregate_name: str = "score",
    ) -> None:
        self._mutation_field = mutation_field
        self._aggregate_name = aggregate_name
        if render_contract is None:
            from whetstone.testing.toy.experiment import toy_template_render_contract

            render_contract = toy_template_render_contract()
        self._render_contract = render_contract

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
        request: EvalRequest,
        execution_policy: ProviderExecutionPolicy,
        concurrency: int,
        max_wall_seconds: float | None,
        partial_log: PartialLog | None,
        prompt_cache: PromptResultCache | None,
    ) -> InternalEvalResult:
        _ = (
            execution_policy,
            concurrency,
            max_wall_seconds,
            partial_log,
            prompt_cache,
        )
        self.preflight(request.candidate)
        num_samples = sampling.sample_plan.num_samples
        outputs: list[GenerationRowOutput] = []
        task_rows: list[TaskRows] = []
        task_hashes = sampling.task_set.task_hashes
        for task_index, (task, task_hash) in enumerate(
            zip(sampling.tasks, task_hashes, strict=True)
        ):
            row_values: list[RowValue] = []
            for sample_index in range(num_samples):
                prompt = self.rendered_prompt(request.candidate, task, max_budget=None)
                generation = prompt
                score = score_generation(
                    generation=generation,
                    gold=_task_gold(task),
                    task_id=_task_id(task),
                )
                outputs.append(
                    GenerationRowOutput(
                        candidate_id=request.candidate.candidate_id,
                        task_id=_task_id(task),
                        task_index=task_index,
                        sample_index=sample_index,
                        row_state=ExecutedRowState.SUCCESS,
                        executed_component_steps=(),
                        output_text=generation,
                        score=score,
                        submission_result=None,
                    )
                )
                row_values.append(RowValue(value=score))
            task_rows.append(TaskRows(task_hash=task_hash, rows=tuple(row_values)))

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
            outputs=tuple(outputs),
        )
