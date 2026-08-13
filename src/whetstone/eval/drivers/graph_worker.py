from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from whetstone.eval.drivers.graph_row import execute_rollout_graph, graph_result_to_row_fields
from whetstone.eval.drivers.graph_row_request import GraphRowRequest
from whetstone.eval.drivers.row_common import RolloutRowOutput
from whetstone.eval.eval_procedure import EvalProcedureRunner
from whetstone.eval.protocol import EvalTaskView
from whetstone.execution._file_lock import ensure_private_directory
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.graph.llm_call_run_node import (
    EvalRunNodeDeps,
    LlmCallRunNodeDeps,
    ProviderCallConfigResolver,
)
from whetstone.experiment.graph.run_node_registry import build_run_node
from whetstone.provider.language_model import PlainPromptAdapter
from whetstone.provider.llm_call import LlmCallContext
from whetstone.provider.policy import ProviderExecutionPolicy
from whetstone.testing.fakes.eval_procedure import FakeEvalProcedureRunner
from whetstone.testing.fakes.transport import FakeLlmTransport

__all__ = ["run_row"]


@dataclass(frozen=True, slots=True)
class _WorkerTask:
    task_id: str
    prompt_inputs: dict[str, str]
    gold: str = ""


def _resolve_provider_call_config(
    request: GraphRowRequest,
) -> ProviderCallConfigResolver:
    provider_config = request.parsed_provider_call_config

    def resolve(_ref: object) -> Any:
        return provider_config

    return resolve  # type: ignore[return-value]


def _open_partial_log(path: str | None) -> PartialLog | None:
    if path is None:
        return None
    partial_path = Path(path).resolve()
    ensure_private_directory(partial_path.parent)
    return PartialLog(partial_path)


def _open_prompt_cache(path: str | None) -> PromptResultCache | None:
    if path is None:
        return None
    cache_root = Path(path).resolve()
    ensure_private_directory(cache_root)
    return PromptResultCache(root=cache_root)


def _rollout_row_output_for_request(
    request: GraphRowRequest,
    *,
    eval_runner: EvalProcedureRunner,
    execution_policy: ProviderExecutionPolicy,
) -> tuple[RolloutRowOutput, tuple[str, ...]]:
    task = _WorkerTask(
        task_id=request.task_id,
        prompt_inputs=dict(request.prompt_inputs),
        gold=request.gold,
    )
    partial_log = _open_partial_log(request.partial_log_path)
    prompt_cache = _open_prompt_cache(request.prompt_cache_path)
    row_identities: list[str] = []
    llm_context = LlmCallContext(
        execution_policy=execution_policy,
        transport=FakeLlmTransport(
            transport_policy=execution_policy.transport_policy
        ),
        prompt_adapter=PlainPromptAdapter(),
        prompt_cache=prompt_cache,
        partial_log=partial_log,
    )
    run_node = build_run_node(
        llm_deps=LlmCallRunNodeDeps(
            context=llm_context,
            resolve_provider_call_config=_resolve_provider_call_config(request),
            graph_hash=request.rollout_graph_hash,
            rng_seed=request.rng_seed,
            task_id=request.task_id,
            seed_index=request.seed_index,
            drive_ordinal=0,
            phase=request.split_role,
            unit=request.candidate_id,
            split_role=request.split_role,
            request_identity_sink=row_identities,
        ),
        eval_deps=EvalRunNodeDeps(runner=eval_runner, task=task),  # type: ignore[arg-type]
    )
    result = execute_rollout_graph(
        graph=request.parsed_graph_config,
        inputs={request.graph_external_input_field: request.rendered_prompt},
        run_node=run_node,
    )
    output = graph_result_to_row_fields(
        result,
        candidate_id=request.candidate_id,
        task_id=request.task_id,
        task_index=request.task_index,
        seed_index=request.seed_index,
    )
    return output, tuple(row_identities)


def run_row(payload: JsonValue) -> JsonValue:
    """Subprocess entrypoint for one graph rollout row."""
    request = GraphRowRequest.model_validate(payload)
    execution_policy = ProviderExecutionPolicy.model_validate(
        request.execution_policy
    )
    if execution_policy.identity_hash != request.execution_policy_hash:
        raise ValueError("execution_policy_hash does not match worker policy")
    output, row_identities = _rollout_row_output_for_request(
        request,
        eval_runner=FakeEvalProcedureRunner(),
        execution_policy=execution_policy,
    )
    return _rollout_row_output_to_json(output, row_identities)


def _rollout_row_output_to_json(
    output: RolloutRowOutput,
    request_identities: tuple[str, ...],
) -> dict[str, object]:
    return {
        "candidate_id": output.candidate_id,
        "task_id": output.task_id,
        "task_index": output.task_index,
        "seed_index": output.seed_index,
        "row_state": output.row_state.value,
        "trace_steps": [
            step.model_dump(mode="json") for step in output.trace_steps
        ],
        "output_text": output.output_text,
        "score": output.score,
        "failure_code": output.failure_code,
        "finish_reason": output.finish_reason,
        "provider_error": output.provider_error,
        "submission_result": output.submission_result,
        "request_identities": list(request_identities),
    }
