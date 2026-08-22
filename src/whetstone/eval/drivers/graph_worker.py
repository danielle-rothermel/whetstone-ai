from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, JsonValue

from whetstone.eval.drivers.graph_row import execute_rollout_graph, graph_result_to_row_fields
from whetstone.eval.drivers.graph_row_request import (
    GraphRowRequest,
    resolve_import_path,
)
from whetstone.eval.drivers.row_common import RolloutRowOutput
from whetstone.eval.eval_procedure import EvalProcedureRunner
from whetstone.eval.protocol import EvalTaskView
from dr_store.localfs import ensure_private_directory
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.graph.llm_call_run_node import (
    EvalRunNodeDeps,
    LlmCallRunNodeDeps,
    ProviderCallConfigResolver,
)
from whetstone.experiment.graph.run_node_registry import build_run_node
from whetstone.provider.driver import TransportCall
from whetstone.provider.language_model import (
    PlainPromptAdapter,
    StructuredPromptAdapter,
)
from whetstone.provider.llm_call import LlmCallContext
from whetstone.provider.policy import ProviderExecutionPolicy

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
    transport: TransportCall,
    prompt_adapter: PlainPromptAdapter | StructuredPromptAdapter,
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
        transport=transport,
        prompt_adapter=prompt_adapter,
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


def _reconstruct_transport(
    path: str, execution_policy: ProviderExecutionPolicy
) -> TransportCall:
    factory = resolve_import_path(path)
    if not callable(factory):
        raise TypeError(f"{path!r} does not resolve to a callable")
    transport = factory(execution_policy)
    if not callable(transport):
        raise TypeError(f"{path!r} did not return a transport callable")
    return transport


def _reconstruct_eval_runner(path: str) -> EvalProcedureRunner:
    runner_type = resolve_import_path(path)
    if not callable(runner_type):
        raise TypeError(f"{path!r} does not resolve to a callable")
    runner = runner_type()
    if not isinstance(runner, EvalProcedureRunner):
        raise TypeError(f"{path!r} did not construct an EvalProcedureRunner")
    return runner


def _reconstruct_prompt_adapter(
    path: str, payload: object
) -> PlainPromptAdapter | StructuredPromptAdapter:
    adapter_type = resolve_import_path(path)
    if not isinstance(adapter_type, type) or not issubclass(
        adapter_type, BaseModel
    ):
        raise TypeError(f"{path!r} does not resolve to a model type")
    adapter = adapter_type.model_validate(payload)
    if not isinstance(adapter, (PlainPromptAdapter, StructuredPromptAdapter)):
        raise TypeError(
            f"{path!r} did not reconstruct a supported prompt adapter"
        )
    return adapter


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
        eval_runner=_reconstruct_eval_runner(request.eval_runner),
        execution_policy=execution_policy,
        transport=_reconstruct_transport(
            request.transport_factory, execution_policy
        ),
        prompt_adapter=_reconstruct_prompt_adapter(
            request.prompt_adapter_type, request.prompt_adapter
        ),
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
        "prompt_tokens": output.prompt_tokens,
        "completion_tokens": output.completion_tokens,
        "provider_cost": output.provider_cost,
        "cache_hit": output.cache_hit,
        "submission_result": output.submission_result,
        "request_identities": list(request_identities),
    }
