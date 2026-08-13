from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dr_graph import GraphConfig
from dr_providers import ProviderCallConfig
from pydantic import BaseModel, ConfigDict, JsonValue, StrictInt, StrictStr

from whetstone.eval.drivers.row_common import RolloutRowOutput
from whetstone.eval.traces import ExecutedComponentStep, ExecutedRowState
from whetstone.execution.fanout import FanoutStatus

__all__ = [
    "GraphRowRequest",
    "decode_graph_row_output",
    "rollout_row_output_from_worker_payload",
]


class GraphRowRequest(BaseModel):
    """Strict JSON payload for one subprocess graph rollout row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: StrictStr
    task_id: StrictStr
    task_index: StrictInt
    seed_index: StrictInt
    split_role: StrictStr
    rendered_prompt: StrictStr
    graph_config: JsonValue
    rollout_graph_hash: StrictStr
    provider_call_config: JsonValue
    rng_seed: StrictInt
    mutation_field: StrictStr
    graph_external_input_field: StrictStr = "prompt"
    eval_procedure_config_hash: StrictStr
    execution_policy_hash: StrictStr
    prompt_inputs: dict[str, StrictStr] = {}
    gold: StrictStr = ""
    transport_api_key_env: StrictStr = "WHETSTONE_TOY_API_KEY"

    @property
    def parsed_graph_config(self) -> GraphConfig:
        if not isinstance(self.graph_config, dict):
            raise ValueError("graph_config must be a JSON object")
        return GraphConfig.model_validate(self.graph_config)

    @property
    def parsed_provider_call_config(self) -> ProviderCallConfig:
        if not isinstance(self.provider_call_config, dict):
            raise ValueError("provider_call_config must be a JSON object")
        return ProviderCallConfig.model_validate(self.provider_call_config)


def _normalize_trace_step_payload(item: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(item)
    for key in ("input_field_names", "output_field_names"):
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = tuple(str(field) for field in value)
    return normalized


def rollout_row_output_from_worker_payload(
    payload: Mapping[str, object],
) -> RolloutRowOutput:
    row_state_raw = payload.get("row_state")
    if isinstance(row_state_raw, str):
        row_state = ExecutedRowState(row_state_raw)
    else:
        row_state = ExecutedRowState.SUCCESS
    trace_steps_raw = payload.get("trace_steps", ())
    trace_steps_list: list[ExecutedComponentStep] = []
    if isinstance(trace_steps_raw, list):
        for item in trace_steps_raw:
            if isinstance(item, dict):
                trace_steps_list.append(
                    ExecutedComponentStep.model_validate(
                        _normalize_trace_step_payload(item)
                    )
                )
    trace_steps = tuple(trace_steps_list)
    return RolloutRowOutput(
        candidate_id=str(payload["candidate_id"]),
        task_id=str(payload["task_id"]),
        task_index=int(payload["task_index"]),
        seed_index=int(payload["seed_index"]),
        row_state=row_state,
        trace_steps=trace_steps,  # type: ignore[arg-type]
        output_text=(
            None
            if payload.get("output_text") is None
            else str(payload["output_text"])
        ),
        score=(
            None
            if payload.get("score") is None
            else float(payload["score"])  # type: ignore[arg-type]
        ),
        failure_code=str(payload.get("failure_code") or ""),
        finish_reason=(
            None
            if payload.get("finish_reason") is None
            else str(payload["finish_reason"])
        ),
        provider_error=(
            None
            if payload.get("provider_error") is None
            else dict(payload["provider_error"])  # type: ignore[arg-type]
        ),
        submission_result=payload.get("submission_result"),
    )


def decode_graph_row_output(
    payload: Mapping[str, object],
    *,
    request: GraphRowRequest,
    fanout_status: FanoutStatus | None = None,
) -> RolloutRowOutput:
    if fanout_status is FanoutStatus.NOT_DISPATCHED:
        return RolloutRowOutput(
            candidate_id=request.candidate_id,
            task_id=request.task_id,
            task_index=request.task_index,
            seed_index=request.seed_index,
            row_state=ExecutedRowState.MISSING,
            trace_steps=(),
            output_text=None,
            score=None,
            failure_code="not-dispatched",
        )
    if fanout_status is FanoutStatus.UNIT_TIMEOUT:
        return RolloutRowOutput(
            candidate_id=request.candidate_id,
            task_id=request.task_id,
            task_index=request.task_index,
            seed_index=request.seed_index,
            row_state=ExecutedRowState.MISSING,
            trace_steps=(),
            output_text=None,
            score=None,
            failure_code="unit-timeout",
        )
    if fanout_status is FanoutStatus.OPERATION_DEADLINE:
        return RolloutRowOutput(
            candidate_id=request.candidate_id,
            task_id=request.task_id,
            task_index=request.task_index,
            seed_index=request.seed_index,
            row_state=ExecutedRowState.MISSING,
            trace_steps=(),
            output_text=None,
            score=None,
            failure_code="deadline",
        )
    return rollout_row_output_from_worker_payload(payload)
