from __future__ import annotations

import importlib
from collections.abc import Mapping
from enum import UNIQUE, StrEnum, verify
from typing import Any

from dr_graph import GraphConfig
from dr_providers import ProviderCallConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    StrictInt,
    StrictStr,
    field_validator,
)

from whetstone.eval.drivers.row_common import RolloutRowOutput
from whetstone.eval.traces import ExecutedComponentStep, ExecutedRowState

__all__ = [
    "GraphRowRequest",
    "RowDispatchStatus",
    "decode_graph_row_output",
    "import_path_for_callable",
    "import_path_for_type",
    "resolve_import_path",
    "rollout_row_output_from_worker_payload",
    "worker_request_identities",
]


@verify(UNIQUE)
class RowDispatchStatus(StrEnum):
    """How one dispatched rollout row ended, from the driver's view.

    The values are this driver's row failure-code vocabulary and are
    persisted in row outputs, so they are pinned by a golden test.
    """

    COMPLETED = "completed"
    UNIT_TIMEOUT = "unit-timeout"
    OPERATION_DEADLINE = "deadline"
    NOT_DISPATCHED = "not-dispatched"


def _split_import_path(path: str) -> tuple[str, str]:
    module_name, separator, object_name = path.partition(":")
    if (
        not separator
        or not module_name
        or not object_name
        or ":" in object_name
        or "." in object_name
        or any(not part.isidentifier() for part in module_name.split("."))
        or not object_name.isidentifier()
    ):
        raise ValueError(
            "import path must be 'importable.module:top_level_name'"
        )
    return module_name, object_name


def resolve_import_path(path: str) -> object:
    """Import a top-level ``module:name`` symbol for worker reconstruction."""
    module_name, object_name = _split_import_path(path)
    module = importlib.import_module(module_name)
    try:
        candidate = getattr(module, object_name)
    except AttributeError as exc:
        raise ValueError(
            f"{path!r} does not name an attribute of {module_name}"
        ) from exc
    if (
        getattr(candidate, "__module__", None) != module_name
        or getattr(candidate, "__name__", None) != object_name
    ):
        raise TypeError(
            f"{path!r} does not name a top-level object defined in its module"
        )
    return candidate


def _import_path_for_top_level(obj: object) -> str:
    name = getattr(obj, "__name__", None)
    module_name = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(module_name, str)
        or not module_name
    ):
        raise ValueError("object is not a top-level importable symbol")
    if qualname != name:
        raise ValueError(
            f"{name!r} is not a top-level symbol (qualname={qualname!r})"
        )
    path = f"{module_name}:{name}"
    resolved = resolve_import_path(path)
    if resolved is not obj:
        raise ValueError(f"{path!r} does not resolve to the provided object")
    return path


def import_path_for_callable(obj: object) -> str:
    if not callable(obj):
        raise TypeError("object is not callable")
    return _import_path_for_top_level(obj)


def import_path_for_type(obj: type[Any]) -> str:
    if not isinstance(obj, type):
        raise TypeError("object is not a type")
    return _import_path_for_top_level(obj)


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
    execution_policy: JsonValue
    execution_policy_hash: StrictStr
    prompt_inputs: dict[str, StrictStr] = {}
    gold: StrictStr = ""
    transport_api_key_env: StrictStr = "WHETSTONE_TOY_API_KEY"
    transport_factory: StrictStr
    eval_runner: StrictStr
    """Top-level ``module:Name`` of a zero-arg, stateless ``EvalProcedureRunner``.

    Workers reconstruct via ``runner_type()`` and transfer no constructor
    state. Required-arg constructors fail with TypeError; defaulted instance
    state silently rebuilds with defaults. If a stateful runner is ever
    needed, adopt the prompt-adapter ``model_dump`` / ``model_validate``
    payload pattern.
    """
    prompt_adapter_type: StrictStr
    prompt_adapter: JsonValue
    partial_log_path: StrictStr | None = None
    prompt_cache_path: StrictStr | None = None

    @field_validator("transport_factory", "eval_runner", "prompt_adapter_type")
    @classmethod
    def _collaborator_path_is_top_level(cls, value: str) -> str:
        _split_import_path(value)
        return value

    @field_validator("prompt_adapter")
    @classmethod
    def _prompt_adapter_is_object(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("prompt_adapter must be a JSON object")
        return value

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
        prompt_tokens=(
            None
            if payload.get("prompt_tokens") is None
            else int(payload["prompt_tokens"])  # type: ignore[call-overload]
        ),
        completion_tokens=(
            None
            if payload.get("completion_tokens") is None
            else int(payload["completion_tokens"])  # type: ignore[call-overload]
        ),
        provider_cost=(
            None
            if payload.get("provider_cost") is None
            else float(payload["provider_cost"])  # type: ignore[arg-type]
        ),
        submission_result=payload.get("submission_result"),
    )


def worker_request_identities(payload: Mapping[str, object]) -> tuple[str, ...]:
    raw = payload.get("request_identities", ())
    if not isinstance(raw, list):
        return ()
    return tuple(str(item) for item in raw if isinstance(item, str))


def decode_graph_row_output(
    payload: Mapping[str, object],
    *,
    request: GraphRowRequest,
    dispatch_status: RowDispatchStatus | None = None,
) -> RolloutRowOutput:
    if dispatch_status is RowDispatchStatus.NOT_DISPATCHED:
        return RolloutRowOutput(
            candidate_id=request.candidate_id,
            task_id=request.task_id,
            task_index=request.task_index,
            seed_index=request.seed_index,
            row_state=ExecutedRowState.MISSING,
            trace_steps=(),
            output_text=None,
            score=None,
            failure_code=RowDispatchStatus.NOT_DISPATCHED.value,
        )
    if dispatch_status is RowDispatchStatus.UNIT_TIMEOUT:
        return RolloutRowOutput(
            candidate_id=request.candidate_id,
            task_id=request.task_id,
            task_index=request.task_index,
            seed_index=request.seed_index,
            row_state=ExecutedRowState.MISSING,
            trace_steps=(),
            output_text=None,
            score=None,
            failure_code=RowDispatchStatus.UNIT_TIMEOUT.value,
        )
    if dispatch_status is RowDispatchStatus.OPERATION_DEADLINE:
        return RolloutRowOutput(
            candidate_id=request.candidate_id,
            task_id=request.task_id,
            task_index=request.task_index,
            seed_index=request.seed_index,
            row_state=ExecutedRowState.MISSING,
            trace_steps=(),
            output_text=None,
            score=None,
            failure_code=RowDispatchStatus.OPERATION_DEADLINE.value,
        )
    return rollout_row_output_from_worker_payload(payload)
