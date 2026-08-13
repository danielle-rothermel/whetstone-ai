from __future__ import annotations

from typing import Any

from dr_store import ObjectStore
from dr_store.content_addressing import format_object_reference, parse_object_reference
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, model_validator

from whetstone.coordination.eval_service import EvalDispatchMode
from whetstone.core.identity import require_full_hash

OPTIM_PIPELINE_KEY = "whetstone.optim.v1"
OPTIM_PIPELINE_VERSION = 1

STAGE_OPTIM_STEP = "optim_step"
STAGE_EVAL_ROW = "eval_row"
STAGE_EVAL_FANIN = "eval_fanin"
STAGE_RUN_COMPLETION = "run_completion"

OPTIM_WORK_INPUT_SCHEMA = "whetstone.optim_work_input"
OPTIM_WORK_INPUT_SCHEMA_VERSION = 1


class OptimWorkInput(BaseModel):
    """Content-addressed platform member payload for one optimization run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: StrictInt = OPTIM_WORK_INPUT_SCHEMA_VERSION
    run_id: StrictStr
    controller_identity_hash: StrictStr
    control_identity_hash: StrictStr
    dispatch_mode: EvalDispatchMode = EvalDispatchMode.INLINE

    @model_validator(mode="after")
    def _validate(self) -> OptimWorkInput:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        require_full_hash(
            self.controller_identity_hash,
            field="controller_identity_hash",
        )
        require_full_hash(
            self.control_identity_hash,
            field="control_identity_hash",
        )
        if self.schema_version != OPTIM_WORK_INPUT_SCHEMA_VERSION:
            raise ValueError("schema_version is fixed")
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def persist_work_input(store: ObjectStore, work_input: OptimWorkInput) -> str:
    reference, _ = store.put(
        OPTIM_WORK_INPUT_SCHEMA,
        work_input.record_content(),
    )
    return format_object_reference(reference)


def load_work_input(store: ObjectStore, input_reference: str) -> OptimWorkInput:
    parsed = parse_object_reference(input_reference)
    if parsed.schema != OPTIM_WORK_INPUT_SCHEMA:
        raise ValueError("work input reference has the wrong schema")
    record = store.get(parsed)
    return OptimWorkInput.model_validate(record)


__all__ = [
    "OPTIM_PIPELINE_KEY",
    "OPTIM_PIPELINE_VERSION",
    "OPTIM_WORK_INPUT_SCHEMA",
    "OPTIM_WORK_INPUT_SCHEMA_VERSION",
    "STAGE_EVAL_FANIN",
    "STAGE_EVAL_ROW",
    "STAGE_OPTIM_STEP",
    "STAGE_RUN_COMPLETION",
    "OptimWorkInput",
    "load_work_input",
    "persist_work_input",
]
