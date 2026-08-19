from __future__ import annotations

from typing import Any
from uuid import uuid4

from dr_store import ObjectStore
from dr_store.content_addressing import format_object_reference, parse_object_reference
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, model_validator

from whetstone.coordination.eval_service import EvalDispatchMode
from whetstone.core.identity import require_full_hash
from whetstone.optim.contracts import OptimEvalRequest

OPTIM_PIPELINE_KEY = "whetstone.optim.v1"
OPTIM_PIPELINE_VERSION = 1

STAGE_OPTIM_STEP = "optim_step"
STAGE_EVAL_ROW = "eval_row"
STAGE_EVAL_FANIN = "eval_fanin"
STAGE_RUN_COMPLETION = "run_completion"

OPTIM_WORK_INPUT_SCHEMA = "whetstone.optim_work_input"
OPTIM_WORK_INPUT_SCHEMA_VERSION = 1

PLATFORM_EVAL_ROW_INPUT_SCHEMA = "whetstone.platform_eval_row_input"
PLATFORM_EVAL_ROW_INPUT_SCHEMA_VERSION = 1
PLATFORM_EVAL_FANIN_INPUT_SCHEMA = "whetstone.platform_eval_fanin_input"
PLATFORM_EVAL_FANIN_INPUT_SCHEMA_VERSION = 1
PLATFORM_EVAL_BATCH_SCHEMA = "whetstone.platform_eval_batch"
PLATFORM_EVAL_BATCH_SCHEMA_VERSION = 1
PLATFORM_RUN_MANIFEST_SCHEMA = "whetstone.platform_run_manifest"
PLATFORM_RUN_MANIFEST_SCHEMA_VERSION = 1


class OptimWorkInput(BaseModel):
    """Content-addressed platform member payload for one optimization run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: StrictInt = OPTIM_WORK_INPUT_SCHEMA_VERSION
    run_id: StrictStr
    controller_identity_hash: StrictStr
    control_identity_hash: StrictStr
    dispatch_mode: EvalDispatchMode = EvalDispatchMode.INLINE
    platform_stage_index: StrictInt = 0
    platform_run_key: StrictStr = ""
    work_key: StrictStr = ""

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
        if self.platform_stage_index < 0:
            raise ValueError("platform_stage_index must be non-negative")
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EvalRowInput(BaseModel):
    """One platform eval-row work item."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: StrictInt = PLATFORM_EVAL_ROW_INPUT_SCHEMA_VERSION
    batch_id: StrictStr
    optim_eval_request: OptimEvalRequest
    task_id: StrictStr
    seed_index: StrictInt

    @model_validator(mode="after")
    def _validate(self) -> EvalRowInput:
        if not self.batch_id:
            raise ValueError("batch_id must be non-empty")
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if self.seed_index < 0:
            raise ValueError("seed_index must be non-negative")
        if self.schema_version != PLATFORM_EVAL_ROW_INPUT_SCHEMA_VERSION:
            raise ValueError("schema_version is fixed")
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EvalFaninInput(BaseModel):
    """Join input after all eval rows in one batch succeed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: StrictInt = PLATFORM_EVAL_FANIN_INPUT_SCHEMA_VERSION
    batch_id: StrictStr
    optim_eval_request: OptimEvalRequest

    @model_validator(mode="after")
    def _validate(self) -> EvalFaninInput:
        if not self.batch_id:
            raise ValueError("batch_id must be non-empty")
        if self.schema_version != PLATFORM_EVAL_FANIN_INPUT_SCHEMA_VERSION:
            raise ValueError("schema_version is fixed")
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EvalBatch(BaseModel):
    """Fan-out manifest linking deferred intents to eval-row inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: StrictInt = PLATFORM_EVAL_BATCH_SCHEMA_VERSION
    batch_id: StrictStr
    run_id: StrictStr
    step_index: StrictInt
    optim_step_stage_index: StrictInt
    row_input_refs: tuple[StrictStr, ...]
    fanin_input_ref: StrictStr
    work_state_ref: StrictStr
    pending_step_result_ref: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> EvalBatch:
        if not self.batch_id:
            raise ValueError("batch_id must be non-empty")
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if self.optim_step_stage_index < 0:
            raise ValueError("optim_step_stage_index must be non-negative")
        if not self.row_input_refs:
            raise ValueError("row_input_refs must be non-empty")
        if not self.fanin_input_ref:
            raise ValueError("fanin_input_ref must be non-empty")
        if not self.work_state_ref:
            raise ValueError("work_state_ref must be non-empty")
        if not self.pending_step_result_ref:
            raise ValueError("pending_step_result_ref must be non-empty")
        if self.schema_version != PLATFORM_EVAL_BATCH_SCHEMA_VERSION:
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


def persist_eval_row_input(store: ObjectStore, row_input: EvalRowInput) -> str:
    reference, _ = store.put(
        PLATFORM_EVAL_ROW_INPUT_SCHEMA,
        row_input.record_content(),
    )
    return format_object_reference(reference)


def load_eval_row_input(store: ObjectStore, input_reference: str) -> EvalRowInput:
    parsed = parse_object_reference(input_reference)
    if parsed.schema != PLATFORM_EVAL_ROW_INPUT_SCHEMA:
        raise ValueError("eval row input reference has the wrong schema")
    record = store.get(parsed)
    return EvalRowInput.model_validate(record)


def persist_eval_fanin_input(
    store: ObjectStore,
    fanin_input: EvalFaninInput,
) -> str:
    reference, _ = store.put(
        PLATFORM_EVAL_FANIN_INPUT_SCHEMA,
        fanin_input.record_content(),
    )
    return format_object_reference(reference)


def load_eval_fanin_input(store: ObjectStore, input_reference: str) -> EvalFaninInput:
    parsed = parse_object_reference(input_reference)
    if parsed.schema != PLATFORM_EVAL_FANIN_INPUT_SCHEMA:
        raise ValueError("eval fan-in input reference has the wrong schema")
    record = store.get(parsed)
    return EvalFaninInput.model_validate(record)


def persist_eval_batch(store: ObjectStore, batch: EvalBatch) -> str:
    reference, _ = store.put(
        PLATFORM_EVAL_BATCH_SCHEMA,
        batch.record_content(),
    )
    formatted = format_object_reference(reference)
    store.bind(f"whetstone.platform_eval_batch:{batch.batch_id}", reference)
    return formatted


def load_eval_batch(store: ObjectStore, input_reference: str) -> EvalBatch:
    parsed = parse_object_reference(input_reference)
    if parsed.schema != PLATFORM_EVAL_BATCH_SCHEMA:
        raise ValueError("eval batch reference has the wrong schema")
    record = store.get(parsed)
    return EvalBatch.model_validate(record)


def load_eval_batch_by_id(store: ObjectStore, batch_id: str) -> EvalBatch:
    bound = store.resolve(f"whetstone.platform_eval_batch:{batch_id}")
    if bound is None:
        raise ValueError(f"eval batch is not bound: {batch_id!r}")
    return EvalBatch.model_validate(store.get(bound))


def new_batch_id() -> str:
    return uuid4().hex


class OptimRunMemberEntry(BaseModel):
    """One optimization member in a platform run manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    work_key: StrictStr
    run_id: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> OptimRunMemberEntry:
        if not self.work_key:
            raise ValueError("work_key must be non-empty")
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        return self


class OptimRunManifest(BaseModel):
    """Run-level manifest referenced by dr-platform run completion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: StrictInt = PLATFORM_RUN_MANIFEST_SCHEMA_VERSION
    platform_run_key: StrictStr
    membership_digest: StrictStr
    members: tuple[OptimRunMemberEntry, ...]

    @model_validator(mode="after")
    def _validate(self) -> OptimRunManifest:
        if not self.platform_run_key:
            raise ValueError("platform_run_key must be non-empty")
        if not self.membership_digest:
            raise ValueError("membership_digest must be non-empty")
        if not self.members:
            raise ValueError("members must be non-empty")
        if self.schema_version != PLATFORM_RUN_MANIFEST_SCHEMA_VERSION:
            raise ValueError("schema_version is fixed")
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def persist_run_manifest(store: ObjectStore, manifest: OptimRunManifest) -> str:
    reference, _ = store.put(
        PLATFORM_RUN_MANIFEST_SCHEMA,
        manifest.record_content(),
    )
    return format_object_reference(reference)


def load_run_manifest(store: ObjectStore, manifest_reference: str) -> OptimRunManifest:
    parsed = parse_object_reference(manifest_reference)
    if parsed.schema != PLATFORM_RUN_MANIFEST_SCHEMA:
        raise ValueError("run manifest reference has the wrong schema")
    record = store.get(parsed)
    return OptimRunManifest.model_validate(record)


__all__ = [
    "EvalBatch",
    "EvalFaninInput",
    "EvalRowInput",
    "OPTIM_PIPELINE_KEY",
    "OPTIM_PIPELINE_VERSION",
    "OPTIM_WORK_INPUT_SCHEMA",
    "OPTIM_WORK_INPUT_SCHEMA_VERSION",
    "PLATFORM_EVAL_BATCH_SCHEMA",
    "PLATFORM_EVAL_FANIN_INPUT_SCHEMA",
    "PLATFORM_EVAL_ROW_INPUT_SCHEMA",
    "STAGE_EVAL_FANIN",
    "STAGE_EVAL_ROW",
    "STAGE_OPTIM_STEP",
    "STAGE_RUN_COMPLETION",
    "OptimRunManifest",
    "OptimRunMemberEntry",
    "OptimWorkInput",
    "PLATFORM_RUN_MANIFEST_SCHEMA",
    "load_eval_batch",
    "load_run_manifest",
    "load_eval_batch_by_id",
    "load_eval_fanin_input",
    "load_eval_row_input",
    "load_work_input",
    "new_batch_id",
    "persist_eval_batch",
    "persist_eval_fanin_input",
    "persist_eval_row_input",
    "persist_run_manifest",
    "persist_work_input",
]
