from __future__ import annotations

from typing import Any

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
PLATFORM_EVAL_ROW_INPUT_SCHEMA_VERSION = 2
PLATFORM_DEFERRAL_JOIN_INPUT_SCHEMA = "whetstone.platform_deferral_join_input"
PLATFORM_DEFERRAL_JOIN_INPUT_SCHEMA_VERSION = 2
PLATFORM_RUN_MANIFEST_SCHEMA = "whetstone.platform_run_manifest"
PLATFORM_RUN_MANIFEST_SCHEMA_VERSION = 1
PLATFORM_RUN_RESULT_SCHEMA = "whetstone.platform_run_result"
PLATFORM_RUN_RESULT_SCHEMA_VERSION = 1


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
    work_state_ref: StrictStr
    deferral_origin_stage_index: StrictInt
    row_ordinal: StrictInt
    optim_eval_request: OptimEvalRequest
    task_id: StrictStr
    seed_index: StrictInt

    @model_validator(mode="after")
    def _validate(self) -> EvalRowInput:
        if not self.work_state_ref:
            raise ValueError("work_state_ref must be non-empty")
        if self.deferral_origin_stage_index < 0:
            raise ValueError("deferral_origin_stage_index must be non-negative")
        if self.row_ordinal < 0:
            raise ValueError("row_ordinal must be non-negative")
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if self.seed_index < 0:
            raise ValueError("seed_index must be non-negative")
        if self.schema_version != PLATFORM_EVAL_ROW_INPUT_SCHEMA_VERSION:
            raise ValueError("schema_version is fixed")
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class DeferralJoinInput(BaseModel):
    """Join pointer for one deferred eval episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: StrictInt = PLATFORM_DEFERRAL_JOIN_INPUT_SCHEMA_VERSION
    work_state_ref: StrictStr
    deferral_optim_step_stage_index: StrictInt
    primary_optim_eval_request: OptimEvalRequest
    row_input_refs: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def _validate(self) -> DeferralJoinInput:
        if not self.work_state_ref:
            raise ValueError("work_state_ref must be non-empty")
        if self.deferral_optim_step_stage_index < 0:
            raise ValueError("deferral_optim_step_stage_index must be non-negative")
        if not self.row_input_refs:
            raise ValueError("row_input_refs must be non-empty")
        if any(not row_input_ref for row_input_ref in self.row_input_refs):
            raise ValueError("row_input_refs entries must be non-empty")
        if self.schema_version != PLATFORM_DEFERRAL_JOIN_INPUT_SCHEMA_VERSION:
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


def persist_deferral_join_input(
    store: ObjectStore,
    join_input: DeferralJoinInput,
) -> str:
    reference, _ = store.put(
        PLATFORM_DEFERRAL_JOIN_INPUT_SCHEMA,
        join_input.record_content(),
    )
    return format_object_reference(reference)


def load_deferral_join_input(
    store: ObjectStore,
    input_reference: str,
) -> DeferralJoinInput:
    parsed = parse_object_reference(input_reference)
    if parsed.schema != PLATFORM_DEFERRAL_JOIN_INPUT_SCHEMA:
        raise ValueError("deferral join input reference has the wrong schema")
    record = store.get(parsed)
    return DeferralJoinInput.model_validate(record)


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
        work_keys = tuple(member.work_key for member in self.members)
        run_ids = tuple(member.run_id for member in self.members)
        if len(set(work_keys)) != len(work_keys):
            raise ValueError("member work_key values must be unique")
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("member run_id values must be unique")
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


class OptimRunMemberResult(BaseModel):
    """One member's terminal OptimResult reference in a platform run result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    work_key: StrictStr
    run_id: StrictStr
    result_reference: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> OptimRunMemberResult:
        if not self.work_key:
            raise ValueError("work_key must be non-empty")
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.result_reference:
            raise ValueError("result_reference must be non-empty")
        return self


class OptimPlatformRunResult(BaseModel):
    """Run-level completion artifact referenced by dr-platform run output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: StrictInt = PLATFORM_RUN_RESULT_SCHEMA_VERSION
    platform_run_key: StrictStr
    membership_digest: StrictStr
    member_results: tuple[OptimRunMemberResult, ...]

    @model_validator(mode="after")
    def _validate(self) -> OptimPlatformRunResult:
        if not self.platform_run_key:
            raise ValueError("platform_run_key must be non-empty")
        if not self.membership_digest:
            raise ValueError("membership_digest must be non-empty")
        if not self.member_results:
            raise ValueError("member_results must be non-empty")
        work_keys = tuple(item.work_key for item in self.member_results)
        run_ids = tuple(item.run_id for item in self.member_results)
        if len(set(work_keys)) != len(work_keys):
            raise ValueError("member work_key values must be unique")
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("member run_id values must be unique")
        if self.schema_version != PLATFORM_RUN_RESULT_SCHEMA_VERSION:
            raise ValueError("schema_version is fixed")
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def persist_run_result(
    store: ObjectStore,
    run_result: OptimPlatformRunResult,
) -> str:
    reference, _ = store.put(
        PLATFORM_RUN_RESULT_SCHEMA,
        run_result.record_content(),
    )
    return format_object_reference(reference)


def load_run_result(
    store: ObjectStore,
    result_reference: str,
) -> OptimPlatformRunResult:
    parsed = parse_object_reference(result_reference)
    if parsed.schema != PLATFORM_RUN_RESULT_SCHEMA:
        raise ValueError("run result reference has the wrong schema")
    record = store.get(parsed)
    return OptimPlatformRunResult.model_validate(record)


__all__ = [
    "DeferralJoinInput",
    "EvalRowInput",
    "OPTIM_PIPELINE_KEY",
    "OPTIM_PIPELINE_VERSION",
    "OPTIM_WORK_INPUT_SCHEMA",
    "OPTIM_WORK_INPUT_SCHEMA_VERSION",
    "PLATFORM_DEFERRAL_JOIN_INPUT_SCHEMA",
    "PLATFORM_EVAL_ROW_INPUT_SCHEMA",
    "STAGE_EVAL_FANIN",
    "STAGE_EVAL_ROW",
    "STAGE_OPTIM_STEP",
    "STAGE_RUN_COMPLETION",
    "OptimPlatformRunResult",
    "OptimRunManifest",
    "OptimRunMemberEntry",
    "OptimRunMemberResult",
    "OptimWorkInput",
    "PLATFORM_RUN_MANIFEST_SCHEMA",
    "PLATFORM_RUN_RESULT_SCHEMA",
    "load_deferral_join_input",
    "load_eval_row_input",
    "load_run_manifest",
    "load_run_result",
    "load_work_input",
    "persist_deferral_join_input",
    "persist_eval_row_input",
    "persist_run_manifest",
    "persist_run_result",
    "persist_work_input",
]
