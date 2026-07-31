"""Exact serializable contracts for Tool definitions, calls, and results."""

from __future__ import annotations

from collections.abc import Callable
from enum import UNIQUE, StrEnum, verify
from typing import Any

from dr_code.eval import EvalConfig
from dr_code.eval.identity import SCHEMA_EVAL_CONFIG
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationInfo,
    field_validator,
    model_validator,
)

from whetstone.optimization.identity import (
    IdentityHash,
    ImmutableJsonObject,
    NonEmptyId,
    NonNegativeInt,
    OpaqueKey,
    TerminalFailure,
    TypedRef,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.optimization.reward import RewardRef

__all__ = [
    "EVAL_CONFIG_SCHEMA",
    "GLOBAL_CAPACITY_SCOPE_ID",
    "RUN_CAPACITY_SUBJECT_SCHEMA",
    "STEP_CAPACITY_SUBJECT_SCHEMA",
    "TOOL_CALL_SCHEMA",
    "TOOL_CONFIG_SCHEMA",
    "TOOL_CONFIG_SCHEMA_VERSION",
    "TOOL_DEFINITION_SCHEMA",
    "TOOL_DEFINITION_SCHEMA_VERSION",
    "TOOL_RESULT_SCHEMA",
    "RefusalClass",
    "RuntimeToolHandle",
    "ToolCall",
    "ToolCallRef",
    "ToolCapacity",
    "ToolCapacityBinding",
    "ToolCapacityScope",
    "ToolConfig",
    "ToolConfigRef",
    "ToolDefinition",
    "ToolDefinitionRef",
    "ToolRefusal",
    "ToolResult",
    "ToolResultRef",
    "tool_call_reference",
    "tool_capacity_binding",
    "tool_config_reference",
    "tool_definition_reference",
    "tool_result_reference",
]

TOOL_DEFINITION_SCHEMA = "whetstone.tool_definition"
TOOL_DEFINITION_SCHEMA_VERSION = 1
TOOL_CONFIG_SCHEMA = "whetstone.tool_config"
TOOL_CONFIG_SCHEMA_VERSION = 1
TOOL_CALL_SCHEMA = "whetstone.tool_call"
TOOL_RESULT_SCHEMA = "whetstone.tool_result"
EVAL_CONFIG_SCHEMA = SCHEMA_EVAL_CONFIG
GLOBAL_CAPACITY_SCOPE_ID = "global"
RUN_CAPACITY_SUBJECT_SCHEMA = "whetstone.optimization_run"
STEP_CAPACITY_SUBJECT_SCHEMA = "whetstone.optimization_step_request"


def _require_ordered_sequence(value: Any, info: ValidationInfo) -> Any:
    """Accept only the deliberate Python representations of a JSON array."""
    if type(value) not in (list, tuple):
        raise ValueError(
            f"{info.field_name} must be an ordered tuple or JSON array"
        )
    return value


@verify(UNIQUE)
class RefusalClass(StrEnum):
    """Closed pre-execution refusal taxonomy."""

    AUTHORIZATION = "authorization"
    CAPACITY = "capacity"
    BUDGET = "budget"
    VALIDATION = "validation"


@verify(UNIQUE)
class ToolCapacityScope(StrEnum):
    """Identity domain within which accepted-call capacity is counted."""

    GLOBAL = "global"
    RUN = "run"
    STEP = "step"


class ToolCapacityBinding(BaseModel):
    """Exact authority subject whose accepted calls share one capacity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: ToolCapacityScope
    subject_ref: TypedRef | None = None

    @model_validator(mode="after")
    def _validate(self) -> ToolCapacityBinding:
        if self.scope is ToolCapacityScope.GLOBAL:
            if self.subject_ref is not None:
                raise ValueError(
                    "GLOBAL Tool Capacity requires no subject_ref"
                )
            return self
        if self.subject_ref is None:
            raise ValueError(
                f"{self.scope.value.upper()} Tool Capacity requires "
                "subject_ref"
            )
        expected_schema = (
            RUN_CAPACITY_SUBJECT_SCHEMA
            if self.scope is ToolCapacityScope.RUN
            else STEP_CAPACITY_SUBJECT_SCHEMA
        )
        if self.subject_ref.schema_name != expected_schema:
            raise ValueError(
                f"{self.scope.value.upper()} Tool Capacity subject_ref must "
                f"use schema {expected_schema!r}"
            )
        return self

    @property
    def capacity_scope_id(self) -> NonEmptyId:
        if self.scope is ToolCapacityScope.GLOBAL:
            return NonEmptyId(GLOBAL_CAPACITY_SCOPE_ID)
        assert self.subject_ref is not None
        return NonEmptyId(self.subject_ref.content_hash)


def tool_capacity_binding(
    scope: ToolCapacityScope,
    subject_ref: TypedRef | None = None,
) -> ToolCapacityBinding:
    return ToolCapacityBinding(scope=scope, subject_ref=subject_ref)


class ToolDefinition(BaseModel):
    """Versioned interface definition for one Tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: NonEmptyId
    version: NonNegativeInt = NonNegativeInt(1)
    input_fields: tuple[NonEmptyId, ...]
    output_fields: tuple[NonEmptyId, ...]
    refusal_classes: tuple[RefusalClass, ...] = tuple(RefusalClass)
    expansion_semantics: NonEmptyId | None = None

    @field_validator(
        "input_fields",
        "output_fields",
        "refusal_classes",
        mode="before",
    )
    @classmethod
    def _validate_ordered_fields(cls, value: Any, info: ValidationInfo) -> Any:
        return _require_ordered_sequence(value, info)

    @model_validator(mode="after")
    def _validate(self) -> ToolDefinition:
        if self.version == 0:
            raise ValueError("Tool Definition version must be positive")
        if not self.input_fields:
            raise ValueError("a Tool Definition must declare input_fields")
        if not self.output_fields:
            raise ValueError("a Tool Definition must declare output_fields")
        for label, values in (
            ("input_fields", self.input_fields),
            ("output_fields", self.output_fields),
            ("refusal_classes", self.refusal_classes),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"Tool Definition {label} must be unique")
        if not self.refusal_classes:
            raise ValueError("a Tool Definition must declare refusal_classes")
        return self

    def identity_payload(self) -> dict[str, Any]:
        # Persisted identity keys are a pinned wire contract.
        return {
            "tool_name": self.tool_name,
            "version": self.version,
            "input_fields": list(self.input_fields),
            "output_fields": list(self.output_fields),
            "refusal_classes": [
                refusal_class.value for refusal_class in self.refusal_classes
            ],
            "expansion_semantics": self.expansion_semantics,
        }

    def identity_hash(self) -> IdentityHash:
        return compute_identity_hash(
            schema=TOOL_DEFINITION_SCHEMA,
            schema_version=TOOL_DEFINITION_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ToolDefinitionRef(BaseModel):
    """An exact Tool Definition record plus both addressing dimensions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: ToolDefinition
    record_ref: TypedRef
    identity_hash: IdentityHash

    @model_validator(mode="after")
    def _validate(self) -> ToolDefinitionRef:
        expected_ref = typed_ref_for_record(
            TOOL_DEFINITION_SCHEMA, self.record.record_content()
        )
        if self.record_ref != expected_ref:
            raise ValueError(
                "Tool Definition record_ref must address the exact record"
            )
        if self.identity_hash != self.record.identity_hash():
            raise ValueError(
                "Tool Definition identity_hash must match the exact record"
            )
        return self


def tool_definition_reference(
    definition: ToolDefinition,
) -> ToolDefinitionRef:
    return ToolDefinitionRef(
        record=definition,
        record_ref=typed_ref_for_record(
            TOOL_DEFINITION_SCHEMA, definition.record_content()
        ),
        identity_hash=definition.identity_hash(),
    )


class ToolCapacity(BaseModel):
    """Accepted-call ceiling and its explicit identity scope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_accepted_calls: NonNegativeInt
    scope: ToolCapacityScope


class ToolConfig(BaseModel):
    """Complete serializable config materialized from one exact Definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    definition: ToolDefinitionRef
    endpoint_key: OpaqueKey
    eval_config: EvalConfig
    reward_policy_hash: IdentityHash
    capacity: ToolCapacity
    timeout_policy_ref: TypedRef | None = None
    operational_policy_refs: tuple[TypedRef, ...] = ()
    store_namespace_key: OpaqueKey
    idempotent_replay: StrictBool = True

    @field_validator("operational_policy_refs", mode="before")
    @classmethod
    def _validate_operational_policy_refs(
        cls, value: Any, info: ValidationInfo
    ) -> Any:
        return _require_ordered_sequence(value, info)

    @model_validator(mode="after")
    def _validate(self) -> ToolConfig:
        if len(set(self.operational_policy_refs)) != len(
            self.operational_policy_refs
        ):
            raise ValueError(
                "Tool Config operational_policy_refs must be unique"
            )
        IdentityHash(self.eval_config.config_identity_hash)
        return self

    @property
    def tool_name(self) -> str:
        return self.definition.record.tool_name

    @property
    def eval_config_ref(self) -> TypedRef:
        return typed_ref_for_record(
            EVAL_CONFIG_SCHEMA, self.eval_config.model_dump(mode="json")
        )

    @property
    def eval_config_identity_hash(self) -> IdentityHash:
        return IdentityHash(self.eval_config.config_identity_hash)

    def identity_payload(self) -> dict[str, Any]:
        # Persisted identity keys are a pinned wire contract.
        return {
            "definition": {
                "record": self.definition.record.identity_payload(),
                "record_ref": {
                    "schema_name": self.definition.record_ref.schema_name,
                    "content_hash": self.definition.record_ref.content_hash,
                },
                "identity_hash": self.definition.identity_hash,
            },
            "endpoint_key": self.endpoint_key,
            "eval_config": {
                "definition_ref": {
                    "definition_id": (
                        self.eval_config.definition_ref.definition_id
                    ),
                    "version": self.eval_config.definition_ref.version,
                    "schema_name": (
                        self.eval_config.definition_ref.schema_name
                    ),
                    "identity_hash": (
                        self.eval_config.definition_ref.identity_hash
                    ),
                },
                "sampling_config_hash": (
                    self.eval_config.sampling_config_hash
                ),
                "evaluation_procedure_config_hash": (
                    self.eval_config.evaluation_procedure_config_hash
                ),
                "aggregation_config_hash": (
                    self.eval_config.aggregation_config_hash
                ),
                "config_identity_hash": (
                    self.eval_config.config_identity_hash
                ),
            },
            "reward_policy_hash": self.reward_policy_hash,
            "capacity": {
                "max_accepted_calls": self.capacity.max_accepted_calls,
                "scope": self.capacity.scope.value,
            },
            "timeout_policy_ref": (
                None
                if self.timeout_policy_ref is None
                else {
                    "schema_name": self.timeout_policy_ref.schema_name,
                    "content_hash": self.timeout_policy_ref.content_hash,
                }
            ),
            "operational_policy_refs": [
                {
                    "schema_name": reference.schema_name,
                    "content_hash": reference.content_hash,
                }
                for reference in self.operational_policy_refs
            ],
            "store_namespace_key": self.store_namespace_key,
            "idempotent_replay": self.idempotent_replay,
        }

    def identity_hash(self) -> IdentityHash:
        return compute_identity_hash(
            schema=TOOL_CONFIG_SCHEMA,
            schema_version=TOOL_CONFIG_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ToolConfigRef(BaseModel):
    """An exact Tool Config record plus both addressing dimensions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: ToolConfig
    record_ref: TypedRef
    identity_hash: IdentityHash

    @model_validator(mode="after")
    def _validate(self) -> ToolConfigRef:
        expected_ref = typed_ref_for_record(
            TOOL_CONFIG_SCHEMA, self.record.record_content()
        )
        if self.record_ref != expected_ref:
            raise ValueError(
                "Tool Config record_ref must address the exact record"
            )
        if self.identity_hash != self.record.identity_hash():
            raise ValueError(
                "Tool Config identity_hash must match the exact record"
            )
        return self


def tool_config_reference(config: ToolConfig) -> ToolConfigRef:
    return ToolConfigRef(
        record=config,
        record_ref=typed_ref_for_record(
            TOOL_CONFIG_SCHEMA, config.record_content()
        ),
        identity_hash=config.identity_hash(),
    )


class RuntimeToolHandle:
    """Non-serializable callable constructed only at execution."""

    __slots__ = ("_binding", "_config", "_execute")

    def __init__(
        self,
        config: ToolConfig,
        binding: ToolCapacityBinding,
        execute: Callable[[ToolCall], ToolResult],
    ) -> None:
        if binding.scope is not config.capacity.scope:
            raise ValueError(
                "Runtime Tool binding scope must match the exact Tool Config"
            )
        self._config = config
        self._binding = binding
        self._execute = execute

    @property
    def config(self) -> ToolConfig:
        return self._config

    @property
    def binding(self) -> ToolCapacityBinding:
        return self._binding

    @property
    def tool_config_ref(self) -> ToolConfigRef:
        return tool_config_reference(self._config)

    @property
    def tool_config_hash(self) -> IdentityHash:
        return self.tool_config_ref.identity_hash

    def __call__(self, call: ToolCall) -> ToolResult:
        if call.tool_config != self.tool_config_ref:
            raise ValueError(
                "Tool Call config must match the Runtime Tool Handle"
            )
        if call.capacity_binding != self._binding:
            raise ValueError(
                "Tool Call capacity binding must match the Runtime Tool Handle"
            )
        return self._execute(call)


class ToolCall(BaseModel):
    """One exact config-bound Tool Call and its immutable arguments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: NonEmptyId
    tool_config: ToolConfigRef
    capacity_binding: ToolCapacityBinding
    args: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )

    @model_validator(mode="after")
    def _validate(self) -> ToolCall:
        if self.capacity_binding.scope is not self.capacity_scope:
            raise ValueError(
                "Tool Call capacity binding scope must match the exact "
                "Tool Config"
            )
        expected = set(self.tool_config.record.definition.record.input_fields)
        actual = set(self.args)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                "Tool Call args must exactly match Definition input_fields; "
                f"missing={missing}, unexpected={unexpected}"
            )
        return self

    @property
    def tool_config_hash(self) -> IdentityHash:
        return self.tool_config.identity_hash

    @property
    def store_namespace_key(self) -> OpaqueKey:
        return self.tool_config.record.store_namespace_key

    @property
    def capacity_scope(self) -> ToolCapacityScope:
        return self.tool_config.record.capacity.scope

    @property
    def capacity_scope_id(self) -> NonEmptyId:
        return self.capacity_binding.capacity_scope_id

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ToolCallRef(BaseModel):
    """An exact persisted Tool Call record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: ToolCall
    record_ref: TypedRef

    @model_validator(mode="after")
    def _validate(self) -> ToolCallRef:
        expected = typed_ref_for_record(
            TOOL_CALL_SCHEMA, self.record.record_content()
        )
        if self.record_ref != expected:
            raise ValueError(
                "Tool Call record_ref must address the exact call record"
            )
        return self


def tool_call_reference(call: ToolCall) -> ToolCallRef:
    return ToolCallRef(
        record=call,
        record_ref=typed_ref_for_record(
            TOOL_CALL_SCHEMA, call.record_content()
        ),
    )


class ToolRefusal(BaseModel):
    """Typed pre-execution outcome; it never consumes capacity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    refusal_class: RefusalClass
    reason: NonEmptyId


class ToolResult(BaseModel):
    """Exactly success, pre-execution refusal, or terminal failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call: ToolCallRef
    output: ImmutableJsonObject | None = None
    refusal: ToolRefusal | None = None
    terminal_failure: TerminalFailure | None = None
    evaluation_evidence_refs: tuple[TypedRef, ...] = ()
    reward: RewardRef | None = None
    provenance_note: NonEmptyId | None = None
    provenance_ordinal: NonNegativeInt | None = None

    @field_validator("evaluation_evidence_refs", mode="before")
    @classmethod
    def _validate_evidence_refs(cls, value: Any, info: ValidationInfo) -> Any:
        return _require_ordered_sequence(value, info)

    @model_validator(mode="after")
    def _validate(self) -> ToolResult:
        terminal_variants = (
            self.output is not None,
            self.refusal is not None,
            self.terminal_failure is not None,
        )
        if sum(terminal_variants) != 1:
            raise ValueError(
                "a Tool Result must model exactly success, pre-execution "
                "refusal, or terminal failure"
            )
        definition = self.call.record.tool_config.record.definition.record
        if len(set(self.evaluation_evidence_refs)) != len(
            self.evaluation_evidence_refs
        ):
            raise ValueError(
                "Tool Result evaluation_evidence_refs must be unique"
            )
        if self.refusal is not None:
            if self.refusal.refusal_class not in definition.refusal_classes:
                raise ValueError(
                    "Tool Refusal class is not declared by the exact "
                    "Tool Definition"
                )
            if self.evaluation_evidence_refs or self.reward is not None:
                raise ValueError(
                    "a pre-execution refusal carries no evaluation evidence "
                    "or Reward"
                )
        if self.terminal_failure is not None and self.reward is not None:
            raise ValueError("a failed Tool Result carries no Reward")
        if (
            self.reward is not None
            and self.reward.record.reward_policy_hash
            != self.tool_config.record.reward_policy_hash
        ):
            raise ValueError(
                "Tool Result Reward policy must match the exact Tool Config"
            )
        if (
            self.reward is not None
            and self.reward.record.evidence_refs
            != self.evaluation_evidence_refs
        ):
            raise ValueError(
                "Tool Result Reward evidence_refs must exactly equal "
                "evaluation_evidence_refs"
            )
        if self.output is not None:
            expected = set(definition.output_fields)
            actual = set(self.output)
            if actual != expected:
                missing = sorted(expected - actual)
                unexpected = sorted(actual - expected)
                raise ValueError(
                    "Tool Result output must exactly match Definition "
                    f"output_fields; missing={missing}, "
                    f"unexpected={unexpected}"
                )
        if self.refusal is not None:
            if self.provenance_ordinal is not None:
                raise ValueError(
                    "a pre-execution refusal has no provenance ordinal"
                )
        elif self.provenance_ordinal is None or self.provenance_ordinal == 0:
            raise ValueError(
                "a non-refused Tool Result requires a positive provenance "
                "ordinal"
            )
        return self

    @property
    def call_id(self) -> NonEmptyId:
        return self.call.record.call_id

    @property
    def tool_config(self) -> ToolConfigRef:
        return self.call.record.tool_config

    @property
    def tool_config_hash(self) -> IdentityHash:
        return self.tool_config.identity_hash

    @property
    def store_namespace_key(self) -> OpaqueKey:
        return self.call.record.store_namespace_key

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ToolResultRef(BaseModel):
    """An exact persisted terminal Tool Result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: ToolResult
    record_ref: TypedRef

    @model_validator(mode="after")
    def _validate(self) -> ToolResultRef:
        expected = typed_ref_for_record(
            TOOL_RESULT_SCHEMA, self.record.record_content()
        )
        if self.record_ref != expected:
            raise ValueError(
                "Tool Result record_ref must address the exact result record"
            )
        return self


def tool_result_reference(result: ToolResult) -> ToolResultRef:
    return ToolResultRef(
        record=result,
        record_ref=typed_ref_for_record(
            TOOL_RESULT_SCHEMA, result.record_content()
        ),
    )
