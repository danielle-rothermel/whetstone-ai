"""Tool lifecycle: Definition -> Config -> Runtime Handle -> Call/Result.

The optimizer tool boundary, per the vocabulary and Workstream 7:

* :class:`ToolDefinition` — a versioned, variable-bearing interface Definition
  owned by Whetstone. It declares the call input/output shapes, expansion
  semantics, refusal classes, and required provenance, and materializes one or
  more Tool Configs. Addressed by its Identity Hash.

* :class:`ToolConfig` — a complete, validated, **serializable** Config
  materialized from exactly one Tool Definition. It carries the typed Tool
  Definition reference and Identity Hash; endpoint/service identity; an
  ordinary Eval Config typed reference and Identity Hash bound through an
  ``internal`` Evaluation Role; a reusable Reward Policy reference; capacity;
  timeout and operational-policy references; the Tool Call Store namespace; and
  idempotency and refusal rules. It is addressed by its own Identity Hash —
  the ``tool_config_hash`` half of the Tool Call Store key. It carries **no**
  runtime handle, client, connection, or closure.

* :class:`RuntimeToolHandle` — a non-serializable bound callable constructed
  from a Tool Config **only at the execution boundary**. It is deliberately a
  plain (non-pydantic, non-JSON) object so it cannot be embedded in a Step
  Request, state, history, or Result; the request-validation lane rejects it.

* :class:`ToolCall`, :class:`ToolResult`, :class:`ToolRefusal` — the request,
  the immutable terminal result, and the typed non-execution outcome.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.optimization.identity import (
    TypedRef,
    compute_identity_hash,
    reject_non_json,
    require_full_hash,
)

__all__ = [
    "TOOL_CONFIG_SCHEMA",
    "TOOL_CONFIG_SCHEMA_VERSION",
    "TOOL_DEFINITION_SCHEMA",
    "TOOL_DEFINITION_SCHEMA_VERSION",
    "TOOL_RESULT_SCHEMA",
    "RefusalClass",
    "RuntimeToolHandle",
    "ToolCall",
    "ToolCapacity",
    "ToolConfig",
    "ToolDefinition",
    "ToolRefusal",
    "ToolResult",
    "tool_result_reference",
]

TOOL_DEFINITION_SCHEMA = "whetstone.tool_definition"
TOOL_DEFINITION_SCHEMA_VERSION = 1
TOOL_CONFIG_SCHEMA = "whetstone.tool_config"
TOOL_CONFIG_SCHEMA_VERSION = 1
# The Tool Result is a stored *record* (Content Hash), not an Identity.
TOOL_RESULT_SCHEMA = "whetstone.tool_result"


class RefusalClass(StrEnum):
    """Typed Tool Refusal classes (non-execution outcomes)."""

    AUTHORIZATION = "authorization"
    CAPACITY = "capacity"
    BUDGET = "budget"
    VALIDATION = "validation"


class ToolDefinition(BaseModel):
    """Versioned, variable-bearing Tool interface Definition.

    Declares the closed shape a Tool Config completes: the input/output field
    names, the fixed-axis expansion semantics, the declared refusal classes,
    and the required provenance fields. It has an Identity Hash and
    materializes one or more Tool Configs, which reference it by that hash.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: StrictStr
    version: StrictInt = 1
    input_fields: tuple[str, ...]
    output_fields: tuple[str, ...]
    refusal_classes: tuple[RefusalClass, ...] = tuple(RefusalClass)
    required_provenance_fields: tuple[str, ...] = ()
    expansion_semantics: StrictStr | None = None

    @model_validator(mode="after")
    def _validate(self) -> ToolDefinition:
        if not self.tool_name:
            raise ValueError("tool_name must be non-empty")
        if not self.input_fields:
            raise ValueError("a Tool Definition must declare input_fields")
        if not self.output_fields:
            raise ValueError("a Tool Definition must declare output_fields")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "version": self.version,
            "input_fields": list(self.input_fields),
            "output_fields": list(self.output_fields),
            "refusal_classes": [c.value for c in self.refusal_classes],
            "required_provenance_fields": list(
                self.required_provenance_fields
            ),
            "expansion_semantics": self.expansion_semantics,
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=TOOL_DEFINITION_SCHEMA,
            schema_version=TOOL_DEFINITION_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class ToolCapacity(BaseModel):
    """Declared maximum accepted calls under a scope/window.

    Consumption is accounted exactly once only by the Tool Call Store's
    absent->accepted transition; this record carries only the ceiling.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_accepted_calls: StrictInt
    scope: StrictStr = "run"

    @model_validator(mode="after")
    def _validate(self) -> ToolCapacity:
        if self.max_accepted_calls < 0:
            raise ValueError("max_accepted_calls cannot be negative")
        return self


class ToolConfig(BaseModel):
    """Complete validated serializable Tool Config.

    Carries the typed Tool Definition reference + Identity Hash and every
    binding the tool needs: endpoint identity, the ordinary Eval Config
    reference + Identity Hash (bound through an ``internal`` Evaluation Role),
    the Reward Policy reference, capacity, timeout/operational-policy refs, the
    Tool Call Store namespace, and idempotency + refusal rules. It carries no
    runtime handle: the whole record is strict JSON.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: StrictStr
    # Typed Tool Definition reference + Identity Hash of the owning Definition.
    tool_definition_ref: StrictStr
    tool_definition_identity_hash: StrictStr

    # Endpoint / service identity.
    endpoint: StrictStr

    # Ordinary Eval Config reference + Identity Hash, bound through an
    # ``internal`` Evaluation Role at execution.
    eval_config_ref: StrictStr
    eval_config_identity_hash: StrictStr

    # Reusable Reward Policy reference (Identity Hash of the policy).
    reward_policy_ref: StrictStr

    # Capacity and operational policy references.
    capacity: ToolCapacity
    timeout_policy_ref: StrictStr | None = None
    operational_policy_refs: tuple[str, ...] = ()

    # Tool Call Store namespace.
    store_namespace: StrictStr

    # Idempotency and refusal rules.
    idempotent_replay: bool = True
    refusal_classes: tuple[RefusalClass, ...] = tuple(RefusalClass)

    @model_validator(mode="after")
    def _validate(self) -> ToolConfig:
        if not self.tool_name:
            raise ValueError("tool_name must be non-empty")
        if not self.endpoint:
            raise ValueError("endpoint must be non-empty")
        if not self.store_namespace:
            raise ValueError("store_namespace must be non-empty")
        require_full_hash(
            self.tool_definition_identity_hash,
            field="tool_definition_identity_hash",
        )
        require_full_hash(
            self.eval_config_identity_hash,
            field="eval_config_identity_hash",
        )
        require_full_hash(self.reward_policy_ref, field="reward_policy_ref")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_definition_ref": self.tool_definition_ref,
            "tool_definition_identity_hash": (
                self.tool_definition_identity_hash
            ),
            "endpoint": self.endpoint,
            "eval_config_ref": self.eval_config_ref,
            "eval_config_identity_hash": self.eval_config_identity_hash,
            "reward_policy_ref": self.reward_policy_ref,
            "capacity": {
                "max_accepted_calls": self.capacity.max_accepted_calls,
                "scope": self.capacity.scope,
            },
            "timeout_policy_ref": self.timeout_policy_ref,
            "operational_policy_refs": list(self.operational_policy_refs),
            "store_namespace": self.store_namespace,
            "idempotent_replay": self.idempotent_replay,
            "refusal_classes": [c.value for c in self.refusal_classes],
        }

    def identity_hash(self) -> str:
        """The Tool Config Identity Hash — the ``tool_config_hash`` half of
        the Tool Call Store key."""
        return compute_identity_hash(
            schema=TOOL_CONFIG_SCHEMA,
            schema_version=TOOL_CONFIG_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class RuntimeToolHandle:
    """Non-serializable bound callable constructed from a Tool Config.

    Deliberately a plain object (not a pydantic model, not JSON): it holds the
    Tool Config and the live execution callable, and exists only during a
    tool-using Step's execution boundary. Because it is not JSON-able, it can
    never be embedded in a Step Request, state, history, or Result — the
    request-validation lane rejects any attempt to carry one.
    """

    __slots__ = ("_config", "_execute")

    def __init__(
        self,
        config: ToolConfig,
        execute: Callable[[ToolCall], ToolResult],
    ) -> None:
        self._config = config
        self._execute = execute

    @property
    def config(self) -> ToolConfig:
        return self._config

    @property
    def tool_config_hash(self) -> str:
        return self._config.identity_hash()

    def __call__(self, call: ToolCall) -> ToolResult:
        return self._execute(call)


class ToolCall(BaseModel):
    """A Tool Call: caller input plus a stable Tool Call ID.

    Carries the Tool Config Identity Hash and the Tool Call Store namespace so
    it addresses exactly one Tool Call Store Entry ``(tool_config_hash,
    call_id)`` and terminal Tool Result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: StrictStr
    tool_config_hash: StrictStr
    store_namespace: StrictStr
    args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> ToolCall:
        if not self.call_id:
            raise ValueError("call_id must be non-empty")
        require_full_hash(self.tool_config_hash, field="tool_config_hash")
        if not self.store_namespace:
            raise ValueError("store_namespace must be non-empty")
        reject_non_json(self.args, field="args")
        return self


class ToolRefusal(BaseModel):
    """Typed non-execution outcome; never masquerades as a measurement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    refusal_class: RefusalClass
    reason: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> ToolRefusal:
        if not self.reason:
            raise ValueError("refusal reason must be non-empty")
        return self


class ToolResult(BaseModel):
    """Immutable terminal result for exactly one Tool Call.

    Carries the Tool Config typed reference + Identity Hash and the terminal
    Tool Call Store Entry identity, and contains **either** typed output (with
    optional evaluation-evidence refs and a produced Reward) **or** a
    Refusal/failure — never an unnamed score field. Stored by Content Hash.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: StrictStr
    tool_config_ref: StrictStr
    tool_config_hash: StrictStr
    store_namespace: StrictStr

    # EITHER typed output OR a refusal (never both, never neither).
    output: dict[str, Any] | None = None
    refusal: ToolRefusal | None = None

    # Internal evaluation evidence produced by the tool's internal-role
    # evaluation (Rollout Result / aggregate refs), referenced not duplicated.
    evaluation_evidence_refs: tuple[TypedRef, ...] = ()
    # The named Reward the tool's Reward Policy produced (internal only).
    reward: dict[str, Any] | None = None

    provenance_note: StrictStr | None = None
    provenance_ordinal: StrictInt | None = None

    @model_validator(mode="after")
    def _validate(self) -> ToolResult:
        if not self.call_id:
            raise ValueError("call_id must be non-empty")
        require_full_hash(self.tool_config_hash, field="tool_config_hash")
        has_output = self.output is not None
        has_refusal = self.refusal is not None
        if has_output and has_refusal:
            raise ValueError(
                "a Tool Result carries EITHER output OR a refusal, never both"
            )
        if not has_output and not has_refusal:
            raise ValueError(
                "a Tool Result must carry typed output or a typed refusal"
            )
        if has_refusal and (self.evaluation_evidence_refs or self.reward):
            raise ValueError(
                "a refused Tool Result carries no evaluation evidence or "
                "Reward: a refusal never masquerades as a measurement"
            )
        if self.output is not None:
            reject_non_json(self.output, field="output")
        if self.reward is not None:
            reject_non_json(self.reward, field="reward")
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def tool_result_reference(result: ToolResult) -> TypedRef:
    """The typed Object Reference (by Content Hash) for a Tool Result."""
    from whetstone.optimization.identity import typed_ref_for_record

    return typed_ref_for_record(TOOL_RESULT_SCHEMA, result.record_content())
