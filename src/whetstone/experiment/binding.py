from __future__ import annotations

from typing import Any, Literal

from dr_code.eval import EvalConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from whetstone.core.identity import (
    IdentityHash,
    IdentityRef,
    NonEmptyId,
    NonNegativeInt,
    TypedRef,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.core.roles import EvaluationRole
from whetstone.provider.policy import PROVIDER_EXECUTION_POLICY_SCHEMA

EVAL_CONFIG_RECORD_SCHEMA = "dr_code.eval_config"
EVALUATION_BINDING_SCHEMA = "whetstone.evaluation_binding"
EVALUATION_BINDING_SCHEMA_VERSION = 2


def _require_ordered_sequence(value: Any, info: ValidationInfo) -> Any:
    if type(value) not in (list, tuple):
        raise ValueError(
            f"{info.field_name} must be an ordered tuple or JSON array"
        )
    return value


__all__ = [
    "EVALUATION_BINDING_SCHEMA",
    "EVALUATION_BINDING_SCHEMA_VERSION",
    "EVAL_CONFIG_RECORD_SCHEMA",
    "EvalConfigRef",
    "EvaluationBinding",
    "ExecutionEnvironmentFingerprint",
    "eval_config_reference",
]


class EvalConfigRef(BaseModel):
    """Exact typed Eval Config record and both addressing dimensions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: EvalConfig
    record_ref: TypedRef
    identity_hash: IdentityHash

    @model_validator(mode="after")
    def _validate(self) -> EvalConfigRef:
        if self.identity_hash != self.record.config_identity_hash:
            raise ValueError(
                "Eval Config identity_hash must match the exact typed record"
            )
        expected = typed_ref_for_record(
            EVAL_CONFIG_RECORD_SCHEMA, self.record.model_dump(mode="json")
        )
        if self.record_ref != expected:
            raise ValueError(
                "Eval Config record_ref must address the exact typed record"
            )
        return self


def eval_config_reference(eval_config: EvalConfig) -> EvalConfigRef:
    return EvalConfigRef(
        record=eval_config,
        record_ref=typed_ref_for_record(
            EVAL_CONFIG_RECORD_SCHEMA, eval_config.model_dump(mode="json")
        ),
        identity_hash=eval_config.config_identity_hash,
    )


class ExecutionEnvironmentFingerprint(BaseModel):
    """Exact realized dependency, code, and runtime environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dependency_versions: tuple[tuple[NonEmptyId, NonEmptyId], ...] = ()
    code_revision: NonEmptyId | None = None
    runtime_identity: NonEmptyId | None = None

    @field_validator("dependency_versions", mode="before")
    @classmethod
    def _validate_dependency_input(
        cls, value: Any, info: ValidationInfo
    ) -> Any:
        ordered = _require_ordered_sequence(value, info)
        for index, dependency in enumerate(ordered):
            if type(dependency) not in (list, tuple):
                raise ValueError(
                    "dependency_versions entries must be ordered "
                    f"package/version pairs; entry {index} is unordered"
                )
        return ordered

    @field_validator("dependency_versions")
    @classmethod
    def _canonicalize_dependencies(
        cls,
        value: tuple[tuple[NonEmptyId, NonEmptyId], ...],
    ) -> tuple[tuple[NonEmptyId, NonEmptyId], ...]:
        package_names = [package for package, _ in value]
        if len(set(package_names)) != len(package_names):
            raise ValueError("dependency package names must be unique")
        # Dependency order has no semantic meaning. Canonicalize it so the
        # same realized environment has one stable identity.
        return tuple(sorted(value, key=lambda dependency: dependency[0]))


class EvaluationBinding(BaseModel):
    """Immutable policy and environment binding for one evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2]
    eval_config: EvalConfigRef
    role: EvaluationRole
    authority_principal: NonEmptyId | None = None
    campaign: NonEmptyId
    provider_execution_policy_ref: IdentityRef | None = None
    retry_policy_ref: TypedRef | None = None
    operational_policy_refs: tuple[TypedRef, ...] = ()
    environment_fingerprint: ExecutionEnvironmentFingerprint = Field(
        default_factory=ExecutionEnvironmentFingerprint
    )
    provenance_note: NonEmptyId | None = None
    provenance_ordinal: NonNegativeInt | None = None

    @field_validator("provider_execution_policy_ref")
    @classmethod
    def _validate_provider_execution_policy_ref(
        cls, value: IdentityRef | None
    ) -> IdentityRef | None:
        if (
            value is not None
            and value.record_ref.schema_name
            != PROVIDER_EXECUTION_POLICY_SCHEMA
        ):
            raise ValueError(
                "provider_execution_policy_ref must use schema "
                f"{PROVIDER_EXECUTION_POLICY_SCHEMA!r}"
            )
        return value

    @field_validator("operational_policy_refs", mode="before")
    @classmethod
    def _validate_operational_policy_refs(
        cls, value: Any, info: ValidationInfo
    ) -> Any:
        return _require_ordered_sequence(value, info)

    @model_validator(mode="after")
    def _validate(self) -> EvaluationBinding:
        if (
            self.role is EvaluationRole.OFFICIAL
            and self.authority_principal is None
        ):
            raise ValueError(
                "authority_principal is required for official evaluation"
            )
        if (
            self.role is EvaluationRole.INTERNAL
            and self.authority_principal is not None
        ):
            raise ValueError(
                "authority_principal must be absent for internal evaluation"
            )
        return self

    def identity_payload(self) -> dict[str, Any]:
        # These persisted identity keys are an explicit wire contract. Never
        # derive them by iterating over model fields.
        return {
            "schema_version": self.schema_version,
            "eval_config": self.eval_config.model_dump(mode="json"),
            "role": self.role.value,
            "authority_principal": self.authority_principal,
            "campaign": self.campaign,
            "provider_execution_policy_ref": (
                None
                if self.provider_execution_policy_ref is None
                else self.provider_execution_policy_ref.model_dump(mode="json")
            ),
            "retry_policy_ref": (
                None
                if self.retry_policy_ref is None
                else self.retry_policy_ref.model_dump(mode="json")
            ),
            "operational_policy_refs": [
                ref.model_dump(mode="json")
                for ref in self.operational_policy_refs
            ],
            "environment_fingerprint": self.environment_fingerprint.model_dump(
                mode="json"
            ),
            "provenance_note": self.provenance_note,
            "provenance_ordinal": self.provenance_ordinal,
        }

    def identity_hash(self) -> IdentityHash:
        return compute_identity_hash(
            schema=EVALUATION_BINDING_SCHEMA,
            schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
