from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.core.identity import (
    ImmutableJsonObject,
    compute_identity_hash,
    require_full_hash,
)
from whetstone.evaluation import SamplePlan, SamplingConfig, TaskSet
from whetstone.evaluation.config import (
    SCHEMA_EVAL_CONFIG,
    SCHEMA_SAMPLING_CONFIG,
    identity_hash_for,
)
from whetstone.experiment.binding import (
    EvalConfigRef,
    eval_config_reference,
)

Miprov2EvalPurpose = Literal[
    "bootstrap",
    "baseline",
    "sample",
    "promotion",
]

MIPROV2_EVALUATION_EXECUTION_POLICY_SCHEMA = (
    "whetstone.miprov2_evaluation_execution_policy"
)
MIPROV2_EVALUATION_EXECUTION_POLICY_SCHEMA_VERSION = 1
MIPROV2_EVAL_BINDING_REQUEST_SCHEMA = "whetstone.miprov2_eval_binding_request"
MIPROV2_EVAL_BINDING_REQUEST_SCHEMA_VERSION = 1
MIPROV2_EVAL_BINDING_SCHEMA = "whetstone.miprov2_eval_binding"
MIPROV2_EVAL_BINDING_SCHEMA_VERSION = 1


class Miprov2EvaluationExecutionPolicy(BaseModel):
    """Effect-specific, identity-bearing evaluator and provider controls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_threads: StrictInt | None
    max_errors: StrictInt
    provide_traceback: StrictBool | None
    task_model_identity_hash: StrictStr
    provider_execution_policy_hash: StrictStr
    provider_parameters: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    generation_id: StrictInt | None = None
    copy_task_model: StrictBool = False

    @model_validator(mode="after")
    def _validate_policy(self) -> Miprov2EvaluationExecutionPolicy:
        if self.num_threads is not None and self.num_threads <= 0:
            raise ValueError("num_threads must be positive when present")
        if self.max_errors <= 0:
            raise ValueError("max_errors must be positive")
        if self.generation_id is not None and self.generation_id < 0:
            raise ValueError("generation_id cannot be negative")
        for field in (
            "task_model_identity_hash",
            "provider_execution_policy_hash",
        ):
            require_full_hash(getattr(self, field), field=field)
        return self

    def model_post_init(self, _context: Any) -> None:
        if not isinstance(self.provider_parameters, ImmutableJsonObject):
            object.__setattr__(
                self,
                "provider_parameters",
                ImmutableJsonObject(self.provider_parameters),
            )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if deep:
            payload = self.model_dump(mode="json")
            payload.update(update or {})
            return type(self).model_validate(payload)
        copied = super().model_copy(update=update, deep=deep)
        copied.model_post_init(None)
        return copied

    def identity_payload(self) -> dict[str, Any]:
        return {
            "num_threads": self.num_threads,
            "max_errors": self.max_errors,
            "provide_traceback": self.provide_traceback,
            "task_model_identity_hash": self.task_model_identity_hash,
            "provider_execution_policy_hash": (
                self.provider_execution_policy_hash
            ),
            "provider_parameters": self.provider_parameters.to_json(),
            "generation_id": self.generation_id,
            "copy_task_model": self.copy_task_model,
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_EVALUATION_EXECUTION_POLICY_SCHEMA,
            schema_version=(
                MIPROV2_EVALUATION_EXECUTION_POLICY_SCHEMA_VERSION
            ),
            payload=self.identity_payload(),
        )


def _canonical_eval_config_hash(eval_config: EvalConfigRef) -> str:
    record = eval_config.record
    return identity_hash_for(
        schema=SCHEMA_EVAL_CONFIG,
        payload={
            "definition_identity": record.definition_ref.identity_hash,
            "sampling_config": record.sampling_config_hash,
            "evaluation_procedure_config": (
                record.evaluation_procedure_config_hash
            ),
            "aggregation_config": record.aggregation_config_hash,
        },
    )


def _require_canonical_eval_config(
    eval_config: EvalConfigRef,
    *,
    field: str,
) -> None:
    record = eval_config.record
    for nested_field, value in (
        ("definition_ref.identity_hash", record.definition_ref.identity_hash),
        ("sampling_config_hash", record.sampling_config_hash),
        (
            "evaluation_procedure_config_hash",
            record.evaluation_procedure_config_hash,
        ),
        ("aggregation_config_hash", record.aggregation_config_hash),
        ("config_hash", record.config_hash),
    ):
        require_full_hash(value, field=f"{field}.{nested_field}")
    if eval_config.config_hash != _canonical_eval_config_hash(eval_config):
        raise ValueError(f"{field} has a non-canonical Eval Config identity")
    expected_reference = eval_config_reference(record)
    if eval_config.record_ref != expected_reference.record_ref:
        raise ValueError(
            f"{field} record_ref does not address its exact record"
        )


def _canonical_sampling_config_hash(
    sampling_config: SamplingConfig,
) -> str:
    return identity_hash_for(
        schema=SCHEMA_SAMPLING_CONFIG,
        payload={
            "definition_identity": (
                sampling_config.definition_ref.identity_hash
            ),
            "assignment": [
                [name, value] for name, value in sampling_config.assignment
            ],
        },
    )


def derive_eval_config_reference(
    source_eval_config: EvalConfigRef,
    sampling_config: SamplingConfig,
) -> EvalConfigRef:
    """Derive the exact Eval Config obtained by replacing only sampling."""

    _require_canonical_eval_config(
        source_eval_config,
        field="source_eval_config",
    )
    require_full_hash(
        sampling_config.definition_ref.identity_hash,
        field="sampling_config.definition_ref.identity_hash",
    )
    require_full_hash(
        sampling_config.config_hash,
        field="sampling_config.config_hash",
    )
    expected_sampling_hash = _canonical_sampling_config_hash(sampling_config)
    if sampling_config.config_hash != expected_sampling_hash:
        raise ValueError("sampling_config has a non-canonical identity")
    source = source_eval_config.record
    derived = source.model_copy(
        update={
            "sampling_config_hash": sampling_config.config_hash,
            "config_hash": identity_hash_for(
                schema=SCHEMA_EVAL_CONFIG,
                payload={
                    "definition_identity": source.definition_ref.identity_hash,
                    "sampling_config": sampling_config.config_hash,
                    "evaluation_procedure_config": (
                        source.evaluation_procedure_config_hash
                    ),
                    "aggregation_config": source.aggregation_config_hash,
                },
            ),
        }
    )
    return eval_config_reference(derived)


class Miprov2EvalConfigBindingRequest(BaseModel):
    """Request for one exact ordered-task Eval Config derivation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    control_identity_hash: StrictStr
    source_eval_config: EvalConfigRef
    purpose: Miprov2EvalPurpose
    effect_identity_hash: StrictStr
    execution_policy: Miprov2EvaluationExecutionPolicy
    task_batch_hashes: tuple[StrictStr, ...]
    num_samples: StrictInt = 1

    @model_validator(mode="after")
    def _validate_request(self) -> Miprov2EvalConfigBindingRequest:
        require_full_hash(
            self.control_identity_hash,
            field="control_identity_hash",
        )
        require_full_hash(
            self.effect_identity_hash,
            field="effect_identity_hash",
        )
        _require_canonical_eval_config(
            self.source_eval_config,
            field="source_eval_config",
        )
        if not self.task_batch_hashes or self.num_samples <= 0:
            raise ValueError(
                "Eval Config derivation requires tasks and positive repeats"
            )
        if len(set(self.task_batch_hashes)) != len(self.task_batch_hashes):
            raise ValueError("Eval Config derivation tasks must be unique")
        for task_hash in self.task_batch_hashes:
            require_full_hash(
                task_hash,
                field="task_batch_hash",
            )
        return self

    def identity_payload(self) -> dict[str, Any]:
        # Persisted identity keys are an explicit wire contract. Exact refs
        # and the nested policy use their canonical identity projections.
        return {
            "control_identity_hash": self.control_identity_hash,
            "source_eval_config": self.source_eval_config.model_dump(
                mode="json"
            ),
            "purpose": self.purpose,
            "effect_identity_hash": self.effect_identity_hash,
            "execution_policy": self.execution_policy.identity_payload(),
            "task_batch_hashes": list(self.task_batch_hashes),
            "num_samples": self.num_samples,
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_EVAL_BINDING_REQUEST_SCHEMA,
            schema_version=MIPROV2_EVAL_BINDING_REQUEST_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class Miprov2EvalConfigBinding(BaseModel):
    """Auditable derivation of one exact ordered-task Eval Config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: Miprov2EvalConfigBindingRequest
    task_set: TaskSet
    sample_plan: SamplePlan
    sampling_config: SamplingConfig
    eval_config: EvalConfigRef

    @model_validator(mode="after")
    def _validate_binding(self) -> Miprov2EvalConfigBinding:
        tasks = self.request.task_batch_hashes
        if (
            self.task_set.task_hashes != tasks
            or self.task_set.selection_rule is not None
        ):
            raise ValueError("bound Task Set has the wrong ordered tasks")
        if (
            self.sample_plan.task_hashes != tasks
            or self.sample_plan.num_samples != self.request.num_samples
        ):
            raise ValueError("bound Sample Plan conflicts with request")
        task_set_hash = self.task_set.identity_hash()
        sample_plan_hash = self.sample_plan.identity_hash()
        require_full_hash(task_set_hash, field="task_set.identity_hash")
        require_full_hash(
            sample_plan_hash,
            field="sample_plan.identity_hash",
        )
        expected_assignment = (
            ("task_set_hash", task_set_hash),
            ("sample_plan_hash", sample_plan_hash),
        )
        if self.sampling_config.assignment != expected_assignment:
            raise ValueError(
                "bound Sampling Config does not bind the exact ordered "
                "Task Set and Sample Plan"
            )
        require_full_hash(
            self.sampling_config.definition_ref.identity_hash,
            field="sampling_config.definition_ref.identity_hash",
        )
        expected_sampling_hash = _canonical_sampling_config_hash(
            self.sampling_config
        )
        if self.sampling_config.config_hash != expected_sampling_hash:
            raise ValueError("bound Sampling Config identity is not canonical")
        expected_eval_config = derive_eval_config_reference(
            self.request.source_eval_config,
            self.sampling_config,
        )
        if self.eval_config != expected_eval_config:
            raise ValueError(
                "bound Eval Config is not the canonical source derivation"
            )
        return self

    def identity_payload(self) -> dict[str, Any]:
        # Persisted identity keys are an explicit wire contract. Nested
        # records and exact refs use their canonical JSON projections.
        return {
            "request": self.request.identity_payload(),
            "task_set": self.task_set.model_dump(mode="json"),
            "sample_plan": self.sample_plan.model_dump(mode="json"),
            "sampling_config": self.sampling_config.model_dump(mode="json"),
            "eval_config": self.eval_config.model_dump(mode="json"),
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_EVAL_BINDING_SCHEMA,
            schema_version=MIPROV2_EVAL_BINDING_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class Miprov2EvalConfigResolver(Protocol):
    """Injected authority that materializes ordered-task Eval Configs."""

    def resolve(
        self,
        request: Miprov2EvalConfigBindingRequest,
    ) -> Miprov2EvalConfigBinding: ...


__all__ = [
    "MIPROV2_EVALUATION_EXECUTION_POLICY_SCHEMA",
    "MIPROV2_EVALUATION_EXECUTION_POLICY_SCHEMA_VERSION",
    "MIPROV2_EVAL_BINDING_REQUEST_SCHEMA",
    "MIPROV2_EVAL_BINDING_REQUEST_SCHEMA_VERSION",
    "MIPROV2_EVAL_BINDING_SCHEMA",
    "MIPROV2_EVAL_BINDING_SCHEMA_VERSION",
    "Miprov2EvalConfigBinding",
    "Miprov2EvalConfigBindingRequest",
    "Miprov2EvalConfigResolver",
    "Miprov2EvalPurpose",
    "Miprov2EvaluationExecutionPolicy",
    "derive_eval_config_reference",
]
