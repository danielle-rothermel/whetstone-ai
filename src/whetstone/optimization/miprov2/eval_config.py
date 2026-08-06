from __future__ import annotations

from typing import Literal, Protocol

from dr_code.eval import RepeatPlan, SamplingConfig, TaskSet
from dr_code.eval.identity import (
    SCHEMA_EVAL_CONFIG,
    SCHEMA_SAMPLING_CONFIG,
    identity_hash_for,
)
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
    rollout_id: StrictInt | None = None
    copy_task_model: StrictBool = False

    @model_validator(mode="after")
    def _validate_policy(self) -> Miprov2EvaluationExecutionPolicy:
        if self.num_threads is not None and self.num_threads <= 0:
            raise ValueError("num_threads must be positive when present")
        if self.max_errors <= 0:
            raise ValueError("max_errors must be positive")
        if self.rollout_id is not None and self.rollout_id < 0:
            raise ValueError("rollout_id cannot be negative")
        for field in (
            "task_model_identity_hash",
            "provider_execution_policy_hash",
        ):
            require_full_hash(getattr(self, field), field=field)
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema="whetstone.miprov2_evaluation_execution_policy",
            schema_version=1,
            payload=self.model_dump(mode="json"),
        )


def _canonical_eval_config_identity(eval_config: EvalConfigRef) -> str:
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
        ("config_identity_hash", record.config_identity_hash),
    ):
        require_full_hash(value, field=f"{field}.{nested_field}")
    if eval_config.identity_hash != _canonical_eval_config_identity(
        eval_config
    ):
        raise ValueError(f"{field} has a non-canonical Eval Config identity")


def _canonical_sampling_config_identity(
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
        sampling_config.config_identity_hash,
        field="sampling_config.config_identity_hash",
    )
    expected_sampling_identity = _canonical_sampling_config_identity(
        sampling_config
    )
    if sampling_config.config_identity_hash != expected_sampling_identity:
        raise ValueError("sampling_config has a non-canonical identity")
    source = source_eval_config.record
    derived = source.model_copy(
        update={
            "sampling_config_hash": sampling_config.config_identity_hash,
            "config_identity_hash": identity_hash_for(
                schema=SCHEMA_EVAL_CONFIG,
                payload={
                    "definition_identity": source.definition_ref.identity_hash,
                    "sampling_config": sampling_config.config_identity_hash,
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
    task_batch_identities: tuple[StrictStr, ...]
    repeat_count: StrictInt = 1

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
        if not self.task_batch_identities or self.repeat_count <= 0:
            raise ValueError(
                "Eval Config derivation requires tasks and positive repeats"
            )
        if len(set(self.task_batch_identities)) != len(
            self.task_batch_identities
        ):
            raise ValueError("Eval Config derivation tasks must be unique")
        for task_identity in self.task_batch_identities:
            require_full_hash(
                task_identity,
                field="task_batch_identity",
            )
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema="whetstone.miprov2_eval_binding_request",
            schema_version=1,
            payload=self.model_dump(mode="json"),
        )


class Miprov2EvalConfigBinding(BaseModel):
    """Auditable derivation of one exact ordered-task Eval Config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: Miprov2EvalConfigBindingRequest
    task_set: TaskSet
    repeat_plan: RepeatPlan
    sampling_config: SamplingConfig
    eval_config: EvalConfigRef

    @model_validator(mode="after")
    def _validate_binding(self) -> Miprov2EvalConfigBinding:
        tasks = self.request.task_batch_identities
        if (
            self.task_set.task_identities != tasks
            or self.task_set.selection_rule is not None
        ):
            raise ValueError("bound Task Set has the wrong ordered tasks")
        if (
            self.repeat_plan.task_identities != tasks
            or self.repeat_plan.repeat_count != self.request.repeat_count
        ):
            raise ValueError("bound Repeat Plan conflicts with request")
        task_set_identity = self.task_set.identity_hash()
        repeat_plan_identity = self.repeat_plan.identity_hash()
        require_full_hash(task_set_identity, field="task_set.identity_hash")
        require_full_hash(
            repeat_plan_identity,
            field="repeat_plan.identity_hash",
        )
        expected_assignment = (
            ("task_set_hash", task_set_identity),
            ("repeat_plan_hash", repeat_plan_identity),
        )
        if self.sampling_config.assignment != expected_assignment:
            raise ValueError(
                "bound Sampling Config does not bind the exact ordered "
                "Task Set and Repeat Plan"
            )
        require_full_hash(
            self.sampling_config.definition_ref.identity_hash,
            field="sampling_config.definition_ref.identity_hash",
        )
        expected_sampling_identity = _canonical_sampling_config_identity(
            self.sampling_config
        )
        if (
            self.sampling_config.config_identity_hash
            != expected_sampling_identity
        ):
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

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema="whetstone.miprov2_eval_binding",
            schema_version=1,
            payload=self.model_dump(mode="json"),
        )


class Miprov2EvalConfigResolver(Protocol):
    """Injected authority that materializes ordered-task Eval Configs."""

    def resolve(
        self,
        request: Miprov2EvalConfigBindingRequest,
    ) -> Miprov2EvalConfigBinding: ...


__all__ = [
    "Miprov2EvalConfigBinding",
    "Miprov2EvalConfigBindingRequest",
    "Miprov2EvalConfigResolver",
    "Miprov2EvalPurpose",
    "Miprov2EvaluationExecutionPolicy",
    "derive_eval_config_reference",
]
