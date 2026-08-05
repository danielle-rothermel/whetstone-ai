"""Exact-reference evidence bridge for durable MIPROv2 effects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from dr_store import BindingConflictError, ObjectStore
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.evaluation.schema import (
    EVALUATION_COMPONENT_TRACES_SCHEMA,
    EvaluationComponentTraces,
    EvaluationComponentTracesRef,
    EvaluationEvidence,
    EvaluationEvidenceRef,
    EvaluationFailureEvidence,
    EvaluationFailureEvidenceRef,
)
from whetstone.evaluation_role import EvaluationRole
from whetstone.optimization.identity import (
    ImmutableJsonObject,
    TypedRef,
    compute_identity_hash,
    require_full_hash,
)
from whetstone.optimization.miprov2_bootstrap import (
    BootstrapAttemptPlan,
    BootstrapRolloutResult,
)
from whetstone.optimization.miprov2_demo import ObservedTraceStep
from whetstone.optimization.miprov2_eval_config import (
    Miprov2EvalConfigBinding,
    Miprov2EvaluationExecutionPolicy,
)
from whetstone.optimization.miprov2_study import (
    EVALUATION_FAILURE_SCHEMA,
    Miprov2EvaluationObservation,
)
from whetstone.optimization.reward import REWARD_SCHEMA, RewardRef
from whetstone.optimization.schema import (
    EVALUATION_EVIDENCE_SCHEMA,
    CandidateRef,
    EvalConfigRef,
    EvaluationBinding,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
)

MIPROV2_INTENT_CONTEXT_SCHEMA = "whetstone.miprov2_intent_context"
MIPROV2_INTENT_CONTEXT_SCHEMA_VERSION = 2
MIPROV2_SELECTED_COMPONENT_STEP_SCHEMA = (
    "whetstone.miprov2_selected_component_step"
)
MIPROV2_SELECTED_COMPONENT_STEP_SCHEMA_VERSION = 1


class _ExecutedComponentStep(Protocol):
    @property
    def trace_index(self) -> int: ...

    @property
    def component_id(self) -> str: ...

    @property
    def input_field_names(self) -> tuple[str, ...]: ...

    @property
    def output_field_names(self) -> tuple[str, ...]: ...

    @property
    def inputs(self) -> ImmutableJsonObject: ...

    @property
    def outputs(self) -> ImmutableJsonObject: ...


class Miprov2RowAccounting(BaseModel):
    """Exact projection of canonical evaluation row accounting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    planned: StrictInt
    present: StrictInt
    missing: StrictInt
    failed: StrictInt
    invalid: StrictInt


class Miprov2IntentContext(BaseModel):
    """Typed context persisted before one MIPROv2 Evaluation Intent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = MIPROV2_INTENT_CONTEXT_SCHEMA_VERSION
    control_identity_hash: StrictStr
    run_id: StrictStr
    effect_kind: Literal["bootstrap", "baseline", "sample", "promotion"]
    effect_identity_hash: StrictStr
    intent_id: StrictStr
    candidate: CandidateRef
    task_batch_identities: tuple[StrictStr, ...]
    eval_config: EvalConfigRef
    eval_config_binding: Miprov2EvalConfigBinding
    evaluation_binding: EvaluationBinding
    execution_policy: Miprov2EvaluationExecutionPolicy
    reward_policy_hash: StrictStr
    bootstrap_attempt: BootstrapAttemptPlan | None = None
    optimizable_component_id: StrictStr | None = None
    optimizable_trace_index: StrictInt | None = None

    @model_validator(mode="after")
    def _validate_context(self) -> Miprov2IntentContext:
        EvaluationBinding.model_validate(
            self.evaluation_binding.model_dump(mode="json")
        )
        for field in (
            "control_identity_hash",
            "effect_identity_hash",
            "reward_policy_hash",
        ):
            require_full_hash(getattr(self, field), field=field)
        if not self.run_id or not self.intent_id:
            raise ValueError(
                "intent context run_id and intent_id are required"
            )
        if not self.task_batch_identities:
            raise ValueError("intent context task batch cannot be empty")
        for identity in self.task_batch_identities:
            require_full_hash(identity, field="task_batch_identity")
        request = self.eval_config_binding.request
        expected_purpose = {
            "bootstrap": "bootstrap",
            "baseline": "baseline",
            "sample": "sample",
            "promotion": "promotion",
        }[self.effect_kind]
        if (
            request.control_identity_hash != self.control_identity_hash
            or request.purpose != expected_purpose
            or request.effect_identity_hash != self.effect_identity_hash
            or request.task_batch_identities != self.task_batch_identities
            or request.repeat_count != 1
            or request.execution_policy != self.execution_policy
            or self.eval_config_binding.eval_config != self.eval_config
            or self.evaluation_binding.eval_config != self.eval_config
            or self.evaluation_binding.role is not EvaluationRole.INTERNAL
        ):
            raise ValueError(
                "intent context conflicts with exact Eval Config binding"
            )
        if self.effect_kind == "bootstrap":
            if (
                self.bootstrap_attempt is None
                or self.optimizable_component_id is None
                or self.optimizable_trace_index is None
            ):
                raise ValueError(
                    "bootstrap context requires its attempt and exact "
                    "optimizable graph position"
                )
            if self.optimizable_component_id not in {"generate", "encode"}:
                raise ValueError(
                    "bootstrap optimizable component must be "
                    "generate or encode"
                )
            if self.optimizable_trace_index != 0:
                raise ValueError(
                    "the supported optimizable component occupies "
                    "trace index 0"
                )
            if self.task_batch_identities != (
                self.bootstrap_attempt.task_identity,
            ):
                raise ValueError(
                    "bootstrap context must bind its exact single task"
                )
            if (
                self.effect_identity_hash
                != self.bootstrap_attempt.identity_hash()
            ):
                raise ValueError(
                    "bootstrap context effect identity does not match attempt"
                )
        elif any(
            value is not None
            for value in (
                self.bootstrap_attempt,
                self.optimizable_component_id,
                self.optimizable_trace_index,
            )
        ):
            raise ValueError(
                "non-bootstrap context cannot carry bootstrap trace selection"
            )
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_INTENT_CONTEXT_SCHEMA,
            schema_version=MIPROV2_INTENT_CONTEXT_SCHEMA_VERSION,
            payload=self.model_dump(mode="json"),
        )


class Miprov2ResolvedEvaluation(BaseModel):
    """Algorithm observation derived only from canonical persisted evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    context: Miprov2IntentContext
    reward_value: float
    normalized_score: float
    evaluation: Miprov2EvaluationObservation
    row_accounting: Miprov2RowAccounting


@dataclass(frozen=True, slots=True)
class _ResolvedEvidence:
    context: Miprov2IntentContext
    evidence_ref: TypedRef
    evidence: EvaluationEvidence
    component_traces_ref: TypedRef
    component_traces: EvaluationComponentTraces
    reward_ref: RewardRef
    row_accounting: Miprov2RowAccounting


def persist_miprov2_intent_context(
    store: ObjectStore,
    context: Miprov2IntentContext,
) -> TypedRef:
    ref, _ = store.put(
        MIPROV2_INTENT_CONTEXT_SCHEMA,
        context.model_dump(mode="json"),
    )
    typed = TypedRef(schema_name=ref.schema, content_hash=ref.content_hash)
    key = (
        "whetstone.miprov2_intent_context:"
        f"{context.run_id}:{context.intent_id}"
    )
    try:
        store.bind(key, ref)
    except BindingConflictError as error:
        existing = store.resolve(key)
        if existing != ref:
            raise ValueError(
                "MIPROv2 intent identity is bound to another exact context"
            ) from error
    return typed


def load_miprov2_intent_context(
    store: ObjectStore,
    intent: EvaluationIntent,
) -> Miprov2IntentContext:
    key = (
        f"whetstone.miprov2_intent_context:{intent.run_id}:{intent.intent_id}"
    )
    resolved = store.resolve(key)
    if resolved is None:
        raise ValueError("MIPROv2 intent has no persisted exact context")
    context = Miprov2IntentContext.model_validate(store.get(resolved))
    if (
        intent.intent_id,
        intent.run_id,
        intent.candidate,
        intent.target_eval_config,
        intent.evaluation_binding,
        intent.expected_reward_policy_hash,
    ) != (
        context.intent_id,
        context.run_id,
        context.candidate,
        context.eval_config,
        context.evaluation_binding,
        context.reward_policy_hash,
    ):
        raise ValueError("MIPROv2 intent conflicts with persisted context")
    expected_purpose = {
        "bootstrap": "miprov2_bootstrap",
        "baseline": "miprov2_baseline",
        "sample": "miprov2_sample",
        "promotion": "miprov2_promotion",
    }[context.effect_kind]
    if intent.purpose != expected_purpose:
        raise ValueError("MIPROv2 intent purpose conflicts with context")
    return context


def _row_accounting(evidence: EvaluationEvidence) -> Miprov2RowAccounting:
    accounting = Miprov2RowAccounting.model_validate(
        evidence.row_accounting.model_dump(mode="json")
    )
    counts = (
        accounting.present,
        accounting.missing,
        accounting.failed,
        accounting.invalid,
    )
    if any(count < 0 for count in counts):
        raise ValueError("evaluation row accounting cannot be negative")
    if accounting.planned != sum(counts):
        raise ValueError("evaluation row accounting is not exhaustive")
    return accounting


def _selected_step_identity(step: _ExecutedComponentStep) -> str:
    return compute_identity_hash(
        schema=MIPROV2_SELECTED_COMPONENT_STEP_SCHEMA,
        schema_version=MIPROV2_SELECTED_COMPONENT_STEP_SCHEMA_VERSION,
        payload={
            "trace_index": step.trace_index,
            "component_id": step.component_id,
            "input_field_names": list(step.input_field_names),
            "output_field_names": list(step.output_field_names),
            "inputs": step.inputs.to_json(),
            "outputs": step.outputs.to_json(),
        },
    )


@dataclass(frozen=True, slots=True)
class Miprov2EvidenceResolver:
    """Resolve only the exact records named by one Intent Resolution."""

    store: ObjectStore

    def _resolve_completed(
        self,
        resolution: IntentResolution,
    ) -> _ResolvedEvidence:
        context = load_miprov2_intent_context(self.store, resolution.intent)
        if resolution.outcome is not IntentOutcome.COMPLETED:
            raise ValueError(
                "MIPROv2 requires a completed measured resolution"
            )
        evidence_ref = resolution.evaluation_result_ref
        if (
            evidence_ref is None
            or evidence_ref.schema_name != EVALUATION_EVIDENCE_SCHEMA
        ):
            raise ValueError("MIPROv2 requires canonical evaluation evidence")
        evidence = EvaluationEvidence.model_validate(
            self.store.get(evidence_ref.reference)
        )
        EvaluationEvidenceRef(record=evidence, record_ref=evidence_ref)
        expected_binding = resolution.intent.evaluation_binding
        if (
            evidence.candidate,
            evidence.evaluation_binding,
            evidence.purpose,
            evidence.task_identities,
            evidence.repeat_count,
        ) != (
            context.candidate,
            expected_binding,
            resolution.intent.purpose,
            context.task_batch_identities,
            1,
        ):
            raise ValueError(
                "evaluation evidence conflicts with exact MIPROv2 context"
            )
        if expected_binding.role is not EvaluationRole.INTERNAL:
            raise ValueError("MIPROv2 requires an internal Evaluation Binding")
        accounting = _row_accounting(evidence)
        if accounting.planned != len(context.task_batch_identities):
            raise ValueError("evaluation row plan conflicts with task batch")

        traces_ref = evidence.component_traces_ref
        if traces_ref.schema_name != EVALUATION_COMPONENT_TRACES_SCHEMA:
            raise ValueError(
                "evaluation component traces have the wrong schema"
            )
        traces = EvaluationComponentTraces.model_validate_json(
            json.dumps(self.store.get(traces_ref.reference))
        )
        EvaluationComponentTracesRef(record=traces, record_ref=traces_ref)
        if (
            traces.candidate,
            traces.evaluation_binding,
            traces.evaluation_role,
            traces.graph_hash,
            traces.purpose,
            traces.split_role,
            traces.task_identities,
            traces.repeat_count,
        ) != (
            evidence.candidate,
            evidence.evaluation_binding,
            EvaluationRole.INTERNAL,
            evidence.graph_hash,
            evidence.purpose,
            "internal",
            evidence.task_identities,
            evidence.repeat_count,
        ):
            raise ValueError(
                "component traces conflict with exact evaluation evidence"
            )

        reward_ref = resolution.reward_ref
        if (
            reward_ref is None
            or reward_ref.record_ref.schema_name != REWARD_SCHEMA
        ):
            raise ValueError("MIPROv2 evaluation has no canonical Reward")
        if evidence.reward_ref != reward_ref:
            raise ValueError(
                "evaluation evidence and resolution Reward refs differ"
            )
        reward = reward_ref.record
        if (
            reward.reward_policy_hash != context.reward_policy_hash
            or reward.evidence_role is not EvaluationRole.INTERNAL
            or reward.evidence_refs != resolution.reward_evidence_refs
        ):
            raise ValueError(
                "Reward conflicts with MIPROv2 policy or evidence"
            )
        return _ResolvedEvidence(
            context=context,
            evidence_ref=evidence_ref,
            evidence=evidence,
            component_traces_ref=traces_ref,
            component_traces=traces,
            reward_ref=reward_ref,
            row_accounting=accounting,
        )

    def resolve_evaluation(
        self,
        resolution: IntentResolution,
    ) -> Miprov2ResolvedEvaluation:
        """Project one exact completed study evaluation."""

        resolved = self._resolve_completed(resolution)
        context = resolved.context
        if context.effect_kind == "bootstrap":
            raise ValueError("bootstrap evidence is not a study observation")
        purpose = cast(
            Literal[
                "miprov2_baseline",
                "miprov2_sample",
                "miprov2_promotion",
            ],
            resolution.intent.purpose,
        )
        normalized_score = round(resolved.reward_ref.record.value * 100, 2)
        observation = Miprov2EvaluationObservation(
            run_id=context.run_id,
            intent_id=context.intent_id,
            effect_identity_hash=context.effect_identity_hash,
            purpose=purpose,
            candidate=context.candidate,
            task_batch_identities=context.task_batch_identities,
            eval_config=context.eval_config,
            eval_config_binding=context.eval_config_binding,
            evaluation_binding=resolution.intent.evaluation_binding,
            evaluation_result_ref=resolved.evidence_ref,
            expected_reward_policy_hash=context.reward_policy_hash,
            reward_ref=resolved.reward_ref,
            normalized_score=normalized_score,
        )
        return Miprov2ResolvedEvaluation(
            context=context,
            reward_value=resolved.reward_ref.record.value,
            normalized_score=normalized_score,
            evaluation=observation,
            row_accounting=resolved.row_accounting,
        )

    def resolve_evaluation_failure(
        self,
        resolution: IntentResolution,
    ) -> Miprov2ResolvedEvaluation:
        """Map exact terminal failure evidence to a zero observation."""

        context = load_miprov2_intent_context(self.store, resolution.intent)
        if context.effect_kind == "bootstrap":
            raise ValueError("bootstrap failures use the bootstrap mapper")
        if resolution.outcome is not IntentOutcome.FAILED:
            raise ValueError("failure mapping requires a failed resolution")
        failure_ref = resolution.evaluation_result_ref
        if (
            failure_ref is None
            or failure_ref.schema_name != EVALUATION_FAILURE_SCHEMA
        ):
            raise ValueError(
                "failed evaluation requires exact failure evidence"
            )
        failure = EvaluationFailureEvidence.model_validate(
            self.store.get(failure_ref.reference)
        )
        EvaluationFailureEvidenceRef(record=failure, record_ref=failure_ref)
        if (
            failure.candidate,
            failure.evaluation_binding,
            failure.purpose,
        ) != (
            context.candidate,
            resolution.intent.evaluation_binding,
            resolution.intent.purpose,
        ):
            raise ValueError(
                "evaluation failure conflicts with intent context"
            )
        if (
            resolution.reward_ref is not None
            or resolution.reward_evidence_refs
        ):
            raise ValueError("failed evaluation cannot carry Reward evidence")
        purpose = cast(
            Literal[
                "miprov2_baseline",
                "miprov2_sample",
                "miprov2_promotion",
            ],
            resolution.intent.purpose,
        )
        observation = Miprov2EvaluationObservation(
            run_id=context.run_id,
            intent_id=context.intent_id,
            effect_identity_hash=context.effect_identity_hash,
            purpose=purpose,
            candidate=context.candidate,
            task_batch_identities=context.task_batch_identities,
            eval_config=context.eval_config,
            eval_config_binding=context.eval_config_binding,
            evaluation_binding=resolution.intent.evaluation_binding,
            evaluation_result_ref=failure_ref,
            expected_reward_policy_hash=context.reward_policy_hash,
            reward_ref=None,
            normalized_score=0.0,
        )
        rows = len(context.task_batch_identities)
        return Miprov2ResolvedEvaluation(
            context=context,
            reward_value=0.0,
            normalized_score=0.0,
            evaluation=observation,
            row_accounting=Miprov2RowAccounting(
                planned=rows,
                present=0,
                missing=0,
                failed=rows,
                invalid=0,
            ),
        )

    def resolve_bootstrap(
        self,
        resolution: IntentResolution,
    ) -> BootstrapRolloutResult:
        """Select the exact configured step from a successful trace row."""

        resolved = self._resolve_completed(resolution)
        context = resolved.context
        if context.effect_kind != "bootstrap":
            raise ValueError("resolution is not a MIPROv2 bootstrap effect")
        accounting = resolved.row_accounting
        if (
            accounting.planned != 1
            or accounting.present != 1
            or accounting.missing
            or accounting.failed
            or accounting.invalid
        ):
            raise ValueError("bootstrap requires exactly one successful row")
        if len(resolved.component_traces.rows) != 1:
            raise ValueError(
                "bootstrap requires exactly one component trace row"
            )
        row = resolved.component_traces.rows[0]
        if (
            row.task_identity != context.task_batch_identities[0]
            or row.repeat != 0
            or row.executed_component_trace.row_state.value != "success"
        ):
            raise ValueError(
                "bootstrap trace row conflicts with task, repeat, or success"
            )
        assert context.optimizable_component_id is not None
        assert context.optimizable_trace_index is not None
        matching = tuple(
            step
            for step in row.executed_component_trace.executed_component_steps
            if step.component_id == context.optimizable_component_id
        )
        if len(matching) != 1:
            raise ValueError(
                "bootstrap trace must contain the optimizable component "
                "exactly once"
            )
        selected = matching[0]
        if selected.trace_index != context.optimizable_trace_index:
            raise ValueError(
                "bootstrap optimizable component occupies another "
                "graph position"
            )
        assert context.bootstrap_attempt is not None
        return BootstrapRolloutResult(
            attempt_identity_hash=context.bootstrap_attempt.identity_hash(),
            source_rollout_identity=resolved.evidence_ref.content_hash,
            source_trace_identity=resolved.component_traces_ref.content_hash,
            source_output_identity=_selected_step_identity(selected),
            source_score_identity=resolved.reward_ref.record_ref.content_hash,
            metric_present=True,
            score=resolved.reward_ref.record.value,
            trace_steps=(
                ObservedTraceStep(
                    trace_index=selected.trace_index,
                    component_id=selected.component_id,
                    inputs=selected.inputs.to_json(),
                    outputs=selected.outputs.to_json(),
                ),
            ),
        )

    def resolve_bootstrap_failure(
        self,
        resolution: IntentResolution,
    ) -> BootstrapRolloutResult:
        """Preserve exact failure evidence without inventing score or trace."""

        context = load_miprov2_intent_context(self.store, resolution.intent)
        if (
            context.effect_kind != "bootstrap"
            or context.bootstrap_attempt is None
        ):
            raise ValueError("failure context is not a bootstrap attempt")
        if resolution.outcome is not IntentOutcome.FAILED:
            raise ValueError(
                "bootstrap failure mapping requires FAILED outcome"
            )
        failure_ref = resolution.evaluation_result_ref
        if (
            failure_ref is None
            or failure_ref.schema_name != EVALUATION_FAILURE_SCHEMA
        ):
            raise ValueError(
                "failed bootstrap requires exact failure evidence"
            )
        failure = EvaluationFailureEvidence.model_validate(
            self.store.get(failure_ref.reference)
        )
        EvaluationFailureEvidenceRef(record=failure, record_ref=failure_ref)
        if (
            failure.candidate,
            failure.evaluation_binding,
            failure.purpose,
        ) != (
            context.candidate,
            context.evaluation_binding,
            resolution.intent.purpose,
        ):
            raise ValueError("bootstrap failure conflicts with exact context")
        if (
            resolution.reward_ref is not None
            or resolution.reward_evidence_refs
        ):
            raise ValueError("failed bootstrap cannot carry Reward evidence")
        evidence_hash = failure_ref.content_hash
        return BootstrapRolloutResult(
            attempt_identity_hash=context.bootstrap_attempt.identity_hash(),
            source_rollout_identity=evidence_hash,
            source_trace_identity=evidence_hash,
            source_output_identity=evidence_hash,
            source_score_identity=evidence_hash,
            metric_present=False,
            score=None,
            error=resolution.detail.message,
        )


__all__ = [
    "MIPROV2_INTENT_CONTEXT_SCHEMA",
    "MIPROV2_INTENT_CONTEXT_SCHEMA_VERSION",
    "MIPROV2_SELECTED_COMPONENT_STEP_SCHEMA",
    "MIPROV2_SELECTED_COMPONENT_STEP_SCHEMA_VERSION",
    "Miprov2EvidenceResolver",
    "Miprov2IntentContext",
    "Miprov2ResolvedEvaluation",
    "Miprov2RowAccounting",
    "load_miprov2_intent_context",
    "persist_miprov2_intent_context",
]
