"""Exact ObjectStore evidence bridge for durable MIPROv2 effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from dr_store import ObjectStore
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.identity import (
    TypedRef,
    compute_identity_hash,
    reject_non_json,
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
    REWARD_SCHEMA,
    EvaluationBinding,
    VerifiedEvaluationCitation,
)
from whetstone.optimization.reward import Reward, RewardInputCitation
from whetstone.optimization.schema import (
    CandidateRef,
    EvalConfigRef,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
)

MIPROV2_INTENT_CONTEXT_SCHEMA = "whetstone.miprov2_intent_context"
MIPROV2_INTENT_CONTEXT_SCHEMA_VERSION = 1
EVALUATION_EVIDENCE_SCHEMA = "whetstone.evaluation_evidence"
ROLLOUT_AGGREGATE_SCHEMA = "whetstone.rollout_aggregate"


class Miprov2RowAccounting(BaseModel):
    """Dependency-safe exact projection of evaluation row accounting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    planned: StrictInt
    present: StrictInt
    missing: StrictInt
    failed: StrictInt
    invalid: StrictInt


class _RolloutAggregateProjection(BaseModel):
    """Exact fields required to bind evidence to its aggregate artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: StrictStr
    graph_hash: StrictStr
    eval_config_hash: StrictStr
    evaluation_context_id: StrictStr
    task_count: StrictInt
    repeat_count: StrictInt
    aggregation_output: dict[str, object]
    rows_present: StrictInt
    rows_missing: StrictInt
    rows_failed: StrictInt
    rows_invalid: StrictInt


class Miprov2BootstrapTraceProjection(BaseModel):
    """Expected native trace fields for one ordered teacher component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: StrictStr
    inputs: dict[str, object]
    output_field: StrictStr

    @model_validator(mode="after")
    def _validate_projection(self) -> Miprov2BootstrapTraceProjection:
        if not self.component_id or not self.output_field:
            raise ValueError("bootstrap trace component fields are required")
        reject_non_json(self.inputs, field="bootstrap component inputs")
        return self


class Miprov2IntentContext(BaseModel):
    """Typed context persisted before one MIPROv2 evaluation Intent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    control_identity_hash: StrictStr
    run_id: StrictStr
    effect_kind: Literal["bootstrap", "baseline", "sample", "promotion"]
    effect_identity_hash: StrictStr
    intent_id: StrictStr
    candidate: CandidateRef
    task_batch_identities: tuple[StrictStr, ...]
    eval_config: EvalConfigRef
    eval_config_binding: Miprov2EvalConfigBinding
    execution_policy: Miprov2EvaluationExecutionPolicy
    reward_policy_hash: StrictStr
    bootstrap_attempt: BootstrapAttemptPlan | None = None
    trace_components: tuple[Miprov2BootstrapTraceProjection, ...] = ()

    @model_validator(mode="after")
    def _validate_context(self) -> Miprov2IntentContext:
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
        ):
            raise ValueError(
                "intent context conflicts with exact Eval Config binding"
            )
        if self.effect_kind == "bootstrap":
            if self.bootstrap_attempt is None or not self.trace_components:
                raise ValueError(
                    "bootstrap context requires attempt and component traces"
                )
            assert self.bootstrap_attempt is not None
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
            component_ids = tuple(
                item.component_id for item in self.trace_components
            )
            if len(component_ids) != len(set(component_ids)):
                raise ValueError(
                    "bootstrap trace component ids must be unique"
                )
        elif self.bootstrap_attempt is not None or self.trace_components:
            raise ValueError(
                "non-bootstrap context cannot carry bootstrap projection"
            )
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_INTENT_CONTEXT_SCHEMA,
            schema_version=MIPROV2_INTENT_CONTEXT_SCHEMA_VERSION,
            payload=self.model_dump(mode="json"),
        )


class Miprov2ResolvedEvaluation(BaseModel):
    """Score and provenance derived only from canonical persisted evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    context: Miprov2IntentContext
    reward_value: float
    normalized_score: float
    evaluation: EvaluationBinding
    row_accounting: Miprov2RowAccounting


@dataclass(frozen=True, slots=True)
class _ResolvedEvidence:
    """Internal common evidence projection shared by eval and bootstrap."""

    context: Miprov2IntentContext
    evidence_ref: TypedRef
    row_accounting: Miprov2RowAccounting
    repeat_count: int
    outputs_ref: TypedRef
    aggregate_ref: TypedRef
    reward_ref: TypedRef
    reward: Reward


def persist_miprov2_intent_context(
    store: ObjectStore,
    context: Miprov2IntentContext,
) -> TypedRef:
    ref, _ = store.put(
        MIPROV2_INTENT_CONTEXT_SCHEMA,
        context.model_dump(mode="json"),
    )
    typed = TypedRef(schema_name=ref.schema, content_hash=ref.content_hash)
    return typed


def load_miprov2_intent_context(
    store: ObjectStore,
    intent: EvaluationIntent,
) -> Miprov2IntentContext:
    if intent.context_policy_ref is None:
        raise ValueError("MIPROv2 intent has no persisted effect context")
    ref = TypedRef(
        schema_name=MIPROV2_INTENT_CONTEXT_SCHEMA,
        content_hash=intent.context_policy_ref,
    )
    context = Miprov2IntentContext.model_validate(store.get(ref.reference))
    expected = (
        context.intent_id,
        context.run_id,
        context.candidate,
        context.eval_config,
    )
    actual = (
        intent.intent_id,
        intent.run_id,
        intent.candidate,
        intent.target_eval_config,
    )
    if actual != expected:
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


def _resolve_miprov2_evidence(
    store: ObjectStore,
    resolution: IntentResolution,
) -> _ResolvedEvidence:
    """Validate one resolution without assigning optimizer semantics."""

    # The evaluation package eagerly imports its engine, which depends back on
    # optimization through environment construction. Import its contract only
    # once the module graph is initialized and evidence is actually resolved.
    from whetstone.evaluation.schema import (
        EVALUATION_OUTPUTS_SCHEMA,
        EvaluationEvidence,
    )

    context = load_miprov2_intent_context(store, resolution.intent)
    if resolution.outcome is not IntentOutcome.COMPLETED:
        raise ValueError("MIPROv2 requires a completed measured resolution")
    if len(resolution.evaluation_evidence_refs) != 1:
        raise ValueError(
            "MIPROv2 requires exactly one evaluation evidence ref"
        )
    evidence_ref = resolution.evaluation_evidence_refs[0]
    if evidence_ref.schema_name != EVALUATION_EVIDENCE_SCHEMA:
        raise ValueError("MIPROv2 requires canonical evaluation evidence")
    evidence = EvaluationEvidence.model_validate(
        store.get(evidence_ref.reference)
    )
    expected = (
        context.candidate,
        context.eval_config,
        EvaluationRole.INTERNAL,
        context.intent_id,
        resolution.intent.purpose,
        context.task_batch_identities,
    )
    actual = (
        evidence.candidate,
        evidence.eval_config,
        evidence.evaluation_role,
        evidence.evaluation_context_id,
        evidence.purpose,
        evidence.task_identities,
    )
    if actual != expected:
        raise ValueError(
            "evaluation evidence conflicts with MIPROv2 intent context"
        )
    if evidence.row_accounting.planned != (
        len(context.task_batch_identities) * evidence.repeat_count
    ):
        raise ValueError("evaluation row accounting conflicts with task batch")
    counts = (
        evidence.row_accounting.present,
        evidence.row_accounting.missing,
        evidence.row_accounting.failed,
        evidence.row_accounting.invalid,
    )
    if any(count < 0 for count in counts):
        raise ValueError("evaluation row accounting cannot be negative")
    if sum(counts) != evidence.row_accounting.planned:
        raise ValueError("evaluation row accounting is not exhaustive")
    if (
        evidence.repeat_count <= 0
        or len(evidence.per_task_values) != len(context.task_batch_identities)
        or len(evidence.per_task_counts) != len(context.task_batch_identities)
    ):
        raise ValueError("evaluation per-task evidence has the wrong shape")
    if evidence.outputs_ref.schema_name != EVALUATION_OUTPUTS_SCHEMA:
        raise ValueError("evaluation outputs ref has the wrong schema")
    if evidence.aggregate_ref.schema_name != ROLLOUT_AGGREGATE_SCHEMA:
        raise ValueError("evaluation aggregate ref has the wrong schema")
    aggregate = _RolloutAggregateProjection.model_validate(
        store.get(evidence.aggregate_ref.reference)
    )
    aggregate_output = aggregate.aggregation_output
    if (
        aggregate.name,
        aggregate.graph_hash,
        aggregate.eval_config_hash,
        aggregate.evaluation_context_id,
        aggregate.task_count,
        aggregate.repeat_count,
        aggregate_output.get("value"),
        aggregate_output.get("status"),
        aggregate.rows_present,
        aggregate.rows_missing,
        aggregate.rows_failed,
        aggregate.rows_invalid,
    ) != (
        evidence.aggregate_name,
        evidence.graph_hash,
        context.eval_config.identity_hash,
        context.intent_id,
        len(context.task_batch_identities),
        evidence.repeat_count,
        evidence.aggregate_value,
        evidence.aggregate_status,
        *counts,
    ):
        raise ValueError("evaluation aggregate conflicts with evidence")
    if resolution.reward_ref is None or evidence.reward_ref is None:
        raise ValueError("MIPROv2 evaluation has no canonical Reward")
    if resolution.reward_ref != evidence.reward_ref:
        raise ValueError("resolution and evidence Reward refs differ")
    if resolution.reward_ref.schema_name != REWARD_SCHEMA:
        raise ValueError("MIPROv2 Reward ref has the wrong schema")
    reward = Reward.model_validate(store.get(resolution.reward_ref.reference))
    if (
        reward.reward_policy_hash != context.reward_policy_hash
        or reward.evidence_role is not EvaluationRole.INTERNAL
    ):
        raise ValueError("Reward conflicts with MIPROv2 policy or role")
    if reward.evidence_ref_content_hash != evidence.aggregate_ref.content_hash:
        raise ValueError("Reward cites another evaluation aggregate")
    return _ResolvedEvidence(
        context=context,
        evidence_ref=evidence_ref,
        row_accounting=Miprov2RowAccounting.model_validate(
            evidence.row_accounting.model_dump(mode="json")
        ),
        repeat_count=evidence.repeat_count,
        outputs_ref=evidence.outputs_ref,
        aggregate_ref=evidence.aggregate_ref,
        reward_ref=resolution.reward_ref,
        reward=reward,
    )


def resolve_miprov2_evaluation(
    store: ObjectStore,
    resolution: IntentResolution,
) -> Miprov2ResolvedEvaluation:
    """Validate and project one exact completed evaluation resolution."""

    resolved = _resolve_miprov2_evidence(store, resolution)
    context = resolved.context
    if context.effect_kind == "bootstrap":
        raise ValueError(
            "bootstrap evidence cannot be folded as a study evaluation"
        )
    purpose = cast(
        Literal[
            "miprov2_baseline",
            "miprov2_sample",
            "miprov2_promotion",
        ],
        resolution.intent.purpose,
    )
    normalized_score = round(resolved.reward.value * 100, 2)
    citation = VerifiedEvaluationCitation(
        run_id=context.run_id,
        intent_id=context.intent_id,
        effect_identity_hash=context.effect_identity_hash,
        purpose=purpose,
        candidate_identity_hash=context.candidate.identity_hash,
        task_batch_identities=context.task_batch_identities,
        validation_eval_source_identity_hash=(
            context.eval_config_binding.request.source_eval_config.identity_hash
        ),
        eval_config_identity_hash=context.eval_config.identity_hash,
        eval_config_binding_identity_hash=(
            context.eval_config_binding.identity_hash()
        ),
        reward_policy_hash=context.reward_policy_hash,
        evidence_ref=resolved.evidence_ref,
        reward_ref=resolved.reward_ref,
        normalized_score=normalized_score,
    )
    binding = EvaluationBinding(
        run_id=context.run_id,
        intent_id=context.intent_id,
        effect_identity_hash=context.effect_identity_hash,
        purpose=purpose,
        candidate_identity_hash=context.candidate.identity_hash,
        task_batch_identities=context.task_batch_identities,
        eval_config=context.eval_config,
        eval_config_binding=context.eval_config_binding,
        reward_policy_hash=context.reward_policy_hash,
        reward_ref=resolved.reward_ref,
        evidence_citations=(citation,),
        normalized_score=normalized_score,
    )
    return Miprov2ResolvedEvaluation(
        context=context,
        reward_value=resolved.reward.value,
        normalized_score=normalized_score,
        evaluation=binding,
        row_accounting=resolved.row_accounting,
    )


def resolve_miprov2_evaluation_failure(
    store: ObjectStore,
    resolution: IntentResolution,
) -> Miprov2ResolvedEvaluation:
    """Map a terminal evaluation exception to frozen DSPy's score zero."""

    context = load_miprov2_intent_context(store, resolution.intent)
    if context.effect_kind == "bootstrap":
        raise ValueError("bootstrap failures use the bootstrap mapper")
    if resolution.outcome is IntentOutcome.COMPLETED:
        raise ValueError("completed evaluation must use measured evidence")
    if len(resolution.evaluation_evidence_refs) != 1:
        raise ValueError("failed evaluation requires exactly one evidence ref")
    evidence_ref = resolution.evaluation_evidence_refs[0]
    if evidence_ref.schema_name != EVALUATION_FAILURE_SCHEMA:
        raise ValueError(
            "failed evaluation requires canonical failure evidence"
        )
    failure = store.get(evidence_ref.reference)
    if not isinstance(failure, dict):
        raise ValueError("evaluation failure evidence must be an object")
    expected = (
        context.candidate,
        context.eval_config,
        resolution.intent.purpose,
    )
    actual = (
        CandidateRef.model_validate(failure.get("candidate")),
        EvalConfigRef.model_validate(failure.get("eval_config")),
        failure.get("purpose"),
    )
    if actual != expected:
        raise ValueError("evaluation failure conflicts with intent context")
    reward = Reward(
        reward_name="miprov2_evaluation_failure",
        value=0.0,
        reward_policy_hash=context.reward_policy_hash,
        evidence_role=EvaluationRole.INTERNAL,
        input_citations=(
            RewardInputCitation(
                name="evaluation_failure",
                value=0.0,
                contributed=0.0,
            ),
        ),
        evidence_ref_content_hash=evidence_ref.content_hash,
    )
    reward_ref_raw, _ = store.put(REWARD_SCHEMA, reward.record_content())
    reward_ref = TypedRef(
        schema_name=reward_ref_raw.schema,
        content_hash=reward_ref_raw.content_hash,
    )
    purpose = cast(
        Literal[
            "miprov2_baseline",
            "miprov2_sample",
            "miprov2_promotion",
        ],
        resolution.intent.purpose,
    )
    binding_identity = context.eval_config_binding.identity_hash()
    citation = VerifiedEvaluationCitation(
        run_id=context.run_id,
        intent_id=context.intent_id,
        effect_identity_hash=context.effect_identity_hash,
        purpose=purpose,
        candidate_identity_hash=context.candidate.identity_hash,
        task_batch_identities=context.task_batch_identities,
        validation_eval_source_identity_hash=(
            context.eval_config_binding.request.source_eval_config.identity_hash
        ),
        eval_config_identity_hash=context.eval_config.identity_hash,
        eval_config_binding_identity_hash=binding_identity,
        reward_policy_hash=context.reward_policy_hash,
        evidence_ref=evidence_ref,
        reward_ref=reward_ref,
        normalized_score=0.0,
    )
    evaluation = EvaluationBinding(
        run_id=context.run_id,
        intent_id=context.intent_id,
        effect_identity_hash=context.effect_identity_hash,
        purpose=purpose,
        candidate_identity_hash=context.candidate.identity_hash,
        task_batch_identities=context.task_batch_identities,
        eval_config=context.eval_config,
        eval_config_binding=context.eval_config_binding,
        reward_policy_hash=context.reward_policy_hash,
        reward_ref=reward_ref,
        evidence_citations=(citation,),
        normalized_score=0.0,
    )
    rows = len(context.task_batch_identities)
    return Miprov2ResolvedEvaluation(
        context=context,
        reward_value=0.0,
        normalized_score=0.0,
        evaluation=evaluation,
        row_accounting=Miprov2RowAccounting(
            planned=rows,
            present=0,
            missing=0,
            failed=rows,
            invalid=0,
        ),
    )


def resolve_miprov2_bootstrap(
    store: ObjectStore,
    resolution: IntentResolution,
) -> BootstrapRolloutResult:
    """Map exact single-task evidence/output rows to a bootstrap result."""

    # See the package-cycle constraint in _resolve_miprov2_evidence.
    from whetstone.evaluation.schema import EvaluationOutputsRecord

    resolved = _resolve_miprov2_evidence(store, resolution)
    context = resolved.context
    if context.effect_kind != "bootstrap":
        raise ValueError("resolution is not a MIPROv2 bootstrap effect")
    if (
        resolved.row_accounting.planned != 1
        or resolved.row_accounting.present != 1
        or resolved.row_accounting.missing
        or resolved.row_accounting.failed
        or resolved.row_accounting.invalid
    ):
        raise ValueError(
            "bootstrap requires exactly one successful task output row"
        )
    if resolved.repeat_count != 1:
        raise ValueError("bootstrap requires repeat_count=1")
    output_record = EvaluationOutputsRecord.model_validate(
        store.get(resolved.outputs_ref.reference)
    )
    if len(output_record.outputs) != 1:
        raise ValueError("bootstrap requires exactly one output row")
    row = output_record.outputs[0]
    output_text = row.output_text
    if output_text is None or row.failure_code != "":
        raise ValueError("bootstrap output row is not a successful generation")
    expected_candidate_id = context.candidate.record.candidate_id
    if (
        output_record.candidate_id != expected_candidate_id
        or row.candidate_id != expected_candidate_id
        or row.task_identity != context.task_batch_identities[0]
        or row.repeat != 0
    ):
        raise ValueError(
            "bootstrap output row conflicts with candidate, task, or repeat"
        )
    assert context.bootstrap_attempt is not None
    trace_steps: tuple[ObservedTraceStep, ...]
    component_traces = row.component_trace_steps
    if len(context.trace_components) == 1 and not component_traces:
        projection = context.trace_components[0]
        trace_steps = (
            ObservedTraceStep(
                trace_index=0,
                component_id=projection.component_id,
                inputs=projection.inputs,
                outputs={projection.output_field: output_text},
            ),
        )
    else:
        if len(component_traces) != len(context.trace_components):
            raise ValueError(
                "bootstrap output must carry every ordered component trace"
            )
        observed: list[ObservedTraceStep] = []
        for trace_index, (projection, trace) in enumerate(
            zip(
                context.trace_components,
                component_traces,
                strict=True,
            )
        ):
            inputs = trace.inputs
            outputs = trace.outputs
            if (
                trace.component_id != projection.component_id
                or inputs != projection.inputs
                or tuple(outputs) != (projection.output_field,)
                or type(outputs[projection.output_field]) is not str
            ):
                raise ValueError(
                    "bootstrap component trace conflicts with context"
                )
            observed.append(
                ObservedTraceStep(
                    trace_index=trace_index,
                    component_id=projection.component_id,
                    inputs=inputs,
                    outputs=outputs,
                )
            )
        trace_steps = tuple(observed)
    row_identity = compute_identity_hash(
        schema="whetstone.miprov2_bootstrap_output_row",
        schema_version=1,
        payload=row.model_dump(mode="json"),
    )
    return BootstrapRolloutResult(
        attempt_identity_hash=context.bootstrap_attempt.identity_hash(),
        source_rollout_identity=resolved.aggregate_ref.content_hash,
        source_trace_identity=resolved.outputs_ref.content_hash,
        source_output_identity=row_identity,
        source_score_identity=resolved.reward_ref.content_hash,
        metric_present=True,
        score=resolved.reward.value,
        trace_steps=trace_steps,
    )


def failure_bootstrap_result(
    *,
    context: Miprov2IntentContext,
    resolution: IntentResolution,
) -> BootstrapRolloutResult:
    """Preserve a terminal failed rollout without inventing score or trace."""

    if context.effect_kind != "bootstrap" or context.bootstrap_attempt is None:
        raise ValueError("failure context is not a bootstrap attempt")
    if resolution.intent.intent_id != context.intent_id:
        raise ValueError("failure resolution belongs to another intent")
    if resolution.outcome is IntentOutcome.COMPLETED:
        raise ValueError("completed bootstrap must use the evidence mapper")
    evidence_hash = compute_identity_hash(
        schema=EVALUATION_FAILURE_SCHEMA,
        schema_version=1,
        payload=resolution.model_dump(mode="json"),
    )
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
    "Miprov2BootstrapTraceProjection",
    "Miprov2IntentContext",
    "Miprov2ResolvedEvaluation",
    "Miprov2RowAccounting",
    "failure_bootstrap_result",
    "load_miprov2_intent_context",
    "persist_miprov2_intent_context",
    "resolve_miprov2_bootstrap",
    "resolve_miprov2_evaluation",
    "resolve_miprov2_evaluation_failure",
]
