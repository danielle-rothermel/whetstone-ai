from __future__ import annotations

import json
from functools import cache

from dr_serialize import Jsonable
from dr_store import ObjectStore

from tests.envs.support import (
    execution_policy,
    in_process_internal_row_job_factory,
)
from whetstone.coordination.evaluation_claims import (
    EVALUATION_INTENT_CLAIM_SCHEMA,
    EVALUATION_RESULT_ATTESTATION_SCHEMA,
    EvaluationIntentClaim,
    EvaluationResultAttestation,
)
from whetstone.coordination.evaluation_service import EngineEvaluationService
from whetstone.core.identity import TypedRef
from whetstone.core.roles import EvaluationRole
from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.envs.generation_graph import LLM_NODE_ID, render_prompt
from whetstone.envs.registry import env_spec
from whetstone.evaluation.drivers.internal import (
    InternalRowJobFactory,
    InternalRowOutcome,
    InternalRowRequest,
    _llm_component_step,
)
from whetstone.evaluation.engine import (
    EngineEvaluation,
    EvaluationEngine,
)
from whetstone.evaluation.schema import (
    EvaluationComponentTraces,
    EvaluationEvidence,
)
from whetstone.evaluation.traces import ExecutedRowState
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.binding import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvaluationBinding,
)
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optimization.contracts import (
    INTENT_RESOLUTION_SCHEMA,
    INTENT_RESOLUTION_SCHEMA_VERSION,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    ResolutionClass,
    ResolutionDetail,
)
from whetstone.provider.policy import ProviderExecutionPolicy

_DEFAULT_ROW_JOB_FACTORY = in_process_internal_row_job_factory()


def _uncached_experiment(*, num_samples: int = 1) -> EnvExperiment:
    return build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        num_samples=num_samples,
    )


@cache
def _cached_experiment(num_samples: int) -> EnvExperiment:
    return _uncached_experiment(num_samples=num_samples)


def _experiment(*, num_samples: int = 1) -> EnvExperiment:
    return _cached_experiment(num_samples)


def _engine(
    tmp_path,
    *,
    store: ObjectStore,
    row_job_factory: InternalRowJobFactory = _DEFAULT_ROW_JOB_FACTORY,
    num_samples: int = 1,
    partial: bool = False,
    cache: bool = False,
    role: EvaluationRole = EvaluationRole.INTERNAL,
    provider_policy: ProviderExecutionPolicy | None = None,
    max_wall_seconds: float | None = None,
) -> EvaluationEngine:
    experiment = _experiment(num_samples=num_samples)
    sampling = (
        experiment.eval_configs.internal
        if role is EvaluationRole.INTERNAL
        else experiment.eval_configs.official
    )
    return EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=sampling,
        execution_policy=provider_policy or execution_policy(),
        row_job_factory=row_job_factory,
        max_wall_seconds=max_wall_seconds,
        partial_log=PartialLog(tmp_path / "partials.jsonl")
        if partial
        else None,
        prompt_cache=PromptResultCache(tmp_path / "cache") if cache else None,
    )


def _binding(
    engine: EvaluationEngine,
    *,
    role: EvaluationRole = EvaluationRole.INTERNAL,
    campaign: str = "evaluation-test",
) -> EvaluationBinding:
    return EvaluationBinding(
        schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
        eval_config=engine.eval_config_ref,
        role=role,
        authority_principal=(
            "test-authority" if role is EvaluationRole.OFFICIAL else None
        ),
        campaign=campaign,
        provider_execution_policy_ref=engine.provider_execution_policy_ref,
    )


def _intent(
    engine: EvaluationEngine,
    *,
    intent_id: str,
    purpose: str,
    candidate: Candidate | None = None,
    role: EvaluationRole = EvaluationRole.INTERNAL,
) -> EvaluationIntent:
    return EvaluationIntent(
        intent_id=intent_id,
        candidate=candidate_reference(
            candidate or engine.experiment.initial_candidate
        ),
        target_eval_config=engine.eval_config_ref,
        evaluation_binding=_binding(engine, role=role, campaign=intent_id),
        purpose=purpose,
        run_id="run",
        step_index=0,
        expected_reward_policy_hash=(
            engine.experiment.reward_policy.identity_hash()
            if role is EvaluationRole.INTERNAL
            else None
        ),
    )


def _completed_resolution(
    intent: EvaluationIntent,
    evaluated: EngineEvaluation,
) -> IntentResolution:
    reward_ref = evaluated.evidence.reward_ref
    return IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        intent=intent,
        outcome=IntentOutcome.COMPLETED,
        detail=ResolutionDetail(
            classification=ResolutionClass.MEASURED,
            message="candidate evaluated under exact sampling binding",
        ),
        evaluation_result_ref=evaluated.evidence_ref,
        reward_evidence_refs=(
            () if reward_ref is None else reward_ref.record.evidence_refs
        ),
        resolved_eval_config=intent.target_eval_config,
        reward_ref=reward_ref,
    )


def _bind_without_validation(
    *,
    store: ObjectStore,
    service: EngineEvaluationService,
    intent: EvaluationIntent,
    resolution: IntentResolution,
) -> None:
    reference, _ = store.put(
        INTENT_RESOLUTION_SCHEMA,
        resolution.model_dump(mode="json"),
    )
    store.bind(service._key(intent), reference)


def _bind_with_forged_terminal_attestation(
    *,
    store: ObjectStore,
    service: EngineEvaluationService,
    intent: EvaluationIntent,
    resolution: IntentResolution,
) -> None:
    attestation = EvaluationResultAttestation(
        graph_hash=service._engine.experiment.generation_graph.graph_hash,
        resolution=resolution,
    )
    attestation_ref = _put_typed(
        store,
        EVALUATION_RESULT_ATTESTATION_SCHEMA,
        attestation.record_content(),
    )
    claim = EvaluationIntentClaim(
        intent_ref=service._intent_ref(intent),
        owner_id="forged-restart-fixture",
        event_ordinal=0,
        generation=0,
        heartbeat_ordinal=0,
        expires_at=0.0,
        result_attestation_ref=attestation_ref,
    )
    claim_reference, _ = store.put(
        EVALUATION_INTENT_CLAIM_SCHEMA,
        claim.model_dump(mode="json"),
    )
    store.bind(service._claim_key(intent, 0), claim_reference)
    _bind_without_validation(
        store=store,
        service=service,
        intent=intent,
        resolution=resolution,
    )


def _publish_attestation(
    *,
    service: EngineEvaluationService,
    intent: EvaluationIntent,
    resolution: IntentResolution,
) -> None:
    service._persist_intent_targets(intent)
    owned = service._claim(intent)
    assert owned is not None
    service._publish_result_attestation(
        intent=intent,
        resolution=resolution,
        owned=owned,
    )


def _put_typed(
    store: ObjectStore,
    schema: str,
    content: Jsonable,
) -> TypedRef:
    reference, _ = store.put(schema, content)
    return TypedRef(
        schema_name=reference.schema,
        content_hash=reference.content_hash,
    )


def _load_component_traces(
    store: ObjectStore,
    evidence: EvaluationEvidence,
) -> EvaluationComponentTraces:
    return EvaluationComponentTraces.model_validate_json(
        json.dumps(store.get(evidence.component_traces_ref.reference))
    )


def _successful_internal_outcome(
    request: InternalRowRequest,
) -> InternalRowOutcome:
    output_text = request.instance.gold
    prompt = render_prompt(
        env_spec(request.env_name),
        request.candidate,
        request.instance.to_instance(),
    )
    return InternalRowOutcome(
        score=1.0,
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=(
            _llm_component_step(
                trace_index=0,
                component_id=LLM_NODE_ID,
                prompt=prompt,
                generation=output_text,
            ),
        ),
        output_text=output_text,
        finish_reason="stop",
    )
