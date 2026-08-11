from __future__ import annotations

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.coordination.evaluation_restart_support import (
    assert_restart_rejects_forgery,
    evaluate_intent_bundle,
)
from tests.envs.support import process_row_job_factory
from tests.evaluation.support import (
    _binding,
    _engine,
    _intent,
    _load_component_traces,
)
from tests.optimization.support import (
    make_harness,
    make_intent,
    proposal_request,
    proposal_run,
    proposed_candidate,
    python_format_contract,
    registry,
)
from whetstone.coordination.evaluation_service import EngineEvaluationService
from whetstone.core.effects.models import ReplayPolicy
from whetstone.core.identity import TypedRef
from whetstone.evaluation.drivers.code_comp.direct import DirectRowRequest
from whetstone.evaluation.schema import (
    EvaluationEvidence,
    EvaluationOutputsRecord,
)
from whetstone.execution.fanout import ProcessJob
from whetstone.experiment.candidate import Candidate
from whetstone.optimization.adapters import AdapterOutput
from whetstone.optimization.contracts import (
    BudgetDelta,
    IntentOutcome,
    ResolutionClass,
    StepMode,
    StepStatus,
)


def test_service_rejects_provider_policy_mismatch_before_execution(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "service-policy.sqlite"))
    submitted: list[DirectRowRequest] = []

    def reject_submission(request: DirectRowRequest) -> ProcessJob:
        submitted.append(request)
        raise AssertionError("invalid binding must not create a process job")

    engine = _engine(
        tmp_path,
        store=store,
        row_job_factory=reject_submission,
    )
    intent = _intent(
        engine,
        intent_id="service-policy",
        purpose="provider-policy",
    )
    mismatched = intent.model_copy(
        update={
            "evaluation_binding": intent.evaluation_binding.model_copy(
                update={
                    "provider_execution_policy_ref": TypedRef(
                        schema_name="whetstone.provider_execution_policy",
                        content_hash="f" * 64,
                    )
                }
            )
        }
    )

    resolution = EngineEvaluationService(
        store=store,
        engine=engine,
    ).resolve_evaluation_intent(mismatched)

    assert resolution.outcome is IntentOutcome.REJECTED
    assert resolution.detail.classification is ResolutionClass.VALIDATION
    assert resolution.evaluation_result_ref is None
    assert submitted == []


def test_invalid_intent_rejects_without_provider_spend(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "reject.sqlite"))
    submitted: list[DirectRowRequest] = []

    def record_submission(request: DirectRowRequest) -> ProcessJob:
        submitted.append(request)
        raise AssertionError("invalid candidate must not create a process job")

    engine = _engine(
        tmp_path,
        store=store,
        row_job_factory=record_submission,
    )
    from whetstone.envs.code_comp.constants import MUTATION_FIELD

    invalid = Candidate(
        candidate_id="invalid",
        base_ref=engine.experiment.initial_candidate.base_ref,
        payload={MUTATION_FIELD: "Use {input_code}."},
    )
    intent = _intent(
        engine,
        intent_id="invalid-intent",
        candidate=invalid,
        purpose="preflight",
    )

    resolution = EngineEvaluationService(
        store=store, engine=engine
    ).resolve_evaluation_intent(intent)

    assert resolution.outcome is IntentOutcome.REJECTED
    assert resolution.evaluation_result_ref is None
    assert resolution.reward_evidence_refs == ()
    assert submitted == []


@pytest.mark.process_integration
def test_resolution_and_prompt_results_replay_after_restart(tmp_path) -> None:
    database = tmp_path / "restart.sqlite"
    store = ObjectStore(SqliteBackend(database))
    delegated = process_row_job_factory(
        "tests.envs.process_workers:drive_d1_success"
    )
    submitted: list[DirectRowRequest] = []

    def record_submission(request: DirectRowRequest) -> ProcessJob:
        submitted.append(request)
        return delegated(request)

    engine = _engine(
        tmp_path,
        store=store,
        row_job_factory=record_submission,
        partial=True,
        cache=True,
    )
    candidate = engine.experiment.initial_candidate
    intent = _intent(
        engine,
        intent_id="restart-intent",
        candidate=candidate,
        purpose="restart",
    )
    first = EngineEvaluationService(
        store=store, engine=engine
    ).resolve_evaluation_intent(intent)
    assert len(submitted) == 1
    assert first.reward_ref is not None
    assert first.reward_evidence_refs == first.reward_ref.record.evidence_refs
    assert first.evaluation_result_ref is not None
    first_evidence = EvaluationEvidence.model_validate(
        store.get(first.evaluation_result_ref.reference)
    )
    first_outputs = EvaluationOutputsRecord.model_validate(
        store.get(first_evidence.outputs_ref.reference)
    )
    assert first_outputs.component_traces_ref == (
        first_evidence.component_traces_ref
    )
    assert _load_component_traces(store, first_evidence).rows

    fresh_store = ObjectStore(SqliteBackend(database))

    def reject_submission(_request: DirectRowRequest) -> ProcessJob:
        raise AssertionError("durable resolution must replay")

    fresh_engine = _engine(
        tmp_path,
        store=fresh_store,
        row_job_factory=reject_submission,
        partial=True,
        cache=True,
    )
    replay = EngineEvaluationService(
        store=fresh_store, engine=fresh_engine
    ).resolve_evaluation_intent(intent)

    assert replay == first
    assert len(submitted) == 1


@pytest.mark.parametrize("forgery", ("candidate",))
@pytest.mark.process_integration
def test_restart_rejects_forged_or_incomplete_result_graphs(
    tmp_path,
    forgery: str,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / f"forged-{forgery}.sqlite"))
    engine = _engine(tmp_path, store=store)
    intent = _intent(
        engine,
        intent_id=f"forged-{forgery}",
        purpose="graph-validation",
    )
    bundle = evaluate_intent_bundle(engine, intent)
    assert_restart_rejects_forgery(
        store=store,
        engine=engine,
        bundle=bundle,
        forgery=forgery,
    )


@pytest.mark.process_integration
def test_concrete_evaluation_service_reaches_harness_boundary(
    tmp_path,
) -> None:
    class EngineProposalAdapter:
        @property
        def key(self) -> str:
            return "proposal-test"

        @property
        def mode(self) -> StepMode:
            return StepMode.PROPOSAL_ONLY

        @property
        def required_replay_policy(self) -> ReplayPolicy:
            return ReplayPolicy.IDEMPOTENT

        def invoke(self, request, handles) -> AdapterOutput:
            assert handles == ()
            base = request.candidates[0]
            from whetstone.envs.code_comp.constants import MUTATION_FIELD

            template = str(base.payload[MUTATION_FIELD])
            proposed = proposed_candidate(
                base,
                "harness-evaluation",
                text=f"{template}\n\nBe precise.",
            )
            proposed = Candidate(
                candidate_id=proposed.candidate_id,
                base_ref=proposed.base_ref,
                payload={MUTATION_FIELD: proposed.payload[MUTATION_FIELD]},
            )
            intent = make_intent(
                proposed,
                run_id=request.run_id,
                step_index=request.step_index,
                binding=_binding(engine, campaign=request.run_id),
                reward_policy=engine.experiment.reward_policy,
            )
            return AdapterOutput(
                proposed_candidates=(proposed,),
                accepted_candidates=(proposed,),
                evaluation_intents=(intent,),
                budget_delta=BudgetDelta(consumed={"generations": 1}),
                proposed_status=StepStatus.COMPLETE,
            )

    store = ObjectStore(SqliteBackend(tmp_path / "harness-service.sqlite"))
    engine = _engine(tmp_path, store=store)
    service = EngineEvaluationService(store=store, engine=engine)
    render_contract = python_format_contract(
        available_fields=("input_code",),
        required_fields=(),
    )
    run = proposal_run(
        reward_policy=engine.experiment.reward_policy,
        template_render_contract=render_contract,
    )
    request = proposal_request(
        run=run,
        candidates=(engine.experiment.initial_candidate,),
    )
    adapter = EngineProposalAdapter()
    assert adapter.required_replay_policy is ReplayPolicy.IDEMPOTENT
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        evaluation_service=service,
        adapter_replay_policy=ReplayPolicy.IDEMPOTENT,
    )

    result, _result_ref = harness.run_step(request)

    assert service.replay_policy is ReplayPolicy.DURABLE_WORKFLOW
    assert len(result.resolved_intents) == 1
    assert result.resolved_intents[0].outcome is IntentOutcome.COMPLETED
    service.validate_resolution_graph(result.resolved_intents[0])
