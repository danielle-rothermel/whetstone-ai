"""Successful real MIPROv2 adapter flow through the durable harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from dr_store import ObjectStore, SqliteBackend

from tests.optimization.miprov2.support import (
    make_minimal_miprov2_runtime,
    persist_test_record,
    resolve_miprov2_eval_config_binding,
)
from tests.optimization.support import make_harness, registry
from whetstone.core.effects.authority import EffectAuthority
from whetstone.core.effects.models import ReplayPolicy
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.schema import (
    EVALUATION_COMPONENT_TRACES_SCHEMA,
    EVALUATION_OUTPUTS_SCHEMA,
    EvaluationComponentTraceRow,
    EvaluationComponentTraces,
    EvaluationEvidence,
    RowAccounting,
)
from whetstone.evaluation.schema_names import EVALUATION_EVIDENCE_SCHEMA
from whetstone.evaluation.traces import (
    ExecutedComponentTracePayload,
    ExecutedRowState,
)
from whetstone.experiment.candidate import candidate_reference
from whetstone.experiment.reward import (
    RewardPolicy,
    apply_reward_policy,
    reward_reference,
)
from whetstone.optimization.contracts import (
    INTENT_RESOLUTION_SCHEMA_VERSION,
    BudgetState,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    OptimizationRun,
    OutputContract,
    ResolutionClass,
    ResolutionDetail,
    StepMode,
    StepStatus,
    optimization_run_reference,
    step_request_reference,
    step_result_reference,
)
from whetstone.optimization.miprov2.adapter import (
    MIPROV2_STATE_KEY,
    Miprov2Adapter,
)
from whetstone.optimization.miprov2.eval_config import (
    Miprov2EvalConfigBinding,
    Miprov2EvalConfigBindingRequest,
    Miprov2EvalConfigResolver,
)
from whetstone.optimization.miprov2.evidence import Miprov2IntentContext
from whetstone.optimization.miprov2.runtime import Miprov2State
from whetstone.optimization.proposal.proposer import (
    DurableProposalExecutor,
    FakeProposerTransport,
    ProposalExecutorDurabilityContract,
    _durable_proposal_executor,
)

GRAPH_HASH = "a" * 64


class _CanonicalEvalConfigResolver:
    def resolve(
        self,
        request: Miprov2EvalConfigBindingRequest,
    ) -> Miprov2EvalConfigBinding:
        return resolve_miprov2_eval_config_binding(request)


def _proposal_executor() -> DurableProposalExecutor:
    def execute(*, config, request, transport, count):
        return transport.draft(config, request, count)

    return _durable_proposal_executor(
        durability_contract=ProposalExecutorDurabilityContract(
            recovery_policy=ReplayPolicy.DURABLE_WORKFLOW,
            policy_identity_hash="c" * 64,
        ),
        execute=execute,
    )


@dataclass(slots=True)
class _PersistingEvaluationService:
    store: ObjectStore
    reward_policy: RewardPolicy
    calls: list[EvaluationIntent] = field(default_factory=list)
    validation_calls: list[IntentResolution] = field(default_factory=list)

    @property
    def replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.DURABLE_WORKFLOW

    def resolve_evaluation_intent(
        self,
        intent: EvaluationIntent,
    ) -> IntentResolution:
        self.calls.append(intent)
        score = {
            "miprov2_baseline": 0.1,
            "miprov2_sample": 0.9,
        }[intent.purpose]
        task_identities = _load_context_tasks(self.store, intent)
        traces = EvaluationComponentTraces(
            schema_version=1,
            candidate=intent.candidate,
            evaluation_binding=intent.evaluation_binding,
            evaluation_role=EvaluationRole.INTERNAL,
            graph_hash=GRAPH_HASH,
            purpose=intent.purpose,
            split_role="internal",
            task_identities=task_identities,
            repeat_count=1,
            rows=tuple(
                EvaluationComponentTraceRow(
                    instance_id=f"instance-{index}",
                    task_identity=task_identity,
                    repeat=0,
                    executed_component_trace=ExecutedComponentTracePayload(
                        row_state=ExecutedRowState.SUCCESS,
                        executed_component_steps=(),
                    ),
                )
                for index, task_identity in enumerate(task_identities)
            ),
        )
        traces_ref = persist_test_record(
            self.store,
            EVALUATION_COMPONENT_TRACES_SCHEMA,
            traces.record_content(),
        )
        outputs_ref = persist_test_record(
            self.store,
            EVALUATION_OUTPUTS_SCHEMA,
            {"intent_id": intent.intent_id},
        )
        aggregate_ref = persist_test_record(
            self.store,
            "whetstone.rollout_aggregate",
            {"intent_id": intent.intent_id, "score": score},
        )
        reward = apply_reward_policy(
            self.reward_policy,
            aggregates={"score": score},
            evidence_role=EvaluationRole.INTERNAL,
            evidence_refs=(aggregate_ref,),
        )
        reward_ref = reward_reference(reward)
        persist_test_record(
            self.store,
            reward_ref.record_ref.schema_name,
            reward.record_content(),
        )
        evidence = EvaluationEvidence(
            schema_version=2,
            candidate=intent.candidate,
            evaluation_binding=intent.evaluation_binding,
            graph_hash=GRAPH_HASH,
            graph_config_ref="flow-graph",
            purpose=intent.purpose,
            dataset_identity="miprov2-flow-dataset",
            task_identities=task_identities,
            repeat_count=1,
            per_task_values=(score,) * len(task_identities),
            per_task_counts=(1,) * len(task_identities),
            row_accounting=RowAccounting(
                planned=len(task_identities),
                present=len(task_identities),
                missing=0,
                failed=0,
                invalid=0,
            ),
            component_traces_ref=traces_ref,
            outputs_ref=outputs_ref,
            aggregate_ref=aggregate_ref,
            aggregate_name="score",
            aggregate_value=score,
            aggregate_status="measured",
            reward_ref=reward_ref,
        )
        evidence_ref = persist_test_record(
            self.store,
            EVALUATION_EVIDENCE_SCHEMA,
            evidence.record_content(),
        )
        return IntentResolution(
            schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
            intent=intent,
            outcome=IntentOutcome.COMPLETED,
            detail=ResolutionDetail(
                classification=ResolutionClass.MEASURED,
                message="evaluation completed",
            ),
            evaluation_result_ref=evidence_ref,
            reward_evidence_refs=(aggregate_ref,),
            resolved_eval_config=intent.target_eval_config,
            reward_ref=reward_ref,
        )

    def validate_resolution_graph(self, resolution: IntentResolution) -> None:
        self.validation_calls.append(resolution)


def _load_context_tasks(
    store: ObjectStore,
    intent: EvaluationIntent,
) -> tuple[str, ...]:
    key = (
        f"whetstone.miprov2_intent_context:{intent.run_id}:{intent.intent_id}"
    )
    ref = store.resolve(key)
    assert ref is not None
    context = Miprov2IntentContext.model_validate(store.get(ref))
    return context.task_batch_identities


def _adapter(
    *,
    store: ObjectStore,
    state,
    driver,
):
    transport = FakeProposerTransport(
        {},
        default=("Instruction: improved {query}.",),
        execution_policy_hash=state.control.provider_execution_policy_hash,
        prompt_adapter_identity_hash=(
            state.control.prompt_adapter_identity_hash
        ),
    )
    adapter = Miprov2Adapter(
        store=store,
        proposer_config=state.control.prompt_model,
        transport=transport,
        eval_config_resolver=cast(
            Miprov2EvalConfigResolver,
            _CanonicalEvalConfigResolver(),
        ),
        proposal_executor=_proposal_executor(),
        driver=driver,
    )
    return adapter, transport


def test_real_adapter_reaches_terminal_and_replays_without_duplicate_effects(
    tmp_path,
) -> None:
    database = tmp_path / "miprov2-adapter-flow.sqlite"
    store = ObjectStore(SqliteBackend(database))
    driver, initial_state = make_minimal_miprov2_runtime()
    adapter, transport = _adapter(
        store=store,
        state=initial_state,
        driver=driver,
    )
    run = optimization_run_reference(
        OptimizationRun(
            run_id=initial_state.run_id,
            optimizer_config=initial_state.control.reference(),
            adapter_key=adapter.key,
            mode=StepMode.PROPOSAL_ONLY,
            terminal_output_contract=OutputContract(returned_proposal_count=1),
            template_render_contract=(
                initial_state.control.template_render_contract
            ),
            reward_policy=initial_state.control.reward_policy,
        )
    )
    service = _PersistingEvaluationService(
        store,
        initial_state.control.reward_policy,
    )
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=run,
        effect_authority=EffectAuthority.memory(),
        evaluation_service=service,
        adapter_replay_policy=ReplayPolicy.DURABLE_WORKFLOW,
    )
    initial_budget = BudgetState(
        remaining={
            "bootstrap_rollouts": 0,
            "proposal_calls": 2,
            "evaluations": 2,
            "task_rows": 6,
        }
    )
    requests = []
    results = []
    refs = []
    request = adapter.build_step_request(
        run=run,
        step_index=0,
        initial_state=initial_state,
        initial_budget=initial_budget,
    )
    for step_index in range(7):
        requests.append(request)
        result, result_ref = harness.run_step(request)
        results.append(result)
        refs.append(result_ref)
        assert result_ref == step_result_reference(result).record_ref
        assert result.request == step_request_reference(request)
        if step_index == 6:
            break
        request = adapter.build_step_request(
            step_index=step_index + 1,
            prior_result=result,
            prior_result_ref=result_ref,
        )
        assert request.prior_step_result_ref == result_ref
        assert request.prior_state_ref == result.state_ref
        assert request.budget == result.budget

    assert tuple(item.step_index for item in requests) == tuple(range(7))
    assert tuple(item.kind_label for item in requests) == (
        "proposal_model",
        "proposal_model",
        "eval_config_binding",
        "baseline_evaluation",
        "eval_config_binding",
        "sample_evaluation",
        "complete",
    )
    assert tuple(result.status for result in results) == (
        StepStatus.CONTINUE,
    ) * 6 + (StepStatus.COMPLETE,)
    assert tuple(result.budget_delta.consumed for result in results) == (
        {"proposal_calls": 1},
        {"proposal_calls": 1},
        {},
        {"evaluations": 1, "task_rows": 3},
        {},
        {"evaluations": 1, "task_rows": 3},
        {},
    )
    assert results[-1].budget.consumed == {
        "proposal_calls": 2,
        "evaluations": 2,
        "task_rows": 6,
    }
    assert "bootstrap_rollouts" not in results[-1].budget.consumed
    assert results[-1].budget.remaining == {
        "bootstrap_rollouts": 0,
        "proposal_calls": 0,
        "evaluations": 0,
        "task_rows": 0,
    }
    assert len(results[-1].accepted_candidates) == 1
    assert len(transport.calls) == 2
    assert [intent.purpose for intent in service.calls] == [
        "miprov2_baseline",
        "miprov2_sample",
    ]
    assert results[3].proposed_candidates == ()
    assert results[3].resolved_intents[
        0
    ].intent.candidate == candidate_reference(requests[3].candidates[0])
    assert results[5].proposed_candidates == (service.calls[1].candidate,)
    assert results[-1].accepted_candidates == (service.calls[1].candidate,)
    evidence = []
    reward_values = []
    for result in (results[3], results[5]):
        resolution = result.resolved_intents[0]
        assert resolution.evaluation_result_ref is not None
        evidence.append(
            EvaluationEvidence.model_validate(
                store.get(resolution.evaluation_result_ref.reference)
            )
        )
        assert resolution.reward_ref is not None
        reward_values.append(resolution.reward_ref.record.value)
    assert [item.aggregate_value for item in evidence] == [0.1, 0.9]
    assert reward_values == [0.1, 0.9]
    assert results[-1].state_ref is not None
    terminal_snapshot = store.get(results[-1].state_ref.reference)
    assert isinstance(terminal_snapshot, dict)
    terminal_state = Miprov2State.model_validate(
        terminal_snapshot[MIPROV2_STATE_KEY]
    )
    assert terminal_state.terminal_result is not None
    assert terminal_state.terminal_result.stats is not None
    assert tuple(
        item.score for item in terminal_state.terminal_result.stats.score_data
    ) == (10.0, 90.0)

    reopened_store = ObjectStore(SqliteBackend(database))
    replay_adapter, replay_transport = _adapter(
        store=reopened_store,
        state=initial_state,
        driver=driver,
    )
    replay_service = _PersistingEvaluationService(
        reopened_store,
        initial_state.control.reward_policy,
    )
    replay_harness = make_harness(
        store=reopened_store,
        adapter_registry=registry(replay_adapter),
        run=run,
        effect_authority=EffectAuthority.memory(),
        evaluation_service=replay_service,
        adapter_replay_policy=ReplayPolicy.DURABLE_WORKFLOW,
    )

    assert [replay_harness.run_step(item) for item in requests] == list(
        zip(results, refs, strict=True)
    )
    assert replay_transport.calls == []
    assert replay_service.calls == []
