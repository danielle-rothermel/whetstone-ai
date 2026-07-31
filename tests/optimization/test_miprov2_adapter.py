from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, get_type_hints

import pytest
from dr_code.eval import (
    DefinitionRef,
    EvalConfig,
    RepeatPlan,
    SamplingDefinition,
    TaskSet,
)
from dr_code.eval.identity import SCHEMA_EVAL_CONFIG, identity_hash_for
from dr_store import ObjectStore, SqliteBackend
from whetstone_envs.core import Instance

from whetstone.envs.registry import EnvSpec
from whetstone.envs.rollout_definition import (
    PromptInputError,
    initial_candidate,
    render_prompt,
    validate_candidate_prompt,
)
from whetstone.graph.rollout import EvaluationRole
from whetstone.lm.boundary import PlainPromptAdapter
from whetstone.optimization.adapters import MappingAdapterRegistry
from whetstone.optimization.harness import OptimizationHarness
from whetstone.optimization.identity import TypedRef
from whetstone.optimization.miprov2 import Miprov2Adapter
from whetstone.optimization.miprov2_bootstrap import BootstrapRolloutResult
from whetstone.optimization.miprov2_control import (
    Miprov2ComponentSpec,
    Miprov2InjectedDefaults,
    Miprov2ProgramLayout,
    configure_miprov2,
)
from whetstone.optimization.miprov2_demo import LabeledTaskDemo
from whetstone.optimization.miprov2_eval_config import (
    Miprov2EvalConfigBinding,
    Miprov2EvalConfigBindingRequest,
    derive_eval_config_reference,
)
from whetstone.optimization.miprov2_evidence import (
    EVALUATION_EVIDENCE_SCHEMA,
    ROLLOUT_AGGREGATE_SCHEMA,
    Miprov2IntentContext,
    Miprov2ResolvedEvaluation,
    load_miprov2_intent_context,
    resolve_miprov2_bootstrap,
)
from whetstone.optimization.miprov2_proposal import (
    Miprov2DatasetExample,
    Miprov2PromptComponent,
    Miprov2ProposalResponse,
)
from whetstone.optimization.miprov2_rng import Miprov2DurableBindings
from whetstone.optimization.miprov2_runtime import (
    Miprov2Driver,
    Miprov2EffectBudget,
    Miprov2State,
    render_miprov2_candidate,
)
from whetstone.optimization.miprov2_study import Miprov2Study
from whetstone.optimization.prompt_program import (
    PROMPT_PROGRAM_PAYLOAD_FIELD,
    PromptProgram,
    PromptProgramComponent,
    PromptProgramExample,
)
from whetstone.optimization.proposer import (
    FakeProposerTransport,
    ProposalDraft,
    ProposalRequest,
    ProposerConfig,
    prompt_adapter_identity_hash,
)
from whetstone.optimization.reward import (
    REWARD_RECORD_SCHEMA,
    Reward,
    RewardInputCitation,
)
from whetstone.optimization.schema import (
    BudgetState,
    Candidate,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    OptimizationStepRequest,
    OutputContract,
    ResolutionClass,
    ResolutionDetail,
    StepKind,
    StepMode,
    candidate_reference,
    eval_config_reference,
)

FULL_A = "a" * 64
FULL_B = "b" * 64
FULL_C = "c" * 64
FULL_D = "d" * 64
RUN_ID = "miprov2-adapter-test"
COMPONENT_ID = "user_prompt_template"
TASKS = tuple(
    hashlib.sha256(f"task-{index}".encode()).hexdigest() for index in range(4)
)


def _source_eval_config(sampling_hash: str) -> EvalConfig:
    definition = DefinitionRef(
        definition_id="test-eval",
        version="1",
        schema_name="dr_code.eval_definition",
        identity_hash=FULL_A,
    )
    identity = identity_hash_for(
        schema=SCHEMA_EVAL_CONFIG,
        payload={
            "definition_identity": definition.identity_hash,
            "sampling_config": sampling_hash,
            "evaluation_procedure_config": FULL_C,
            "aggregation_config": FULL_D,
        },
    )
    return EvalConfig(
        definition_ref=definition,
        sampling_config_hash=sampling_hash,
        evaluation_procedure_config_hash=FULL_C,
        aggregation_config_hash=FULL_D,
        config_identity_hash=identity,
    )


def _base_candidate() -> Candidate:
    return Candidate(
        candidate_id="base",
        base_ref="route-a",
        payload={
            COMPONENT_ID: "Answer {input}.",
            "fixed": "unchanged",
        },
    )


def _proposer_config() -> ProposerConfig:
    return ProposerConfig(
        provider_call_config_ref="provider://miprov2",
        provider_call_config_hash=FULL_A,
        temperature=1.0,
    )


def _control(
    *,
    multi_component: bool = False,
    minibatch: bool = False,
    minibatch_size: int = 2,
    max_bootstrapped_demos: int = 0,
    max_labeled_demos: int = 1,
    track_stats: bool = True,
    num_trials: int = 1,
    num_threads: int | None = None,
    provide_traceback: bool | None = None,
    teacher_settings: dict[str, Any] | None = None,
    teacher_compiled: bool | None = None,
    num_candidates: int = 2,
):
    prompt_adapter = PlainPromptAdapter()
    adapter_hash = prompt_adapter_identity_hash(prompt_adapter)
    payload = dict(_base_candidate().payload)
    layout = None
    if multi_component:
        payload["system_prompt"] = "Be concise."
        layout = Miprov2ProgramLayout(
            layout_id="unsupported-two-component-layout",
            component_specs=(
                Miprov2ComponentSpec(
                    component_id=COMPONENT_ID,
                    candidate_field=COMPONENT_ID,
                    prompt_format_identity_hash=adapter_hash,
                    required_placeholders=("input",),
                ),
                Miprov2ComponentSpec(
                    component_id="system",
                    candidate_field="system_prompt",
                    prompt_format_identity_hash=adapter_hash,
                ),
            ),
        )
    base = candidate_reference(
        _base_candidate().model_copy(update={"payload": payload})
    )
    validation = eval_config_reference(_source_eval_config(FULL_B))
    defaults = Miprov2InjectedDefaults(
        prompt_model=_proposer_config(),
        bootstrap_eval_source=eval_config_reference(
            _source_eval_config(FULL_C)
        ),
        validation_eval_source=validation,
        reward_policy_hash=FULL_D,
        provider_execution_policy_hash=FULL_B,
        task_model_identity_hash=FULL_C,
        prompt_adapter=prompt_adapter,
        max_errors=3,
        validation_eval_source_is_metric_authority=True,
    )
    return configure_miprov2(
        metric=validation,
        auto=None,
        num_candidates=num_candidates,
        num_trials=num_trials,
        num_threads=num_threads,
        provide_traceback=provide_traceback,
        teacher_settings=teacher_settings,
        teacher_compiled=teacher_compiled,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        minibatch=minibatch,
        minibatch_size=minibatch_size,
        base_candidate=base,
        trainset=TASKS[:2],
        valset=TASKS[2:],
        program_layout=layout,
        program_aware_proposer=False,
        data_aware_proposer=False,
        tip_aware_proposer=False,
        fewshot_aware_proposer=False,
        track_stats=track_stats,
        defaults=defaults,
    )


def _bindings(control) -> Miprov2DurableBindings:
    return Miprov2DurableBindings(
        control_identity_hash=control.identity_hash(),
        prompt_route_identity_hash=control.prompt_model.identity_hash(),
        task_route_identity_hash=control.task_model_identity_hash,
        execution_policy_identity_hash=(
            control.provider_execution_policy_hash
        ),
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
        base_candidate_identity_hash=control.base_candidate.identity_hash,
        teacher_candidate_identity_hash=(
            control.teacher_candidate.identity_hash
        ),
    )


def _labeled() -> tuple[LabeledTaskDemo, ...]:
    return tuple(
        LabeledTaskDemo(
            source_task_identity=task,
            inputs_by_component={COMPONENT_ID: {"input": f"q-{index}"}},
            outputs_by_component={COMPONENT_ID: {"output": f"a-{index}"}},
        )
        for index, task in enumerate(TASKS[:2])
    )


def _proposal_components() -> tuple[Miprov2PromptComponent, ...]:
    return (
        Miprov2PromptComponent(
            component_id=COMPONENT_ID,
            template="Answer {input}.",
            allowed_placeholders=("input",),
            rendering_rules="Format input into the native template.",
            example_execution="Answer q-0.",
        ),
    )


def _proposal_examples() -> tuple[Miprov2DatasetExample, ...]:
    return tuple(
        Miprov2DatasetExample(
            task_identity=task,
            rendered_record=f"input=q-{index}; output=a-{index}",
        )
        for index, task in enumerate(TASKS[:2])
    )


def _start(
    *,
    driver: Miprov2Driver | None = None,
    control=None,
    labeled_trainset: tuple[LabeledTaskDemo, ...] | None = None,
    budget: Miprov2EffectBudget | None = None,
    run_id: str = RUN_ID,
) -> tuple[Miprov2Driver, Miprov2State]:
    active_driver = driver or Miprov2Driver()
    active_control = control or _control()
    active_labeled = labeled_trainset or _labeled()
    active_components = _proposal_components()
    field_order = {COMPONENT_ID: ("input", "output")}
    if len(active_control.component_ids) > 1:
        active_labeled = tuple(
            item.model_copy(
                update={
                    "inputs_by_component": {
                        **item.inputs_by_component,
                        "system": {},
                    },
                    "outputs_by_component": {
                        **item.outputs_by_component,
                        "system": {"output": "concise"},
                    },
                }
            )
            for item in active_labeled
        )
        active_components = (
            *active_components,
            Miprov2PromptComponent(
                component_id="system",
                template="Be concise.",
                allowed_placeholders=(),
                rendering_rules="Render the system instruction.",
                example_execution="Be concise.",
            ),
        )
        field_order["system"] = ("output",)
    state = active_driver.start(
        run_id=run_id,
        control=active_control,
        bindings=_bindings(active_control),
        labeled_trainset=active_labeled,
        proposal_components=active_components,
        proposal_trainset=_proposal_examples(),
        component_field_order=field_order,
        budget=budget
        or Miprov2EffectBudget(
            bootstrap_rollouts=0,
            proposal_calls=2,
            evaluations=2,
        ),
    )
    return active_driver, _roundtrip(state)


def _roundtrip(state: Miprov2State) -> Miprov2State:
    return Miprov2State.model_validate_json(state.model_dump_json())


class _ExactEvalConfigResolver:
    def __init__(self) -> None:
        self.calls: list[Miprov2EvalConfigBindingRequest] = []

    def resolve(
        self,
        request: Miprov2EvalConfigBindingRequest,
    ) -> Miprov2EvalConfigBinding:
        self.calls.append(request)
        suffix = request.identity_hash()[:20]
        task_set = TaskSet(
            manifest_id=f"miprov2-tasks-{suffix}",
            version="1",
            dataset_revision="test",
            task_identities=request.task_batch_identities,
        )
        repeat_plan = RepeatPlan(
            plan_id=f"miprov2-repeats-{suffix}",
            version="1",
            task_identities=request.task_batch_identities,
            repeat_count=request.repeat_count,
        )
        sampling = SamplingDefinition(
            definition_id="miprov2-test-sampling",
            version="1",
        ).materialize(
            {
                "task_set_hash": task_set.identity_hash(),
                "repeat_plan_hash": repeat_plan.identity_hash(),
            }
        )
        return Miprov2EvalConfigBinding(
            request=request,
            task_set=task_set,
            repeat_plan=repeat_plan,
            sampling_config=sampling,
            eval_config=derive_eval_config_reference(
                request.source_eval_config,
                sampling,
            ),
        )


@dataclass
class _EffectJournal:
    drafts: dict[str, tuple[ProposalDraft, ...]] = field(default_factory=dict)


class _RecordingEffectExecutor:
    """Tiny durable executor fake shared by restarted adapter instances."""

    durability_scope_identity_hash = FULL_B

    def __init__(self, journal: _EffectJournal) -> None:
        self._journal = journal
        self.accepted: list[str] = []

    def execute(
        self,
        *,
        config: ProposerConfig,
        request: ProposalRequest,
        transport,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        identity = request.identity_hash()
        self.accepted.append(identity)
        persisted = self._journal.drafts.get(identity)
        if persisted is not None:
            return persisted
        drafts = transport.draft(config, request, count)
        self._journal.drafts[identity] = drafts
        return drafts


def _transport(*, default: str = "Improve {input}."):
    return FakeProposerTransport(
        {},
        default=(default,),
        execution_policy_hash=FULL_B,
        prompt_adapter_identity_hash=(
            prompt_adapter_identity_hash(PlainPromptAdapter())
        ),
    )


def _adapter(
    *,
    store: ObjectStore,
    resolver: _ExactEvalConfigResolver,
    executor: _RecordingEffectExecutor,
    transport=None,
    driver: Miprov2Driver | None = None,
) -> Miprov2Adapter:
    return Miprov2Adapter(
        store=store,
        proposer_config=_proposer_config(),
        transport=transport or _transport(),
        eval_config_resolver=resolver,
        proposal_effect_executor=executor,
        driver=driver,
    )


def _request(state: Miprov2State, *, ordinal: int) -> OptimizationStepRequest:
    counts = state.effect_counts
    return OptimizationStepRequest(
        run_id=state.run_id,
        step_id=f"{state.run_id}:miprov2-step-{ordinal}",
        optimizer_config_hash=state.control.identity_hash(),
        adapter_key="miprov2",
        mode=StepMode.PROPOSAL_ONLY,
        kind=StepKind.PROPOSAL,
        step_index=0,
        candidates=(state.control.base_candidate.record,),
        pools={"miprov2_state": state.model_dump(mode="json")},
        budget=BudgetState(
            consumed={
                label: counts[label]
                for label in (
                    "bootstrap_rollouts",
                    "proposal_calls",
                    "evaluations",
                )
                if counts[label]
            },
            remaining={
                "bootstrap_rollouts": (
                    state.budget.bootstrap_rollouts
                    - counts["bootstrap_rollouts"]
                ),
                "proposal_calls": (
                    state.budget.proposal_calls - counts["proposal_calls"]
                ),
                "evaluations": (
                    state.budget.evaluations - counts["evaluations"]
                ),
                "task_rows": 4,
            },
        ),
        output_contract=OutputContract(returned_proposal_count=1),
    )


def _state(output) -> Miprov2State:
    return Miprov2State.model_validate(output.state_delta["miprov2_state"])


def _finish_proposals(
    driver: Miprov2Driver,
    state: Miprov2State,
) -> Miprov2State:
    for ordinal in range(2):
        plan = driver.plan(_roundtrip(state))
        assert plan.kind == "proposal_model"
        assert plan.proposal_request is not None
        state = driver.fold_proposal(
            plan.state,
            Miprov2ProposalResponse(
                request_identity_hash=plan.proposal_request.identity_hash,
                text=f"Improved-{ordinal} {{input}}.",
                evidence={"ordinal": ordinal},
            ),
        )
    return _roundtrip(state)


def _typed_put(
    store: ObjectStore,
    schema: str,
    record: dict[str, Any],
) -> TypedRef:
    ref, _ = store.put(schema, record)
    return TypedRef(schema_name=ref.schema, content_hash=ref.content_hash)


def _resolution(
    store: ObjectStore,
    intent: EvaluationIntent,
    *,
    score: float,
    output_record: dict[str, Any] | None = None,
) -> IntentResolution:
    from whetstone.evaluation.schema import EVALUATION_OUTPUTS_SCHEMA

    # Evaluation rows are represented by the exact ordered task identities in
    # the persisted MIPROv2 context. The context is intentionally reloaded by
    # fold_resolution instead of being accepted from this test helper.
    context = load_miprov2_intent_context(store, intent)
    task_count = len(context.task_batch_identities)
    outputs_ref = _typed_put(
        store,
        EVALUATION_OUTPUTS_SCHEMA,
        output_record or {"outputs": []},
    )
    aggregate_ref = _typed_put(
        store,
        ROLLOUT_AGGREGATE_SCHEMA,
        {
            "name": "accuracy",
            "graph_hash": FULL_A,
            "eval_config_hash": intent.target_eval_config.identity_hash,
            "evaluation_context_id": intent.intent_id,
            "task_count": task_count,
            "repeat_count": 1,
            "aggregation_output": {"value": score, "status": "ok"},
            "rows_present": task_count,
            "rows_missing": 0,
            "rows_failed": 0,
            "rows_invalid": 0,
        },
    )
    reward = Reward(
        reward_name="miprov2-test-reward",
        value=score,
        reward_policy_hash=FULL_D,
        evidence_role=EvaluationRole.INTERNAL,
        input_citations=(
            RewardInputCitation(
                name="accuracy",
                value=score,
                contributed=score,
            ),
        ),
        evidence_ref_content_hash=aggregate_ref.content_hash,
    )
    reward_ref = _typed_put(
        store,
        REWARD_RECORD_SCHEMA,
        reward.record_content(),
    )
    evidence = {
        "candidate": intent.candidate.model_dump(mode="json"),
        "eval_config": intent.target_eval_config.model_dump(mode="json"),
        "graph_hash": FULL_A,
        "graph_config_ref": "graph://miprov2-test",
        "evaluation_role": EvaluationRole.INTERNAL.value,
        "evaluation_context_id": intent.intent_id,
        "purpose": intent.purpose,
        "dataset_identity": "test",
        "task_identities": list(context.task_batch_identities),
        "repeat_count": 1,
        "per_task_values": [score for _ in range(task_count)],
        "per_task_counts": [1 for _ in range(task_count)],
        "row_accounting": {
            "planned": task_count,
            "present": task_count,
            "missing": 0,
            "failed": 0,
            "invalid": 0,
        },
        "outputs_ref": outputs_ref.model_dump(mode="json"),
        "aggregate_ref": aggregate_ref.model_dump(mode="json"),
        "aggregate_name": "accuracy",
        "aggregate_value": score,
        "aggregate_status": "ok",
        "reward_ref": reward_ref.model_dump(mode="json"),
        "cache": {
            "partial_row_count": 0,
            "cache_hit_count": 0,
            "source_call_ids": [],
        },
        "concurrency_halved": False,
        "deadline_reached": False,
        "guard_timeouts": 0,
    }
    evidence_ref = _typed_put(
        store,
        EVALUATION_EVIDENCE_SCHEMA,
        evidence,
    )
    return IntentResolution(
        intent=intent,
        outcome=IntentOutcome.COMPLETED,
        detail=ResolutionDetail(
            classification=ResolutionClass.MEASURED,
            message="measured",
        ),
        evaluation_evidence_refs=(evidence_ref,),
        resolved_eval_config=intent.target_eval_config,
        reward_ref=reward_ref,
    )


def _plan_bootstrap_intent(
    store: ObjectStore,
    *,
    labeled_trainset: tuple[LabeledTaskDemo, ...] | None = None,
) -> tuple[EvaluationIntent, Miprov2IntentContext]:
    driver, state = _start(
        control=_control(
            max_bootstrapped_demos=1,
            max_labeled_demos=0,
        ),
        labeled_trainset=labeled_trainset,
        budget=Miprov2EffectBudget(
            bootstrap_rollouts=1,
            proposal_calls=2,
            evaluations=2,
        ),
    )
    adapter = _adapter(
        store=store,
        resolver=_ExactEvalConfigResolver(),
        executor=_RecordingEffectExecutor(_EffectJournal()),
        driver=driver,
    )
    state = _state(adapter.invoke(_request(state, ordinal=0), ()))
    rollout = adapter.invoke(_request(state, ordinal=1), ())
    intent = rollout.evaluation_intents[0]
    return intent, load_miprov2_intent_context(store, intent)


def _bootstrap_output_record(
    context: Miprov2IntentContext,
    *,
    candidate_id: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "outputs": [
            {
                "candidate_id": candidate_id,
                "instance_id": "bootstrap-instance",
                "task_identity": context.task_batch_identities[0],
                "repeat": 0,
                "rendered_prompt": "flattened provider prompt",
                "output_text": "generated answer",
                "score": 0.8,
                "failure_code": "",
                "component_trace_steps": [],
                "finish_reason": "stop",
                "provider_error": None,
                "max_budget": None,
                "over_budget": None,
            }
        ],
    }


class _ScoredEvaluationService:
    def __init__(self, store: ObjectStore) -> None:
        self._store = store
        self.intents: list[EvaluationIntent] = []

    def resolve_evaluation_intent(
        self,
        intent: EvaluationIntent,
    ) -> IntentResolution:
        self.intents.append(intent)
        score = 0.1 if intent.purpose == "miprov2_baseline" else 0.9
        return _resolution(self._store, intent, score=score)


def _snapshot_state(store: ObjectStore, state_ref: TypedRef) -> Miprov2State:
    snapshot = store.get(state_ref.reference)
    if not isinstance(snapshot, dict):
        raise AssertionError("MIPROv2 state snapshot must be an object")
    return Miprov2State.model_validate(snapshot["miprov2_state"])


def test_start_rejects_multicomponent_before_any_effect() -> None:
    control = _control(multi_component=True)
    with pytest.raises(ValueError, match="only one provider trace"):
        _start(control=control)


def test_runtime_preserves_compiled_zero_demo_teacher_semantics() -> None:
    _, state = _start(
        control=_control(
            max_bootstrapped_demos=1,
            max_labeled_demos=1,
            num_candidates=3,
            teacher_compiled=True,
        )
    )
    bootstrap = next(
        plan
        for plan in state.bootstrap_plans
        if plan.kind.value == "bootstrap"
    )

    assert bootstrap.teacher is not None
    assert bootstrap.teacher.reset_before_labeled_compile is False
    assert bootstrap.teacher.labeled_selection is None


def test_runtime_input_binding_roundtrips_and_rejects_tamper() -> None:
    _, state = _start()
    assert _roundtrip(state) == state
    record = state.model_dump(mode="json")
    record["labeled_trainset"][0]["inputs_by_component"][COMPONENT_ID][
        "input"
    ] = "tampered"
    with pytest.raises(ValueError, match="exact content binding"):
        Miprov2State.model_validate(record)


def test_runtime_rejects_self_consistent_noncanonical_rng_and_ledger() -> None:
    _, state = _start()
    rng = state.rng_checkpoint.state.restore()
    result = rng.randint(0, 10**9)
    extra_checkpoint = state.rng_checkpoint.append(
        rng=rng,
        phase="proposal",
        operation="randint",
        arguments=(0, 10**9),
        result=result,
    )
    rng_record = state.model_dump(mode="json")
    rng_record["rng_checkpoint"] = extra_checkpoint.model_dump(mode="json")
    with pytest.raises(ValueError, match="canonical evidence replay"):
        Miprov2State.model_validate(rng_record)

    ledger_record = state.model_dump(mode="json")
    ledger_record["completed_effects"] = [
        {
            "kind": "proposal_calls",
            "identity_hash": hashlib.sha256(b"fabricated").hexdigest(),
            "task_rows": 0,
        }
    ]
    with pytest.raises(ValueError, match="completed-effect ledger"):
        Miprov2State.model_validate(ledger_record)


def test_runtime_rejects_bootstrap_cursor_tampering() -> None:
    driver, state = _start(
        control=_control(
            max_bootstrapped_demos=1,
            max_labeled_demos=0,
        ),
        budget=Miprov2EffectBudget(
            bootstrap_rollouts=2,
            proposal_calls=2,
            evaluations=2,
        ),
    )
    binding_plan = driver.plan(state)
    assert binding_plan.eval_config_binding is not None
    bound = driver.fold_eval_config_binding(
        binding_plan.state,
        _ExactEvalConfigResolver().resolve(binding_plan.eval_config_binding),
    )
    rollout_plan = driver.plan(bound)
    attempt = rollout_plan.bootstrap_rollout
    assert attempt is not None
    folded = driver.fold_bootstrap(
        rollout_plan.state,
        BootstrapRolloutResult(
            attempt_identity_hash=attempt.identity_hash(),
            source_rollout_identity=FULL_A,
            source_trace_identity=FULL_B,
            source_output_identity=FULL_C,
            source_score_identity=FULL_D,
            metric_present=True,
            score=1.0,
            trace_steps=(),
        ),
    )
    record = folded.model_dump(mode="json")
    record["bootstrap_plan_index"] += 1

    with pytest.raises(ValueError, match="canonical bootstrap output"):
        Miprov2State.model_validate(record)


def test_dynamic_eval_config_binds_exact_baseline_tasks() -> None:
    driver, state = _start()
    state = _finish_proposals(driver, state)

    binding_plan = driver.plan(state)
    assert binding_plan.kind == "eval_config_binding"
    request = binding_plan.eval_config_binding
    assert request is not None
    assert request.purpose == "baseline"
    assert request.task_batch_identities == TASKS[2:]
    assert request.source_eval_config == state.control.validation_eval_source
    assert request.execution_policy.num_threads is None
    assert request.execution_policy.max_errors == 3
    assert request.execution_policy.provide_traceback is None

    resolver = _ExactEvalConfigResolver()
    binding = resolver.resolve(request)
    state = driver.fold_eval_config_binding(binding_plan.state, binding)
    baseline = driver.plan(_roundtrip(state))

    assert baseline.kind == "baseline_evaluation"
    assert baseline.evaluation is not None
    assert baseline.evaluation.eval_config == binding.eval_config
    assert binding.task_set.task_identities == TASKS[2:]
    assert (
        binding.eval_config.record.sampling_config_hash
        == binding.sampling_config.config_identity_hash
    )
    assert binding.eval_config != request.source_eval_config


def test_runtime_rejects_self_consistent_noncanonical_baseline_spec() -> None:
    driver, state = _start()
    state = _finish_proposals(driver, state)
    planned = driver.plan(state).state
    assert planned.pending_evaluation_spec is not None
    assert planned.pending_eval_binding_request is not None
    foreign = _base_candidate().model_copy(
        update={"candidate_id": "foreign-baseline"}
    )
    tampered_spec = planned.pending_evaluation_spec.model_copy(
        update={
            "candidate": foreign,
            "categorical_combination_identity_hash": FULL_A,
        }
    )
    tampered_request = planned.pending_eval_binding_request.model_copy(
        update={"effect_identity_hash": tampered_spec.identity_hash()}
    )
    record = planned.model_dump(mode="json")
    record["pending_evaluation_spec"] = tampered_spec.model_dump(mode="json")
    record["pending_eval_binding_request"] = tampered_request.model_dump(
        mode="json"
    )

    with pytest.raises(ValueError, match="canonical study replay"):
        Miprov2State.model_validate(record)


def test_every_effect_binds_exact_evaluation_execution_policy() -> None:
    control = _control(
        max_bootstrapped_demos=1,
        max_labeled_demos=0,
        num_threads=7,
        provide_traceback=True,
        teacher_settings={
            "temperature": 0.3,
            "top_p": 0.8,
            "route": {"tags": ["teacher"]},
        },
    )
    driver, state = _start(
        control=control,
        budget=Miprov2EffectBudget(
            bootstrap_rollouts=2,
            proposal_calls=2,
            evaluations=2,
        ),
    )
    bootstrap = driver.plan(state)
    assert bootstrap.eval_config_binding is not None
    policy = bootstrap.eval_config_binding.execution_policy
    assert policy.num_threads == 7
    assert policy.max_errors == 3
    assert policy.provide_traceback is True
    assert policy.provider_parameters == {
        "temperature": 0.3,
        "extra_body": {
            "top_p": 0.8,
            "route": {"tags": ["teacher"]},
        },
    }
    assert policy.rollout_id is None
    assert policy.copy_task_model is False
    assert len(policy.identity_hash()) == 64


def test_teacher_excludes_equal_content_under_another_identity() -> None:
    duplicate_content = tuple(
        LabeledTaskDemo(
            source_task_identity=task,
            inputs_by_component={COMPONENT_ID: {"input": "same"}},
            outputs_by_component={COMPONENT_ID: {"output": "same"}},
        )
        for task in TASKS[:2]
    )
    driver, state = _start(
        control=_control(
            max_bootstrapped_demos=1,
            max_labeled_demos=1,
            num_candidates=3,
        ),
        labeled_trainset=duplicate_content,
        budget=Miprov2EffectBudget(
            bootstrap_rollouts=1,
            proposal_calls=3,
            evaluations=2,
        ),
    )
    plan = driver.plan(state)
    teacher = plan.state.pending_bootstrap_candidate
    assert teacher is not None
    program = PromptProgram.model_validate(
        teacher.record.payload[PROMPT_PROGRAM_PAYLOAD_FIELD]
    )
    assert program.components[0].examples == ()


def test_bootstrap_preserves_all_native_inputs_not_rendered_prompt(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "miprov2-bootstrap.sqlite"))
    labeled = tuple(
        LabeledTaskDemo(
            source_task_identity=task,
            inputs_by_component={
                COMPONENT_ID: {
                    "input": f"q-{index}",
                    "context": f"context-{index}",
                }
            },
            outputs_by_component={COMPONENT_ID: {"output": f"a-{index}"}},
        )
        for index, task in enumerate(TASKS[:2])
    )
    intent, context = _plan_bootstrap_intent(
        store,
        labeled_trainset=labeled,
    )
    candidate_id = intent.candidate.record.candidate_id
    output_record = _bootstrap_output_record(
        context,
        candidate_id=candidate_id,
    )
    projection = context.trace_components[0]
    output_record["outputs"][0]["component_trace_steps"] = [
        {
            "component_id": projection.component_id,
            "inputs": projection.inputs,
            "outputs": {projection.output_field: "generated answer"},
        }
    ]
    resolution = _resolution(
        store,
        intent,
        score=0.8,
        output_record=output_record,
    )

    resolved = resolve_miprov2_bootstrap(store, resolution)

    trace = resolved.trace_steps[0]
    assert trace.inputs == context.trace_components[0].inputs
    assert set(trace.inputs) == {"input", "context"}
    assert "rendered_prompt" not in trace.inputs
    assert trace.outputs == {"output": "generated answer"}


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("unknown_field", "Extra inputs are not permitted"),
        ("coercible_rendered_prompt", "valid string"),
        ("coercible_output_text", "valid string"),
        ("coercible_max_budget", "valid integer"),
        ("malformed_component_trace", "Extra inputs are not permitted"),
        ("nonfinite_score", "finite number"),
    ),
)
def test_bootstrap_rejects_canonical_output_schema_drift(
    tmp_path,
    monkeypatch,
    case: str,
    message: str,
) -> None:
    from whetstone.evaluation.schema import EVALUATION_OUTPUTS_SCHEMA

    store = ObjectStore(SqliteBackend(tmp_path / f"{case}.sqlite"))
    intent, context = _plan_bootstrap_intent(store)
    candidate_id = intent.candidate.record.candidate_id
    output_record = _bootstrap_output_record(
        context,
        candidate_id=candidate_id,
    )
    resolution = _resolution(
        store,
        intent,
        score=0.8,
        output_record=output_record,
    )
    row = output_record["outputs"][0]
    if case == "unknown_field":
        row["unexpected"] = "drift"
    elif case == "coercible_rendered_prompt":
        row["rendered_prompt"] = 1
    elif case == "coercible_output_text":
        row["output_text"] = 1
    elif case == "coercible_max_budget":
        row["max_budget"] = "100"
    elif case == "malformed_component_trace":
        projection = context.trace_components[0]
        row["component_trace_steps"] = [
            {
                "component_id": projection.component_id,
                "inputs": projection.inputs,
                "outputs": {projection.output_field: "generated answer"},
                "unexpected": "drift",
            }
        ]
    elif case == "nonfinite_score":
        row["score"] = float("nan")
    else:
        raise AssertionError(f"unhandled test case: {case}")
    canonical_get = store.get

    def _get_with_drift(reference):
        if reference.schema == EVALUATION_OUTPUTS_SCHEMA:
            return output_record
        return canonical_get(reference)

    monkeypatch.setattr(store, "get", _get_with_drift)

    with pytest.raises(ValueError, match=message):
        resolve_miprov2_bootstrap(store, resolution)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("no_rows", "exactly one output row"),
        ("extra_row", "exactly one output row"),
        ("candidate", "candidate, task, or repeat"),
        ("task", "candidate, task, or repeat"),
        ("repeat", "candidate, task, or repeat"),
        ("missing_output", "not a successful generation"),
        ("failure", "not a successful generation"),
        ("trace_count", "every ordered component trace"),
        ("trace_component", "component trace conflicts with context"),
        ("trace_inputs", "component trace conflicts with context"),
        ("trace_outputs", "component trace conflicts with context"),
        ("trace_output_type", "component trace conflicts with context"),
    ),
)
def test_bootstrap_rejects_output_semantic_mismatch(
    tmp_path,
    case: str,
    message: str,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / f"{case}.sqlite"))
    intent, context = _plan_bootstrap_intent(store)
    candidate_id = intent.candidate.record.candidate_id
    output_record = _bootstrap_output_record(
        context,
        candidate_id=candidate_id,
    )
    row = output_record["outputs"][0]
    projection = context.trace_components[0]
    trace = {
        "component_id": projection.component_id,
        "inputs": projection.inputs,
        "outputs": {projection.output_field: "generated answer"},
    }
    row["component_trace_steps"] = [trace]
    if case == "no_rows":
        output_record["outputs"] = []
    elif case == "extra_row":
        output_record["outputs"].append({**row, "repeat": 1})
    elif case == "candidate":
        output_record["candidate_id"] = "other-candidate"
        row["candidate_id"] = "other-candidate"
    elif case == "task":
        row["task_identity"] = TASKS[1]
    elif case == "repeat":
        row["repeat"] = 1
    elif case == "missing_output":
        row["output_text"] = None
    elif case == "failure":
        row["failure_code"] = "provider_error"
    elif case == "trace_count":
        row["component_trace_steps"].append(trace)
    elif case == "trace_component":
        trace["component_id"] = "other-component"
    elif case == "trace_inputs":
        trace["inputs"] = {"input": "other"}
    elif case == "trace_outputs":
        trace["outputs"] = {"other-output": "generated answer"}
    elif case == "trace_output_type":
        trace["outputs"] = {projection.output_field: 1}
    else:
        raise AssertionError(f"unhandled test case: {case}")
    resolution = _resolution(
        store,
        intent,
        score=0.8,
        output_record=output_record,
    )

    with pytest.raises(ValueError, match=message):
        resolve_miprov2_bootstrap(store, resolution)


def test_evaluations_have_no_direct_scalar_fold_lane() -> None:
    parameters = inspect.signature(Miprov2Driver.fold_evaluation).parameters

    assert tuple(parameters) == ("self", "state", "resolved")
    assert "score" not in parameters
    assert get_type_hints(Miprov2Driver.fold_evaluation)["resolved"] is (
        Miprov2ResolvedEvaluation
    )
    source = inspect.getsource(Miprov2Adapter)
    assert "score=" not in source
    assert "resolve_miprov2_evaluation" in source
    assert "fold_resolution" in source


def test_instruction_and_structured_examples_remain_separate() -> None:
    driver, state = _start()
    state = _finish_proposals(driver, state)
    plan = driver.plan(state)
    assert plan.kind == "eval_config_binding"
    state = plan.state
    assert state.study_demo_candidates is not None
    labeled_index = next(
        index
        for index, demo_set in enumerate(state.study_demo_candidates)
        if demo_set.demos_for(COMPONENT_ID)
    )
    params = (
        ("0_predictor_instruction", 1),
        ("0_predictor_demos", labeled_index),
    )
    space = driver._space(state)
    candidate = render_miprov2_candidate(
        control=state.control,
        instruction_pools=state.instruction_pools,
        demo_candidates=state.study_demo_candidates,
        params=params,
        categorical_combination_identity_hash=(
            space.combination_identity_hash(params)
        ),
    )

    instruction = candidate.payload[COMPONENT_ID]
    program = PromptProgram.model_validate(
        candidate.payload[PROMPT_PROGRAM_PAYLOAD_FIELD]
    )
    assert instruction == "Improved-1 {input}."
    assert "q-0" not in instruction
    assert "a-0" not in instruction
    assert program.components[0].examples
    first_example = program.components[0].examples[0]
    index = first_example.inputs["input"].removeprefix("q-")
    assert first_example.outputs == {"output": f"a-{index}"}


def test_execution_surface_renders_demos_and_fails_before_provider() -> None:
    class _Surface:
        naive_template = "Answer {question}."
        ceiling_template = "Carefully answer {question}."

        @staticmethod
        def render(template, instance):
            return template.format(**dict(instance.prompt_inputs))

    env = cast(
        EnvSpec,
        SimpleNamespace(name="test", surface=_Surface()),
    )
    instance = cast(
        Instance,
        SimpleNamespace(
            prompt_inputs={"question": "What is the current task?"}
        ),
    )
    legacy = initial_candidate(env)
    legacy_rendered = render_prompt(env, legacy, instance)
    program = PromptProgram(
        components=(
            PromptProgramComponent(
                component_id=COMPONENT_ID,
                candidate_field=COMPONENT_ID,
                examples=(
                    PromptProgramExample(
                        inputs={"question": "What is the demo?"},
                        outputs={"answer": "The exact demo answer."},
                    ),
                ),
            ),
        )
    )
    structured = legacy.model_copy(
        update={
            "payload": {
                **legacy.payload,
                PROMPT_PROGRAM_PAYLOAD_FIELD: program.model_dump(mode="json"),
            }
        }
    )

    rendered = render_prompt(env, structured, instance)

    assert render_prompt(env, legacy, instance) == legacy_rendered
    assert "question: What is the demo?" in rendered
    assert "answer: The exact demo answer." in rendered
    assert rendered.index("### Example 1") < rendered.index(
        "Now complete the current task."
    )
    assert rendered.endswith(legacy_rendered)

    mismatched_program = program.model_copy(
        update={
            "components": (
                program.components[0].model_copy(
                    update={"candidate_field": "another_prompt_field"}
                ),
            )
        }
    )
    mismatched = structured.model_copy(
        update={
            "payload": {
                **structured.payload,
                PROMPT_PROGRAM_PAYLOAD_FIELD: mismatched_program.model_dump(
                    mode="json"
                ),
            }
        }
    )
    with pytest.raises(ValueError, match="exactly one prompt-program"):
        render_prompt(env, mismatched, instance)

    invalid = structured.model_copy(
        update={
            "payload": {
                **structured.payload,
                COMPONENT_ID: "Reveal {private_gold}.",
            }
        }
    )
    provider_calls: list[Candidate] = []

    def validate_then_call_provider() -> None:
        validate_candidate_prompt(env, invalid, (instance,))
        provider_calls.append(invalid)

    with pytest.raises(PromptInputError, match="private_gold"):
        validate_then_call_provider()
    assert provider_calls == []


def test_provider_result_replays_without_a_duplicate_call_after_restart(
    tmp_path,
) -> None:
    database = tmp_path / "miprov2-replay.sqlite"
    driver, initial = _start()
    journal = _EffectJournal()
    first_transport = _transport()
    first = _adapter(
        store=ObjectStore(SqliteBackend(database)),
        resolver=_ExactEvalConfigResolver(),
        executor=_RecordingEffectExecutor(journal),
        transport=first_transport,
        driver=driver,
    )

    first_output = first.invoke(_request(initial, ordinal=0), ())
    assert len(first_transport.calls) == 1

    restarted_transport = _transport(default="must-not-run {input}.")
    restarted = _adapter(
        store=ObjectStore(SqliteBackend(database)),
        resolver=_ExactEvalConfigResolver(),
        executor=_RecordingEffectExecutor(journal),
        transport=restarted_transport,
        driver=Miprov2Driver(),
    )
    replay = restarted.invoke(_request(_roundtrip(initial), ordinal=0), ())

    assert restarted_transport.calls == []
    assert replay.state_delta == first_output.state_delta
    assert replay.budget_delta == first_output.budget_delta


def test_proposal_claims_are_namespaced_by_run_in_shared_store(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "miprov2-two-runs.sqlite"))
    first_driver, first_state = _start(run_id="miprov2-run-a")
    second_driver, second_state = _start(run_id="miprov2-run-b")
    first_plan = first_driver.plan(first_state)
    second_plan = second_driver.plan(second_state)
    assert first_plan.proposal_request is not None
    assert second_plan.proposal_request is not None
    assert (
        first_plan.proposal_request.identity_hash
        == second_plan.proposal_request.identity_hash
    )

    journal = _EffectJournal()
    transport = _transport()
    first = _adapter(
        store=store,
        resolver=_ExactEvalConfigResolver(),
        executor=_RecordingEffectExecutor(journal),
        transport=transport,
        driver=first_driver,
    )
    second = _adapter(
        store=store,
        resolver=_ExactEvalConfigResolver(),
        executor=_RecordingEffectExecutor(journal),
        transport=transport,
        driver=second_driver,
    )

    first_output = first.invoke(_request(first_state, ordinal=0), ())
    second_output = second.invoke(_request(second_state, ordinal=0), ())

    assert _state(first_output).run_id == "miprov2-run-a"
    assert _state(second_output).run_id == "miprov2-run-b"
    assert len(transport.calls) == 2
    assert transport.calls[0][1].run_id == "miprov2-run-a"
    assert transport.calls[1][1].run_id == "miprov2-run-b"


def test_typed_evidence_bridge_rejects_tampered_intent_context(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "miprov2-tamper.sqlite"))
    driver, state = _start()
    adapter = _adapter(
        store=store,
        resolver=_ExactEvalConfigResolver(),
        executor=_RecordingEffectExecutor(_EffectJournal()),
        driver=driver,
    )
    for ordinal in range(2):
        state = _state(adapter.invoke(_request(state, ordinal=ordinal), ()))
    state = _state(adapter.invoke(_request(state, ordinal=2), ()))
    output = adapter.invoke(_request(state, ordinal=3), ())
    intent = output.evaluation_intents[0]
    resolution = _resolution(store, intent, score=0.25)
    tampered_intent = intent.model_copy(update={"purpose": "miprov2_sample"})
    tampered = resolution.model_copy(update={"intent": tampered_intent})

    with pytest.raises(ValueError, match="purpose conflicts with context"):
        adapter.fold_resolution(_state(output), tampered)


def test_harness_folds_prior_resolutions_without_state_pool_injection(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "miprov2-harness.sqlite"))
    driver, initial = _start()
    service = _ScoredEvaluationService(store)
    adapter = _adapter(
        store=store,
        resolver=_ExactEvalConfigResolver(),
        executor=_RecordingEffectExecutor(_EffectJournal()),
        driver=driver,
    )
    harness = OptimizationHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry({"miprov2": adapter}),
        evaluation_service=service,
    )
    prior = None
    prior_ref = None
    results = []
    for step_index in range(7):
        if step_index == 0:
            request = adapter.build_step_request(
                step_index=0,
                initial_state=initial,
                initial_budget=BudgetState(
                    remaining={
                        "bootstrap_rollouts": 0,
                        "proposal_calls": 2,
                        "evaluations": 2,
                        "task_rows": 4,
                    }
                ),
            )
        else:
            request = adapter.build_step_request(
                step_index=step_index,
                prior_result=prior,
                prior_result_ref=prior_ref,
            )
        if step_index:
            assert request.pools == {}
        assert request.output_contract.returned_proposal_count == (
            1 if step_index == 6 else 0
        )
        if step_index == 1:
            assert prior_ref is not None
            with pytest.raises(ValueError, match="exact record"):
                adapter.build_step_request(
                    step_index=step_index,
                    prior_result=prior,
                    prior_result_ref=prior_ref.model_copy(
                        update={"content_hash": FULL_A}
                    ),
                )
            assert request.prior_state_ref is not None
            tampered = request.model_copy(
                update={
                    "prior_state_ref": request.prior_state_ref.model_copy(
                        update={"content_hash": FULL_A}
                    )
                }
            )
            with pytest.raises(ValueError, match="exact state snapshot"):
                adapter.invoke(tampered, ())
        prior, prior_ref = harness.run_step(request)
        results.append(prior)

    assert results[3].resolved_intents[0].intent.purpose == (
        "miprov2_baseline"
    )
    after_baseline_ref = results[4].state_ref
    assert after_baseline_ref is not None
    after_baseline = _snapshot_state(store, after_baseline_ref)
    assert after_baseline.study_transcript is not None
    assert after_baseline.study_transcript.baseline.score == 10.0
    assert results[5].resolved_intents[0].intent.purpose == "miprov2_sample"
    assert prior is not None
    assert prior.accepted_candidates == (
        results[5].resolved_intents[0].intent.candidate,
    )
    final_state_ref = prior.state_ref
    assert final_state_ref is not None
    final_state = _snapshot_state(store, final_state_ref)
    assert final_state.effect_counts["task_rows"] == 4
    assert [intent.purpose for intent in service.intents] == [
        "miprov2_baseline",
        "miprov2_sample",
    ]


def test_harness_rejects_budget_state_drift_before_proposal_endpoint(
    tmp_path,
) -> None:
    store = ObjectStore(
        SqliteBackend(tmp_path / "miprov2-budget-drift.sqlite")
    )
    driver, initial = _start()
    transport = _transport()
    executor = _RecordingEffectExecutor(_EffectJournal())
    adapter = _adapter(
        store=store,
        resolver=_ExactEvalConfigResolver(),
        executor=executor,
        transport=transport,
        driver=driver,
    )
    harness = OptimizationHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry({"miprov2": adapter}),
        evaluation_service=_ScoredEvaluationService(store),
    )
    request = adapter.build_step_request(
        step_index=0,
        initial_state=initial,
        initial_budget=BudgetState(
            remaining={
                "bootstrap_rollouts": 0,
                "proposal_calls": 1,
                "evaluations": 2,
                "task_rows": 4,
            }
        ),
    )

    with pytest.raises(ValueError, match="disagrees with durable state"):
        harness.run_step(request)

    assert transport.calls == []
    assert executor.accepted == []
    assert harness.resolve_step_result(RUN_ID, 0) is None


def test_harness_exhaustion_stops_before_proposal_endpoint(tmp_path) -> None:
    store = ObjectStore(
        SqliteBackend(tmp_path / "miprov2-budget-exhausted.sqlite")
    )
    driver, initial = _start(
        budget=Miprov2EffectBudget(
            bootstrap_rollouts=0,
            proposal_calls=0,
            evaluations=2,
        )
    )
    transport = _transport()
    executor = _RecordingEffectExecutor(_EffectJournal())
    adapter = _adapter(
        store=store,
        resolver=_ExactEvalConfigResolver(),
        executor=executor,
        transport=transport,
        driver=driver,
    )
    harness = OptimizationHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry({"miprov2": adapter}),
        evaluation_service=_ScoredEvaluationService(store),
    )
    with pytest.raises(ValueError, match="proposal_calls budget exhausted"):
        adapter.build_step_request(
            step_index=0,
            initial_state=initial,
            initial_budget=BudgetState(
                remaining={
                    "bootstrap_rollouts": 0,
                    "proposal_calls": 0,
                    "evaluations": 2,
                    "task_rows": 4,
                }
            ),
        )

    assert transport.calls == []
    assert executor.accepted == []
    assert harness.resolve_step_result(RUN_ID, 0) is None


def test_full_size_minibatch_preserves_order_and_rng_checkpoint(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "miprov2-full-batch.sqlite"))
    resolver = _ExactEvalConfigResolver()
    driver, state = _start(control=_control(minibatch=True, minibatch_size=2))
    adapter = _adapter(
        store=store,
        resolver=resolver,
        executor=_RecordingEffectExecutor(_EffectJournal()),
        driver=driver,
    )
    for ordinal in range(2):
        state = _state(adapter.invoke(_request(state, ordinal=ordinal), ()))
    state = _state(adapter.invoke(_request(state, ordinal=2), ()))
    baseline_output = adapter.invoke(_request(state, ordinal=3), ())
    state = adapter.fold_resolution(
        _state(baseline_output),
        _resolution(
            store,
            baseline_output.evaluation_intents[0],
            score=0.1,
        ),
    )
    before = state.rng_checkpoint

    tamper_state = driver.plan(state).state
    assert tamper_state.pending_evaluation_spec is not None
    assert tamper_state.pending_evaluation_spec.suggestion is not None
    assert tamper_state.pending_eval_binding_request is not None
    original_suggestion = tamper_state.pending_evaluation_spec.suggestion
    foreign_suggestion = original_suggestion.model_copy(
        update={"trial_number": (original_suggestion.trial_number + 10)}
    )
    tampered_spec = tamper_state.pending_evaluation_spec.model_copy(
        update={"suggestion": foreign_suggestion}
    )
    tampered_request = tamper_state.pending_eval_binding_request.model_copy(
        update={"effect_identity_hash": tampered_spec.identity_hash()}
    )
    record = tamper_state.model_dump(mode="json")
    record["pending_evaluation_spec"] = tampered_spec.model_dump(mode="json")
    record["pending_eval_binding_request"] = tampered_request.model_dump(
        mode="json"
    )
    with pytest.raises(ValueError, match="canonical study replay"):
        Miprov2State.model_validate(record)

    sample_binding = adapter.invoke(_request(state, ordinal=4), ())
    sample_state = _state(sample_binding)
    after = sample_state.rng_checkpoint

    assert resolver.calls[-1].purpose == "sample"
    assert resolver.calls[-1].task_batch_identities == TASKS[2:]
    assert after == before


def _complete_for_track_stats(
    directory: Path,
    *,
    track_stats: bool,
) -> Miprov2State:
    store = ObjectStore(
        SqliteBackend(directory / f"miprov2-track-stats-{track_stats}.sqlite")
    )
    driver, state = _start(control=_control(track_stats=track_stats))
    adapter = _adapter(
        store=store,
        resolver=_ExactEvalConfigResolver(),
        executor=_RecordingEffectExecutor(_EffectJournal()),
        driver=driver,
    )
    for ordinal in range(2):
        state = _state(adapter.invoke(_request(state, ordinal=ordinal), ()))
    state = _state(adapter.invoke(_request(state, ordinal=2), ()))
    baseline = adapter.invoke(_request(state, ordinal=3), ())
    state = adapter.fold_resolution(
        _state(baseline),
        _resolution(
            store,
            baseline.evaluation_intents[0],
            score=0.9,
        ),
    )
    state = _state(adapter.invoke(_request(state, ordinal=4), ()))
    sample = adapter.invoke(_request(state, ordinal=5), ())
    state = adapter.fold_resolution(
        _state(sample),
        _resolution(
            store,
            sample.evaluation_intents[0],
            score=0.1,
        ),
    )
    return _state(adapter.invoke(_request(state, ordinal=6), ()))


def test_terminal_result_gates_only_detailed_track_stats(tmp_path) -> None:
    tracked = _complete_for_track_stats(tmp_path, track_stats=True)
    untracked = _complete_for_track_stats(tmp_path, track_stats=False)
    tracked_result = tracked.terminal_result
    untracked_result = untracked.terminal_result

    assert tracked_result is not None
    assert untracked_result is not None
    assert tracked_result.winner == untracked_result.winner
    assert tracked_result.winner == tracked.control.base_candidate
    assert tracked_result.winner_score == untracked_result.winner_score == 90.0
    assert tracked_result.track_stats is True
    assert untracked_result.track_stats is False
    assert untracked_result.stats is None

    stats = tracked_result.stats
    assert stats is not None
    assert stats.study_transcript == tracked.study_transcript
    assert stats.fully_evaluated_candidates == (
        tracked.fully_evaluated_candidates
    )
    assert stats.completed_effects == tracked.completed_effects
    assert stats.instruction_pools == tracked.instruction_pools
    assert stats.demo_candidates == tracked.study_demo_candidates
    assert (
        stats.effect_counts
        == tracked.effect_counts
        == {
            "bootstrap_rollouts": 0,
            "proposal_calls": 2,
            "evaluations": 2,
            "task_rows": 4,
        }
    )
    assert stats.cumulative_evaluation_calls == 4
    assert tuple(item.source for item in stats.score_data) == (
        "baseline",
        "sample",
    )
    assert tuple(item.score for item in stats.candidate_programs) == (
        90.0,
        10.0,
    )
    assert stats.mb_candidate_programs == ()
    assert tuple(item.log_key for item in stats.trial_logs) == (1, 2)
    assert stats.prompt_model_total_calls == stats.total_calls == 0

    tampered = tracked.model_dump(mode="json")
    tampered["terminal_result"]["winner_score"] = 89.0
    with pytest.raises(ValueError, match="canonical best full evaluation"):
        Miprov2State.model_validate(tampered)


def test_nonminibatch_repeated_suggestion_retains_ordered_evaluations(
    tmp_path,
    monkeypatch,
) -> None:
    def repeat_baseline_params(study, trial):
        del trial
        return study.space.baseline_params

    monkeypatch.setattr(Miprov2Study, "_suggest", repeat_baseline_params)
    store = ObjectStore(SqliteBackend(tmp_path / "miprov2-repeat.sqlite"))
    driver, state = _start(
        control=_control(num_trials=2),
        budget=Miprov2EffectBudget(
            bootstrap_rollouts=0,
            proposal_calls=2,
            evaluations=3,
        ),
    )
    adapter = _adapter(
        store=store,
        resolver=_ExactEvalConfigResolver(),
        executor=_RecordingEffectExecutor(_EffectJournal()),
        driver=driver,
    )
    for ordinal in range(2):
        state = _state(adapter.invoke(_request(state, ordinal=ordinal), ()))
    state = _state(adapter.invoke(_request(state, ordinal=2), ()))
    baseline = adapter.invoke(_request(state, ordinal=3), ())
    state = adapter.fold_resolution(
        _state(baseline),
        _resolution(
            store,
            baseline.evaluation_intents[0],
            score=0.1,
        ),
    )

    repeated_refs = []
    for offset, score in enumerate((0.5, 0.6), start=4):
        state = _state(adapter.invoke(_request(state, ordinal=offset), ()))
        sample = adapter.invoke(_request(state, ordinal=offset + 10), ())
        repeated_refs.append(sample.evaluation_intents[0].candidate)
        state = adapter.fold_resolution(
            _state(sample),
            _resolution(
                store,
                sample.evaluation_intents[0],
                score=score,
            ),
        )

    assert repeated_refs[0] == repeated_refs[1]
    assert state.fully_evaluated_candidates == (
        state.control.base_candidate,
        repeated_refs[0],
        repeated_refs[1],
    )
    complete = _state(adapter.invoke(_request(state, ordinal=20), ()))
    assert complete.terminal_result is not None
    assert complete.terminal_result.winner == repeated_refs[0]
    assert complete.terminal_result.winner_score == 60.0


def test_minimal_production_flow_accounts_rows_and_returns_exact_winner(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "miprov2-flow.sqlite"))
    resolver = _ExactEvalConfigResolver()
    driver, state = _start()
    transport = _transport()
    adapter = _adapter(
        store=store,
        resolver=resolver,
        executor=_RecordingEffectExecutor(_EffectJournal()),
        transport=transport,
        driver=driver,
    )

    for ordinal in range(2):
        proposal = adapter.invoke(_request(state, ordinal=ordinal), ())
        assert proposal.budget_delta.consumed == {"proposal_calls": 1}
        state = _state(proposal)

    baseline_binding = adapter.invoke(_request(state, ordinal=2), ())
    assert baseline_binding.evaluation_intents == ()
    state = _state(baseline_binding)
    baseline_output = adapter.invoke(_request(state, ordinal=3), ())
    baseline_intent = baseline_output.evaluation_intents[0]
    assert baseline_intent.candidate == state.control.base_candidate
    assert baseline_output.budget_delta.consumed == {
        "evaluations": 1,
        "task_rows": 2,
    }
    state = adapter.fold_resolution(
        _state(baseline_output),
        _resolution(store, baseline_intent, score=0.1),
    )

    sample_binding = adapter.invoke(_request(state, ordinal=4), ())
    state = _state(sample_binding)
    sample_output = adapter.invoke(_request(state, ordinal=5), ())
    sample_intent = sample_output.evaluation_intents[0]
    assert sample_intent.candidate != state.control.base_candidate
    assert sample_output.budget_delta.consumed == {
        "evaluations": 1,
        "task_rows": 2,
    }
    state = adapter.fold_resolution(
        _state(sample_output),
        _resolution(store, sample_intent, score=0.9),
    )

    complete = adapter.invoke(_request(state, ordinal=6), ())
    assert complete.accepted_candidates == (sample_intent.candidate.record,)
    assert (
        candidate_reference(complete.accepted_candidates[0])
        == sample_intent.candidate
    )
    final_state = _state(complete)
    assert final_state.accepted_candidate_ref == sample_intent.candidate
    assert final_state.fully_evaluated_candidates == (
        state.control.base_candidate,
        sample_intent.candidate,
    )
    assert final_state.effect_counts == {
        "bootstrap_rollouts": 0,
        "proposal_calls": 2,
        "evaluations": 2,
        "task_rows": 4,
    }
    assert [request.purpose for request in resolver.calls] == [
        "baseline",
        "sample",
    ]


@pytest.mark.parametrize(
    "forbidden",
    [
        "_seeded_tpe_choice",
        "_materialize_demonstrations",
        "DemoPair",
        "itertools.combinations",
        "combination_candidates",
        '"promotion": "noop"',
        "retry_until_distinct",
    ],
)
def test_adapter_source_has_no_approximation_markers(forbidden: str) -> None:
    root = Path(__file__).parents[2] / "src" / "whetstone" / "optimization"
    assert forbidden not in (root / "miprov2.py").read_text()
    assert forbidden not in (root / "miprov2_runtime.py").read_text()
