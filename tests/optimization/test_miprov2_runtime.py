from __future__ import annotations

import pytest
from dr_code.eval import (
    DefinitionRef,
    EvalConfig,
    RepeatPlan,
    SamplingDefinition,
    TaskSet,
)
from dr_code.eval.identity import SCHEMA_EVAL_CONFIG, identity_hash_for
from pydantic import ValidationError

from tests.optimization.support import FULL_A, FULL_C, FULL_D
from tests.optimization.test_miprov2_control import _configure, _defaults
from whetstone.evaluation_role import EvaluationRole
from whetstone.optimization.identity import TypedRef
from whetstone.optimization.miprov2_demo import LabeledTaskDemo
from whetstone.optimization.miprov2_eval_config import (
    Miprov2EvalConfigBinding,
    Miprov2EvalConfigBindingRequest,
    derive_eval_config_reference,
)
from whetstone.optimization.miprov2_evidence import (
    Miprov2IntentContext,
    Miprov2ResolvedEvaluation,
    Miprov2RowAccounting,
)
from whetstone.optimization.miprov2_proposal import (
    Miprov2DatasetExample,
    Miprov2PromptComponent,
    Miprov2ProposalResponse,
)
from whetstone.optimization.miprov2_rng import (
    Miprov2DurableBindings,
    Miprov2RngCheckpoint,
)
from whetstone.optimization.miprov2_runtime import (
    Miprov2Driver,
    Miprov2EffectBudget,
    Miprov2State,
)
from whetstone.optimization.miprov2_study import (
    EVALUATION_EVIDENCE_SCHEMA,
    Miprov2EvaluationObservation,
)
from whetstone.optimization.reward import (
    apply_reward_policy,
    reward_reference,
)
from whetstone.optimization.schema import (
    EvaluationBinding,
    candidate_reference,
    eval_config_reference,
)


def _canonical_eval_source(sampling_hash: str):
    definition = DefinitionRef(
        definition_id="runtime-eval",
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
    return eval_config_reference(
        EvalConfig(
            definition_ref=definition,
            sampling_config_hash=sampling_hash,
            evaluation_procedure_config_hash=FULL_C,
            aggregation_config_hash=FULL_D,
            config_identity_hash=identity,
        )
    )


def _runtime(*, proposal_calls: int = 2, track_stats: bool = True):
    bootstrap_source = _canonical_eval_source("1" * 64)
    validation_source = _canonical_eval_source("2" * 64)
    defaults = _defaults().model_copy(
        update={
            "bootstrap_eval_source": bootstrap_source,
            "validation_eval_source": validation_source,
            "evaluation_binding": EvaluationBinding(
                schema_version=2,
                eval_config=validation_source,
                role=EvaluationRole.INTERNAL,
                campaign="miprov2-runtime-test",
            ),
        }
    )
    control = _configure(
        max_bootstrapped_demos=0,
        max_labeled_demos=1,
        program_aware_proposer=False,
        data_aware_proposer=False,
        tip_aware_proposer=False,
        fewshot_aware_proposer=False,
        num_trials=1,
        track_stats=track_stats,
        defaults=defaults,
    )
    component_id = control.component_ids[0]
    labeled = tuple(
        LabeledTaskDemo(
            source_task_identity=task_identity,
            inputs_by_component={component_id: {"query": f"q-{index}"}},
            outputs_by_component={component_id: {"answer": f"a-{index}"}},
        )
        for index, task_identity in enumerate(control.trainset_task_identities)
    )
    proposal_trainset = tuple(
        Miprov2DatasetExample(
            task_identity=task_identity,
            rendered_record=f"query=q-{index}; answer=a-{index}",
        )
        for index, task_identity in enumerate(control.trainset_task_identities)
    )
    bindings = Miprov2DurableBindings(
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
    driver = Miprov2Driver()
    state = driver.start(
        run_id="miprov2-runtime-test",
        control=control,
        bindings=bindings,
        labeled_trainset=labeled,
        proposal_components=(
            Miprov2PromptComponent(
                component_id=component_id,
                template=control.base_candidate.record.payload[
                    "user_prompt_template"
                ],
                allowed_placeholders=("query",),
                rendering_rules="Substitute the native query field.",
                example_execution="Answer q-0.",
            ),
        ),
        proposal_trainset=proposal_trainset,
        component_field_order={component_id: ("query", "answer")},
        budget=Miprov2EffectBudget(
            bootstrap_rollouts=0,
            proposal_calls=proposal_calls,
            evaluations=2,
        ),
    )
    return driver, Miprov2State.model_validate_json(state.model_dump_json())


def _resolve_binding(
    request: Miprov2EvalConfigBindingRequest,
) -> Miprov2EvalConfigBinding:
    suffix = request.identity_hash()[:20]
    task_set = TaskSet(
        manifest_id=f"miprov2-runtime-tasks-{suffix}",
        version="1",
        dataset_revision="test",
        task_identities=request.task_batch_identities,
    )
    repeat_plan = RepeatPlan(
        plan_id=f"miprov2-runtime-repeats-{suffix}",
        version="1",
        task_identities=request.task_batch_identities,
        repeat_count=request.repeat_count,
    )
    sampling = SamplingDefinition(
        definition_id="miprov2-runtime-sampling",
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


def _resolved_evaluation(
    state: Miprov2State,
    *,
    score: float,
    nonce: int,
) -> Miprov2ResolvedEvaluation:
    effect = state.pending_evaluation
    binding = state.resolved_eval_binding
    assert effect is not None
    assert binding is not None
    purpose = effect.purpose.removeprefix("miprov2_")
    assert purpose in {"baseline", "sample", "promotion"}
    exact_binding = EvaluationBinding(
        schema_version=2,
        eval_config=effect.eval_config,
        role=EvaluationRole.INTERNAL,
        campaign="miprov2-runtime-test",
    )
    reward = apply_reward_policy(
        state.control.reward_policy,
        aggregates={"score": score / 100},
        evidence_role=EvaluationRole.INTERNAL,
        evidence_refs=(
            TypedRef(
                schema_name="whetstone.rollout_aggregate",
                content_hash=f"{20_000 + nonce:064x}",
            ),
        ),
    )
    observation = Miprov2EvaluationObservation(
        run_id=effect.run_id,
        intent_id=f"miprov2-runtime-intent-{nonce}",
        effect_identity_hash=effect.identity_hash(),
        purpose=effect.purpose,
        candidate=candidate_reference(effect.candidate),
        task_batch_identities=effect.task_batch_identities,
        eval_config=effect.eval_config,
        eval_config_binding=binding,
        evaluation_binding=exact_binding,
        evaluation_result_ref=TypedRef(
            schema_name=EVALUATION_EVIDENCE_SCHEMA,
            content_hash=f"{30_000 + nonce:064x}",
        ),
        expected_reward_policy_hash=state.control.reward_policy_hash,
        reward_ref=reward_reference(reward),
        normalized_score=score,
    )
    context = Miprov2IntentContext(
        control_identity_hash=state.control.identity_hash(),
        run_id=effect.run_id,
        effect_kind=purpose,
        effect_identity_hash=effect.identity_hash(),
        intent_id=observation.intent_id,
        candidate=observation.candidate,
        task_batch_identities=effect.task_batch_identities,
        eval_config=effect.eval_config,
        eval_config_binding=binding,
        evaluation_binding=exact_binding,
        execution_policy=effect.execution_policy,
        reward_policy_hash=state.control.reward_policy_hash,
    )
    row_count = len(effect.task_batch_identities)
    return Miprov2ResolvedEvaluation(
        context=context,
        reward_value=score / 100,
        normalized_score=score,
        evaluation=observation,
        row_accounting=Miprov2RowAccounting(
            planned=row_count,
            present=row_count,
            missing=0,
            failed=0,
            invalid=0,
        ),
    )


def _fold_all_proposals(
    driver: Miprov2Driver,
    state: Miprov2State,
) -> Miprov2State:
    for ordinal in range(2):
        plan = driver.plan(state)
        assert plan.kind == "proposal_model"
        assert plan.proposal_request is not None
        state = driver.fold_proposal(
            plan.state,
            Miprov2ProposalResponse(
                request_identity_hash=plan.proposal_request.identity_hash,
                text=f"Instruction: improved-{ordinal} {{query}}.",
                evidence={"ordinal": ordinal},
            ),
        )
        state = Miprov2State.model_validate_json(state.model_dump_json())
    return state


def _complete_runtime(*, track_stats: bool = True):
    driver, state = _runtime(track_stats=track_stats)
    state = _fold_all_proposals(driver, state)

    baseline_binding = driver.plan(state)
    assert baseline_binding.eval_config_binding is not None
    state = driver.fold_eval_config_binding(
        baseline_binding.state,
        _resolve_binding(baseline_binding.eval_config_binding),
    )
    baseline = driver.plan(state)
    assert baseline.kind == "baseline_evaluation"
    state = driver.fold_evaluation(
        baseline.state,
        _resolved_evaluation(baseline.state, score=10.0, nonce=1),
    )

    sample_binding = driver.plan(state)
    assert sample_binding.eval_config_binding is not None
    state = driver.fold_eval_config_binding(
        sample_binding.state,
        _resolve_binding(sample_binding.eval_config_binding),
    )
    sample = driver.plan(state)
    assert sample.kind == "sample_evaluation"
    assert sample.evaluation is not None
    expected_winner = candidate_reference(sample.evaluation.candidate)
    state = driver.fold_evaluation(
        sample.state,
        _resolved_evaluation(sample.state, score=90.0, nonce=2),
    )
    return driver.plan(state), expected_winner


def test_runtime_input_binding_roundtrips_and_rejects_tamper() -> None:
    _, state = _runtime()

    assert Miprov2State.model_validate_json(state.model_dump_json()) == state
    payload = state.model_dump(mode="json")
    payload["proposal_trainset"][0]["rendered_record"] = "foreign"
    with pytest.raises(ValidationError, match="exact content binding"):
        Miprov2State.model_validate(payload)


def test_runtime_rejects_noncanonical_rng_and_bootstrap_cursor() -> None:
    driver, state = _runtime()
    planned = driver.plan(state).state

    with pytest.raises(ValidationError, match="RNG checkpoint"):
        Miprov2State.model_validate(
            {
                **planned.model_dump(mode="json"),
                "rng_checkpoint": Miprov2RngCheckpoint.seeded(999).model_dump(
                    mode="json"
                ),
            }
        )
    with pytest.raises(ValidationError, match="bootstrap"):
        Miprov2State.model_validate(
            {
                **planned.model_dump(mode="json"),
                "bootstrap_plan_index": 0,
            }
        )


def test_proposal_restart_reconstructs_exact_next_effect() -> None:
    driver, state = _runtime()
    first = driver.plan(state)
    assert first.proposal_request is not None
    restarted = Miprov2State.model_validate_json(first.state.model_dump_json())

    replay = driver.plan(restarted)

    assert replay.kind == "proposal_model"
    assert replay.proposal_request == first.proposal_request


def test_evaluation_binding_binds_exact_tasks_and_execution_policy() -> None:
    driver, state = _runtime()
    state = _fold_all_proposals(driver, state)

    plan = driver.plan(state)

    assert plan.kind == "eval_config_binding"
    assert plan.eval_config_binding is not None
    assert plan.eval_config_binding.purpose == "baseline"
    assert plan.eval_config_binding.task_batch_identities == (
        state.control.valset_task_identities
    )
    assert plan.eval_config_binding.execution_policy.num_threads == (
        state.control.num_threads
    )
    assert plan.eval_config_binding.execution_policy.max_errors == (
        state.control.max_errors
    )
    assert plan.eval_config_binding.execution_policy.provide_traceback == (
        state.control.provide_traceback
    )
    assert (
        plan.eval_config_binding.execution_policy.task_model_identity_hash
        == state.control.task_model_identity_hash
    )
    assert (
        plan.eval_config_binding.execution_policy.provider_execution_policy_hash
        == state.control.provider_execution_policy_hash
    )


def test_proposal_budget_exhaustion_stops_before_next_effect() -> None:
    driver, state = _runtime(proposal_calls=1)
    first = driver.plan(state)
    assert first.proposal_request is not None
    state = driver.fold_proposal(
        first.state,
        Miprov2ProposalResponse(
            request_identity_hash=first.proposal_request.identity_hash,
            text="Instruction: improved {query}.",
            evidence={"ordinal": 0},
        ),
    )

    with pytest.raises(ValueError, match="proposal_calls budget exhausted"):
        driver.plan(state)


def test_minimal_runtime_flow_accounts_rows_and_returns_exact_winner() -> None:
    terminal, expected_winner = _complete_runtime()

    assert terminal.kind == "complete"
    assert terminal.state.accepted_candidate_ref == expected_winner
    assert terminal.state.terminal_result is not None
    assert terminal.state.terminal_result.winner_score == 90.0
    assert terminal.state.effect_counts == {
        "bootstrap_rollouts": 0,
        "proposal_calls": 2,
        "evaluations": 2,
        "task_rows": 6,
    }
    stats = terminal.state.terminal_result.stats
    assert stats is not None
    assert tuple(item.source for item in stats.score_data) == (
        "baseline",
        "sample",
    )
    assert tuple(item.score for item in stats.score_data) == (10.0, 90.0)


def test_terminal_result_gates_detailed_stats_only() -> None:
    terminal, expected_winner = _complete_runtime(track_stats=False)

    assert terminal.kind == "complete"
    assert terminal.state.accepted_candidate_ref == expected_winner
    assert terminal.state.terminal_result is not None
    assert terminal.state.terminal_result.winner_score == 90.0
    assert terminal.state.terminal_result.track_stats is False
    assert terminal.state.terminal_result.stats is None
