from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest
from pydantic import ValidationError

from tests.optimization.miprov2.support import (
    make_minimal_miprov2_runtime,
    resolve_miprov2_eval_config_binding,
)
from whetstone.core.identity import TypedRef
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.schema_names import EVALUATION_EVIDENCE_SCHEMA
from whetstone.experiment.binding import EvaluationBinding
from whetstone.experiment.candidate import candidate_reference
from whetstone.experiment.reward import (
    apply_reward_policy,
    reward_reference,
)
from whetstone.optimization.contracts import optimization_run_reference
from whetstone.optimization.miprov2.eval_config import (
    Miprov2EvaluationExecutionPolicy,
)
from whetstone.optimization.miprov2.evidence import (
    Miprov2IntentContext,
    Miprov2ResolvedEvaluation,
    Miprov2RowAccounting,
)
from whetstone.optimization.miprov2.proposal import Miprov2ProposalResponse
from whetstone.optimization.miprov2.rng import (
    Miprov2RngCheckpoint,
)
from whetstone.optimization.miprov2.runtime import (
    Miprov2Driver,
    Miprov2EffectBudget,
    Miprov2EvaluationSpec,
    Miprov2State,
    _input_data_identity,
)
from whetstone.optimization.miprov2.study import (
    Miprov2EvaluationObservation,
)


def _runtime(**kwargs):
    return make_minimal_miprov2_runtime(**kwargs)


def _assert_reparseable_state(state: Miprov2State) -> Miprov2State:
    assert Miprov2State.model_validate_json(state.model_dump_json()) == state
    return state


def test_runtime_state_and_evaluation_identity_payloads_are_pinned() -> None:
    _driver, state = _runtime()
    spec = Miprov2EvaluationSpec(
        run_id=state.run_id,
        ordinal=0,
        purpose="miprov2_baseline",
        candidate=state.control.base_candidate.record,
        categorical_combination_identity_hash="a" * 64,
        task_batch_identities=state.control.valset_task_identities,
        execution_policy=Miprov2EvaluationExecutionPolicy(
            num_threads=state.control.num_threads,
            max_errors=state.control.max_errors,
            provide_traceback=state.control.provide_traceback,
            task_model_identity_hash=state.control.task_model_identity_hash,
            provider_execution_policy_hash=(
                state.control.provider_execution_policy_hash
            ),
        ),
    )

    assert spec.identity_payload() == {
        "run_id": state.run_id,
        "ordinal": 0,
        "purpose": "miprov2_baseline",
        "candidate": state.control.base_candidate.record.model_dump(
            mode="json"
        ),
        "categorical_combination_identity_hash": "a" * 64,
        "task_batch_identities": list(state.control.valset_task_identities),
        "execution_policy": spec.execution_policy.model_dump(mode="json"),
        "suggestion": None,
        "promotion_candidate": None,
        "candidate_assembly": None,
    }
    assert spec.identity_hash() == (
        "4405d08efd51adbc3a0fd8f642aae1ff67f513aa8f93f2621529181b1a513ee7"
    )
    assert tuple(state.identity_payload()) == (
        "schema_name",
        "schema_version",
        "run_id",
        "run",
        "control",
        "bindings",
        "rng_checkpoint",
        "labeled_trainset",
        "proposal_components",
        "proposal_trainset",
        "component_field_order",
        "input_data_identity_hash",
        "budget",
        "phase",
        "bootstrap_plans",
        "bootstrap_plan_index",
        "bootstrap_state",
        "demo_candidates",
        "bootstrap_evidence",
        "proposal_state",
        "instruction_pools",
        "study_demo_candidates",
        "study_transcript",
        "pending_bootstrap",
        "pending_bootstrap_candidate",
        "pending_proposal",
        "pending_evaluation_spec",
        "pending_eval_binding_request",
        "resolved_eval_binding",
        "pending_evaluation",
        "pending_sample",
        "fully_evaluated_candidates",
        "accepted_candidate",
        "accepted_candidate_ref",
        "terminal_result",
        "completed_effects",
        "failure",
    )
    assert state.identity_payload() == state.model_dump(mode="json")
    assert state.identity_hash() == (
        "ddabdb99f2094e20f51d5d6f1f9e62abc1defbc26565835794323083a97d0048"
    )


def test_runtime_component_order_is_refrozen_by_copy_and_construct() -> None:
    _driver, state = _runtime()
    source = {state.control.component_ids[0]: ["query", "answer"]}
    copied = state.model_copy(update={"component_field_order": source})
    source[state.control.component_ids[0]].append("late")
    assert copied.component_field_order.to_json() == {
        state.control.component_ids[0]: ["query", "answer"]
    }

    constructed_source = {state.control.component_ids[0]: ["query", "answer"]}
    constructed = Miprov2State.model_construct(
        **{
            field: getattr(state, field)
            for field in Miprov2State.model_fields
            if field != "component_field_order"
        },
        component_field_order=constructed_source,
    )
    constructed_source[state.control.component_ids[0]].append("late")
    assert constructed.component_field_order == state.component_field_order


@pytest.mark.parametrize(
    "fields",
    (
        pytest.param(("query", "answer"), id="tuple-fields"),
        pytest.param(["query", "answer"], id="list-fields"),
    ),
)
def test_runtime_mapping_component_order_preserves_constructor_boundary(
    fields: tuple[str, ...] | list[str],
) -> None:
    _driver, state = _runtime()
    component_id = state.control.component_ids[0]
    source = MappingProxyType({component_id: fields})
    state_values: dict[str, Any] = {
        field: getattr(state, field) for field in Miprov2State.model_fields
    }
    state_values["component_field_order"] = source

    normal = Miprov2State(**state_values)
    copied = state.model_copy(update={"component_field_order": source})
    constructed = Miprov2State.model_construct(**state_values)

    if isinstance(fields, list):
        fields.append("late")
    for detached in (normal, copied, constructed):
        assert detached.component_field_order.to_json() == {
            component_id: ["query", "answer"]
        }
        assert detached.identity_hash() == state.identity_hash()
        assert (
            Miprov2State.model_validate_json(detached.model_dump_json())
            == detached
        )


@pytest.mark.parametrize(
    "source",
    (
        pytest.param(
            MappingProxyType({1: ("query", "answer")}),
            id="non-string-key",
        ),
        pytest.param(
            MappingProxyType({"": ("query", "answer")}),
            id="empty-component-id",
        ),
        pytest.param(
            MappingProxyType({"generate": ("query", object())}),
            id="non-json-value",
        ),
    ),
)
def test_runtime_mapping_component_order_stays_strict(source: object) -> None:
    _driver, state = _runtime()
    state_values: dict[str, Any] = {
        field: getattr(state, field) for field in Miprov2State.model_fields
    }
    state_values["component_field_order"] = source

    with pytest.raises(ValidationError):
        Miprov2State(**state_values)
    with pytest.raises(ValidationError):
        state.model_copy(update={"component_field_order": source})
    with pytest.raises(ValidationError):
        Miprov2State.model_construct(**state_values)


@pytest.mark.parametrize(
    "fields",
    (
        pytest.param(
            "query",
            id="scalar-string-fields",
        ),
        pytest.param(
            b"query",
            id="scalar-bytes-fields",
        ),
        pytest.param(
            (1,),
            id="numeric-field",
        ),
        pytest.param(
            (True,),
            id="boolean-field",
        ),
        pytest.param(
            ("query", 2),
            id="mixed-fields",
        ),
        pytest.param(
            (),
            id="empty-fields",
        ),
        pytest.param(
            ("", "answer"),
            id="empty-field",
        ),
        pytest.param(
            ("query", "query"),
            id="duplicate-fields",
        ),
    ),
)
def test_runtime_component_order_contract_is_consistent(
    fields: object,
) -> None:
    _driver, state = _runtime()
    source = MappingProxyType({state.control.component_ids[0]: fields})
    state_values: dict[str, Any] = {
        field: getattr(state, field) for field in Miprov2State.model_fields
    }
    state_values["component_field_order"] = source
    state_values["input_data_identity_hash"] = _input_data_identity(
        control=state.control,
        labeled_trainset=state.labeled_trainset,
        proposal_components=state.proposal_components,
        proposal_trainset=state.proposal_trainset,
        component_field_order=source,
    )

    with pytest.raises(ValidationError):
        Miprov2State(**state_values)
    with pytest.raises(ValidationError):
        state.model_copy(update={"component_field_order": source})
    with pytest.raises(ValidationError):
        Miprov2State.model_construct(**state_values)


def test_runtime_copy_rejects_input_change_with_stale_identity_hash() -> None:
    _driver, state = _runtime()

    with pytest.raises(ValidationError, match="exact content binding"):
        state.model_copy(
            update={
                "component_field_order": {
                    state.control.component_ids[0]: ("query",)
                }
            }
        )


def test_runtime_copy_accepts_coherent_input_identity_update() -> None:
    _driver, state = _runtime()
    source = MappingProxyType({state.control.component_ids[0]: ["query"]})
    input_data_identity_hash = _input_data_identity(
        control=state.control,
        labeled_trainset=state.labeled_trainset,
        proposal_components=state.proposal_components,
        proposal_trainset=state.proposal_trainset,
        component_field_order=source,
    )

    copied = state.model_copy(
        update={
            "component_field_order": source,
            "input_data_identity_hash": input_data_identity_hash,
        }
    )
    source[state.control.component_ids[0]].append("late")

    assert copied.component_field_order.to_json() == {
        state.control.component_ids[0]: ["query"]
    }
    assert Miprov2State.model_validate_json(copied.model_dump_json()) == copied


def test_runtime_tuple_fields_are_refrozen_by_copy_and_construct() -> None:
    driver, state = _runtime()
    state = driver.plan(_fold_all_proposals(driver, state)).state

    copy_pools = [list(pool) for pool in state.instruction_pools]
    copy_effects = list(state.completed_effects)
    copy_trainset = list(state.proposal_trainset)
    copied = state.model_copy(
        update={
            "instruction_pools": copy_pools,
            "completed_effects": copy_effects,
            "proposal_trainset": copy_trainset,
        }
    )
    copy_pools[0].append("late {query}")
    copy_effects.clear()
    copy_trainset.clear()
    assert copied.instruction_pools == state.instruction_pools
    assert copied.completed_effects == state.completed_effects
    assert copied.proposal_trainset == state.proposal_trainset

    construct_pools = [list(pool) for pool in state.instruction_pools]
    construct_effects = list(state.completed_effects)
    construct_trainset = list(state.proposal_trainset)
    construct_values: dict[str, Any] = {
        field: getattr(state, field) for field in Miprov2State.model_fields
    }
    construct_values.update(
        instruction_pools=construct_pools,
        completed_effects=construct_effects,
        proposal_trainset=construct_trainset,
    )
    constructed = Miprov2State.model_construct(**construct_values)
    construct_pools[0].append("late {query}")
    construct_effects.clear()
    construct_trainset.clear()
    assert constructed.instruction_pools == state.instruction_pools
    assert constructed.completed_effects == state.completed_effects
    assert constructed.proposal_trainset == state.proposal_trainset


def test_runtime_nested_models_are_detached_by_copy_and_construct() -> None:
    _driver, state = _runtime()
    source = state.proposal_trainset[0]
    replacement = [source, *state.proposal_trainset[1:]]

    copied = state.model_copy(update={"proposal_trainset": replacement})
    construct_values: dict[str, Any] = {
        field: getattr(state, field) for field in Miprov2State.model_fields
    }
    construct_values["proposal_trainset"] = replacement
    constructed = Miprov2State.model_construct(**construct_values)
    copied_hash = copied.identity_hash()
    constructed_hash = constructed.identity_hash()

    assert copied.proposal_trainset[0] is not source
    assert constructed.proposal_trainset[0] is not source
    assert copied.proposal_trainset[0] is not constructed.proposal_trainset[0]
    assert copied.control is not state.control
    assert constructed.control is not state.control
    assert copied.budget is not state.budget
    assert constructed.budget is not state.budget
    assert copied.control.program_layout is not state.control.program_layout
    assert (
        copied.control.program_layout.component_specs[0]
        is not state.control.program_layout.component_specs[0]
    )

    replacement.clear()
    object.__setattr__(source, "rendered_record", "tampered after copy")

    assert copied.identity_hash() == copied_hash
    assert constructed.identity_hash() == constructed_hash
    assert Miprov2State.model_validate_json(copied.model_dump_json()) == copied
    assert (
        Miprov2State.model_validate_json(constructed.model_dump_json())
        == constructed
    )


@pytest.mark.parametrize(
    "invalid_task_rows",
    (
        pytest.param(True, id="bool"),
        pytest.param(1.5, id="float"),
        pytest.param("1", id="string"),
    ),
)
def test_runtime_nested_strict_int_cannot_bypass_state_revalidation(
    invalid_task_rows: object,
) -> None:
    _driver, state = _runtime()
    forged_budget = Miprov2EffectBudget.model_construct(
        bootstrap_rollouts=state.budget.bootstrap_rollouts,
        proposal_calls=state.budget.proposal_calls,
        evaluations=state.budget.evaluations,
        task_rows=invalid_task_rows,
    )
    state_values: dict[str, Any] = {
        field: getattr(state, field) for field in Miprov2State.model_fields
    }
    state_values["budget"] = forged_budget

    with pytest.raises(ValidationError, match="valid integer"):
        state.model_copy(update={"budget": forged_budget})
    with pytest.raises(ValidationError, match="valid integer"):
        Miprov2State.model_construct(**state_values)


def test_runtime_rejects_same_hash_foreign_control_address() -> None:
    _driver, state = _runtime()
    foreign_optimizer_config = state.control.reference().model_copy(
        update={
            "record_ref": TypedRef(
                schema_name="whetstone.miprov2_optimizer_config",
                content_hash="9" * 64,
            )
        }
    )
    foreign_run = optimization_run_reference(
        state.run.record.model_copy(
            update={"optimizer_config": foreign_optimizer_config}
        )
    )
    payload = state.model_dump(mode="json")
    payload["run"] = foreign_run.model_dump(mode="json")

    with pytest.raises(ValidationError, match="resolved control"):
        Miprov2State.model_validate(payload)


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
    driver, state = make_minimal_miprov2_runtime(track_stats=track_stats)
    state = _fold_all_proposals(driver, state)

    baseline_binding = driver.plan(state)
    assert baseline_binding.eval_config_binding is not None
    state = driver.fold_eval_config_binding(
        baseline_binding.state,
        resolve_miprov2_eval_config_binding(
            baseline_binding.eval_config_binding
        ),
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
        resolve_miprov2_eval_config_binding(
            sample_binding.eval_config_binding
        ),
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
    _, state = make_minimal_miprov2_runtime()

    assert Miprov2State.model_validate_json(state.model_dump_json()) == state
    payload = state.model_dump(mode="json")
    payload["proposal_trainset"][0]["rendered_record"] = "foreign"
    with pytest.raises(ValidationError, match="exact content binding"):
        Miprov2State.model_validate(payload)


def test_runtime_rejects_noncanonical_rng_and_bootstrap_cursor() -> None:
    driver, state = make_minimal_miprov2_runtime()
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
    driver, state = make_minimal_miprov2_runtime()
    first = driver.plan(state)
    assert first.proposal_request is not None
    restarted = Miprov2State.model_validate_json(first.state.model_dump_json())

    replay = driver.plan(restarted)

    assert replay.kind == "proposal_model"
    assert replay.proposal_request == first.proposal_request


def test_evaluation_binding_binds_exact_tasks_and_execution_policy() -> None:
    driver, state = make_minimal_miprov2_runtime()
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
    driver, state = make_minimal_miprov2_runtime(proposal_calls=1)
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
