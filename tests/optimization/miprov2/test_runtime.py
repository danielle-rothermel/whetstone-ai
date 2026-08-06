from __future__ import annotations

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
from whetstone.optimization.miprov2.evidence import (
    Miprov2IntentContext,
    Miprov2ResolvedEvaluation,
    Miprov2RowAccounting,
)
from whetstone.optimization.miprov2.proposal import Miprov2ProposalResponse
from whetstone.optimization.miprov2.rng import (
    Miprov2RngCheckpoint,
)
from whetstone.optimization.miprov2.runtime import Miprov2Driver, Miprov2State
from whetstone.optimization.miprov2.study import (
    Miprov2EvaluationObservation,
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
