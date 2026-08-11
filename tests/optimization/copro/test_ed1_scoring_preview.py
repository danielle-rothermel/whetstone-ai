from __future__ import annotations

from collections.abc import Sequence

import pytest
from dr_store import MemoryBackend, ObjectStore

from tests.envs.support import execution_policy, synthetic_ed1_tasks
from whetstone.envs.ed1_runtime import Ed1RuntimeProbe
from whetstone.envs.ed1_scoring import CodeScore, CodeScoringInput
from whetstone.envs.encdec_rollout import build_encoder_provider_call_config
from whetstone.experiment.task_selection import (
    TaskRoleSelection,
    TaskSplitRole,
)
from whetstone.optimization.copro.adapter import HISTORY_PROPOSAL
from whetstone.optimization.copro.ed1_dry_run import (
    DummyCoproProposerConfig,
    DummyCoproProposerTransport,
    Ed1CoproRoundAttempt,
    Ed1CoproSweepRanges,
)
from whetstone.optimization.copro.ed1_scoring_preview import (
    Ed1CoproCandidateProgress,
    Ed1CoproRoundFailure,
    Ed1ScoringRuntimeSummary,
    run_ed1_copro_scoring_preview,
)
from whetstone.optimization.copro.ed1_scoring_preview_worker import (
    DUMMY_ALTERNATE_PASSING_BODY,
    DUMMY_FAILING_BODY,
)
from whetstone.optimization.copro.ed1_task_model import (
    Ed1TaskModelConfig,
    Ed1TaskModelKind,
)


def _score(
    inputs: Sequence[CodeScoringInput],
    *,
    max_wall_seconds: float | None = None,
) -> tuple[CodeScore, ...]:
    del max_wall_seconds
    return tuple(
        CodeScore(
            passed="return None" not in item.raw_submission,
            infrastructure_unknown=False,
            outcome=(
                "passed"
                if "return None" not in item.raw_submission
                else "tests_failed"
            ),
        )
        for item in inputs
    )


def _task_model() -> Ed1TaskModelConfig:
    return Ed1TaskModelConfig(
        kind=Ed1TaskModelKind.DUMMY,
        provider_call_config=build_encoder_provider_call_config(
            "test/task-model"
        ),
        execution_policy=execution_policy(),
    )


def test_scoring_preview_runs_real_engine_and_folds_two_rounds() -> None:
    runtime = Ed1ScoringRuntimeSummary(
        evaluation_python="/copied/python",
        dr_code_version="0.1.5",
        runtime_identity_hash="a" * 64,
        probe=Ed1RuntimeProbe(
            implementation="CPython",
            numpy_version="2.0.0",
            python_executable="/copied/python",
            python_version="3.13.0",
        ),
    )
    proposal_attempts: list[Ed1CoproRoundAttempt] = []
    candidate_progress: list[Ed1CoproCandidateProgress] = []
    transcript = run_ed1_copro_scoring_preview(
        store=ObjectStore(MemoryBackend()),
        tasks=synthetic_ed1_tasks(1),
        sweep=Ed1CoproSweepRanges(
            budget_ratios=(None,),
            breadths=(3,),
            depths=(2,),
        ),
        proposer_kind="scripted_test",
        proposer_config=DummyCoproProposerConfig(
            bodies=(
                "Describe exact behavior for faithful reconstruction",
                DUMMY_FAILING_BODY,
                DUMMY_ALTERNATE_PASSING_BODY,
            )
        ),
        proposer_transport=DummyCoproProposerTransport(),
        task_model=_task_model(),
        task_selection=TaskRoleSelection(
            manifest_content_hash="b" * 64,
            pool_key="ed1",
            role=TaskSplitRole.TRAIN,
            task_ids=("Synthetic/0",),
        ),
        batch_scorer=_score,
        runtime=runtime,
        concurrency=4,
        proposal_observer=proposal_attempts.append,
        candidate_observer=candidate_progress.append,
    )

    assert transcript.preflight.passed is True
    assert transcript.task_model.kind is Ed1TaskModelKind.DUMMY
    assert transcript.concurrency == 4
    assert transcript.task_selection is not None
    assert transcript.task_selection.role is TaskSplitRole.TRAIN
    point = transcript.points[0]
    assert len(point.rounds) == 2
    assert len(proposal_attempts) == 2
    assert all(attempt.succeeded for attempt in proposal_attempts)
    assert all(
        attempt.proposal_call.proposer_kind == "scripted_test"
        for attempt in proposal_attempts
    )
    evaluation_counts = [
        len(round_record.evaluations) for round_record in point.rounds
    ]
    assert evaluation_counts == [3, 3]
    assert point.rounds[1].preview.round_plan.proposal_mode == HISTORY_PROPOSAL
    assert len(point.rounds[1].preview.round_plan.instruction_history) == 3
    assert point.finalization.total_calls == 6
    assert len(candidate_progress) == 12
    assert [item.result is not None for item in candidate_progress] == [
        False,
        True,
    ] * 6
    assert all(item.candidate_count == 3 for item in candidate_progress)
    assert point.finalization.ranked_attempts[0].reward > 0.9
    failed = [
        item
        for round_record in point.rounds
        for item in round_record.evaluations
        if item.attempt.instruction == DUMMY_FAILING_BODY
    ]
    assert failed
    assert all(item.primary_value == 0.0 for item in failed)
    assert all(item.attempt.reward == 0.0 for item in failed)
    first = point.rounds[0].evaluations[0]
    steps = first.component_traces.rows[0].executed_component_trace
    assert len(steps.executed_component_steps) == 2
    assert first.evidence.evaluation_binding == point.evaluation_binding


def test_scoring_preview_observes_rejected_round_before_failure() -> None:
    runtime = Ed1ScoringRuntimeSummary(
        evaluation_python="/copied/python",
        dr_code_version="0.1.5",
        runtime_identity_hash="a" * 64,
        probe=Ed1RuntimeProbe(
            implementation="CPython",
            numpy_version="2.0.0",
            python_executable="/copied/python",
            python_version="3.13.0",
        ),
    )
    observed: list[Ed1CoproRoundAttempt] = []

    with pytest.raises(Ed1CoproRoundFailure) as error:
        run_ed1_copro_scoring_preview(
            store=ObjectStore(MemoryBackend()),
            tasks=synthetic_ed1_tasks(1),
            sweep=Ed1CoproSweepRanges(
                budget_ratios=(None,),
                breadths=(2,),
                depths=(1,),
            ),
            proposer_kind="dummy",
            proposer_config=DummyCoproProposerConfig(
                bodies=("Explain {input_code}",)
            ),
            proposer_transport=DummyCoproProposerTransport(),
            task_model=_task_model(),
            batch_scorer=_score,
            runtime=runtime,
            proposal_observer=observed.append,
        )

    assert observed == [error.value.attempt]
    assert observed[0].terminal_failure is not None
    assert observed[0].rejections[0].proposed_body == "Explain {input_code}"
