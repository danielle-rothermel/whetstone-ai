from __future__ import annotations

import json

import pytest

from whetstone.envs.ed1 import (
    DECODER_TEMPLATE,
    ENCODER_BODY_A,
    ENCODER_FRAME,
    ENCODER_FRAME_NO_BUDGET,
    ed1_initial_candidate,
)
from whetstone.optimization.copro.adapter import CoproDriver
from whetstone.optimization.copro.ed1_contract import (
    ed1_copro_proposal_contract,
)
from whetstone.optimization.copro.ed1_dry_run import (
    DummyCoproProposerConfig,
    DummyCoproProposerTransport,
    Ed1CoproDryRunTranscript,
    Ed1CoproPreviewTask,
    Ed1CoproProposalRejectionKind,
    Ed1CoproSweepRanges,
    attempt_ed1_copro_round,
    run_ed1_copro_dry_run,
)
from whetstone.optimization.proposal.mutation import MUTATION_FIELD


def _task() -> Ed1CoproPreviewTask:
    return Ed1CoproPreviewTask(
        task_id="HumanEval/0",
        input_code="def add(a, b):\n    return a + b",
    )


def _proposer() -> DummyCoproProposerConfig:
    return DummyCoproProposerConfig(
        bodies=(
            "Describe the function's behavior for a Python implementer",
            "Summarize the input, output, and required behavior",
            "Explain how to reconstruct an equivalent Python function",
        )
    )


def test_sweep_ranges_expand_in_declared_cartesian_order() -> None:
    sweep = Ed1CoproSweepRanges(
        budget_ratios=(None, 0.5),
        breadths=(3, 4),
        depths=(1,),
    )

    points = sweep.expand()

    assert [point.sweep_ordinal for point in points] == [0, 1, 2, 3]
    assert [(point.budget_ratio, point.copro.breadth) for point in points] == [
        (None, 3),
        (None, 4),
        (0.5, 3),
        (0.5, 4),
    ]


def test_dry_run_logs_baseline_fill_and_body_only_seed_mutations() -> None:
    sweep = Ed1CoproSweepRanges(
        budget_ratios=(None, 0.5),
        breadths=(3,),
        depths=(1,),
    )
    logged: list[str] = []

    transcript = run_ed1_copro_dry_run(
        sweep=sweep,
        preview_task=_task(),
        dummy_proposer=_proposer(),
        log=logged.append,
    )

    assert len(transcript.points) == 2
    assert logged == [transcript.model_dump_json(indent=2)]
    assert (
        json.loads(logged[0])["points"][0]["baseline_prompt"]["body_literal"]
        == ENCODER_BODY_A
    )

    no_budget, budgeted = transcript.points
    assert no_budget.initial_state.completed_rounds == 0
    assert no_budget.initial_state.attempts == ()
    assert no_budget.round_plan.iteration == 0
    assert no_budget.round_plan.include_initial_candidate is True
    assert no_budget.round_plan.proposal_count == 2
    assert no_budget.round_plan.instruction_history == ()
    assert no_budget.baseline_prompt.frame_template == (
        ENCODER_FRAME_NO_BUDGET
    )
    assert no_budget.baseline_prompt.fill.max_budget is None
    assert no_budget.baseline_prompt.rendered_prompt == (
        f"{ENCODER_BODY_A}\n```python\n{_task().input_code}\n```"
    )
    no_budget_call = no_budget.proposal_call
    assert no_budget_call.requested_count == 2
    assert no_budget_call.proposer_kind == "dummy"
    assert no_budget_call.request.base_template == ENCODER_BODY_A
    assert no_budget_call.instruction_contract.budget_mode == "unbudgeted"
    assert no_budget_call.instruction_contract.encoder_frame == (
        ENCODER_FRAME_NO_BUDGET
    )
    assert no_budget_call.instruction_contract.decoder_template == (
        DECODER_TEMPLATE
    )
    proposal_prompt = no_budget_call.request.context["proposal_prompt"]
    assert isinstance(proposal_prompt, str)
    assert "Optimization target: encoder_instruction" in proposal_prompt
    assert ENCODER_FRAME_NO_BUDGET in proposal_prompt
    assert DECODER_TEMPLATE in proposal_prompt
    assert "Return only a replacement encoder instruction body" in (
        proposal_prompt
    )
    assert [draft.template for draft in no_budget_call.drafts] == list(
        _proposer().bodies[:2]
    )
    assert all(
        draft.request_evidence["proposal_request_identity_hash"]
        == no_budget_call.request.identity_hash()
        for draft in no_budget_call.drafts
    )

    expected_budget = round(0.5 * len(_task().input_code))
    assert budgeted.baseline_prompt.frame_template == ENCODER_FRAME
    assert budgeted.baseline_prompt.fill.max_budget == expected_budget
    assert (
        f"Use at most {expected_budget} characters."
        in budgeted.baseline_prompt.rendered_prompt
    )
    assert budgeted.proposal_call.instruction_contract.budget_mode == (
        "budgeted"
    )
    assert budgeted.proposal_call.instruction_contract.encoder_frame == (
        ENCODER_FRAME
    )
    assert "# Encode" not in budgeted.baseline_prompt.rendered_prompt

    baseline_ref = no_budget.baseline_candidate.record_ref
    assert [
        mutation.proposed_body for mutation in no_budget.candidate_mutations
    ] == list(_proposer().bodies[:2])
    for mutation in no_budget.candidate_mutations:
        assert mutation.proposal_ordinal >= 0
        assert mutation.previous_body == ENCODER_BODY_A
        assert mutation.candidate.record.base_ref == baseline_ref
        assert mutation.candidate.record.payload[MUTATION_FIELD] == (
            mutation.proposed_body
        )
        assert mutation.prompt.body_literal == mutation.proposed_body
        assert mutation.prompt.fill.body == mutation.proposed_body
        assert mutation.prompt.rendered_prompt.endswith(
            f"```python\n{_task().input_code}\n```"
        )


def test_round_attempt_preserves_valid_and_rejected_proposal_slots() -> None:
    settings = Ed1CoproSweepRanges(
        budget_ratios=(None,),
        breadths=(3,),
        depths=(1,),
    ).expand()[0]
    state = CoproDriver(settings.copro).initial_state(ed1_initial_candidate())

    attempt = attempt_ed1_copro_round(
        settings=settings,
        state=state,
        preview_task=_task(),
        proposer_kind="dummy",
        proposer_config=DummyCoproProposerConfig(
            bodies=("Explain {input_code}", "Describe exact behavior")
        ),
        transport=DummyCoproProposerTransport(),
        request_ordinal=4,
    )

    assert len(attempt.proposal_call.drafts) == 2
    assert [item.proposal_ordinal for item in attempt.candidate_mutations] == [
        1
    ]
    assert len(attempt.rejections) == 1
    assert attempt.rejections[0].proposal_ordinal == 0
    assert attempt.rejections[0].proposed_body == "Explain {input_code}"
    assert attempt.rejections[0].kind is (
        Ed1CoproProposalRejectionKind.REJECTED
    )
    assert "ed1_invalid_encoder_body" in attempt.rejections[0].reason
    assert attempt.terminal_failure is not None


def test_dry_run_transcript_round_trips_as_json() -> None:
    transcript = run_ed1_copro_dry_run(
        sweep=Ed1CoproSweepRanges(
            budget_ratios=(None,),
            breadths=(2,),
            depths=(1,),
        ),
        preview_task=_task(),
        dummy_proposer=_proposer(),
    )

    assert (
        Ed1CoproDryRunTranscript.model_validate_json(
            transcript.model_dump_json()
        )
        == transcript
    )


def test_dummy_output_uses_the_same_post_generation_validation() -> None:
    scripted = DummyCoproProposerConfig(bodies=("Explain {input_code}",))

    with pytest.raises(ValueError, match="ed1_invalid_encoder_body"):
        run_ed1_copro_dry_run(
            sweep=Ed1CoproSweepRanges(
                budget_ratios=(None,),
                breadths=(2,),
                depths=(1,),
            ),
            preview_task=_task(),
            dummy_proposer=scripted,
        )


@pytest.mark.parametrize(
    "instruction",
    (
        "",
        "Explain the code.",
        "Use at most 100 characters",
        "# Encode\nExplain the code",
    ),
)
def test_instruction_contract_rejects_fixed_frame_content(
    instruction: str,
) -> None:
    contract = ed1_copro_proposal_contract(budget_ratio=0.5)

    with pytest.raises(ValueError):
        contract.validate_instruction(instruction)


def test_dummy_proposer_must_fill_largest_seed_round() -> None:
    with pytest.raises(ValueError, match="fewer bodies"):
        run_ed1_copro_dry_run(
            sweep=Ed1CoproSweepRanges(
                budget_ratios=(None,),
                breadths=(4,),
                depths=(1,),
            ),
            preview_task=_task(),
            dummy_proposer=DummyCoproProposerConfig(bodies=("One", "Two")),
        )


def test_dummy_proposal_rejects_punctuated_baseline_body() -> None:
    with pytest.raises(ValueError, match="must omit terminal punctuation"):
        run_ed1_copro_dry_run(
            sweep=Ed1CoproSweepRanges(
                budget_ratios=(None,),
                breadths=(2,),
                depths=(1,),
            ),
            preview_task=_task(),
            dummy_proposer=DummyCoproProposerConfig(bodies=(ENCODER_BODY_A,)),
        )
