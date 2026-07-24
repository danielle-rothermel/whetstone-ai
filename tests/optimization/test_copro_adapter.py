from __future__ import annotations

from copy import deepcopy

import pytest

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization import (
    BudgetState,
    Candidate,
    CoproAdapter,
    EvaluationIntent,
    FakeProposerTransport,
    IdentityOptimizerAdapter,
    IntentOutcome,
    IntentResolution,
    MappingAdapterRegistry,
    OptimizationHarness,
    OptimizationStepRequest,
    OutputContract,
    ProposerConfig,
    ResolutionClass,
    ResolutionDetail,
    Reward,
    RewardInputCitation,
    StepKind,
    StepMode,
    StepStatus,
    candidate_reference,
    eval_config_reference,
    typed_ref_for_record,
)
from whetstone.optimization.copro import (
    HISTORY_PROPOSAL,
    SEED_PROPOSAL,
    CoproAttempt,
    CoproConfig,
    CoproDriver,
    CoproState,
    rank_attempt_history,
)
from whetstone.optimization.copro_control import COPRO_ALGORITHM_VERSION
from whetstone.optimization.proposal_prompts import (
    COPRO_PROPOSAL_PROMPT_SCHEMA_TAG,
)

from .support import (
    FULL_A,
    RecordingEvaluationService,
    eval_config,
    make_store,
)


def _adapter(script: dict[tuple[str, int], tuple[str, ...]]):
    transport = FakeProposerTransport(
        script,
        execution_policy_hash=FULL_A,
        prompt_adapter_identity_hash=FULL_A,
    )
    adapter = CoproAdapter(
        proposer_config=ProposerConfig(
            provider_call_config_ref="provider://proposal",
            provider_call_config_hash=FULL_A,
            temperature=1.4,
        ),
        transport=transport,
    )
    return adapter, transport


def _candidate(cid: str, text: str) -> Candidate:
    return Candidate(
        candidate_id=cid,
        base_ref="route-a",
        payload={"user_prompt_template": text, "fixed": "unchanged"},
    )


def _request(
    *,
    step_index: int = 0,
    candidates: tuple[Candidate, ...] | None = None,
    history: list[dict[str, object]] | None = None,
    breadth: int = 3,
    depth: int = 1,
    proposal_budget: int | None = None,
) -> OptimizationStepRequest:
    return OptimizationStepRequest(
        run_id="copro-run",
        step_id=f"copro-{step_index}",
        optimizer_config_hash=FULL_A,
        adapter_key="copro",
        mode=StepMode.PROPOSAL_ONLY,
        kind=StepKind.PROPOSAL,
        step_index=step_index,
        prior_step_result_ref=(
            None
            if step_index == 0
            else typed_ref_for_record("test.prior", {"step": step_index - 1})
        ),
        candidates=candidates or (_candidate("baseline", "base {input}"),),
        pools={
            "attempt_history": history or [],
            "valid_template_keys": ["input"],
        },
        hyperparameters={
            "breadth": breadth,
            "depth": depth,
            "init_temperature": 1.4,
            "track_stats": False,
            "round_index": step_index,
            "algorithm_version": COPRO_ALGORITHM_VERSION,
            "proposal_prompt_schema_tag": (COPRO_PROPOSAL_PROMPT_SCHEMA_TAG),
            "provider_execution_policy_hash": FULL_A,
            "prompt_adapter_identity_hash": FULL_A,
            "reward_policy_hash": FULL_A,
            "eval_config": eval_config_reference(eval_config()).model_dump(
                mode="json"
            ),
        },
        budget=BudgetState(
            remaining={
                "proposal_calls": (
                    proposal_budget if proposal_budget is not None else breadth
                )
            }
        ),
        output_contract=OutputContract(returned_proposal_count=breadth),
    )


def _entry(
    occurrence_ordinal: int,
    cid: str,
    template: str,
    reward: float,
    *,
    breadth: int = 3,
) -> dict[str, object]:
    record = _candidate(cid, template)
    return CoproAttempt(
        occurrence_ordinal=occurrence_ordinal,
        round_index=occurrence_ordinal // breadth,
        run_id="copro-run",
        step_index=occurrence_ordinal // breadth,
        intent_id=f"intent-{occurrence_ordinal}",
        candidate=candidate_reference(record),
        eval_config=eval_config_reference(eval_config()),
        reward=reward,
        reward_policy_hash=FULL_A,
        evaluation_evidence_refs=(
            typed_ref_for_record(
                "test.copro_evidence",
                {"occurrence_ordinal": occurrence_ordinal},
            ),
        ),
        reward_ref=typed_ref_for_record(
            "test.copro_reward",
            {"occurrence_ordinal": occurrence_ordinal, "reward": reward},
        ),
    ).model_dump(mode="json")


def test_public_hyperparameter_defaults_match_dspy() -> None:
    config = CoproConfig()

    assert config.breadth == 10
    assert config.depth == 3
    assert config.init_temperature == 1.4
    assert config.track_stats is False
    with pytest.raises(ValueError, match="greater than 1"):
        CoproConfig(breadth=1)
    minimum = CoproDriver(CoproConfig(breadth=2, depth=1)).plan_round(
        iteration=0,
        initial_candidates=(_candidate("base", "base {input}"),),
        attempt_history=(),
    )
    assert minimum.proposal_count == 1
    assert minimum.include_initial_candidate is True


def test_seed_round_generates_breadth_minus_one_then_evaluates_original_last(
    tmp_path,
) -> None:
    adapter, transport = _adapter(
        {
            (SEED_PROPOSAL, 0): (
                '"new {input}"',
                "other {input}",
            )
        }
    )
    store = make_store(tmp_path)
    service = RecordingEvaluationService(store)
    harness = OptimizationHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry(
            {
                "identity": IdentityOptimizerAdapter(),
                "copro": adapter,
            }
        ),
        evaluation_service=service,
    )

    result, _ = harness.run_step(
        _request(
            candidates=(_candidate("copro:copro-run:0", '"base {input}"'),)
        )
    )

    assert result.status is StepStatus.CONTINUE
    assert transport.calls[0][2] == 2
    proposed_ids = [
        item.record.candidate_id for item in result.proposed_candidates
    ]
    assert proposed_ids == [
        "copro:copro-run:0:generated",
        "copro:copro-run:1",
        "copro:copro-run:0",
    ]
    assert len(set(proposed_ids)) == 3
    assert [
        item.record.payload["user_prompt_template"]
        for item in result.proposed_candidates
    ] == ["new {input}", "other {input}", "base {input}"]
    assert len(result.resolved_intents) == 3
    assert (
        len(
            {
                resolution.intent.intent_id
                for resolution in result.resolved_intents
            }
        )
        == 3
    )
    assert all(
        resolution.intent.candidate.identity_hash
        in resolution.intent.intent_id
        for resolution in result.resolved_intents
    )
    assert len(service.resolved) == 3
    assert result.budget.consumed["proposal_calls"] == 2
    assert all(
        resolution.intent.target_eval_config
        == eval_config_reference(eval_config())
        for resolution in result.resolved_intents
    )

    proposal_request = transport.calls[0][1]
    prompt = str(proposal_request.context["proposal_prompt"])
    assert prompt == (
        "You are an instruction optimizer for large language models. I will "
        "give you an initial prompt template. Your task is to propose an "
        "instruction that will lead a good language model to perform the "
        "task well. Don't be afraid to be creative.\n\n"
        "Initial instruction:\n"
        "base {input}\n\n"
        "Return only the improved instruction."
    )
    assert "placeholder" not in prompt.lower()


def test_history_uses_top_unique_attempts_and_stable_ties() -> None:
    # "a" is first-seen, then replaced in place by its strictly better
    # duplicate. It therefore remains ahead of "b" in the terminal tie.
    history: list[dict[str, object]] = [
        _entry(0, "a-old", "a {input}", 0.7),
        _entry(1, "b", "b {input}", 0.9),
        _entry(2, "c", "c {input}", 0.8),
        _entry(3, "a-new", "a {input}", 0.9),
        _entry(4, "d", "d {input}", 0.95),
        _entry(5, "e", "e {input}", 0.1),
    ]
    original = deepcopy(history)
    adapter, transport = _adapter(
        {
            (HISTORY_PROPOSAL, 2): (
                "x {input}",
                "y {input}",
                "z {input}",
            )
        }
    )

    output = adapter.invoke(
        _request(step_index=2, history=history, breadth=3, depth=3),
        (),
    )

    assert history == original
    typed_history = tuple(
        CoproAttempt.model_validate(item) for item in history
    )
    assert [
        item.candidate_id for item in rank_attempt_history(typed_history)
    ] == ["d", "a-new", "b", "c", "e"]
    request = transport.calls[0][1]
    assert transport.calls[0][2] == 3
    assert [
        item["candidate_id"] for item in request.context["prompt_history"]
    ] == ["b", "a-new", "d"]
    prompt = str(request.context["proposal_prompt"])
    assert prompt == (
        "You are an instruction optimizer for large language models. I will "
        "give you some task instructions I've tried, along with their "
        "corresponding validation scores. The instructions are arranged in "
        "increasing order based on their scores, where higher scores indicate "
        "better quality.\n\n"
        "Your task is to propose a new instruction that will lead a good "
        "language model to perform the task even better. Don't be afraid to "
        "be creative.\n\n"
        "Instruction #1: b {input}\n"
        "Resulting Score #1: 0.9\n"
        "Instruction #2: a {input}\n"
        "Resulting Score #2: 0.9\n"
        "Instruction #3: d {input}\n"
        "Resulting Score #3: 0.95\n\n"
        "Return only the improved instruction."
    )
    assert "Base template:" not in prompt
    assert "base {input}" not in prompt
    assert (
        output.state_delta["globally_best_measured"]["candidate"]["record"][
            "candidate_id"
        ]
        == "d"
    )
    assert len(output.evaluation_intents) == 3
    assert output.proposed_status is StepStatus.CONTINUE


def test_duplicates_are_evaluated_before_history_deduplication() -> None:
    adapter, _ = _adapter(
        {
            (SEED_PROPOSAL, 0): (
                "duplicate {input}",
                "duplicate {input}",
            )
        }
    )

    output = adapter.invoke(_request(), ())

    assert len(output.evaluation_intents) == 3
    assert [
        item.payload["user_prompt_template"]
        for item in output.proposed_candidates
    ] == [
        "duplicate {input}",
        "duplicate {input}",
        "base {input}",
    ]


def test_history_uses_all_unique_attempts_below_breadth() -> None:
    driver = CoproDriver(CoproConfig(breadth=4, depth=2))
    attempts = tuple(
        CoproAttempt.model_validate(item)
        for item in (
            _entry(0, "a-old", "a {input}", 0.1, breadth=4),
            _entry(1, "a-best", "a {input}", 0.4, breadth=4),
            _entry(2, "b-best", "b {input}", 0.3, breadth=4),
            _entry(3, "b-old", "b {input}", 0.2, breadth=4),
        )
    )

    plan = driver.plan_round(
        iteration=1,
        initial_candidates=(_candidate("base", "base {input}"),),
        attempt_history=attempts,
    )

    assert [item["candidate_id"] for item in plan.prompt_history] == [
        "b-best",
        "a-best",
    ]


def test_driver_owns_exact_round_counts_terminal_ranking_and_statistics() -> (
    None
):
    config = CoproConfig(breadth=3, depth=2, track_stats=True)
    driver = CoproDriver(config)
    initial = (_candidate("baseline", "base {input}"),)
    first = tuple(
        CoproAttempt.model_validate(item)
        for item in (
            _entry(0, "x", "x {input}", 0.2),
            _entry(1, "y", "y {input}", 0.6),
            _entry(2, "baseline", "base {input}", 0.4),
        )
    )
    second = tuple(
        CoproAttempt.model_validate(item)
        for item in (
            _entry(3, "x2", "x {input}", 0.7),
            _entry(4, "z", "z {input}", 0.9),
            _entry(5, "w", "w {input}", 0.8),
        )
    )

    seed = driver.plan_round(
        iteration=0,
        initial_candidates=initial,
        attempt_history=(),
    )
    history = driver.plan_round(
        iteration=1,
        initial_candidates=initial,
        attempt_history=first,
    )

    assert seed.proposal_count == 2
    assert seed.include_initial_candidate is True
    assert history.proposal_count == 3
    assert history.include_initial_candidate is False
    assert seed.proposal_count + history.proposal_count == 3 * 2 - 1
    assert 3 + 3 == config.breadth * config.depth
    assert [
        item.candidate_id for item in driver.terminal_ranking(first + second)
    ] == ["z", "w", "x2", "y", "baseline"]

    stats = driver.statistics((first, second))
    assert stats.total_calls == 6
    assert stats.results_latest.average[0] == pytest.approx(0.4)
    assert stats.results_latest.std[0] == pytest.approx(0.1632993161855452)
    assert stats.results_best.max[1] == 0.9
    assert stats.results_best.min[1] == 0.4

    state = driver.initial_state(initial[0])
    after_first = driver.fold_round(state, first)
    restored = driver.restore_state(
        initial_candidate=initial[0],
        attempts=first,
    )
    assert restored == after_first
    assert driver.advance(restored) == history
    completed = driver.fold_round(restored, second)
    final = driver.finalize(completed)
    assert final.total_calls == 6
    assert final.statistics == stats
    assert [item.candidate_id for item in final.ranked_attempts] == [
        "z",
        "w",
        "x2",
        "y",
        "baseline",
    ]

    without_stats = CoproDriver(
        CoproConfig(breadth=3, depth=2, track_stats=False)
    ).finalize(completed)
    assert without_stats.total_calls == 6
    assert without_stats.statistics is None
    forged = CoproState(
        initial_candidate=initial[0],
        completed_rounds=2,
        attempts=(),
        total_calls=0,
    )
    with pytest.raises(ValueError, match="occurrence history"):
        driver.finalize(forged)


def test_attempt_folds_only_matching_reward_and_evaluation_provenance() -> (
    None
):
    eval_ref = eval_config_reference(eval_config())
    evidence_ref = typed_ref_for_record("test.evidence", {"value": 1})
    aggregate_ref = typed_ref_for_record("test.aggregate", {"score": 0.75})
    reward = Reward(
        reward_name="quality",
        value=0.75,
        reward_policy_hash=FULL_A,
        evidence_role=EvaluationRole.INTERNAL,
        input_citations=(
            RewardInputCitation(
                name="score",
                value=0.75,
                contributed=0.75,
            ),
        ),
        evidence_ref_content_hash=aggregate_ref.content_hash,
    )
    reward_ref = typed_ref_for_record(
        "whetstone.reward", reward.record_content()
    )
    intent = EvaluationIntent(
        intent_id="copro-run:0:0",
        candidate=candidate_reference(_candidate("a", "a {input}")),
        target_eval_config=eval_ref,
        context_role=EvaluationRole.INTERNAL,
        purpose=SEED_PROPOSAL,
        run_id="copro-run",
        step_index=0,
    )
    resolution = IntentResolution(
        intent=intent,
        outcome=IntentOutcome.COMPLETED,
        detail=ResolutionDetail(
            classification=ResolutionClass.MEASURED,
            message="measured",
        ),
        evaluation_evidence_refs=(evidence_ref,),
        resolved_eval_config=eval_ref,
        reward_ref=reward_ref,
    )

    attempt = CoproAttempt.from_resolution(
        occurrence_ordinal=0,
        round_index=0,
        resolution=resolution,
        reward=reward,
        expected_run_id="copro-run",
        expected_eval_config=eval_ref,
        expected_reward_policy_hash=FULL_A,
    )

    assert attempt.reward == 0.75
    assert attempt.reward_policy_hash == FULL_A
    assert attempt.eval_config == eval_ref
    with pytest.raises(ValueError, match="Reward Policy"):
        CoproAttempt.from_resolution(
            occurrence_ordinal=0,
            round_index=0,
            resolution=resolution,
            reward=reward,
            expected_run_id="copro-run",
            expected_eval_config=eval_ref,
            expected_reward_policy_hash="f" * 64,
        )
    with pytest.raises(ValueError, match="reward_ref"):
        CoproAttempt.from_resolution(
            occurrence_ordinal=0,
            round_index=0,
            resolution=resolution,
            reward=reward.model_copy(update={"value": 0.8}),
            expected_run_id="copro-run",
            expected_eval_config=eval_ref,
            expected_reward_policy_hash=FULL_A,
        )


def test_rejects_multi_seed_and_mismatched_temperature() -> None:
    adapter, _ = _adapter({})
    with pytest.raises(ValueError, match="exactly one initial candidate"):
        adapter.invoke(
            _request(
                candidates=(
                    _candidate("a", "a {input}"),
                    _candidate("b", "b {input}"),
                )
            ),
            (),
        )

    request = _request().model_copy(
        update={
            "hyperparameters": {
                **_request().hyperparameters,
                "init_temperature": 0.7,
            }
        }
    )
    with pytest.raises(ValueError, match="temperature"):
        adapter.invoke(request, ())


def test_transport_identity_must_match_durable_request() -> None:
    adapter, transport = _adapter({})
    request = _request().model_copy(
        update={
            "hyperparameters": {
                **_request().hyperparameters,
                "provider_execution_policy_hash": "f" * 64,
            }
        }
    )

    with pytest.raises(ValueError, match="execution policy"):
        adapter.invoke(request, ())

    assert transport.calls == []


def test_seed_failure_counts_only_generated_slots() -> None:
    adapter, _ = _adapter({(SEED_PROPOSAL, 0): ("", "valid {input}")})

    output = adapter.invoke(_request(), ())

    assert output.proposed_status is StepStatus.FAILED
    assert [item.candidate_id for item in output.proposed_candidates] == [
        "copro:copro-run:1",
        "baseline",
    ]
    assert output.accepted_candidates == ()
    assert output.evaluation_intents == ()
    assert output.budget_delta.consumed["proposal_calls"] == 2
    paid = output.state_delta["proposer_evidence"]
    assert len(paid) == 2
    assert [item["disposition"] for item in paid] == [
        "provider_failed",
        "accepted",
    ]


def test_placeholder_authority_is_required_and_rejections_keep_evidence() -> (
    None
):
    adapter, _ = _adapter(
        {
            (SEED_PROPOSAL, 0): (
                "bad {unknown}",
                "valid {input}",
            )
        }
    )
    request = _request()
    missing_authority = request.model_copy(
        update={"pools": {"attempt_history": []}}
    )
    with pytest.raises(ValueError, match="valid_template_keys authority"):
        adapter.invoke(missing_authority, ())

    output = adapter.invoke(request, ())

    assert output.proposed_status is StepStatus.FAILED
    evidence = output.state_delta["proposer_evidence"]
    assert len(evidence) == 2
    assert evidence[0]["disposition"] == "rejected"
    assert "unknown" in evidence[0]["reason"]
    assert evidence[1]["disposition"] == "accepted"


def test_malformed_placeholder_rejection_keeps_paid_slot_evidence() -> None:
    adapter, _ = _adapter(
        {(SEED_PROPOSAL, 0): ("broken {input", "valid {input}")}
    )

    output = adapter.invoke(_request(), ())

    assert output.proposed_status is StepStatus.FAILED
    evidence = output.state_delta["proposer_evidence"]
    assert evidence[0]["disposition"] == "rejected"
    assert "malformed" in evidence[0]["reason"]
    assert evidence[1]["disposition"] == "accepted"


def test_removed_placeholder_is_rejected_after_generation() -> None:
    adapter, _ = _adapter(
        {(SEED_PROPOSAL, 0): ("no input token", "valid {input}")}
    )

    output = adapter.invoke(_request(), ())

    assert output.proposed_status is StepStatus.FAILED
    evidence = output.state_delta["proposer_evidence"]
    assert evidence[0]["disposition"] == "rejected"
    assert "removes required placeholders" in evidence[0]["reason"]


def test_history_and_fold_validation_fail_closed() -> None:
    adapter, _ = _adapter({})
    malformed = _request(
        step_index=1,
        depth=2,
        history=[{"candidate_id": "legacy-untyped"}],
    )
    with pytest.raises(ValueError):
        adapter.invoke(malformed, ())

    driver = CoproDriver(CoproConfig(breadth=3, depth=1))
    initial = _candidate("baseline", "base {input}")
    attempts = tuple(
        CoproAttempt.model_validate(item)
        for item in (
            _entry(0, "a", "a {input}", 0.1),
            _entry(1, "b", "b {input}", 0.2),
            _entry(2, "c", "c {input}", 0.3),
        )
    )
    gap = attempts[2].model_copy(update={"occurrence_ordinal": 3})
    with pytest.raises(ValueError, match="contiguous"):
        driver.fold_round(
            driver.initial_state(initial),
            (*attempts[:2], gap),
        )

    divergent_record = Candidate(
        candidate_id="c",
        base_ref="route-a",
        payload={
            "user_prompt_template": "c {input}",
            "fixed": "changed",
        },
    )
    divergent = attempts[2].model_copy(
        update={"candidate": candidate_reference(divergent_record)}
    )
    with pytest.raises(ValueError, match="outside"):
        driver.fold_round(
            driver.initial_state(initial),
            (*attempts[:2], divergent),
        )


def test_depth_two_final_adapter_round_still_continues() -> None:
    adapter, _ = _adapter(
        {
            (HISTORY_PROPOSAL, 1): (
                "x {input}",
                "y {input}",
                "z {input}",
            )
        }
    )
    history = [
        _entry(0, "a", "a {input}", 0.1),
        _entry(1, "b", "b {input}", 0.2),
        _entry(2, "c", "c {input}", 0.3),
    ]

    output = adapter.invoke(
        _request(step_index=1, depth=2, history=history),
        (),
    )

    assert output.proposed_status is StepStatus.CONTINUE
    assert [item.candidate_id for item in output.proposed_candidates] == [
        "copro:copro-run:3",
        "copro:copro-run:4",
        "copro:copro-run:5",
    ]


def test_registry_key_and_mode_conform() -> None:
    adapter, _ = _adapter({})
    registry = MappingAdapterRegistry({"copro": adapter})

    assert registry.resolve("copro") is adapter
    assert adapter.key == "copro"
    assert adapter.mode is StepMode.PROPOSAL_ONLY
