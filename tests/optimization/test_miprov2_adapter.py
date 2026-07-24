from __future__ import annotations

from typing import Any

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.envs.support import (
    FakeTransport,
    constant_reply,
    execution_policy,
)
from whetstone.envs.factory import build_env_experiment
from whetstone.evaluation import (
    EvaluationEngine,
    EvaluationEvidence,
    EvaluationRequest,
    RowAccounting,
)
from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization import (
    BASELINE_FULL,
    BOOTSTRAP,
    COMPLETION,
    MINIBATCH,
    MIPROV2_DEMO_SET_SCHEMA,
    POOL_CONSTRUCTION,
    PROMOTION_FULL,
    AdapterOutput,
    BudgetState,
    Candidate,
    DemoPair,
    DemoSetArtifact,
    DemoSetArtifactRef,
    DemoSetIdentity,
    FakeProposerTransport,
    InstructionIdentity,
    IntentOutcome,
    IntentResolution,
    Miprov2Adapter,
    Miprov2Driver,
    OptimizationStepRequest,
    OutputContract,
    ProposerConfig,
    ResolutionClass,
    ResolutionDetail,
    Reward,
    RewardInputCitation,
    StepKind,
    StepMode,
    TrialCombinationIdentity,
    TypedRef,
    candidate_reference,
    eval_config_reference,
    typed_ref_for_record,
)

from .support import FULL_A, FULL_C, eval_config, make_store


def _candidate(cid: str, template: str) -> Candidate:
    return Candidate(
        candidate_id=cid,
        base_ref="route-a",
        payload={"user_prompt_template": template},
    )


def _adapter(store):
    transport = FakeProposerTransport(
        {("pool_construction", 0): ("pool instruction {input}",)},
        execution_policy_hash=FULL_A,
        prompt_adapter_identity_hash=FULL_A,
    )
    adapter = Miprov2Adapter(
        store=store,
        proposer_config=ProposerConfig(
            provider_call_config_ref="provider://proposal",
            provider_call_config_hash=FULL_A,
            temperature=1.0,
        ),
        transport=transport,
    )
    return adapter, transport


def _hyper() -> dict[str, Any]:
    exact = eval_config_reference(eval_config()).model_dump(mode="json")
    return {
        "bootstrap_eval_config": exact,
        "minibatch_eval_config": exact,
        "full_eval_config": exact,
        "num_demo_set_candidates": 1,
        "num_instruction_candidates": 2,
        "instruction_attempt_cap": 2,
        "num_trials": 1,
        "minibatch_full_eval_steps": 1,
        "returned_proposal_count": 1,
        "seed": 7,
    }


def _request(
    kind: str,
    index: int,
    *,
    state: dict[str, Any],
    candidates: tuple[Candidate, ...],
    output_count: int = 0,
    hyperparameters: dict[str, Any] | None = None,
) -> OptimizationStepRequest:
    return OptimizationStepRequest(
        run_id="mipro-run",
        step_id=f"mipro-{index}",
        optimizer_config_hash=FULL_A,
        adapter_key="miprov2",
        mode=StepMode.PROPOSAL_ONLY,
        kind=StepKind.PROPOSAL,
        kind_label=kind,
        step_index=index,
        prior_step_result_ref=(
            None
            if index == 0
            else typed_ref_for_record("test.prior", {"step": index - 1})
        ),
        candidates=candidates,
        pools=state,
        hyperparameters=hyperparameters or _hyper(),
        budget=BudgetState(
            remaining={"proposal_calls": 2, "search_rollouts": 2}
        ),
        output_contract=OutputContract(returned_proposal_count=output_count),
    )


def _resolution(
    store,
    intent,
    value: float,
    *,
    demo_row_count: int = 1,
) -> IntentResolution:
    reward = Reward(
        reward_name="reward",
        value=value,
        reward_policy_hash=FULL_C,
        evidence_role=EvaluationRole.INTERNAL,
        input_citations=(
            RewardInputCitation(
                name="score",
                value=value,
                contributed=value,
            ),
        ),
    )
    reward_ref, _ = store.put("whetstone.reward", reward.record_content())
    output_record = {
        "candidate_id": intent.candidate.record.candidate_id,
        "outputs": [
            {
                "rendered_prompt": f"observed input {value}-{index}",
                "output_text": f"observed output {value}-{index}",
                "failure_code": None,
            }
            for index in range(demo_row_count)
        ],
    }
    outputs_ref, _ = store.put("whetstone.evaluation_outputs", output_record)
    aggregate_ref, _ = store.put("test.aggregate", {"value": value})
    evidence = EvaluationEvidence(
        candidate=intent.candidate,
        eval_config=intent.target_eval_config,
        graph_hash=FULL_A,
        graph_config_ref=FULL_A,
        evaluation_role=EvaluationRole.INTERNAL,
        evaluation_context_id=intent.intent_id,
        purpose=intent.purpose,
        task_identities=tuple(
            f"task-{index}" for index in range(demo_row_count)
        ),
        repeat_count=1,
        per_task_values=(value,) * demo_row_count,
        per_task_counts=(1,) * demo_row_count,
        row_accounting=RowAccounting(
            planned=demo_row_count,
            present=demo_row_count,
            missing=0,
            failed=0,
            invalid=0,
        ),
        outputs_ref=TypedRef(
            schema_name=outputs_ref.schema,
            content_hash=outputs_ref.content_hash,
        ),
        aggregate_ref=TypedRef(
            schema_name=aggregate_ref.schema,
            content_hash=aggregate_ref.content_hash,
        ),
        aggregate_name="score",
        aggregate_value=value,
        aggregate_status="ok",
        reward_ref=TypedRef(
            schema_name=reward_ref.schema,
            content_hash=reward_ref.content_hash,
        ),
    )
    evidence_ref, _ = store.put(
        "whetstone.evaluation_evidence", evidence.record_content()
    )
    return IntentResolution(
        intent=intent,
        outcome=IntentOutcome.COMPLETED,
        detail=ResolutionDetail(
            classification=ResolutionClass.MEASURED,
            message="measured",
        ),
        evaluation_evidence_refs=(
            TypedRef(
                schema_name=evidence_ref.schema,
                content_hash=evidence_ref.content_hash,
            ),
        ),
        resolved_eval_config=intent.target_eval_config,
        reward_ref=TypedRef(
            schema_name=reward_ref.schema,
            content_hash=reward_ref.content_hash,
        ),
    )


def test_full_production_cadence_and_state_folding(tmp_path) -> None:
    store = make_store(tmp_path)
    driver = Miprov2Driver(store)
    adapter, transport = _adapter(store)
    baseline = _candidate("baseline", "base {input}")
    state: dict[str, Any] = {}

    assert driver.next_plan(state, _hyper()).kind == BOOTSTRAP
    bootstrap = adapter.invoke(
        _request(BOOTSTRAP, 0, state=state, candidates=(baseline,)), ()
    )
    assert len(bootstrap.evaluation_intents) == 1
    state = driver.advance(
        state,
        bootstrap,
        (_resolution(store, bootstrap.evaluation_intents[0], 0.2),),
    )

    assert driver.next_plan(state, _hyper()).kind == POOL_CONSTRUCTION
    pool = adapter.invoke(
        _request(POOL_CONSTRUCTION, 1, state=state, candidates=(baseline,)),
        (),
    )
    state = driver.advance(state, pool, ())
    assert pool.evaluation_intents == ()
    assert state["pool_frozen"] is True
    assert len(state["combination_candidates"]) == 2
    assert "proposal_prompt" in transport.calls[0][1].context

    assert driver.next_plan(state, _hyper()).kind == BASELINE_FULL
    baseline_step = adapter.invoke(
        _request(BASELINE_FULL, 2, state=state, candidates=(baseline,)),
        (),
    )
    state = driver.advance(
        state,
        baseline_step,
        (_resolution(store, baseline_step.evaluation_intents[0], 0.2),),
    )
    default_id = state["default_combination_id"]
    assert (
        baseline_step.evaluation_intents[0].candidate.record.candidate_id
        == default_id
    )
    assert (
        baseline_step.evaluation_intents[0].candidate.record.payload[
            "user_prompt_template"
        ]
        == baseline.payload["user_prompt_template"]
    )
    assert state["study_state"]["study_observations"] == [
        {
            "candidate_id": default_id,
            "value": 0.2,
            "purpose": BASELINE_FULL,
        }
    ]

    assert driver.next_plan(state, _hyper()).kind == MINIBATCH
    trial = adapter.invoke(
        _request(MINIBATCH, 3, state=state, candidates=()), ()
    )
    state = driver.advance(
        state,
        trial,
        (_resolution(store, trial.evaluation_intents[0], 0.7),),
    )

    assert driver.next_plan(state, _hyper()).kind == PROMOTION_FULL
    promotion = adapter.invoke(
        _request(PROMOTION_FULL, 4, state=state, candidates=()), ()
    )
    resolutions = (
        ()
        if not promotion.evaluation_intents
        else (_resolution(store, promotion.evaluation_intents[0], 0.8),)
    )
    state = driver.advance(state, promotion, resolutions)

    plan = driver.next_plan(state, _hyper())
    assert plan.kind == COMPLETION
    assert plan.returned_proposal_count == 1
    completion = adapter.invoke(
        _request(
            COMPLETION,
            5,
            state=state,
            candidates=(),
            output_count=plan.returned_proposal_count,
        ),
        (),
    )
    assert completion.proposed_status.value == "complete"
    assert len(completion.accepted_candidates) == 1
    assert completion.evaluation_intents == ()


def test_default_demo_target_materializes_four_distinct_sets(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    driver = Miprov2Driver(store)
    adapter, _transport = _adapter(store)
    baseline = _candidate("baseline", "base {input}")
    hyperparameters = _hyper()
    hyperparameters.pop("num_demo_set_candidates")
    state: dict[str, Any] = {}

    plan = driver.next_plan(state, hyperparameters)
    assert plan.kind == BOOTSTRAP
    bootstrap = adapter.invoke(
        _request(
            BOOTSTRAP,
            0,
            state=state,
            candidates=(baseline,),
            hyperparameters=hyperparameters,
        ),
        (),
    )
    resolution = _resolution(
        store,
        bootstrap.evaluation_intents[0],
        0.2,
        demo_row_count=3,
    )
    state = driver.advance(state, bootstrap, (resolution,))

    entries = tuple(
        DemoSetArtifactRef.model_validate(item)
        for item in state["demo_set_pool"]
    )
    artifacts = tuple(
        DemoSetArtifact.model_validate(store.get(entry.artifact_ref.reference))
        for entry in entries
    )
    assert len(entries) == 4
    assert len({entry.identity_hash for entry in entries}) == 4
    assert len({entry.artifact_ref for entry in entries}) == 4
    assert [len(artifact.demo_set.pairs) for artifact in artifacts] == [
        0,
        1,
        1,
        1,
    ]
    assert all(
        artifact.source_evidence_ref == resolution.evaluation_evidence_refs[0]
        for artifact in artifacts
    )
    assert driver.next_plan(state, hyperparameters).kind == POOL_CONSTRUCTION


def test_default_demo_target_refuses_a_repeated_no_progress_bootstrap(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    driver = Miprov2Driver(store)
    adapter, _transport = _adapter(store)
    baseline = _candidate("baseline", "base {input}")
    hyperparameters = _hyper()
    hyperparameters.pop("num_demo_set_candidates")
    state: dict[str, Any] = {}

    for index in range(2):
        assert driver.next_plan(state, hyperparameters).kind == BOOTSTRAP
        bootstrap = adapter.invoke(
            _request(
                BOOTSTRAP,
                index,
                state=state,
                candidates=(baseline,),
                hyperparameters=hyperparameters,
            ),
            (),
        )
        state = driver.advance(
            state,
            bootstrap,
            (
                _resolution(
                    store,
                    bootstrap.evaluation_intents[0],
                    0.2,
                ),
            ),
        )

    assert len(state["demo_set_pool"]) == 2
    with pytest.raises(
        ValueError,
        match="cannot materialize the required 4 distinct demo sets",
    ):
        driver.next_plan(state, hyperparameters)


def test_combination_identity_interoperates_with_candidate_pool() -> None:
    instruction = InstructionIdentity(
        instruction_text="pool instruction {input}"
    )
    demo = DemoSetIdentity()
    combination = TrialCombinationIdentity(
        instruction_hash=instruction.identity_hash(),
        demo_set_hash=demo.identity_hash(),
    )
    rebuilt = TrialCombinationIdentity.model_validate(
        combination.model_dump(mode="json")
    )

    assert rebuilt.identity_hash() == combination.identity_hash()


def test_driver_folds_noop_promotion_to_completion(tmp_path) -> None:
    store = make_store(tmp_path)
    driver = Miprov2Driver(store)
    demo = DemoSetIdentity(
        pairs=(
            DemoPair(
                rendered_input="input",
                observed_output="output",
            ),
        )
    )
    artifact = DemoSetArtifact(
        demo_set=demo,
        source_evidence_ref=typed_ref_for_record(
            "test.evidence", {"source": "demo"}
        ),
    )
    artifact_ref, _ = store.put(
        MIPROV2_DEMO_SET_SCHEMA, artifact.model_dump(mode="json")
    )
    pool_entry = DemoSetArtifactRef(
        identity_hash=demo.identity_hash(),
        artifact_ref=TypedRef(
            schema_name=artifact_ref.schema,
            content_hash=artifact_ref.content_hash,
        ),
    )
    state = {
        "pool_frozen": True,
        "demo_set_pool": [pool_entry.model_dump(mode="json")],
        "study_state": {
            "baseline_complete": True,
            "trials_completed": 1,
        },
    }
    advanced = driver.advance(
        state,
        AdapterOutput(
            state_delta={
                "promotion": "noop",
                "reason": "all observed combinations fully evaluated",
            }
        ),
        (),
    )

    assert driver.next_plan(advanced, _hyper()).kind == COMPLETION


def test_minibatch_acquisition_uses_observed_study_scores(tmp_path) -> None:
    store = make_store(tmp_path)
    adapter, _transport = _adapter(store)
    low = _candidate("low", "low {input}")
    high = _candidate("high", "high {input}")
    state = {
        "combination_candidates": [
            low.model_dump(mode="json"),
            high.model_dump(mode="json"),
        ],
        "study_state": {
            "study_observations": [
                {
                    "candidate_id": "low",
                    "value": 0.1,
                    "purpose": MINIBATCH,
                },
                {
                    "candidate_id": "high",
                    "value": 0.9,
                    "purpose": MINIBATCH,
                },
            ],
            "trials_completed": 2,
        },
    }

    output = adapter.invoke(
        _request(MINIBATCH, 3, state=state, candidates=()), ()
    )

    assert output.evaluation_intents[0].candidate == candidate_reference(high)
    assert output.state_delta["acquisition"]["policy"] == (
        "seeded_categorical_tpe/v1"
    )

    restarted = adapter.invoke(
        _request(MINIBATCH, 3, state=state, candidates=()), ()
    )
    assert (
        restarted.evaluation_intents[0].candidate
        == output.evaluation_intents[0].candidate
    )


def test_pool_seeds_base_instructions_and_empty_demo(tmp_path) -> None:
    store = make_store(tmp_path)
    source_ref, _ = store.put("test.bootstrap_evidence", {"complete": True})
    empty_demo = DemoSetIdentity()
    artifact = DemoSetArtifact(
        demo_set=empty_demo,
        source_evidence_ref=TypedRef(
            schema_name=source_ref.schema,
            content_hash=source_ref.content_hash,
        ),
    )
    artifact_ref, _ = store.put(
        MIPROV2_DEMO_SET_SCHEMA, artifact.model_dump(mode="json")
    )
    entry = DemoSetArtifactRef(
        identity_hash=empty_demo.identity_hash(),
        artifact_ref=TypedRef(
            schema_name=artifact_ref.schema,
            content_hash=artifact_ref.content_hash,
        ),
    )
    adapter, transport = _adapter(store)
    base_a = _candidate("a", "template A {input}")
    base_b = _candidate("b", "template B {input}")
    hyperparameters = {
        **_hyper(),
        "num_instruction_candidates": 3,
    }

    output = adapter.invoke(
        _request(
            POOL_CONSTRUCTION,
            1,
            state={"demo_set_pool": [entry.model_dump(mode="json")]},
            candidates=(base_a, base_b),
            hyperparameters=hyperparameters,
        ),
        (),
    )

    assert output.state_delta["instruction_pool"] == [
        "template A {input}",
        "template B {input}",
        "pool instruction {input}",
    ]
    assert len(transport.calls) == 1
    combinations = tuple(
        Candidate.model_validate(item)
        for item in output.state_delta["combination_candidates"]
    )
    assert len(combinations) == 3
    assert {
        candidate.payload["demo_set_identity_hash"]
        for candidate in combinations
    } == {empty_demo.identity_hash()}
    assert output.state_delta["default_combination_id"] == (
        combinations[0].candidate_id
    )
    default_template = combinations[0].payload["user_prompt_template"]
    assert default_template == base_a.payload["user_prompt_template"]


def test_distinct_demo_artifacts_change_execution_and_evidence(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "demo-execution.sqlite"))
    source_ref, _ = store.put("test.bootstrap_evidence", {"complete": True})
    source = TypedRef(
        schema_name=source_ref.schema,
        content_hash=source_ref.content_hash,
    )

    def persist_demo(observed_output: str | None) -> DemoSetArtifactRef:
        demo = DemoSetIdentity(
            pairs=(
                ()
                if observed_output is None
                else (
                    DemoPair(
                        rendered_input="A solved training input",
                        observed_output=observed_output,
                    ),
                )
            )
        )
        artifact = DemoSetArtifact(
            demo_set=demo,
            source_evidence_ref=source,
        )
        reference, _ = store.put(
            MIPROV2_DEMO_SET_SCHEMA, artifact.model_dump(mode="json")
        )
        return DemoSetArtifactRef(
            identity_hash=demo.identity_hash(),
            artifact_ref=TypedRef(
                schema_name=reference.schema,
                content_hash=reference.content_hash,
            ),
        )

    alpha = persist_demo("alpha")
    beta = persist_demo("beta")
    empty = persist_demo(None)
    experiment = build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=1,
    )
    base = experiment.initial_candidate
    instruction = (
        str(base.payload["user_prompt_template"])
        + "\nReturn only the final label."
    )
    adapter = Miprov2Adapter(
        store=store,
        proposer_config=ProposerConfig(
            provider_call_config_ref="provider://proposal",
            provider_call_config_hash=FULL_A,
            temperature=1.0,
        ),
        transport=FakeProposerTransport(
            {("pool_construction", 0): (instruction,)},
            execution_policy_hash=FULL_A,
            prompt_adapter_identity_hash=FULL_A,
        ),
    )
    pool = adapter.invoke(
        _request(
            POOL_CONSTRUCTION,
            1,
            state={
                "demo_set_pool": [
                    empty.model_dump(mode="json"),
                    alpha.model_dump(mode="json"),
                    beta.model_dump(mode="json"),
                ]
            },
            candidates=(base,),
        ),
        (),
    )
    combinations = tuple(
        Candidate.model_validate(item)
        for item in pool.state_delta["combination_candidates"]
    )
    assert len(combinations) == 6
    combinations = tuple(
        candidate
        for candidate in combinations
        if candidate.payload["instruction_template"] == instruction
        and candidate.payload["demo_set_identity_hash"]
        in {alpha.identity_hash, beta.identity_hash}
    )
    assert len(combinations) == 2

    transport = FakeTransport(reply=constant_reply("wrong"))
    engine = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        transport=transport,
    )
    evaluated = tuple(
        engine.evaluate(
            EvaluationRequest(
                candidate=candidate,
                evaluation_role=EvaluationRole.INTERNAL,
                evaluation_context_id=f"demo-{index}",
                purpose=MINIBATCH,
            )
        )
        for index, candidate in enumerate(combinations)
    )

    rendered = [
        request.transcript.messages[-1].content for request in transport.served
    ]
    assert rendered[0] != rendered[1]
    assert {"alpha", "beta"} == {
        "alpha" if "alpha" in prompt else "beta" for prompt in rendered
    }
    assert tuple(result.evidence.candidate for result in evaluated) == tuple(
        candidate_reference(candidate) for candidate in combinations
    )
    assert {
        result.evidence.candidate.record.payload["demo_set_identity_hash"]
        for result in evaluated
    } == {alpha.identity_hash, beta.identity_hash}
    for result, prompt in zip(evaluated, rendered, strict=True):
        output = store.get(result.evidence.outputs_ref.reference)
        assert isinstance(output, dict)
        rows = output["outputs"]
        assert isinstance(rows, list)
        row = rows[0]
        assert isinstance(row, dict)
        assert row["rendered_prompt"] == prompt
