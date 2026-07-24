from __future__ import annotations

from typing import Any

import pytest
from dr_serialize import Jsonable
from dr_store import ObjectStore, SqliteBackend

from tests.envs.support import FakeTransport, constant_reply, execution_policy
from tests.optimization.support import FULL_A, candidate, eval_config
from whetstone.code_eval.power import PowerConfig, analyze_power
from whetstone.envs.factory import build_env_experiment
from whetstone.evaluation import (
    EngineToolEvaluator,
    EvaluationEngine,
    EvaluationEvidence,
    RowAccounting,
)
from whetstone.evaluation.schema import REWARD_SCHEMA
from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization import (
    BudgetState,
    Candidate,
    CodexAdapter,
    CoproAdapter,
    CoproControl,
    EvaluateCandidateServer,
    EvaluatingToolExecutor,
    FakeCodexRunner,
    FakeProposerTransport,
    GepaAdapter,
    IdentityOptimizerAdapter,
    InProcessMcpProcess,
    IntentOutcome,
    IntentResolution,
    MappingAdapterRegistry,
    Miprov2Adapter,
    OptimizationHarness,
    OptimizationStepRequest,
    OptimizationStepResult,
    OutputContract,
    ProposerConfig,
    ResolutionClass,
    ResolutionDetail,
    Reward,
    RewardInputCitation,
    ScriptedAgentCall,
    StepKind,
    StepMode,
    StepStatus,
    ToolCallStore,
    ToolCapacity,
    ToolConfig,
    ToolDefinition,
    eval_config_reference,
    template_placeholder_fields,
    typed_ref_for_record,
)
from whetstone.runner.optimization_run import (
    OPTIMIZATION_RUN_CONTROL_SCHEMA,
    OptimizationControllerError,
    OptimizationRunControl,
    OptimizationRunServices,
    Optimizer,
    _copro_round_attempts,
    derive_power_sampling,
    derive_powered_control,
    run_optimization,
)

PROVIDER_POLICY_HASH = "e" * 64
PROMPT_ADAPTER_HASH = "f" * 64


def _services(path) -> OptimizationRunServices:
    store = ObjectStore(SqliteBackend(path))
    harness = OptimizationHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry(
            {"identity": IdentityOptimizerAdapter()}
        ),
    )
    return OptimizationRunServices(store=store, harness=harness)


class _ScoringService:
    def __init__(self, store: ObjectStore) -> None:
        self.store = store
        self.calls: list[str] = []

    def resolve_evaluation_intent(self, intent) -> IntentResolution:
        self.calls.append(intent.intent_id)
        value = 0.9 if intent.candidate.record.candidate_id == "A" else 0.5
        reward = Reward(
            reward_name="score",
            value=value,
            reward_policy_hash="c" * 64,
            evidence_role=intent.context_role,
            input_citations=(
                RewardInputCitation(
                    name="score",
                    value=value,
                    contributed=value,
                ),
            ),
            evidence_ref_content_hash=typed_ref_for_record(
                "test.aggregate", {"value": value}
            ).content_hash,
        )
        reward_ref = typed_ref_for_record(
            REWARD_SCHEMA, reward.record_content()
        )
        self.store.put(REWARD_SCHEMA, reward.record_content())
        outputs: Jsonable = {
            "candidate_id": intent.candidate.record.candidate_id,
            "outputs": [
                {
                    "rendered_prompt": f"observed input {value}",
                    "output_text": f"observed output {value}",
                    "failure_code": None,
                }
            ],
        }
        outputs_ref = typed_ref_for_record(
            "whetstone.evaluation_outputs", outputs
        )
        self.store.put("whetstone.evaluation_outputs", outputs)
        aggregate: Jsonable = {"value": value}
        aggregate_ref = typed_ref_for_record("test.aggregate", aggregate)
        self.store.put("test.aggregate", aggregate)
        evidence = EvaluationEvidence(
            candidate=intent.candidate,
            eval_config=intent.target_eval_config,
            graph_hash=FULL_A,
            graph_config_ref=FULL_A,
            evaluation_role=EvaluationRole.INTERNAL,
            evaluation_context_id=intent.intent_id,
            purpose=intent.purpose,
            task_identities=("task",),
            repeat_count=1,
            per_task_values=(value,),
            per_task_counts=(1,),
            row_accounting=RowAccounting(
                planned=1,
                present=1,
                missing=0,
                failed=0,
                invalid=0,
            ),
            outputs_ref=outputs_ref,
            aggregate_ref=aggregate_ref,
            aggregate_name="score",
            aggregate_value=value,
            aggregate_status="ok",
            reward_ref=reward_ref,
        )
        evidence_ref = typed_ref_for_record(
            "whetstone.evaluation_evidence", evidence.record_content()
        )
        self.store.put(
            "whetstone.evaluation_evidence", evidence.record_content()
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


def _proposal_config() -> ProposerConfig:
    return ProposerConfig(
        provider_call_config_ref="provider://proposal",
        provider_call_config_hash=FULL_A,
    )


def _copro_proposal_config() -> ProposerConfig:
    return _proposal_config().model_copy(update={"temperature": 1.4})


def _fake_proposer(
    script,
    *,
    default=(),
    strict: bool = True,
) -> FakeProposerTransport:
    return FakeProposerTransport(
        script,
        default=default,
        execution_policy_hash=PROVIDER_POLICY_HASH,
        prompt_adapter_identity_hash=PROMPT_ADAPTER_HASH,
        strict=strict,
    )


def _copro_control(
    metric,
    *,
    breadth: int = 2,
    depth: int = 1,
    track_stats: bool = False,
) -> CoproControl:
    return CoproControl(
        prompt_model=_copro_proposal_config(),
        metric=metric,
        reward_policy_hash="c" * 64,
        breadth=breadth,
        depth=depth,
        init_temperature=1.4,
        track_stats=track_stats,
        provider_execution_policy_hash=PROVIDER_POLICY_HASH,
        prompt_adapter_identity_hash=PROMPT_ADAPTER_HASH,
    )


def _prompt_candidate(candidate_id: str, text: str) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        base_ref="route-a",
        payload={"user_prompt_template": text},
    )


def _valid_keys(record: Candidate) -> list[str]:
    return list(
        dict.fromkeys(
            template_placeholder_fields(
                str(record.payload["user_prompt_template"])
            )
        )
    )


def test_identity_is_one_pure_durable_step_and_resumes(tmp_path) -> None:
    path = tmp_path / "cell.sqlite"
    control = OptimizationRunControl(
        run_id="identity-cell",
        optimizer=Optimizer.IDENTITY,
        candidates=(candidate(),),
        budget=BudgetState(remaining={"rollouts": 8}),
    )

    first = run_optimization(control, _services(path))
    restarted = run_optimization(control, _services(path))

    assert first.result.status is StepStatus.COMPLETE
    assert len(first.result.step_result_refs) == 1
    assert first.result_ref == restarted.result_ref
    assert first.trace_ref == restarted.trace_ref
    assert first.result.proposals[0].candidate.record == candidate()


def test_run_id_is_bound_to_one_exact_control_before_terminal_reuse(
    tmp_path,
) -> None:
    path = tmp_path / "bound-control.sqlite"
    services = _services(path)
    control = OptimizationRunControl(
        run_id="bound-identity",
        optimizer=Optimizer.IDENTITY,
        candidates=(candidate(),),
        hyperparameters={"revision": 1},
    )

    completed = run_optimization(control, services)
    binding = services.store.resolve(
        f"{OPTIMIZATION_RUN_CONTROL_SCHEMA}:{control.run_id}"
    )
    assert binding is not None
    assert binding.content_hash == control.config_hash
    assert services.store.get(binding) == control.record.record_content()

    changed = OptimizationRunControl(
        run_id=control.run_id,
        optimizer=Optimizer.IDENTITY,
        candidates=(candidate(),),
        hyperparameters={"revision": 2},
    )
    with pytest.raises(
        OptimizationControllerError, match="already bound to control"
    ):
        run_optimization(changed, services)

    assert services.harness.resolve_optimization_result(control.run_id) == (
        completed.result_ref
    )


def test_prior_request_must_match_bound_run_control(tmp_path) -> None:
    services = _services(tmp_path / "foreign-prior.sqlite")
    control = OptimizationRunControl(
        run_id="foreign-prior",
        optimizer=Optimizer.IDENTITY,
        candidates=(candidate(),),
    )
    services.harness.run_step(
        OptimizationStepRequest(
            run_id=control.run_id,
            step_id="foreign-prior:0:identity",
            optimizer_config_hash=FULL_A,
            adapter_key="identity",
            mode=StepMode.PURE,
            kind=StepKind.IDENTITY,
            step_index=0,
            candidates=control.candidates,
            budget=control.budget,
            output_contract=OutputContract(returned_proposal_count=1),
        )
    )

    with pytest.raises(
        OptimizationControllerError, match="bound Optimization Run Control"
    ):
        run_optimization(control, services)


def test_proposer_config_is_exact_control_and_adapter_identity(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "proposer.sqlite"))
    configured = _copro_proposal_config()
    mismatched = configured.model_copy(update={"temperature": 0.25})
    adapter = CoproAdapter(
        proposer_config=mismatched,
        transport=_fake_proposer({}),
    )
    services = OptimizationRunServices(
        store=store,
        harness=OptimizationHarness(
            store=store,
            adapter_registry=MappingAdapterRegistry(
                {"identity": IdentityOptimizerAdapter(), "copro": adapter}
            ),
        ),
    )
    metric = eval_config_reference(eval_config("b" * 64))
    control = OptimizationRunControl(
        run_id="proposer-mismatch",
        optimizer=Optimizer.COPRO,
        candidates=(candidate(),),
        pools={"valid_template_keys": []},
        eval_configs={"internal": metric},
        proposer_config=configured,
        copro_control=_copro_control(metric),
    )

    with pytest.raises(
        OptimizationControllerError, match="Proposer Config does not match"
    ):
        run_optimization(control, services)

    assert (
        store.resolve(f"{OPTIMIZATION_RUN_CONTROL_SCHEMA}:{control.run_id}")
        is None
    )


def test_algorithm_controls_require_exact_config_scopes() -> None:
    internal = eval_config_reference(eval_config())

    with pytest.raises(ValueError, match=r"bootstrap.*full.*minibatch"):
        OptimizationRunControl(
            run_id="mipro-cell",
            optimizer=Optimizer.MIPROV2,
            candidates=(candidate(),),
            eval_configs={"internal": internal},
        )


def test_power_materializes_exact_sampling_and_control_identity() -> None:
    experiment = build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=4,
        split_sizes=(3, 1, 1),
        repeats=1,
    )
    base = experiment.eval_configs.internal
    power = analyze_power(
        naive_per_task=(0.0, 0.0, 0.0),
        ceiling_per_task=(1.0, 1.0, 1.0),
        pool_ceiling=3,
        anchor_repeats=1,
        config=PowerConfig(repeat_cap=2, trials=10),
    )
    derived = derive_power_sampling(base, power, minimum_n_tasks=2)
    internal_ref = eval_config_reference(
        experiment.eval_configs.internal.eval_config
    )
    control = OptimizationRunControl(
        run_id="powered-copro",
        optimizer=Optimizer.COPRO,
        candidates=(experiment.initial_candidate,),
        pools={
            "valid_template_keys": _valid_keys(experiment.initial_candidate)
        },
        eval_configs={"internal": internal_ref},
        proposer_config=_copro_proposal_config(),
        copro_control=_copro_control(internal_ref),
    )
    powered = derive_powered_control(
        control,
        samplings={"internal": derived.sampling},
    )

    assert len(derived.sampling.instances) == derived.used_n_tasks
    assert derived.sampling.repeat_plan.repeat_count == derived.used_repeats
    assert powered.eval_configs["internal"].identity_hash == (
        derived.sampling.eval_config.config_identity_hash
    )
    assert powered.copro_control is not None
    assert powered.copro_control.metric == powered.eval_configs["internal"]
    assert powered.config_hash != control.config_hash

    with pytest.raises(ValueError, match="one exact Tool Config"):
        OptimizationRunControl(
            run_id="codex-cell",
            optimizer=Optimizer.CODEX,
            candidates=(candidate(),),
        )


def test_copro_runs_exact_rounds_returns_full_ranking_and_restarts(
    tmp_path,
) -> None:
    database = tmp_path / "copro.sqlite"
    store = ObjectStore(SqliteBackend(database))
    service = _ScoringService(store)
    metric = eval_config_reference(eval_config("b" * 64))
    transport = _fake_proposer(
        {
            ("seed_proposal", 0): ("seed candidate",),
            ("history_proposal", 1): (
                "history candidate one",
                "history candidate two",
            ),
        }
    )
    adapter = CoproAdapter(
        proposer_config=_copro_proposal_config(),
        transport=transport,
    )
    harness = OptimizationHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry(
            {"identity": IdentityOptimizerAdapter(), "copro": adapter}
        ),
        evaluation_service=service,
    )
    control = OptimizationRunControl(
        run_id="copro-cell",
        optimizer=Optimizer.COPRO,
        candidates=(_prompt_candidate("A", "baseline template"),),
        output_count=1,
        budget=BudgetState(remaining={"proposal_calls": 3}),
        pools={"valid_template_keys": []},
        eval_configs={"internal": metric},
        proposer_config=_copro_proposal_config(),
        copro_control=_copro_control(
            metric,
            breadth=2,
            depth=2,
            track_stats=True,
        ),
    )

    execution = run_optimization(
        control,
        OptimizationRunServices(
            store=store,
            harness=harness,
            evaluation_service=service,
        ),
    )

    assert execution.result.status is StepStatus.COMPLETE
    assert [call[2] for call in transport.calls] == [1, 2]
    assert len(service.calls) == 4
    assert len(execution.result.step_result_refs) == 3
    assert [
        proposal.candidate.record.candidate_id
        for proposal in execution.result.proposals
    ] == [
        "A",
        "copro:copro-cell:0",
        "copro:copro-cell:2",
        "copro:copro-cell:3",
    ]
    assert [step.dispositions for step in execution.trace.steps] == [
        ("completed", "completed"),
        ("completed", "completed"),
        (),
    ]
    seed = OptimizationStepResult.model_validate(
        store.get(execution.trace.steps[0].result_ref.reference)
    )
    assert [
        item.intent.candidate.record.candidate_id
        for item in seed.resolved_intents
    ] == ["copro:copro-cell:0", "A"]
    assert all(
        item.intent.purpose == "seed_proposal"
        for item in seed.resolved_intents
    )
    history_step = OptimizationStepResult.model_validate(
        store.get(execution.trace.steps[1].result_ref.reference)
    )
    history_request = OptimizationStepRequest.model_validate(
        store.get(history_step.request_ref.reference)
    )
    h0 = history_request.pools["attempt_history"]
    assert len(h0) == 2
    assert [item["occurrence_ordinal"] for item in h0] == [0, 1]
    assert all(item["candidate"] for item in h0)
    assert all(item["evaluation_evidence_refs"] for item in h0)
    assert all(item["reward_ref"] for item in h0)
    terminal = OptimizationStepResult.model_validate(
        store.get(execution.trace.steps[2].result_ref.reference)
    )
    terminal_request = OptimizationStepRequest.model_validate(
        store.get(terminal.request_ref.reference)
    )
    finalization = terminal_request.pools["copro_finalization"]
    assert finalization["total_calls"] == 4
    assert finalization["statistics"]["total_calls"] == 4
    assert len(finalization["ranked_attempts"]) == 4
    assert control.copro_control is not None
    assert (
        control.record.copro_control_identity_hash
        == control.copro_control.identity_hash()
    )

    fresh_store = ObjectStore(SqliteBackend(database))
    fresh_service = _ScoringService(fresh_store)
    fresh_adapter = CoproAdapter(
        proposer_config=_copro_proposal_config(),
        transport=_fake_proposer({}),
    )
    restarted = run_optimization(
        control,
        OptimizationRunServices(
            store=fresh_store,
            harness=OptimizationHarness(
                store=fresh_store,
                adapter_registry=MappingAdapterRegistry(
                    {
                        "identity": IdentityOptimizerAdapter(),
                        "copro": fresh_adapter,
                    }
                ),
                evaluation_service=fresh_service,
            ),
            evaluation_service=fresh_service,
        ),
    )
    assert restarted.result_ref == execution.result_ref
    assert fresh_adapter.invocations == 0
    assert fresh_service.calls == []


def test_copro_control_rejects_duplicate_state_and_multiple_seeds() -> None:
    metric = eval_config_reference(eval_config("b" * 64))
    common: dict[str, Any] = {
        "run_id": "copro-invalid",
        "optimizer": Optimizer.COPRO,
        "eval_configs": {"internal": metric},
        "proposer_config": _copro_proposal_config(),
        "copro_control": _copro_control(metric),
    }
    with pytest.raises(ValueError, match="exactly one initial candidate"):
        OptimizationRunControl(
            **common,
            candidates=(
                _prompt_candidate("A", "one"),
                _prompt_candidate("B", "two"),
            ),
            pools={"valid_template_keys": []},
        )
    with pytest.raises(ValueError, match="noncanonical"):
        OptimizationRunControl(
            **common,
            candidates=(_prompt_candidate("A", "one"),),
            pools={
                "valid_template_keys": [],
                "attempt_history": [],
            },
        )
    with pytest.raises(ValueError, match="owned by CoproControl"):
        OptimizationRunControl(
            **common,
            candidates=(_prompt_candidate("A", "one"),),
            pools={"valid_template_keys": []},
            hyperparameters={"breadth": 2},
        )


def test_copro_rejects_duplicated_omitted_or_reordered_occurrences(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "copro-order.sqlite"))
    service = _ScoringService(store)
    metric = eval_config_reference(eval_config("b" * 64))
    adapter = CoproAdapter(
        proposer_config=_copro_proposal_config(),
        transport=_fake_proposer({("seed_proposal", 0): ("seed candidate",)}),
    )
    control = OptimizationRunControl(
        run_id="copro-order",
        optimizer=Optimizer.COPRO,
        candidates=(_prompt_candidate("A", "baseline template"),),
        budget=BudgetState(remaining={"proposal_calls": 1}),
        pools={"valid_template_keys": []},
        eval_configs={"internal": metric},
        proposer_config=_copro_proposal_config(),
        copro_control=_copro_control(metric, breadth=2, depth=1),
    )
    execution = run_optimization(
        control,
        OptimizationRunServices(
            store=store,
            harness=OptimizationHarness(
                store=store,
                adapter_registry=MappingAdapterRegistry(
                    {
                        "identity": IdentityOptimizerAdapter(),
                        "copro": adapter,
                    }
                ),
                evaluation_service=service,
            ),
            evaluation_service=service,
        ),
    )
    seed = OptimizationStepResult.model_validate(
        store.get(execution.trace.steps[0].result_ref.reference)
    )
    first, second = seed.resolved_intents
    corruptions = (
        seed.model_copy(update={"resolved_intents": (second, first)}),
        seed.model_copy(update={"resolved_intents": (first, first)}),
        seed.model_copy(update={"resolved_intents": (first,)}),
    )
    for corrupted in corruptions:
        with pytest.raises(OptimizationControllerError):
            _copro_round_attempts(control, corrupted, store)

    terminal = OptimizationStepResult.model_validate(
        store.get(execution.trace.steps[1].result_ref.reference)
    )
    terminal_request = OptimizationStepRequest.model_validate(
        store.get(terminal.request_ref.reference)
    )
    assert "statistics" not in terminal_request.pools["copro_finalization"]


def test_copro_restart_reconstructs_typed_occurrences_before_next_round(
    tmp_path,
) -> None:
    database = tmp_path / "copro-mid-round.sqlite"
    store = ObjectStore(SqliteBackend(database))
    service = _ScoringService(store)
    metric = eval_config_reference(eval_config("b" * 64))
    first_adapter = CoproAdapter(
        proposer_config=_copro_proposal_config(),
        transport=_fake_proposer({("seed_proposal", 0): ("seed candidate",)}),
    )
    control = OptimizationRunControl(
        run_id="copro-restart",
        optimizer=Optimizer.COPRO,
        candidates=(_prompt_candidate("A", "baseline template"),),
        budget=BudgetState(remaining={"proposal_calls": 3}),
        pools={"valid_template_keys": []},
        eval_configs={"internal": metric},
        proposer_config=_copro_proposal_config(),
        copro_control=_copro_control(metric, breadth=2, depth=2),
    )
    checkpoints = 0

    def stop_before_second_round(_label: str) -> None:
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 2:
            raise RuntimeError("simulated controller crash")

    with pytest.raises(RuntimeError, match="simulated controller crash"):
        run_optimization(
            control,
            OptimizationRunServices(
                store=store,
                harness=OptimizationHarness(
                    store=store,
                    adapter_registry=MappingAdapterRegistry(
                        {
                            "identity": IdentityOptimizerAdapter(),
                            "copro": first_adapter,
                        }
                    ),
                    evaluation_service=service,
                ),
                evaluation_service=service,
                before_paid_step=stop_before_second_round,
            ),
        )
    assert first_adapter.invocations == 1
    assert len(service.calls) == 2

    fresh_store = ObjectStore(SqliteBackend(database))
    fresh_service = _ScoringService(fresh_store)
    fresh_adapter = CoproAdapter(
        proposer_config=_copro_proposal_config(),
        transport=_fake_proposer(
            {
                ("history_proposal", 1): (
                    "history one",
                    "history two",
                )
            }
        ),
    )
    resumed = run_optimization(
        control,
        OptimizationRunServices(
            store=fresh_store,
            harness=OptimizationHarness(
                store=fresh_store,
                adapter_registry=MappingAdapterRegistry(
                    {
                        "identity": IdentityOptimizerAdapter(),
                        "copro": fresh_adapter,
                    }
                ),
                evaluation_service=fresh_service,
            ),
            evaluation_service=fresh_service,
        ),
    )

    assert resumed.result.status is StepStatus.COMPLETE
    assert fresh_adapter.invocations == 1
    assert len(fresh_service.calls) == 2
    assert len(resumed.result.proposals) == 4


def test_failed_draft_is_typed_and_restart_does_not_reinvoke(
    tmp_path,
) -> None:
    database = tmp_path / "failed-copro.sqlite"
    store = ObjectStore(SqliteBackend(database))
    service = _ScoringService(store)
    metric = eval_config_reference(eval_config("b" * 64))
    adapter = CoproAdapter(
        proposer_config=_copro_proposal_config(),
        transport=_fake_proposer({}),
    )
    harness = OptimizationHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry(
            {"identity": IdentityOptimizerAdapter(), "copro": adapter}
        ),
        evaluation_service=service,
    )
    control = OptimizationRunControl(
        run_id="failed-copro-cell",
        optimizer=Optimizer.COPRO,
        candidates=(_prompt_candidate("A", "baseline template"),),
        budget=BudgetState(remaining={"proposal_calls": 1}),
        pools={"valid_template_keys": []},
        eval_configs={"internal": metric},
        proposer_config=_copro_proposal_config(),
        copro_control=_copro_control(metric, breadth=2, depth=1),
    )

    first = run_optimization(
        control,
        OptimizationRunServices(
            store=store,
            harness=harness,
            evaluation_service=service,
        ),
    )
    fresh_store = ObjectStore(SqliteBackend(database))
    fresh_service = _ScoringService(fresh_store)
    fresh_adapter = CoproAdapter(
        proposer_config=_copro_proposal_config(),
        transport=_fake_proposer({}),
    )
    restarted = run_optimization(
        control,
        OptimizationRunServices(
            store=fresh_store,
            harness=OptimizationHarness(
                store=fresh_store,
                adapter_registry=MappingAdapterRegistry(
                    {
                        "identity": IdentityOptimizerAdapter(),
                        "copro": fresh_adapter,
                    }
                ),
                evaluation_service=fresh_service,
            ),
            evaluation_service=fresh_service,
        ),
    )

    assert first.result.status is StepStatus.FAILED
    assert first.result.proposals == ()
    assert service.calls == []
    assert first.result_ref == restarted.result_ref
    assert fresh_adapter.invocations == 0


def test_miprov2_runs_full_driver_cadence_with_exact_configs(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "mipro.sqlite"))
    service = _ScoringService(store)
    adapter = Miprov2Adapter(
        store=store,
        proposer_config=_proposal_config(),
        transport=_fake_proposer(
            {("pool_construction", 0): ("instruction candidate",)}
        ),
    )
    harness = OptimizationHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry({"miprov2": adapter}),
        evaluation_service=service,
    )
    control = OptimizationRunControl(
        run_id="mipro-cell",
        optimizer=Optimizer.MIPROV2,
        candidates=(_prompt_candidate("A", "baseline template"),),
        budget=BudgetState(
            remaining={"proposal_calls": 1, "search_rollouts": 1}
        ),
        hyperparameters={
            "num_demo_set_candidates": 1,
            "num_instruction_candidates": 1,
            "instruction_attempt_cap": 1,
            "num_trials": 1,
            "minibatch_full_eval_steps": 1,
            "seed": 7,
        },
        eval_configs={
            "bootstrap": eval_config_reference(eval_config("b" * 64)),
            "minibatch": eval_config_reference(eval_config("c" * 64)),
            "full": eval_config_reference(eval_config("d" * 64)),
        },
        proposer_config=_proposal_config(),
    )

    execution = run_optimization(
        control,
        OptimizationRunServices(
            store=store,
            harness=harness,
            evaluation_service=service,
        ),
    )

    assert execution.result.status is StepStatus.COMPLETE
    assert len(execution.result.step_result_refs) == 6
    purposes = [
        resolution.intent.purpose
        for step in execution.trace.steps
        for resolution in (
            OptimizationStepResult.model_validate(
                store.get(step.result_ref.reference)
            ).resolved_intents
        )
    ]
    assert purposes == [
        "bootstrap",
        "baseline_full",
        "minibatch",
    ]


def _experiment():
    return build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=1,
    )


def _tool_config(
    engine: EvaluationEngine,
    *,
    name: str,
    namespace: str,
    endpoint: str,
) -> ToolConfig:
    definition = ToolDefinition(
        tool_name=name,
        input_fields=("base_ref", "model_route", "template", "task_ids"),
        output_fields=("objective_values", "evaluation_evidence_ref"),
    )
    return ToolConfig(
        tool_name=name,
        tool_definition_ref=f"tooldef://{name}",
        tool_definition_identity_hash=definition.identity_hash(),
        endpoint=endpoint,
        eval_config_ref=engine.eval_config_ref.record_ref.content_hash,
        eval_config_identity_hash=engine.eval_config_ref.identity_hash,
        reward_policy_ref=engine.experiment.reward_policy.identity_hash(),
        capacity=ToolCapacity(max_accepted_calls=10),
        store_namespace=namespace,
    )


def test_gepa_runs_real_tool_evaluation_seam(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "gepa.sqlite"))
    experiment = _experiment()
    engine = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        transport=FakeTransport(reply=constant_reply("wrong")),
    )
    base = experiment.initial_candidate
    reflection = _fake_proposer(
        {
            ("gepa_reflection", 0): (
                f"{base.payload['user_prompt_template']}\nBe concise.",
            )
        }
    )
    adapter = GepaAdapter(
        reflection_config=_proposal_config(),
        reflection_transport=reflection,
    )
    tool_store = ToolCallStore(store)
    executor = EvaluatingToolExecutor(
        EngineToolEvaluator(engine), experiment.reward_policy
    )
    configs = (
        _tool_config(
            engine,
            name="evaluate_minibatch",
            namespace="gepa-minibatch",
            endpoint="tool://evaluate_minibatch",
        ),
        _tool_config(
            engine,
            name="evaluate_subset",
            namespace="gepa-subset",
            endpoint="tool://evaluate_subset",
        ),
    )
    harness = OptimizationHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry({"gepa": adapter}),
        tool_executor=executor,
        tool_store=tool_store,
    )
    control = OptimizationRunControl(
        run_id="gepa-cell",
        optimizer=Optimizer.GEPA,
        candidates=(base,),
        pools={"task_pool": list(engine.sampling.task_set.task_identities)},
        hyperparameters={
            "minibatch_size": 1,
            "max_reflection_attempts_per_step": 1,
            "max_reflection_lm_calls": 1,
        },
        tool_configs=configs,
        reflection_config=_proposal_config(),
    )

    execution = run_optimization(
        control,
        OptimizationRunServices(store=store, harness=harness),
    )

    assert execution.result.status is StepStatus.COMPLETE
    assert execution.trace.steps[0].evidence_refs
    assert reflection.calls


def test_codex_runs_one_opaque_step_through_real_mcp_seam(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "codex.sqlite"))
    experiment = _experiment()
    engine = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        transport=FakeTransport(reply=constant_reply("wrong")),
    )
    base = experiment.initial_candidate
    config = _tool_config(
        engine,
        name="evaluate_candidate",
        namespace="codex-cell",
        endpoint="mcp://whetstone/evaluate_candidate",
    )
    tool_store = ToolCallStore(store)
    executor = EvaluatingToolExecutor(
        EngineToolEvaluator(engine), experiment.reward_policy
    )
    server = EvaluateCandidateServer(
        tool_config=config,
        store=tool_store,
        executor=executor,
    )
    proposal = Candidate(
        candidate_id="codex-proposal",
        base_ref=base.base_ref,
        payload={
            **base.payload,
            "user_prompt_template": (
                f"{base.payload['user_prompt_template']}\nBe concise."
            ),
        },
    )
    runner = FakeCodexRunner(
        process=InProcessMcpProcess(server),
        scripted_calls=(
            ScriptedAgentCall(
                call_id="codex-call-1",
                base_ref=base.base_ref,
                model_route=base.base_ref,
                template=base.payload["user_prompt_template"],
            ),
        ),
        final_proposals=(proposal,),
    )
    adapter = CodexAdapter(
        runner,
        store=store,
        tool_store=tool_store,
    )
    harness = OptimizationHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry({"codex": adapter}),
        tool_executor=executor,
        tool_store=tool_store,
    )
    control = OptimizationRunControl(
        run_id="codex-cell",
        optimizer=Optimizer.CODEX,
        candidates=(base,),
        tool_configs=(config,),
    )

    execution = run_optimization(
        control,
        OptimizationRunServices(store=store, harness=harness),
    )
    restarted = run_optimization(
        control,
        OptimizationRunServices(store=store, harness=harness),
    )

    assert execution.result.status is StepStatus.COMPLETE
    assert len(execution.result.step_result_refs) == 1
    assert restarted.result_ref == execution.result_ref
    assert len(runner.observed_payloads) == 1
    assert runner.observed_payloads[0]["refused"] is False
    assert [
        call.call_id
        for call in tool_store.namespace_calls(
            config.store_namespace, config.identity_hash()
        )
    ] == ["codex-call-1"]
