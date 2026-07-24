from __future__ import annotations

from dataclasses import replace

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.envs.support import FakeTransport, constant_reply, execution_policy
from tests.optimization.support import eval_config
from tests.runner.test_optimization_run import (
    _copro_control,
    _copro_proposal_config,
    _fake_proposer,
    _proposal_config,
    _ScoringService,
    _tool_config,
)
from whetstone.envs.factory import build_env_experiment
from whetstone.evaluation import (
    EngineToolEvaluator,
    EvaluationEngine,
)
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.optimization import (
    BudgetState,
    CodexAdapter,
    CoproAdapter,
    EvaluateCandidateServer,
    EvaluatingToolExecutor,
    FakeCodexRunner,
    GepaAdapter,
    IdentityOptimizerAdapter,
    InProcessMcpProcess,
    MappingAdapterRegistry,
    Miprov2Adapter,
    OptimizationHarness,
    ScriptedAgentCall,
    ToolCallStore,
    eval_config_reference,
    template_placeholder_fields,
)
from whetstone.runner.budget import BudgetGuard, CreditsSnapshot, StopLossError
from whetstone.runner.cell import CellConfig, run_cell
from whetstone.runner.dryrun import run_dry_cell
from whetstone.runner.ledger import Ledger
from whetstone.runner.optimization_run import (
    OptimizationControllerError,
    OptimizationRunControl,
    OptimizationRunServices,
    Optimizer,
)


def _controller_stack(
    name: Optimizer,
    store: ObjectStore,
    experiment,
) -> tuple[OptimizationRunControl, OptimizationRunServices]:
    base = experiment.initial_candidate
    service = _ScoringService(store)
    if name is Optimizer.COPRO:
        adapter = CoproAdapter(
            proposer_config=_copro_proposal_config(),
            transport=_fake_proposer(
                {
                    ("seed_proposal", 0): (
                        f"{base.payload['user_prompt_template']}\nBe concise.",
                    )
                }
            ),
        )
        harness = OptimizationHarness(
            store=store,
            adapter_registry=MappingAdapterRegistry(
                {"identity": IdentityOptimizerAdapter(), "copro": adapter}
            ),
            evaluation_service=service,
        )
        metric = eval_config_reference(eval_config("b" * 64))
        control = OptimizationRunControl(
            run_id="copro:c18:a0",
            optimizer=name,
            candidates=(base,),
            budget=BudgetState(remaining={"proposal_calls": 1}),
            pools={
                "valid_template_keys": list(
                    dict.fromkeys(
                        template_placeholder_fields(
                            str(base.payload["user_prompt_template"])
                        )
                    )
                )
            },
            eval_configs={"internal": metric},
            proposer_config=_copro_proposal_config(),
            copro_control=_copro_control(metric, breadth=2, depth=1),
        )
        return control, OptimizationRunServices(
            store=store,
            harness=harness,
            evaluation_service=service,
        )
    if name is Optimizer.MIPROV2:
        adapter = Miprov2Adapter(
            store=store,
            proposer_config=_proposal_config(),
            transport=_fake_proposer(
                {
                    ("pool_construction", 0): (
                        f"{base.payload['user_prompt_template']}\nBe concise.",
                    )
                }
            ),
        )
        harness = OptimizationHarness(
            store=store,
            adapter_registry=MappingAdapterRegistry({"miprov2": adapter}),
            evaluation_service=service,
        )
        control = OptimizationRunControl(
            run_id="miprov2:c18:a0",
            optimizer=name,
            candidates=(base,),
            budget=BudgetState(
                remaining={"proposal_calls": 1, "search_rollouts": 1}
            ),
            hyperparameters={
                "num_demo_set_candidates": 1,
                "num_instruction_candidates": 1,
                "instruction_attempt_cap": 1,
                "num_trials": 1,
                "minibatch_full_eval_steps": 1,
            },
            eval_configs={
                "bootstrap": eval_config_reference(eval_config("b" * 64)),
                "minibatch": eval_config_reference(eval_config("c" * 64)),
                "full": eval_config_reference(eval_config("d" * 64)),
            },
            proposer_config=_proposal_config(),
        )
        return control, OptimizationRunServices(
            store=store,
            harness=harness,
            evaluation_service=service,
        )
    internal = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        transport=FakeTransport(reply=constant_reply("wrong")),
    )
    executor = EvaluatingToolExecutor(
        EngineToolEvaluator(internal), experiment.reward_policy
    )
    tool_store = ToolCallStore(store)
    if name is Optimizer.GEPA:
        adapter = GepaAdapter(
            reflection_config=_proposal_config(),
            reflection_transport=_fake_proposer(
                {
                    ("gepa_reflection", 0): (
                        f"{base.payload['user_prompt_template']}\nBe concise.",
                    )
                }
            ),
        )
        configs = (
            _tool_config(
                internal,
                name="evaluate_minibatch",
                namespace="cell-gepa-minibatch",
                endpoint="tool://evaluate_minibatch",
            ),
            _tool_config(
                internal,
                name="evaluate_subset",
                namespace="cell-gepa-subset",
                endpoint="tool://evaluate_subset",
            ),
        )
        harness = OptimizationHarness(
            store=store,
            adapter_registry=MappingAdapterRegistry({"gepa": adapter}),
            tool_executor=executor,
            tool_store=tool_store,
        )
        return (
            OptimizationRunControl(
                run_id="gepa:c18:a0",
                optimizer=name,
                candidates=(base,),
                pools={
                    "task_pool": list(
                        internal.sampling.task_set.task_identities
                    )
                },
                hyperparameters={
                    "minibatch_size": 1,
                    "max_reflection_attempts_per_step": 1,
                    "max_reflection_lm_calls": 1,
                },
                tool_configs=configs,
                reflection_config=_proposal_config(),
            ),
            OptimizationRunServices(store=store, harness=harness),
        )
    config = _tool_config(
        internal,
        name="evaluate_candidate",
        namespace="cell-codex",
        endpoint="mcp://whetstone/evaluate_candidate",
    )
    server = EvaluateCandidateServer(
        tool_config=config,
        store=tool_store,
        executor=executor,
    )
    proposal = base.model_copy(
        update={
            "candidate_id": "codex-cell-proposal",
            "payload": {
                **base.payload,
                "user_prompt_template": (
                    f"{base.payload['user_prompt_template']}\nBe concise."
                ),
            },
        }
    )
    adapter = CodexAdapter(
        FakeCodexRunner(
            process=InProcessMcpProcess(server),
            scripted_calls=(
                ScriptedAgentCall(
                    call_id="cell-codex-call",
                    base_ref=base.base_ref,
                    model_route=base.base_ref,
                    template=base.payload["user_prompt_template"],
                ),
            ),
            final_proposals=(proposal,),
        ),
        store=store,
        tool_store=tool_store,
    )
    harness = OptimizationHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry({"codex": adapter}),
        tool_executor=executor,
        tool_store=tool_store,
    )
    return (
        OptimizationRunControl(
            run_id="codex:c18:a0",
            optimizer=name,
            candidates=(base,),
            tool_configs=(config,),
        ),
        OptimizationRunServices(store=store, harness=harness),
    )


def test_identity_cell_uses_canonical_optimization_and_official_eval(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "cell.sqlite"))
    experiment = build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=1,
    )
    official_transport = FakeTransport(reply=constant_reply("wrong"))
    official = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(),
        transport=official_transport,
        prompt_cache=PromptResultCache(tmp_path / "cache"),
    )
    harness = OptimizationHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry(
            {"identity": IdentityOptimizerAdapter()}
        ),
    )
    control = OptimizationRunControl(
        run_id="identity:c18:a0",
        optimizer=Optimizer.IDENTITY,
        candidates=(experiment.initial_candidate,),
    )
    ledger = Ledger(tmp_path / "run")

    outcome = run_dry_cell(
        CellConfig(
            env="c18",
            attempt=0,
            canonical=True,
            task_model="openai/test",
            proposer_model="none",
            lane="test",
            baseline=experiment.initial_candidate,
            optimization=control,
            official_engine=official,
            optimization_services=OptimizationRunServices(
                store=store,
                harness=harness,
            ),
            ledger=ledger,
        )
    )

    assert outcome.record.status == "no-improvement"
    assert outcome.record.artifacts.optimization_result_ref is not None
    trace = outcome.record.artifacts.optimization_trace_ref
    assert trace is not None
    assert (ledger.root / trace).exists()
    assert outcome.record.artifacts.best_candidate_id == (
        experiment.initial_candidate.candidate_id
    )
    assert outcome.record.official_repeats_used == 1
    assert outcome.record.controls.prompt_cache is not None
    assert outcome.record.controls.prompt_cache.hits > 0
    assert ledger.cells() == [outcome.record]
    ledger.append_cell(
        outcome.record.model_copy(
            update={
                "cell_id": "identity:c18:a1",
                "attempt": 1,
            }
        )
    )
    config = CellConfig(
        env="c18",
        attempt=0,
        canonical=True,
        task_model="openai/test",
        proposer_model="none",
        lane="test",
        baseline=experiment.initial_candidate,
        optimization=control,
        official_engine=official,
        optimization_services=OptimizationRunServices(
            store=store,
            harness=harness,
        ),
        ledger=ledger,
        credits_fetcher=lambda: (_ for _ in ()).throw(
            AssertionError("completed reuse must not inspect credits")
        ),
    )
    served_before_resume = len(official_transport.served)
    resumed = run_cell(config)
    assert resumed.skipped is True
    assert resumed.record == outcome.record
    assert len(official_transport.served) == served_before_resume

    changed_candidate = experiment.ceiling_candidate
    assert changed_candidate is not None
    with pytest.raises(
        OptimizationControllerError,
        match="already bound to control",
    ):
        run_cell(
            replace(
                config,
                baseline=changed_candidate,
                optimization=replace(
                    control,
                    candidates=(changed_candidate,),
                ),
            )
        )

    changed_official = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        transport=FakeTransport(reply=constant_reply("wrong")),
    )
    with pytest.raises(ValueError, match="already bound to control"):
        run_cell(replace(config, official_engine=changed_official))

    with pytest.raises(ValueError, match="already bound to control"):
        run_cell(
            replace(
                config,
                task_model="openai/changed",
                lane="changed",
            )
        )


def test_completed_cell_rejects_changed_proposer_config(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "proposer-binding.sqlite"))
    experiment = build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=1,
    )
    official = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(),
        transport=FakeTransport(reply=constant_reply("wrong")),
    )
    control, services = _controller_stack(
        Optimizer.COPRO,
        store,
        experiment,
    )
    ledger = Ledger(tmp_path / "proposer-binding")
    config = CellConfig(
        env="c18",
        attempt=0,
        canonical=True,
        task_model="openai/test",
        proposer_model="test-proposer",
        lane="test",
        baseline=experiment.initial_candidate,
        optimization=control,
        official_engine=official,
        optimization_services=services,
        ledger=ledger,
    )
    first = run_cell(config)
    assert first.record.is_completed()

    changed_proposer = _proposal_config().model_copy(
        update={"temperature": 0.5}
    )
    assert control.copro_control is not None
    changed_copro_control = control.copro_control.model_copy(
        update={
            "prompt_model": changed_proposer,
            "init_temperature": 0.5,
        }
    )
    changed_adapter = CoproAdapter(
        proposer_config=changed_proposer,
        transport=_fake_proposer({}),
    )
    changed_service = _ScoringService(store)
    changed_services = OptimizationRunServices(
        store=store,
        harness=OptimizationHarness(
            store=store,
            adapter_registry=MappingAdapterRegistry(
                {
                    "identity": IdentityOptimizerAdapter(),
                    "copro": changed_adapter,
                }
            ),
            evaluation_service=changed_service,
        ),
        evaluation_service=changed_service,
    )

    with pytest.raises(
        OptimizationControllerError,
        match="already bound to control",
    ):
        run_cell(
            replace(
                config,
                optimization=replace(
                    control,
                    proposer_config=changed_proposer,
                    copro_control=changed_copro_control,
                ),
                optimization_services=changed_services,
            )
        )


def test_official_arms_resume_without_transport_redrive_after_crash(
    tmp_path,
    monkeypatch,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "official-resume.sqlite"))
    experiment = build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=1,
    )
    transport = FakeTransport(reply=constant_reply("wrong"))
    official = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(),
        transport=transport,
    )
    harness = OptimizationHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry(
            {"identity": IdentityOptimizerAdapter()}
        ),
    )
    control = OptimizationRunControl(
        run_id="identity:c18:a0",
        optimizer=Optimizer.IDENTITY,
        candidates=(experiment.initial_candidate,),
    )
    ledger = Ledger(tmp_path / "official-resume")
    config = CellConfig(
        env="c18",
        attempt=0,
        canonical=True,
        task_model="openai/test",
        proposer_model="none",
        lane="test",
        baseline=experiment.initial_candidate,
        ceiling=experiment.ceiling_candidate,
        optimization=control,
        official_engine=official,
        optimization_services=OptimizationRunServices(
            store=store,
            harness=harness,
        ),
        ledger=ledger,
    )
    append_cell = Ledger.append_cell

    def crash_before_ledger(_ledger, _record) -> None:
        raise RuntimeError("simulated crash before cell ledger append")

    monkeypatch.setattr(Ledger, "append_cell", crash_before_ledger)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_cell(config)

    served_before_restart = len(transport.served)
    assert served_before_restart == 3
    for arm in ("baseline", "ceiling", "best"):
        assert store.resolve(
            f"whetstone.runner.official_arm_binding:{config.cell_id}#{arm}"
        )

    monkeypatch.setattr(Ledger, "append_cell", append_cell)
    outcome = run_cell(config)

    assert outcome.record.status == "no-improvement"
    assert len(transport.served) == served_before_restart


@pytest.mark.parametrize(
    "optimizer",
    [
        Optimizer.COPRO,
        Optimizer.MIPROV2,
        Optimizer.GEPA,
        Optimizer.CODEX,
    ],
)
def test_optimizing_cells_use_the_same_canonical_seams(
    tmp_path,
    optimizer: Optimizer,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / f"{optimizer.value}.sqlite"))
    experiment = build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=1,
    )
    official = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(),
        transport=FakeTransport(reply=constant_reply("wrong")),
    )
    control, services = _controller_stack(
        optimizer,
        store,
        experiment,
    )

    outcome = run_dry_cell(
        CellConfig(
            env="c18",
            attempt=0,
            canonical=True,
            task_model="openai/test",
            proposer_model="test-proposer",
            lane="test",
            baseline=experiment.initial_candidate,
            optimization=control,
            official_engine=official,
            optimization_services=services,
            ledger=Ledger(tmp_path / f"run-{optimizer.value}"),
        )
    )

    assert outcome.optimization is not None
    assert outcome.record.optimizer == optimizer.value
    assert outcome.record.artifacts.optimization_result_ref == (
        outcome.optimization.result_ref
    )
    assert outcome.record.artifacts.best_candidate_id is not None


def test_failed_draft_is_distinct_and_has_no_candidate_or_credit_spend(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "failed.sqlite"))
    experiment = build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=1,
    )
    official = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(),
        transport=FakeTransport(reply=constant_reply("wrong")),
    )
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
        run_id="copro:c18:a0",
        optimizer=Optimizer.COPRO,
        candidates=(experiment.initial_candidate,),
        budget=BudgetState(remaining={"proposal_calls": 1}),
        pools={
            "valid_template_keys": list(
                dict.fromkeys(
                    template_placeholder_fields(
                        str(
                            experiment.initial_candidate.payload[
                                "user_prompt_template"
                            ]
                        )
                    )
                )
            )
        },
        eval_configs={"internal": metric},
        proposer_config=_copro_proposal_config(),
        copro_control=_copro_control(metric, breadth=2, depth=1),
    )
    ledger = Ledger(tmp_path / "failed-run")

    outcome = run_cell(
        CellConfig(
            env="c18",
            attempt=0,
            canonical=True,
            task_model="openai/test",
            proposer_model="test-proposer",
            lane="test",
            baseline=experiment.initial_candidate,
            optimization=control,
            official_engine=official,
            optimization_services=OptimizationRunServices(
                store=store,
                harness=harness,
                evaluation_service=service,
            ),
            ledger=ledger,
            credits_fetcher=lambda: CreditsSnapshot(100.0, 0.0, "snapshot"),
        )
    )

    assert outcome.record.status == "proposer-failure"
    assert outcome.record.artifacts.best_candidate_id is None
    assert outcome.record.best_official is None
    assert outcome.record.spend_usd == 0.0
    assert service.calls == []
    assert [record.phase for record in ledger.spend_records()] == [
        "before",
        "checkpoint:official:baseline",
        "checkpoint:optimization:0:seed_proposal",
        "after",
    ]


def test_stop_loss_blocks_next_paid_arm_before_transport(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "stop-loss.sqlite"))
    experiment = build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=1,
    )
    transport = FakeTransport(reply=constant_reply("wrong"))
    official = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(),
        transport=transport,
    )
    control = OptimizationRunControl(
        run_id="identity:c18:a0",
        optimizer=Optimizer.IDENTITY,
        candidates=(experiment.initial_candidate,),
    )
    snapshots = iter(
        (
            CreditsSnapshot(100.0, 0.0, "start"),
            CreditsSnapshot(100.0, 0.0, "baseline"),
            CreditsSnapshot(100.0, 5.0, "best"),
        )
    )
    ledger = Ledger(tmp_path / "stop-loss")

    with pytest.raises(StopLossError, match=r"spent \$5\.00"):
        run_cell(
            CellConfig(
                env="c18",
                attempt=0,
                canonical=True,
                task_model="openai/test",
                proposer_model="none",
                lane="test",
                baseline=experiment.initial_candidate,
                optimization=control,
                official_engine=official,
                optimization_services=OptimizationRunServices(
                    store=store,
                    harness=OptimizationHarness(
                        store=store,
                        adapter_registry=MappingAdapterRegistry(
                            {"identity": IdentityOptimizerAdapter()}
                        ),
                    ),
                ),
                ledger=ledger,
                budget_guard=BudgetGuard(expected_cell_usd=2.0),
                credits_fetcher=lambda: next(snapshots),
            )
        )

    assert len(transport.served) == 1
    assert [record.phase for record in ledger.spend_records()] == [
        "before",
        "checkpoint:official:baseline",
        "checkpoint:official:best",
    ]
