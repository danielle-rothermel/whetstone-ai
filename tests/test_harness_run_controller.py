from __future__ import annotations

from whetstone.coordination.runtime_bootstrap import copro_run_request
from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA, OptimResult


def test_harness_run_controller_completes_copro_run(copro_launch) -> None:
    runtime, launch = copro_launch
    request = copro_run_request(
        launch,
        controller_identity_hash=runtime.controller.runtime_hash,
    )
    result_ref = runtime.controller.drive(request)
    assert result_ref.schema_name == OPTIM_RESULT_SCHEMA
    result = OptimResult.model_validate(runtime.store.get(result_ref.reference))
    assert len(result.proposals) == 1
    assert result.step_results
    assert result.step_results[-1].record.status.value in {
        "complete",
        "failed",
    }


def test_runtime_bootstrap_registers_copro_adapter(toy_runtime) -> None:
    runtime, _control = toy_runtime
    adapter = runtime.adapter_registry.resolve("copro")
    assert adapter.key == "copro"
    assert runtime.controller.runtime_hash
    assert runtime.eval_service.replay_policy.value == "durable_workflow"


def test_load_launch_deserializes_gepa_control(tmp_path) -> None:
    from tests.test_gepa_harness_adapter import _toy_gepa_control

    from whetstone.coordination.harness_run_controller import OptimRunLaunch
    from whetstone.testing.runtime import register_toy_runtime
    from dr_store.sync import open_sqlite
    from whetstone.optim.contracts import OptimRun, OutputContract, StepMode
    from whetstone.testing.toy.experiment import build_toy_experiment

    from whetstone.testing.toy.experiment import TOY_MUTATION_FIELD, toy_template_render_contract

    sqlite_path = str(tmp_path / "gepa-load.sqlite")
    with open_sqlite(sqlite_path) as store:
        runtime = register_toy_runtime(store=store)
        experiment = build_toy_experiment(num_seeds=1)
        control = _toy_gepa_control(
            max_metric_calls=2,
            sqlite_path=sqlite_path,
        )
        run = OptimRun(
            run_id="gepa-platform-run",
            optimizer_config=control.reference(),
            adapter_key="gepa",
            mode=StepMode.PROPOSAL_ONLY,
            terminal_output_contract=OutputContract(returned_proposal_count=1),
            template_render_contract=toy_template_render_contract(),
            mutation_field=TOY_MUTATION_FIELD,
            reward_policy=experiment.reward_policy,
        )
        launch = OptimRunLaunch(
            run=run,
            initial_candidate=experiment.initial_candidate,
            control=control,
        )
        runtime.controller.bind_launch(launch)
        loaded = runtime.controller.load_launch("gepa-platform-run")
        assert loaded.control.identity_hash() == control.identity_hash()
