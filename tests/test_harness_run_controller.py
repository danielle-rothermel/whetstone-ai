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
