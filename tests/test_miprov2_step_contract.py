"""MIPROv2 continuation is a pure function of the prior Step Result."""

from __future__ import annotations

from dr_store.sync import open_sqlite

from whetstone.testing.runtime import (
    build_miprov2_adapter,
    build_toy_copro_control,
    prepare_toy_miprov2_run,
    register_toy_runtime,
)
from whetstone.coordination.step_request_builder import StepRequestBuilder
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.contracts import (
    StepStatus,
    optimization_run_reference,
    step_result_reference,
)
from whetstone.optim.miprov2.adapter import MIPROV2_ADAPTER_KEY, MIPROV2_STATE_KEY
from whetstone.optim.miprov2.control import Miprov2DemoMode
from whetstone.testing.toy.miprov2 import build_toy_miprov2_control


def test_build_next_is_identical_from_the_same_prior(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "roundtrip.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        control = build_toy_miprov2_control(
            engine=engine, demo_mode=Miprov2DemoMode.ZEROSHOT
        )
        adapter = build_miprov2_adapter(
            store=store, control=control, engine=engine
        )
        runtime = register_toy_runtime(
            store=store,
            engine=engine,
            copro_control=build_toy_copro_control(engine=engine),
            extra_adapters={MIPROV2_ADAPTER_KEY: adapter},
        )
        launch = prepare_toy_miprov2_run(
            runtime, run_id="round-trip", control=control, engine=engine
        )
        builder = StepRequestBuilder(store=store)
        first = builder.build_first(
            run=optimization_run_reference(launch.run),
            adapter_key=MIPROV2_ADAPTER_KEY,
            initial_candidate=launch.initial_candidate,
            control=control,
            extra_pools=(
                None
                if launch.extra_pools is None
                else dict(launch.extra_pools)
            ),
        )
        runtime.harness.bind_run(launch.run)
        result, result_ref = runtime.harness.run_step(first)
        assert result.status is StepStatus.CONTINUE

        next_one = builder.build_next(
            prior=result,
            prior_ref=result_ref,
            prior_results=(result,),
            control=control,
            mutation_field=launch.run.mutation_field,
        )
        next_two = builder.build_next(
            prior=result,
            prior_ref=result_ref,
            prior_results=(result,),
            control=control,
            mutation_field=launch.run.mutation_field,
        )

        assert next_one.record_content() == next_two.record_content()
        assert next_one.kind_label == next_two.kind_label
        assert (
            next_one.prior_step_result_ref
            == step_result_reference(result).record_ref
        )
        assert next_one.prior_state_ref == result.state_ref
        assert MIPROV2_STATE_KEY not in next_one.pools
        assert next_one.pools == next_two.pools


def test_complete_step_contract_uses_the_run_complete_cardinality() -> None:
    from types import SimpleNamespace

    from whetstone.optim.contracts import OutputContract, StepStatus
    from whetstone.optim.miprov2.adapter import MIPROV2_COMPLETE
    from whetstone.optim.miprov2.step_contract import miprov2_step_output_contract

    terminal = OutputContract(
        returned_proposal_count=0,
        terminal_proposal_count=1,
    )
    run = SimpleNamespace(record=SimpleNamespace(terminal_output_contract=terminal))
    contract = miprov2_step_output_contract(run, kind_label=MIPROV2_COMPLETE)
    assert contract.returned_proposal_count == 0
    assert contract.terminal_proposal_count == 1
    assert contract.accepted_count_for(StepStatus.COMPLETE) == 1
    assert contract.honors_terminal(terminal)
