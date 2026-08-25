"""MIPROv2 must bootstrap through the control's declared mutation field.

The mutated field is per-experiment configuration, not a fixed name. A
run whose field is not the toy default must bootstrap exactly like one
that uses it.

The run is the expensive part and the assertions only read it, so each
demo mode is driven once for the whole module.
"""

from __future__ import annotations

import pytest

from dr_store.sync import open_sqlite

from whetstone.coordination.harness_run_controller import RunRequest
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.miprov2.adapter import MIPROV2_ADAPTER_KEY
from whetstone.optim.miprov2.control import Miprov2DemoMode
from whetstone.testing.runtime import (
    build_miprov2_adapter,
    build_toy_copro_control,
    prepare_toy_miprov2_run,
    register_toy_runtime,
)
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
)
from whetstone.testing.toy.miprov2 import build_toy_miprov2_control

from tests.test_miprov2_harness_e2e import BOOTSTRAP_PURPOSE, _Run

#: Deliberately not the toy default, matching real experiments that name
#: their mutated field something else.
ALTERNATE_MUTATION_FIELD = "prompt_template"


@pytest.fixture(
    scope="module",
    params=list(Miprov2DemoMode),
    ids=lambda mode: mode.value,
)
def alternate_field_run(request, tmp_path_factory):
    """One completed run per demo mode under the alternate mutation field."""

    demo_mode: Miprov2DemoMode = request.param
    run_id = f"mf-{demo_mode.value}"
    experiment = build_toy_experiment(
        num_seeds=1, mutation_field=ALTERNATE_MUTATION_FIELD
    )
    assert ALTERNATE_MUTATION_FIELD in experiment.initial_candidate.payload, (
        "the toy experiment must carry the alternate field"
    )

    directory = tmp_path_factory.mktemp(f"mutation-{demo_mode.value}")
    with open_sqlite(str(directory / f"{run_id}.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(
            store,
            experiment=experiment,
            mutation_field=ALTERNATE_MUTATION_FIELD,
        )
        control = build_toy_miprov2_control(
            engine=engine,
            experiment=experiment,
            demo_mode=demo_mode,
            mutation_field=ALTERNATE_MUTATION_FIELD,
        )
        assert control.mutation_field == ALTERNATE_MUTATION_FIELD
        adapter = build_miprov2_adapter(
            store=store, control=control, engine=engine
        )
        runtime = register_toy_runtime(
            store=store,
            engine=engine,
            copro_control=build_toy_copro_control(engine=engine),
            extra_adapters={MIPROV2_ADAPTER_KEY: adapter},
        )
        prepare_toy_miprov2_run(
            runtime, run_id=run_id, control=control, engine=engine
        )
        terminal_ref = runtime.controller.drive(
            RunRequest(
                controller_identity_hash=runtime.controller.runtime_hash,
                run_id=run_id,
                control_identity_hash=control.identity_hash(),
            )
        )
        yield _Run(
            store=store,
            runtime=runtime,
            control=control,
            terminal_ref=terminal_ref,
            run_id=run_id,
        )


def test_bootstrap_reads_the_controls_mutation_field(
    alternate_field_run,
) -> None:
    assert ALTERNATE_MUTATION_FIELD != TOY_MUTATION_FIELD
    run = alternate_field_run
    demo_mode = run.control.demo_mode

    assert run.intents_for(BOOTSTRAP_PURPOSE), (
        f"{demo_mode.value} must bootstrap through the eval engine "
        "under the alternate mutation field"
    )
    assert run.final_state.bootstrap_plans
    teacher = run.final_state.control.teacher_candidate
    assert ALTERNATE_MUTATION_FIELD in teacher.record.payload
