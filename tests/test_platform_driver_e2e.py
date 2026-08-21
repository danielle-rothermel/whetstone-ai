from __future__ import annotations

from dr_store.content_addressing import parse_object_reference

from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA, OptimResult
from whetstone.platform.contracts import (
    STAGE_OPTIM_STEP,
    OptimWorkInput,
    persist_work_input,
)
from whetstone.platform.step_executor import (
    OPTIM_WORK_STATE_SCHEMA,
    execute_optim_step_sync,
    execute_run_completion_sync,
)


def test_platform_driver_inline_copro_chain(copro_launch) -> None:
    runtime, launch = copro_launch
    control = launch.control
    assert control is not None
    run_id = launch.run.run_id
    work_input = OptimWorkInput(
        run_id=run_id,
        controller_identity_hash=runtime.controller.runtime_hash,
        control_identity_hash=control.identity_hash(),
    )
    input_reference = persist_work_input(runtime.store, work_input)
    stage_index = 0
    current_ref = input_reference

    while True:
        completion = execute_optim_step_sync(
            runtime,
            input_reference=current_ref,
        )
        assert completion.output_reference
        if not completion.successors:
            parsed = parse_object_reference(completion.output_reference)
            assert parsed.schema == OPTIM_WORK_STATE_SCHEMA
            terminal_ref = execute_run_completion_sync(
                runtime,
                input_reference=completion.output_reference,
            )
            parsed_result = parse_object_reference(terminal_ref)
            assert parsed_result.schema == OPTIM_RESULT_SCHEMA
            result = OptimResult.model_validate(
                runtime.store.get(parsed_result)
            )
            assert result.run.record.run_id == run_id
            assert len(result.proposals) == 1
            return
        successor = completion.successors[0]
        assert successor.stage_index == stage_index + 1
        assert successor.stage_key.value == STAGE_OPTIM_STEP
        current_ref = successor.input_reference
        stage_index = successor.stage_index

    raise AssertionError("platform driver chain did not reach run completion")


def test_platform_driver_copro_chain_honors_a_split_terminal_contract(
    toy_runtime,
) -> None:
    """COPRO finalization selects the run's COMPLETE cardinality.

    A run terminal contract may bind ``returned_proposal_count`` and
    ``terminal_proposal_count`` to different values. The harness checks a
    COMPLETE step's accepted candidates against ``accepted_count_for``, so
    finalizing on the continuing count would emit the wrong cardinality and
    the run would be rejected. COPRO's own contracts never set
    ``terminal_proposal_count``; this pins the caller-supplied split form.
    """
    from uuid import uuid4

    from whetstone.coordination.harness_run_controller import OptimRunLaunch
    from whetstone.experiment.candidate import candidate_reference
    from whetstone.optim.contracts import OptimRun, OutputContract, StepMode
    from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY
    from whetstone.testing.toy.experiment import (
        TOY_MUTATION_FIELD,
        build_toy_experiment,
        toy_template_render_contract,
    )

    runtime, control = toy_runtime
    experiment = build_toy_experiment(num_seeds=1)
    run_id = f"copro-split-terminal-{uuid4().hex[:8]}"
    terminal = OutputContract(
        returned_proposal_count=2,
        terminal_proposal_count=1,
    )
    assert terminal.returned_proposal_count != terminal.terminal_proposal_count

    run = OptimRun(
        run_id=run_id,
        optimizer_config=control.reference(),
        adapter_key=COPRO_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=terminal,
        template_render_contract=toy_template_render_contract(),
        initial_candidate_ref=candidate_reference(
            experiment.initial_candidate
        ),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )
    launch = OptimRunLaunch(
        run=run,
        initial_candidate=experiment.initial_candidate,
        control=control,
    )
    runtime.controller.bind_launch(launch)

    work_input = OptimWorkInput(
        run_id=run_id,
        controller_identity_hash=runtime.controller.runtime_hash,
        control_identity_hash=control.identity_hash(),
    )
    current_ref = persist_work_input(runtime.store, work_input)

    while True:
        completion = execute_optim_step_sync(
            runtime,
            input_reference=current_ref,
        )
        assert completion.output_reference
        if not completion.successors:
            terminal_ref = execute_run_completion_sync(
                runtime,
                input_reference=completion.output_reference,
            )
            result = OptimResult.model_validate(
                runtime.store.get(parse_object_reference(terminal_ref))
            )
            # The COMPLETE cardinality of 1, not the continuing count of 2.
            assert len(result.proposals) == 1
            return
        current_ref = completion.successors[0].input_reference
