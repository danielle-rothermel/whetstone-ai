from __future__ import annotations

import hashlib
import random

import pytest

from whetstone.optimization.miprov2.demo import (
    ComponentDemo,
    ComponentDemoSequence,
    ComponentDemoSet,
    DemoSourceKind,
    LabeledTaskDemo,
)
from whetstone.optimization.miprov2.proposal import (
    MIPROV2_DEMO_BRIDGE_VERSION,
    NO_TASK_DEMOS,
    TIP_TEXTS,
    Miprov2ComponentDemoCandidates,
    Miprov2DatasetExample,
    Miprov2DemoField,
    Miprov2DemoSet,
    Miprov2InstructionGenerationFailed,
    Miprov2PromptComponent,
    Miprov2ProposalDemo,
    Miprov2ProposalRequest,
    Miprov2ProposalResponse,
    Miprov2ProposalState,
    fold_proposal_response,
    plan_next_proposal_request,
    proposal_candidates_from_demo_sets,
    start_miprov2_proposal,
)
from whetstone.optimization.miprov2.rng import (
    Miprov2DurableBindings,
    Miprov2RngCheckpoint,
    Miprov2RngDraw,
)


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _bindings() -> Miprov2DurableBindings:
    return Miprov2DurableBindings(
        control_identity_hash=_identity("control"),
        prompt_route_identity_hash=_identity("prompt-route"),
        task_route_identity_hash=_identity("task-route"),
        execution_policy_identity_hash=_identity("execution-policy"),
        prompt_adapter_identity_hash=_identity("prompt-adapter"),
        base_candidate_identity_hash=_identity("base"),
        teacher_candidate_identity_hash=_identity("teacher"),
    )


def _component(
    component_id: str = "user_prompt_template",
    template: str = "Answer {input}.",
) -> Miprov2PromptComponent:
    return Miprov2PromptComponent(
        component_id=component_id,
        template=template,
        allowed_placeholders=("input",),
        rendering_rules=(
            "Substitute the named input without DSPy field labels."
        ),
        example_execution="Answer blue. -> blue",
    )


def _dataset(count: int) -> tuple[Miprov2DatasetExample, ...]:
    return tuple(
        Miprov2DatasetExample(
            task_identity=f"{index + 1:064x}",
            rendered_record=f"input={index}; expected={index}",
        )
        for index in range(count)
    )


def _start(
    *,
    components: tuple[Miprov2PromptComponent, ...] | None = None,
    dataset_count: int = 1,
    demos: tuple[Miprov2ComponentDemoCandidates, ...] | None = None,
    candidates: int = 2,
    seed: int = 2,
    data_aware: bool = False,
    program_aware: bool = True,
    tip_aware: bool = True,
    fewshot_aware: bool = True,
    batch_size: int = 10,
) -> Miprov2ProposalState:
    return start_miprov2_proposal(
        bindings=_bindings(),
        components=components or (_component(),),
        trainset=_dataset(dataset_count),
        demo_candidates=demos,
        num_candidates=candidates,
        view_data_batch_size=batch_size,
        init_temperature=0.7,
        data_aware=data_aware,
        program_aware=program_aware,
        tip_aware=tip_aware,
        fewshot_aware=fewshot_aware,
        rng_checkpoint=Miprov2RngCheckpoint.seeded(seed),
    )


def _next(
    state: Miprov2ProposalState,
) -> tuple[Miprov2ProposalState, Miprov2ProposalRequest]:
    planned = plan_next_proposal_request(state)
    assert planned.request is not None
    return planned.state, planned.request


def _fold(
    state: Miprov2ProposalState,
    request: Miprov2ProposalRequest,
    text: str = "ok",
    *,
    failed: bool = False,
) -> Miprov2ProposalState:
    response = Miprov2ProposalResponse(
        request_identity_hash=request.identity_hash,
        text="" if failed else text,
        failed=failed,
        failure_detail="scripted failure" if failed else None,
        evidence={"request": request.effect_ordinal},
    )
    return fold_proposal_response(state, response)


def _finish_proposals(
    state: Miprov2ProposalState,
) -> tuple[Miprov2ProposalState, tuple[Miprov2ProposalRequest, ...]]:
    requests: list[Miprov2ProposalRequest] = []
    effects_per_proposal = 3 if state.program_aware else 1
    expected_effect_count = (
        len(state.components) * state.proposal_count * effects_per_proposal
    )
    for _transition in range(expected_effect_count + 1):
        planned = plan_next_proposal_request(state)
        state = planned.state
        if planned.request is None:
            assert len(requests) == expected_effect_count, (
                "proposal machine terminated before its exact effect count; "
                f"requests={requests!r}; state={state.model_dump_json()}"
            )
            return state, tuple(requests)
        request = planned.request
        requests.append(request)
        if request.effect == "program_description":
            text = f"Program description {request.component_index}"
        elif request.effect == "component_description":
            text = f"Component description {request.component_index}"
        elif request.effect == "instruction_proposal":
            text = (
                f"Instruction: generated-{request.component_index}-"
                f"{request.proposal_index} {{input}}"
            )
        else:
            raise AssertionError(f"unexpected effect {request.effect}")
        state = _fold(state, request, text)
    raise AssertionError(
        "proposal machine did not terminate after its exact effect count; "
        f"requests={requests!r}; state={state.model_dump_json()}"
    )


def _demo(
    label: str,
    *,
    augmented_key_present: bool = True,
    augmented: bool | None = True,
) -> Miprov2ProposalDemo:
    return Miprov2ProposalDemo(
        fields=(
            Miprov2DemoField(name="input", value=f"in-{label}"),
            Miprov2DemoField(name="output", value=f"out-{label}"),
        ),
        augmented_key_present=augmented_key_present,
        augmented=augmented,
    )


def test_dataset_summary_has_one_initial_nine_followups_and_final() -> None:
    state = _start(
        dataset_count=120,
        candidates=1,
        data_aware=True,
        program_aware=False,
        tip_aware=False,
        batch_size=10,
    )
    effects: list[str] = []
    while state.stage != "proposal_select":
        state, request = _next(state)
        effects.append(request.effect)
        text = (
            "Summary: compact dataset"
            if request.effect == "dataset_final"
            else f"observation-{request.effect_ordinal}"
        )
        state = _fold(state, request, text)

    assert effects == [
        "dataset_initial",
        *(["dataset_followup"] * 9),
        "dataset_final",
    ]
    assert state.dataset_descriptor_calls == 10
    assert state.dataset_summary == "compact dataset"
    assert all(item.request.temperature == 1.0 for item in state.evidence)
    assert all(item.request.rollout_id is None for item in state.evidence)


def test_dataset_complete_skips_are_cumulative_and_observations_concat() -> (
    None
):
    state = _start(
        dataset_count=200,
        candidates=1,
        data_aware=True,
        program_aware=False,
        tip_aware=False,
        batch_size=10,
    )
    state, request = _next(state)
    state = _fold(state, request, "A")

    scripted = (
        "COMPLETE first",
        "B",
        "complete second",
        "C",
        "COMPLETE third",
        "D",
        "COMPLETE fourth",
        "COMPLETE fifth",
    )
    for text in scripted:
        state, request = _next(state)
        assert request.effect == "dataset_followup"
        state = _fold(state, request, text)

    state, request = _next(state)
    assert request.effect == "dataset_final"
    assert request.fields[0].value == "ABCD"
    state = _fold(state, request, "dataset summary")

    assert state.dataset_complete_skips == 5
    assert state.dataset_observations == "ABCD"
    assert (
        sum(item.rejection_reason == "COMPLETE" for item in state.evidence)
        == 5
    )


def test_one_component_snapshot_none_tip_and_displaced_first_result() -> None:
    assert TIP_TEXTS == (
        ("none", ""),
        (
            "creative",
            "Don't be afraid to be creative when creating the new "
            "instruction!",
        ),
        ("simple", "Keep the instruction clear and concise."),
        (
            "description",
            "Make sure your instruction is very informative and descriptive.",
        ),
        (
            "high_stakes",
            "The instruction should include a high stakes scenario in which "
            "the LM must solve the task!",
        ),
        (
            "persona",
            "Include a persona that is relevant to the task in the "
            'instruction (ie. "You are a ...")',
        ),
    )
    state = _start(seed=2, candidates=2)
    state, requests = _finish_proposals(state)

    assert [request.effect for request in requests] == [
        "program_description",
        "component_description",
        "instruction_proposal",
        "program_description",
        "component_description",
        "instruction_proposal",
    ]
    first_instruction = requests[2]
    assert first_instruction.selected_tip_key == "none"
    assert [field.name for field in first_instruction.fields] == [
        "Whetstone prompt-component graph",
        "Program description",
        "Selected component",
        "Component description",
        "Task demonstrations",
        "Basic instruction",
    ]
    expected_prompt = "\n".join(
        [
            "Use the information below to generate one complete replacement "
            "instruction for the selected Whetstone prompt component.",
            "",
            "## Whetstone prompt-component graph",
            "Component 0: user_prompt_template",
            "Current complete template:",
            "Answer {input}.",
            "Allowed placeholders: {input}",
            "Rendering rules: Substitute the named input without DSPy field "
            "labels.",
            "Example execution: Answer blue. -> blue",
            "",
            "## Program description",
            "Program description 0",
            "",
            "## Selected component",
            "Component id: user_prompt_template",
            "Current complete template:",
            "Answer {input}.",
            "Allowed placeholders: {input}",
            "Rendering rules: Substitute the named input without DSPy field "
            "labels.",
            "Example execution: Answer blue. -> blue",
            "",
            "## Component description",
            "Component description 0",
            "",
            "## Task demonstrations",
            "No task demos provided.",
            "",
            "## Basic instruction",
            "Answer {input}.",
            "",
            "Return only the complete replacement instruction. Preserve "
            "every required native {placeholder} occurrence.",
        ]
    )
    assert first_instruction.prompt == expected_prompt
    assert state.instruction_pools == (
        ("Answer {input}.", "generated-0-1 {input}"),
    )
    assert state.instruction_slots[0].generated_instruction == (
        "generated-0-0 {input}"
    )
    assert state.instruction_slots[0].displaced_by_original is True
    assert len(state.evidence) == 6


def test_demo_rotation_uses_key_presence_and_reference_field_order() -> None:
    demos = (
        Miprov2ComponentDemoCandidates(
            component_id="user_prompt_template",
            demo_sets=(
                Miprov2DemoSet(examples=(_demo("zero"),)),
                Miprov2DemoSet(
                    examples=(
                        _demo("one-true"),
                        _demo(
                            "one-absent",
                            augmented_key_present=False,
                            augmented=None,
                        ),
                    )
                ),
                Miprov2DemoSet(examples=(_demo("two"),)),
                Miprov2DemoSet(examples=(_demo("three"),)),
            ),
        ),
    )
    state = _start(
        demos=demos,
        candidates=4,
        program_aware=False,
        tip_aware=False,
    )
    seen: dict[int, str] = {}
    trace: list[tuple[int | None, str]] = []
    expected_effect_count = len(state.components) * state.proposal_count
    for _transition in range(expected_effect_count + 1):
        planned = plan_next_proposal_request(state)
        state = planned.state
        if planned.request is None:
            assert len(trace) == expected_effect_count, (
                "demo-rotation proposal machine terminated before its exact "
                f"effect count; trace={trace!r}; "
                f"state={state.model_dump_json()}"
            )
            break
        request = planned.request
        trace.append((request.proposal_index, request.effect))
        assert request.effect == "instruction_proposal"
        field_map = {field.name: field.value for field in request.fields}
        assert request.proposal_index is not None
        seen[request.proposal_index] = field_map["Task demonstrations"]
        state = _fold(
            state,
            request,
            f"generated-{request.proposal_index} {{input}}",
        )
    else:
        raise AssertionError(
            "demo-rotation proposal machine did not terminate after its exact "
            f"effect count; trace={trace!r}; state={state.model_dump_json()}"
        )

    assert seen[0] == NO_TASK_DEMOS
    assert seen[1] == (
        "input: in-one-true\noutput: out-one-true\n\n"
        "input: in-two\noutput: out-two\n\n"
        "input: in-three\noutput: out-three\n\n"
    )
    assert "one-absent" not in seen[1]


def test_two_components_are_predictor_major_with_exact_rng_draws() -> None:
    components = (
        _component("first", "First {input}."),
        _component("second", "Second {input}."),
    )
    state = _start(components=components, candidates=2, seed=19)
    state, requests = _finish_proposals(state)

    assert [
        (request.component_index, request.proposal_index, request.effect)
        for request in requests
    ] == [
        (0, 0, "program_description"),
        (0, 0, "component_description"),
        (0, 0, "instruction_proposal"),
        (0, 1, "program_description"),
        (0, 1, "component_description"),
        (0, 1, "instruction_proposal"),
        (1, 0, "program_description"),
        (1, 0, "component_description"),
        (1, 0, "instruction_proposal"),
        (1, 1, "program_description"),
        (1, 1, "component_description"),
        (1, 1, "instruction_proposal"),
    ]

    oracle = random.Random(19)
    keys = tuple(key for key, _ in TIP_TEXTS)
    expected: list[tuple[str, str | int]] = []
    for _ in range(4):
        expected.append(("choice", oracle.choice(keys)))
        expected.append(("randint", oracle.randint(0, 10**9)))
    assert [
        (draw.operation, draw.result) for draw in state.rng_checkpoint.draws
    ] == expected
    assert len({request.rollout_id for request in requests[:3]}) == 1
    assert requests[0].rollout_id != requests[3].rollout_id
    assert state.instruction_pools[0][0] == "First {input}."
    assert state.instruction_pools[1][0] == "Second {input}."


def test_program_failure_skips_component_call_but_keeps_aware_fields() -> None:
    state = _start(candidates=1, seed=2)
    state, program_request = _next(state)
    assert program_request.effect == "program_description"
    state = _fold(state, program_request, failed=True)

    state, instruction_request = _next(state)
    assert instruction_request.effect == "instruction_proposal"
    assert [field.name for field in instruction_request.fields[:4]] == [
        "Whetstone prompt-component graph",
        "Program description",
        "Selected component",
        "Component description",
    ]
    values = {field.name: field.value for field in instruction_request.fields}
    assert values["Program description"] == "Not available"
    assert values["Component description"] == "Not provided"


def test_placeholder_rejection_is_charged_without_entering_pool() -> None:
    state = _start(
        candidates=2,
        program_aware=False,
        tip_aware=False,
    )
    state, first = _next(state)
    state = _fold(state, first, "discarded but invalid")
    state, second = _next(state)
    state = _fold(state, second, "removes input")
    state = plan_next_proposal_request(state).state

    assert state.stage == "complete"
    assert state.effect_count == 2
    assert len(state.evidence) == 2
    assert state.instruction_pools == (("Answer {input}.",),)
    assert state.instruction_slots[0].displaced_by_original is True
    assert state.instruction_slots[1].rejection_reason == (
        "removes required placeholders: input"
    )


def test_response_binding_and_pending_request_are_replay_safe() -> None:
    state = _start(
        candidates=1,
        program_aware=False,
        tip_aware=False,
    )
    planned = plan_next_proposal_request(state)
    replayed = plan_next_proposal_request(planned.state)

    assert replayed == planned
    assert replayed.state.model_dump(mode="json") == (
        Miprov2ProposalState.model_validate_json(
            replayed.state.model_dump_json()
        ).model_dump(mode="json")
    )
    with pytest.raises(ValueError, match="belongs to another request"):
        fold_proposal_response(
            planned.state,
            Miprov2ProposalResponse(
                request_identity_hash="0" * 64,
                text="new {input}",
            ),
        )


def test_instruction_generation_failure_is_terminal() -> None:
    state = _start(
        candidates=1,
        program_aware=False,
        tip_aware=False,
    )
    state, request = _next(state)

    failed = _fold(state, request, failed=True)
    assert failed.stage == "failed"
    assert failed.terminal_failure_response is not None
    restored = Miprov2ProposalState.model_validate_json(
        failed.model_dump_json()
    )
    with pytest.raises(
        Miprov2InstructionGenerationFailed,
        match="scripted failure",
    ) as caught:
        plan_next_proposal_request(restored)
    assert caught.value.state == restored


def test_pending_request_is_reconstructed_and_tampering_fails_closed() -> None:
    state = _start(candidates=1)
    state, request = _next(state)
    payload = state.model_dump(mode="json")
    payload["selected_tip_key"] = (
        "creative" if request.selected_tip_key != "creative" else "simple"
    )

    with pytest.raises(
        ValueError,
        match="does not match reconstructed state",
    ):
        Miprov2ProposalState.model_validate(payload)


def test_shared_rng_transcript_is_appended_without_reset() -> None:
    prior = Miprov2RngDraw(
        ordinal=0,
        phase="bootstrap",
        operation="randint",
        arguments=(1, 3),
        result=2,
    )
    rng = random.Random(7)
    assert rng.randint(1, 3) == prior.result
    checkpoint = Miprov2RngCheckpoint.seeded(7).append(
        rng=rng,
        phase="bootstrap",
        operation="randint",
        arguments=(1, 3),
        result=prior.result,
    )
    state = start_miprov2_proposal(
        bindings=_bindings(),
        components=(_component(),),
        trainset=_dataset(1),
        demo_candidates=None,
        num_candidates=1,
        data_aware=False,
        program_aware=False,
        tip_aware=True,
        rng_checkpoint=checkpoint,
    )
    state, _request = _next(state)

    assert state.rng_checkpoint.draws[0] == prior
    assert [draw.ordinal for draw in state.rng_checkpoint.draws] == [0, 1, 2]
    assert [draw.phase for draw in state.rng_checkpoint.draws[1:]] == [
        "proposal",
        "proposal",
    ]


def test_bootstrap_demo_bridge_preserves_order_fields_and_key_presence() -> (
    None
):
    assert MIPROV2_DEMO_BRIDGE_VERSION == "whetstone_component_demo_bridge/v1"
    labeled_task = LabeledTaskDemo(
        source_task_identity=_identity("task"),
        inputs_by_component={"user_prompt_template": {"input": "question"}},
        outputs_by_component={"user_prompt_template": {"output": "answer"}},
    )
    labeled = labeled_task.for_component("user_prompt_template")
    bootstrapped = ComponentDemo(
        component_id="user_prompt_template",
        source_kind=DemoSourceKind.BOOTSTRAPPED,
        inputs={"input": "boot-question"},
        outputs={"output": "boot-answer"},
        augmented=True,
        source_task_identity=_identity("boot-task"),
        source_rollout_identity=_identity("rollout"),
        source_trace_identity=_identity("trace"),
        source_output_identity=_identity("output"),
        source_score_identity=_identity("score"),
        source_trace_index=0,
        score=1.0,
        acceptance_identity_hash=_identity("acceptance"),
    )
    candidate_sets = (
        ComponentDemoSet(
            candidate_seed=-3,
            components=(
                ComponentDemoSequence(
                    component_id="user_prompt_template",
                    demos=(labeled,),
                ),
            ),
        ),
        ComponentDemoSet(
            candidate_seed=-1,
            components=(
                ComponentDemoSequence(
                    component_id="user_prompt_template",
                    demos=(bootstrapped,),
                ),
            ),
        ),
    )

    (bridged,) = proposal_candidates_from_demo_sets(
        candidate_sets,
        components=(_component(),),
        component_field_order={
            "user_prompt_template": ("output", "input"),
        },
    )

    assert len(bridged.demo_sets) == 2
    raw, augmented = (demo_set.examples[0] for demo_set in bridged.demo_sets)
    assert [field.name for field in raw.fields] == ["output", "input"]
    assert [field.value for field in raw.fields] == ["answer", "question"]
    assert raw.augmented_key_present is False
    assert raw.augmented is None
    assert augmented.augmented_key_present is True
    assert augmented.augmented is True


def test_crash_roundtrip_after_every_proposal_effect() -> None:
    state = _start(
        dataset_count=12,
        candidates=1,
        data_aware=True,
        batch_size=10,
    )
    dataset_descriptor_count = min(
        (len(state.trainset) + state.view_data_batch_size - 1)
        // state.view_data_batch_size,
        10,
    )
    expected_effect_count = (
        dataset_descriptor_count
        + 1
        + len(state.components) * state.proposal_count * 3
    )
    trace: list[str] = []
    for _transition in range(expected_effect_count + 1):
        planned = plan_next_proposal_request(state)
        state = Miprov2ProposalState.model_validate_json(
            planned.state.model_dump_json()
        )
        replayed = plan_next_proposal_request(state)
        assert replayed.request == planned.request
        if replayed.request is None:
            assert len(trace) == expected_effect_count, (
                "crash-roundtrip proposal machine terminated before its exact "
                f"effect count; trace={trace!r}; "
                f"state={state.model_dump_json()}"
            )
            break
        request = replayed.request
        trace.append(request.effect)
        response_text = {
            "dataset_initial": "initial observation",
            "dataset_followup": "followup observation",
            "dataset_final": "Summary: dataset",
            "program_description": "Program: description",
            "component_description": "component description",
            "instruction_proposal": "Instruction: replacement {input}",
        }[request.effect]
        state = _fold(state, request, response_text)
        state = Miprov2ProposalState.model_validate_json(
            state.model_dump_json()
        )
    else:
        raise AssertionError(
            "crash-roundtrip proposal machine did not terminate after its "
            f"exact effect count; trace={trace!r}; "
            f"state={state.model_dump_json()}"
        )

    assert state.stage == "complete"
    assert state.instruction_pools == (("Answer {input}.",),)


def test_outer_bindings_change_proposal_request_identity() -> None:
    first = _start(
        candidates=1,
        program_aware=False,
        tip_aware=False,
    )
    changed_bindings = _bindings().model_copy(
        update={"prompt_route_identity_hash": _identity("another-route")}
    )
    second = first.model_copy(update={"bindings": changed_bindings})

    first_request = plan_next_proposal_request(first).request
    second_request = plan_next_proposal_request(second).request

    assert first_request is not None
    assert second_request is not None
    assert first_request.prompt == second_request.prompt
    assert first_request.identity_hash != second_request.identity_hash


def test_bridge_version_and_augmented_marker_are_identity_bound() -> None:
    bindings = _bindings()
    assert bindings.demo_bridge_version == MIPROV2_DEMO_BRIDGE_VERSION

    candidate = Miprov2ComponentDemoCandidates(
        component_id="user_prompt_template",
        demo_sets=(),
    )
    assert candidate.bridge_version == MIPROV2_DEMO_BRIDGE_VERSION
    assert _demo("false-marker", augmented=False).augmented is False
    with pytest.raises(ValueError, match="presence must match"):
        _demo(
            "bad-presence",
            augmented_key_present=False,
            augmented=True,
        )


def test_canonical_replay_rejects_slot_and_pool_tampering() -> None:
    complete, _requests = _finish_proposals(
        _start(
            candidates=2,
            program_aware=False,
            tip_aware=True,
        )
    )
    assert complete.stage == "complete"
    slots = list(complete.instruction_slots)
    second = slots[1].model_copy(
        update={
            "generated_instruction": "forged {input}",
            "pool_instruction": "forged {input}",
        }
    )
    slots[1] = second
    forged = complete.model_copy(
        update={
            "instruction_slots": tuple(slots),
            "instruction_pools": (
                (complete.instruction_pools[0][0], "forged {input}"),
            ),
        }
    )
    with pytest.raises(ValueError, match="canonical evidence replay"):
        Miprov2ProposalState.model_validate(forged.model_dump(mode="json"))


def test_canonical_replay_rejects_rng_and_evidence_projection_tampering() -> (
    None
):
    complete, _requests = _finish_proposals(
        _start(
            candidates=1,
            program_aware=False,
            tip_aware=True,
        )
    )
    wrong_rng = complete.model_copy(
        update={"rng_checkpoint": complete.initial_rng_checkpoint}
    )
    with pytest.raises(ValueError, match="canonical evidence replay"):
        Miprov2ProposalState.model_validate(wrong_rng.model_dump(mode="json"))

    item = complete.evidence[0]
    forged_item = item.model_copy(update={"parsed_text": "forged {input}"})
    forged_evidence = complete.model_copy(update={"evidence": (forged_item,)})
    with pytest.raises(ValueError, match="decision does not match"):
        Miprov2ProposalState.model_validate(
            forged_evidence.model_dump(mode="json")
        )
