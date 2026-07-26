from __future__ import annotations

import inspect
import math
import random
from typing import Any

import pytest

from tests.optimization.support import (
    FULL_A,
    FULL_B,
    FULL_C,
    FULL_D,
    eval_config,
)
from whetstone.lm.boundary import PlainPromptAdapter
from whetstone.optimization.miprov2_control import (
    MIPROV2_ALGORITHM_VERSION,
    MIPROV2_OPTUNA_VERSION,
    MIPROV2_PHASE_SCHEMA_MANIFEST,
    MIPROV2_PROPOSER_OUTPUT_PARSER_SCHEMA,
    MIPROV2_REFERENCE_COMMIT,
    MIPROV2_RESULT_SCHEMA,
    MIPROV2_STATE_SCHEMA,
    MIPROV2_TEACHER_COMPILED_PAYLOAD_FIELD,
    Miprov2ComponentSpec,
    Miprov2Control,
    Miprov2InjectedDefaults,
    Miprov2ProgramLayout,
    configure_miprov2,
)
from whetstone.optimization.proposer import (
    ProposerConfig,
    prompt_adapter_identity_hash,
)
from whetstone.optimization.schema import (
    Candidate,
    candidate_reference,
    eval_config_reference,
)

TASKS = tuple(f"{index:064x}" for index in range(1, 1206))


def _candidate(
    candidate_id: str = "base",
    *,
    base_ref: str = "route-a",
    payload: dict[str, Any] | None = None,
):
    return candidate_reference(
        Candidate(
            candidate_id=candidate_id,
            base_ref=base_ref,
            payload=payload
            if payload is not None
            else {"user_prompt_template": "Answer {input}."},
        )
    )


def _prompt_model(*, temperature: float = 1.0) -> ProposerConfig:
    return ProposerConfig(
        provider_call_config_ref="provider://mipro-proposer",
        provider_call_config_hash=FULL_A,
        temperature=temperature,
    )


def _defaults(
    *,
    prompt_model: ProposerConfig | None = None,
    prompt_adapter: PlainPromptAdapter | None = None,
    bootstrap_eval_hash: str = FULL_A,
    validation_eval_hash: str = FULL_C,
    metric_authority: bool = True,
) -> Miprov2InjectedDefaults:
    return Miprov2InjectedDefaults(
        prompt_model=(
            _prompt_model() if prompt_model is None else prompt_model
        ),
        bootstrap_eval_source=eval_config_reference(
            eval_config(bootstrap_eval_hash)
        ),
        validation_eval_source=eval_config_reference(
            eval_config(validation_eval_hash)
        ),
        reward_policy_hash=FULL_D,
        provider_execution_policy_hash=FULL_B,
        task_model_identity_hash=FULL_C,
        prompt_adapter=prompt_adapter or PlainPromptAdapter(),
        max_errors=17,
        validation_eval_source_is_metric_authority=metric_authority,
    )


def _configure(**overrides: Any):
    values: dict[str, Any] = {
        "base_candidate": _candidate(),
        "trainset": TASKS[:5],
        "defaults": _defaults(),
    }
    values.update(overrides)
    return configure_miprov2(**values)


def _layout(
    *component_specs: Miprov2ComponentSpec,
    layout_id: str = "test-layout",
) -> Miprov2ProgramLayout:
    return Miprov2ProgramLayout(
        layout_id=layout_id,
        component_specs=component_specs,
    )


def test_public_signature_preserves_dspy_defaults() -> None:
    signature = inspect.signature(configure_miprov2)
    expected = {
        "metric": None,
        "prompt_model": None,
        "task_model": None,
        "teacher_settings": None,
        "max_bootstrapped_demos": 4,
        "max_labeled_demos": 4,
        "auto": "light",
        "num_candidates": None,
        "num_threads": None,
        "max_errors": None,
        "seed": 9,
        "init_temperature": 1.0,
        "verbose": False,
        "track_stats": True,
        "log_dir": None,
        "metric_threshold": None,
        "teacher": None,
        "teacher_compiled": None,
        "valset": None,
        "num_trials": None,
        "run_max_bootstrapped_demos": None,
        "run_max_labeled_demos": None,
        "run_seed": None,
        "minibatch": True,
        "minibatch_size": 35,
        "minibatch_full_eval_steps": 5,
        "program_aware_proposer": True,
        "data_aware_proposer": True,
        "view_data_batch_size": 10,
        "tip_aware_proposer": True,
        "fewshot_aware_proposer": True,
        "requires_permission_to_run": None,
        "provide_traceback": None,
        "program_layout": None,
    }
    for name, default in expected.items():
        assert signature.parameters[name].default == default


def test_default_control_resolves_light_mode_and_dataset_split() -> None:
    control = _configure()
    pre_auto = TASKS[1:5]
    rng = random.Random(9)
    indices = tuple(rng.sample(range(4), 4))

    assert control.trainset_task_identities == (TASKS[0],)
    assert control.auto_validation_sample_indices == indices
    assert control.valset_task_identities == tuple(
        pre_auto[index] for index in indices
    )
    assert control.num_instruct_candidates == 3
    assert control.num_fewshot_candidates == 6
    assert control.num_trials == int(max(4 * math.log2(6), 9))
    assert control.minibatch is False
    assert control.max_errors == 17
    assert control.component_ids == ("user_prompt_template",)
    assert len(control.component_specs) == 1
    assert control.component_specs[0].candidate_field == (
        "user_prompt_template"
    )
    assert control.component_specs[0].required_placeholders == ("input",)
    assert (
        control.component_specs[0].prompt_format_identity_hash
        == control.prompt_adapter_identity_hash
    )
    component_payload = control.identity_payload()["program_layout"]["config"][
        "component_specs"
    ][0]
    assert component_payload["identity_hash"] == (
        control.component_specs[0].identity_hash()
    )
    assert component_payload["config"] == (
        control.component_specs[0].identity_payload()
    )


def test_seed_zero_falls_back_to_constructor_seed_before_sampling() -> None:
    control = _configure(seed=23, run_seed=0)
    expected_rng = random.Random(23)

    assert control.seed == 23
    assert control.auto_validation_sample_indices == tuple(
        expected_rng.sample(range(4), 4)
    )


def test_explicit_nonzero_run_seed_overrides_constructor_seed() -> None:
    control = _configure(seed=23, run_seed=7)

    assert control.seed == 7
    assert control.auto_validation_sample_indices == tuple(
        random.Random(7).sample(range(4), 4)
    )


@pytest.mark.parametrize(
    ("mode", "n", "val_size"),
    [
        ("light", 6, 100),
        ("medium", 12, 300),
        ("heavy", 18, 1000),
    ],
)
def test_auto_modes_match_reference_settings(
    mode: str,
    n: int,
    val_size: int,
) -> None:
    valset = TASKS[100:1200]
    control = _configure(
        auto=mode,
        valset=valset,
        minibatch=False,
    )

    assert len(control.valset_task_identities) == val_size
    assert control.minibatch is True
    assert control.num_instruct_candidates == int(n * 0.5)
    assert control.num_fewshot_candidates == n
    assert control.num_trials == int(max(4 * math.log2(n), 1.5 * n))


def test_zero_shot_auto_uses_all_n_instruction_candidates() -> None:
    base = _candidate(
        payload={
            "first_prompt": "Use {input} and repeat {input}.",
            "second_prompt": "Check {context}.",
        }
    )
    adapter_hash = prompt_adapter_identity_hash(PlainPromptAdapter())
    specs = (
        Miprov2ComponentSpec(
            component_id="first",
            candidate_field="first_prompt",
            prompt_format_identity_hash=adapter_hash,
            required_placeholders=("input", "input"),
        ),
        Miprov2ComponentSpec(
            component_id="second",
            candidate_field="second_prompt",
            prompt_format_identity_hash=adapter_hash,
            required_placeholders=("context",),
        ),
    )
    control = _configure(
        base_candidate=base,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        program_layout=_layout(*specs),
    )

    assert control.component_ids == ("first", "second")
    assert control.component_specs == specs
    assert control.zeroshot_opt is True
    assert control.num_instruct_candidates == 6
    assert control.num_fewshot_candidates == 6
    assert control.num_trials == int(max(4 * math.log2(6), 9))


def test_manual_mode_preserves_explicit_validation_order() -> None:
    control = _configure(
        auto=None,
        num_candidates=5,
        num_trials=11,
        valset=(TASKS[12], TASKS[10], TASKS[11]),
        minibatch=False,
    )

    assert control.trainset_task_identities == (*TASKS[:5],)
    assert control.valset_task_identities == (
        TASKS[12],
        TASKS[10],
        TASKS[11],
    )
    assert control.auto_validation_sample_indices is None
    assert control.num_instruct_candidates == 5
    assert control.num_fewshot_candidates == 5
    assert control.num_trials == 11
    assert control.minibatch is False


def test_invalid_auto_precedes_compile_permission_and_dataset_errors() -> None:
    with pytest.raises(ValueError, match="Invalid value for auto"):
        _configure(
            auto="fast",
            requires_permission_to_run=True,
            trainset=(),
        )


def test_permission_error_precedes_manual_and_dataset_errors() -> None:
    with pytest.raises(ValueError, match="User confirmation is removed"):
        _configure(
            auto=None,
            requires_permission_to_run=True,
            trainset=(),
        )


def test_manual_recommendation_error_precedes_missing_dataset_error() -> None:
    with pytest.raises(
        ValueError,
        match=r"num_trials must also be provided.*~9",
    ):
        _configure(
            auto=None,
            num_candidates=5,
            trainset=(),
        )


def test_manual_missing_candidate_error_matches_reference_precedence() -> None:
    with pytest.raises(
        ValueError,
        match="num_candidates must also be provided",
    ):
        _configure(
            auto=None,
            num_trials=5,
            trainset=(),
        )


def test_auto_conflict_precedes_missing_dataset_error() -> None:
    with pytest.raises(
        ValueError,
        match="num_candidates and num_trials cannot be set",
    ):
        _configure(num_trials=5, trainset=())


def test_dataset_validation_matches_reference() -> None:
    with pytest.raises(ValueError, match="Trainset cannot be empty"):
        _configure(trainset=())
    with pytest.raises(ValueError, match="at least 2 examples"):
        _configure(trainset=(TASKS[0],))
    with pytest.raises(ValueError, match="Validation set"):
        _configure(valset=())


def test_minibatch_size_check_runs_only_when_minibatching() -> None:
    with pytest.raises(
        ValueError,
        match=r"Minibatch size cannot exceed.*60",
    ):
        _configure(
            auto=None,
            num_candidates=2,
            num_trials=2,
            valset=TASKS[100:160],
            minibatch=True,
            minibatch_size=61,
        )

    control = _configure(
        auto=None,
        num_candidates=2,
        num_trials=2,
        valset=(TASKS[100],),
        minibatch=False,
        minibatch_size=100,
    )
    assert control.minibatch_size == 100


def test_false_permission_records_reference_deprecation_warning() -> None:
    control = _configure(requires_permission_to_run=False)

    assert control.deprecated_warnings == (
        "'requires_permission_to_run' is deprecated and will be removed in "
        "a future version.",
    )


def test_reference_dependency_and_effect_authorities_are_bound() -> None:
    control = _configure()
    payload = control.identity_payload()

    assert payload["algorithm_version"] == MIPROV2_ALGORITHM_VERSION
    assert payload["reference_commit"] == MIPROV2_REFERENCE_COMMIT
    assert payload["optuna_version"] == MIPROV2_OPTUNA_VERSION
    assert payload["task_model_identity_hash"] == FULL_C
    assert payload["reward_policy_hash"] == FULL_D
    assert payload["provider_execution_policy_hash"] == FULL_B
    assert payload["source_trainset_task_identities"] == [
        *TASKS[:5],
    ]

    changed_seed = _configure(seed=10)
    changed_tasks = _configure(trainset=(*TASKS[:4], TASKS[20]))
    assert control.identity_hash() != changed_seed.identity_hash()
    assert control.identity_hash() != changed_tasks.identity_hash()


def test_bootstrap_and_validation_sources_are_independent() -> None:
    control = _configure()
    metric_override = eval_config_reference(eval_config(FULL_B))
    changed_validation = _configure(metric=metric_override)
    changed_bootstrap = _configure(
        defaults=_defaults(bootstrap_eval_hash=FULL_B)
    )

    assert changed_validation.bootstrap_eval_source == (
        control.bootstrap_eval_source
    )
    assert changed_validation.validation_eval_source == metric_override
    assert changed_validation.identity_hash() != control.identity_hash()
    assert changed_bootstrap.validation_eval_source == (
        control.validation_eval_source
    )
    assert changed_bootstrap.identity_hash() != control.identity_hash()


def test_dataset_rng_replay_resumes_after_auto_sampling() -> None:
    control = _configure(seed=31)
    replayed = control.replay_dataset_rng()
    expected = random.Random(31)
    expected.sample(range(4), 4)

    assert replayed.randint(0, 10**9) == expected.randint(0, 10**9)


def test_explicit_equal_teacher_matches_default_teacher_identity() -> None:
    base = _candidate()
    implicit = _configure(base_candidate=base)
    explicit = _configure(base_candidate=base, teacher=base)

    assert implicit.identity_hash() == explicit.identity_hash()


def test_distinct_teacher_route_with_matching_component_structure() -> None:
    teacher = _candidate(
        "teacher",
        base_ref="teacher-route",
        payload={"user_prompt_template": "Teach from {input}."},
    )

    control = _configure(teacher=teacher)

    assert control.teacher_candidate == teacher
    assert control.teacher_candidate.record.base_ref == "teacher-route"


def test_compiled_zero_demo_teacher_is_explicitly_bound() -> None:
    teacher = _candidate(
        "teacher",
        base_ref="teacher-route",
        payload={
            "user_prompt_template": "Teach from {input}.",
            MIPROV2_TEACHER_COMPILED_PAYLOAD_FIELD: True,
        },
    )

    control = _configure(teacher=teacher)

    assert control.teacher_compiled is True
    assert control.identity_payload()["teacher_compiled"] is True


def test_zero_demo_teacher_defaults_to_uncompiled() -> None:
    control = _configure(
        teacher=_candidate(
            "teacher",
            base_ref="teacher-route",
            payload={"user_prompt_template": "Teach from {input}."},
        )
    )

    assert control.teacher_compiled is False


def test_teacher_compiled_conflict_is_rejected() -> None:
    teacher = _candidate(
        "teacher",
        base_ref="teacher-route",
        payload={
            "user_prompt_template": "Teach from {input}.",
            MIPROV2_TEACHER_COMPILED_PAYLOAD_FIELD: True,
        },
    )

    with pytest.raises(ValueError, match="conflicts"):
        _configure(teacher=teacher, teacher_compiled=False)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({}, "must be a non-empty string"),
        ({"user_prompt_template": ""}, "must be a non-empty string"),
        ({"user_prompt_template": 7}, "must be a non-empty string"),
        (
            {"user_prompt_template": "Teach from {different}."},
            "placeholder structure",
        ),
    ],
)
def test_teacher_component_structure_mismatch_is_rejected(
    payload: dict[str, Any],
    match: str,
) -> None:
    teacher = _candidate(
        "teacher",
        base_ref="teacher-route",
        payload=payload,
    )

    with pytest.raises(ValueError, match=match):
        _configure(teacher=teacher)


def test_component_specs_reject_spoofed_candidate_structure() -> None:
    adapter_hash = prompt_adapter_identity_hash(PlainPromptAdapter())
    spec = Miprov2ComponentSpec(
        component_id="invented",
        candidate_field="missing_prompt",
        prompt_format_identity_hash=adapter_hash,
        required_placeholders=("input",),
    )

    with pytest.raises(ValueError, match=r"missing_prompt.*non-empty string"):
        _configure(program_layout=_layout(spec))


def test_component_specs_reject_duplicate_ids_and_candidate_fields() -> None:
    adapter_hash = prompt_adapter_identity_hash(PlainPromptAdapter())
    first = Miprov2ComponentSpec(
        component_id="same",
        candidate_field="first",
        prompt_format_identity_hash=adapter_hash,
        required_placeholders=("input",),
    )
    duplicate_id = Miprov2ComponentSpec(
        component_id="same",
        candidate_field="second",
        prompt_format_identity_hash=adapter_hash,
        required_placeholders=("input",),
    )
    duplicate_field = Miprov2ComponentSpec(
        component_id="other",
        candidate_field="first",
        prompt_format_identity_hash=adapter_hash,
        required_placeholders=("input",),
    )

    with pytest.raises(ValueError, match="component_ids must be unique"):
        _layout(first, duplicate_id)
    with pytest.raises(ValueError, match="candidate fields must be unique"):
        _layout(first, duplicate_field)


def test_component_spec_binds_prompt_format_and_placeholder_multiplicity() -> (
    None
):
    adapter_hash = prompt_adapter_identity_hash(PlainPromptAdapter())
    wrong_format = Miprov2ComponentSpec(
        component_id="prompt",
        candidate_field="user_prompt_template",
        prompt_format_identity_hash=FULL_B,
        required_placeholders=("input",),
    )
    wrong_multiplicity = Miprov2ComponentSpec(
        component_id="prompt",
        candidate_field="user_prompt_template",
        prompt_format_identity_hash=adapter_hash,
        required_placeholders=("input", "input"),
    )

    with pytest.raises(ValueError, match="prompt format conflicts"):
        _configure(program_layout=_layout(wrong_format))
    with pytest.raises(ValueError, match="placeholder structure"):
        _configure(program_layout=_layout(wrong_multiplicity))


def test_whetstone_temperature_and_layout_safety_labels() -> None:
    with pytest.raises(ValueError, match=r"Whetstone safety.*temperature"):
        _configure(
            prompt_model=_prompt_model(temperature=0.5),
        )
    with pytest.raises(
        ValueError,
        match=r"Whetstone safety.*layout components",
    ):
        _layout()


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"max_bootstrapped_demos": -1}, "max_bootstrapped_demos"),
        ({"max_labeled_demos": -1}, "max_labeled_demos"),
        ({"run_max_bootstrapped_demos": -1}, "run_max_bootstrapped_demos"),
        ({"run_max_labeled_demos": -1}, "run_max_labeled_demos"),
        (
            {
                "auto": None,
                "num_candidates": 0,
                "num_trials": 1,
                "minibatch": False,
            },
            "num_candidates",
        ),
        (
            {
                "auto": None,
                "num_candidates": 2,
                "num_trials": 0,
                "minibatch": False,
            },
            "num_trials",
        ),
        ({"num_threads": 0}, "num_threads"),
        ({"max_errors": 0}, "max_errors"),
        ({"minibatch_size": 0}, "minibatch_size"),
        ({"minibatch_full_eval_steps": 0}, "minibatch_full_eval_steps"),
        ({"view_data_batch_size": 0}, "view_data_batch_size"),
    ],
)
def test_whetstone_integer_safety_boundaries(
    overrides: dict[str, Any],
    field: str,
) -> None:
    with pytest.raises(ValueError, match=rf"Whetstone safety: {field}"):
        _configure(**overrides)


@pytest.mark.parametrize("field", ["init_temperature", "metric_threshold"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_whetstone_numeric_controls_must_be_finite(
    field: str,
    value: float,
) -> None:
    overrides: dict[str, Any] = {field: value}
    if field == "init_temperature":
        overrides["prompt_model"] = _prompt_model(temperature=value)
    with pytest.raises(ValueError, match=rf"Whetstone safety: {field}"):
        _configure(**overrides)


def test_zero_demo_maxima_arbitrary_seed_and_zero_threshold_are_allowed() -> (
    None
):
    control = _configure(
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        seed=-123,
        metric_threshold=0.0,
    )

    assert control.zeroshot_opt is True
    assert control.seed == -123
    assert control.metric_threshold == 0.0


def test_task_identities_must_be_full_hashes() -> None:
    with pytest.raises(
        ValueError,
        match="source_trainset_task_identities must be a full",
    ):
        _configure(trainset=("not-a-task-hash", TASKS[1]))


def test_falsey_task_model_uses_injected_default() -> None:
    control = _configure(task_model="")

    assert control.task_model_identity_hash == FULL_C


def test_falsey_prompt_model_uses_injected_default() -> None:
    class FalseyProposerConfig(ProposerConfig):
        def __bool__(self) -> bool:
            return False

    explicit = FalseyProposerConfig(
        provider_call_config_ref="provider://falsey",
        provider_call_config_hash=FULL_B,
        temperature=1.0,
    )
    control = _configure(
        prompt_model=explicit,
        defaults=_defaults(prompt_model=_prompt_model()),
    )

    assert control.prompt_model.provider_call_config_ref == (
        "provider://mipro-proposer"
    )
    assert control.prompt_model.provider_call_config_hash == FULL_A


def test_metric_requires_an_explicit_or_injected_authority() -> None:
    with pytest.raises(ValueError, match="metric is required"):
        _configure(defaults=_defaults(metric_authority=False))

    metric = eval_config_reference(eval_config(FULL_B))
    explicit = _configure(
        metric=metric,
        defaults=_defaults(metric_authority=False),
    )
    injected = _configure()

    assert explicit.validation_eval_source == metric
    assert explicit.metric_authority == "explicit"
    assert injected.metric_authority == "injected_default"


@pytest.mark.parametrize("value", [0, 1, "", object()])
def test_permission_requires_an_actual_boolean(value: Any) -> None:
    with pytest.raises(
        ValueError,
        match="requires_permission_to_run must be a boolean",
    ):
        _configure(requires_permission_to_run=value)


def test_reference_errors_precede_whetstone_safety_checks() -> None:
    with pytest.raises(ValueError, match="num_trials must also be provided"):
        _configure(
            auto=None,
            num_candidates=5,
            max_bootstrapped_demos=-1,
            trainset=(),
        )
    with pytest.raises(
        ValueError,
        match="num_candidates and num_trials cannot be set",
    ):
        _configure(
            num_trials=5,
            max_bootstrapped_demos=-1,
            trainset=(),
        )
    with pytest.raises(ValueError, match="Trainset cannot be empty"):
        _configure(
            max_bootstrapped_demos=-1,
            trainset=(),
        )
    with pytest.raises(ValueError, match="Minibatch size cannot exceed"):
        _configure(
            auto=None,
            num_candidates=2,
            num_trials=2,
            base_candidate=_candidate(payload={"fixed": "not-a-prompt"}),
            valset=TASKS[100:160],
            minibatch_size=61,
        )


def test_program_layout_is_an_ordered_identity_authority() -> None:
    adapter_hash = prompt_adapter_identity_hash(PlainPromptAdapter())
    specs = (
        Miprov2ComponentSpec(
            component_id="first",
            candidate_field="first_prompt",
            prompt_format_identity_hash=adapter_hash,
            required_placeholders=("input",),
        ),
        Miprov2ComponentSpec(
            component_id="second",
            candidate_field="second_prompt",
            prompt_format_identity_hash=adapter_hash,
            required_placeholders=("context",),
        ),
    )
    reversed_specs = tuple(reversed(specs))
    base = _candidate(
        payload={
            "first_prompt": "First {input}.",
            "second_prompt": "Second {context}.",
        }
    )
    forward = _configure(
        base_candidate=base,
        program_layout=_layout(*specs),
    )
    reverse = _configure(
        base_candidate=base,
        program_layout=_layout(*reversed_specs),
    )

    assert forward.component_specs == specs
    assert forward.program_layout.identity_hash() != (
        reverse.program_layout.identity_hash()
    )
    assert forward.identity_hash() != reverse.identity_hash()


def test_control_identity_binds_all_algorithm_schema_authorities() -> None:
    payload = _configure().identity_payload()

    assert payload["proposer_output_parser"]["schema"] == (
        MIPROV2_PROPOSER_OUTPUT_PARSER_SCHEMA
    )
    assert payload["phase_schema_manifest"] == [
        {"schema": schema, "schema_version": version}
        for schema, version in MIPROV2_PHASE_SCHEMA_MANIFEST
    ]
    assert payload["state_schema"]["schema"] == MIPROV2_STATE_SCHEMA
    assert payload["result_schema"]["schema"] == MIPROV2_RESULT_SCHEMA


def test_nested_control_inputs_are_snapshotted_and_frozen() -> None:
    metadata = {"labels": ["original"]}
    teacher_settings = {"route": {"tags": ["teacher"]}}
    base = _candidate(
        payload={
            "user_prompt_template": "Answer {input}.",
            "metadata": metadata,
        }
    )
    control = _configure(
        base_candidate=base,
        teacher_settings=teacher_settings,
    )
    identity = control.identity_hash()

    metadata["labels"].append("caller-mutation")
    teacher_settings["route"]["tags"].append("caller-mutation")
    assert control.identity_hash() == identity

    with pytest.raises(TypeError, match="immutable"):
        control.base_candidate.record.payload["metadata"]["labels"].append(
            "control-mutation"
        )
    with pytest.raises(TypeError, match="immutable"):
        control.teacher_settings["route"]["tags"].append("control-mutation")
    assert control.identity_hash() == identity


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("trainset_task_identities", (TASKS[20],)),
        ("valset_task_identities", (TASKS[20],)),
        ("auto_validation_sample_indices", ()),
        ("num_instruct_candidates", 4),
        ("num_fewshot_candidates", 5),
        ("num_trials", 999),
        ("minibatch", True),
        ("zeroshot_opt", True),
    ],
)
def test_deserialization_rejects_tampered_auto_derivations(
    field: str,
    replacement: Any,
) -> None:
    payload = _configure().model_dump(mode="python")
    payload[field] = replacement

    with pytest.raises(ValueError, match=r"resolved MIPROv2|resolved auto"):
        Miprov2Control.model_validate(payload)


def test_deserialization_rejects_manual_and_source_dataset_tampering() -> None:
    manual = _configure(
        auto=None,
        num_candidates=3,
        num_trials=4,
        valset=TASKS[10:13],
        minibatch=False,
    )

    missing_candidates = manual.model_dump(mode="python")
    missing_candidates["num_candidates"] = None
    with pytest.raises(ValueError, match="requires num_candidates"):
        Miprov2Control.model_validate(missing_candidates)

    sampled_manual = manual.model_dump(mode="python")
    sampled_manual["auto_validation_sample_indices"] = (0,)
    with pytest.raises(ValueError, match="cannot carry auto sample"):
        Miprov2Control.model_validate(sampled_manual)

    empty_source = manual.model_dump(mode="python")
    empty_source["source_trainset_task_identities"] = ()
    with pytest.raises(ValueError, match="source trainset cannot be empty"):
        Miprov2Control.model_validate(empty_source)


def test_persisted_optimizer_identity_conflict_is_rejected() -> None:
    control = _configure()

    control.require_identity_hash(control.identity_hash())
    with pytest.raises(ValueError, match="conflicts"):
        control.require_identity_hash(FULL_A)
