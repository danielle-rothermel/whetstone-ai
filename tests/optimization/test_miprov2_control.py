from __future__ import annotations

import inspect
import math
from typing import Any

import pytest
from dr_providers import openrouter_chat_config

from tests.optimization.support import (
    FULL_A,
    FULL_B,
    FULL_C,
    candidate,
    eval_config,
    internal_reward_policy,
    python_format_contract,
)
from whetstone.evaluation_role import EvaluationRole
from whetstone.lm.boundary import PlainPromptAdapter
from whetstone.optimization.identity import IdentityRef, typed_ref_for_record
from whetstone.optimization.miprov2_control import (
    MIPROV2_ALGORITHM_VERSION,
    MIPROV2_OPTUNA_VERSION,
    MIPROV2_PROMPT_FORMAT_ADAPTER_VERSION,
    MIPROV2_REFERENCE_COMMIT,
    Miprov2AutoMode,
    Miprov2ComponentSpec,
    Miprov2InjectedDefaults,
    Miprov2ProgramLayout,
    configure_miprov2,
)
from whetstone.optimization.proposer import (
    ProposerConfig,
    prompt_adapter_identity_hash,
)
from whetstone.optimization.schema import (
    EvaluationBinding,
    candidate_reference,
    eval_config_reference,
)

TASKS = tuple(f"{index:064x}" for index in range(1, 8))


def _defaults() -> Miprov2InjectedDefaults:
    provider = openrouter_chat_config(model="proposal-model")
    validation = eval_config_reference(eval_config(FULL_B))
    return Miprov2InjectedDefaults(
        prompt_model=ProposerConfig(
            provider_call_config=IdentityRef(
                record_ref=typed_ref_for_record(
                    "dr_providers.provider_call_config",
                    provider.model_dump(mode="json"),
                ),
                identity_hash=provider.identity_hash,
            )
        ),
        bootstrap_eval_source=eval_config_reference(eval_config(FULL_A)),
        validation_eval_source=validation,
        reward_policy=internal_reward_policy(),
        evaluation_binding=EvaluationBinding(
            schema_version=2,
            eval_config=validation,
            role=EvaluationRole.INTERNAL,
            campaign="miprov2-control-test",
        ),
        provider_execution_policy_hash=FULL_B,
        task_model_identity_hash=FULL_C,
        prompt_adapter=PlainPromptAdapter(),
        template_render_contract=python_format_contract(),
        max_errors=3,
        validation_eval_source_is_metric_authority=True,
    )


def _configure(**updates):
    values: dict[str, Any] = {
        "base_candidate": candidate_reference(
            candidate("base", text="Answer {query}.")
        ),
        "trainset": TASKS[:4],
        "valset": TASKS[4:],
        "auto": None,
        "num_candidates": 2,
        "num_trials": 2,
        "minibatch": False,
        "defaults": _defaults(),
    }
    values.update(updates)
    return configure_miprov2(**values)


def test_public_surface_keeps_frozen_versions() -> None:
    assert MIPROV2_ALGORITHM_VERSION == "dspy_miprov2/v2"
    assert MIPROV2_REFERENCE_COMMIT == (
        "6f68dcdb3ef46d70bf0c12596699ebc44e82d6b0"
    )
    assert MIPROV2_OPTUNA_VERSION == "4.8.0"
    assert MIPROV2_PROMPT_FORMAT_ADAPTER_VERSION


def test_default_layout_admits_one_generate_component() -> None:
    control = _configure()

    assert control.component_ids == ("generate",)
    assert control.base_candidate.record.payload["user_prompt_template"] == (
        "Answer {query}."
    )
    assert control.reference().identity_hash == control.identity_hash()


def test_ed1_layout_admits_exact_encoder_component() -> None:
    adapter_hash = prompt_adapter_identity_hash(PlainPromptAdapter())
    layout = Miprov2ProgramLayout(
        layout_id="ed1",
        component_specs=(
            Miprov2ComponentSpec(
                component_id="encode",
                prompt_format_identity_hash=adapter_hash,
            ),
        ),
    )

    assert _configure(program_layout=layout).component_ids == ("encode",)


@pytest.mark.parametrize(
    "component_ids",
    (("decode",), ("generate", "encode")),
)
def test_layout_rejects_nonoptimizable_or_multiple_components(
    component_ids: tuple[str, ...],
) -> None:
    adapter_hash = prompt_adapter_identity_hash(PlainPromptAdapter())

    with pytest.raises(ValueError, match=r"optimizes|exactly one"):
        Miprov2ProgramLayout(
            layout_id="invalid",
            component_specs=tuple(
                Miprov2ComponentSpec(
                    component_id=component_id,
                    prompt_format_identity_hash=adapter_hash,
                )
                for component_id in component_ids
            ),
        )


def test_control_preserves_non_prompt_payload_data() -> None:
    base = type(candidate("base")).model_validate(
        {
            **candidate("base", text="Answer {query}.").model_dump(
                mode="json"
            ),
            "payload": {
                "user_prompt_template": "Answer {query}.",
                "fixed": {"nested": [1, 2, 3]},
            },
        }
    )

    control = _configure(base_candidate=candidate_reference(base))

    assert control.base_candidate.record.payload["fixed"] == {
        "nested": [1, 2, 3]
    }


def test_control_roundtrip_rejects_derived_identity_tampering() -> None:
    control = _configure()
    record = control.model_dump(mode="json")
    record["program_layout"]["component_specs"][0]["component_id"] = "decode"

    with pytest.raises(ValueError):
        type(control).model_validate(record)


@pytest.mark.parametrize(
    ("mode", "instructions", "fewshot", "trials"),
    (
        ("light", 3, 6, 10),
        ("medium", 6, 12, 18),
        ("heavy", 9, 18, 27),
    ),
)
def test_auto_modes_match_frozen_reference_settings(
    mode: Miprov2AutoMode,
    instructions: int,
    fewshot: int,
    trials: int,
) -> None:
    control = _configure(
        auto=mode,
        num_candidates=None,
        num_trials=None,
    )

    assert (
        control.num_instruct_candidates,
        control.num_fewshot_candidates,
        control.num_trials,
    ) == (instructions, fewshot, trials)


def test_zeroshot_auto_uses_all_instruction_candidates() -> None:
    control = _configure(
        auto="light",
        num_candidates=None,
        num_trials=None,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
    )

    assert control.zeroshot_opt is True
    assert control.num_instruct_candidates == 6
    assert control.num_trials == 9


def test_default_dataset_split_and_seed_zero_match_reference() -> None:
    control = _configure(valset=None, run_seed=0, seed=9)

    assert control.trainset_task_identities == TASKS[:1]
    assert control.valset_task_identities == TASKS[1:4]
    assert control.seed == 9


def test_explicit_nonzero_run_seed_controls_auto_sampling() -> None:
    first = _configure(
        auto="light",
        num_candidates=None,
        num_trials=None,
        valset=None,
        run_seed=4,
    )
    replay = _configure(
        auto="light",
        num_candidates=None,
        num_trials=None,
        valset=None,
        run_seed=4,
    )

    assert first.seed == 4
    assert first.auto_validation_sample_indices == (
        replay.auto_validation_sample_indices
    )
    assert first.valset_task_identities == replay.valset_task_identities


def test_manual_recommendation_and_auto_conflicts_match_reference() -> None:
    with pytest.raises(ValueError, match="recommend setting"):
        _configure(num_trials=None)
    with pytest.raises(ValueError, match="would be overridden"):
        _configure(auto="light")


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        (
            {
                "auto": "invalid",
                "trainset": (),
                "valset": (),
                "num_candidates": None,
                "num_trials": None,
            },
            "Invalid value for auto",
        ),
        (
            {
                "auto": None,
                "num_trials": None,
                "trainset": (),
                "valset": (),
            },
            "recommend setting",
        ),
        (
            {
                "auto": "light",
                "trainset": (),
                "valset": (),
            },
            "would be overridden",
        ),
    ),
)
def test_reference_argument_errors_precede_dataset_validation(
    updates: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _configure(**updates)


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"num_threads": 0}, "num_threads"),
        ({"max_errors": 0}, "max_errors"),
        ({"init_temperature": math.inf}, "init_temperature"),
        ({"metric_threshold": math.nan}, "metric_threshold"),
        ({"minibatch_size": 0}, "minibatch_size"),
    ),
)
def test_numeric_safety_boundaries_are_preserved(
    updates: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _configure(**updates)


def test_task_identities_must_be_full_hashes() -> None:
    with pytest.raises(ValueError, match="trainset_task_identities"):
        _configure(trainset=("not-a-hash",), valset=TASKS[4:])


def test_nested_settings_are_snapshotted_and_immutable() -> None:
    settings = {"nested": {"items": [1, 2]}}
    control = _configure(teacher_settings=settings)
    settings["nested"]["items"].append(3)

    assert control.teacher_settings == {"nested": {"items": [1, 2]}}
    with pytest.raises(TypeError):
        control.teacher_settings["new"] = True


def test_bootstrap_and_validation_sources_remain_independent() -> None:
    control = _configure()

    assert control.bootstrap_eval_source != control.validation_eval_source
    assert control.evaluation_binding.eval_config == (
        control.validation_eval_source
    )


def test_metric_requires_explicit_or_injected_authority() -> None:
    defaults = _defaults().model_copy(
        update={"validation_eval_source_is_metric_authority": False}
    )

    with pytest.raises(ValueError, match="metric is required"):
        _configure(defaults=defaults)


def test_public_signature_preserves_dspy_defaults() -> None:
    parameters = inspect.signature(configure_miprov2).parameters

    assert parameters["auto"].default == "light"
    assert parameters["seed"].default == 9
    assert parameters["minibatch"].default is True
    assert parameters["minibatch_size"].default == 35
    assert parameters["minibatch_full_eval_steps"].default == 5


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"trainset": ()}, "Trainset cannot be empty"),
        (
            {"trainset": (TASKS[0],), "valset": None},
            "at least 2 examples",
        ),
        ({"valset": ()}, "Validation set must have at least 1 example"),
    ),
)
def test_dataset_validation_matches_frozen_reference(
    updates: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _configure(**updates)


def test_minibatch_size_is_checked_only_when_minibatching() -> None:
    valset = (TASKS[4],)

    with pytest.raises(ValueError, match="Minibatch size cannot exceed"):
        _configure(valset=valset, minibatch=True, minibatch_size=2)

    control = _configure(valset=valset, minibatch=False, minibatch_size=2)
    assert control.minibatch is False
    assert control.minibatch_size == 2


def test_dataset_rng_replay_resumes_after_auto_sampling() -> None:
    control = _configure(
        auto="light",
        num_candidates=None,
        num_trials=None,
        valset=None,
        run_seed=4,
    )
    replay = control.replay_dataset_rng()

    assert replay.random() == control.replay_dataset_rng().random()
    tampered = control.model_dump(mode="json")
    assert tampered["auto_validation_sample_indices"] is not None
    tampered["auto_validation_sample_indices"] = list(
        reversed(tampered["auto_validation_sample_indices"])
    )
    with pytest.raises(ValueError, match="validation sample"):
        type(control).model_validate(tampered)


def test_teacher_defaults_and_explicit_compiled_state_are_bound() -> None:
    default = _configure()
    teacher = candidate_reference(candidate("teacher", text="Teach {query}."))
    explicit = _configure(teacher=teacher, teacher_compiled=True)

    assert default.teacher_candidate == default.base_candidate
    assert default.teacher_compiled is False
    assert explicit.teacher_candidate == teacher
    assert explicit.teacher_compiled is True
    assert explicit.identity_hash() != default.identity_hash()


def test_teacher_requires_the_same_hard_cut_prompt_surface() -> None:
    invalid_teacher = candidate_reference(
        type(candidate("teacher")).model_validate(
            {
                **candidate("teacher").model_dump(mode="json"),
                "payload": {"fixed": "no prompt surface"},
            }
        )
    )

    with pytest.raises(ValueError, match="teacher candidate component field"):
        _configure(teacher=invalid_teacher)
    with pytest.raises(ValueError, match="teacher_compiled must be a boolean"):
        _configure(teacher_compiled=1)


def test_component_spec_binds_the_prompt_adapter_identity() -> None:
    layout = Miprov2ProgramLayout(
        layout_id="foreign-adapter",
        component_specs=(
            Miprov2ComponentSpec(
                component_id="generate",
                prompt_format_identity_hash=FULL_A,
            ),
        ),
    )

    with pytest.raises(ValueError, match="prompt format conflicts"):
        _configure(program_layout=layout)


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"num_threads": True}, "num_threads must be an integer"),
        ({"max_bootstrapped_demos": -1}, "must be nonnegative"),
        ({"max_labeled_demos": -1}, "must be nonnegative"),
        ({"minibatch_full_eval_steps": 0}, "must be positive"),
        ({"view_data_batch_size": 0}, "must be positive"),
    ),
)
def test_integer_safety_boundaries_remain_strict(
    updates: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _configure(**updates)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("algorithm_version", "foreign", "algorithm_version is fixed"),
        ("reference_commit", "0" * 40, "reference_commit is fixed"),
        ("optuna_version", "0.0.0", "optuna_version is fixed"),
        (
            "candidate_renderer_version",
            "foreign",
            "candidate renderer is fixed",
        ),
        ("phase_schema_manifest", [], "phase schema manifest is fixed"),
        ("state_schema_version", 99, "state schema identity is fixed"),
        ("result_schema_version", 99, "result schema identity is fixed"),
    ),
)
def test_control_deserialization_rejects_schema_authority_drift(
    field: str, value: Any, message: str
) -> None:
    control = _configure()
    record = control.model_dump(mode="json")
    record[field] = value

    with pytest.raises(ValueError, match=message):
        type(control).model_validate(record)


def test_persisted_optimizer_identity_conflict_is_rejected() -> None:
    control = _configure()

    control.require_identity_hash(control.identity_hash())
    with pytest.raises(ValueError, match="conflicts with resolved"):
        control.require_identity_hash(FULL_A)
