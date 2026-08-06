from __future__ import annotations

import inspect
from typing import Any

import pytest
from pydantic import ValidationError

from tests.optimization.support import (
    FULL_A,
    FULL_B,
    FULL_C,
    FULL_D,
    eval_config,
)
from whetstone.core.identity import (
    IdentityRef,
    typed_ref_for_record,
)
from whetstone.experiment.binding import eval_config_reference
from whetstone.optimization.gepa.control import (
    GEPA_AUTO_CANDIDATES,
    GepaControl,
    configure_gepa,
    gepa_auto_budget,
)
from whetstone.optimization.proposal.proposer import ProposerConfig


def _configure(**overrides) -> GepaControl:
    values: dict[str, Any] = {
        "reflection_model": ProposerConfig(
            provider_call_config=IdentityRef(
                record_ref=typed_ref_for_record(
                    "dr_providers.provider_call_config",
                    {"provider_call_config_ref": "provider://reflection"},
                ),
                identity_hash=FULL_A,
            ),
        ),
        "metric": eval_config_reference(eval_config()),
        "reward_policy_hash": FULL_B,
        "evaluation_execution_policy_hash": FULL_C,
        "proposal_execution_policy_hash": FULL_A,
        "proposal_prompt_adapter_identity_hash": FULL_B,
        "proposal_durability_policy_identity_hash": FULL_D,
        "task_model_identity_hash": FULL_D,
        "prompt_format_identity_hash": FULL_A,
        "prompt_binding_identity_hash": FULL_B,
        "trainset_task_identities": (FULL_A, FULL_B),
        "valset_task_identities": (FULL_C,),
        "component_names": ("alpha", "beta"),
        "num_predictors": 2,
        "max_metric_calls": 40,
    }
    values.update(overrides)
    return configure_gepa(**values)


def test_public_defaults_match_frozen_dspy_wrapper() -> None:
    parameters = inspect.signature(configure_gepa).parameters

    assert parameters["auto"].default is None
    assert parameters["max_full_evals"].default is None
    assert parameters["max_metric_calls"].default is None
    assert parameters["reflection_minibatch_size"].default == 3
    assert parameters["candidate_selection_strategy"].default == "pareto"
    assert parameters["skip_perfect_score"].default is True
    assert parameters["add_format_failure_as_feedback"].default is False
    assert parameters["component_selector"].default == "round_robin"
    assert parameters["use_merge"].default is True
    assert parameters["max_merge_invocations"].default == 5
    assert parameters["failure_score"].default == 0.0
    assert parameters["perfect_score"].default == 1.0
    assert parameters["track_stats"].default is False
    assert parameters["track_best_outputs"].default is False
    assert parameters["warn_on_score_mismatch"].default is True
    assert parameters["seed"].default == 0


def test_auto_presets_and_budget_arithmetic_match_frozen_dspy() -> None:
    assert GEPA_AUTO_CANDIDATES == {
        "light": 6,
        "medium": 12,
        "heavy": 18,
    }
    expected = {"light": 790, "medium": 1110, "heavy": 1325}
    for mode, metric_calls in expected.items():
        control = _configure(
            auto=mode,
            max_metric_calls=None,
            trainset_task_identities=(FULL_A, FULL_B),
            valset_task_identities=tuple(
                f"{index:064x}" for index in range(10)
            ),
        )
        assert control.resolved_max_metric_calls == metric_calls

    assert (
        gepa_auto_budget(
            num_predictors=2,
            num_candidates=6,
            valset_size=10,
        )
        == 790
    )


def test_max_full_evals_copies_dspy_valset_none_arithmetic() -> None:
    omitted = _configure(
        max_metric_calls=None,
        max_full_evals=4,
        valset_task_identities=None,
    )
    explicit = _configure(
        max_metric_calls=None,
        max_full_evals=4,
    )

    assert omitted.resolved_max_metric_calls == 8
    assert omitted.valset_task_identities == omitted.trainset_task_identities
    assert explicit.resolved_max_metric_calls == 12


def test_budget_mode_and_non_replayable_controls_fail_closed() -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        _configure(auto="light")
    with pytest.raises(ValueError, match="Exactly one"):
        _configure(max_metric_calls=None)

    dumped = _configure().model_dump(mode="json")
    dumped["seed"] = None
    with pytest.raises(ValidationError):
        GepaControl.model_validate(dumped)
    dumped = _configure().model_dump(mode="json")
    dumped["max_merge_invocations"] = None
    with pytest.raises(ValidationError):
        GepaControl.model_validate(dumped)
    with pytest.raises(ValueError, match="Teacher is not supported"):
        _configure(teacher=object())
    with pytest.raises(ValueError, match="auto must be one of"):
        _configure(auto="surprise", max_metric_calls=None)
    with pytest.raises(ValueError, match="unique non-empty order"):
        _configure(component_names=("alpha", "alpha"))
    with pytest.raises(ValueError, match="ordered component count"):
        _configure(component_names=("alpha",), num_predictors=2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_threads": 2}, "num_threads is not yet supported"),
    ],
)
def test_unsupported_operational_controls_reject_nondefaults(
    overrides, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _configure(**overrides)


def test_adapter_feedback_controls_are_supported_and_identity_bound() -> None:
    base = _configure()
    configured = _configure(
        add_format_failure_as_feedback=True,
        failure_score=-1.0,
        warn_on_score_mismatch=False,
    )

    assert configured.add_format_failure_as_feedback is True
    assert configured.failure_score == -1.0
    assert configured.warn_on_score_mismatch is False
    assert configured.identity_hash() != base.identity_hash()


def test_track_best_outputs_requires_stats() -> None:
    with pytest.raises(ValueError, match="track_stats must be True"):
        _configure(track_best_outputs=True)

    control = _configure(track_stats=True, track_best_outputs=True)
    assert control.track_stats is True
    assert control.track_best_outputs is True


def test_identity_binds_prompt_data_execution_and_upstream_source() -> None:
    control = _configure()
    payload = control.identity_payload()

    assert payload["prompt_format_identity_hash"] == FULL_A
    assert payload["prompt_binding_identity_hash"] == FULL_B
    assert payload["source_trainset_task_identities"] == [FULL_A, FULL_B]
    assert payload["source_valset_task_identities"] == [FULL_C]
    assert payload["reflection_model_identity_hash"] == (
        control.reflection_model.identity_hash()
    )
    assert payload["metric_identity_hash"] == control.metric.identity_hash
    assert len(payload["gepa_source_manifest_hash"]) == 64
    assert len(payload["merge_policy_identity_hash"]) == 64
    assert (
        control.identity_hash()
        != _configure(prompt_binding_identity_hash=FULL_D).identity_hash()
    )


@pytest.mark.parametrize(
    "field",
    [
        "evaluation_execution_policy_hash",
        "proposal_execution_policy_hash",
        "proposal_prompt_adapter_identity_hash",
        "proposal_durability_policy_identity_hash",
    ],
)
def test_identity_separately_binds_each_runtime_authority(field) -> None:
    control = _configure()
    changed = GepaControl.model_validate(
        control.model_copy(update={field: "9" * 64})
    )

    assert changed.identity_hash() != control.identity_hash()


def test_identity_binds_component_order() -> None:
    assert (
        _configure().identity_hash()
        != _configure(component_names=("beta", "alpha")).identity_hash()
    )


def test_upstream_projection_disables_noncanonical_side_effects() -> None:
    kwargs = _configure().upstream_kwargs()

    assert kwargs["run_dir"] is None
    assert kwargs["use_wandb"] is False
    assert kwargs["use_mlflow"] is False
    assert kwargs["display_progress_bar"] is False
    assert kwargs["use_cloudpickle"] is False
    assert kwargs["cache_evaluation"] is False
    assert kwargs["val_evaluation_policy"] == "full_eval"
    assert "stop_callbacks" not in kwargs
    assert "callbacks" not in kwargs
