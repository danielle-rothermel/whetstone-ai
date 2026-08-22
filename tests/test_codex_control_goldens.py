"""Persisted identity of the Codex-direct optimizer control.

The control is the run's persisted authority, so its schema name, schema
version, and the exact shape of its identity payload are pinned here: a
field added, renamed, or dropped changes the identity hash and must be a
deliberate schema bump rather than a quiet drift.
"""

from __future__ import annotations

import pytest
from dr_store.sync import open_sqlite

from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.codex.containment import (
    CODEX_CONTAINMENT_PROFILE,
    CODEX_DENIED_FEATURES,
    CODEX_FILESYSTEM_POLICY,
    CODEX_NETWORK_POLICY,
)
from whetstone.optim.codex.control import (
    CODEX_ADAPTER_SCHEMA_VERSION,
    CODEX_ALGORITHM,
    CODEX_ALGORITHM_VERSION,
    CODEX_CONTROL_SCHEMA,
    CODEX_CONTROL_SCHEMA_VERSION,
    CodexControl,
    CodexReasoningEffort,
    configure_codex,
)

_PLACEHOLDER_REWARD_HASH = "a" * 64
_PLACEHOLDER_EXECUTION_HASH = "b" * 64
_PLACEHOLDER_TASK_MODEL_HASH = "c" * 64
_PLACEHOLDER_TASK_HASHES = ("d" * 64, "e" * 64)

#: The identity hash of :func:`_toy_control`. Regenerate deliberately.
_TOY_CONTROL_HASH = (
    "96cde1c8e0130a75b7c3868ca74697b7"
    "b98675f55bf529156687579309dd1602"
)


def _eval_config_ref(tmp_path):
    with open_sqlite(str(tmp_path / "codex-control.sqlite")) as store:
        return ReferenceEvalRuntimeConfig().build_engine(store).eval_config_ref


def _toy_control(tmp_path) -> CodexControl:
    """A control whose supplied hashes are placeholders.

    ``eval_config_ref`` is deliberately not one: it is a live reference
    engine binding, and ``identity_payload`` hashes both the full
    ``eval_config_ref`` dump and the derived
    ``eval_config_identity_hash``. So ``_TOY_CONTROL_HASH`` is coupled to
    the reference engine's identity as well as to this control's own
    payload shape, and a change to the toy eval definition moves it. That
    coupling is intended -- the eval binding is part of what a control
    identity means -- but it is why a golden break here can originate
    outside this file.
    """
    return configure_codex(
        model="toy-codex-model",
        max_tool_calls=3,
        eval_config_ref=_eval_config_ref(tmp_path),
        reward_policy_hash=_PLACEHOLDER_REWARD_HASH,
        evaluation_execution_policy_hash=_PLACEHOLDER_EXECUTION_HASH,
        task_model_identity_hash=_PLACEHOLDER_TASK_MODEL_HASH,
        internal_task_hashes=_PLACEHOLDER_TASK_HASHES,
        wall_seconds=120.0,
    )


def test_control_schema_name_and_version_are_pinned() -> None:
    assert CODEX_CONTROL_SCHEMA == "whetstone.codex_optimizer_config"
    assert CODEX_CONTROL_SCHEMA_VERSION == 1
    assert CODEX_ALGORITHM == "codex"
    assert CODEX_ALGORITHM_VERSION == "whetstone.codex_direct/v1"
    assert CODEX_ADAPTER_SCHEMA_VERSION == "whetstone.codex_output_artifact/v1"


def test_identity_payload_names_the_algorithm_and_the_eval_binding(
    tmp_path,
) -> None:
    control = _toy_control(tmp_path)
    payload = control.identity_payload()

    assert payload["algorithm"] == CODEX_ALGORITHM
    assert (
        payload["eval_config_identity_hash"]
        == control.eval_config_ref.config_hash
    )
    # The containment posture is part of run identity: a run that denied a
    # different feature set is a different run.
    assert payload["denied_features"] == list(CODEX_DENIED_FEATURES)
    assert payload["containment_profile"] == CODEX_CONTAINMENT_PROFILE
    assert payload["network_policy"] == CODEX_NETWORK_POLICY
    assert payload["filesystem_policy"] == CODEX_FILESYSTEM_POLICY
    # Derived properties stay out of the payload.
    for absent in ("reference", "identity_hash", "record_content"):
        assert absent not in payload


def test_the_identity_payload_field_set_is_pinned(tmp_path) -> None:
    assert set(_toy_control(tmp_path).identity_payload()) == {
        "adapter_schema_version",
        "algorithm",
        "algorithm_version",
        "codex_binary",
        "containment_profile",
        "denied_features",
        "eval_config_identity_hash",
        "eval_config_ref",
        "evaluation_execution_policy_hash",
        "filesystem_policy",
        "internal_task_hashes",
        "max_output_bytes",
        "max_tool_calls",
        "model",
        "mutation_field",
        "network_policy",
        "reasoning_effort",
        "reward_policy_hash",
        "task_model_identity_hash",
        "wall_seconds",
    }


def test_the_toy_control_identity_hash_is_stable(tmp_path) -> None:
    assert _toy_control(tmp_path).identity_hash() == _TOY_CONTROL_HASH


def test_max_tool_calls_is_required_and_positive(tmp_path) -> None:
    reference = _eval_config_ref(tmp_path)
    with pytest.raises(ValueError, match="max_tool_calls must be positive"):
        configure_codex(
            model="toy-codex-model",
            max_tool_calls=0,
            eval_config_ref=reference,
            reward_policy_hash=_PLACEHOLDER_REWARD_HASH,
            evaluation_execution_policy_hash=_PLACEHOLDER_EXECUTION_HASH,
            task_model_identity_hash=_PLACEHOLDER_TASK_MODEL_HASH,
            internal_task_hashes=_PLACEHOLDER_TASK_HASHES,
        )


def test_internal_task_hashes_must_be_non_empty_and_unique(tmp_path) -> None:
    reference = _eval_config_ref(tmp_path)
    common = {
        "model": "toy-codex-model",
        "max_tool_calls": 2,
        "eval_config_ref": reference,
        "reward_policy_hash": _PLACEHOLDER_REWARD_HASH,
        "evaluation_execution_policy_hash": _PLACEHOLDER_EXECUTION_HASH,
        "task_model_identity_hash": _PLACEHOLDER_TASK_MODEL_HASH,
    }
    with pytest.raises(ValueError, match="must be non-empty"):
        configure_codex(**common, internal_task_hashes=())
    with pytest.raises(ValueError, match="unique identities"):
        configure_codex(
            **common, internal_task_hashes=("d" * 64, "d" * 64)
        )


def test_the_containment_posture_is_fixed(tmp_path) -> None:
    control = _toy_control(tmp_path)
    with pytest.raises(ValueError, match="denied_features is fixed"):
        control.model_copy(update={"denied_features": ["shell_tool"]})


def test_model_copy_revalidates_through_the_full_boundary(tmp_path) -> None:
    control = _toy_control(tmp_path)
    with pytest.raises(ValueError, match="max_tool_calls must be positive"):
        control.model_copy(update={"max_tool_calls": 0})
    widened = control.model_copy(update={"max_tool_calls": 9})
    assert widened.max_tool_calls == 9
    assert widened.identity_hash() != control.identity_hash()


def test_reasoning_effort_defaults_to_medium(tmp_path) -> None:
    assert (
        _toy_control(tmp_path).reasoning_effort is CodexReasoningEffort.MEDIUM
    )


def test_the_single_step_is_the_only_hyperparameter_iteration(
    tmp_path,
) -> None:
    control = _toy_control(tmp_path)
    assert control.step_hyperparameters(iteration=0)["max_tool_calls"] == 3
    with pytest.raises(ValueError, match="exactly one opaque step"):
        control.step_hyperparameters(iteration=1)
