"""Rollout Key / Execution Key / Evaluation Context (deliverable 5)."""

from __future__ import annotations

import pytest

from tests.graph.support import fake_hash
from whetstone.graph.rollout import (
    EnvironmentAttestation,
    EvaluationContext,
    EvaluationRole,
    RolloutExecutionKey,
    RolloutKey,
    rollout_execution_key,
)


def _rollout_key() -> RolloutKey:
    return RolloutKey(
        graph_hash=fake_hash("a"),
        eval_config_hash=fake_hash("b"),
        task_identity="humaneval/0",
        repeat_id="r0",
    )


def _context(
    *, role: EvaluationRole = EvaluationRole.INTERNAL, authority=None
) -> EvaluationContext:
    return EvaluationContext(
        eval_config_hash=fake_hash("b"),
        role=role,
        authority=authority,
        campaign="camp-1",
        provider_execution_policy_ref="pep-1",
        environment=EnvironmentAttestation(
            dependency_versions=(("dr-graph", "0.1.0"),),
            code_revision="deadbeef",
        ),
    )


def test_rollout_key_is_exact_four_field_measurement_cell() -> None:
    key = _rollout_key()
    assert set(key.model_dump()) == {
        "graph_hash",
        "eval_config_hash",
        "task_identity",
        "repeat_id",
    }


def test_rollout_key_requires_full_hashes() -> None:
    with pytest.raises(ValueError, match="graph_hash"):
        RolloutKey(
            graph_hash="short",
            eval_config_hash=fake_hash("b"),
            task_identity="t",
            repeat_id="r0",
        )


def test_execution_key_adds_evaluation_context_id() -> None:
    key = _rollout_key()
    context = _context()
    exec_key = rollout_execution_key(rollout_key=key, context=context)
    assert isinstance(exec_key, RolloutExecutionKey)
    assert exec_key.rollout_key == key
    assert exec_key.evaluation_context_id == context.evaluation_context_id()
    assert len(exec_key.evaluation_context_id) == 64


def test_execution_key_requires_matching_eval_config_hash() -> None:
    key = _rollout_key()
    mismatched = EvaluationContext(
        eval_config_hash=fake_hash("c"),
        role=EvaluationRole.INTERNAL,
        campaign="camp-1",
    )
    with pytest.raises(ValueError, match="does not match"):
        rollout_execution_key(rollout_key=key, context=mismatched)


def test_official_context_requires_authority() -> None:
    with pytest.raises(ValueError, match="authority is required"):
        _context(role=EvaluationRole.OFFICIAL, authority=None)


def test_internal_context_forbids_authority() -> None:
    with pytest.raises(ValueError, match="authority must be absent"):
        _context(role=EvaluationRole.INTERNAL, authority="auth-x")


def test_same_config_under_both_roles_keeps_one_eval_config_hash() -> None:
    internal = _context(role=EvaluationRole.INTERNAL)
    official = _context(role=EvaluationRole.OFFICIAL, authority="auth-x")
    # Same ordinary Eval Config referenced under either role; the role
    # changes the Evaluation Context id, not the eval_config_hash.
    assert internal.eval_config_hash == official.eval_config_hash
    assert internal.evaluation_context_id() != (
        official.evaluation_context_id()
    )


def test_context_id_is_stable_and_deterministic() -> None:
    assert _context().evaluation_context_id() == (
        _context().evaluation_context_id()
    )


def test_role_is_closed_internal_or_official() -> None:
    assert {r.value for r in EvaluationRole} == {"internal", "official"}
