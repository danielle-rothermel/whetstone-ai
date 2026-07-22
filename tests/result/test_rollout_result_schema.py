"""Rollout Result schema invariants.

Proves the terminal Rollout Result carries every required field group, the
facts-or-failure exclusivity, the nested Graph Run Result reference (never
duplicate) invariant, the no-Platform-Stage-state invariant, and the absence
of any Materialization Record reference.
"""

from __future__ import annotations

import pytest
from dr_graph import GraphRunResult
from dr_store import MemoryBackend, ObjectStore

from whetstone.result import (
    ROLLOUT_RESULT_SCHEMA,
    ExhaustedCausalFailure,
    RolloutResult,
    ScoreFact,
    rollout_result_reference,
)

from .support import (
    execution_key,
    failure_rollout_result,
    graph_run_result,
    provider_attempt,
    success_rollout_result,
)


def test_success_result_carries_all_required_field_groups() -> None:
    result = success_rollout_result()
    # Rollout Execution Key.
    assert result.rollout_execution_key == execution_key()
    # GraphConfig ref + Graph Hash; Eval Config ref + Identity Hash.
    assert result.graph_config_ref
    assert len(result.graph_hash) == 64
    assert result.eval_config_ref
    assert len(result.eval_config_hash) == 64
    # Evaluation Context + authority identities.
    assert result.evaluation_context_id
    # Input identities.
    assert "task_identity" in result.input_identities
    # Nested native GraphRunResult.
    assert isinstance(result.graph_run_result, GraphRunResult)
    # Metric Facts + named Scores.
    assert result.metric_facts
    assert result.scores == (ScoreFact(name="reward", value=1.0),)
    assert result.exhausted_failure is None
    # Provider Call Attempt observation slots.
    assert result.provider_call_attempts
    # Platform Stage Attempt / Durability Replay evidence slots.
    assert result.stage_attempt_evidence.platform_stage_attempt_id == "psa-1"
    assert result.stage_attempt_evidence.durability_replay_count == 0


def test_failure_result_carries_exhausted_causal_failure() -> None:
    result = failure_rollout_result()
    assert result.exhausted_failure is not None
    assert result.exhausted_failure.failure_class == "rate_limited"
    # No facts/scores when the outcome is an exhausted causal failure.
    assert result.metric_facts == ()
    assert result.scores == ()


def test_facts_and_failure_are_mutually_exclusive() -> None:
    key = execution_key()
    with pytest.raises(ValueError, match="never both"):
        RolloutResult(
            rollout_execution_key=key,
            graph_config_ref="g",
            graph_hash=key.rollout_key.graph_hash,
            eval_config_ref="e",
            eval_config_hash=key.rollout_key.eval_config_hash,
            evaluation_context_id=key.evaluation_context_id,
            graph_run_result=graph_run_result(),
            scores=(ScoreFact(name="reward", value=1.0),),
            exhausted_failure=ExhaustedCausalFailure(
                failure_class="permanent",
                failure_exception_type="x",
                underlying_exception_type="y",
                message="z",
            ),
        )


def test_result_requires_facts_or_failure() -> None:
    key = execution_key()
    with pytest.raises(ValueError, match="must carry Metric Facts"):
        RolloutResult(
            rollout_execution_key=key,
            graph_config_ref="g",
            graph_hash=key.rollout_key.graph_hash,
            eval_config_ref="e",
            eval_config_hash=key.rollout_key.eval_config_hash,
            evaluation_context_id=key.evaluation_context_id,
            graph_run_result=graph_run_result(),
        )


def test_nested_graph_run_result_must_match_graph_hash() -> None:
    key = execution_key()
    with pytest.raises(
        ValueError, match=r"nested graph_run_result\.graph_hash"
    ):
        RolloutResult(
            rollout_execution_key=key,
            graph_config_ref="g",
            graph_hash=key.rollout_key.graph_hash,
            eval_config_ref="e",
            eval_config_hash=key.rollout_key.eval_config_hash,
            evaluation_context_id=key.evaluation_context_id,
            graph_run_result=graph_run_result(graph_hash="c" * 64),
            scores=(ScoreFact(name="reward", value=1.0),),
        )


def test_graph_run_result_references_held_provider_bodies() -> None:
    """The nested Graph Run Result references, never orphans, provider bodies.

    Every ``attempt_evidence_ref`` on the nested Graph Run Result MUST resolve
    to a Provider Call Attempt observation held by the enclosing Rollout
    Result: the bodies live once on the Rollout Result and the Graph Run
    Result only points at them.
    """
    key = execution_key()
    # The nested Graph Run Result references an attempt evidence ref that no
    # enclosing Provider Call Attempt observation provides -> rejected.
    with pytest.raises(ValueError, match="not held by an enclosing Provider"):
        RolloutResult(
            rollout_execution_key=key,
            graph_config_ref="g",
            graph_hash=key.rollout_key.graph_hash,
            eval_config_ref="e",
            eval_config_hash=key.rollout_key.eval_config_hash,
            evaluation_context_id=key.evaluation_context_id,
            graph_run_result=graph_run_result(
                attempt_evidence_refs=("attempt-does-not-exist",)
            ),
            scores=(ScoreFact(name="reward", value=1.0),),
            provider_call_attempts=(provider_attempt(),),
        )


def test_graph_run_result_reference_resolves_when_held() -> None:
    result = success_rollout_result()
    # Sanity: the built success result nests a Graph Run Result that
    # references the single held attempt observation's evidence_ref.
    (attempt,) = result.provider_call_attempts
    assert result.graph_run_result.attempt_evidence_refs == (
        attempt.evidence_ref,
    )


def test_no_materialization_record_reference_field() -> None:
    """Absence test: the Rollout Result schema has no Materialization Record
    reference field, and extra fields are forbidden so one cannot be added at
    construction time."""
    field_names = set(RolloutResult.model_fields)
    for name in field_names:
        assert "materialization" not in name.lower()
        assert "material_record" not in name.lower()
    # extra="forbid" rejects any smuggled-in materialization reference. Using
    # ``model_validate`` with a dict keeps the forbidden field out of the
    # static call signature while proving the runtime rejection.
    key = execution_key()
    payload: dict[str, object] = {
        "rollout_execution_key": key,
        "graph_config_ref": "g",
        "graph_hash": key.rollout_key.graph_hash,
        "eval_config_ref": "e",
        "eval_config_hash": key.rollout_key.eval_config_hash,
        "evaluation_context_id": key.evaluation_context_id,
        "graph_run_result": graph_run_result(),
        "scores": (ScoreFact(name="reward", value=1.0),),
        "materialization_record_ref": "materialization://forbidden",
    }
    with pytest.raises(ValueError, match="materialization_record_ref"):
        RolloutResult.model_validate(payload)


def test_no_platform_stage_state_in_nested_graph_run_result() -> None:
    """The nested Graph Run Result contains no Platform Stage state.

    dr-graph's GraphRunResult schema has no stage/platform fields; the stage
    evidence lives only on the enclosing Rollout Result's dedicated slot.
    """
    grr_fields = set(GraphRunResult.model_fields)
    for name in grr_fields:
        assert "stage" not in name.lower()
        assert "platform" not in name.lower()


def test_record_content_roundtrips_through_dr_store() -> None:
    """The Rollout Result persists and reads back verified through dr-store,
    with a Content Hash (never an Identity Hash)."""
    result = success_rollout_result()
    store = ObjectStore(MemoryBackend())
    reference, _status = store.put(
        ROLLOUT_RESULT_SCHEMA, result.record_content()
    )
    assert reference == rollout_result_reference(result)
    assert reference.schema == ROLLOUT_RESULT_SCHEMA
    assert store.get(reference) == result.record_content()
    # A Rollout Result is content-addressed only; there is no identity_hash
    # method or field.
    assert not hasattr(result, "identity_hash")
    assert "identity_hash" not in RolloutResult.model_fields
