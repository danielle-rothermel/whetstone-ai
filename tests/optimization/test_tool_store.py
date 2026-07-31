"""Atomic Tool admission, exact-chain persistence, and capacity scope."""

from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, ClassVar, LiteralString, cast
from uuid import uuid4

import pytest
from dr_store import (
    MemoryBackend,
    ObjectConflictError,
    ObjectStore,
    PutOutcome,
    SqliteBackend,
)
from pydantic import ValidationError

from whetstone.evaluation_role import EvaluationRole
from whetstone.optimization import tool_store as tool_store_module
from whetstone.optimization.effect_authority import (
    EffectAuthority,
    TerminalConflictError,
)
from whetstone.optimization.identity import TypedRef, typed_ref_for_record
from whetstone.optimization.reward import (
    REWARD_SCHEMA,
    Reward,
    RewardInputCitation,
    RewardPolicy,
    RewardTerm,
    reward_reference,
)
from whetstone.optimization.tool_store import (
    ToolAdmissionAuthority,
    ToolAdmissionSchemaMismatchError,
    ToolCallState,
    ToolCallStore,
    ToolCallStoreConflictError,
    ToolCallStoreEntry,
    tool_effect_request,
)
from whetstone.optimization.tools import (
    GLOBAL_CAPACITY_SCOPE_ID,
    RUN_CAPACITY_SUBJECT_SCHEMA,
    STEP_CAPACITY_SUBJECT_SCHEMA,
    RefusalClass,
    ToolCall,
    ToolCapacity,
    ToolCapacityBinding,
    ToolCapacityScope,
    ToolConfig,
    ToolDefinition,
    ToolRefusal,
    ToolResult,
    tool_call_reference,
    tool_config_reference,
    tool_definition_reference,
)

from .processes import (
    in_process_start_methods,
    join_processes,
    terminate_processes,
)
from .support import eval_config
from .tool_store_spawn import admit_once, load_terminal_result_once

FULL_A = "a" * 64
FULL_B = "b" * 64


def _config(
    *,
    capacity: int = 2,
    namespace: str = "tool-ns",
    scope: ToolCapacityScope = ToolCapacityScope.RUN,
    reward_policy_hash: str = FULL_B,
) -> ToolConfig:
    definition = ToolDefinition(
        tool_name="evaluate_candidate",
        input_fields=("model_route", "template"),
        output_fields=("rollout_refs", "accepted_ordinal"),
    )
    return ToolConfig(
        definition=tool_definition_reference(definition),
        endpoint_key="evaluate_candidate",
        eval_config=eval_config(FULL_A),
        reward_policy_hash=reward_policy_hash,
        capacity=ToolCapacity(
            max_accepted_calls=capacity,
            scope=scope,
        ),
        store_namespace_key=namespace,
    )


def _call(
    config: ToolConfig,
    call_id: str,
    *,
    template: str | None = None,
    scope_id: str = "run-1",
) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        tool_config=tool_config_reference(config),
        capacity_binding=_binding(config.capacity.scope, scope_id),
        args={
            "model_route": "route",
            "template": template if template is not None else call_id,
        },
    )


def _binding(
    scope: ToolCapacityScope,
    subject: str = "run-1",
) -> ToolCapacityBinding:
    if scope is ToolCapacityScope.GLOBAL:
        return ToolCapacityBinding(scope=scope)
    schema = (
        RUN_CAPACITY_SUBJECT_SCHEMA
        if scope is ToolCapacityScope.RUN
        else STEP_CAPACITY_SUBJECT_SCHEMA
    )
    return ToolCapacityBinding(
        scope=scope,
        subject_ref=typed_ref_for_record(schema, {"subject": subject}),
    )


def _store(
    database,
    *,
    effect_authority: EffectAuthority | None = None,
) -> ToolCallStore:
    return ToolCallStore(
        ObjectStore(SqliteBackend(database)),
        ToolAdmissionAuthority.sqlite(database),
        effect_authority or EffectAuthority.memory(),
    )


def _success(call: ToolCall, ordinal: int) -> ToolResult:
    return ToolResult(
        call=tool_call_reference(call),
        output={
            "rollout_refs": [],
            "accepted_ordinal": ordinal,
        },
        provenance_ordinal=ordinal,
    )


def _reward(policy: RewardPolicy) -> Reward:
    return Reward(
        reward_name="reward",
        value=1.0,
        reward_policy=policy,
        evidence_role=EvaluationRole.INTERNAL,
        input_citations=(
            RewardInputCitation(
                name="score",
                value=1.0,
                contributed=1.0,
            ),
        ),
        evidence_refs=(_reward_evidence_ref(),),
    )


def _reward_evidence_ref() -> TypedRef:
    return typed_ref_for_record(
        "whetstone.test.reward_evidence",
        {"evidence": "score"},
    )


def test_admission_replay_debits_once_and_completion_is_exact(
    tmp_path,
) -> None:
    database = tmp_path / "tool.sqlite"
    config = _config(capacity=2)
    call = _call(config, "c1")
    store = _store(database)

    accepted = store.admit(call, config)
    assert accepted.state is ToolCallState.ACCEPTED
    assert accepted.capacity_debit_ordinal == 1
    assert store.admit(call, config) == accepted
    assert store.accepted_count(config, _binding(ToolCapacityScope.RUN)) == 1

    result = _success(call, 1)
    with pytest.raises(
        ValueError,
        match="exact authoritative EffectTerminal proof",
    ):
        store.complete(result)

    authority = store.effect_authority
    acquisition = authority.acquire(
        tool_effect_request(call),
        owner_id="test-owner",
        attempt_id="test-attempt",
        lease_duration=timedelta(seconds=10),
    )
    assert acquisition.lease is not None
    result_ref = store.persist_result(result)
    terminal = authority.succeed(acquisition.lease, result_ref=result_ref)
    completed = store.complete(
        result,
        terminal=terminal,
    )
    assert completed.state is ToolCallState.COMPLETED
    assert completed.effect_terminal == terminal
    assert (
        store.complete(
            result,
            terminal=terminal,
        )
        == completed
    )
    assert store.load_terminal_result(completed) == result

    wrong_request = completed.model_dump(mode="json")
    wrong_request["effect_terminal"]["request"]["request_identity_hash"] = (
        FULL_B
    )
    with pytest.raises(
        ValidationError,
        match="belongs to another exact Tool request",
    ):
        ToolCallStoreEntry.model_validate(wrong_request)

    wrong_result = completed.model_dump(mode="json")
    wrong_result["effect_terminal"]["result_ref"]["content_hash"] = FULL_A
    with pytest.raises(
        ValidationError,
        match="references another exact Tool Result",
    ):
        ToolCallStoreEntry.model_validate(wrong_result)

    recovery = completed.model_dump(mode="json")
    recovery["effect_terminal"]["outcome"] = "recovery_required"
    recovery["effect_terminal"]["result_ref"] = None
    recovery["effect_terminal"]["failure"] = {
        "code": "recovery",
        "message": "manual recovery required",
        "details": {},
    }
    with pytest.raises(
        ValidationError,
        match="recovery-required effects have no completed Tool Result",
    ):
        ToolCallStoreEntry.model_validate(recovery)


def test_completed_terminal_survives_spawned_restart(tmp_path) -> None:
    database = tmp_path / "spawned-completion.sqlite"
    authority = EffectAuthority.sqlite(database)
    store = _store(database, effect_authority=authority)
    config = _config(capacity=1)
    call = _call(config, "spawned-restart")
    store.admit(call, config)
    result = _success(call, 1)
    acquisition = authority.acquire(
        tool_effect_request(call),
        owner_id="spawned-owner",
        attempt_id="spawned-attempt",
        lease_duration=timedelta(seconds=10),
    )
    assert acquisition.lease is not None
    terminal = authority.succeed(
        acquisition.lease,
        result_ref=store.persist_result(result),
    )
    completed = store.complete(result, terminal=terminal)

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=load_terminal_result_once,
        args=(str(database), call.model_dump(mode="json"), queue),
    )
    process.start()
    process.join(timeout=30)

    assert process.exitcode == 0
    record = queue.get(timeout=5)
    assert "error" not in record
    assert record["entry"] == completed.model_dump(mode="json")
    assert record["result"] == result.model_dump(mode="json")


def test_poisoned_admission_terminal_fails_bound_authority_verification(
    tmp_path,
) -> None:
    database = tmp_path / "poisoned-completion.sqlite"
    authority = EffectAuthority.sqlite(database)
    store = _store(database, effect_authority=authority)
    config = _config(capacity=1)
    call = _call(config, "poisoned")
    store.admit(call, config)
    result = _success(call, 1)
    acquisition = authority.acquire(
        tool_effect_request(call),
        owner_id="real-owner",
        attempt_id="real-attempt",
        lease_duration=timedelta(seconds=10),
    )
    assert acquisition.lease is not None
    terminal = authority.succeed(
        acquisition.lease,
        result_ref=store.persist_result(result),
    )
    store.complete(result, terminal=terminal)

    with sqlite3.connect(database) as connection:
        raw = connection.execute(
            """
            SELECT entry_json FROM whetstone_tool_admission_entry
            WHERE store_namespace_key = ? AND call_id = ?
            """,
            (str(call.store_namespace_key), str(call.call_id)),
        ).fetchone()
        assert raw is not None
        poisoned = json.loads(str(raw[0]))
        poisoned["effect_terminal"]["owner_id"] = "poisoned-owner"
        connection.execute(
            """
            UPDATE whetstone_tool_admission_entry SET entry_json = ?
            WHERE store_namespace_key = ? AND call_id = ?
            """,
            (
                json.dumps(poisoned),
                str(call.store_namespace_key),
                str(call.call_id),
            ),
        )

    restarted = _store(
        database,
        effect_authority=EffectAuthority.sqlite(database),
    )
    entry = restarted.get(call)
    assert entry is not None
    with pytest.raises(TerminalConflictError, match="not authoritative"):
        restarted.load_terminal_result(entry)


def test_completion_rejects_terminal_from_caller_selected_authority(
    tmp_path,
) -> None:
    store = _store(tmp_path / "canonical.sqlite")
    config = _config(capacity=1)
    call = _call(config, "foreign-proof")
    store.admit(call, config)
    result = _success(call, 1)
    result_ref = store.persist_result(result)
    foreign_authority = EffectAuthority.memory()
    acquisition = foreign_authority.acquire(
        tool_effect_request(call),
        owner_id="foreign-owner",
        attempt_id="foreign-attempt",
        lease_duration=timedelta(seconds=10),
    )
    assert acquisition.lease is not None
    foreign_terminal = foreign_authority.succeed(
        acquisition.lease,
        result_ref=result_ref,
    )

    with pytest.raises(
        TerminalConflictError,
        match="not authoritative",
    ):
        store.complete(result, terminal=foreign_terminal)

    entry = store.get(call)
    assert entry is not None
    assert entry.state is ToolCallState.ACCEPTED


def test_persist_result_stores_embedded_reward_before_public_result_load(
    tmp_path,
) -> None:
    object_database = tmp_path / "objects.sqlite"
    objects = ObjectStore(SqliteBackend(object_database))
    store = ToolCallStore(
        objects,
        ToolAdmissionAuthority.memory(),
        EffectAuthority.memory(),
    )
    policy = RewardPolicy(
        policy_name="tool",
        terms=(RewardTerm(name="score", weight=1.0),),
    )
    config = _config(reward_policy_hash=str(policy.identity_hash()))
    call = _call(config, "rewarded")
    reward = reward_reference(_reward(policy))
    result = ToolResult(
        call=tool_call_reference(call),
        output={"rollout_refs": [], "accepted_ordinal": 1},
        reward=reward,
        evaluation_evidence_refs=reward.record.evidence_refs,
        provenance_ordinal=1,
    )

    result_ref = store.persist_result(result)

    restarted_objects = ObjectStore(SqliteBackend(object_database))
    assert reward.record_ref.schema_name == REWARD_SCHEMA
    assert (
        Reward.model_validate(
            restarted_objects.get(reward.record_ref.reference)
        )
        == reward.record
    )
    assert (
        ToolCallStore(
            restarted_objects,
            ToolAdmissionAuthority.memory(),
            EffectAuthority.memory(),
        ).load_result(result_ref, expected_call=call)
        == result
    )


class _CollisionBackend(MemoryBackend):
    def put_object(
        self,
        *,
        schema: str,
        content_hash: str,
        canonical: str,
    ) -> PutOutcome:
        del content_hash, canonical
        return PutOutcome(
            inserted=False,
            stored_schema=schema,
            stored_canonical="{}",
        )


def test_persist_result_rejects_embedded_reward_collision() -> None:
    policy = RewardPolicy(
        policy_name="tool",
        terms=(RewardTerm(name="score", weight=1.0),),
    )
    config = _config(reward_policy_hash=str(policy.identity_hash()))
    reward = reward_reference(_reward(policy))
    result = ToolResult(
        call=tool_call_reference(_call(config, "collision")),
        output={"rollout_refs": [], "accepted_ordinal": 1},
        reward=reward,
        evaluation_evidence_refs=reward.record.evidence_refs,
        provenance_ordinal=1,
    )
    store = ToolCallStore(
        ObjectStore(_CollisionBackend()),
        ToolAdmissionAuthority.memory(),
        EffectAuthority.memory(),
    )

    with pytest.raises(ObjectConflictError):
        store.persist_result(result)


def test_capacity_refusal_is_terminal_exact_and_consumes_nothing(
    tmp_path,
) -> None:
    database = tmp_path / "tool.sqlite"
    config = _config(capacity=1)
    store = _store(database)
    store.admit(_call(config, "accepted"), config)

    refused = store.admit(_call(config, "refused"), config)
    assert refused.state is ToolCallState.REFUSED
    assert refused.capacity_debit_ordinal is None
    assert refused.refusal is not None
    assert refused.refusal.refusal_class is RefusalClass.CAPACITY
    result = store.load_terminal_result(refused)
    assert result.refusal == refused.refusal
    assert result.call == refused.tool_call
    assert store.accepted_count(config, _binding(ToolCapacityScope.RUN)) == 1


def test_refused_terminal_load_rejects_mismatched_persisted_refusal(
    tmp_path,
) -> None:
    database = tmp_path / "refusal-poison.sqlite"
    config = _config(capacity=1)
    call = _call(config, "refused")
    store = _store(database)
    refusal = ToolRefusal(
        refusal_class=RefusalClass.AUTHORIZATION,
        reason="not authorized",
    )
    entry = store.refuse(call, config, refusal=refusal)
    other_result = ToolResult(
        call=tool_call_reference(call),
        refusal=ToolRefusal(
            refusal_class=RefusalClass.AUTHORIZATION,
            reason="different refusal",
        ),
    )
    other_ref = store.persist_result(other_result)
    poisoned = entry.model_dump(mode="json")
    poisoned["tool_result_ref"] = other_ref.model_dump(mode="json")
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE whetstone_tool_admission_entry SET entry_json = ?
            WHERE store_namespace_key = ? AND call_id = ?
            """,
            (
                json.dumps(poisoned),
                str(call.store_namespace_key),
                str(call.call_id),
            ),
        )

    durable = store.get(call)
    assert durable is not None
    with pytest.raises(
        ValueError,
        match="admission refusal disagree",
    ):
        store.load_terminal_result(durable)


def test_noncapacity_refusal_consumes_no_capacity(tmp_path) -> None:
    database = tmp_path / "refusal.sqlite"
    config = _config(capacity=1)
    store = _store(database)
    refused = store.refuse(
        _call(config, "refused"),
        config,
        refusal=ToolRefusal(
            refusal_class=RefusalClass.AUTHORIZATION,
            reason="not authorized",
        ),
    )

    assert refused.state is ToolCallState.REFUSED
    assert refused.capacity_debit_ordinal is None
    assert store.accepted_count(config, _binding(ToolCapacityScope.RUN)) == 0
    accepted = store.admit(_call(config, "accepted"), config)
    assert accepted.capacity_debit_ordinal == 1


def test_completed_result_is_terminally_immutable(tmp_path) -> None:
    database = tmp_path / "completed.sqlite"
    config = _config(capacity=1)
    call = _call(config, "call")
    store = _store(database)
    store.admit(call, config)
    first_result = _success(call, 1)
    authority = store.effect_authority
    acquisition = authority.acquire(
        tool_effect_request(call),
        owner_id="test-owner",
        attempt_id="test-attempt",
        lease_duration=timedelta(seconds=10),
    )
    assert acquisition.lease is not None
    terminal = authority.succeed(
        acquisition.lease,
        result_ref=store.persist_result(first_result),
    )
    completed = store.complete(
        first_result,
        terminal=terminal,
    )

    with pytest.raises(
        ValueError,
        match="another exact Tool Result",
    ):
        store.complete(
            ToolResult(
                call=tool_call_reference(call),
                output={"rollout_refs": [], "accepted_ordinal": 2},
                provenance_ordinal=1,
            ),
            terminal=terminal,
        )

    assert store.get(call) == completed
    assert store.accepted_count(config, _binding(ToolCapacityScope.RUN)) == 1


def test_completion_rejects_provenance_ordinal_mismatch(
    tmp_path,
) -> None:
    database = tmp_path / "provenance-poison.sqlite"
    config = _config(capacity=1)
    call = _call(config, "call")
    store = _store(database)
    store.admit(call, config)
    result = _success(call, 2)
    acquisition = store.effect_authority.acquire(
        tool_effect_request(call),
        owner_id="owner",
        attempt_id="attempt",
        lease_duration=timedelta(seconds=10),
    )
    assert acquisition.lease is not None
    terminal = store.effect_authority.succeed(
        acquisition.lease,
        result_ref=store.persist_result(result),
    )
    with pytest.raises(ValueError, match="provenance ordinal disagrees"):
        store.complete(result, terminal=terminal)

    entry = store.get(call)
    assert entry is not None
    assert entry.state is ToolCallState.ACCEPTED


@pytest.mark.parametrize("ordinal", [None, 0])
def test_completion_requires_positive_provenance_ordinal(
    tmp_path,
    ordinal: int | None,
) -> None:
    database = tmp_path / "provenance-positive.sqlite"
    config = _config(capacity=1)
    call = _call(config, "call")
    store = _store(database)
    store.admit(call, config)
    valid_result = ToolResult(
        call=tool_call_reference(call),
        output={"rollout_refs": [], "accepted_ordinal": 1},
        provenance_ordinal=1,
    )
    hostile_result = ToolResult.model_construct(
        call=valid_result.call,
        output=valid_result.output,
        refusal=None,
        terminal_failure=None,
        evaluation_evidence_refs=(),
        reward=None,
        provenance_note=None,
        provenance_ordinal=ordinal,
    )
    acquisition = store.effect_authority.acquire(
        tool_effect_request(call),
        owner_id="owner",
        attempt_id="attempt",
        lease_duration=timedelta(seconds=10),
    )
    assert acquisition.lease is not None
    terminal = store.effect_authority.succeed(
        acquisition.lease,
        result_ref=store.persist_result(valid_result),
    )

    with pytest.raises(ValueError, match="positive provenance ordinal"):
        store.complete(hostile_result, terminal=terminal)

    entry = store.get(call)
    assert entry is not None
    assert entry.state is ToolCallState.ACCEPTED


def test_completed_terminal_load_rejects_capacity_projection_divergence(
    tmp_path,
) -> None:
    database = tmp_path / "capacity-projection-poison.sqlite"
    config = _config(capacity=1)
    call = _call(config, "call")
    store = _store(database)
    store.admit(call, config)
    result = _success(call, 1)
    acquisition = store.effect_authority.acquire(
        tool_effect_request(call),
        owner_id="owner",
        attempt_id="attempt",
        lease_duration=timedelta(seconds=10),
    )
    assert acquisition.lease is not None
    terminal = store.effect_authority.succeed(
        acquisition.lease,
        result_ref=store.persist_result(result),
    )
    completed = store.complete(result, terminal=terminal)
    poisoned = completed.model_dump(mode="json")
    poisoned["capacity_debit_ordinal"] = 2
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE whetstone_tool_admission_entry SET entry_json = ?
            WHERE store_namespace_key = ? AND call_id = ?
            """,
            (
                json.dumps(poisoned),
                str(call.store_namespace_key),
                str(call.call_id),
            ),
        )

    durable = store.get(call)
    assert durable is not None
    with pytest.raises(ValueError, match="durable admission projection"):
        store.load_terminal_result(durable)


def test_terminal_load_rejects_non_durable_entry_projection(tmp_path) -> None:
    database = tmp_path / "entry-projection.sqlite"
    config = _config(capacity=1)
    call = _call(config, "refused")
    store = _store(database)
    entry = store.refuse(
        call,
        config,
        refusal=ToolRefusal(
            refusal_class=RefusalClass.AUTHORIZATION,
            reason="not authorized",
        ),
    )
    forged = entry.model_copy(
        update={
            "refusal": ToolRefusal(
                refusal_class=RefusalClass.AUTHORIZATION,
                reason="forged refusal",
            )
        }
    )

    with pytest.raises(ValueError, match="durable admission decision"):
        store.load_terminal_result(forged)


def test_divergent_input_for_same_namespace_call_conflicts(tmp_path) -> None:
    database = tmp_path / "tool.sqlite"
    config = _config(capacity=2)
    first = _store(database)
    first.admit(_call(config, "same", template="one"), config)

    second = _store(database)
    with pytest.raises(ToolCallStoreConflictError, match="different exact"):
        second.admit(_call(config, "same", template="two"), config)
    assert second.accepted_count(config, _binding(ToolCapacityScope.RUN)) == 1


def test_config_namespace_and_scope_chain_are_exact(tmp_path) -> None:
    database = tmp_path / "tool.sqlite"
    config = _config(capacity=1, namespace="namespace-a")
    store = _store(database)
    call = _call(config, "same", scope_id="run-a")
    store.admit(call, config)

    other_config = _config(capacity=1, namespace="namespace-b")
    with pytest.raises(ValueError, match="exact supplied Tool Config"):
        store.admit(call, other_config)

    assert (
        store.accepted_count(config, _binding(ToolCapacityScope.RUN, "run-a"))
        == 1
    )
    assert (
        store.accepted_count(config, _binding(ToolCapacityScope.RUN, "run-b"))
        == 0
    )
    separate_scope = store.admit(
        _call(config, "other", scope_id="run-b"), config
    )
    assert separate_scope.capacity_debit_ordinal == 1


def test_global_scope_rejects_caller_partitioning(tmp_path) -> None:
    database = tmp_path / "global.sqlite"
    config = _config(capacity=1, scope=ToolCapacityScope.GLOBAL)
    store = _store(database)

    with pytest.raises(ValueError, match="requires no subject_ref"):
        ToolCapacityBinding(
            scope=ToolCapacityScope.GLOBAL,
            subject_ref=typed_ref_for_record(
                RUN_CAPACITY_SUBJECT_SCHEMA, {"subject": "forged"}
            ),
        )

    first = store.admit(
        _call(config, "first", scope_id=GLOBAL_CAPACITY_SCOPE_ID),
        config,
    )
    second = store.admit(
        _call(config, "second", scope_id=GLOBAL_CAPACITY_SCOPE_ID),
        config,
    )
    assert first.state is ToolCallState.ACCEPTED
    assert second.state is ToolCallState.REFUSED
    assert (
        store.accepted_count(
            config,
            _binding(ToolCapacityScope.GLOBAL),
        )
        == 1
    )


def test_tool_admission_persisted_literals_are_pinned() -> None:
    assert GLOBAL_CAPACITY_SCOPE_ID == "global"
    assert tool_store_module._SCHEMA_TABLE == (
        "whetstone_tool_admission_schema"
    )
    assert tool_store_module._ENTRY_TABLE == ("whetstone_tool_admission_entry")
    assert tool_store_module._CAPACITY_TABLE == (
        "whetstone_tool_admission_capacity"
    )
    assert tool_store_module._SCHEMA_COMPONENT == "tool_admission"
    assert tool_store_module._SCHEMA_VERSION == 2
    assert (
        tool_store_module._ENTRY_LOCK_DOMAIN
        == "whetstone.tool_admission.entry_lock.v1"
    )
    assert tool_store_module._SQLITE_SCHEMA_COLUMNS == (
        ("component", "TEXT", True, 1),
        ("version", "INTEGER", True, 0),
    )
    assert tool_store_module._SQLITE_ENTRY_COLUMNS == (
        ("store_namespace_key", "TEXT", True, 1),
        ("call_id", "TEXT", True, 2),
        ("entry_json", "TEXT", True, 0),
    )
    assert tool_store_module._SQLITE_CAPACITY_COLUMNS == (
        ("store_namespace_key", "TEXT", True, 1),
        ("tool_config_hash", "TEXT", True, 2),
        ("capacity_scope", "TEXT", True, 3),
        ("capacity_scope_id", "TEXT", True, 4),
        ("max_accepted_calls", "INTEGER", True, 0),
        ("consumed", "INTEGER", True, 0),
    )
    assert tool_store_module._POSTGRES_SCHEMA_COLUMNS == (
        ("component", "text", True, "pg_catalog", "C", "c", True, -1),
        ("version", "bigint", True, None, None, None, None, None),
    )
    assert tool_store_module._POSTGRES_ENTRY_COLUMNS == (
        (
            "store_namespace_key",
            "text",
            True,
            "pg_catalog",
            "C",
            "c",
            True,
            -1,
        ),
        ("call_id", "text", True, "pg_catalog", "C", "c", True, -1),
        ("entry_json", "text", True, "pg_catalog", "C", "c", True, -1),
    )
    assert tool_store_module._POSTGRES_CAPACITY_COLUMNS == (
        (
            "store_namespace_key",
            "text",
            True,
            "pg_catalog",
            "C",
            "c",
            True,
            -1,
        ),
        (
            "tool_config_hash",
            "text",
            True,
            "pg_catalog",
            "C",
            "c",
            True,
            -1,
        ),
        (
            "capacity_scope",
            "text",
            True,
            "pg_catalog",
            "C",
            "c",
            True,
            -1,
        ),
        (
            "capacity_scope_id",
            "text",
            True,
            "pg_catalog",
            "C",
            "c",
            True,
            -1,
        ),
        ("max_accepted_calls", "bigint", True, None, None, None, None, None),
        ("consumed", "bigint", True, None, None, None, None, None),
    )
    assert [state.value for state in ToolCallState] == [
        "accepted",
        "refused",
        "completed",
    ]
    assert [scope.value for scope in ToolCapacityScope] == [
        "global",
        "run",
        "step",
    ]


def test_sqlite_admission_schema_pins_every_storage_class(tmp_path) -> None:
    database = tmp_path / "storage-classes.sqlite"
    ToolAdmissionAuthority.sqlite(database)
    expected_columns = {
        "whetstone_tool_admission_schema": ("component", "version"),
        "whetstone_tool_admission_entry": (
            "store_namespace_key",
            "call_id",
            "entry_json",
        ),
        "whetstone_tool_admission_capacity": (
            "store_namespace_key",
            "tool_config_hash",
            "capacity_scope",
            "capacity_scope_id",
            "max_accepted_calls",
            "consumed",
        ),
    }
    with sqlite3.connect(database) as connection:
        for table, columns in expected_columns.items():
            row = connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type = 'table' AND name = ?
                """,
                (table,),
            ).fetchone()
            assert row is not None
            compact_sql = "".join(row[0].split())
            for column in columns:
                assert f"typeof({column})=" in compact_sql


def test_sqlite_initialization_migrates_exact_unversioned_schema(
    tmp_path,
) -> None:
    database = tmp_path / "unversioned.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(tool_store_module._SQLITE_CREATE_ENTRY)
        connection.execute(tool_store_module._SQLITE_CREATE_CAPACITY)

    ToolAdmissionAuthority.sqlite(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            """
            SELECT component, version
            FROM whetstone_tool_admission_schema
            """
        ).fetchall() == [("tool_admission", 2)]


def test_sqlite_initialization_rejects_truncated_table(tmp_path) -> None:
    database = tmp_path / "truncated.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE whetstone_tool_admission_entry (
                store_namespace_key TEXT NOT NULL
            )
            """
        )

    with pytest.raises(
        RuntimeError,
        match=r"owned table inventory.*whetstone_tool_admission_entry",
    ):
        ToolAdmissionAuthority.sqlite(database)


def test_sqlite_initialization_rejects_unaudited_capacity_table(
    tmp_path,
) -> None:
    database = tmp_path / "capacity-only.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(tool_store_module._SQLITE_CREATE_CAPACITY)
        connection.execute(
            """
            INSERT INTO whetstone_tool_admission_capacity (
                store_namespace_key, tool_config_hash, capacity_scope,
                capacity_scope_id, max_accepted_calls, consumed
            ) VALUES ('namespace', 'config', 'run', 'run-1', 2, 1)
            """
        )

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match=r"owned table inventory.*whetstone_tool_admission_capacity",
    ):
        ToolAdmissionAuthority.sqlite(database)

    with sqlite3.connect(database) as connection:
        owned_tables = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name LIKE 'whetstone_tool_admission_%'
            ORDER BY name
            """
        ).fetchall()
        consumed = connection.execute(
            "SELECT consumed FROM whetstone_tool_admission_capacity"
        ).fetchall()
    assert owned_tables == [("whetstone_tool_admission_capacity",)]
    assert consumed == [(1,)]


def test_sqlite_initialization_rejects_columns_with_wrong_constraints(
    tmp_path,
) -> None:
    database = tmp_path / "wrong-constraints.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(tool_store_module._SQLITE_CREATE_ENTRY)
        connection.execute(
            """
            CREATE TABLE whetstone_tool_admission_capacity (
                store_namespace_key TEXT NOT NULL,
                tool_config_hash TEXT NOT NULL,
                capacity_scope TEXT NOT NULL,
                capacity_scope_id TEXT NOT NULL,
                max_accepted_calls INTEGER NOT NULL,
                consumed INTEGER NOT NULL,
                PRIMARY KEY (
                    store_namespace_key, tool_config_hash, capacity_scope,
                    capacity_scope_id
                )
            )
            """
        )

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match=r"incompatible Tool admission table "
        r"'whetstone_tool_admission_capacity'.*table definition",
    ):
        ToolAdmissionAuthority.sqlite(database)


def test_sqlite_initialization_rejects_unknown_schema_version(
    tmp_path,
) -> None:
    database = tmp_path / "future.sqlite"
    ToolAdmissionAuthority.sqlite(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE whetstone_tool_admission_schema
            SET version = 3
            WHERE component = 'tool_admission'
            """
        )

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match="expected exact schema version 2, found 3",
    ):
        ToolAdmissionAuthority.sqlite(database)


def test_sqlite_initialization_rejects_real_schema_version(tmp_path) -> None:
    database = tmp_path / "real-version.sqlite"
    ToolAdmissionAuthority.sqlite(database)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            """
            UPDATE whetstone_tool_admission_schema
            SET version = 2.5
            WHERE component = 'tool_admission'
            """
        )

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match="schema version",
    ):
        ToolAdmissionAuthority.sqlite(database)


@pytest.mark.parametrize(
    "field",
    ["max_accepted_calls", "consumed"],
)
def test_sqlite_capacity_decode_rejects_real_counters(
    tmp_path,
    field: str,
) -> None:
    database = tmp_path / f"real-{field}.sqlite"
    config = _config(capacity=2)
    store = _store(database)
    store.admit(_call(config, "accepted"), config)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            f"""
            UPDATE whetstone_tool_admission_capacity
            SET {field} = 1.5
            """
        )

    with pytest.raises(
        RuntimeError,
        match=rf"{field} is not an integer",
    ):
        if field == "max_accepted_calls":
            store.admit(_call(config, "second"), config)
        else:
            store.accepted_count(config, _binding(ToolCapacityScope.RUN))


def test_sqlite_admission_decode_rejects_blob_json(tmp_path) -> None:
    database = tmp_path / "blob-entry.sqlite"
    config = _config(capacity=1)
    call = _call(config, "accepted")
    store = _store(database)
    store.admit(call, config)
    with sqlite3.connect(database) as connection:
        raw = connection.execute(
            """
            SELECT entry_json FROM whetstone_tool_admission_entry
            WHERE store_namespace_key = ? AND call_id = ?
            """,
            (str(call.store_namespace_key), str(call.call_id)),
        ).fetchone()
        assert raw is not None
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            """
            UPDATE whetstone_tool_admission_entry SET entry_json = ?
            WHERE store_namespace_key = ? AND call_id = ?
            """,
            (
                sqlite3.Binary(raw[0].encode()),
                str(call.store_namespace_key),
                str(call.call_id),
            ),
        )

    with pytest.raises(RuntimeError, match="entry is not JSON text"):
        store.get(call)


def _run_spawned_admissions(
    database: Path,
    config: ToolConfig,
    calls: tuple[tuple[str, str], ...],
    *,
    start_method: str,
    hold_transaction: bool,
) -> list[dict[str, Any]]:
    context = cast(Any, multiprocessing.get_context(start_method))
    queue = context.Queue()
    start = context.Event()
    ready = [context.Event() for _ in calls]
    attempted = [context.Event() for _ in calls]
    acquired = [context.Event() for _ in calls]
    processes = [
        context.Process(
            target=admit_once,
            args=(
                str(
                    database.with_name(
                        f"{database.stem}-objects-{index}.sqlite"
                    )
                ),
                str(database),
                config.model_dump(mode="json"),
                call_id,
                template,
                ready[index],
                start,
                attempted[index],
                acquired[index],
                queue,
            ),
        )
        for index, (call_id, template) in enumerate(calls)
    ]
    started: list[Any] = []
    coordinator: sqlite3.Connection | None = None
    try:
        for process in processes:
            process.start()
            started.append(process)
        assert all(signal.wait(timeout=30) for signal in ready)
        if hold_transaction:
            coordinator = sqlite3.connect(database, isolation_level=None)
            coordinator.execute("BEGIN IMMEDIATE")
        start.set()
        assert all(signal.wait(timeout=30) for signal in attempted)
        if coordinator is not None:
            assert not any(signal.is_set() for signal in acquired)
            coordinator.rollback()
            coordinator.close()
            coordinator = None
        assert all(signal.wait(timeout=30) for signal in acquired)
        records = [queue.get(timeout=30) for _ in processes]
        join_processes(processes, timeout=30)
        return records
    finally:
        start.set()
        if coordinator is not None:
            coordinator.rollback()
            coordinator.close()
        terminate_processes(started, timeout=30)


@pytest.mark.sqlite_contention
@pytest.mark.parametrize("start_method", in_process_start_methods())
def test_spawned_sqlite_capacity_race_is_atomic(
    tmp_path: Path,
    start_method: str,
) -> None:
    database = tmp_path / "race.sqlite"
    config = _config(capacity=4)
    # Initialize tables before processes start; each process still opens fully
    # independent ObjectStore and admission-authority instances.
    _store(database)
    records = _run_spawned_admissions(
        database,
        config,
        tuple((f"call-{index}", f"template-{index}") for index in range(12)),
        start_method=start_method,
        hold_transaction=True,
    )
    assert not [record for record in records if "error" in record]
    accepted = [record for record in records if record["state"] == "accepted"]
    refused = [record for record in records if record["state"] == "refused"]
    assert len(accepted) == 4
    assert len(refused) == 8
    assert sorted(record["ordinal"] for record in accepted) == [1, 2, 3, 4]
    assert (
        _store(database).accepted_count(
            config, _binding(ToolCapacityScope.RUN)
        )
        == 4
    )


@pytest.mark.sqlite_contention
@pytest.mark.parametrize("start_method", in_process_start_methods())
def test_spawned_global_capacity_has_one_process_shared_bucket(
    tmp_path: Path,
    start_method: str,
) -> None:
    database = tmp_path / "global-race.sqlite"
    config = _config(capacity=1, scope=ToolCapacityScope.GLOBAL)
    _store(database)
    records = _run_spawned_admissions(
        database,
        config,
        tuple((f"global-{index}", f"template-{index}") for index in range(8)),
        start_method=start_method,
        hold_transaction=False,
    )
    assert not [record for record in records if "error" in record]
    assert sum(record["state"] == "accepted" for record in records) == 1
    assert sum(record["state"] == "refused" for record in records) == 7
    assert (
        _store(database).accepted_count(
            config,
            _binding(ToolCapacityScope.GLOBAL),
        )
        == 1
    )


@pytest.mark.sqlite_contention
@pytest.mark.parametrize("start_method", in_process_start_methods())
def test_spawned_same_call_replay_has_one_ordinal(
    tmp_path: Path,
    start_method: str,
) -> None:
    database = tmp_path / "same.sqlite"
    config = _config(capacity=4)
    _store(database)
    records = _run_spawned_admissions(
        database,
        config,
        (("same", "same-template"),) * 6,
        start_method=start_method,
        hold_transaction=False,
    )
    assert records == [{"state": "accepted", "ordinal": 1} for _ in range(6)]
    assert (
        _store(database).accepted_count(
            config, _binding(ToolCapacityScope.RUN)
        )
        == 1
    )


class _PostgresCursor:
    _columns: ClassVar[dict[str, list[tuple[Any, ...]]]] = {
        "whetstone_tool_admission_schema": [
            (
                "component",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
            ("version", "bigint", "NO", None, None, None, None, None),
        ],
        "whetstone_tool_admission_entry": [
            (
                "store_namespace_key",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
            (
                "call_id",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
            (
                "entry_json",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
        ],
        "whetstone_tool_admission_capacity": [
            (
                "store_namespace_key",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
            (
                "tool_config_hash",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
            (
                "capacity_scope",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
            (
                "capacity_scope_id",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
            (
                "max_accepted_calls",
                "bigint",
                "NO",
                None,
                None,
                None,
                None,
                None,
            ),
            (
                "consumed",
                "bigint",
                "NO",
                None,
                None,
                None,
                None,
                None,
            ),
        ],
    }

    def __init__(self, recorder: _PostgresRecorder) -> None:
        self._recorder = recorder
        self._rows: list[tuple[Any, ...]] = []
        self.rowcount = -1

    def execute(
        self,
        query: str,
        params: tuple[Any, ...] | None = None,
    ) -> None:
        self._recorder.queries.append((query, params))
        self._rows = []
        self.rowcount = -1
        normalized = " ".join(query.split())
        if normalized == "SHOW server_encoding":
            self._rows = [(self._recorder.server_encoding,)]
        elif "FROM information_schema.tables" in normalized:
            self._rows = [(table,) for table in sorted(self._recorder.tables)]
        elif normalized.startswith("CREATE TABLE IF NOT EXISTS"):
            self._recorder.tables.add(normalized.split()[5])
        elif "FROM information_schema.columns" in normalized:
            assert params is not None
            self._rows = list(self._recorder.columns[str(params[0])])
        elif "FROM pg_catalog.pg_constraint" in normalized:
            self._rows = [
                constraint
                for constraint in self._recorder.constraints
                if constraint[0] in self._recorder.tables
            ]
        elif normalized.startswith(
            "SELECT version FROM whetstone_tool_admission_schema"
        ):
            if self._recorder.schema_version is not None:
                self._rows = [(self._recorder.schema_version,)]
        elif normalized.startswith(
            "INSERT INTO whetstone_tool_admission_schema"
        ):
            assert params == ("tool_admission", 2)
            self._recorder.schema_version = 2
        elif normalized.startswith(
            "SELECT component, version FROM whetstone_tool_admission_schema"
        ):
            if self._recorder.schema_version is not None:
                self._rows = [
                    ("tool_admission", self._recorder.schema_version)
                ]
        elif normalized.startswith(
            "SELECT entry_json FROM whetstone_tool_admission_entry"
        ):
            assert params is not None
            entry = self._recorder.entries.get(
                (str(params[0]), str(params[1]))
            )
            if entry is not None:
                self._rows = [(entry,)]
        elif normalized.startswith(
            "INSERT INTO whetstone_tool_admission_capacity"
        ):
            assert params is not None
            scope = tuple(str(value) for value in params[:4])
            self._recorder.capacity.setdefault(
                scope,
                (int(params[4]), 0),
            )
        elif normalized.startswith(
            "SELECT max_accepted_calls, consumed "
            "FROM whetstone_tool_admission_capacity"
        ):
            assert params is not None
            scope = tuple(str(value) for value in params)
            self._rows = [self._recorder.capacity[scope]]
        elif normalized.startswith(
            "UPDATE whetstone_tool_admission_capacity SET consumed"
        ):
            assert params is not None
            scope = tuple(str(value) for value in params[1:])
            maximum, _ = self._recorder.capacity[scope]
            self._recorder.capacity[scope] = (maximum, int(params[0]))
            self.rowcount = 1
        elif normalized.startswith(
            "INSERT INTO whetstone_tool_admission_entry"
        ):
            assert params is not None
            key = (str(params[0]), str(params[1]))
            self._recorder.entries[key] = str(params[2])
            self.rowcount = 1

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows.pop(0) if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows = self._rows
        self._rows = []
        return rows

    def __enter__(self) -> _PostgresCursor:
        return self

    def __exit__(self, *args: object) -> None:
        pass


class _PostgresConnection:
    def __init__(self, recorder: _PostgresRecorder) -> None:
        self._recorder = recorder

    def cursor(self) -> _PostgresCursor:
        return _PostgresCursor(self._recorder)

    def __enter__(self) -> _PostgresConnection:
        self._recorder.entered += 1
        return self

    def __exit__(self, *args: object) -> None:
        self._recorder.exited += 1


class _PostgresRecorder:
    def __init__(self) -> None:
        self.queries: list[tuple[str, tuple[Any, ...] | None]] = []
        self.schema_version: int | None = None
        self.tables: set[str] = set()
        self.columns = {
            table: list(columns)
            for table, columns in _PostgresCursor._columns.items()
        }
        self.constraints = list(tool_store_module._POSTGRES_CONSTRAINTS)
        self.server_encoding = "UTF8"
        self.entries: dict[tuple[str, str], str] = {}
        self.capacity: dict[
            tuple[str, ...],
            tuple[int, int],
        ] = {}
        self.entered = 0
        self.exited = 0

    def connect(self, dsn: str) -> _PostgresConnection:
        assert dsn == "postgresql://tool-admission-test"
        return _PostgresConnection(self)


def test_postgresql_initialization_rejects_truncated_table() -> None:
    recorder = _PostgresRecorder()
    recorder.columns["whetstone_tool_admission_entry"].pop()

    with pytest.raises(
        RuntimeError,
        match="incompatible Tool admission table "
        "'whetstone_tool_admission_entry'",
    ):
        ToolAdmissionAuthority.postgresql(
            "postgresql://tool-admission-test",
            _connect=recorder.connect,
        )


def test_postgresql_initialization_rejects_unaudited_capacity_table() -> None:
    recorder = _PostgresRecorder()
    recorder.tables = {"whetstone_tool_admission_capacity"}
    recorder.capacity[("namespace", "config", "run", "run-1")] = (2, 1)

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match=r"owned table inventory.*whetstone_tool_admission_capacity",
    ):
        ToolAdmissionAuthority.postgresql(
            "postgresql://tool-admission-test",
            _connect=recorder.connect,
        )

    statements = [" ".join(query.split()) for query, _ in recorder.queries]
    assert not any(
        statement.startswith("CREATE TABLE") for statement in statements
    )
    assert recorder.capacity == {
        ("namespace", "config", "run", "run-1"): (2, 1)
    }


def test_postgresql_initialization_requires_utf8_server_encoding() -> None:
    recorder = _PostgresRecorder()
    recorder.server_encoding = "SQL_ASCII"

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match=r"server_encoding.*UTF8.*SQL_ASCII",
    ) as raised:
        ToolAdmissionAuthority.postgresql(
            "postgresql://tool-admission-test",
            _connect=recorder.connect,
        )

    assert raised.value.table == "<database>"
    assert raised.value.aspect == "server_encoding"
    assert [" ".join(query.split()) for query, _ in recorder.queries] == [
        "SHOW server_encoding"
    ]


def test_postgresql_initialization_rejects_non_c_text_collation() -> None:
    recorder = _PostgresRecorder()
    recorder.columns["whetstone_tool_admission_entry"][0] = (
        "store_namespace_key",
        "text",
        "NO",
        "public",
        "case_insensitive",
        "i",
        False,
        -1,
    )

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match=r"whetstone_tool_admission_entry.*columns.*case_insensitive",
    ):
        ToolAdmissionAuthority.postgresql(
            "postgresql://tool-admission-test",
            _connect=recorder.connect,
        )


@pytest.mark.parametrize(
    ("table", "columns", "replacement"),
    [
        ("whetstone_tool_admission_schema", ("version",), None),
        (
            "whetstone_tool_admission_capacity",
            ("capacity_scope",),
            "capacity_scope = ANY (ARRAY['global'::text, 'run'::text])",
        ),
        (
            "whetstone_tool_admission_capacity",
            ("max_accepted_calls",),
            "max_accepted_calls > 0",
        ),
        (
            "whetstone_tool_admission_capacity",
            ("consumed", "max_accepted_calls"),
            "consumed >= 0",
        ),
    ],
    ids=(
        "missing-version-positive",
        "wrong-capacity-scope-enum",
        "wrong-maximum-nonnegative",
        "wrong-consumed-bounds",
    ),
)
def test_postgresql_initialization_rejects_wrong_check_constraints(
    table: str,
    columns: tuple[str, ...],
    replacement: str | None,
) -> None:
    recorder = _PostgresRecorder()
    check_index = next(
        index
        for index, constraint in enumerate(recorder.constraints)
        if constraint[0] == table
        and constraint[1] == "c"
        and constraint[2] == columns
    )
    if replacement is None:
        recorder.constraints.pop(check_index)
    else:
        constraint = recorder.constraints[check_index]
        recorder.constraints[check_index] = (
            *constraint[:3],
            replacement,
            *constraint[4:],
        )

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match=(
            rf"incompatible Tool admission table {table!r}: expected exact "
            r"PRIMARY KEY and CHECK constraints"
        ),
    ) as raised:
        ToolAdmissionAuthority.postgresql(
            "postgresql://tool-admission-test",
            _connect=recorder.connect,
        )

    assert raised.value.table == table
    assert raised.value.aspect == "PRIMARY KEY and CHECK constraints"
    assert "apply the Tool admission schema migration" in str(raised.value)


@pytest.mark.parametrize(
    ("constraint_type", "columns", "flag_index", "wrong_value"),
    [
        (
            "p",
            (
                "store_namespace_key",
                "tool_config_hash",
                "capacity_scope",
                "capacity_scope_id",
            ),
            4,
            True,
        ),
        (
            "p",
            (
                "store_namespace_key",
                "tool_config_hash",
                "capacity_scope",
                "capacity_scope_id",
            ),
            5,
            True,
        ),
        (
            "p",
            (
                "store_namespace_key",
                "tool_config_hash",
                "capacity_scope",
                "capacity_scope_id",
            ),
            6,
            False,
        ),
        (
            "p",
            (
                "store_namespace_key",
                "tool_config_hash",
                "capacity_scope",
                "capacity_scope_id",
            ),
            7,
            False,
        ),
        ("c", ("consumed", "max_accepted_calls"), 4, True),
        ("c", ("consumed", "max_accepted_calls"), 5, True),
        ("c", ("consumed", "max_accepted_calls"), 6, False),
        ("c", ("consumed", "max_accepted_calls"), 7, True),
    ],
    ids=(
        "primary-key-deferrable",
        "primary-key-initially-deferred",
        "primary-key-not-validated",
        "primary-key-wrong-no-inherit",
        "check-deferrable",
        "check-initially-deferred",
        "check-not-validated",
        "check-no-inherit",
    ),
)
def test_postgresql_initialization_rejects_wrong_constraint_flags(
    constraint_type: str,
    columns: tuple[str, ...],
    flag_index: int,
    wrong_value: bool,
) -> None:
    recorder = _PostgresRecorder()
    table = "whetstone_tool_admission_capacity"
    index = next(
        index
        for index, constraint in enumerate(recorder.constraints)
        if constraint[0] == table
        and constraint[1] == constraint_type
        and constraint[2] == columns
    )
    changed = list(recorder.constraints[index])
    changed[flag_index] = wrong_value
    recorder.constraints[index] = tuple(changed)

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match=(
            r"whetstone_tool_admission_capacity.*"
            r"PRIMARY KEY and CHECK constraints"
        ),
    ) as raised:
        ToolAdmissionAuthority.postgresql(
            "postgresql://tool-admission-test",
            _connect=recorder.connect,
        )

    assert "deferrable=" in str(raised.value)
    assert "deferred=" in str(raised.value)
    assert "validated=" in str(raised.value)
    assert "no_inherit=" in str(raised.value)


def test_postgresql_initialization_rejects_unknown_schema_version() -> None:
    recorder = _PostgresRecorder()
    recorder.schema_version = 3
    recorder.tables = {
        "whetstone_tool_admission_schema",
        "whetstone_tool_admission_entry",
        "whetstone_tool_admission_capacity",
    }

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match="expected exact schema version 2, found 3",
    ):
        ToolAdmissionAuthority.postgresql(
            "postgresql://tool-admission-test",
            _connect=recorder.connect,
        )


def test_postgresql_adapter_uses_versioned_schema_and_nul_free_lock(
    tmp_path,
) -> None:
    recorder = _PostgresRecorder()
    authority = ToolAdmissionAuthority.postgresql(
        "postgresql://tool-admission-test",
        _connect=recorder.connect,
    )
    config = _config(
        capacity=1,
        namespace="postgres-adapter",
        scope=ToolCapacityScope.GLOBAL,
    )
    store = ToolCallStore(
        ObjectStore(SqliteBackend(tmp_path / "objects.sqlite")),
        authority,
        EffectAuthority.memory(),
    )

    first = store.admit(
        _call(config, "first", scope_id=GLOBAL_CAPACITY_SCOPE_ID),
        config,
    )
    second = store.admit(
        _call(config, "second", scope_id=GLOBAL_CAPACITY_SCOPE_ID),
        config,
    )

    assert first.state is ToolCallState.ACCEPTED
    assert first.capacity_debit_ordinal == 1
    assert second.state is ToolCallState.REFUSED
    assert recorder.entered == recorder.exited == 3
    statements = "\n".join(query for query, _ in recorder.queries)
    assert "CREATE TABLE IF NOT EXISTS whetstone_tool_admission_schema" in (
        statements
    )
    assert "FROM information_schema.columns" in statements
    assert "FROM pg_catalog.pg_constraint AS constraint_record" in statements
    assert "constraint_record.condeferrable" in statements
    assert "constraint_record.condeferred" in statements
    assert "constraint_record.convalidated" in statements
    assert "constraint_record.connoinherit" in statements
    assert "pg_get_expr(" in statements
    assert "consrc" not in statements
    assert "FOR UPDATE" in statements
    assert "chr(0)" not in statements
    assert "\x00" not in statements
    lock_params = [
        params
        for query, params in recorder.queries
        if " ".join(query.split()) == "SELECT pg_advisory_xact_lock(%s)"
    ]
    assert lock_params == [
        (tool_store_module._entry_lock_key(("postgres-adapter", "first")),),
        (tool_store_module._entry_lock_key(("postgres-adapter", "second")),),
    ]


def test_postgresql_entry_lock_digest_is_pinned_and_unambiguous() -> None:
    assert (
        tool_store_module._entry_lock_key(("namespace", "call"))
        == 5219561813675110560
    )
    assert tool_store_module._entry_lock_key(
        ("a", "bc")
    ) != tool_store_module._entry_lock_key(("ab", "c"))


@pytest.mark.skipif(
    "WHETSTONE_TEST_POSTGRES_DSN" not in os.environ,
    reason=(
        "WHETSTONE_TEST_POSTGRES_DSN is not configured; adapter SQL is "
        "covered separately, but PostgreSQL integration did not run"
    ),
)
def test_postgresql_configured_dsn_admits_with_global_capacity(
    tmp_path,
) -> None:
    authority = ToolAdmissionAuthority.postgresql(
        os.environ["WHETSTONE_TEST_POSTGRES_DSN"]
    )
    namespace = f"tool-admission-test-{uuid4()}"
    config = _config(
        capacity=1,
        namespace=namespace,
        scope=ToolCapacityScope.GLOBAL,
    )
    store = ToolCallStore(
        ObjectStore(SqliteBackend(Path(tmp_path) / "postgres-objects.sqlite")),
        authority,
        EffectAuthority.memory(),
    )

    first = store.admit(
        _call(config, "first", scope_id=GLOBAL_CAPACITY_SCOPE_ID),
        config,
    )
    second = store.admit(
        _call(config, "second", scope_id=GLOBAL_CAPACITY_SCOPE_ID),
        config,
    )

    assert first.state is ToolCallState.ACCEPTED
    assert first.capacity_debit_ordinal == 1
    assert second.state is ToolCallState.REFUSED


@pytest.mark.skipif(
    "WHETSTONE_TEST_POSTGRES_DSN" not in os.environ,
    reason="WHETSTONE_TEST_POSTGRES_DSN is required for live collation checks",
)
def test_postgresql_17_rejects_case_insensitive_admission_schema() -> None:
    from psycopg import connect
    from psycopg.sql import SQL, Identifier

    dsn = os.environ["WHETSTONE_TEST_POSTGRES_DSN"]
    schema = f"tool_ci_{uuid4().hex}"

    @contextmanager
    def connect_in_schema(configured_dsn: str) -> Iterator[Any]:
        with connect(configured_dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    SQL("SET LOCAL search_path TO {}, pg_catalog").format(
                        Identifier(schema)
                    )
                )
            yield connection

    with connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SHOW server_version_num")
            version = cursor.fetchone()
            assert version is not None
            if int(version[0]) // 10_000 != 17:
                pytest.skip(
                    "live case-insensitive check requires PostgreSQL 17"
                )
            cursor.execute(SQL("CREATE SCHEMA {}").format(Identifier(schema)))
            cursor.execute(
                SQL(
                    """
                CREATE COLLATION {}.case_insensitive (
                    provider = icu,
                    locale = 'und-u-ks-level2',
                    deterministic = false
                )
                """
                ).format(Identifier(schema))
            )
            cursor.execute(
                SQL("SET LOCAL search_path TO {}, pg_catalog").format(
                    Identifier(schema)
                )
            )
            for create_sql in (
                tool_store_module._POSTGRES_CREATE_SCHEMA,
                tool_store_module._POSTGRES_CREATE_ENTRY,
                tool_store_module._POSTGRES_CREATE_CAPACITY,
            ):
                cursor.execute(
                    SQL(
                        cast(
                            LiteralString,
                            create_sql.replace(
                                'COLLATE "C"',
                                f'COLLATE "{schema}".case_insensitive',
                            ),
                        )
                    )
                )
            cursor.execute(
                """
                INSERT INTO whetstone_tool_admission_schema (
                    component, version
                ) VALUES (%s, %s)
                """,
                (
                    tool_store_module._SCHEMA_COMPONENT,
                    tool_store_module._SCHEMA_VERSION,
                ),
            )

    try:
        with pytest.raises(
            ToolAdmissionSchemaMismatchError,
            match=r"whetstone_tool_admission_schema.*columns.*"
            r"case_insensitive",
        ):
            ToolAdmissionAuthority.postgresql(
                dsn,
                _connect=connect_in_schema,
            )
    finally:
        with connect(dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    SQL("DROP SCHEMA {} CASCADE").format(Identifier(schema))
                )
