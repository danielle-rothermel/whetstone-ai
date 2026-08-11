from __future__ import annotations

import json
import multiprocessing
import sqlite3
from datetime import timedelta

import pytest
from dr_store import (
    ObjectConflictError,
    ObjectStore,
    SqliteBackend,
)
from pydantic import ValidationError

from tests.optimization.processes import (
    join_processes,
    terminate_processes,
)
from tests.optimization.tools.store_spawn import (
    load_terminal_result_once,
)
from tests.optimization.tools.support import (
    CollisionBackend,
    capacity_binding,
    reward_record,
    sqlite_store,
    successful_result,
    tool_call,
    tool_config,
)
from whetstone.core.effects.authority import (
    EffectAuthority,
    TerminalConflictError,
)
from whetstone.core.identity import typed_ref_for_record
from whetstone.experiment.reward import (
    REWARD_SCHEMA,
    Reward,
    RewardPolicy,
    RewardTerm,
    reward_reference,
)
from whetstone.optimization.tools.admission import (
    ToolCallState,
    ToolCallStoreConflictError,
    ToolCallStoreEntry,
    tool_effect_request,
)
from whetstone.optimization.tools.contracts import (
    GLOBAL_CAPACITY_SCOPE_ID,
    RUN_CAPACITY_SUBJECT_SCHEMA,
    RefusalClass,
    ToolCapacityBinding,
    ToolCapacityScope,
    ToolRefusal,
    ToolResult,
    tool_call_reference,
)
from whetstone.optimization.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)

FULL_A = "a" * 64
FULL_B = "b" * 64


def test_admission_replay_debits_once_and_completion_is_exact(
    tmp_path,
) -> None:
    database = tmp_path / "tool.sqlite"
    config = tool_config(capacity=2)
    call = tool_call(config, "c1")
    store = sqlite_store(database)

    accepted = store.admit(call, config)
    assert accepted.state is ToolCallState.ACCEPTED
    assert accepted.capacity_debit_ordinal == 1
    assert store.admit(call, config) == accepted
    assert (
        store.accepted_count(config, capacity_binding(ToolCapacityScope.RUN))
        == 1
    )

    result = successful_result(call, 1)
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
    wrong_request["effect_terminal"]["request"]["request_hash"] = FULL_B
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


@pytest.mark.process_integration
def test_completed_terminal_survives_spawned_restart(tmp_path) -> None:
    database = tmp_path / "spawned-completion.sqlite"
    authority = EffectAuthority.sqlite(database)
    store = sqlite_store(database, effect_authority=authority)
    config = tool_config(capacity=1)
    call = tool_call(config, "spawned-restart")
    store.admit(call, config)
    result = successful_result(call, 1)
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
    started = []
    try:
        process.start()
        started.append(process)
        join_processes(started, timeout=30)
        record = queue.get(timeout=5)
        assert "error" not in record
        assert record["entry"] == completed.model_dump(mode="json")
        assert record["result"] == result.model_dump(mode="json")
    finally:
        terminate_processes(started, timeout=30)


def test_poisoned_admission_terminal_fails_bound_authority_verification(
    tmp_path,
) -> None:
    database = tmp_path / "poisoned-completion.sqlite"
    authority = EffectAuthority.sqlite(database)
    store = sqlite_store(database, effect_authority=authority)
    config = tool_config(capacity=1)
    call = tool_call(config, "poisoned")
    store.admit(call, config)
    result = successful_result(call, 1)
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

    restarted = sqlite_store(
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
    store = sqlite_store(tmp_path / "canonical.sqlite")
    config = tool_config(capacity=1)
    call = tool_call(config, "foreign-proof")
    store.admit(call, config)
    result = successful_result(call, 1)
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
    config = tool_config(reward_policy_hash=str(policy.identity_hash()))
    call = tool_call(config, "rewarded")
    reward = reward_reference(reward_record(policy))
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


def test_persist_result_rejects_embedded_reward_collision() -> None:
    policy = RewardPolicy(
        policy_name="tool",
        terms=(RewardTerm(name="score", weight=1.0),),
    )
    config = tool_config(reward_policy_hash=str(policy.identity_hash()))
    reward = reward_reference(reward_record(policy))
    result = ToolResult(
        call=tool_call_reference(tool_call(config, "collision")),
        output={"rollout_refs": [], "accepted_ordinal": 1},
        reward=reward,
        evaluation_evidence_refs=reward.record.evidence_refs,
        provenance_ordinal=1,
    )
    store = ToolCallStore(
        ObjectStore(CollisionBackend()),
        ToolAdmissionAuthority.memory(),
        EffectAuthority.memory(),
    )

    with pytest.raises(ObjectConflictError):
        store.persist_result(result)


def test_capacity_refusal_is_terminal_exact_and_consumes_nothing(
    tmp_path,
) -> None:
    database = tmp_path / "tool.sqlite"
    config = tool_config(capacity=1)
    store = sqlite_store(database)
    store.admit(tool_call(config, "accepted"), config)

    refused = store.admit(tool_call(config, "refused"), config)
    assert refused.state is ToolCallState.REFUSED
    assert refused.capacity_debit_ordinal is None
    assert refused.refusal is not None
    assert refused.refusal.refusal_class is RefusalClass.CAPACITY
    result = store.load_terminal_result(refused)
    assert result.refusal == refused.refusal
    assert result.call == refused.tool_call
    assert (
        store.accepted_count(config, capacity_binding(ToolCapacityScope.RUN))
        == 1
    )


def test_refused_terminal_load_rejects_mismatched_persisted_refusal(
    tmp_path,
) -> None:
    database = tmp_path / "refusal-poison.sqlite"
    config = tool_config(capacity=1)
    call = tool_call(config, "refused")
    store = sqlite_store(database)
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
    config = tool_config(capacity=1)
    store = sqlite_store(database)
    refused = store.refuse(
        tool_call(config, "refused"),
        config,
        refusal=ToolRefusal(
            refusal_class=RefusalClass.AUTHORIZATION,
            reason="not authorized",
        ),
    )

    assert refused.state is ToolCallState.REFUSED
    assert refused.capacity_debit_ordinal is None
    assert (
        store.accepted_count(config, capacity_binding(ToolCapacityScope.RUN))
        == 0
    )
    accepted = store.admit(tool_call(config, "accepted"), config)
    assert accepted.capacity_debit_ordinal == 1


def test_completed_result_is_terminally_immutable(tmp_path) -> None:
    database = tmp_path / "completed.sqlite"
    config = tool_config(capacity=1)
    call = tool_call(config, "call")
    store = sqlite_store(database)
    store.admit(call, config)
    first_result = successful_result(call, 1)
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
    assert (
        store.accepted_count(config, capacity_binding(ToolCapacityScope.RUN))
        == 1
    )


def test_completion_rejects_provenance_ordinal_mismatch(
    tmp_path,
) -> None:
    database = tmp_path / "provenance-poison.sqlite"
    config = tool_config(capacity=1)
    call = tool_call(config, "call")
    store = sqlite_store(database)
    store.admit(call, config)
    result = successful_result(call, 2)
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
    config = tool_config(capacity=1)
    call = tool_call(config, "call")
    store = sqlite_store(database)
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
    config = tool_config(capacity=1)
    call = tool_call(config, "call")
    store = sqlite_store(database)
    store.admit(call, config)
    result = successful_result(call, 1)
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
    config = tool_config(capacity=1)
    call = tool_call(config, "refused")
    store = sqlite_store(database)
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
    config = tool_config(capacity=2)
    first = sqlite_store(database)
    first.admit(tool_call(config, "same", template="one"), config)

    second = sqlite_store(database)
    with pytest.raises(ToolCallStoreConflictError, match="different exact"):
        second.admit(tool_call(config, "same", template="two"), config)
    assert (
        second.accepted_count(config, capacity_binding(ToolCapacityScope.RUN))
        == 1
    )


def test_config_namespace_and_scope_chain_are_exact(tmp_path) -> None:
    database = tmp_path / "tool.sqlite"
    config = tool_config(capacity=1, namespace="namespace-a")
    store = sqlite_store(database)
    call = tool_call(config, "same", scope_id="run-a")
    store.admit(call, config)

    other_config = tool_config(capacity=1, namespace="namespace-b")
    with pytest.raises(ValueError, match="exact supplied Tool Config"):
        store.admit(call, other_config)

    assert (
        store.accepted_count(
            config, capacity_binding(ToolCapacityScope.RUN, "run-a")
        )
        == 1
    )
    assert (
        store.accepted_count(
            config, capacity_binding(ToolCapacityScope.RUN, "run-b")
        )
        == 0
    )
    separate_scope = store.admit(
        tool_call(config, "other", scope_id="run-b"), config
    )
    assert separate_scope.capacity_debit_ordinal == 1


def test_global_scope_rejects_caller_partitioning(tmp_path) -> None:
    database = tmp_path / "global.sqlite"
    config = tool_config(capacity=1, scope=ToolCapacityScope.GLOBAL)
    store = sqlite_store(database)

    with pytest.raises(ValueError, match="requires no subject_ref"):
        ToolCapacityBinding(
            scope=ToolCapacityScope.GLOBAL,
            subject_ref=typed_ref_for_record(
                RUN_CAPACITY_SUBJECT_SCHEMA, {"subject": "forged"}
            ),
        )

    first = store.admit(
        tool_call(config, "first", scope_id=GLOBAL_CAPACITY_SCOPE_ID),
        config,
    )
    second = store.admit(
        tool_call(config, "second", scope_id=GLOBAL_CAPACITY_SCOPE_ID),
        config,
    )
    assert first.state is ToolCallState.ACCEPTED
    assert second.state is ToolCallState.REFUSED
    assert (
        store.accepted_count(
            config,
            capacity_binding(ToolCapacityScope.GLOBAL),
        )
        == 1
    )
