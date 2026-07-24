"""Result Store specialization and the terminal persistence path.

Proves the Rollout Execution Key canonical string encoding, the
absent->bind / same->idempotent / different->conflict contract with no
overwrite path, the complete persistence path (put -> reference -> bind), the
deliberate-re-evaluation-needs-a-new-key rule, and a two-writer concurrency
race with exactly one winner.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from dr_store import (
    MemoryBackend,
    ObjectReference,
    ObjectStore,
    SqliteBackend,
)

from whetstone.graph.rollout import EvaluationRole
from whetstone.result import (
    ROLLOUT_RESULT_SCHEMA,
    ResultBindStatus,
    ResultStore,
    ResultStoreConflictError,
    encode_rollout_execution_key,
    persist_rollout_result,
    rollout_result_reference,
)

from .support import (
    evaluation_context,
    execution_key,
    full_hash,
    success_rollout_result,
)

# ---------------------------------------------------------------------------
# Canonical key encoding
# ---------------------------------------------------------------------------


def test_key_encoding_is_deterministic_and_injective() -> None:
    key = execution_key()
    encoded = encode_rollout_execution_key(key)
    # Deterministic: same key -> same string.
    assert encode_rollout_execution_key(key) == encoded
    # Version-tagged, self-describing canonical JSON array.
    decoded = json.loads(encoded)
    assert decoded[0] == "whetstone.rollout_execution_key/v1"
    assert decoded[1] == key.rollout_key.graph_hash
    assert decoded[-1] == key.evaluation_context_id
    # No incidental whitespace (stable across processes).
    assert " " not in encoded.replace(" ", "") or ": " not in encoded


def test_distinct_keys_encode_distinctly() -> None:
    a = execution_key(task_identity="task-1")
    b = execution_key(task_identity="task-2")
    assert encode_rollout_execution_key(a) != encode_rollout_execution_key(b)


def test_equal_keys_encode_identically() -> None:
    a = execution_key(repeat_id="r0")
    b = execution_key(repeat_id="r0")
    assert encode_rollout_execution_key(a) == encode_rollout_execution_key(b)


# ---------------------------------------------------------------------------
# Binding contract
# ---------------------------------------------------------------------------


def _result_store() -> ResultStore:
    return ResultStore(ObjectStore(MemoryBackend()))


def _reference(char: str) -> ObjectReference:
    return ObjectReference(
        schema=ROLLOUT_RESULT_SCHEMA, content_hash=full_hash(char)
    )


def test_absent_key_binds() -> None:
    rstore = _result_store()
    key = execution_key()
    status = rstore.bind(key, _reference("a"))
    assert status is ResultBindStatus.BOUND
    assert rstore.resolve(key) == _reference("a")


def test_same_reference_replays_idempotently() -> None:
    rstore = _result_store()
    key = execution_key()
    ref = _reference("a")
    assert rstore.bind(key, ref) is ResultBindStatus.BOUND
    assert rstore.bind(key, ref) is ResultBindStatus.IDEMPOTENT
    assert rstore.bind(key, ref) is ResultBindStatus.IDEMPOTENT


def test_different_reference_conflicts_preserving_winner() -> None:
    rstore = _result_store()
    key = execution_key()
    winner = _reference("a")
    loser = _reference("b")
    assert rstore.bind(key, winner) is ResultBindStatus.BOUND
    with pytest.raises(ResultStoreConflictError) as excinfo:
        rstore.bind(key, loser)
    error = excinfo.value
    assert error.existing == winner
    assert error.requested == loser
    # The winner is preserved and never overwritten.
    assert rstore.resolve(key) == winner


def test_resolve_returns_none_when_unbound() -> None:
    rstore = _result_store()
    assert rstore.resolve(execution_key()) is None


def test_bind_rejects_non_rollout_result_reference() -> None:
    rstore = _result_store()
    wrong = ObjectReference(
        schema="whetstone.materialization_record",
        content_hash=full_hash("a"),
    )
    with pytest.raises(ValueError, match="typed Rollout Result Object"):
        rstore.bind(execution_key(), wrong)


def test_no_overwrite_or_clear_api() -> None:
    """The Result Store exposes no overwrite/clear/rebind/delete surface."""
    public = {name for name in dir(ResultStore) if not name.startswith("_")}
    for forbidden in (
        "overwrite",
        "clear",
        "rebind",
        "delete",
        "remove",
        "unbind",
        "replace",
        "set",
        "update",
    ):
        assert forbidden not in public, forbidden
    assert public == {"bind", "resolve", "store"}


# ---------------------------------------------------------------------------
# Persistence path
# ---------------------------------------------------------------------------


def test_persist_puts_then_binds() -> None:
    rstore = _result_store()
    result = success_rollout_result()
    reference, status = persist_rollout_result(rstore, result)
    assert status is ResultBindStatus.BOUND
    assert reference == rollout_result_reference(result)
    # The reference resolves back to the exact stored record.
    assert rstore.store.get(reference) == result.record_content()
    # The key is bound to that reference.
    assert rstore.resolve(result.rollout_execution_key) == reference


def test_persist_same_result_replays_idempotently() -> None:
    rstore = _result_store()
    result = success_rollout_result()
    ref1, status1 = persist_rollout_result(rstore, result)
    ref2, status2 = persist_rollout_result(rstore, result)
    assert status1 is ResultBindStatus.BOUND
    assert status2 is ResultBindStatus.IDEMPOTENT
    assert ref1 == ref2


def test_same_key_different_result_conflicts_not_supersedes() -> None:
    """Same Rollout Execution Key + a different Result conflicts.

    A divergent terminal Result for the same execution identity does not
    supersede the durable winner; the binding is preserved and a typed
    conflict is raised.
    """
    rstore = _result_store()
    key = execution_key()
    winner = success_rollout_result(key=key, score=1.0)
    divergent = success_rollout_result(key=key, score=99.0)
    assert rollout_result_reference(winner) != rollout_result_reference(
        divergent
    )
    winner_ref, _ = persist_rollout_result(rstore, winner)
    with pytest.raises(ResultStoreConflictError) as excinfo:
        persist_rollout_result(rstore, divergent)
    assert excinfo.value.existing == winner_ref
    # The winner is still bound; the divergent Result did not overwrite it.
    assert rstore.resolve(key) == winner_ref


def test_reevaluation_requires_new_evaluation_context_id() -> None:
    """A new Evaluation Context ID yields a new key; both Results coexist."""
    rstore = _result_store()
    ctx_a = evaluation_context(campaign="camp-a")
    ctx_b = evaluation_context(campaign="camp-b")
    key_a = execution_key(context=ctx_a)
    key_b = execution_key(context=ctx_b)
    # Different Evaluation Context IDs -> different Rollout Execution Keys.
    assert key_a.evaluation_context_id != key_b.evaluation_context_id
    ref_a, status_a = persist_rollout_result(
        rstore, success_rollout_result(key=key_a)
    )
    ref_b, status_b = persist_rollout_result(
        rstore, success_rollout_result(key=key_b)
    )
    assert status_a is ResultBindStatus.BOUND
    assert status_b is ResultBindStatus.BOUND
    # Both keys are independently bound; neither superseded the other.
    assert rstore.resolve(key_a) == ref_a
    assert rstore.resolve(key_b) == ref_b


def test_reevaluation_requires_new_repeat_id() -> None:
    """A new Repeat ID yields a new key; both Results coexist."""
    rstore = _result_store()
    key_r0 = execution_key(repeat_id="r0")
    key_r1 = execution_key(repeat_id="r1")
    assert encode_rollout_execution_key(
        key_r0
    ) != encode_rollout_execution_key(key_r1)
    persist_rollout_result(rstore, success_rollout_result(key=key_r0))
    persist_rollout_result(rstore, success_rollout_result(key=key_r1))
    assert rstore.resolve(key_r0) is not None
    assert rstore.resolve(key_r1) is not None
    assert rstore.resolve(key_r0) != rstore.resolve(key_r1)


def test_official_authority_result_persists() -> None:
    ctx = evaluation_context(
        role=EvaluationRole.OFFICIAL, authority="lab-authority"
    )
    key = execution_key(context=ctx)
    result = success_rollout_result(key=key)
    rstore = _result_store()
    _ref, status = persist_rollout_result(rstore, result)
    assert status is ResultBindStatus.BOUND


# ---------------------------------------------------------------------------
# Concurrency: two writers, one winner
# ---------------------------------------------------------------------------


def test_two_writers_one_winner_same_reference(tmp_path) -> None:
    """Two threads binding the SAME reference: one BOUND, rest IDEMPOTENT."""
    store = ObjectStore(SqliteBackend(tmp_path / "same.db"))
    rstore = ResultStore(store)
    key = execution_key()
    ref = _reference("a")

    def worker(_: int) -> ResultBindStatus:
        return rstore.bind(key, ref)

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(worker, range(16)))

    bound = [s for s in statuses if s is ResultBindStatus.BOUND]
    idempotent = [s for s in statuses if s is ResultBindStatus.IDEMPOTENT]
    assert len(bound) == 1
    assert len(idempotent) == 15
    assert rstore.resolve(key) == ref


def test_two_writers_one_winner_different_references(tmp_path) -> None:
    """Two threads race distinct references for one key: exactly one wins."""
    store = ObjectStore(SqliteBackend(tmp_path / "diff.db"))
    rstore = ResultStore(store)
    key = execution_key()
    references = [_reference(c) for c in "abcdef012345"]

    winners: list[ObjectReference] = []
    conflicts: list[ObjectReference] = []

    def worker(ref: ObjectReference) -> None:
        try:
            rstore.bind(key, ref)
            winners.append(ref)
        except ResultStoreConflictError as exc:
            conflicts.append(exc.requested)
            # Every conflict preserves the same durable winner.
            assert exc.existing == rstore.resolve(key)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, references))

    assert len(winners) == 1
    assert len(conflicts) == len(references) - 1
    bound = rstore.resolve(key)
    assert bound == winners[0]
    # No conflicting candidate ever became the terminal binding.
    assert bound not in conflicts
