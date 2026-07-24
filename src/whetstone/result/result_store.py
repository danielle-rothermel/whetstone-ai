"""The Whetstone Result Store: a thin specialization of dr-store binding.

The Result Store is the authoritative Whetstone-owned specialization of
dr-store's generic atomic key-to-reference binding. It owns exactly three
Whetstone concerns on top of that generic primitive and no more:

1. **Key construction.** The binding key is one Rollout Execution Key, in a
   canonical string encoding this module defines
   (:func:`encode_rollout_execution_key`). dr-store sees only an opaque
   string; Whetstone owns its meaning and encoding.

2. **Typed value.** The bound value is a typed Rollout Result Object
   Reference — a :class:`~dr_store.ObjectReference` under the
   ``whetstone.rollout_result`` record schema carrying the Rollout Result's
   Content Hash.

3. **The specialized binding contract, unchanged from dr-store:**
   absent key -> atomically bind (:data:`ResultBindStatus.BOUND`); same
   reference -> idempotent success (:data:`ResultBindStatus.IDEMPOTENT`);
   different reference -> typed conflict that preserves the durable winner
   (:class:`ResultStoreConflictError`) and never overwrites. There is NO
   overwrite/clear/rebind API.

Whetstone puts no Whetstone policy into dr-store: dr-store still exposes only
its generic ``bind``/``resolve``. The Result Store wraps them, translating the
Rollout Execution Key to the canonical string and re-typing the generic
:class:`~dr_store.BindingConflictError` as a Whetstone
:class:`ResultStoreConflictError`.

Deliberate re-evaluation is a *new key*, never an overwrite: to produce a
second terminal Result for the same measurement cell you must vary the
Evaluation Context ID or the Repeat ID, which changes the Rollout Execution
Key. Binding a different Result under the *same* key conflicts rather than
supersedes.
"""

from __future__ import annotations

import enum
import json
from typing import TYPE_CHECKING

from dr_store import (
    BindingConflictError,
    BindStatus,
    ObjectReference,
    ObjectStore,
)

from whetstone.result.rollout_result import rollout_result_reference
from whetstone.result.schema import ROLLOUT_RESULT_SCHEMA

if TYPE_CHECKING:
    from whetstone.graph.rollout import RolloutExecutionKey
    from whetstone.result.rollout_result import RolloutResult

__all__ = [
    "ROLLOUT_RESULT_SCHEMA",
    "ResultBindStatus",
    "ResultBinding",
    "ResultStore",
    "ResultStoreConflictError",
    "encode_rollout_execution_key",
    "persist_rollout_result",
]


# Version tag on the canonical key encoding so the string form is
# self-describing and can never be confused with a differently versioned
# encoding.
_KEY_ENCODING_VERSION = "whetstone.rollout_execution_key/v1"


def encode_rollout_execution_key(key: RolloutExecutionKey) -> str:
    """Canonically encode a Rollout Execution Key as an opaque store key.

    The encoding is deterministic and injective: two Rollout Execution Keys
    encode to the same string if and only if they are equal. It is a
    version-tagged canonical JSON array of the key's four measurement fields
    plus the Evaluation Context ID, produced with sorted keys and no
    incidental whitespace so it is stable across processes.

    dr-store treats this string as opaque; Whetstone alone assigns it meaning.
    """
    rollout_key = key.rollout_key
    payload = [
        _KEY_ENCODING_VERSION,
        rollout_key.graph_hash,
        rollout_key.eval_config_hash,
        rollout_key.task_identity,
        rollout_key.repeat_id,
        key.evaluation_context_id,
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class ResultBindStatus(enum.Enum):
    """Non-conflicting outcome of binding a terminal Rollout Result.

    ``BOUND`` -- an unbound Rollout Execution Key acquired this reference.
    ``IDEMPOTENT`` -- the key was already bound to the *same* reference; the
    replay is success without any replacement write. A different reference is
    not a status: it raises :class:`ResultStoreConflictError`.
    """

    BOUND = "bound"
    IDEMPOTENT = "idempotent"


class ResultStoreConflictError(Exception):
    """A Rollout Execution Key is already bound to a different Result.

    The existing binding (the durable winner) is preserved unchanged and
    exposed as ``existing`` so the caller can inspect it; ``requested`` is the
    losing candidate. There is no overwrite path: a divergent Result never
    supersedes the winner or authorizes a retry replacement.
    """

    def __init__(
        self,
        *,
        key: RolloutExecutionKey,
        encoded_key: str,
        existing: ObjectReference,
        requested: ObjectReference,
    ) -> None:
        self.key = key
        self.encoded_key = encoded_key
        self.existing = existing
        self.requested = requested
        super().__init__(
            f"Rollout Execution Key {encoded_key} already bound to "
            f"({existing.schema!r}, {existing.content_hash}); refusing to "
            f"bind divergent Result ({requested.schema!r}, "
            f"{requested.content_hash})"
        )


class ResultBinding:
    """Authoritative Whetstone Result Store over a dr-store ObjectStore.

    Owns key construction, the typed Rollout Result Object Reference value,
    and the absent->bind / same->idempotent / different->conflict contract.
    Delegates all durability and atomicity to the underlying
    :class:`~dr_store.ObjectStore`; adds no second persistence path.
    """

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    @property
    def store(self) -> ObjectStore:
        """The underlying dr-store ObjectStore (for result puts)."""
        return self._store

    def bind(
        self,
        key: RolloutExecutionKey,
        reference: ObjectReference,
    ) -> ResultBindStatus:
        """Atomically bind a Rollout Execution Key to a Result reference.

        The value MUST be a typed Rollout Result Object Reference (a
        reference under the ``whetstone.rollout_result`` record schema).
        Absent key -> bind (``BOUND``); same reference -> ``IDEMPOTENT``;
        different reference -> :class:`ResultStoreConflictError` preserving
        the winner. No overwrite path exists.
        """
        if reference.schema != ROLLOUT_RESULT_SCHEMA:
            raise ValueError(
                "Result Store binds only typed Rollout Result Object "
                f"References under schema {ROLLOUT_RESULT_SCHEMA!r}, got "
                f"{reference.schema!r}"
            )
        encoded = encode_rollout_execution_key(key)
        try:
            status = self._store.bind(encoded, reference)
        except BindingConflictError as conflict:
            raise ResultStoreConflictError(
                key=key,
                encoded_key=encoded,
                existing=conflict.existing,
                requested=conflict.requested,
            ) from conflict
        if status is BindStatus.BOUND:
            return ResultBindStatus.BOUND
        return ResultBindStatus.IDEMPOTENT

    def resolve(self, key: RolloutExecutionKey) -> ObjectReference | None:
        """Return the Result reference bound to ``key``, or ``None``.

        Read-only; never mutates and exposes no overwrite path. A ``None``
        result means the key is unbound (an operator retry precondition).
        """
        return self._store.resolve(encode_rollout_execution_key(key))


# ``ResultStore`` is the canonical public name for the specialization; the
# implementation lives on ``ResultBinding``.
ResultStore = ResultBinding


def persist_rollout_result(
    result_store: ResultBinding,
    result: RolloutResult,
) -> tuple[ObjectReference, ResultBindStatus]:
    """Persist and uniquely bind one complete terminal Rollout Result.

    The complete persistence path, in one authoritative sequence:

    1. Immutably ``put`` the complete Rollout Result through dr-store, which
       verifies its Content Hash and returns the typed Rollout Result Object
       Reference carrying that Content Hash.
    2. Atomically ``bind`` that reference under the Rollout Execution Key
       through the Result Store compare-and-set.

    Returns the reference and the bind status. Raises
    :class:`ResultStoreConflictError` when the key is already bound to a
    *different* Result — deliberate re-evaluation requires a new Evaluation
    Context ID or Repeat ID (a new key), never an overwrite of an existing
    binding.

    The nested Graph Run Result is persisted only as part of this one
    enclosing record; it has no separate authoritative persistence path.
    """
    reference, _put_status = result_store.store.put(
        ROLLOUT_RESULT_SCHEMA, result.record_content()
    )
    # Cross-check: the reference the store computed matches the reference the
    # record resolves under, so a caller cannot bind a Result under a
    # mismatched reference.
    expected = rollout_result_reference(result)
    if reference != expected:
        raise ValueError(
            "stored Rollout Result reference does not match the record's own "
            "content-addressed reference"
        )
    status = result_store.bind(result.rollout_execution_key, reference)
    return reference, status
