"""Whetstone terminal Rollout Result and the Result Store specialization.

This package owns the two Whetstone concerns that sit on top of dr-store's
generic content-addressed persistence and generic atomic key-to-reference
binding:

* :mod:`whetstone.result.rollout_result` — the immutable terminal
  :class:`RolloutResult` schema. It maps directly to one Rollout and its
  Rollout Execution Key, nests a native dr-graph
  :class:`~dr_graph.GraphRunResult` (referencing — never duplicating —
  provider bodies the enclosing Rollout Result holds), and carries either
  Metric Facts plus named Scores or an exhausted causal failure, together
  with Provider Call Attempt observations and Platform Stage Attempt /
  Durability Replay evidence. It carries NO Materialization Record reference
  and NO official-specific result role or type.

* :mod:`whetstone.result.result_store` — the authoritative Whetstone-owned
  Result Store: a thin domain specialization of dr-store's atomic binding
  whose key is the Rollout Execution Key (in a canonical string encoding
  Whetstone owns) and whose value is a typed Rollout Result Object
  Reference. Absent binds; the same reference replays idempotently; a
  different reference conflicts and never overwrites the winner. It exposes
  the complete-Result persistence path (put through dr-store -> Object
  Reference -> bind under the key) and NO overwrite/clear API.
"""

from __future__ import annotations

from whetstone.result.result_store import (
    ResultBinding,
    ResultBindStatus,
    ResultStore,
    ResultStoreConflictError,
    encode_rollout_execution_key,
    persist_rollout_result,
)
from whetstone.result.rollout_result import (
    ExhaustedCausalFailure,
    PlatformStageAttemptEvidence,
    ProviderCallAttemptObservation,
    RolloutResult,
    ScoreFact,
    rollout_result_reference,
)
from whetstone.result.schema import ROLLOUT_RESULT_SCHEMA

__all__ = [
    "ROLLOUT_RESULT_SCHEMA",
    "ExhaustedCausalFailure",
    "PlatformStageAttemptEvidence",
    "ProviderCallAttemptObservation",
    "ResultBindStatus",
    "ResultBinding",
    "ResultStore",
    "ResultStoreConflictError",
    "RolloutResult",
    "ScoreFact",
    "encode_rollout_execution_key",
    "persist_rollout_result",
    "rollout_result_reference",
]
