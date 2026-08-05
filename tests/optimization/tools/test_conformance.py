"""Public Tool admission semantics shared by every backend."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from tests.optimization.tools.conformance_support import (
    ToolAdmissionBackend,
    conformance_store,
)
from tests.optimization.tools.support import (
    capacity_binding,
    successful_result,
    tool_call,
    tool_config,
)
from whetstone.optimization.tools.admission import (
    ToolCallState,
    ToolCallStoreConflictError,
    tool_effect_request,
)
from whetstone.optimization.tools.contracts import (
    RefusalClass,
    ToolCapacityScope,
)
from whetstone.optimization.tools.facade import ToolCallStore


@pytest.fixture(
    name="store",
    params=("memory", "sqlite", "postgresql"),
)
def store_fixture(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[ToolCallStore]:
    backend: ToolAdmissionBackend = request.param
    with conformance_store(backend, tmp_path) as store:
        yield store


def test_admit_replay(store: ToolCallStore) -> None:
    config = tool_config(capacity=2)
    call = tool_call(config, "replay")

    admitted = store.admit(call, config)

    assert admitted.state is ToolCallState.ACCEPTED
    assert admitted.capacity_debit_ordinal == 1
    assert store.admit(call, config) == admitted
    assert (
        store.accepted_count(
            config,
            capacity_binding(ToolCapacityScope.RUN),
        )
        == 1
    )


def test_capacity_refusal(store: ToolCallStore) -> None:
    config = tool_config(capacity=1)

    accepted = store.admit(tool_call(config, "accepted"), config)
    refused = store.admit(tool_call(config, "refused"), config)

    assert accepted.state is ToolCallState.ACCEPTED
    assert refused.state is ToolCallState.REFUSED
    assert refused.capacity_debit_ordinal is None
    assert refused.refusal is not None
    assert refused.refusal.refusal_class is RefusalClass.CAPACITY


def test_namespace_conflict(store: ToolCallStore) -> None:
    config = tool_config(capacity=2, namespace="conflict")
    store.admit(tool_call(config, "same", template="first"), config)

    with pytest.raises(ToolCallStoreConflictError, match="different exact"):
        store.admit(tool_call(config, "same", template="second"), config)


def test_terminal_replay(store: ToolCallStore) -> None:
    config = tool_config(capacity=1)
    call = tool_call(config, "terminal")
    store.admit(call, config)
    result = successful_result(call, 1)
    acquisition = store.effect_authority.acquire(
        tool_effect_request(call),
        owner_id="conformance-owner",
        attempt_id="conformance-attempt",
        lease_duration=timedelta(minutes=1),
    )
    assert acquisition.lease is not None
    terminal = store.effect_authority.succeed(
        acquisition.lease,
        result_ref=store.persist_result(result),
    )

    completed = store.complete(result, terminal=terminal)

    assert store.complete(result, terminal=terminal) == completed
    assert store.load_terminal_result(completed) == result


def test_scope_isolation(store: ToolCallStore) -> None:
    config = tool_config(capacity=1)

    first = store.admit(tool_call(config, "first", scope_id="run-a"), config)
    second = store.admit(
        tool_call(config, "second", scope_id="run-b"),
        config,
    )

    assert first.capacity_debit_ordinal == 1
    assert second.capacity_debit_ordinal == 1
    assert (
        store.accepted_count(
            config,
            capacity_binding(ToolCapacityScope.RUN, "run-a"),
        )
        == 1
    )
    assert (
        store.accepted_count(
            config,
            capacity_binding(ToolCapacityScope.RUN, "run-b"),
        )
        == 1
    )
