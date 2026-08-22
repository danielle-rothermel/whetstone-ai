"""``admitted_entries`` names which calls a scope paid for, on every backend.

``accepted_count`` says how many evaluations a scope bought;
``admitted_entries`` says which ones, and the Codex adapter reconciles an
agent's self-report against it. That makes three properties load-bearing,
and all three are backend-implemented rather than shared: the memory
backend filters a dict, while the SQLite and PostgreSQL backends narrow
by namespace in SQL and decode the rest. So the same behavioral test runs
against all three.

The properties: entries are scoped by the full
``(namespace, config hash, scope, scope_id)`` key, so two runs sharing one
Tool Config cannot read each other's ledger; they come back in
``capacity_debit_ordinal`` order; and an over-cap REFUSED call debited no
capacity, so it is absent -- which is what makes ``len(admitted_entries)``
agree with ``accepted_count``.

The PostgreSQL case is marked ``integration`` and runs against the
database named by ``WHETSTONE_TEST_DATABASE_URL``.
"""

from __future__ import annotations

import os

import pytest
from dr_store.sync import open_sqlite
from sqlalchemy import make_url

from tests.codex_support import (
    toy_capacity_binding,
    toy_codex_control,
    toy_codex_run,
    toy_tool_args,
)
from whetstone.core.identity import ImmutableJsonObject
from whetstone.core.leasing import EffectLeaseAuthority
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.tools.admission import (
    _CAPACITY_TABLE,
    _ENTRY_TABLE,
    _SCHEMA_TABLE,
    ToolCallState,
)
from whetstone.optim.tools.contracts import (
    ToolCall,
    tool_config_reference,
)
from whetstone.optim.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)

MAX_ACCEPTED_CALLS = 2

#: The two runs deliberately share one ``store_namespace_key`` so they
#: share a Tool Config identity. Only the capacity scope id -- the run
#: ref -- separates their ledgers, which is exactly the isolation
#: ``admitted_entries`` has to enforce on its own.
SHARED_NAMESPACE = "whetstone.codex.mcp:admitted-entries-shared"


def _libpq_dsn(database_url: str) -> str:
    """Strip the SQLAlchemy driver suffix to get a plain libpq DSN."""
    url = make_url(database_url).set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


def _postgres_authority() -> ToolAdmissionAuthority:
    database_url = os.environ.get(
        "WHETSTONE_TEST_DATABASE_URL",
        "postgresql+psycopg:///whetstone_platform_test",
    )
    dsn = _libpq_dsn(database_url)
    # The admission tables are durable and shared across the integration
    # suite, and one sibling test deliberately leaves malformed ones
    # behind to prove mismatch detection. Dropping the whole owned
    # inventory first makes this test independent of suite order and of
    # whatever rows a previous run left in the shared namespace; the
    # backend recreates all three on ``initialize``.
    import psycopg

    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"DROP TABLE IF EXISTS {_ENTRY_TABLE}, "
                f"{_CAPACITY_TABLE}, {_SCHEMA_TABLE} CASCADE"
            )
        connection.commit()
    return ToolAdmissionAuthority.postgresql(dsn)


def _memory_authority(tmp_path) -> ToolAdmissionAuthority:
    return ToolAdmissionAuthority.memory()


def _sqlite_authority(tmp_path) -> ToolAdmissionAuthority:
    return ToolAdmissionAuthority.sqlite(str(tmp_path / "admission.sqlite"))


BACKENDS = [
    pytest.param(_memory_authority, id="memory"),
    pytest.param(_sqlite_authority, id="sqlite"),
    pytest.param(
        lambda tmp_path: _postgres_authority(),
        id="postgres",
        marks=pytest.mark.integration,
    ),
]


def _call(*, call_id, config, run, candidate, engine, template) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        tool_config=tool_config_reference(config),
        capacity_binding=toy_capacity_binding(run),
        args=ImmutableJsonObject(
            toy_tool_args(
                candidate=candidate, engine=engine, template=template
            )
        ),
    )


@pytest.mark.parametrize("build_authority", BACKENDS)
def test_admitted_entries_names_exactly_what_this_scope_paid_for(
    build_authority, tmp_path
) -> None:
    """One shared behavioral contract, three admission backends."""
    with open_sqlite(str(tmp_path / "records.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        control = toy_codex_control(
            engine=engine, max_tool_calls=MAX_ACCEPTED_CALLS
        )
        # Both runs are built against the same control and namespace, so
        # they resolve to the identical Tool Config hash.
        run_a, config, candidate = toy_codex_run(
            control=control,
            engine=engine,
            run_id="admitted-entries-run-a",
        )
        run_b, config_b, _candidate_b = toy_codex_run(
            control=control,
            engine=engine,
            run_id="admitted-entries-run-b",
        )
        config = config.model_copy(
            update={"store_namespace_key": SHARED_NAMESPACE}
        )
        config_b = config_b.model_copy(
            update={"store_namespace_key": SHARED_NAMESPACE}
        )
        assert config.identity_hash() == config_b.identity_hash()

        tool_store = ToolCallStore(
            store,
            build_authority(tmp_path),
            EffectLeaseAuthority.memory(),
        )

        # Run A fills its cap, then tries one more.
        for ordinal in range(1, MAX_ACCEPTED_CALLS + 1):
            tool_store.admit(
                _call(
                    call_id=f"a{ordinal}",
                    config=config,
                    run=run_a,
                    candidate=candidate,
                    engine=engine,
                    template=f"Answer {{prompt}} in {ordinal} words.",
                ),
                config,
            )
        over_cap = tool_store.admit(
            _call(
                call_id="a-over",
                config=config,
                run=run_a,
                candidate=candidate,
                engine=engine,
                template="Answer {prompt} in one word.",
            ),
            config,
        )
        # Run B shares the Tool Config and buys one call of its own.
        tool_store.admit(
            _call(
                call_id="b1",
                config=config_b,
                run=run_b,
                candidate=candidate,
                engine=engine,
                template="Answer {prompt} for run b.",
            ),
            config_b,
        )

        binding_a = toy_capacity_binding(run_a)
        binding_b = toy_capacity_binding(run_b)
        admitted_a = tool_store.admitted_entries(config, binding_a)
        admitted_b = tool_store.admitted_entries(config_b, binding_b)

    # The over-cap call was refused, so it debited no capacity and is not
    # part of what run A paid for.
    assert over_cap.state is ToolCallState.REFUSED
    assert over_cap.capacity_debit_ordinal is None
    assert "a-over" not in {
        entry.tool_call.record.call_id for entry in admitted_a
    }

    # Scoping: run B's call never appears in run A's ledger, and vice
    # versa, even though both share one Tool Config and one namespace.
    assert [entry.tool_call.record.call_id for entry in admitted_a] == [
        "a1",
        "a2",
    ]
    assert [entry.tool_call.record.call_id for entry in admitted_b] == ["b1"]

    # Ordering is the capacity debit ordinal, dense and one-based.
    assert [entry.capacity_debit_ordinal for entry in admitted_a] == [1, 2]
    assert [entry.capacity_debit_ordinal for entry in admitted_b] == [1]

    # The two views of the same ledger agree.
    assert len(admitted_a) == tool_store.accepted_count(config, binding_a)
    assert len(admitted_b) == tool_store.accepted_count(config_b, binding_b)
