"""Per-run Tool capacity is what stops a Codex agent buying more evals.

The capacity cap is advisory to the agent -- a refused call returns
``is_error`` and the agent keeps running until its own wall or turn
budget -- but it is absolute to the ledger: call ``N+1`` is refused, not
evaluated. These tests assert on the returned admission state, never on
elapsed time.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from dr_store.sync import open_sqlite

from tests.codex_support import (
    toy_capacity_binding,
    toy_codex_control,
    toy_codex_run,
    toy_tool_args,
)
from whetstone.core.identity import ImmutableJsonObject
from whetstone.core.leasing import EffectLeaseAuthority, ReplayPolicy
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.tools.admission import ToolCallState
from whetstone.optim.tools.contracts import (
    RefusalClass,
    ToolCall,
    ToolCapacityScope,
    tool_config_reference,
)
from whetstone.optim.tools.evaluator import EngineToolEvaluator
from whetstone.optim.tools.execution import EvaluatingToolExecutor
from whetstone.optim.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)

MAX_ACCEPTED_CALLS = 2


@pytest.fixture
def codex_admission(tmp_path):
    """A runtime handle whose per-run capacity admits two calls."""
    with open_sqlite(str(tmp_path / "admission.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        control = toy_codex_control(
            engine=engine, max_tool_calls=MAX_ACCEPTED_CALLS
        )
        run, config, candidate = toy_codex_run(
            control=control, engine=engine
        )
        effect_authority = EffectLeaseAuthority.memory()
        tool_store = ToolCallStore(
            store,
            ToolAdmissionAuthority.memory(),
            effect_authority,
        )
        executor = EvaluatingToolExecutor(
            EngineToolEvaluator(engine),
            engine.reward_policy,
            effect_authority,
            owner_id="codex-admission-test",
            replay_policy=ReplayPolicy.IDEMPOTENT,
            lease_duration=timedelta(minutes=5),
        )
        handle = executor.runtime_handle(
            config, tool_store, toy_capacity_binding(run)
        )
        yield handle, tool_store, config, run, candidate, engine


def _call(*, call_id, config, run, candidate, engine, template):
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


def test_the_capacity_scope_is_the_run(codex_admission) -> None:
    handle, _store, config, run, _candidate, _engine = codex_admission

    assert config.capacity.scope is ToolCapacityScope.RUN
    assert config.capacity.max_accepted_calls == MAX_ACCEPTED_CALLS
    assert handle.binding.subject_ref is not None
    assert handle.binding.subject_ref.schema_name == "whetstone.optim_run"


def test_calls_up_to_the_cap_succeed_and_the_next_is_refused(
    codex_admission,
) -> None:
    handle, tool_store, config, run, candidate, engine = codex_admission

    accepted = [
        handle(
            _call(
                call_id=f"c{ordinal}",
                config=config,
                run=run,
                candidate=candidate,
                engine=engine,
                template=f"Answer {{prompt}} in {ordinal} words.",
            )
        )
        for ordinal in range(1, MAX_ACCEPTED_CALLS + 1)
    ]

    for ordinal, result in enumerate(accepted, start=1):
        assert result.refusal is None
        assert result.terminal_failure is None
        assert result.output is not None
        assert result.provenance_ordinal == ordinal

    over_cap = handle(
        _call(
            call_id="c-over",
            config=config,
            run=run,
            candidate=candidate,
            engine=engine,
            template="Answer {prompt} in one word.",
        )
    )

    assert over_cap.refusal is not None
    assert over_cap.refusal.refusal_class is RefusalClass.CAPACITY
    assert over_cap.output is None
    # A refusal is terminal at admission: it carries no evidence and no
    # capacity ordinal, so it cannot be mistaken for a paid evaluation.
    assert over_cap.evaluation_evidence_refs == ()
    assert over_cap.reward is None
    assert over_cap.provenance_ordinal is None
    assert (
        tool_store.accepted_count(config, handle.binding)
        == MAX_ACCEPTED_CALLS
    )


def test_a_refused_call_is_recorded_and_replays_as_refused(
    codex_admission,
) -> None:
    handle, tool_store, config, run, candidate, engine = codex_admission
    for ordinal in range(1, MAX_ACCEPTED_CALLS + 1):
        handle(
            _call(
                call_id=f"c{ordinal}",
                config=config,
                run=run,
                candidate=candidate,
                engine=engine,
                template=f"Answer {{prompt}} in {ordinal} words.",
            )
        )
    over_cap_call = _call(
        call_id="c-over",
        config=config,
        run=run,
        candidate=candidate,
        engine=engine,
        template="Answer {prompt} in one word.",
    )

    first = handle(over_cap_call)
    replayed = handle(over_cap_call)

    assert replayed == first
    entry = tool_store.find_entry(
        store_namespace_key=str(config.store_namespace_key),
        call_id="c-over",
    )
    assert entry is not None
    assert entry.state is ToolCallState.REFUSED


def test_capacity_is_scoped_to_one_run(codex_admission) -> None:
    handle, tool_store, config, run, candidate, engine = codex_admission
    for ordinal in range(1, MAX_ACCEPTED_CALLS + 1):
        handle(
            _call(
                call_id=f"c{ordinal}",
                config=config,
                run=run,
                candidate=candidate,
                engine=engine,
                template=f"Answer {{prompt}} in {ordinal} words.",
            )
        )

    other_run, _config, _candidate = toy_codex_run(
        control=toy_codex_control(
            engine=engine, max_tool_calls=MAX_ACCEPTED_CALLS
        ),
        engine=engine,
        run_id="codex-other-run",
    )

    # A different run has its own capacity ledger, so the first run's
    # exhausted cap cannot starve it.
    assert tool_store.accepted_count(
        config, toy_capacity_binding(other_run)
    ) == 0
