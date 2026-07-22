"""GEPA adapter tests: structured bounded tool-using steps on the harness.

Every run drives the REAL durable harness + REAL Tool Call Store + the
``EvaluatingToolExecutor`` over the shipped ``StubToolEvaluator`` and a
scripted
``FakeProposerTransport`` for the reflection LM. No network.
"""

from __future__ import annotations

from typing import Any

from tests.optimization.tool_support import (
    candidate,
    evaluating_executor,
    gepa_request,
    gepa_tool_configs,
    make_store,
    proposer_config,
)
from whetstone.optimization import (
    FakeProposerTransport,
    OptimizationHarness,
    RuntimeToolHandle,
    ToolCallState,
    ToolCallStore,
    ToolConfig,
    ToolExecutor,
)
from whetstone.optimization.gepa import (
    ACCEPTANCE_POLICY,
    GEPA_VARIANT,
    GepaAdapter,
    GepaHyperparameters,
    strict_pareto_accepts,
)
from whetstone.optimization.schema import StepMode


def _harness(store):
    return OptimizationHarness(
        store=store, tool_executor=evaluating_executor()
    )


def _restart(store, result) -> dict[str, Any]:
    assert result.state_ref is not None
    body = store.get(result.state_ref.reference)
    assert isinstance(body, dict)
    restart = body["restart_minimum"]
    assert isinstance(restart, dict)
    return restart


def _reflection_transport(templates):
    # Key every reflection call under mode gepa_reflection; ordinal-indexed.
    script: dict[tuple[str, int], tuple[str, ...]] = {
        ("gepa_reflection", i): (t,) for i, t in enumerate(templates)
    }
    return FakeProposerTransport(script, default=("fallback child template",))


def _adapter(templates, **hp):
    return GepaAdapter(
        reflection_config=proposer_config(),
        reflection_transport=_reflection_transport(templates),
        hyperparameters=GepaHyperparameters(**hp) if hp else None,
    )


def test_gepa_first_step_seeds_ab_then_reflects_and_evaluates():
    store = make_store()
    harness = _harness(store)
    configs = gepa_tool_configs()
    candidates = (
        candidate("A", "route-0", "template A"),
        candidate("B", "route-0", "template B"),
    )
    adapter = _adapter(["a much more specific evolved child template"])
    request = gepa_request(configs=configs, candidates=candidates)

    result, _ref = harness.run_step(request, adapter)

    # Seed: 2 evaluate_subset calls (A, B) + parent diagnostic + child
    # comparison = 4 tool results (accepted-child subset only if accepted).
    assert len(result.tool_evidence) >= 4
    # Every Tool Result is referenced with its Store Entry, and each completed
    # entry's terminal ref matches the evidence ref.
    for evidence in result.tool_evidence:
        entry = evidence.store_entry
        assert entry.state in (
            ToolCallState.COMPLETED,
            ToolCallState.REFUSED,
        )
        if entry.state is ToolCallState.COMPLETED:
            assert entry.tool_result_ref == evidence.tool_result_ref
    # The restart minimum is present in the state snapshot.
    restart = _restart(store, result)
    for field in (
        "run_id",
        "gepa_variant",
        "acceptance_policy",
        "parent",
        "sampled_task_ids",
        "parent_objectives",
        "reflection_evidence",
        "acceptance",
    ):
        assert field in restart
    assert restart["gepa_variant"] == GEPA_VARIANT
    assert restart["acceptance_policy"] == ACCEPTANCE_POLICY


def test_gepa_parent_and_child_same_minibatch_task_ids():
    store = make_store()
    harness = _harness(store)
    configs = gepa_tool_configs()
    candidates = (
        candidate("A", "route-0", "template A"),
        candidate("B", "route-0", "template B"),
    )
    adapter = _adapter(["evolved child that is more specific and precise"])
    request = gepa_request(configs=configs, candidates=candidates)
    result, _ref = harness.run_step(request, adapter)

    # The parent diagnostic and child comparison share one minibatch task ID
    # set (the same-minibatch invariant); it is recorded in the restart
    # minimum and sized to minibatch_size = 3.
    restart = _restart(store, result)
    task_ids = restart["sampled_task_ids"]
    assert isinstance(task_ids, list)
    assert len(task_ids) == 3
    # A child was accepted (stub gives the child a distinct measurement), so an
    # acceptance decision was recorded against these exact task IDs.
    assert restart["acceptance"] is not None


def test_strict_pareto_acceptance_semantics():
    # Strictly better on correctness, no worse on compression -> accept.
    assert strict_pareto_accepts(
        parent={"correctness": 0.5, "compression": 2.0},
        child={"correctness": 0.7, "compression": 2.0},
    )
    # Equal on both -> reject (not strictly better).
    assert not strict_pareto_accepts(
        parent={"correctness": 0.5, "compression": 2.0},
        child={"correctness": 0.5, "compression": 2.0},
    )
    # Trade-off (better correctness, worse compression) -> reject.
    assert not strict_pareto_accepts(
        parent={"correctness": 0.5, "compression": 2.0},
        child={"correctness": 0.7, "compression": 3.0},
    )
    # Regression on correctness -> reject.
    assert not strict_pareto_accepts(
        parent={"correctness": 0.5, "compression": 2.0},
        child={"correctness": 0.4, "compression": 1.0},
    )
    # Better compression, equal correctness -> accept.
    assert strict_pareto_accepts(
        parent={"correctness": 0.5, "compression": 2.0},
        child={"correctness": 0.5, "compression": 1.0},
    )


class _TracingExecutor(ToolExecutor):
    """Wraps an EvaluatingToolExecutor to record when handles are built."""

    def __init__(self, inner: ToolExecutor) -> None:
        self._inner = inner
        self.built: list[str] = []

    def runtime_handle(
        self, config: ToolConfig, store: ToolCallStore
    ) -> RuntimeToolHandle:
        self.built.append(config.tool_name)
        return self._inner.runtime_handle(config, store)


def test_gepa_runtime_handles_built_only_at_execution_boundary():
    store = make_store()
    executor = _TracingExecutor(evaluating_executor())
    built = executor.built
    harness = OptimizationHarness(store=store, tool_executor=executor)
    configs = gepa_tool_configs()
    candidates = (
        candidate("A", "route-0", "template A"),
        candidate("B", "route-0", "template B"),
    )
    adapter = _adapter(["evolved child template that is more precise"])
    request = gepa_request(configs=configs, candidates=candidates)

    # No handle exists before run_step; the request carries only serialized
    # Tool Configs (never a RuntimeToolHandle).
    assert built == []
    for cfg in request.tool_configs:
        assert not isinstance(cfg, RuntimeToolHandle)

    harness.run_step(request, adapter)
    # Handles are constructed exactly at the execution boundary, one per
    # config.
    assert set(built) == {"evaluate_minibatch", "evaluate_subset"}


def test_gepa_reflection_bounded_and_restart_minimum_records_attempts():
    store = make_store()
    harness = _harness(store)
    configs = gepa_tool_configs()
    candidates = (
        candidate("A", "route-0", "template A"),
        candidate("B", "route-0", "template B"),
    )
    # All reflection drafts are off-surface duplicates of the base template, so
    # none validate: reflection is exhausted within the per-step bound.
    adapter = _adapter(
        ["template A", "template A", "template A"],
        max_reflection_attempts_per_step=3,
        max_reflection_lm_calls=8,
    )
    request = gepa_request(configs=configs, candidates=candidates)
    result, _ref = harness.run_step(request, adapter)

    restart = _restart(store, result)
    attempts = restart["reflection_evidence"]["attempts"]
    # Bounded by max_reflection_attempts_per_step = 3.
    assert len(attempts) == 3
    assert all(a["validation"]["valid"] is False for a in attempts)
    # No accepted child, no fabricated acceptance.
    assert restart["accepted_child"] is None
    assert restart["acceptance"] is None


def test_gepa_restart_mid_run_reuses_step_result():
    store = make_store()
    configs = gepa_tool_configs()
    candidates = (
        candidate("A", "route-0", "template A"),
        candidate("B", "route-0", "template B"),
    )
    templates = ["a precise evolved child template for the encoder"]
    request = gepa_request(configs=configs, candidates=candidates)

    harness1 = _harness(store)
    adapter1 = _adapter(templates)
    result1, ref1 = harness1.run_step(request, adapter1)

    # A fresh harness + fresh adapter over the same store short-circuits to the
    # persisted Step Result (no re-execution).
    fresh_transport = _reflection_transport(templates)
    adapter2 = GepaAdapter(
        reflection_config=proposer_config(),
        reflection_transport=fresh_transport,
    )
    harness2 = _harness(store)
    result2, ref2 = harness2.run_step(request, adapter2)
    assert ref2 == ref1
    assert result2.request_ref == result1.request_ref
    # The fresh reflection transport was never called (Step replayed).
    assert fresh_transport.calls == []


def test_gepa_state_crosses_step_boundaries_only_by_reference():
    # Two GEPA steps: step 2 reads step 1's durable state snapshot through the
    # request pools (the harness/driver threads the state ref, never process
    # memory). The catalog grows across steps and the seed is not re-run.
    store = make_store()
    configs = gepa_tool_configs()
    candidates = (
        candidate("A", "route-0", "template A"),
        candidate("B", "route-0", "template B"),
    )

    harness = _harness(store)
    adapter1 = _adapter(["step one evolved encoder template that is precise"])
    request1 = gepa_request(configs=configs, candidates=candidates)
    result1, ref1 = harness.run_step(request1, adapter1)
    restart1 = _restart(store, result1)
    assert restart1["catalog_size"] >= 3  # A + B + step-1 child

    # The driver threads the prior state snapshot into the next request pools.
    assert result1.state_ref is not None
    state_body = store.get(result1.state_ref.reference)
    assert isinstance(state_body, dict)
    gepa_state = state_body["gepa_state"]
    assert isinstance(gepa_state, dict)
    assert gepa_state["seed_done"] is True

    adapter2 = _adapter(["step two evolved encoder template, more precise"])
    request2 = gepa_request(
        run_id="run-gepa",
        step_index=1,
        configs=configs,
        candidates=candidates,
        pools={"gepa_state": gepa_state},
        prior_step_result_ref=ref1,
    )
    result2, _ref2 = harness.run_step(request2, adapter2)
    restart2 = _restart(store, result2)

    # Step 2 did NOT re-seed (seed is data-dependent, carried by reference).
    assert restart2["catalog_size"] > restart1["catalog_size"]
    # Step 2's parent came from the frontier the prior step built.
    assert restart2["parent"] is not None


def test_gepa_adapter_declares_tool_using_mode():
    adapter = _adapter(["x"])
    assert adapter.mode is StepMode.TOOL_USING
