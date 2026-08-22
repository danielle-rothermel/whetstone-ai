"""Run cost after a completed in-process run, end to end.

These drive the real evaluation engine and the real graph rollout driver with
a transport that reports usage the way a provider does, then read the spend
back off the persisted evidence. That covers the whole path the plumbing
exists for: provider response, output row, stored evidence, aggregated
``OptimResult.cost``.
"""

from __future__ import annotations

import pytest
from dr_providers import (
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderTransportPolicy,
)
from dr_providers.outcomes.evidence import ProviderHttpRequestEvidence
from dr_providers.outcomes.models import (
    CostInfo,
    ProviderTransportResponse,
    TokenUsage,
)

from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.schema import EvalEvidence
from whetstone.optim.contracts import OptimResult
from whetstone.optim.cost import ProposerCallUsage
from whetstone.optim.cost_aggregation import aggregate_run_cost

PROMPT_TOKENS = 17
COMPLETION_TOKENS = 5
USD_PER_CALL = 0.125

__all__ = [
    "COMPLETION_TOKENS",
    "PROMPT_TOKENS",
    "USD_PER_CALL",
    "usage_reporting_transport_factory",
]


def _usage_reporting_transport(
    *,
    transport_policy: ProviderTransportPolicy,
    usd: float | None,
):
    """A transport that reports token usage, and optionally a price."""

    def _transport(request: ProviderCallRequest) -> ProviderInvocationEvidence:
        messages = request.transcript.messages
        prompt = messages[-1].content if messages else ""
        response = ProviderTransportResponse(
            text=f"generated: {prompt}",
            stop_reason="stop",
            usage=TokenUsage(
                prompt_tokens=PROMPT_TOKENS,
                completion_tokens=COMPLETION_TOKENS,
                total_tokens=PROMPT_TOKENS + COMPLETION_TOKENS,
            ),
            cost=None if usd is None else CostInfo(total_cost=usd),
        )
        return ProviderInvocationEvidence.build(
            request=request,
            policy=transport_policy,
            http_request=ProviderHttpRequestEvidence(
                method="POST",
                url="http://whetstone.fake/llm",
                headers={},
                body={},
                body_bytes=0,
            ),
            outcome=response,
        )

    return _transport


def usage_reporting_transport_factory(usd: float | None):
    """A transport factory reporting fixed per-call usage, and maybe a price.

    Shared with the platform cross-path test so both paths are driven by the
    identical provider behaviour, which is what makes their cost reports
    comparable at all.
    """

    def factory(policy):
        return _usage_reporting_transport(
            transport_policy=policy.transport_policy,
            usd=usd,
        )

    return factory


_transport_factory = usage_reporting_transport_factory


def _evaluate_one_row(store, *, usd: float | None) -> EvalEvidence:
    """Run one real evaluation row and return its persisted evidence."""
    from whetstone.eval.protocol import EvalRequest

    engine = ReferenceEvalRuntimeConfig().build_engine(
        store,
        transport_factory=_transport_factory(usd),
    )
    task_id = engine.sampling.tasks[0].task_id
    completion = engine.for_task_seed(task_id, 0).evaluate_row(
        EvalRequest(
            request_id="cost-e2e",
            candidate=engine.experiment.initial_candidate,
        )
    )
    assert completion.evidence_ref is not None
    return EvalEvidence.model_validate(
        store.get(completion.evidence_ref.reference)
    )


class _Citation:
    def __init__(self, ref) -> None:
        self.eval_result_ref = ref


class _StepResult:
    def __init__(self, *, refs=(), proposer_usage=()) -> None:
        self.resolved_intents = tuple(_Citation(ref) for ref in refs)
        self.search_evidence = ()
        self.tool_evidence = ()
        self.proposer_usage = proposer_usage


def test_task_model_usage_reaches_persisted_output_rows(
    sqlite_store,
) -> None:
    """The provider's usage survives all the way into stored evidence."""
    evidence = _evaluate_one_row(sqlite_store, usd=None)
    outputs = sqlite_store.get(evidence.outputs_ref.reference)
    rows = outputs["outputs"]
    assert rows, "the evaluation persisted no output rows"
    assert all(row["prompt_tokens"] == PROMPT_TOKENS for row in rows)
    assert all(
        row["completion_tokens"] == COMPLETION_TOKENS for row in rows
    )


def test_completed_run_reports_nonzero_task_model_and_proposer_cost(
    sqlite_store,
) -> None:
    """The headline case: a run reports what its two models actually used."""
    evidence = _evaluate_one_row(sqlite_store, usd=None)
    report = aggregate_run_cost(
        store=sqlite_store,
        step_results=(
            _StepResult(
                refs=(evidence_ref := _evidence_ref(sqlite_store, evidence),),
                proposer_usage=(
                    ProposerCallUsage(
                        prompt_tokens=40,
                        completion_tokens=12,
                    ),
                ),
            ),
        ),
    )
    assert evidence_ref is not None
    assert report.task_model.calls >= 1
    assert report.task_model.input_tokens >= PROMPT_TOKENS
    assert report.task_model.output_tokens >= COMPLETION_TOKENS
    assert report.proposer.calls == 1
    assert report.proposer.input_tokens == 40
    assert report.proposer.output_tokens == 12
    # Neither model reported a price, so no total is claimed.
    assert report.task_model.usd is None
    assert report.proposer.usd is None


def test_a_priced_run_reports_a_usd_total(sqlite_store) -> None:
    evidence = _evaluate_one_row(sqlite_store, usd=0.125)
    report = aggregate_run_cost(
        store=sqlite_store,
        step_results=(
            _StepResult(refs=(_evidence_ref(sqlite_store, evidence),)),
        ),
    )
    assert report.task_model.unpriced_calls == 0
    assert report.task_model.usd == pytest.approx(
        0.125 * report.task_model.calls
    )


def _evidence_ref(store, evidence: EvalEvidence):
    """Re-address the persisted evidence record."""
    from whetstone.core.identity import typed_ref_for_record
    from whetstone.eval.schema_names import EVAL_EVIDENCE_SCHEMA

    ref = typed_ref_for_record(
        EVAL_EVIDENCE_SCHEMA, evidence.record_content()
    )
    # It must already be in the store; this is a lookup, not a write.
    store.get(ref.reference)
    return ref


@pytest.fixture
def toy_copro_run(sqlite_store):
    """Drive a real in-process COPRO run over the usage-reporting transport.

    Returns the run's ``RunCostReport`` together with the engine and control
    that determine how many calls it should have made, so a test can derive
    the expected totals rather than read them back off the report.
    """
    from uuid import uuid4

    from whetstone.coordination.runtime_bootstrap import copro_run_request
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.optim.cost import RunCostReport
    from whetstone.testing.runtime import (
        build_toy_copro_control,
        prepare_toy_copro_run,
        register_toy_runtime,
    )

    def _run(*, usd: float | None, breadth: int = 2, depth: int = 1):
        engine = ReferenceEvalRuntimeConfig().build_engine(
            sqlite_store,
            transport_factory=usage_reporting_transport_factory(usd),
        )
        control = build_toy_copro_control(
            breadth=breadth, depth=depth, engine=engine
        )
        runtime = register_toy_runtime(
            store=sqlite_store, engine=engine, copro_control=control
        )
        launch = prepare_toy_copro_run(
            runtime,
            run_id=f"cost-e2e-{uuid4().hex[:8]}",
            control=control,
            terminal_top_k=1,
        )
        result_ref = runtime.controller.drive(
            copro_run_request(
                launch,
                controller_identity_hash=runtime.controller.runtime_hash,
            )
        )
        result = OptimResult.model_validate(
            sqlite_store.get(result_ref.reference)
        )
        return (
            RunCostReport.model_validate(result.cost.to_json()),
            engine,
            control,
        )

    return _run


def test_optim_result_cost_is_populated_by_the_harness(
    copro_launch,
) -> None:
    """The completion path fills ``cost`` rather than leaving it empty."""
    from whetstone.coordination.runtime_bootstrap import copro_run_request

    runtime, launch = copro_launch
    result_ref = runtime.controller.drive(
        copro_run_request(
            launch,
            controller_identity_hash=runtime.controller.runtime_hash,
        )
    )
    result = OptimResult.model_validate(
        runtime.store.get(result_ref.reference)
    )
    cost = result.cost.to_json()
    assert cost, "OptimResult.cost must not be empty on a completed run"
    assert cost["schema_version"] == 1
    # The toy run drives a real proposer call, so its call count is real.
    assert cost["proposer"]["calls"] >= 1
    assert set(cost) == {"schema_version", "task_model", "proposer"}


def test_run_totals_equal_the_usage_the_transport_reported(
    toy_copro_run,
) -> None:
    """Totals are checked against arithmetic, not against the report itself.

    The transport reports a fixed number of tokens and a fixed price per
    call, and the toy control's shape fixes how many calls the run makes, so
    the expected totals are computable in advance. Asserting exact equality
    against that arithmetic is what makes this a check of the plumbing --
    provider response, output row, stored evidence, aggregate -- rather than
    a check that the report agrees with itself.
    """
    report, engine, control = toy_copro_run(usd=USD_PER_CALL)

    # COPRO evaluates each of the ``breadth`` candidates it drafts in its
    # single round, one row per task per seed. Verified to scale with
    # breadth rather than assumed: breadth=3 drives six calls, not four.
    rows_per_candidate = len(engine.sampling.tasks) * engine.sampling.num_seeds
    expected_calls = control.breadth * rows_per_candidate

    assert report.task_model.calls == expected_calls
    assert report.task_model.input_tokens == expected_calls * PROMPT_TOKENS
    assert (
        report.task_model.output_tokens == expected_calls * COMPLETION_TOKENS
    )
    assert report.task_model.usd == pytest.approx(
        expected_calls * USD_PER_CALL
    )
    assert report.task_model.cached_calls == 0
    assert report.task_model.rows_missing_token_breakdown == 0


def test_run_totals_track_the_number_of_calls_the_run_makes(
    toy_copro_run,
) -> None:
    """A wider round costs proportionally more, so the count is not a constant.

    Pinning one run's totals would pass against a report that hard-coded
    them. Driving a second, wider run and asserting the totals move with the
    call count is what makes the arithmetic above load-bearing.
    """
    report, engine, control = toy_copro_run(usd=USD_PER_CALL, breadth=3)

    rows_per_candidate = len(engine.sampling.tasks) * engine.sampling.num_seeds
    expected_calls = control.breadth * rows_per_candidate

    assert control.breadth == 3
    assert report.task_model.calls == expected_calls
    assert report.task_model.input_tokens == expected_calls * PROMPT_TOKENS
    assert report.task_model.usd == pytest.approx(
        expected_calls * USD_PER_CALL
    )
