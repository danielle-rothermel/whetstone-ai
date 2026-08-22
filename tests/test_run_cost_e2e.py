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


def _transport_factory(usd: float | None):
    def factory(policy):
        return _usage_reporting_transport(
            transport_policy=policy.transport_policy,
            usd=usd,
        )

    return factory


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
