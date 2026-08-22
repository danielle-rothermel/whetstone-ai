"""Deriving run cost from evidence persisted in the object store.

These tests drive :func:`aggregate_run_cost` over records written to a real
store, which is the property that matters: the number is re-derived from
durable evidence rather than from counters the run happened to hold in
memory, so a resumed or platform run reports the same total.
"""

from __future__ import annotations

from typing import Any

import pytest
from dr_store.sync import open_sqlite

from whetstone.core.identity import TypedRef
from whetstone.eval.schema import EVAL_OUTPUTS_SCHEMA
from whetstone.eval.schema_names import EVAL_EVIDENCE_SCHEMA
from whetstone.optim.cost import ProposerCallUsage
from whetstone.optim.cost_aggregation import aggregate_run_cost


class _FakeStepResult:
    """Only the fields the aggregator reads off a Step Result."""

    def __init__(
        self,
        *,
        resolved_intents: tuple[Any, ...] = (),
        search_evidence: tuple[Any, ...] = (),
        proposer_usage: tuple[ProposerCallUsage, ...] = (),
    ) -> None:
        self.resolved_intents = resolved_intents
        self.search_evidence = search_evidence
        self.proposer_usage = proposer_usage


class _FakeCitation:
    def __init__(self, ref: TypedRef | None) -> None:
        self.eval_result_ref = ref


def _typed_ref(reference: Any) -> TypedRef:
    return TypedRef(
        schema_name=reference.schema,
        content_hash=reference.content_hash,
    )


def _row(
    *,
    task_index: int,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    provider_cost: float | None,
    cache_hit: bool = False,
    failed: bool = False,
    missing: bool = False,
) -> dict[str, Any]:
    """A minimal output row in its persisted shape.

    ``failed`` / ``missing`` drive the exclusive row state the schema
    enforces: an unscored row carries exactly one of them, and a scored row
    carries neither.
    """
    scored = not (failed or missing)
    return {
        "candidate_id": "candidate-1",
        "task_id": f"task-{task_index}",
        "task_hash": f"hash-{task_index}",
        "task_index": task_index,
        "seed_index": 0,
        "rendered_prompt": "prompt",
        "output_text": "answer" if scored else None,
        "score": 1.0 if scored else None,
        "failed": failed,
        "missing": missing,
        "invalid": False,
        "failure_code": "TRANSPORT_ERROR" if failed else "",
        "finish_reason": "stop" if scored else None,
        "provider_error": None,
        "max_budget": None,
        "over_budget": None,
        "submission_result": None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "provider_cost": provider_cost,
        "cache_hit": cache_hit,
    }


def _persist_evidence(store: Any, rows: list[dict[str, Any]]) -> TypedRef:
    """Write one outputs record and the evidence that cites it."""
    outputs_reference, _ = store.put(
        EVAL_OUTPUTS_SCHEMA,
        {"outputs": rows},
    )
    outputs_ref = _typed_ref(outputs_reference)
    evidence_reference, _ = store.put(
        EVAL_EVIDENCE_SCHEMA,
        {"outputs_ref": outputs_ref.model_dump(mode="json")},
    )
    return _typed_ref(evidence_reference)


def test_task_model_tokens_come_from_persisted_output_rows(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=0,
                    prompt_tokens=10,
                    completion_tokens=4,
                    provider_cost=None,
                ),
                _row(
                    task_index=1,
                    prompt_tokens=6,
                    completion_tokens=2,
                    provider_cost=None,
                ),
            ],
        )
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(resolved_intents=(_FakeCitation(ref),)),
            ),
        )
    assert report.task_model.calls == 2
    assert report.task_model.input_tokens == 16
    assert report.task_model.output_tokens == 6
    assert report.task_model.usd is None


def test_priced_rows_produce_a_usd_total(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=0,
                    prompt_tokens=10,
                    completion_tokens=4,
                    provider_cost=0.2,
                ),
                _row(
                    task_index=1,
                    prompt_tokens=6,
                    completion_tokens=2,
                    provider_cost=0.3,
                ),
            ],
        )
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(resolved_intents=(_FakeCitation(ref),)),
            ),
        )
    assert report.task_model.priced_calls == 2
    assert report.task_model.usd == pytest.approx(0.5)


def test_one_unpriced_row_withholds_the_task_model_usd_total(
    tmp_path,
) -> None:
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=0,
                    prompt_tokens=10,
                    completion_tokens=4,
                    provider_cost=0.2,
                ),
                _row(
                    task_index=1,
                    prompt_tokens=6,
                    completion_tokens=2,
                    provider_cost=None,
                ),
            ],
        )
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(resolved_intents=(_FakeCitation(ref),)),
            ),
        )
    assert report.task_model.calls == 2
    assert report.task_model.input_tokens == 16
    assert report.task_model.priced_calls == 1
    assert report.task_model.unpriced_calls == 1
    assert report.task_model.usd is None


def test_a_failed_row_with_no_usage_evidence_is_not_a_call(tmp_path) -> None:
    """A failed row with no telemetry never reached a billable provider."""
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=0,
                    prompt_tokens=10,
                    completion_tokens=4,
                    provider_cost=None,
                ),
                _row(
                    task_index=1,
                    prompt_tokens=None,
                    completion_tokens=None,
                    provider_cost=None,
                    failed=True,
                ),
            ],
        )
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(resolved_intents=(_FakeCitation(ref),)),
            ),
        )
    assert report.task_model.calls == 1
    assert report.task_model.input_tokens == 10
    assert report.task_model.cached_calls == 0


def test_a_missing_row_with_no_usage_evidence_is_not_a_call(tmp_path) -> None:
    """A row that never ran is not a call however the run ended."""
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=0,
                    prompt_tokens=10,
                    completion_tokens=4,
                    provider_cost=None,
                ),
                _row(
                    task_index=1,
                    prompt_tokens=None,
                    completion_tokens=None,
                    provider_cost=None,
                    missing=True,
                ),
            ],
        )
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(resolved_intents=(_FakeCitation(ref),)),
            ),
        )
    assert report.task_model.calls == 1


def test_a_successful_row_without_telemetry_is_a_billable_call(
    tmp_path,
) -> None:
    """The provider answered, so the run paid for the call regardless.

    A successful row proves a provider call happened. Dropping it because no
    usage came back understates ``calls`` and -- worse -- lets the remaining
    priced rows present a partial ``usd`` as a complete run total. Counting
    it as unpriced is the honest outcome: the call is in ``calls``, and
    ``usd`` is withheld.
    """
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=0,
                    prompt_tokens=10,
                    completion_tokens=4,
                    provider_cost=0.5,
                ),
                _row(
                    task_index=1,
                    prompt_tokens=None,
                    completion_tokens=None,
                    provider_cost=None,
                ),
            ],
        )
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(resolved_intents=(_FakeCitation(ref),)),
            ),
        )
    assert report.task_model.calls == 2
    assert report.task_model.priced_calls == 1
    assert report.task_model.unpriced_calls == 1
    assert report.task_model.rows_missing_token_breakdown == 1
    assert report.task_model.input_tokens == 10
    assert report.task_model.usd is None


def test_a_failed_row_that_was_billed_still_counts(tmp_path) -> None:
    """A failure the provider charged for is spend the run really made."""
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=0,
                    prompt_tokens=10,
                    completion_tokens=4,
                    provider_cost=0.5,
                ),
                _row(
                    task_index=1,
                    prompt_tokens=7,
                    completion_tokens=0,
                    provider_cost=0.25,
                    failed=True,
                ),
            ],
        )
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(resolved_intents=(_FakeCitation(ref),)),
            ),
        )
    assert report.task_model.calls == 2
    assert report.task_model.input_tokens == 17
    assert report.task_model.usd == pytest.approx(0.75)


def test_a_cache_hit_row_is_reported_but_never_billed(tmp_path) -> None:
    """The headline replay case, with the tokens a real cache hit carries.

    A prompt-cache hit replays the original call's response verbatim, so its
    row carries that call's real tokens and price. Billing it again would
    charge the run twice for one provider call, so the row counts only in
    ``cached_calls``.
    """
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=0,
                    prompt_tokens=10,
                    completion_tokens=4,
                    provider_cost=0.2,
                ),
                _row(
                    task_index=1,
                    prompt_tokens=10,
                    completion_tokens=4,
                    provider_cost=0.2,
                    cache_hit=True,
                ),
            ],
        )
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(resolved_intents=(_FakeCitation(ref),)),
            ),
        )
    assert report.task_model.calls == 1
    assert report.task_model.cached_calls == 1
    assert report.task_model.input_tokens == 10
    assert report.task_model.output_tokens == 4
    # One provider call was paid for, so the total is that call's price.
    assert report.task_model.usd == pytest.approx(0.2)


def test_a_cache_hit_does_not_make_the_usd_total_incomplete(tmp_path) -> None:
    """A replayed call is not an unpriced call: it is not a call at all."""
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=0,
                    prompt_tokens=10,
                    completion_tokens=4,
                    provider_cost=0.2,
                ),
                _row(
                    task_index=1,
                    prompt_tokens=8,
                    completion_tokens=3,
                    provider_cost=None,
                    cache_hit=True,
                ),
            ],
        )
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(resolved_intents=(_FakeCitation(ref),)),
            ),
        )
    assert report.task_model.unpriced_calls == 0
    assert report.task_model.cached_calls == 1
    assert report.task_model.usd == pytest.approx(0.2)


def test_a_priced_row_without_a_token_breakdown_keeps_its_price(
    tmp_path,
) -> None:
    """A price is positive evidence a call happened, so it is never dropped.

    Some providers omit the usage block while still returning a price. The
    call must count and its price must reach ``usd``; only its tokens are
    unknown, which ``rows_missing_token_breakdown`` records so the token
    totals are not read as complete.
    """
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=0,
                    prompt_tokens=10,
                    completion_tokens=4,
                    provider_cost=0.1,
                ),
                _row(
                    task_index=1,
                    prompt_tokens=None,
                    completion_tokens=None,
                    provider_cost=0.42,
                ),
            ],
        )
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(resolved_intents=(_FakeCitation(ref),)),
            ),
        )
    assert report.task_model.calls == 2
    assert report.task_model.priced_calls == 2
    assert report.task_model.unpriced_calls == 0
    assert report.task_model.rows_missing_token_breakdown == 1
    assert report.task_model.input_tokens == 10
    assert report.task_model.usd == pytest.approx(0.52)


def test_a_row_with_only_one_token_field_is_a_full_call(tmp_path) -> None:
    """A partial token split still evidences one call with a known side."""
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=0,
                    prompt_tokens=5,
                    completion_tokens=None,
                    provider_cost=None,
                ),
            ],
        )
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(resolved_intents=(_FakeCitation(ref),)),
            ),
        )
    assert report.task_model.calls == 1
    assert report.task_model.input_tokens == 5
    assert report.task_model.output_tokens == 0
    assert report.task_model.rows_missing_token_breakdown == 0
    assert report.task_model.usd is None


def test_a_proposer_call_reported_twice_is_counted_once(tmp_path) -> None:
    """Two Step Results reporting one call must not bill it twice.

    This is the proposer-side counterpart of evidence-ref de-duplication: a
    Step re-driven after a crash re-reports the calls its predecessor already
    paid for, and only the call identity can tell that from real spend.
    """
    call = ProposerCallUsage(
        call_id="reflection-1",
        prompt_tokens=30,
        completion_tokens=9,
        usd=0.4,
    )
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(proposer_usage=(call,)),
                _FakeStepResult(proposer_usage=(call,)),
            ),
        )
    assert report.proposer.calls == 1
    assert report.proposer.input_tokens == 30
    assert report.proposer.usd == pytest.approx(0.4)


def test_distinct_proposer_calls_are_each_counted(tmp_path) -> None:
    """De-duplication keys on identity, so it never merges real calls."""
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(
                    proposer_usage=(
                        ProposerCallUsage(
                            call_id="reflection-1",
                            prompt_tokens=30,
                            completion_tokens=9,
                        ),
                        ProposerCallUsage(
                            call_id="reflection-2",
                            prompt_tokens=30,
                            completion_tokens=9,
                        ),
                    )
                ),
            ),
        )
    assert report.proposer.calls == 2
    assert report.proposer.input_tokens == 60


def test_evidence_cited_twice_is_counted_once(tmp_path) -> None:
    """A replayed evaluation must not inflate the run total."""
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=0,
                    prompt_tokens=10,
                    completion_tokens=4,
                    provider_cost=None,
                )
            ],
        )
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(resolved_intents=(_FakeCitation(ref),)),
                # A later Step cites the same evaluation, as a replay does.
                _FakeStepResult(search_evidence=(_FakeCitation(ref),)),
            ),
        )
    assert report.task_model.calls == 1
    assert report.task_model.input_tokens == 10


def test_search_evidence_is_counted_alongside_resolved_intents(
    tmp_path,
) -> None:
    """Evaluations driven inside an optimizer's own search are paid too."""
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        intent_ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=0,
                    prompt_tokens=10,
                    completion_tokens=4,
                    provider_cost=None,
                )
            ],
        )
        search_ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=1,
                    prompt_tokens=7,
                    completion_tokens=1,
                    provider_cost=None,
                )
            ],
        )
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(
                    resolved_intents=(_FakeCitation(intent_ref),),
                    search_evidence=(_FakeCitation(search_ref),),
                ),
            ),
        )
    assert report.task_model.calls == 2
    assert report.task_model.input_tokens == 17
    assert report.task_model.output_tokens == 5


def test_proposer_usage_totals_across_steps(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(
                    proposer_usage=(
                        ProposerCallUsage(
                            prompt_tokens=30,
                            completion_tokens=9,
                            usd=0.4,
                        ),
                    )
                ),
                _FakeStepResult(
                    proposer_usage=(
                        ProposerCallUsage(
                            prompt_tokens=20,
                            completion_tokens=5,
                            usd=0.1,
                        ),
                    )
                ),
            ),
        )
    assert report.proposer.calls == 2
    assert report.proposer.input_tokens == 50
    assert report.proposer.output_tokens == 14
    assert report.proposer.usd == pytest.approx(0.5)


def test_roles_are_totalled_independently(tmp_path) -> None:
    """An unpriced task model must not suppress a priced proposer total."""
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        ref = _persist_evidence(
            store,
            [
                _row(
                    task_index=0,
                    prompt_tokens=10,
                    completion_tokens=4,
                    provider_cost=None,
                )
            ],
        )
        report = aggregate_run_cost(
            store=store,
            step_results=(
                _FakeStepResult(
                    resolved_intents=(_FakeCitation(ref),),
                    proposer_usage=(
                        ProposerCallUsage(
                            prompt_tokens=8,
                            completion_tokens=2,
                            usd=0.05,
                        ),
                    ),
                ),
            ),
        )
    assert report.task_model.usd is None
    assert report.proposer.usd == pytest.approx(0.05)


def test_a_run_with_no_evidence_reports_empty_roles(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "cost.sqlite")) as store:
        report = aggregate_run_cost(
            store=store,
            step_results=(_FakeStepResult(),),
        )
    assert report.task_model.calls == 0
    assert report.proposer.calls == 0
    assert report.task_model.usd is None
