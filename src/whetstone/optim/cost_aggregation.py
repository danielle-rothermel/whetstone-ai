"""Derive one run's spend from the evidence persisted in the object store.

This is the single owner of how ``OptimResult.cost`` is computed. Both the
in-process harness path and the platform run-completion path call
:func:`aggregate_run_cost`, so a run reports the same spend however it ran and
however often it was resumed.

Everything is re-derived from persisted records, never from counters held in
memory:

* task-model usage comes from the ``EvalOutputRow`` entries inside each
  ``EvalEvidence`` record a Step cites, reached through the Step's
  ``resolved_intents`` and ``search_evidence``;
* proposer usage comes from ``OptimStepResult.proposer_usage``, which the
  optimizer's adapter records as it drives each proposer call.

Both roles de-duplicate. An evaluation that a Step cites more than once is
counted once, keyed by its evidence reference; a proposer call reported more
than once is counted once, keyed by its ``call_id``. That is what makes a
replay free: GEPA re-drives its whole reflection prefix from the durable
effect cache on every Step, and a resumed Step re-drives the prefix it had
already paid for, so without both keys a replayed call would be billed again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, StrictInt

from whetstone.core.identity import TypedRef
from whetstone.eval.schema_names import EVAL_EVIDENCE_SCHEMA
from whetstone.optim.cost import (
    RunCostReport,
    UsageObservation,
    aggregate_role_cost,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore

    from whetstone.optim.contracts import OptimStepResult

__all__ = ["aggregate_run_cost"]


class EvalOutputRowUsage(BaseModel):
    """The usage fields of one persisted ``EvalOutputRow``.

    A narrow read-side projection: cost aggregation needs three fields, so it
    parses only those and ignores the rest of the row. The field names are
    the persisted spelling pinned by
    ``tests/test_eval_evidence_schema_golden.py``.
    """

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    prompt_tokens: StrictInt | None = None
    completion_tokens: StrictInt | None = None
    provider_cost: float | None = None
    #: The prompt cache replayed this row's call, so the usage above is the
    #: original call's and this row was not paid for again.
    cache_hit: bool = False


def _evidence_refs(result: OptimStepResult) -> tuple[TypedRef, ...]:
    """Every evaluation-evidence ref one Step paid for, in order."""
    refs: list[TypedRef] = []
    for resolution in result.resolved_intents:
        ref = resolution.eval_result_ref
        if ref is not None and ref.schema_name == EVAL_EVIDENCE_SCHEMA:
            refs.append(ref)
    for evidence in result.search_evidence:
        ref = evidence.eval_result_ref
        if ref is not None and ref.schema_name == EVAL_EVIDENCE_SCHEMA:
            refs.append(ref)
    return tuple(refs)


def _task_model_observations(
    store: ObjectStore,
    refs: tuple[TypedRef, ...],
) -> tuple[UsageObservation, ...]:
    """Load the output rows behind each distinct evidence ref."""
    observations: list[UsageObservation] = []
    for ref in refs:
        evidence = store.get(ref.reference)
        outputs_reference = evidence.get("outputs_ref")
        if not isinstance(outputs_reference, dict):
            continue
        outputs_ref = TypedRef.model_validate(outputs_reference)
        record = store.get(outputs_ref.reference)
        rows = record.get("outputs") if isinstance(record, dict) else None
        if not isinstance(rows, list):
            continue
        # Only the usage fields are read, so cost aggregation stays decoupled
        # from the rest of the outputs record's shape.
        for row in (EvalOutputRowUsage.model_validate(item) for item in rows):
            observation = _row_observation(row)
            if observation is not None:
                observations.append(observation)
    return tuple(observations)


def _row_observation(row: EvalOutputRowUsage) -> UsageObservation | None:
    """Classify one output row as billable, cached, or no call at all.

    Any recorded usage -- a token count *or* a provider-reported price -- is
    positive evidence the provider answered, so the row is a call. A price
    without a token split still counts, and still contributes its price;
    ``rows_missing_token_breakdown`` records that its tokens are unknown so
    the token totals are not read as complete.

    A row the prompt cache replayed carries the original call's usage
    verbatim. It is reported as a cached call and contributes nothing
    billable, which is what keeps a cache hit from being charged twice.

    A row with no tokens and no price evidences no provider call at all -- a
    missing row, or a failure before the provider answered -- and is dropped.
    """
    has_tokens = (
        row.prompt_tokens is not None or row.completion_tokens is not None
    )
    if not has_tokens and row.provider_cost is None:
        return None
    if row.cache_hit:
        return UsageObservation(cached=True)
    return UsageObservation(
        input_tokens=row.prompt_tokens or 0,
        output_tokens=row.completion_tokens or 0,
        usd=row.provider_cost,
        missing_token_breakdown=not has_tokens,
    )


def aggregate_run_cost(
    *,
    store: ObjectStore,
    step_results: tuple[OptimStepResult, ...],
) -> RunCostReport:
    """Total one run's task-model and proposer spend from persisted evidence."""
    seen: set[tuple[str, str]] = set()
    seen_proposer_calls: set[str] = set()
    distinct_refs: list[TypedRef] = []
    proposer: list[UsageObservation] = []
    for result in step_results:
        for ref in _evidence_refs(result):
            key = (ref.schema_name, ref.content_hash)
            if key in seen:
                continue
            seen.add(key)
            distinct_refs.append(ref)
        for usage in result.proposer_usage:
            # An identified call is counted once however many Step Results
            # report it. An unidentified one has nothing to key on, so it is
            # counted every time rather than silently collapsed together.
            if usage.call_id:
                if usage.call_id in seen_proposer_calls:
                    continue
                seen_proposer_calls.add(usage.call_id)
            proposer.append(usage.observation())
    return RunCostReport(
        task_model=aggregate_role_cost(
            _task_model_observations(store, tuple(distinct_refs))
        ),
        proposer=aggregate_role_cost(tuple(proposer)),
    )
