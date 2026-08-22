"""Derive one run's spend from the evidence persisted in the object store.

This is the single owner of how ``OptimResult.cost`` is computed. Both the
in-process harness path and the platform run-completion path call
:func:`aggregate_run_cost`, so a run reports the same spend however it ran and
however often it was resumed.

Everything is re-derived from persisted records, never from counters held in
memory:

* task-model usage comes from the ``EvalOutputRow`` entries inside each
  ``EvalEvidence`` record a Step cites, reached through the Step's
  ``resolved_intents``, ``search_evidence``, and ``tool_evidence``;
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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
)

from whetstone.core.identity import TypedRef
from whetstone.eval.schema_names import (
    EVAL_EVIDENCE_SCHEMA,
    EVAL_FAILURE_SCHEMA,
)
from whetstone.execution.call_support import (
    PROVIDER_ERROR_KEY,
    evidences_provider_response,
)
from whetstone.optim.cost import (
    RunCostReport,
    UsageObservation,
    aggregate_role_cost,
)
from whetstone.optim.tools.evaluator import (
    TOOL_EVAL_FAILURE_EVIDENCE_REF_KEY as _TOOL_FAILURE_EVIDENCE_REF_KEY,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore

    from whetstone.optim.contracts import OptimStepResult

__all__ = ["aggregate_run_cost"]


class EvalOutputRowUsage(BaseModel):
    """The usage fields of one persisted ``EvalOutputRow``.

    A narrow read-side projection: cost aggregation needs the usage fields
    plus enough row state to tell a call that happened from one that never
    did, and ignores the rest of the row. The field names are the persisted
    spelling pinned by ``tests/test_eval_evidence_schema_golden.py``.
    """

    model_config = ConfigDict(
        extra="ignore", allow_inf_nan=False, populate_by_name=True
    )

    prompt_tokens: StrictInt | None = None
    completion_tokens: StrictInt | None = None
    provider_cost: float | None = None
    #: The prompt cache replayed this row's call, so the usage above is the
    #: original call's and this row was not paid for again.
    cache_hit: bool = False
    #: Row state. A row that is neither failed nor missing reached the
    #: provider and got an answer back, which is a billable call whether or
    #: not the provider reported any usage telemetry alongside it.
    failed: bool = False
    missing: bool = False
    #: The failure body ``call_telemetry`` persists for a failed row. It
    #: carries a rejected response exactly when the provider generated one
    #: the classifier then turned down, which distinguishes a response-level
    #: semantic failure from a transport failure that got nothing back.
    provider_error: dict[str, Any] | None = Field(
        default=None, alias=PROVIDER_ERROR_KEY
    )

    @property
    def response_rejected(self) -> bool:
        """The provider answered and the classifier rejected the answer."""
        return evidences_provider_response(self.provider_error)


#: Evidence records that can carry output rows run cost must read. A failed
#: evaluation counts: work the provider was paid for before the failure is
#: still spend, and the failure evidence references those rows when any
#: survived. Excluding it would drop every already-billed call an
#: evaluation-level failure interrupted.
_COSTED_EVIDENCE_SCHEMAS = frozenset({EVAL_EVIDENCE_SCHEMA, EVAL_FAILURE_SCHEMA})



def _evidence_refs(result: OptimStepResult) -> tuple[TypedRef, ...]:
    """Every evaluation-evidence ref one Step paid for, in order.

    A Step cites the evaluations it paid for through whichever channel its
    mode uses, and all three are spend. A PROPOSAL_ONLY Step resolves
    intents; a searching Step adds search evidence; a TOOL_USING Step --
    the Codex arm -- drives every one of its evaluations through a tool and
    cites them only from ``tool_evidence``. The Codex arm has no proposer
    at all, since the agent does the proposing, so reading only the first
    two channels would report an entire Codex run as free.

    The three are unioned rather than treated as alternatives: the caller
    de-duplicates by ref, so a Step that somehow cited one evaluation
    through two channels is still paid for once.
    """
    refs: list[TypedRef] = []
    for resolution in result.resolved_intents:
        ref = resolution.eval_result_ref
        if ref is not None and ref.schema_name in _COSTED_EVIDENCE_SCHEMAS:
            refs.append(ref)
    for evidence in result.search_evidence:
        ref = evidence.eval_result_ref
        if ref is not None and ref.schema_name in _COSTED_EVIDENCE_SCHEMAS:
            refs.append(ref)
    for tool_evidence in result.tool_evidence:
        # A refused Tool Call carries no evaluation refs by construction,
        # and a *failed* one may still carry them: the provider work done
        # before the tool failed was billed exactly like any other
        # interrupted evaluation, which is why failure evidence is costed.
        record = tool_evidence.result.record
        for ref in record.evaluation_evidence_refs:
            if ref.schema_name in _COSTED_EVIDENCE_SCHEMAS:
                refs.append(ref)
        failure_ref = _tool_failure_evidence_ref(record)
        if failure_ref is not None:
            refs.append(failure_ref)
    return tuple(refs)


def _tool_failure_evidence_ref(record: Any) -> TypedRef | None:
    """The failure evidence ref a terminally failed Tool Result cites.

    A tool-mediated evaluation that produced ``EvalFailureEvidence``
    reaches its rows by a different route than a successful one. The
    executor builds the failed ``ToolResult`` from the evaluator's
    ``TerminalFailure`` alone, so ``evaluation_evidence_refs`` is empty
    and the only citation of the persisted evidence is the typed ref the
    evaluator put in ``details``. When that evaluation failed *after*
    provider rows were produced and persisted, those rows were billed --
    reading only the success channel drops them from task-model spend
    while the intent path counts the identical failure.

    The ref is validated rather than trusted: ``details`` is an
    open-ended JSON body, so a value that is not a well-formed
    ``TypedRef`` under the failure schema contributes nothing instead of
    raising inside cost aggregation. It is read as a ``Mapping`` rather
    than a ``dict`` because ``details`` is an ``ImmutableJsonObject``,
    whose nested objects are mappings and not dicts.
    """
    failure = getattr(record, "terminal_failure", None)
    if failure is None:
        return None
    raw = failure.details.get(_TOOL_FAILURE_EVIDENCE_REF_KEY)
    if not isinstance(raw, Mapping):
        return None
    try:
        ref = TypedRef.model_validate(dict(raw))
    except ValidationError:
        return None
    if ref.schema_name != EVAL_FAILURE_SCHEMA:
        return None
    return ref


def _task_model_observations(
    store: ObjectStore,
    refs: tuple[TypedRef, ...],
) -> tuple[UsageObservation, ...]:
    """Load the output rows behind each distinct evidence ref.

    Successful and failed evaluation evidence are read the same way: both
    reference an outputs record when they have rows, and a failure that left
    none simply contributes nothing.
    """
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

    A row the prompt cache replayed is reported as a cached call and
    contributes nothing billable, which is what keeps a cache hit from being
    charged twice. ``cache_hit`` is itself the evidence that cached work was
    served, so the row counts even when the original response carried no
    usage telemetry at all -- a provider that omits usage should look cheap,
    not absent.

    Otherwise the row is billable when the provider left usage telemetry --
    a token count or a provider-reported price -- or when the row otherwise
    evidences that the provider answered. Two row states evidence that. A row
    that is neither failed nor missing has an accepted generation. A *failed*
    row whose ``provider_error`` carries a ``rejected_response`` also has a
    generation: the provider produced it and charged for it, and only the
    classifier rejected it, which is a response-level semantic failure rather
    than a transport failure. Either way the call happened and was paid for
    even when no telemetry came back with it. Such a row counts as a call and
    as an *unpriced* one, which withholds ``usd`` rather than letting a
    partial sum present itself as a run total;
    ``rows_missing_token_breakdown`` records that its tokens are missing from
    the token totals.

    A row carries a token *breakdown* only when both directions are present.
    One direction still evidences a call, but the absent side is carried into
    the totals as zero, so the row is billable *and* flagged as missing its
    breakdown -- the same rule ``ProposerCallUsage.observation`` applies, so
    the two cost roles agree on what a breakdown is.

    Only a row with no telemetry that went missing, or failed without any
    provider response, evidences no provider call -- a failure before the
    provider answered, or a row that never ran -- and is dropped.
    """
    # Two different questions are asked of the same two fields. *Any* token
    # count is evidence the provider answered, so one direction is enough to
    # evidence a call. Only *both* directions are a token breakdown, though:
    # the absent side is carried into the totals as zero, so a row reporting
    # one direction publishes an understated token total. Collapsing these
    # into one test would either drop a call or hide that understatement.
    has_any_token_count = (
        row.prompt_tokens is not None or row.completion_tokens is not None
    )
    has_token_breakdown = (
        row.prompt_tokens is not None and row.completion_tokens is not None
    )
    has_telemetry = has_any_token_count or row.provider_cost is not None
    if row.cache_hit:
        return UsageObservation(cached=True)
    if not has_telemetry and row.missing:
        return None
    if not has_telemetry and row.failed and not row.response_rejected:
        return None
    return UsageObservation(
        input_tokens=row.prompt_tokens or 0,
        output_tokens=row.completion_tokens or 0,
        usd=row.provider_cost,
        missing_token_breakdown=not has_token_breakdown,
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
