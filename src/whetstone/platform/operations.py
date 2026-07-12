"""Thin, payload-free operator commands over persisted platform identities."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Annotated, Any

import typer
from dbos import DBOSClient
from dr_platform import (
    AttemptExecutionState,
    CancellationInspection,
    CancellationInspectionDisposition,
    CancellationRequest,
    EligibilityReference,
    FailureSnapshot,
    NextAttemptReason,
    NextAttemptRequest,
    OperationWaitOptions,
    PlatformSchema,
    RetryDisposition,
    cancel_operation,
    health_report,
    inspect_operation,
    list_attempts,
    list_items,
    list_operations,
    list_throttle_states,
    request_next_attempt,
    wait_operation,
)
from dr_platform import (
    FailureClass as PlatformFailureClass,
)
from dr_serialize import sha256_json_digest
from pydantic import BaseModel, ConfigDict
from sqlalchemy import and_, create_engine, select

from whetstone.db import io
from whetstone.db import schema as application_schema
from whetstone.eval_failures.policy import summarize_exception
from whetstone.platform.acceptance import (
    RequiredScoringProfile,
    evaluate_strict_acceptance,
    load_acceptance,
    load_current_acceptance,
)
from whetstone.platform.runtime import resolve_application_database_url
from whetstone.platform.targets import target_registry
from whetstone.records.models import GenerationRunStatus

APP = typer.Typer(no_args_is_help=True)
PLATFORM_SCHEMA = PlatformSchema(prefix="whetstone")


class AttemptPreview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    source_attempt: int
    workflow_id: str
    execution_state: str
    has_children: bool | None = None
    shared_reference: bool | None = None


class MutationPreview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str
    request_identity: dict[str, Any]
    operation_key: str
    platform_cut_version: int
    affected_attempts: tuple[AttemptPreview, ...]
    eligible: bool
    exhausted: bool
    rejection_detail: str | None = None
    preview_digest: str


def _preview_digest(value: dict[str, Any]) -> str:
    return sha256_json_digest(value)


def _mutation_preview(**values: Any) -> MutationPreview:
    digest = _preview_digest(values)
    return MutationPreview(**values, preview_digest=digest)


def _emit(value: Any, *, as_json: bool) -> None:
    if as_json and isinstance(value, tuple):
        typer.echo(
            json.dumps(
                [member.model_dump(mode="json") for member in value],
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    elif as_json and hasattr(value, "model_dump_json"):
        typer.echo(value.model_dump_json())
    elif as_json:
        typer.echo(str(value))
    elif isinstance(value, tuple):
        for member in value:
            typer.echo(member.model_dump_json())
    elif hasattr(value, "model_dump"):
        typer.echo(str(value.model_dump(mode="json")))
    else:
        typer.echo(str(value))


def _engine():
    return create_engine(resolve_application_database_url())


def _cancel_preview(
    *,
    engine: Any,
    canceller: _DbosCanceller,
    request: CancellationRequest,
) -> MutationPreview:
    operation = inspect_operation(
        request.operation_key, engine=engine, schema=PLATFORM_SCHEMA
    ).operation
    items = []
    cursor = None
    while True:
        page = list_items(
            request.operation_key,
            engine=engine,
            cursor=cursor,
            limit=500,
            schema=PLATFORM_SCHEMA,
        )
        items.extend(page)
        if len(page) < 500:
            break
        final_item = page[-1].item
        cursor = (final_item.item_index, final_item.item_id)
    attempts: list[AttemptPreview] = []
    with engine.connect() as connection:
        for item in items:
            attempt = item.current_attempt
            if attempt is None:
                continue
            terminal = attempt.execution_state in {
                AttemptExecutionState.SUCCEEDED,
                AttemptExecutionState.ERROR,
                AttemptExecutionState.RECOVERY_EXHAUSTED,
                AttemptExecutionState.CANCELLED,
                AttemptExecutionState.MISSING,
            }
            inspection = (
                None
                if terminal
                else canceller.inspect(workflow_id=attempt.workflow_id)
            )
            shared = connection.execute(
                select(PLATFORM_SCHEMA.item_attempts.c.item_id)
                .join(
                    PLATFORM_SCHEMA.items,
                    PLATFORM_SCHEMA.items.c.item_id
                    == PLATFORM_SCHEMA.item_attempts.c.item_id,
                )
                .where(
                    and_(
                        PLATFORM_SCHEMA.item_attempts.c.workflow_id
                        == attempt.workflow_id,
                        PLATFORM_SCHEMA.item_attempts.c.item_id
                        != attempt.item_id,
                        PLATFORM_SCHEMA.items.c.current_attempt
                        == PLATFORM_SCHEMA.item_attempts.c.attempt,
                        PLATFORM_SCHEMA.item_attempts.c.execution_state.not_in(
                            [state.value for state in (
                                AttemptExecutionState.SUCCEEDED,
                                AttemptExecutionState.ERROR,
                                AttemptExecutionState.RECOVERY_EXHAUSTED,
                                AttemptExecutionState.CANCELLED,
                                AttemptExecutionState.MISSING,
                            )]
                        ),
                    )
                )
                .limit(1)
            ).first()
            attempts.append(
                AttemptPreview(
                    item_id=attempt.item_id,
                    source_attempt=attempt.attempt,
                    workflow_id=attempt.workflow_id,
                    execution_state=attempt.execution_state.value,
                    has_children=(
                        inspection.has_children if inspection else None
                    ),
                    shared_reference=shared is not None,
                )
            )
    return _mutation_preview(
        command="cancel",
        request_identity=request.model_dump(mode="json"),
        operation_key=request.operation_key,
        platform_cut_version=operation.platform_cut_version,
        affected_attempts=tuple(attempts),
        eligible=True,
        exhausted=False,
        rejection_detail=None,
    )


def _authoritative_cancel_retry(
    *, engine: Any, item_id: str, source_attempt: int, requested_by: str
) -> tuple[NextAttemptRequest, MutationPreview]:
    with engine.connect() as connection:
        item = connection.execute(
            select(PLATFORM_SCHEMA.items).where(
                PLATFORM_SCHEMA.items.c.item_id == item_id
            )
        ).mappings().one()
        source = connection.execute(
            select(PLATFORM_SCHEMA.item_attempts).where(
                and_(
                    PLATFORM_SCHEMA.item_attempts.c.item_id == item_id,
                    PLATFORM_SCHEMA.item_attempts.c.attempt == source_attempt,
                )
            )
        ).mappings().one()
        operation = connection.execute(
            select(PLATFORM_SCHEMA.operations).where(
                PLATFORM_SCHEMA.operations.c.operation_key
                == item["operation_key"]
            )
        ).mappings().one()
    provenance_id = (
        source["foreign_cancellation_request_id"]
        or source["cancellation_request_id"]
    )
    eligibility_values = {
        "item_id": item_id,
        "source_attempt": source_attempt,
        "cancellation_request_id": provenance_id,
        "cancellation_origin": source["cancellation_origin"],
        "cancellation_disposition": source["cancellation_disposition"],
    }
    eligibility = EligibilityReference(
        kind="whetstone_cancellation",
        record_id=str(provenance_id or "unavailable"),
        digest=sha256_json_digest(eligibility_values),
    )
    request_key = sha256_json_digest(
        {
            "reason": NextAttemptReason.OPERATOR_CANCEL_RETRY.value,
            "eligibility": eligibility.model_dump(mode="json"),
            "operation_key": item["operation_key"],
            "platform_cut_version": operation["platform_cut_version"],
            "current_attempt": item["current_attempt"],
            "policy_max_attempts": operation["retry_policy"]["max_attempts"],
        }
    )
    request = NextAttemptRequest(
        item_id=item_id,
        source_attempt=source_attempt,
        request_key=request_key,
        reason=NextAttemptReason.OPERATOR_CANCEL_RETRY,
        eligibility=eligibility,
        requested_by=requested_by,
        operator_confirmed_at=datetime.now(UTC),
    )
    policy_max = int(operation["retry_policy"]["max_attempts"])
    exhausted = source_attempt + 1 >= policy_max
    eligible = (
        source["execution_state"] == AttemptExecutionState.CANCELLED.value
        and provenance_id is not None
        and item["current_attempt"] == source_attempt
        and not exhausted
    )
    preview = _mutation_preview(
        command="next-attempt",
        request_identity={
            **request.model_dump(mode="json"),
            "operator_confirmed_at": None,
        },
        operation_key=str(item["operation_key"]),
        platform_cut_version=int(operation["platform_cut_version"]),
        affected_attempts=(
            AttemptPreview(
                item_id=item_id,
                source_attempt=source_attempt,
                workflow_id=str(source["workflow_id"]),
                execution_state=str(source["execution_state"]),
            ),
        ),
        eligible=eligible,
        exhausted=exhausted,
        rejection_detail=(None if eligible else "source is not eligible"),
    )
    return request, preview


def _domain_outcome_request(
    *, engine: Any, kind: str, record_id: str, requested_by: str
) -> NextAttemptRequest:
    with engine.connect() as connection:
        if kind == "generation_run":
            row = connection.execute(
                select(application_schema.generation_runs).where(
                    application_schema.generation_runs.c.generation_run_id
                    == record_id
                )
            ).mappings().one()
            record = io.generation_run_record_from_row(dict(row))
            if record.status not in {
                GenerationRunStatus.ERROR,
                GenerationRunStatus.BLOCKED,
            }:
                raise ValueError(
                    "Generation Run is not a failed domain outcome"
                )
            item_id = record.platform_item_id
            source_attempt = record.platform_attempt
            payload = record.model_dump(mode="json")
            eligibility_kind = "whetstone_generation_run"
        elif kind == "score_harness_failure":
            row = connection.execute(
                select(application_schema.score_harness_failures).where(
                    application_schema.score_harness_failures.c.score_harness_failure_id
                    == record_id
                )
            ).mappings().one()
            record = io.score_harness_failure_record_from_row(dict(row))
            item_id = record.platform_item_id
            source_attempt = record.platform_attempt
            payload = record.model_dump(mode="json")
            eligibility_kind = "whetstone_score_harness_failure"
        else:
            raise ValueError(f"unsupported domain outcome kind: {kind}")
    digest = sha256_json_digest(payload)
    eligibility = EligibilityReference(
        kind=eligibility_kind, record_id=record_id, digest=digest
    )
    return NextAttemptRequest(
        item_id=item_id,
        source_attempt=source_attempt,
        request_key=sha256_json_digest(
            {
                "reason": NextAttemptReason.DOMAIN_OUTCOME.value,
                "eligibility": eligibility.model_dump(mode="json"),
            }
        ),
        reason=NextAttemptReason.DOMAIN_OUTCOME,
        eligibility=eligibility,
        requested_by=requested_by,
    )


class _DbosCanceller:
    """Minimal DBOS adapter: it reads status and never requests recursion."""

    def __init__(self) -> None:
        self.client = DBOSClient()

    def inspect(self, *, workflow_id: str) -> CancellationInspection:
        rows = self.client.list_workflows(
            workflow_ids=[workflow_id],
            limit=2,
            load_input=False,
            load_output=False,
        )
        if not rows:
            return CancellationInspection(
                workflow_id=workflow_id,
                disposition=CancellationInspectionDisposition.ABSENT,
            )
        status = str(rows[0].status)
        disposition = {
            "SUCCESS": CancellationInspectionDisposition.SUCCEEDED,
            "CANCELLED": CancellationInspectionDisposition.CANCELLED,
            "ERROR": CancellationInspectionDisposition.ERROR,
        }.get(status, CancellationInspectionDisposition.ACTIVE)
        failure = None
        retry_disposition = None
        if disposition is CancellationInspectionDisposition.ERROR:
            classified_rows = self.client.list_workflows(
                workflow_ids=[workflow_id],
                limit=2,
                load_input=False,
                load_output=True,
            )
            if (
                len(classified_rows) != 1
                or str(classified_rows[0].status) != "ERROR"
                or classified_rows[0].error is None
            ):
                raise RuntimeError(
                    "authoritative DBOS error classification is unavailable"
                )
            summary = summarize_exception(classified_rows[0].error)
            failure_class = PlatformFailureClass(summary.failure_class.value)
            failure = FailureSnapshot(
                failure_class=failure_class,
                error_type=summary.failure_exception_type,
                underlying_exception_type=summary.underlying_exception_type,
                message=summary.message,
                metadata=summary.failure_metadata,
            )
            retry_disposition = (
                RetryDisposition.RETRYABLE
                if summary.is_recoverable
                else RetryDisposition.PERMANENT
            )
        try:
            children = self.client.list_workflows(
                parent_workflow_id=workflow_id,
                limit=1,
                load_input=False,
                load_output=False,
            )
        except Exception as error:
            # A missing topology observation must never be treated as a
            # childless top-level workflow.
            raise RuntimeError(
                "unable to inspect workflow topology"
            ) from error
        return CancellationInspection(
            workflow_id=workflow_id,
            disposition=disposition,
            has_children=bool(children),
            dbos_status=status,
            failure=failure,
            retry_disposition=retry_disposition,
        )

    def cancel_workflow(
        self, *, workflow_id: str, cancel_children: bool
    ) -> None:
        self.client.cancel_workflow(
            workflow_id, cancel_children=cancel_children
        )


@APP.command("list")
def operation_list(
    limit: Annotated[int, typer.Option(min=1, max=500)] = 100,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    engine = _engine()
    try:
        _emit(
            list_operations(
                engine=engine, limit=limit, schema=PLATFORM_SCHEMA
            ),
            as_json=as_json,
        )
    finally:
        engine.dispose()


@APP.command("show")
def operation_show(operation_key: str, as_json: bool = False) -> None:
    engine = _engine()
    try:
        _emit(
            inspect_operation(
                operation_key, engine=engine, schema=PLATFORM_SCHEMA
            ),
            as_json=as_json,
        )
    finally:
        engine.dispose()


@APP.command("items")
def operation_items(
    operation_key: str, limit: int = 100, as_json: bool = False
) -> None:
    engine = _engine()
    try:
        _emit(
            list_items(
                operation_key,
                engine=engine,
                limit=limit,
                schema=PLATFORM_SCHEMA,
            ),
            as_json=as_json,
        )
    finally:
        engine.dispose()


@APP.command("attempts")
def operation_attempts(
    operation_key: str, limit: int = 100, as_json: bool = False
) -> None:
    engine = _engine()
    try:
        _emit(
            list_attempts(
                operation_key,
                engine=engine,
                limit=limit,
                schema=PLATFORM_SCHEMA,
            ),
            as_json=as_json,
        )
    finally:
        engine.dispose()


@APP.command("wait")
def operation_wait(
    operation_key: str,
    timeout_seconds: Annotated[float, typer.Option(min=0.1)] = 60.0,
    poll_interval_seconds: Annotated[float, typer.Option(min=0.1)] = 1.0,
    as_json: bool = False,
) -> None:
    engine = _engine()
    try:
        result = wait_operation(
            operation_key,
            engine=engine,
            resolver=target_registry(),
            options=OperationWaitOptions(
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                clock=lambda: datetime.now(UTC),
                sleeper=time.sleep,
            ),
            schema=PLATFORM_SCHEMA,
        )
        _emit(result, as_json=as_json)
    finally:
        engine.dispose()


@APP.command("evaluate")
def acceptance_evaluate(
    experiment_name: str,
    scoring_profile_id: str,
    scoring_profile_version: str,
    parser_profile_id: str,
    parser_version: str,
    dataset_name: str,
    dataset_split: str,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Evaluate and, when fully current, promote strict acceptance."""
    profile = RequiredScoringProfile(
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        parser_profile_id=parser_profile_id,
        parser_version=parser_version,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    engine = _engine()
    try:
        with engine.begin() as connection:
            result = evaluate_strict_acceptance(
                connection,
                experiment_name=experiment_name,
                required_profiles=(profile,),
            )
        _emit(result, as_json=as_json)
    finally:
        engine.dispose()


@APP.command("show-current")
def acceptance_show_current(
    experiment_name: str,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Load current acceptance only after revalidating its Platform cut."""
    engine = _engine()
    try:
        with engine.connect() as connection:
            result = load_current_acceptance(
                connection, experiment_name=experiment_name
            )
        _emit(result, as_json=as_json)
    finally:
        engine.dispose()


@APP.command("show-acceptance")
def acceptance_show_historical(
    acceptance_id: str,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Load one immutable historical acceptance evaluation by identity."""
    engine = _engine()
    try:
        with engine.connect() as connection:
            result = load_acceptance(connection, acceptance_id=acceptance_id)
        _emit(result, as_json=as_json)
    finally:
        engine.dispose()


@APP.command("throttle")
def throttle(as_json: bool = False) -> None:
    engine = _engine()
    try:
        with engine.connect() as connection:
            _emit(
                list_throttle_states(
                    connection, schema=PLATFORM_SCHEMA
                ),
                as_json=as_json,
            )
    finally:
        engine.dispose()


@APP.command("health")
def health(
    queued_age_threshold_seconds: int | None = None,
    active_age_threshold_seconds: int | None = None,
    no_progress_after_seconds: int | None = None,
    as_json: bool = False,
) -> None:
    engine = _engine()
    try:
        report = health_report(
            engine=engine,
            now=datetime.now(UTC),
            queued_age_threshold_seconds=queued_age_threshold_seconds,
            active_age_threshold_seconds=active_age_threshold_seconds,
            no_progress_after_seconds=no_progress_after_seconds,
            schema=PLATFORM_SCHEMA,
        )
        _emit(report, as_json=as_json)
    finally:
        engine.dispose()


@APP.command("cancel")
def cancel(
    operation_key: str,
    request_id: str,
    requested_by: str,
    confirm: bool = False,
    preview_digest: str | None = None,
    as_json: bool = False,
) -> None:
    engine = _engine()
    canceller = _DbosCanceller()
    try:
        request = CancellationRequest(
            operation_key=operation_key,
            request_id=request_id,
            requested_by=requested_by,
        )
        preview = _cancel_preview(
            engine=engine, canceller=canceller, request=request
        )
        if not confirm:
            _emit(preview, as_json=as_json)
            return
        if preview_digest is None:
            raise typer.BadParameter(
                "--preview-digest from the exact preview is required"
            )
        if preview_digest != preview.preview_digest:
            raise typer.BadParameter(
                "preview drift detected; inspect and confirm the new preview"
            )
        result = cancel_operation(
            request,
            engine=engine,
            canceller=canceller,
            schema=PLATFORM_SCHEMA,
        )
        _emit(result, as_json=as_json)
    finally:
        canceller.client.destroy()
        engine.dispose()


@APP.command("next-attempt")
def next_attempt(
    item_id: str,
    source_attempt: int,
    request_key: str,
    eligibility_kind: str,
    eligibility_record_id: str,
    eligibility_digest: str,
    requested_by: str,
    confirm: bool = False,
    preview_digest: str | None = None,
    as_json: bool = False,
) -> None:
    engine = _engine()
    try:
        request, preview = _authoritative_cancel_retry(
            engine=engine,
            item_id=item_id,
            source_attempt=source_attempt,
            requested_by=requested_by,
        )
        # These positional fields remain accepted for CLI compatibility, but
        # cannot authorize the transition; persisted provenance owns identity.
        _ = (
            request_key,
            eligibility_kind,
            eligibility_record_id,
            eligibility_digest,
        )
        if not confirm:
            _emit(preview, as_json=as_json)
            return
        if preview_digest is None:
            raise typer.BadParameter(
                "--preview-digest from the exact preview is required"
            )
        if preview_digest != preview.preview_digest:
            raise typer.BadParameter(
                "preview drift detected; inspect and confirm the new preview"
            )
        result = request_next_attempt(
            request,
            engine=engine,
            resolver=target_registry(),
            schema=PLATFORM_SCHEMA,
        )
        _emit(result, as_json=as_json)
    finally:
        engine.dispose()


@APP.command("generation-next-attempt")
def generation_next_attempt(
    generation_run_id: str,
    requested_by: str,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Request a retry from one authoritative failed Generation Run."""
    engine = _engine()
    try:
        request = _domain_outcome_request(
            engine=engine,
            kind="generation_run",
            record_id=generation_run_id,
            requested_by=requested_by,
        )
        _emit(
            request_next_attempt(
                request,
                engine=engine,
                resolver=target_registry(),
                schema=PLATFORM_SCHEMA,
            ),
            as_json=as_json,
        )
    finally:
        engine.dispose()


@APP.command("scoring-next-attempt")
def scoring_next_attempt(
    score_harness_failure_id: str,
    requested_by: str,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Request a retry from one authoritative Score Harness Failure."""
    engine = _engine()
    try:
        request = _domain_outcome_request(
            engine=engine,
            kind="score_harness_failure",
            record_id=score_harness_failure_id,
            requested_by=requested_by,
        )
        _emit(
            request_next_attempt(
                request,
                engine=engine,
                resolver=target_registry(),
                schema=PLATFORM_SCHEMA,
            ),
            as_json=as_json,
        )
    finally:
        engine.dispose()
