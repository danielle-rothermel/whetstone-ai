"""Thin, payload-free operator commands over persisted platform identities."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Annotated, Any

import typer
from dbos import DBOSClient
from dr_platform import (
    CancellationInspection,
    CancellationInspectionDisposition,
    CancellationRequest,
    EligibilityReference,
    NextAttemptReason,
    NextAttemptRequest,
    OperationWaitOptions,
    PlatformSchema,
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
from sqlalchemy import create_engine

from whetstone.platform.runtime import resolve_application_database_url
from whetstone.platform.targets import target_registry

APP = typer.Typer(no_args_is_help=True)


def _emit(value: Any, *, as_json: bool) -> None:
    if as_json and hasattr(value, "model_dump_json"):
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
        if disposition is CancellationInspectionDisposition.ERROR:
            # The kernel requires a classified error for this case; leave it
            # active until bounded reconciliation classifies it safely.
            disposition = CancellationInspectionDisposition.ACTIVE
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
        _emit(list_operations(engine=engine, limit=limit), as_json=as_json)
    finally:
        engine.dispose()


@APP.command("show")
def operation_show(operation_key: str, as_json: bool = False) -> None:
    engine = _engine()
    try:
        _emit(inspect_operation(operation_key, engine=engine), as_json=as_json)
    finally:
        engine.dispose()


@APP.command("items")
def operation_items(
    operation_key: str, limit: int = 100, as_json: bool = False
) -> None:
    engine = _engine()
    try:
        _emit(
            list_items(operation_key, engine=engine, limit=limit),
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
            list_attempts(operation_key, engine=engine, limit=limit),
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
        )
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
                    connection, schema=PlatformSchema(prefix="whetstone")
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
    as_json: bool = False,
) -> None:
    if not confirm:
        raise typer.BadParameter("--confirm is required for cancellation")
    engine = _engine()
    canceller = _DbosCanceller()
    try:
        result = cancel_operation(
            CancellationRequest(
                operation_key=operation_key,
                request_id=request_id,
                requested_by=requested_by,
            ),
            engine=engine,
            canceller=canceller,
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
    as_json: bool = False,
) -> None:
    if not confirm:
        raise typer.BadParameter("--confirm is required for a cancel retry")
    engine = _engine()
    try:
        request = NextAttemptRequest(
            item_id=item_id,
            source_attempt=source_attempt,
            request_key=request_key,
            reason=NextAttemptReason.OPERATOR_CANCEL_RETRY,
            eligibility=EligibilityReference(
                kind=eligibility_kind,
                record_id=eligibility_record_id,
                digest=eligibility_digest,
            ),
            requested_by=requested_by,
            operator_confirmed_at=datetime.now(UTC),
        )
        result = request_next_attempt(
            request, engine=engine, resolver=target_registry()
        )
        _emit(result, as_json=as_json)
    finally:
        engine.dispose()
