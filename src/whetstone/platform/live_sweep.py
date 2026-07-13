"""Fail-closed operator commands for the immutable HumanEval live sweep.

The ledger is deliberately local to an operator run: it contains identities and
money facts, never prompts, provider headers, or credentials.  ``--execute``
is the only path that can call Platform submission; provider work is performed
later by the existing worker.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from dr_platform import (
    AttemptRecord,
    EligibilityReference,
    NextAttemptReason,
    NextAttemptRequest,
    list_attempts,
    request_next_attempt,
)
from dr_platform.items import item_id
from dr_platform.status import AttemptExecutionState
from dr_providers import FailureClass
from dr_serialize import sha256_json_digest
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine

from whetstone.db import schema as whetstone_schema
from whetstone.lm.boundary import OUTPUT_FIELD_TEXT
from whetstone.platform.dataset_snapshot import (
    HumanEvalSnapshot,
    load_humaneval_snapshot,
)
from whetstone.platform.runtime import resolve_application_database_url
from whetstone.platform.spec_builder import (
    iter_experiment_specs_from_file,
    load_model_config_fragment,
)
from whetstone.platform.submission import submit_prediction_specs
from whetstone.platform.targets import target_registry
from whetstone.records import (
    DatasetSnapshotIdentityPayload,
    GenerationRunStatus,
)

APP = typer.Typer(no_args_is_help=True)
EXPECTED_CELLS = 5_904
CANARY_CELLS = 12
GENERATION_CEILING_USD = Decimal("4.62")
MAX_RETRIES_PER_CELL = 2

_PLATFORM_TERMINAL_FAILURES = frozenset(
    {
        AttemptExecutionState.ERROR,
        AttemptExecutionState.RECOVERY_EXHAUSTED,
        AttemptExecutionState.CANCELLED,
        AttemptExecutionState.MISSING,
    }
)
_PLATFORM_IN_FLIGHT = frozenset(
    {
        AttemptExecutionState.NOT_STARTED,
        AttemptExecutionState.ACTIVE,
        AttemptExecutionState.CANCEL_REQUESTED,
    }
)


class AdapterDisposition(StrEnum):
    """Allowlisted outcome of the durable node-adapter boundary."""

    SUCCESS = "success"
    MISSING_OUTPUT = "missing_output"
    BLANK_OUTPUT = "blank_output"
    PARSE_FAILURE = "parse_failure"
    PROVIDER_FAILURE = "provider_failure"
    FAILURE = "failure"


class TypedFailureCode(StrEnum):
    """Stable, non-message failure codes available in existing records."""

    EMPTY_GENERATION = "empty_generation"
    PREDICTION_PARSE = "prediction_parse"
    PROVIDER_RESPONSE_PARSE = "provider_response_parse"
    PROVIDER_FAILURE = "provider_failure"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class LiveSweepDiagnostics:
    """The complete ledger-safe node diagnostic allowlist.

    This deliberately contains no raw node output, response metadata, failure
    message, failure metadata, request data, or provider configuration.
    """

    response_id_hash: str | None
    returned_model: str | None
    finish_reason: str | None
    node_status: str | None
    expected_output_field: str
    output_field_present: bool
    output_nonblank: bool
    parser_profile: str | None
    parser_version: str | None
    parser_status: str | None
    adapter_disposition: AdapterDisposition | None
    typed_failure_class: FailureClass | None
    typed_failure_code: TypedFailureCode | None

    def as_dict(self) -> dict[str, Any]:
        """Return the fixed ledger representation without unavailable facts."""
        return {
            key: value.value if isinstance(value, StrEnum) else value
            for key, value in self.__dict__.items()
            if value is not None
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class CellReconciliation:
    """Read-only cross-store facts for one immutable manifest cell."""

    cell_id: str
    status: str
    platform_attempt: int | None
    retry_count: int
    actual_cost: Decimal | None
    provider_tokens: dict[str, int]
    error_classification: str | None
    score_status: str | None = None
    diagnostics: dict[str, Any] | None = None


def _attempt_facts(
    engine: Engine, operation_key: str
) -> dict[tuple[str, int], AttemptRecord]:
    """Page the public Platform inspector; never read Platform tables here."""
    cursor: tuple[str, int] | None = None
    observations: dict[tuple[str, int], AttemptRecord] = {}
    while True:
        page = list_attempts(
            operation_key,
            engine=engine,
            cursor=cursor,
            limit=100,
        )
        if not page:
            return observations
        for observation in page:
            attempt = observation.attempt
            observations[(attempt.item_id, attempt.attempt)] = attempt
        last = page[-1].attempt
        cursor = (last.item_id, last.attempt)


def _node_cost_facts(
    engine: Engine,
    *,
    prediction_id: str,
    platform_item_id: str,
    platform_attempt: int,
) -> tuple[GenerationRunStatus | None, Decimal | None, dict[str, int]]:
    """Return terminal Whetstone run state and durable node costs only."""
    with engine.connect() as connection:
        generation = (
            connection.execute(
                select(whetstone_schema.generation_runs).where(
                    whetstone_schema.generation_runs.c.prediction_id
                    == prediction_id,
                    whetstone_schema.generation_runs.c.platform_item_id
                    == platform_item_id,
                    whetstone_schema.generation_runs.c.platform_attempt
                    == platform_attempt,
                )
            )
            .mappings()
            .one_or_none()
        )
        if generation is None:
            return None, None, {}
        node_rows = connection.execute(
            select(whetstone_schema.node_attempts.c.usage_cost).where(
                whetstone_schema.node_attempts.c.generation_run_id
                == generation["generation_run_id"]
            )
        ).scalars()
        costs: list[Decimal] = []
        tokens: dict[str, int] = {}
        saw_node_attempt = False
        for usage in node_rows:
            saw_node_attempt = True
            payload = usage or {}
            cost = payload.get("provider_cost")
            if cost is None:
                return GenerationRunStatus(generation["status"]), None, tokens
            try:
                costs.append(_money(cost, field="persisted provider cost"))
            except ValueError:
                return GenerationRunStatus(generation["status"]), None, tokens
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = payload.get("usage_metadata", {}).get(key)
                if isinstance(value, int) and value >= 0:
                    tokens[key] = tokens.get(key, 0) + value
    if not saw_node_attempt:
        return GenerationRunStatus(generation["status"]), None, tokens
    return (
        GenerationRunStatus(generation["status"]),
        sum(costs, Decimal()),
        tokens,
    )


def _score_terminal_status(
    engine: Engine,
    *,
    prediction_id: str,
    platform_item_id: str,
    platform_attempt: int,
) -> str | None:
    """Read the terminal scoring record using the generation stable keys."""
    with engine.connect() as connection:
        generation_run_id = connection.execute(
            select(whetstone_schema.generation_runs.c.generation_run_id).where(
                whetstone_schema.generation_runs.c.prediction_id
                == prediction_id,
                whetstone_schema.generation_runs.c.platform_item_id
                == platform_item_id,
                whetstone_schema.generation_runs.c.platform_attempt
                == platform_attempt,
            )
        ).scalar_one_or_none()
        if generation_run_id is None:
            return None
        score = connection.execute(
            select(whetstone_schema.score_attempts.c.status).where(
                whetstone_schema.score_attempts.c.prediction_id
                == prediction_id,
                whetstone_schema.score_attempts.c.generation_run_id
                == generation_run_id,
                whetstone_schema.score_attempts.c.platform_item_id
                == platform_item_id,
                whetstone_schema.score_attempts.c.platform_attempt
                == platform_attempt,
            )
        ).scalar_one_or_none()
        if score is not None:
            return f"score_{score}"
        harness = connection.execute(
            select(
                whetstone_schema.score_harness_failures.c.score_attempt_id
            ).where(
                whetstone_schema.score_harness_failures.c.prediction_id
                == prediction_id,
                whetstone_schema.score_harness_failures.c.generation_run_id
                == generation_run_id,
                whetstone_schema.score_harness_failures.c.platform_item_id
                == platform_item_id,
                whetstone_schema.score_harness_failures.c.platform_attempt
                == platform_attempt,
            )
        ).scalar_one_or_none()
    return "score_harness_failure" if harness is not None else None


def _safe_diagnostics(
    engine: Engine,
    *,
    prediction_id: str,
    platform_item_id: str,
    platform_attempt: int,
) -> dict[str, Any]:
    """Project allowlisted durable fields; never copy provider payloads."""
    with engine.connect() as connection:
        generation_id = connection.execute(
            select(whetstone_schema.generation_runs.c.generation_run_id).where(
                whetstone_schema.generation_runs.c.prediction_id
                == prediction_id,
                whetstone_schema.generation_runs.c.platform_item_id
                == platform_item_id,
                whetstone_schema.generation_runs.c.platform_attempt
                == platform_attempt,
            )
        ).scalar_one_or_none()
        if generation_id is None:
            return {}
        node = (
            connection.execute(
                select(
                    whetstone_schema.node_attempts.c.status,
                    whetstone_schema.node_attempts.c.model,
                    whetstone_schema.node_attempts.c.output,
                    whetstone_schema.node_attempts.c.response_metadata,
                    whetstone_schema.node_attempts.c.failure,
                )
                .where(
                    whetstone_schema.node_attempts.c.generation_run_id
                    == generation_id
                )
                .order_by(
                    whetstone_schema.node_attempts.c.attempt_index.desc()
                )
            )
            .mappings()
            .first()
        )
        score = (
            connection.execute(
                select(
                    whetstone_schema.score_attempts.c.parser_profile_id,
                    whetstone_schema.score_attempts.c.parser_version,
                    whetstone_schema.score_attempts.c.status,
                )
                .where(
                    whetstone_schema.score_attempts.c.generation_run_id
                    == generation_id
                )
                .order_by(
                    whetstone_schema.score_attempts.c.attempt_index.desc()
                )
            )
            .mappings()
            .first()
        )
    return _project_safe_diagnostics(
        node=dict(node) if node is not None else None,
        score=dict(score) if score is not None else None,
    )


def _project_safe_diagnostics(
    *,
    node: Mapping[str, Any] | None,
    score: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project one node row through the fixed payload-free diagnostic model."""
    output = node.get("output") if node is not None else None
    output_metadata = (
        output.get("metadata") if isinstance(output, Mapping) else None
    )
    response_metadata = (
        node.get("response_metadata") if node is not None else None
    )
    response_id = _allowlisted_string(output_metadata, "response_id")
    if response_id is None:
        response_id = _allowlisted_string(response_metadata, "id")
    model = _allowlisted_string(output_metadata, "model")
    if model is None:
        model = _allowlisted_string(response_metadata, "model")
    if model is None and node is not None:
        model = _allowlisted_string(node, "model")
    finish_reason = _allowlisted_string(output_metadata, "finish_reason")
    if finish_reason is None:
        finish_reason = _allowlisted_string(response_metadata, "finish_reason")
    output_field_present, output_nonblank = _output_field_facts(output)
    failure = node.get("failure") if node is not None else None
    disposition, failure_class, failure_code = _adapter_facts(
        node_status=_allowlisted_string(node, "status"),
        failure=failure,
        output_field_present=output_field_present,
        output_nonblank=output_nonblank,
    )
    diagnostics = LiveSweepDiagnostics(
        response_id_hash=(
            hashlib.sha256(response_id.encode()).hexdigest()
            if response_id is not None
            else None
        ),
        returned_model=model,
        finish_reason=finish_reason,
        node_status=_allowlisted_string(node, "status"),
        expected_output_field=OUTPUT_FIELD_TEXT,
        output_field_present=output_field_present,
        output_nonblank=output_nonblank,
        parser_profile=_allowlisted_string(score, "parser_profile_id"),
        parser_version=_allowlisted_string(score, "parser_version"),
        parser_status=_allowlisted_string(score, "status"),
        adapter_disposition=disposition,
        typed_failure_class=failure_class,
        typed_failure_code=failure_code,
    )
    return diagnostics.as_dict()


def _allowlisted_string(
    payload: Mapping[str, Any] | None, key: str
) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _output_field_facts(output: object) -> tuple[bool, bool]:
    """Report key presence and useful content without map truthiness."""
    if not isinstance(output, Mapping):
        return False, False
    values = output.get("values")
    if not isinstance(values, Mapping) or OUTPUT_FIELD_TEXT not in values:
        return False, False
    value = values.get(OUTPUT_FIELD_TEXT)
    if isinstance(value, str):
        return True, bool(value.strip())
    if isinstance(value, Mapping | tuple | list | set):
        return True, bool(value)
    return True, value is not None


def _adapter_facts(
    *,
    node_status: str | None,
    failure: object,
    output_field_present: bool,
    output_nonblank: bool,
) -> tuple[
    AdapterDisposition | None,
    FailureClass | None,
    TypedFailureCode | None,
]:
    """Classify only stable Whetstone/dr-providers failure taxonomy fields."""
    if node_status == "success":
        if not output_field_present:
            return AdapterDisposition.MISSING_OUTPUT, None, None
        if not output_nonblank:
            return AdapterDisposition.BLANK_OUTPUT, None, None
        return AdapterDisposition.SUCCESS, None, None
    if node_status != "error":
        return None, None, None
    failure_mapping = failure if isinstance(failure, Mapping) else {}
    raw_class = failure_mapping.get("failure_class")
    failure_class = (
        FailureClass(raw_class)
        if isinstance(raw_class, str)
        and raw_class in {item.value for item in FailureClass}
        else None
    )
    error_type = failure_mapping.get("error_type")
    underlying_type = failure_mapping.get("underlying_exception_type")
    type_names = {
        value.rsplit(".", 1)[-1]
        for value in (error_type, underlying_type)
        if isinstance(value, str)
    }
    if "EmptyGenerationError" in type_names:
        return (
            AdapterDisposition.BLANK_OUTPUT,
            failure_class,
            TypedFailureCode.EMPTY_GENERATION,
        )
    if "PredictionParseError" in type_names:
        return (
            AdapterDisposition.PARSE_FAILURE,
            failure_class,
            TypedFailureCode.PREDICTION_PARSE,
        )
    if "ProviderResponseParseError" in type_names:
        return (
            AdapterDisposition.PARSE_FAILURE,
            failure_class,
            TypedFailureCode.PROVIDER_RESPONSE_PARSE,
        )
    metadata = failure_mapping.get("metadata")
    if (
        "ProviderFailureError" in type_names
        or (
            isinstance(metadata, Mapping)
            and "provider_failure" in metadata
        )
    ):
        return (
            AdapterDisposition.PROVIDER_FAILURE,
            failure_class,
            TypedFailureCode.PROVIDER_FAILURE,
        )
    return (
        AdapterDisposition.FAILURE,
        failure_class,
        TypedFailureCode.UNCLASSIFIED,
    )


def reconcile_ledger(
    ledger: SweepLedger,
    *,
    engine: Engine,
) -> list[CellReconciliation]:
    """Reconcile stable IDs, retaining reservations for unknown cost."""
    by_operation = {
        str(row["operation_key"])
        for row in ledger.rows()
        if row["operation_key"] is not None
    }
    platform = {
        operation_key: _attempt_facts(engine, operation_key)
        for operation_key in by_operation
    }
    facts: list[CellReconciliation] = []
    for row in ledger.rows():
        cell_id = str(row["cell_id"])
        retry_count = int(row["retry_count"])
        operation_key = row["operation_key"]
        prediction_id = row["prediction_id"]
        platform_item_id = row["platform_item_id"]
        platform_attempt = row["platform_attempt"]
        if any(
            value is None
            for value in (
                operation_key,
                prediction_id,
                platform_item_id,
                platform_attempt,
            )
        ):
            facts.append(
                CellReconciliation(
                    cell_id=cell_id,
                    status="orphan_reservation",
                    platform_attempt=None,
                    retry_count=retry_count,
                    actual_cost=None,
                    provider_tokens={},
                    error_classification="missing_stable_submission_identity",
                )
            )
            continue
        attempt = platform[str(operation_key)].get(
            (str(platform_item_id), int(platform_attempt))
        )
        if attempt is None:
            facts.append(
                CellReconciliation(
                    cell_id=cell_id,
                    status="unknown",
                    platform_attempt=int(platform_attempt),
                    retry_count=retry_count,
                    actual_cost=None,
                    provider_tokens={},
                    error_classification="platform_attempt_not_observed",
                )
            )
            continue
        run_status, actual_cost, tokens = _node_cost_facts(
            engine,
            prediction_id=str(prediction_id),
            platform_item_id=str(platform_item_id),
            platform_attempt=int(platform_attempt),
        )
        score_status = _score_terminal_status(
            engine,
            prediction_id=str(prediction_id),
            platform_item_id=str(platform_item_id),
            platform_attempt=int(platform_attempt),
        )
        diagnostics = _safe_diagnostics(
            engine,
            prediction_id=str(prediction_id),
            platform_item_id=str(platform_item_id),
            platform_attempt=int(platform_attempt),
        )
        if run_status in {
            GenerationRunStatus.ERROR,
            GenerationRunStatus.BLOCKED,
        }:
            status, error = "typed_failure", f"generation_{run_status.value}"
        elif attempt.execution_state in _PLATFORM_TERMINAL_FAILURES:
            status, error = (
                "typed_failure",
                f"platform_{attempt.execution_state.value}",
            )
        elif (
            attempt.execution_state is AttemptExecutionState.SUCCEEDED
            and run_status
            in {GenerationRunStatus.SUCCESS, GenerationRunStatus.PARTIAL}
        ):
            status, error = "succeeded", None
        elif attempt.execution_state in _PLATFORM_IN_FLIGHT:
            status, error = "in_flight", None
        elif attempt.execution_state is AttemptExecutionState.SUCCEEDED:
            status, error = "incomplete", "missing_terminal_generation_run"
        else:
            status, error = "unknown", "unrecognized_platform_lifecycle"
        facts.append(
            CellReconciliation(
                cell_id=cell_id,
                status=status,
                platform_attempt=int(platform_attempt),
                retry_count=retry_count,
                actual_cost=actual_cost,
                provider_tokens=tokens,
                error_classification=error,
                score_status=score_status,
                diagnostics=diagnostics,
            )
        )
    ledger.reconciliation(facts)
    return facts


def require_known_actual_costs(
    facts: list[CellReconciliation], *, cell_ids: set[str]
) -> None:
    """Fail closed until every selected cell has observed provider cost."""
    stable_statuses = {"succeeded", "typed_failure", "incomplete"}
    selected = {
        fact.cell_id: fact for fact in facts if fact.cell_id in cell_ids
    }
    missing = cell_ids - set(selected)
    unknown = sorted(
        cell_id
        for cell_id, fact in selected.items()
        if fact.actual_cost is None or fact.status not in stable_statuses
    )
    if missing or unknown:
        blocked = sorted(missing) + unknown
        preview = ", ".join(blocked[:5])
        raise RuntimeError(
            "actual provider cost is not established for the bounded page; "
            f"reconcile before dispatching more cells ({preview})"
        )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)


def _money(value: object, *, field: str) -> Decimal:
    """Accept only finite, non-negative JSON numeric values as USD."""
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ValueError(f"{field} must be a JSON number")
    try:
        amount = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field} is not a decimal amount") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return amount


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _stored_money(value: object, *, field: str) -> Decimal:
    if not isinstance(value, str):
        return _money(value, field=field)
    try:
        amount = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{field} is corrupt") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{field} is corrupt")
    return amount


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec_digest(spec: Any) -> str:
    """Digest the exact JSON contract passed to Platform."""
    return sha256_json_digest(
        spec.model_dump(mode="json", exclude={"created_at"})
    )


def _cell_operation_key(cell: Mapping[str, Any]) -> str:
    value = cell.get("operation_key")
    if not isinstance(value, str) or not value:
        raise ValueError("locked cell has no deterministic operation key")
    return value


class SweepLedger:
    """Run-scoped, WAL-backed journal with atomic ceiling checks."""

    def __init__(self, path: Path, *, manifest_hash: str) -> None:
        if not path.is_absolute():
            raise ValueError(
                "ledger path must be absolute and outside the repository"
            )
        self.path = path
        self.manifest_hash = manifest_hash
        self.connection = sqlite3.connect(
            path, isolation_level=None, timeout=30
        )
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sweep_cells (
              manifest_hash TEXT NOT NULL, cell_id TEXT NOT NULL,
              estimated_cost TEXT NOT NULL, reserved_cost TEXT,
              actual_cost TEXT, operation_key TEXT, prediction_id TEXT,
              platform_item_id TEXT, platform_attempt INTEGER,
              attempt_ids_json TEXT NOT NULL DEFAULT '[]',
              status TEXT NOT NULL, retry_count INTEGER NOT NULL DEFAULT 0,
              retry_of_attempt INTEGER, error_classification TEXT,
              provider_tokens_json TEXT, score_status TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL, PRIMARY KEY (manifest_hash, cell_id)
            );
            CREATE TABLE IF NOT EXISTS sweep_events (
              id INTEGER PRIMARY KEY, manifest_hash TEXT NOT NULL,
              cell_id TEXT NOT NULL, event TEXT NOT NULL,
              detail_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            """
        )
        self._ensure_column("attempt_ids_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("score_status", "TEXT")
        self._ensure_column("diagnostics_json", "TEXT")

    def _ensure_column(self, name: str, definition: str) -> None:
        columns = {
            str(row[1])
            for row in self.connection.execute(
                "PRAGMA table_info(sweep_cells)"
            )
        }
        if name not in columns:
            self.connection.execute(
                f"ALTER TABLE sweep_cells ADD COLUMN {name} {definition}"
            )

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def _totals(
        self, connection: sqlite3.Connection
    ) -> tuple[Decimal, Decimal]:
        rows = connection.execute(
            "SELECT actual_cost,reserved_cost FROM sweep_cells "
            "WHERE manifest_hash=?",
            (self.manifest_hash,),
        ).fetchall()
        return (
            sum(
                (
                    _stored_money(row[0], field="stored actual cost")
                    for row in rows
                    if row[0] is not None
                ),
                Decimal(),
            ),
            sum(
                (
                    _stored_money(row[1], field="stored reserved cost")
                    for row in rows
                    if row[1] is not None
                ),
                Decimal(),
            ),
        )

    def reserve(
        self, cells: list[dict[str, Any]], estimates: Mapping[str, object]
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        with self._transaction() as connection:
            actual, reserved = self._totals(connection)
            for cell in cells:
                cell_id = str(cell["cell_id"])
                estimate = estimates.get(cell_id)
                if estimate is None:
                    raise ValueError(
                        f"unknown or invalid cost estimate for {cell_id}"
                    )
                estimate = _money(estimate, field=f"estimate for {cell_id}")
                row = connection.execute(
                    "SELECT status FROM sweep_cells "
                    "WHERE manifest_hash=? AND cell_id=?",
                    (self.manifest_hash, cell_id),
                ).fetchone()
                if row is not None:
                    # A process may die after durable intent and before its
                    # local acknowledgement.  Return that exact intent for
                    # replay; its Platform operation key is immutable.
                    if row[0] == "submitting":
                        selected.append(cell)
                    continue
                if actual + reserved + estimate > GENERATION_CEILING_USD:
                    raise ValueError(
                        "authorized generation ceiling would be exceeded"
                    )
                timestamp = _now()
                connection.execute(
                    "INSERT INTO sweep_cells("
                    "manifest_hash,cell_id,estimated_cost,reserved_cost,"
                    "status,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (
                        self.manifest_hash,
                        cell_id,
                        _decimal_text(estimate),
                        _decimal_text(estimate),
                        "reserved",
                        timestamp,
                        timestamp,
                    ),
                )
                connection.execute(
                    "INSERT INTO sweep_events("
                    "manifest_hash,cell_id,event,detail_json,created_at) "
                    "VALUES(?,?,?,?,?)",
                    (self.manifest_hash, cell_id, "reserved", "{}", timestamp),
                )
                reserved += estimate
                selected.append(cell)
        return selected

    def submitted(
        self,
        cells: list[dict[str, Any]],
        *,
        operation_key: str,
        prediction_ids: dict[str, str],
    ) -> None:
        with self._transaction() as connection:
            for cell in cells:
                cell_id = str(cell["cell_id"])
                connection.execute(
                    "UPDATE sweep_cells SET status='submitted',"
                    "operation_key=?,prediction_id=?,platform_item_id=?,"
                    "platform_attempt=0,attempt_ids_json='[0]',updated_at=? "
                    "WHERE manifest_hash=? AND cell_id=?",
                    (
                        operation_key,
                        prediction_ids.get(cell_id),
                        item_id(
                            operation_key=operation_key,
                            item_key=prediction_ids[cell_id],
                        ),
                        _now(),
                        self.manifest_hash,
                        cell_id,
                    ),
                )

    def submission_intent(
        self,
        cells: list[dict[str, Any]],
        *,
        operation_key: str,
        prediction_ids: dict[str, str],
    ) -> None:
        """Durably bind idempotent Platform identities before submission."""
        with self._transaction() as connection:
            for cell in cells:
                cell_id = str(cell["cell_id"])
                prediction_id = prediction_ids[cell_id]
                connection.execute(
                    "UPDATE sweep_cells SET status='submitting',"
                    "operation_key=?,prediction_id=?,platform_item_id=?,"
                    "platform_attempt=0,"
                    "attempt_ids_json='[0]',updated_at=? "
                    "WHERE manifest_hash=? AND cell_id=? "
                    "AND status IN ('reserved','submitting')",
                    (
                        operation_key,
                        prediction_id,
                        item_id(
                            operation_key=operation_key, item_key=prediction_id
                        ),
                        _now(),
                        self.manifest_hash,
                        cell_id,
                    ),
                )

    def selected_remaining(
        self, all_cells: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT cell_id FROM sweep_cells WHERE manifest_hash=?",
            (self.manifest_hash,),
        ).fetchall()
        excluded = {str(row[0]) for row in rows}
        return [
            cell for cell in all_cells if str(cell["cell_id"]) not in excluded
        ]

    def pending_submission(
        self, all_cells: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        pending = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT cell_id FROM sweep_cells WHERE manifest_hash=? "
                "AND status='submitting'",
                (self.manifest_hash,),
            ).fetchall()
        }
        return [cell for cell in all_cells if str(cell["cell_id"]) in pending]

    def summary(self) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT status, COUNT(*) FROM sweep_cells "
            "WHERE manifest_hash=? GROUP BY status",
            (self.manifest_hash,),
        ).fetchall()
        return {
            str(status): {
                "count": count,
                "reserved_usd": float(
                    sum(
                        (
                            _stored_money(row[1], field="stored reserved cost")
                            for row in self.connection.execute(
                                "SELECT actual_cost,reserved_cost "
                                "FROM sweep_cells "
                                "WHERE manifest_hash=? AND status=?",
                                (self.manifest_hash, status),
                            ).fetchall()
                            if row[1] is not None
                        ),
                        Decimal(),
                    )
                ),
                "actual_usd": float(
                    sum(
                        (
                            _stored_money(row[0], field="stored actual cost")
                            for row in self.connection.execute(
                                "SELECT actual_cost,reserved_cost "
                                "FROM sweep_cells "
                                "WHERE manifest_hash=? AND status=?",
                                (self.manifest_hash, status),
                            ).fetchall()
                            if row[0] is not None
                        ),
                        Decimal(),
                    )
                ),
            }
            for status, count in rows
        } | {
            "manifest_mismatch": {
                "count": self.connection.execute(
                    "SELECT COUNT(*) FROM sweep_cells WHERE manifest_hash!=?",
                    (self.manifest_hash,),
                ).fetchone()[0],
                "reserved_usd": 0,
                "actual_usd": 0,
            }
        }

    def rows(self) -> list[sqlite3.Row]:
        self.connection.row_factory = sqlite3.Row
        return self.connection.execute(
            "SELECT * FROM sweep_cells WHERE manifest_hash=? ORDER BY cell_id",
            (self.manifest_hash,),
        ).fetchall()

    def reconciliation(self, facts: list[CellReconciliation]) -> None:
        with self._transaction() as connection:
            for fact in facts:
                connection.execute(
                    "UPDATE sweep_cells SET status=?,actual_cost=?,"
                    "reserved_cost=CASE WHEN ? IS NULL THEN reserved_cost "
                    "ELSE NULL END,"
                    "error_classification=?,provider_tokens_json=?,"
                    "score_status=?,diagnostics_json=?,"
                    "updated_at=? "
                    "WHERE manifest_hash=? AND cell_id=?",
                    (
                        fact.status,
                        _decimal_text(fact.actual_cost)
                        if fact.actual_cost is not None
                        else None,
                        _decimal_text(fact.actual_cost)
                        if fact.actual_cost is not None
                        else None,
                        fact.error_classification,
                        json.dumps(fact.provider_tokens, sort_keys=True),
                        fact.score_status,
                        json.dumps(fact.diagnostics or {}, sort_keys=True),
                        _now(),
                        self.manifest_hash,
                        fact.cell_id,
                    ),
                )

    def claim_retry(self, fact: CellReconciliation) -> bool:
        if fact.status not in {"typed_failure", "incomplete"}:
            return False
        if (
            fact.platform_attempt is None
            or fact.retry_count >= MAX_RETRIES_PER_CELL
            or fact.actual_cost is None
        ):
            return False
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT estimated_cost FROM sweep_cells "
                "WHERE manifest_hash=? AND cell_id=?",
                (self.manifest_hash, fact.cell_id),
            ).fetchone()
            if row is None:
                return False
            estimate = _stored_money(row[0], field="stored estimate")
            actual, reserved = self._totals(connection)
            if actual + reserved + estimate > GENERATION_CEILING_USD:
                return False
            result = connection.execute(
                "UPDATE sweep_cells SET status='retrying',reserved_cost=?,"
                "retry_of_attempt=?,"
                "updated_at=? WHERE manifest_hash=? AND cell_id=? "
                "AND retry_count<? AND status IN "
                "('typed_failure','incomplete','retrying')",
                (
                    _decimal_text(estimate),
                    fact.platform_attempt,
                    _now(),
                    self.manifest_hash,
                    fact.cell_id,
                    MAX_RETRIES_PER_CELL,
                ),
            )
            return result.rowcount == 1

    def retried(
        self,
        *,
        cell_id: str,
        source_attempt: int,
        created_attempt: int | None,
    ) -> None:
        if created_attempt is None:
            return
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT retry_of_attempt,retry_count,attempt_ids_json "
                "FROM sweep_cells "
                "WHERE manifest_hash=? AND cell_id=?",
                (self.manifest_hash, cell_id),
            ).fetchone()
            if row is None:
                raise ValueError("retry cell disappeared from ledger")
            attempts = json.loads(str(row[2]))
            is_new_attempt = created_attempt not in attempts
            if is_new_attempt:
                attempts.append(created_attempt)
            retry_count = int(row[1])
            if is_new_attempt:
                retry_count += 1
            connection.execute(
                "UPDATE sweep_cells SET status='submitted',retry_count=?,"
                "retry_of_attempt=?,platform_attempt=?,attempt_ids_json=?,"
                "updated_at=? "
                "WHERE manifest_hash=? AND cell_id=?",
                (
                    retry_count,
                    source_attempt,
                    created_attempt,
                    json.dumps(attempts),
                    _now(),
                    self.manifest_hash,
                    cell_id,
                ),
            )


def validate_campaign(
    campaign_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    metadata = _load_json(campaign_dir / "campaign-metadata.json")
    manifest_path = campaign_dir / "manifest.jsonl"
    manifest_hash = _sha256(manifest_path)
    index = _load_json(campaign_dir / "manifest-index.json")
    if index.get("manifest_sha256") != manifest_hash:
        raise typer.BadParameter("manifest hash does not match locked index")
    cells = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline = _load_json(campaign_dir / "legacy-baseline-tasks.json")
    baseline_by_task = {row["task_id"]: row for row in baseline}
    if len(baseline_by_task) != 164:
        raise typer.BadParameter("locked task baseline is incomplete")
    for cell in cells:
        task = baseline_by_task.get(cell["task_id"])
        if task is None or task.get("task_definition_sha256") != cell.get(
            "task_definition_sha256"
        ):
            raise typer.BadParameter(
                "locked task definition does not match cell"
            )
        definition = task.get("legacy_baseline_definition")
        if not isinstance(definition, dict) or sha256_json_digest(
            definition
        ) != task.get("task_definition_sha256"):
            raise typer.BadParameter(
                "locked task definition digest is invalid"
            )
    if (
        len(cells) != EXPECTED_CELLS
        or metadata.get("expected_cell_count") != EXPECTED_CELLS
    ):
        raise typer.BadParameter("campaign must contain exactly 5,904 cells")
    if len({cell["cell_id"] for cell in cells}) != len(cells):
        raise typer.BadParameter("campaign cell IDs must be unique")
    if {cell["budget_key"] for cell in cells} != {
        "direct",
        "1",
        "0.75",
        "0.5",
    }:
        raise typer.BadParameter(
            "campaign budgets do not match approved matrix"
        )
    for model in metadata["models"]:
        fragment = campaign_dir / "models" / f"{model['slug']}-openrouter.json"
        if model["provider_kind"] == "openai":
            fragment = campaign_dir / "models" / "gpt54-nano-openai.json"
        validated = load_model_config_fragment(fragment)
        if any(
            provider.model != model["model"]
            for provider in validated.providers
        ):
            raise typer.BadParameter("model fragment does not match campaign")
    try:
        _specs_for_cells(campaign_dir, metadata, cells)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    return metadata, cells, manifest_hash


def _estimates(path: Path, cells: list[dict[str, Any]]) -> dict[str, Decimal]:
    """Read a locked per-cell estimate artifact; aggregates are unsafe."""
    payload = _load_json(path)
    if payload.get("manifest_sha256") is None or not isinstance(
        payload.get("cells"), dict
    ):
        raise typer.BadParameter(
            "estimate artifact must contain manifest_sha256 and cells map"
        )
    try:
        result = {
            str(key): _money(value, field=f"estimate for {key}")
            for key, value in payload["cells"].items()
        }
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if set(result) != {str(cell["cell_id"]) for cell in cells}:
        raise typer.BadParameter(
            "estimate artifact must price every immutable cell exactly once"
        )
    return result


def _operation_key(metadata: dict[str, Any], suffix: str) -> str:
    return f"{metadata['campaign']}-generation-{suffix}"


def _campaign_snapshot(
    campaign_dir: Path, metadata: Mapping[str, Any]
) -> HumanEvalSnapshot:
    """Load the content-bound snapshot shipped inside the campaign."""
    snapshot_metadata = metadata.get("snapshot")
    if not isinstance(snapshot_metadata, Mapping):
        raise ValueError("campaign snapshot metadata is missing")
    locator = snapshot_metadata.get("path")
    if not isinstance(locator, str) or not locator:
        raise ValueError("campaign snapshot path must be a relative locator")
    relative_path = Path(locator)
    if relative_path.is_absolute():
        raise ValueError("campaign snapshot path must be a relative locator")
    campaign_root = campaign_dir.resolve()
    snapshot_path = (campaign_root / relative_path).resolve()
    if not snapshot_path.is_relative_to(campaign_root):
        raise ValueError("campaign snapshot path must remain inside campaign")

    split = _load_json(campaign_dir / "split-full.json")
    dataset = split.get("dataset")
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("snapshot_path") != locator
    ):
        raise ValueError(
            "campaign snapshot locator does not match locked split"
        )
    expected_identity = DatasetSnapshotIdentityPayload.model_validate(
        {
            "sha256": snapshot_metadata.get("sha256"),
            "header": snapshot_metadata.get("header"),
        }
    )
    snapshot = load_humaneval_snapshot(
        dataset_name=expected_identity.header.dataset_id,
        dataset_split=str(dataset.get("split", "")),
        snapshot_path=snapshot_path,
        expected_identity=expected_identity,
    )
    if snapshot_metadata.get("task_count") != len(snapshot.rows):
        raise ValueError("campaign snapshot task count does not match bytes")
    return snapshot


def _specs_for_cells(
    campaign_dir: Path,
    metadata: Mapping[str, Any],
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    snapshot = _campaign_snapshot(campaign_dir, metadata)
    specs = iter_experiment_specs_from_file(
        campaign_dir / "requested-full-matrix-specification.json",
        configs_root=campaign_dir,
        snapshot=snapshot,
    )
    expected = {
        (
            str(c["task_id"]),
            int(c["repetition_seed"]),
            str(c["model"]),
            c["compression_target"],
        ): str(c["cell_id"])
        for c in cells
    }
    selected: dict[str, Any] = {}
    for spec in specs:
        target = spec.dimensions.values.get("compression_target")
        key = (
            spec.task_id,
            spec.repetition_seed,
            spec.provider_axis.model,
            target,
        )
        cell_id = expected.get(key)
        if cell_id is not None:
            cell = next(cell for cell in cells if cell["cell_id"] == cell_id)
            expected_prompt = hashlib.sha256(
                str(spec.task.inputs.values.get("prompt", "")).encode()
            ).hexdigest()
            actual = {
                "execution_contract": "whetstone_live_sweep_v1",
                "task_definition_sha256": cell.get("task_definition_sha256"),
                "task_prompt_sha256": cell.get("task_prompt_sha256"),
                "provider_kind": spec.provider_axis.provider_kind.value,
                "endpoint_kind": spec.provider_axis.endpoint_kind.value,
                "model": spec.provider_axis.model,
                "provider_parameters": spec.provider_axis.parameters,
                "budget_key": cell.get("budget_key"),
                "compression_target": target,
                "repetition_seed": spec.repetition_seed,
                "prediction_id": spec.prediction_id,
                "spec_sha256": _spec_digest(spec),
                "platform_item_id": item_id(
                    operation_key=_cell_operation_key(cell),
                    item_key=spec.prediction_id,
                ),
            }
            locked = {
                "execution_contract": cell.get("execution_contract"),
                "task_definition_sha256": cell.get("task_definition_sha256"),
                "task_prompt_sha256": expected_prompt,
                "provider_kind": cell.get("provider_kind"),
                "endpoint_kind": cell.get("endpoint_kind"),
                "model": cell.get("model"),
                "provider_parameters": cell.get("provider_parameters"),
                "budget_key": cell.get("budget_key"),
                "compression_target": cell.get("compression_target"),
                "repetition_seed": cell.get("repetition_seed"),
                "prediction_id": cell.get("prediction_id"),
                "spec_sha256": cell.get("spec_sha256"),
                "platform_item_id": cell.get("platform_item_id"),
            }
            if locked != actual:
                raise ValueError(
                    "locked spec contract does not match generated spec for "
                    f"{cell_id}"
                )
            selected[cell_id] = spec
    if len(selected) != len(cells):
        raise ValueError(
            "manifest/spec mapping is incomplete; refusing submission"
        )
    return selected


def _emit(
    command: str,
    *,
    cells: list[dict[str, Any]],
    manifest_hash: str,
    execute: bool,
    ledger: SweepLedger | None = None,
) -> None:
    typer.echo(
        json.dumps(
            {
                "command": command,
                "dry_run": not execute,
                "dispatch": execute,
                "cell_count": len(cells),
                "generation_ceiling_usd": _decimal_text(
                    GENERATION_CEILING_USD
                ),
                "manifest_sha256": manifest_hash,
                "ledger": ledger.summary() if ledger else {},
            },
            sort_keys=True,
        )
    )


def _submit(
    campaign_dir: Path,
    metadata: dict[str, Any],
    cells: list[dict[str, Any]],
    ledger: SweepLedger,
    *,
    suffix: str,
) -> None:
    specs = _specs_for_cells(campaign_dir, metadata, cells)
    engine = create_engine(resolve_application_database_url())
    try:
        for cell in cells:
            cell_id = str(cell["cell_id"])
            spec = specs[cell_id]
            operation_key = _cell_operation_key(cell)
            prediction_ids = {cell_id: spec.prediction_id}
            # This commit is deliberately before the external call. Replaying
            # this deterministic single-cell operation is idempotent.
            ledger.submission_intent(
                [cell],
                operation_key=operation_key,
                prediction_ids=prediction_ids,
            )
            submit_prediction_specs(
                engine,
                operation_key=operation_key,
                experiment_name=metadata["campaign"],
                specs=[spec],
                metadata={
                    "manifest_sha256": ledger.manifest_hash,
                    "operator": "whetstone-live-sweep",
                },
            )
            ledger.submitted(
                [cell],
                operation_key=operation_key,
                prediction_ids=prediction_ids,
            )
    finally:
        engine.dispose()


@APP.command()
def plan(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
) -> None:
    """Validate immutable matrix without touching Platform or providers."""
    _metadata, cells, manifest_hash = validate_campaign(campaign_dir)
    _emit("plan", cells=cells, manifest_hash=manifest_hash, execute=False)


@APP.command("submit-canary")
def submit_canary(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    execute: Annotated[bool, typer.Option("--execute")] = False,
    ledger_path: Annotated[Path | None, typer.Option("--ledger")] = None,
    estimates_path: Annotated[Path | None, typer.Option("--estimates")] = None,
) -> None:
    """Reserve and submit exactly the stable 12-cell canary when confirmed."""
    metadata, cells, manifest_hash = validate_campaign(campaign_dir)
    selected = [
        cell
        for cell in cells
        if cell["task_id"] == "HumanEval/0" and cell["repetition_seed"] == 0
    ]
    if len(selected) != CANARY_CELLS:
        raise typer.BadParameter("canary must contain exactly 12 cells")
    if not execute:
        _emit(
            "submit-canary",
            cells=selected,
            manifest_hash=manifest_hash,
            execute=False,
        )
        return
    if ledger_path is None or estimates_path is None:
        raise typer.BadParameter(
            "--execute requires absolute --ledger and --estimates paths"
        )
    estimates = _estimates(estimates_path, cells)
    if _load_json(estimates_path).get("manifest_sha256") != manifest_hash:
        raise typer.BadParameter(
            "estimate artifact is for a different manifest"
        )
    ledger = SweepLedger(ledger_path, manifest_hash=manifest_hash)
    try:
        reserved = ledger.reserve(selected, estimates)
        if reserved:
            _submit(campaign_dir, metadata, reserved, ledger, suffix="canary")
        _emit(
            "submit-canary",
            cells=reserved,
            manifest_hash=manifest_hash,
            execute=True,
            ledger=ledger,
        )
    finally:
        ledger.close()


@APP.command("submit-remaining")
def submit_remaining(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    execute: Annotated[bool, typer.Option("--execute")] = False,
    ledger_path: Annotated[Path | None, typer.Option("--ledger")] = None,
    estimates_path: Annotated[Path | None, typer.Option("--estimates")] = None,
    page_size: Annotated[int, typer.Option(min=1, max=500)] = 100,
) -> None:
    """Submit cells not already reserved, submitted, or successful."""
    metadata, cells, manifest_hash = validate_campaign(campaign_dir)
    if not execute:
        _emit(
            "submit-remaining",
            cells=cells[CANARY_CELLS:],
            manifest_hash=manifest_hash,
            execute=False,
        )
        return
    if ledger_path is None or estimates_path is None:
        raise typer.BadParameter(
            "--execute requires absolute --ledger and --estimates paths"
        )
    estimates = _estimates(estimates_path, cells)
    ledger = SweepLedger(ledger_path, manifest_hash=manifest_hash)
    try:
        submitted: list[dict[str, Any]] = []
        pending = ledger.pending_submission(cells)
        for start in range(0, len(pending), page_size):
            page = pending[start : start + page_size]
            _submit(
                campaign_dir,
                metadata,
                page,
                ledger,
                suffix=f"replay-{start // page_size:04d}",
            )
            submitted.extend(page)
            engine = create_engine(resolve_application_database_url())
            try:
                require_known_actual_costs(
                    reconcile_ledger(ledger, engine=engine),
                    cell_ids={str(cell["cell_id"]) for cell in page},
                )
            finally:
                engine.dispose()
        existing_ids = {str(row["cell_id"]) for row in ledger.rows()}
        if existing_ids:
            engine = create_engine(resolve_application_database_url())
            try:
                require_known_actual_costs(
                    reconcile_ledger(ledger, engine=engine),
                    cell_ids=existing_ids,
                )
            finally:
                engine.dispose()
        remaining = ledger.selected_remaining(cells)
        for start in range(0, len(remaining), page_size):
            reserved = ledger.reserve(
                remaining[start : start + page_size], estimates
            )
            if reserved:
                _submit(
                    campaign_dir,
                    metadata,
                    reserved,
                    ledger,
                    suffix=f"remaining-{start // page_size:04d}",
                )
                submitted.extend(reserved)
                engine = create_engine(resolve_application_database_url())
                try:
                    require_known_actual_costs(
                        reconcile_ledger(ledger, engine=engine),
                        cell_ids={str(cell["cell_id"]) for cell in reserved},
                    )
                finally:
                    engine.dispose()
        _emit(
            "submit-remaining",
            cells=submitted,
            manifest_hash=manifest_hash,
            execute=True,
            ledger=ledger,
        )
    finally:
        ledger.close()


@APP.command("submit-retry")
def submit_retry(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    execute: Annotated[bool, typer.Option("--execute")] = False,
    ledger_path: Annotated[Path | None, typer.Option("--ledger")] = None,
    owner_approved: Annotated[bool, typer.Option("--owner-approved")] = False,
) -> None:
    """Request at most two typed next Attempts after explicit approval."""
    _metadata, _cells, manifest_hash = validate_campaign(campaign_dir)
    if ledger_path is None:
        if execute:
            raise typer.BadParameter("--execute requires an absolute --ledger")
        typer.echo(
            json.dumps(
                {
                    "command": "submit-retry",
                    "dry_run": True,
                    "dispatch": False,
                    "cell_count": 0,
                    "reason": "no ledger supplied",
                },
                sort_keys=True,
            )
        )
        return
    if execute and not owner_approved:
        raise typer.BadParameter(
            "--execute requires explicit --owner-approved retry authority"
        )
    ledger = SweepLedger(ledger_path, manifest_hash=manifest_hash)
    engine = create_engine(resolve_application_database_url())
    try:
        facts = reconcile_ledger(ledger, engine=engine)
        selected = [
            fact
            for fact in facts
            if fact.status in {"typed_failure", "incomplete"}
            and fact.retry_count < MAX_RETRIES_PER_CELL
            and fact.platform_attempt is not None
        ]
        if execute:
            for fact in selected:
                if fact.platform_attempt is None:
                    continue
                if not ledger.claim_retry(fact):
                    continue
                row = next(
                    row
                    for row in ledger.rows()
                    if row["cell_id"] == fact.cell_id
                )
                request_key = sha256_json_digest(
                    {
                        "manifest_hash": manifest_hash,
                        "cell_id": fact.cell_id,
                        "source_attempt": fact.platform_attempt,
                        "classification": fact.error_classification,
                    }
                )
                result = request_next_attempt(
                    NextAttemptRequest(
                        item_id=str(row["platform_item_id"]),
                        source_attempt=fact.platform_attempt,
                        request_key=request_key,
                        reason=NextAttemptReason.DOMAIN_OUTCOME,
                        eligibility=EligibilityReference(
                            kind="whetstone_live_sweep",
                            record_id=fact.cell_id,
                            digest=request_key,
                        ),
                        requested_by="whetstone-live-sweep-owner-approved",
                        max_attempts=MAX_RETRIES_PER_CELL + 1,
                    ),
                    engine=engine,
                    resolver=target_registry(),
                )
                ledger.retried(
                    cell_id=fact.cell_id,
                    source_attempt=fact.platform_attempt,
                    created_attempt=result.created_attempt,
                )
        _emit(
            "submit-retry",
            cells=[{"cell_id": fact.cell_id} for fact in selected],
            manifest_hash=manifest_hash,
            execute=execute,
            ledger=ledger,
        )
    finally:
        engine.dispose()
        ledger.close()


@APP.command()
def status(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    ledger_path: Annotated[Path | None, typer.Option("--ledger")] = None,
) -> None:
    _metadata, cells, manifest_hash = validate_campaign(campaign_dir)
    if ledger_path is None:
        _emit(
            "status", cells=cells, manifest_hash=manifest_hash, execute=False
        )
        return
    ledger = SweepLedger(ledger_path, manifest_hash=manifest_hash)
    engine = create_engine(resolve_application_database_url())
    try:
        reconcile_ledger(ledger, engine=engine)
        _emit(
            "status",
            cells=cells,
            manifest_hash=manifest_hash,
            execute=False,
            ledger=ledger,
        )
    finally:
        engine.dispose()
        ledger.close()


def main() -> None:
    APP()


if __name__ == "__main__":
    main()
