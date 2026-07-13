"""Fail-closed operator commands for the immutable HumanEval live sweep.

The ledger is deliberately local to an operator run: it contains identities,
lifecycle facts, and observed provider costs, never prompts, provider headers,
or credentials.  ``--execute``
is the only path that can call Platform submission; provider work is performed
later by the existing worker.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from dbos import DBOSClient
from dr_platform import (
    AttemptRecord,
    EligibilityReference,
    NextAttemptReason,
    NextAttemptRequest,
    PlatformSchema,
    list_attempts,
    request_next_attempt,
)
from dr_platform.enqueue_runtime import (
    enqueue_pending_page,
    enqueue_replacement_page,
    recover_call_started_page,
)
from dr_platform.items import item_id
from dr_platform.reconciliation_runtime import DbosLifecycleReader, reconcile
from dr_platform.status import AttemptExecutionState
from dr_providers import FailureClass
from dr_serialize import sha256_json_digest
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine

from whetstone.db import io as db_io
from whetstone.db import schema as whetstone_schema
from whetstone.lm.boundary import OUTPUT_FIELD_TEXT
from whetstone.platform.dataset_snapshot import (
    HumanEvalSnapshot,
    load_humaneval_snapshot,
)
from whetstone.platform.enqueue_runtime import (
    InProcessDbosApi,
    platform_enqueue_runtime,
)
from whetstone.platform.runtime import resolve_application_database_url
from whetstone.platform.spec_builder import (
    iter_experiment_specs_from_file,
    load_model_config_fragment,
)
from whetstone.platform.submission import (
    scoring_target_for_generation_run,
    select_populated_scoring_generation_runs,
    submit_prediction_specs,
    submit_scoring_targets,
)
from whetstone.platform.targets import (
    ScoringTargetSpec,
    target_registry,
)
from whetstone.records import (
    DatasetSnapshotIdentityPayload,
    GenerationRunStatus,
)

APP = typer.Typer(no_args_is_help=True)
PLATFORM_SCHEMA = PlatformSchema(prefix="whetstone")
EXPECTED_CELLS = 5_904
CANARY_CELLS = 12
MAX_RETRIES_PER_CELL = 2
GENERATION_SHARD_SIZE = 100
GENERATION_SHARDS_FILE = "generation-manifest-shards.json"
GENERATION_LOCK_POINTER_FILE = "generation-lock.json"
GENERATION_LOCKS_DIRECTORY = "generation-locks"
# Truncated digest length from the pinned dr-providers response contract
# (dr_providers.response.RESPONSE_ID_HASH_LENGTH, not publicly re-exported).
_RESPONSE_ID_HASH_LENGTH = 16
_RESPONSE_STATUS_VALUES = frozenset(
    {
        "cancelled",
        "completed",
        "failed",
        "in_progress",
        "incomplete",
        "queued",
        "unknown",
    }
)
_INCOMPLETE_REASON_VALUES = frozenset(
    {"content_filter", "max_output_tokens", "unknown"}
)
_OUTPUT_ITEM_TYPE_VALUES = frozenset(
    {"function_call", "message", "reasoning", "unknown"}
)
_CONTENT_PART_TYPE_VALUES = frozenset(
    {"output_text", "refusal", "unknown"}
)

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
    RESPONSE_REFUSAL = "response_refusal"
    RESPONSE_INCOMPLETE_NO_TEXT = "response_incomplete_no_text"
    RESPONSE_FAILED = "response_failed"
    RESPONSE_NO_TEXT = "response_no_text"
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
    response_status: str | None
    incomplete_reason: str | None
    output_item_types: dict[str, int] | None
    content_part_types: dict[str, int] | None
    output_text_len: int | None
    refusal_len: int | None
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


@dataclass(frozen=True)
class FrozenScoringIntent:
    """The one immutable scoring cut journaled for a campaign manifest."""

    operation_key: str
    generation_cut_digest: str
    selection_digest: str
    snapshot_sha256: str
    scoring_profile_id: str
    scoring_profile_version: str
    parser_profile_id: str
    parser_version: str
    targets: tuple[ScoringTargetSpec, ...]
    status: str


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
            schema=PLATFORM_SCHEMA,
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
            select(whetstone_schema.score_attempts.c.status)
            .where(
                whetstone_schema.score_attempts.c.prediction_id
                == prediction_id,
                whetstone_schema.score_attempts.c.generation_run_id
                == generation_run_id,
            )
            .order_by(
                whetstone_schema.score_attempts.c.platform_attempt.desc()
            )
            .limit(1)
        ).scalar_one_or_none()
        if score is not None:
            return f"score_{score}"
        harness = connection.execute(
            select(whetstone_schema.score_harness_failures.c.score_attempt_id)
            .where(
                whetstone_schema.score_harness_failures.c.prediction_id
                == prediction_id,
                whetstone_schema.score_harness_failures.c.generation_run_id
                == generation_run_id,
            )
            .order_by(
                whetstone_schema.score_harness_failures.c.platform_attempt.desc()
            )
            .limit(1)
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
    persisted_response_metadata = (
        node.get("response_metadata") if node is not None else None
    )
    response_metadata = _provider_response_metadata(
        persisted_response_metadata
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
    response_diagnostics = _response_diagnostics(
        response_metadata=response_metadata,
        failure=failure,
    )
    response_status = _allowlisted_enum(
        response_diagnostics,
        "response_status",
        allowed_values=_RESPONSE_STATUS_VALUES,
    )
    if response_status is None:
        response_status = _allowlisted_enum(
            response_metadata,
            "status",
            allowed_values=_RESPONSE_STATUS_VALUES,
        )
    if response_status is None:
        response_status = _allowlisted_enum(
            _provider_failure_metadata(failure),
            "response_status",
            allowed_values=_RESPONSE_STATUS_VALUES,
        )
    diagnostic_response_id_hash = _allowlisted_response_id_hash(
        response_diagnostics
    )
    disposition, failure_class, failure_code = _adapter_facts(
        node_status=_allowlisted_string(node, "status"),
        failure=failure,
        output_field_present=output_field_present,
        output_nonblank=output_nonblank,
    )
    diagnostics = LiveSweepDiagnostics(
        response_id_hash=(
            _hash_response_id(response_id)
            if response_id is not None
            else diagnostic_response_id_hash
        ),
        returned_model=model,
        finish_reason=finish_reason,
        response_status=response_status,
        incomplete_reason=_allowlisted_enum(
            response_diagnostics,
            "incomplete_reason",
            allowed_values=_INCOMPLETE_REASON_VALUES,
        ),
        output_item_types=_allowlisted_counts(
            response_diagnostics,
            "output_item_types",
            allowed_categories=_OUTPUT_ITEM_TYPE_VALUES,
        ),
        content_part_types=_allowlisted_counts(
            response_diagnostics,
            "content_part_types",
            allowed_categories=_CONTENT_PART_TYPE_VALUES,
        ),
        output_text_len=_allowlisted_nonnegative_int(
            response_diagnostics, "output_text_len"
        ),
        refusal_len=_allowlisted_nonnegative_int(
            response_diagnostics, "refusal_len"
        ),
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


def _provider_response_metadata(payload: object) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    nested = payload.get("response_metadata")
    if len(payload) == 1 and isinstance(nested, Mapping):
        return cast(Mapping[str, Any], nested)
    return cast(Mapping[str, Any], payload)


def _response_diagnostics(
    *,
    response_metadata: object,
    failure: object,
) -> Mapping[str, Any] | None:
    if isinstance(response_metadata, Mapping):
        diagnostics = response_metadata.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            return cast(Mapping[str, Any], diagnostics)
    provider_metadata = _provider_failure_metadata(failure)
    if provider_metadata is None:
        return None
    diagnostics = provider_metadata.get("diagnostics")
    return (
        cast(Mapping[str, Any], diagnostics)
        if isinstance(diagnostics, Mapping)
        else None
    )


def _provider_failure_metadata(failure: object) -> Mapping[str, Any] | None:
    if not isinstance(failure, Mapping):
        return None
    failure_metadata = failure.get("metadata")
    if not isinstance(failure_metadata, Mapping):
        return None
    provider_failure = failure_metadata.get("provider_failure")
    if not isinstance(provider_failure, Mapping):
        return None
    provider_metadata = provider_failure.get("metadata")
    return (
        cast(Mapping[str, Any], provider_metadata)
        if isinstance(provider_metadata, Mapping)
        else None
    )


def _allowlisted_string(
    payload: Mapping[str, Any] | None, key: str
) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _hash_response_id(response_id: str) -> str:
    """Digest a response id exactly as pinned dr-providers failures do."""
    return hashlib.sha256(response_id.encode()).hexdigest()[
        :_RESPONSE_ID_HASH_LENGTH
    ]


def _allowlisted_response_id_hash(
    payload: Mapping[str, Any] | None,
) -> str | None:
    value = _allowlisted_string(payload, "response_id_hash")
    if value is None or len(value) != _RESPONSE_ID_HASH_LENGTH:
        return None
    return (
        value
        if all(character in "0123456789abcdef" for character in value)
        else None
    )


def _allowlisted_enum(
    payload: Mapping[str, Any] | None,
    key: str,
    *,
    allowed_values: frozenset[str],
) -> str | None:
    value = _allowlisted_string(payload, key)
    return value if value in allowed_values else None


def _allowlisted_counts(
    payload: Mapping[str, Any] | None,
    key: str,
    *,
    allowed_categories: frozenset[str],
) -> dict[str, int] | None:
    if payload is None:
        return None
    value = payload.get(key)
    if not isinstance(value, Mapping):
        return None
    counts = {
        category: count
        for category, count in value.items()
        if category in allowed_categories
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
    }
    return dict(sorted(counts.items()))


def _allowlisted_nonnegative_int(
    payload: Mapping[str, Any] | None, key: str
) -> int | None:
    if payload is None:
        return None
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


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
    provider_failure = (
        metadata.get("provider_failure")
        if isinstance(metadata, Mapping)
        else None
    )
    provider_failure_code = (
        provider_failure.get("code")
        if isinstance(provider_failure, Mapping)
        else None
    )
    if provider_failure_code == "response_parse_error":
        return (
            AdapterDisposition.PARSE_FAILURE,
            failure_class,
            TypedFailureCode.PROVIDER_RESPONSE_PARSE,
        )
    response_outcome_codes = {
        "response_refusal": TypedFailureCode.RESPONSE_REFUSAL,
        "response_incomplete_no_text": (
            TypedFailureCode.RESPONSE_INCOMPLETE_NO_TEXT
        ),
        "response_failed": TypedFailureCode.RESPONSE_FAILED,
        "response_no_text": TypedFailureCode.RESPONSE_NO_TEXT,
    }
    response_outcome = response_outcome_codes.get(provider_failure_code)
    if response_outcome is not None:
        return (
            AdapterDisposition.PROVIDER_FAILURE,
            failure_class,
            response_outcome,
        )
    if "ProviderFailureError" in type_names or (
        isinstance(metadata, Mapping) and "provider_failure" in metadata
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


def require_terminal_lifecycle(
    facts: list[CellReconciliation], *, cell_ids: set[str]
) -> None:
    """Fail closed until every selected cell has a durable terminal state."""
    stable_statuses = {"succeeded", "typed_failure", "incomplete"}
    selected = {
        fact.cell_id: fact for fact in facts if fact.cell_id in cell_ids
    }
    missing = cell_ids - set(selected)
    unknown = sorted(
        cell_id
        for cell_id, fact in selected.items()
        if fact.status not in stable_statuses
    )
    if missing or unknown:
        blocked = sorted(missing) + unknown
        preview = ", ".join(blocked[:5])
        raise RuntimeError(
            "terminal lifecycle is not established for the bounded page; "
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


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.relock-{os.getpid()}.tmp")
    with temporary.open("wb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _locked_generation_paths(campaign_dir: Path) -> tuple[Path, Path, Path]:
    pointer_path = campaign_dir / GENERATION_LOCK_POINTER_FILE
    if not pointer_path.is_file():
        return (
            campaign_dir / "manifest.jsonl",
            campaign_dir / GENERATION_SHARDS_FILE,
            campaign_dir / "manifest-index.json",
        )
    pointer = _load_json(pointer_path)
    if pointer.get("schema_version") != 1:
        raise typer.BadParameter("unsupported Generation lock pointer")
    generation = pointer.get("generation")
    if not isinstance(generation, str) or not generation:
        raise typer.BadParameter("Generation lock pointer has no generation")
    generation_root = (
        campaign_dir / GENERATION_LOCKS_DIRECTORY / generation
    ).resolve()
    locks_root = (campaign_dir / GENERATION_LOCKS_DIRECTORY).resolve()
    if not generation_root.is_relative_to(locks_root):
        raise typer.BadParameter("Generation lock pointer escapes campaign")
    paths = (
        generation_root / "manifest.jsonl",
        generation_root / GENERATION_SHARDS_FILE,
        generation_root / "manifest-index.json",
    )
    expected_hashes = pointer.get("sha256")
    if not isinstance(expected_hashes, Mapping):
        raise typer.BadParameter("Generation lock pointer has no hash set")
    for path in paths:
        expected_hash = expected_hashes.get(path.name)
        if not path.is_file() or expected_hash != _sha256(path):
            raise typer.BadParameter(
                "Generation lock pointer references an incomplete artifact set"
            )
    return paths


def _write_generation_file(path: Path, payload: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != payload:
            raise typer.BadParameter(
                "content-addressed Generation artifact is inconsistent"
            )
        return
    _atomic_write(path, payload)


def _generation_shards(
    *, campaign: str, cells: list[dict[str, Any]], canary_ids: list[str]
) -> list[dict[str, Any]]:
    by_id = {str(cell["cell_id"]): cell for cell in cells}
    if len(by_id) != len(cells) or len(set(canary_ids)) != CANARY_CELLS:
        raise ValueError("campaign cell and canary identities must be unique")
    if any(cell_id not in by_id for cell_id in canary_ids):
        raise ValueError("canary contains a cell outside the manifest")
    remaining_ids = [
        str(cell["cell_id"])
        for cell in cells
        if str(cell["cell_id"]) not in set(canary_ids)
    ]
    member_groups = [canary_ids] + [
        remaining_ids[start : start + GENERATION_SHARD_SIZE]
        for start in range(0, len(remaining_ids), GENERATION_SHARD_SIZE)
    ]
    shards: list[dict[str, Any]] = []
    for ordinal, member_ids in enumerate(member_groups, start=1):
        members_digest = sha256_json_digest(cast(Any, member_ids))
        operation_key = (
            f"{campaign}-generation-shard-{ordinal:03d}-{members_digest[:16]}"
        )
        shards.append(
            {
                "ordinal": ordinal,
                "kind": "canary" if ordinal == 1 else "remaining",
                "operation_key": operation_key,
                "members_digest": members_digest,
                "cell_ids": member_ids,
            }
        )
    return shards


@APP.command("relock-generation-shards")
def relock_generation_shards(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
) -> None:
    """Publish bounded shards as one crash-consistent artifact generation."""
    metadata = _load_json(campaign_dir / "campaign-metadata.json")
    manifest_path, _, index_path = _locked_generation_paths(
        campaign_dir
    )
    cells = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index = _load_json(index_path)
    current_manifest_hash = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    if index.get("manifest_sha256") != current_manifest_hash:
        raise typer.BadParameter(
            "refusing to re-lock a manifest that does not match its index"
        )
    canary = _load_json(campaign_dir / "canary-12-cells.json")
    canary_ids = [str(value) for value in canary.get("cell_ids", [])]
    expected_canary_ids = [
        str(cell["cell_id"])
        for cell in cells
        if cell.get("task_id") == "HumanEval/0"
        and cell.get("repetition_seed") == 0
    ]
    if canary_ids != expected_canary_ids:
        raise typer.BadParameter(
            "canary selection does not match the locked deterministic rule"
        )
    shards = _generation_shards(
        campaign=str(metadata["campaign"]),
        cells=cells,
        canary_ids=canary_ids,
    )
    by_cell = {
        cell_id: shard for shard in shards for cell_id in shard["cell_ids"]
    }
    for cell in cells:
        shard = by_cell[str(cell["cell_id"])]
        cell["generation_shard_ordinal"] = shard["ordinal"]
        cell["operation_key"] = shard["operation_key"]
        cell["platform_item_id"] = item_id(
            operation_key=shard["operation_key"],
            item_key=str(cell["prediction_id"]),
        )
    manifest_bytes = b"".join(
        (json.dumps(cell, sort_keys=True) + "\n").encode() for cell in cells
    )
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    shard_artifact = {
        "schema_version": 1,
        "campaign": metadata["campaign"],
        "manifest_sha256": manifest_hash,
        "shard_size": GENERATION_SHARD_SIZE,
        "shards": shards,
    }
    shard_bytes = _canonical_json_bytes(shard_artifact)
    index["manifest_sha256"] = manifest_hash
    index["generation_shards_sha256"] = hashlib.sha256(shard_bytes).hexdigest()
    index["generation_shard_count"] = len(shards)
    index_bytes = _canonical_json_bytes(index)
    generation = sha256_json_digest(
        {
            "schema_version": 1,
            "manifest_sha256": manifest_hash,
            "generation_shards_sha256": index["generation_shards_sha256"],
            "manifest_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        }
    )
    generation_dir = campaign_dir / GENERATION_LOCKS_DIRECTORY / generation
    generation_dir.mkdir(parents=True, exist_ok=True)
    _fsync_directory(generation_dir.parent)
    artifacts = {
        "manifest.jsonl": manifest_bytes,
        GENERATION_SHARDS_FILE: shard_bytes,
        "manifest-index.json": index_bytes,
    }
    for name, payload in artifacts.items():
        _write_generation_file(generation_dir / name, payload)
    _fsync_directory(generation_dir)
    pointer = {
        "schema_version": 1,
        "generation": generation,
        "sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in artifacts.items()
        },
    }
    _atomic_write(
        campaign_dir / GENERATION_LOCK_POINTER_FILE,
        _canonical_json_bytes(pointer),
    )
    typer.echo(
        json.dumps(
            {
                "generation": generation,
                "manifest_sha256": manifest_hash,
                "generation_shards_sha256": index["generation_shards_sha256"],
                "generation_shard_count": len(shards),
            },
            sort_keys=True,
        )
    )


class SweepLedger:
    """Run-scoped, WAL-backed journal of durable submission lifecycle."""

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
        self._create_tables()
        self._migrate_legacy_cost_columns()
        self._ensure_column("attempt_ids_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("score_status", "TEXT")
        self._ensure_column("diagnostics_json", "TEXT")
        self._ensure_scoring_targets_column()
        self._ensure_single_scoring_operation()

    def _create_tables(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sweep_cells (
              manifest_hash TEXT NOT NULL, cell_id TEXT NOT NULL,
              actual_cost TEXT, operation_key TEXT, prediction_id TEXT,
              platform_item_id TEXT, platform_attempt INTEGER,
              attempt_ids_json TEXT NOT NULL DEFAULT '[]',
              status TEXT NOT NULL, retry_count INTEGER NOT NULL DEFAULT 0,
              retry_of_attempt INTEGER, error_classification TEXT,
              provider_tokens_json TEXT, score_status TEXT,
              diagnostics_json TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL, PRIMARY KEY (manifest_hash, cell_id)
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sweep_events (
              id INTEGER PRIMARY KEY, manifest_hash TEXT NOT NULL,
              cell_id TEXT NOT NULL, event TEXT NOT NULL,
              detail_json TEXT NOT NULL, created_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sweep_scoring_operations (
              manifest_hash TEXT NOT NULL, operation_key TEXT NOT NULL,
              generation_cut_digest TEXT NOT NULL,
              selection_digest TEXT NOT NULL, snapshot_sha256 TEXT NOT NULL,
              scoring_profile_id TEXT NOT NULL,
              scoring_profile_version TEXT NOT NULL,
              parser_profile_id TEXT NOT NULL, parser_version TEXT NOT NULL,
              item_ids_json TEXT NOT NULL, targets_json TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              PRIMARY KEY (manifest_hash, operation_key)
            )
            """
        )

    def _migrate_legacy_cost_columns(self) -> None:
        """Remove prediction-era columns while retaining durable cell facts."""
        columns = {
            str(row[1])
            for row in self.connection.execute(
                "PRAGMA table_info(sweep_cells)"
            )
        }
        if not {"estimated_cost", "reserved_cost"} & columns:
            return
        retained = (
            "manifest_hash",
            "cell_id",
            "actual_cost",
            "operation_key",
            "prediction_id",
            "platform_item_id",
            "platform_attempt",
            "attempt_ids_json",
            "status",
            "retry_count",
            "retry_of_attempt",
            "error_classification",
            "provider_tokens_json",
            "score_status",
            "diagnostics_json",
            "created_at",
            "updated_at",
        )
        copied = [name for name in retained if name in columns]
        with self._transaction() as connection:
            connection.execute(
                "ALTER TABLE sweep_cells RENAME TO sweep_cells_legacy"
            )
            self._create_tables()
            names = ",".join(copied)
            connection.execute(
                f"INSERT INTO sweep_cells({names}) "
                f"SELECT {names} FROM sweep_cells_legacy"
            )
            connection.execute(
                "UPDATE sweep_cells SET status='pending' "
                "WHERE status='reserved'"
            )
            connection.execute("DROP TABLE sweep_cells_legacy")

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

    def _ensure_scoring_targets_column(self) -> None:
        columns = {
            str(row[1])
            for row in self.connection.execute(
                "PRAGMA table_info(sweep_scoring_operations)"
            )
        }
        if "targets_json" not in columns:
            self.connection.execute(
                "ALTER TABLE sweep_scoring_operations ADD COLUMN "
                "targets_json TEXT"
            )

    def _ensure_single_scoring_operation(self) -> None:
        duplicate = self.connection.execute(
            "SELECT manifest_hash FROM sweep_scoring_operations "
            "GROUP BY manifest_hash HAVING COUNT(*) > 1 LIMIT 1"
        ).fetchone()
        if duplicate is not None:
            raise ValueError(
                "ledger has multiple scoring intents and cannot be replayed "
                "safely"
            )
        self.connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "uq_sweep_scoring_manifest ON "
            "sweep_scoring_operations(manifest_hash)"
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

    def record_intent(
        self, cells: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        with self._transaction() as connection:
            for cell in cells:
                cell_id = str(cell["cell_id"])
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
                timestamp = _now()
                connection.execute(
                    "INSERT INTO sweep_cells("
                    "manifest_hash,cell_id,status,created_at,updated_at) "
                    "VALUES(?,?,?,?,?)",
                    (
                        self.manifest_hash,
                        cell_id,
                        "pending",
                        timestamp,
                        timestamp,
                    ),
                )
                connection.execute(
                    "INSERT INTO sweep_events("
                    "manifest_hash,cell_id,event,detail_json,created_at) "
                    "VALUES(?,?,?,?,?)",
                    (
                        self.manifest_hash,
                        cell_id,
                        "intent_recorded",
                        "{}",
                        timestamp,
                    ),
                )
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
                    "AND status IN ('pending','submitting')",
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
                "AND status IN ('pending','submitting')",
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
        by_status = {
            str(status): {
                "count": count,
                "actual_usd": float(
                    sum(
                        (
                            _stored_money(row[0], field="stored actual cost")
                            for row in self.connection.execute(
                                "SELECT actual_cost FROM sweep_cells "
                                "WHERE manifest_hash=? AND status=?",
                                (self.manifest_hash, status),
                            ).fetchall()
                            if row[0] is not None
                        ),
                        Decimal(),
                    )
                ),
                "unknown_cost_count": self.connection.execute(
                    "SELECT COUNT(*) FROM sweep_cells "
                    "WHERE manifest_hash=? AND status=? "
                    "AND actual_cost IS NULL",
                    (self.manifest_hash, status),
                ).fetchone()[0],
            }
            for status, count in rows
        }
        observed_rows = self.connection.execute(
            "SELECT actual_cost FROM sweep_cells WHERE manifest_hash=?",
            (self.manifest_hash,),
        ).fetchall()
        return by_status | {
            "observed_cost": {
                "actual_usd": float(
                    sum(
                        (
                            _stored_money(row[0], field="stored actual cost")
                            for row in observed_rows
                            if row[0] is not None
                        ),
                        Decimal(),
                    )
                ),
                "unknown_cost_count": sum(
                    1 for row in observed_rows if row[0] is None
                ),
            },
            "manifest_mismatch": {
                "count": self.connection.execute(
                    "SELECT COUNT(*) FROM sweep_cells WHERE manifest_hash!=?",
                    (self.manifest_hash,),
                ).fetchone()[0],
                "actual_usd": 0,
                "unknown_cost_count": 0,
            },
        }

    def rows(self) -> list[sqlite3.Row]:
        self.connection.row_factory = sqlite3.Row
        return self.connection.execute(
            "SELECT * FROM sweep_cells WHERE manifest_hash=? ORDER BY cell_id",
            (self.manifest_hash,),
        ).fetchall()

    def scoring_intent(
        self,
        *,
        operation_key: str,
        generation_cut_digest: str,
        selection_digest: str,
        snapshot_sha256: str,
        scoring_profile_id: str,
        scoring_profile_version: str,
        parser_profile_id: str,
        parser_version: str,
        item_ids: list[str],
        targets: tuple[ScoringTargetSpec, ...],
    ) -> FrozenScoringIntent:
        """Atomically journal the complete scoring identity before enqueue."""
        now = _now()
        targets_json = json.dumps(
            [target.model_dump(mode="json") for target in targets],
            sort_keys=True,
        )
        values = (
            self.manifest_hash,
            operation_key,
            generation_cut_digest,
            selection_digest,
            snapshot_sha256,
            scoring_profile_id,
            scoring_profile_version,
            parser_profile_id,
            parser_version,
            json.dumps(item_ids),
            targets_json,
            "submitting",
            now,
            now,
        )
        with self._transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO sweep_scoring_operations("
                "manifest_hash,operation_key,generation_cut_digest,"
                "selection_digest,snapshot_sha256,scoring_profile_id,"
                "scoring_profile_version,parser_profile_id,parser_version,"
                "item_ids_json,targets_json,status,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                values,
            )
            existing = connection.execute(
                "SELECT operation_key,generation_cut_digest,selection_digest,"
                "snapshot_sha256,scoring_profile_id,scoring_profile_version,"
                "parser_profile_id,parser_version,item_ids_json,targets_json "
                "FROM sweep_scoring_operations WHERE manifest_hash=?",
                (self.manifest_hash,),
            ).fetchone()
            if existing is None or tuple(existing) != values[1:11]:
                raise ValueError(
                    "scoring operation identity changed on replay"
                )
        frozen = self.scoring_operation()
        if frozen is None:
            raise RuntimeError("scoring intent was not persisted")
        return frozen

    def scoring_operation(self) -> FrozenScoringIntent | None:
        row = self.connection.execute(
            "SELECT operation_key,generation_cut_digest,selection_digest,"
            "snapshot_sha256,scoring_profile_id,scoring_profile_version,"
            "parser_profile_id,parser_version,item_ids_json,targets_json,"
            "status FROM "
            "sweep_scoring_operations WHERE manifest_hash=?",
            (self.manifest_hash,),
        ).fetchone()
        if row is None:
            return None
        if row[9] is None:
            raise ValueError(
                "legacy scoring intent has no replayable frozen targets"
            )
        targets = tuple(
            ScoringTargetSpec.model_validate(value)
            for value in json.loads(str(row[9]))
        )
        identity_is_valid = (
            sha256_json_digest(
                [target.model_dump(mode="json") for target in targets]
            )
            == row[2]
            and json.loads(str(row[8]))
            == [
                item_id(operation_key=str(row[0]), item_key=target.item_key)
                for target in targets
            ]
            and {target.dataset_snapshot.sha256 for target in targets}
            == {str(row[3])}
            and {target.scoring_profile_id for target in targets}
            == {str(row[4])}
            and {target.scoring_profile_version for target in targets}
            == {str(row[5])}
            and {target.parser_profile_id for target in targets}
            == {str(row[6])}
            and {target.parser_version for target in targets}
            == {str(row[7])}
        )
        if not targets or not identity_is_valid:
            raise ValueError("frozen scoring intent is corrupt")
        return FrozenScoringIntent(
            operation_key=str(row[0]),
            generation_cut_digest=str(row[1]),
            selection_digest=str(row[2]),
            snapshot_sha256=str(row[3]),
            scoring_profile_id=str(row[4]),
            scoring_profile_version=str(row[5]),
            parser_profile_id=str(row[6]),
            parser_version=str(row[7]),
            targets=targets,
            status=str(row[10]),
        )

    def scoring_submitted(self, *, operation_key: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE sweep_scoring_operations SET status='submitted',"
                "updated_at=? WHERE manifest_hash=? AND operation_key=?",
                (_now(), self.manifest_hash, operation_key),
            )

    def reconciliation(self, facts: list[CellReconciliation]) -> None:
        with self._transaction() as connection:
            for fact in facts:
                connection.execute(
                    "UPDATE sweep_cells SET status=?,actual_cost=?,"
                    "error_classification=?,provider_tokens_json=?,"
                    "score_status=?,diagnostics_json=?,"
                    "updated_at=? "
                    "WHERE manifest_hash=? AND cell_id=?",
                    (
                        fact.status,
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
        ):
            return False
        with self._transaction() as connection:
            result = connection.execute(
                "UPDATE sweep_cells SET status='retrying',retry_of_attempt=?,"
                "updated_at=? WHERE manifest_hash=? AND cell_id=? "
                "AND retry_count<? AND status IN "
                "('typed_failure','incomplete','retrying')",
                (
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
    manifest_path, shards_path, index_path = _locked_generation_paths(
        campaign_dir
    )
    manifest_hash = _sha256(manifest_path)
    index = _load_json(index_path)
    if index.get("manifest_sha256") != manifest_hash:
        raise typer.BadParameter("manifest hash does not match locked index")
    cells = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not shards_path.is_file():
        raise typer.BadParameter(
            "campaign has no locked Generation Manifest shards; run "
            "relock-generation-shards"
        )
    shards_bytes = shards_path.read_bytes()
    if (
        index.get("generation_shards_sha256")
        != hashlib.sha256(shards_bytes).hexdigest()
    ):
        raise typer.BadParameter("Generation shard hash does not match index")
    shard_artifact = json.loads(shards_bytes)
    if shard_artifact.get("manifest_sha256") != manifest_hash:
        raise typer.BadParameter("Generation shards bind a different manifest")
    shards = shard_artifact.get("shards")
    if not isinstance(shards, list) or len(shards) != index.get(
        "generation_shard_count"
    ):
        raise typer.BadParameter("Generation shard inventory is incomplete")
    canary = _load_json(campaign_dir / "canary-12-cells.json")
    expected_shards = _generation_shards(
        campaign=str(metadata["campaign"]),
        cells=cells,
        canary_ids=[str(value) for value in canary.get("cell_ids", [])],
    )
    if shards != expected_shards:
        raise typer.BadParameter("Generation shards are not deterministic")
    shard_members = [
        (str(cell_id), shard)
        for shard in shards
        for cell_id in shard.get("cell_ids", [])
    ]
    if len(shard_members) != len(cells) or len(
        {cell_id for cell_id, _ in shard_members}
    ) != len(cells):
        raise typer.BadParameter("each campaign cell must belong to one shard")
    shard_by_cell = dict(shard_members)
    for cell in cells:
        cell_id = str(cell["cell_id"])
        shard = shard_by_cell.get(cell_id)
        if (
            shard is None
            or cell.get("operation_key") != shard.get("operation_key")
            or cell.get("generation_shard_ordinal") != shard.get("ordinal")
            or cell.get("platform_item_id")
            != item_id(
                operation_key=str(shard.get("operation_key")),
                item_key=str(cell.get("prediction_id")),
            )
        ):
            raise typer.BadParameter(
                "cell identity does not match locked shard"
            )
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
    dispatch: bool | None = None,
) -> None:
    typer.echo(
        json.dumps(
            {
                "command": command,
                "dry_run": not execute,
                "dispatch": execute if dispatch is None else dispatch,
                "cell_count": len(cells),
                "manifest_sha256": manifest_hash,
                "ledger": ledger.summary() if ledger else {},
            },
            sort_keys=True,
        )
    )


def _bounded_submission_groups(
    cells: list[dict[str, Any]], *, fallback_size: int
) -> list[list[dict[str, Any]]]:
    if cells and all("operation_key" in cell for cell in cells):
        groups: dict[str, list[dict[str, Any]]] = {}
        for cell in cells:
            groups.setdefault(_cell_operation_key(cell), []).append(cell)
        return list(groups.values())
    return [
        cells[start : start + fallback_size]
        for start in range(0, len(cells), fallback_size)
    ]


def _non_canary_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordinals = {cell.get("generation_shard_ordinal") for cell in cells}
    if not ordinals or any(
        isinstance(ordinal, bool) or not isinstance(ordinal, int)
        for ordinal in ordinals
    ):
        raise ValueError("campaign cells have no locked shard ordinal")
    if 1 not in ordinals:
        raise ValueError("campaign cells have no locked canary shard")
    return [cell for cell in cells if cell["generation_shard_ordinal"] != 1]


def _expected_generation_operations(
    cells: list[dict[str, Any]],
) -> list[str]:
    by_ordinal: dict[int, str] = {}
    for cell in cells:
        ordinal = cell.get("generation_shard_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise RuntimeError("campaign member has no locked shard ordinal")
        operation_key = _cell_operation_key(cell)
        existing = by_ordinal.setdefault(ordinal, operation_key)
        if existing != operation_key:
            raise RuntimeError("locked shard ordinal has multiple operations")
    return [by_ordinal[ordinal] for ordinal in sorted(by_ordinal)]


def _complete_generation_members(
    cells: list[dict[str, Any]], ledger: SweepLedger
) -> list[dict[str, Any]]:
    """Return the exact terminal member cut or reject before scoring IDs."""
    rows = {str(row["cell_id"]): row for row in ledger.rows()}
    expected_ids = {str(cell["cell_id"]) for cell in cells}
    if set(rows) != expected_ids:
        raise RuntimeError(
            "scoring requires every locked Generation member in the ledger"
        )
    locked_predictions = {
        str(cell["cell_id"]): str(cell["prediction_id"]) for cell in cells
    }
    blocked: list[str] = []
    members: list[dict[str, Any]] = []
    for cell in cells:
        cell_id = str(cell["cell_id"])
        row = rows[cell_id]
        if (
            row["status"] not in {"succeeded", "typed_failure"}
            or row["prediction_id"] != locked_predictions[cell_id]
            or row["operation_key"] != _cell_operation_key(cell)
            or row["platform_item_id"] != cell.get("platform_item_id")
            or row["platform_attempt"] is None
        ):
            blocked.append(cell_id)
        members.append(
            {
                "cell_id": cell_id,
                "prediction_id": row["prediction_id"],
                "operation_key": row["operation_key"],
                "platform_item_id": row["platform_item_id"],
                "platform_attempt": row["platform_attempt"],
                "status": row["status"],
                "error_classification": row["error_classification"],
            }
        )
    if blocked:
        raise RuntimeError(
            "scoring requires every locked Generation member to be terminal "
            f"and fully reconciled ({', '.join(blocked[:5])})"
        )
    return members


def _submit(
    campaign_dir: Path,
    metadata: dict[str, Any],
    cells: list[dict[str, Any]],
    ledger: SweepLedger,
) -> None:
    specs = _specs_for_cells(campaign_dir, metadata, cells)
    engine = create_engine(resolve_application_database_url())
    try:
        by_operation: dict[str, list[dict[str, Any]]] = {}
        for cell in cells:
            by_operation.setdefault(_cell_operation_key(cell), []).append(cell)
        with platform_enqueue_runtime() as runtime:
            for operation_key, shard_cells in by_operation.items():
                prediction_ids = {
                    str(cell["cell_id"]): specs[
                        str(cell["cell_id"])
                    ].prediction_id
                    for cell in shard_cells
                }
                # One FULL-synchronous SQLite transaction fixes every member
                # ID before Platform can register or enqueue the immutable
                # shard.
                ledger.submission_intent(
                    shard_cells,
                    operation_key=operation_key,
                    prediction_ids=prediction_ids,
                )
                submit_prediction_specs(
                    engine,
                    operation_key=operation_key,
                    experiment_name=metadata["campaign"],
                    specs=[
                        specs[str(cell["cell_id"])] for cell in shard_cells
                    ],
                    metadata={
                        "manifest_sha256": ledger.manifest_hash,
                        "generation_shard_ordinal": shard_cells[0][
                            "generation_shard_ordinal"
                        ],
                        "operator": "whetstone-live-sweep",
                    },
                    queue_lookup=runtime.queue_lookup,
                    enqueue_adapter=runtime.enqueue_adapter,
                    workflow_observer=runtime.workflow_observer,
                )
                ledger.submitted(
                    shard_cells,
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
) -> None:
    """Record and submit exactly the stable 12-cell canary when confirmed."""
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
    if ledger_path is None:
        raise typer.BadParameter("--execute requires an absolute --ledger")
    ledger = SweepLedger(ledger_path, manifest_hash=manifest_hash)
    try:
        ledger.record_intent(selected)
        pending = ledger.pending_submission(selected)
        recorded = selected if pending else []
        if recorded:
            _submit(campaign_dir, metadata, recorded, ledger)
        _emit(
            "submit-canary",
            cells=recorded,
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
    page_size: Annotated[int, typer.Option(min=1, max=500)] = 100,
) -> None:
    """Submit cells not already recorded in the durable ledger."""
    metadata, cells, manifest_hash = validate_campaign(campaign_dir)
    non_canary_cells = _non_canary_cells(cells)
    if not execute:
        _emit(
            "submit-remaining",
            cells=non_canary_cells,
            manifest_hash=manifest_hash,
            execute=False,
        )
        return
    if ledger_path is None:
        raise typer.BadParameter("--execute requires an absolute --ledger")
    ledger = SweepLedger(ledger_path, manifest_hash=manifest_hash)
    try:
        submitted: list[dict[str, Any]] = []
        pending = ledger.pending_submission(cells)
        pending_operations = {
            _cell_operation_key(cell)
            for cell in pending
            if "operation_key" in cell
        }
        replay_cells = [
            cell
            for cell in cells
            if (not pending_operations and cell in pending)
            or (
                "operation_key" in cell
                and _cell_operation_key(cell) in pending_operations
            )
        ]
        for page in _bounded_submission_groups(
            replay_cells, fallback_size=page_size
        ):
            _submit(
                campaign_dir,
                metadata,
                page,
                ledger,
            )
            submitted.extend(page)
            engine = create_engine(resolve_application_database_url())
            try:
                require_terminal_lifecycle(
                    reconcile_ledger(ledger, engine=engine),
                    cell_ids={str(cell["cell_id"]) for cell in page},
                )
            finally:
                engine.dispose()
        existing_ids = {str(row["cell_id"]) for row in ledger.rows()}
        if existing_ids:
            engine = create_engine(resolve_application_database_url())
            try:
                require_terminal_lifecycle(
                    reconcile_ledger(ledger, engine=engine),
                    cell_ids=existing_ids,
                )
            finally:
                engine.dispose()
        # A pending canary operation can only originate from a durable intent
        # written by submit-canary. New intent here is always non-canary.
        remaining = ledger.selected_remaining(non_canary_cells)
        for page in _bounded_submission_groups(
            remaining, fallback_size=page_size
        ):
            recorded = ledger.record_intent(page)
            if recorded:
                _submit(
                    campaign_dir,
                    metadata,
                    recorded,
                    ledger,
                )
                submitted.extend(recorded)
                engine = create_engine(resolve_application_database_url())
                try:
                    require_terminal_lifecycle(
                        reconcile_ledger(ledger, engine=engine),
                        cell_ids={str(cell["cell_id"]) for cell in recorded},
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


@APP.command("submit-scoring")
def submit_scoring(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    execute: Annotated[bool, typer.Option("--execute")] = False,
    ledger_path: Annotated[Path | None, typer.Option("--ledger")] = None,
    scoring_profile_id: Annotated[str, typer.Option()] = "humaneval",
    scoring_profile_version: Annotated[str, typer.Option()] = "v1",
) -> None:
    """Freeze once, then replay only the complete campaign scoring cut."""
    metadata, cells, manifest_hash = validate_campaign(campaign_dir)
    if not execute:
        _emit(
            "submit-scoring",
            cells=cells,
            manifest_hash=manifest_hash,
            execute=False,
        )
        return
    if ledger_path is None:
        raise typer.BadParameter("--execute requires an absolute --ledger")
    ledger = SweepLedger(ledger_path, manifest_hash=manifest_hash)
    engine = create_engine(resolve_application_database_url())
    try:
        frozen = ledger.scoring_operation()
        if frozen is None:
            reconcile_ledger(ledger, engine=engine)
            generation_members = _complete_generation_members(cells, ledger)
            with engine.begin() as connection:
                runs = select_populated_scoring_generation_runs(
                    connection, experiment_name=str(metadata["campaign"])
                )
                specs = {
                    str(row["prediction_id"]): (
                        db_io.prediction_spec_record_from_row(dict(row))
                    )
                    for row in connection.execute(
                        select(whetstone_schema.prediction_specs).where(
                            whetstone_schema.prediction_specs.c.experiment_name
                            == metadata["campaign"]
                        )
                    ).mappings()
                }
                generation_relationships = [
                    dict(row)
                    for row in connection.execute(
                        select(
                            whetstone_schema.experiment_operation_manifests.c.operation_key,
                            whetstone_schema.experiment_operation_manifests.c.manifest_digest,
                            whetstone_schema.experiment_operation_manifests.c.accepted_generation_ordinal,
                        )
                        .where(
                            whetstone_schema.experiment_operation_manifests.c.experiment_name
                            == metadata["campaign"],
                            whetstone_schema.experiment_operation_manifests.c.workflow_role
                            == "generation",
                        )
                        .order_by(
                            whetstone_schema.experiment_operation_manifests.c.accepted_generation_ordinal
                        )
                    ).mappings()
                ]
            relationship_keys = [
                str(row["operation_key"]) for row in generation_relationships
            ]
            if relationship_keys != _expected_generation_operations(cells):
                raise RuntimeError(
                    "scoring requires every locked Generation shard "
                    "relationship"
                )
            succeeded_predictions = {
                str(member["prediction_id"])
                for member in generation_members
                if member["status"] == "succeeded"
            }
            runs_by_prediction = {run.prediction_id: run for run in runs}
            if set(runs_by_prediction) != succeeded_predictions:
                raise RuntimeError(
                    "scoreable Generation runs do not match the reconciled cut"
                )
            if not succeeded_predictions.issubset(specs):
                raise RuntimeError(
                    "reconciled Generation cut has missing prediction specs"
                )
            targets = tuple(
                scoring_target_for_generation_run(
                    spec=specs[prediction_id],
                    generation_run=runs_by_prediction[prediction_id],
                    scoring_profile_id=scoring_profile_id,
                    scoring_profile_version=scoring_profile_version,
                )
                for prediction_id in sorted(succeeded_predictions)
            )
            if not targets:
                raise RuntimeError(
                    "complete Generation cut has no populated run to score"
                )
            snapshots = {target.dataset_snapshot.sha256 for target in targets}
            if snapshots != {str(metadata["snapshot"]["sha256"])}:
                raise RuntimeError(
                    "scoring targets do not bind the campaign snapshot"
                )
            generation_run_ids = {
                run.prediction_id: run.generation_run_id for run in runs
            }
            generation_cut_digest = sha256_json_digest(
                cast(
                    Any,
                    {
                        "manifest_sha256": manifest_hash,
                        "generation_relationships": generation_relationships,
                        "generation_members": [
                            member
                            | {
                                "generation_run_id": generation_run_ids.get(
                                    str(member["prediction_id"])
                                )
                            }
                            for member in generation_members
                        ],
                    },
                )
            )
            selection_digest = sha256_json_digest(
                [target.model_dump(mode="json") for target in targets]
            )
            first = targets[0]
            scoring_identity_digest = sha256_json_digest(
                {
                    "manifest_sha256": manifest_hash,
                    "generation_cut_digest": generation_cut_digest,
                    "scoring_profile_id": first.scoring_profile_id,
                    "scoring_profile_version": first.scoring_profile_version,
                    "parser_profile_id": first.parser_profile_id,
                    "parser_version": first.parser_version,
                    "snapshot_sha256": first.dataset_snapshot.sha256,
                }
            )
            operation_key = (
                f"{metadata['campaign']}-scoring-"
                f"{scoring_identity_digest[:24]}"
            )
            frozen = ledger.scoring_intent(
                operation_key=operation_key,
                generation_cut_digest=generation_cut_digest,
                selection_digest=selection_digest,
                snapshot_sha256=first.dataset_snapshot.sha256,
                scoring_profile_id=first.scoring_profile_id,
                scoring_profile_version=first.scoring_profile_version,
                parser_profile_id=first.parser_profile_id,
                parser_version=first.parser_version,
                item_ids=[
                    item_id(
                        operation_key=operation_key, item_key=target.item_key
                    )
                    for target in targets
                ],
                targets=targets,
            )
        if (
            frozen.scoring_profile_id != scoring_profile_id
            or frozen.scoring_profile_version != scoring_profile_version
        ):
            raise RuntimeError(
                "requested scoring profile differs from frozen campaign intent"
            )
        if frozen.status == "submitted":
            _emit(
                "submit-scoring",
                cells=[],
                manifest_hash=manifest_hash,
                execute=True,
                ledger=ledger,
                dispatch=False,
            )
            return
        if frozen.status != "submitting":
            raise RuntimeError("frozen scoring intent has an invalid status")
        with platform_enqueue_runtime() as runtime:
            submit_scoring_targets(
                engine,
                operation_key=frozen.operation_key,
                experiment_name=str(metadata["campaign"]),
                targets=frozen.targets,
                source_generation_operation_key=(
                    f"sharded:{frozen.generation_cut_digest}"
                ),
                metadata={
                    "manifest_sha256": manifest_hash,
                    "generation_cut_digest": frozen.generation_cut_digest,
                    "snapshot_sha256": frozen.snapshot_sha256,
                    "operator": "whetstone-live-sweep",
                },
                queue_lookup=runtime.queue_lookup,
                enqueue_adapter=runtime.enqueue_adapter,
                workflow_observer=runtime.workflow_observer,
            )
        ledger.scoring_submitted(operation_key=frozen.operation_key)
        target_predictions = {
            target.prediction_id for target in frozen.targets
        }
        _emit(
            "submit-scoring",
            cells=[
                cell
                for cell in cells
                if str(cell["prediction_id"])
                in target_predictions
            ],
            manifest_hash=manifest_hash,
            execute=True,
            ledger=ledger,
        )
    finally:
        engine.dispose()
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
                    schema=PLATFORM_SCHEMA,
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


@APP.command("reconcile")
def reconcile_kernel(
    execute: Annotated[bool, typer.Option("--execute")] = False,
    max_cycles: Annotated[int, typer.Option(min=1, max=100)] = 20,
) -> None:
    """Apply authoritative DBOS lifecycle to kernel Attempt state."""
    engine = create_engine(resolve_application_database_url())
    try:
        if not execute:
            attempts = PLATFORM_SCHEMA.item_attempts
            with engine.connect() as connection:
                rows = connection.execute(
                    select(attempts.c.execution_state, func.count())
                    .group_by(attempts.c.execution_state)
                    .order_by(attempts.c.execution_state)
                ).all()
            typer.echo(
                json.dumps(
                    {
                        "command": "reconcile",
                        "execute": False,
                        "attempts": {str(s): int(n) for s, n in rows},
                    },
                    sort_keys=True,
                )
            )
            return
        cycles: list[dict[str, Any]] = []
        with platform_enqueue_runtime() as runtime:
            reader = DbosLifecycleReader(
                cast("DBOSClient", InProcessDbosApi())
            )
            for _ in range(max_cycles):
                result = reconcile(
                    engine,
                    resolver=target_registry(),
                    queue_lookup=runtime.queue_lookup,
                    schema=PLATFORM_SCHEMA,
                    reader=reader,
                    recovery_observer=runtime.workflow_observer,
                    enqueue_adapter=runtime.enqueue_adapter,
                )
                cycles.append(result.model_dump(mode="json"))
                mutated = (
                    result.recovered_call_started_count
                    + result.changed_count
                    + result.enqueue_reset_count
                    + result.execution_retry_count
                    + result.replacement_enqueue_count
                    + result.pending_enqueue_count
                )
                if mutated == 0:
                    break
        typer.echo(
            json.dumps(
                {"command": "reconcile", "execute": True, "cycles": cycles},
                sort_keys=True,
            )
        )
    finally:
        engine.dispose()


@APP.command("recover-enqueues")
def recover_enqueues(
    execute: Annotated[bool, typer.Option("--execute")] = False,
    operation_key: Annotated[str | None, typer.Option()] = None,
    max_rounds: Annotated[int, typer.Option(min=1, max=100)] = 20,
) -> None:
    """Drive pending, expired, and interrupted enqueue Claims to outcomes."""
    engine = create_engine(resolve_application_database_url())
    try:
        if not execute:
            claims = PLATFORM_SCHEMA.enqueue_claims
            with engine.connect() as connection:
                rows = connection.execute(
                    select(claims.c.disposition, func.count())
                    .group_by(claims.c.disposition)
                    .order_by(claims.c.disposition)
                ).all()
            typer.echo(
                json.dumps(
                    {
                        "command": "recover-enqueues",
                        "execute": False,
                        "claims": {str(d): int(n) for d, n in rows},
                    },
                    sort_keys=True,
                )
            )
            return
        rounds: list[dict[str, int]] = []
        with platform_enqueue_runtime() as runtime:
            for _ in range(max_rounds):
                counts = {
                    "pending": len(
                        enqueue_pending_page(
                            engine,
                            resolver=target_registry(),
                            queue_lookup=runtime.queue_lookup,
                            schema=PLATFORM_SCHEMA,
                            adapter=runtime.enqueue_adapter,
                            operation_key=operation_key,
                        ).items
                    ),
                    "replacement": len(
                        enqueue_replacement_page(
                            engine,
                            resolver=target_registry(),
                            queue_lookup=runtime.queue_lookup,
                            schema=PLATFORM_SCHEMA,
                            adapter=runtime.enqueue_adapter,
                            operation_key=operation_key,
                        ).items
                    ),
                    "call_started": len(
                        recover_call_started_page(
                            engine,
                            resolver=target_registry(),
                            queue_lookup=runtime.queue_lookup,
                            schema=PLATFORM_SCHEMA,
                            adapter=runtime.enqueue_adapter,
                            observer=runtime.workflow_observer,
                            operation_key=operation_key,
                        ).items
                    ),
                }
                rounds.append(counts)
                if sum(counts.values()) == 0:
                    break
        typer.echo(
            json.dumps(
                {
                    "command": "recover-enqueues",
                    "execute": True,
                    "rounds": rounds,
                },
                sort_keys=True,
            )
        )
    finally:
        engine.dispose()


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
