#!/usr/bin/env python3
"""Inspect a persisted ED1 baseline behavior matrix without driving calls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from whetstone.envs.ed1 import (
    ed1_blend_config_from_metadata,
    ed1_task_model_from_metadata,
)
from whetstone.evaluation.compression import zstd_compressed_utf8_byte_length
from whetstone.evaluation.metrics.blended import (
    blended_reward,
    compression_score,
)
from whetstone.evaluation.preview.anchor import BaselinePreviewTranscript
from whetstone.execution.call_support import call_telemetry
from whetstone.execution.partials import PartialLog
from whetstone.provider.attempt import ProviderCallResult

try:
    from dr_code.humaneval import (
        extract_humaneval_code as _public_extract_humaneval_code,
    )
except ImportError:  # The overlay is useful inspection evidence, not required.
    _extract_humaneval_code: Callable[..., Any] | None = None
else:
    _extract_humaneval_code = _public_extract_humaneval_code


_COMPLETE_STATES = frozenset({"completed", "complete", "success", "succeeded"})
_COMPLETE_PROCESS_STATES = frozenset(
    {"treatment_completed", "treatment_skipped"}
)
_CACHE_SCHEMA = "whetstone.execution.prompt_cache_entry/v3"
_CACHE_KEY = re.compile(r"[0-9a-f]{64}")
_LOGICAL_CALL_ID = re.compile(
    r"^(?P<candidate>.+):(?P<task>[^:]+)#(?P<repeat>\d+):(?P<leg>enc|dec)$"
)
_ENCODER_CODE = re.compile(r"\n```python\n(?P<code>.*)\n```\Z", re.DOTALL)

REPORT_FILES = (
    "report.json",
    "treatments.jsonl",
    "rows.jsonl",
    "rows.csv",
    "provider_calls.jsonl",
    "partials.jsonl",
    "paired_deltas.jsonl",
    "execution_records.jsonl",
)


class InspectionError(RuntimeError):
    """The persisted matrix is internally inconsistent or invalid."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InspectionError(
            f"cannot read valid JSON at {path}: {exc}"
        ) from exc


def _relative_file(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise InspectionError(f"{field} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise InspectionError(f"{field} must be relative: {value}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise InspectionError(
            f"{field} escapes the output directory: {value}"
        ) from exc
    return resolved


def _process_states(output_dir: Path) -> tuple[dict[str, str], bool]:
    path = output_dir / "process-log.jsonl"
    if not path.exists():
        return {}, False
    states: dict[str, str] = {}
    run_completed = False
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InspectionError(
                f"invalid process log line {number}: {exc}"
            ) from exc
        if not isinstance(item, dict) or not isinstance(
            item.get("state"), str
        ):
            raise InspectionError(
                f"invalid process log record on line {number}"
            )
        state = item["state"]
        treatment_id = item.get("treatment_id")
        if isinstance(treatment_id, str):
            states[treatment_id] = state
        if state == "run_completed":
            run_completed = True
    return states, run_completed


def _manifest_treatments(
    output_dir: Path, manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], bool]:
    raw_treatments = manifest.get("treatments")
    if not isinstance(raw_treatments, list) or not raw_treatments:
        raise InspectionError(
            "run-manifest.json requires a non-empty treatments list"
        )
    process_states, run_completed = _process_states(output_dir)
    plans: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_treatments):
        if not isinstance(raw, dict):
            raise InspectionError(
                f"manifest treatment {index} must be an object"
            )
        treatment_id = raw.get("treatment_id", raw.get("id", raw.get("name")))
        if not isinstance(treatment_id, str) or not treatment_id:
            raise InspectionError(
                f"manifest treatment {index} has no treatment_id"
            )
        if treatment_id in seen_ids:
            raise InspectionError(f"duplicate treatment_id {treatment_id!r}")
        seen_ids.add(treatment_id)
        relative_result = raw.get("result_relative_path")
        if relative_result is None:
            directory = raw.get(
                "directory", raw.get("relative_directory", treatment_id)
            )
            if not isinstance(directory, str):
                raise InspectionError(
                    f"treatment {treatment_id!r} has no directory"
                )
            relative_result = str(Path(directory) / "result.json")
        result_path = _relative_file(
            output_dir,
            relative_result,
            field=f"treatments[{index}].result_relative_path",
        )
        state = raw.get("status")
        if not isinstance(state, str):
            state = process_states.get(treatment_id)
        completed = (
            state in _COMPLETE_STATES or state in _COMPLETE_PROCESS_STATES
        )
        if run_completed and state is None:
            completed = True
        plans.append(
            {
                "treatment_id": treatment_id,
                "plan": raw,
                "result_path": result_path,
                "result_relative_path": str(
                    result_path.relative_to(output_dir.resolve())
                ),
                "state": state or "unavailable",
                "completed": completed,
            }
        )
    return plans, run_completed


def _validate_plan(
    transcript: BaselinePreviewTranscript,
    treatment: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    plan = treatment["plan"]
    treatment_id = treatment["treatment_id"]
    checks: tuple[tuple[str, bool, object, object], ...] = (
        (
            "budget_ratio",
            "budget_ratio" in plan,
            plan.get("budget_ratio"),
            transcript.budget_ratio,
        ),
        (
            "concurrency",
            "concurrency" in plan or "concurrency" in manifest,
            plan.get("concurrency", manifest.get("concurrency")),
            transcript.concurrency,
        ),
        (
            "task_ids",
            "task_ids" in manifest,
            manifest.get("task_ids"),
            list(transcript.task_ids),
        ),
    )
    for field, declared, expected, actual in checks:
        equivalent = expected == actual
        if isinstance(expected, (list, tuple)) and isinstance(
            actual, (list, tuple)
        ):
            equivalent = tuple(expected) == tuple(actual)
        if declared and not equivalent:
            raise InspectionError(
                f"treatment {treatment_id!r} result changed planned {field}"
            )
    task_model = plan.get("task_model")
    if task_model is not None and task_model != ed1_task_model_from_metadata(
        transcript.metadata
    ).model_dump(mode="json"):
        raise InspectionError(
            f"treatment {treatment_id!r} result changed planned task_model"
        )
    repeats = plan.get("repeats", manifest.get("repeats"))
    for arm in (transcript.baseline, transcript.ceiling):
        if repeats is not None and arm.evidence.repeat_count != repeats:
            raise InspectionError(
                f"treatment {treatment_id!r} result changed planned repeats"
            )
    actual_rows = sum(
        arm.evidence.row_accounting.planned
        for arm in (transcript.baseline, transcript.ceiling)
    )
    planned_rows = plan.get("planned_rows")
    if planned_rows is not None and actual_rows != planned_rows:
        raise InspectionError(
            f"treatment {treatment_id!r} has {actual_rows} rows, "
            f"expected {planned_rows}"
        )
    planned_calls = plan.get("planned_provider_calls")
    if planned_calls is not None and planned_calls != actual_rows * 2:
        raise InspectionError(
            f"treatment {treatment_id!r} has inconsistent "
            "planned_provider_calls"
        )


def _trace_values(trace: Any) -> dict[str, str | None]:
    values: dict[str, str | None] = {
        "encoder_prompt": None,
        "encoder_generation": None,
        "decoder_prompt": None,
        "decoder_generation": None,
    }
    for step in trace.executed_component_trace.executed_component_steps:
        prompt = step.inputs.to_json().get("prompt")
        generation = step.outputs.to_json().get("generation")
        if step.component_id == "encode":
            values["encoder_prompt"] = (
                prompt if isinstance(prompt, str) else None
            )
            values["encoder_generation"] = (
                generation if isinstance(generation, str) else None
            )
        elif step.component_id == "decode":
            values["decoder_prompt"] = (
                prompt if isinstance(prompt, str) else None
            )
            values["decoder_generation"] = (
                generation if isinstance(generation, str) else None
            )
    return values


def _input_code(encoder_prompt: str | None) -> str | None:
    if encoder_prompt is None:
        return None
    match = _ENCODER_CODE.search(encoder_prompt)
    return None if match is None else match.group("code")


def _extract_code(decoder_generation: str | None) -> dict[str, object]:
    if decoder_generation is None:
        return {"status": "unavailable", "code": None, "failure_code": None}
    if _extract_humaneval_code is None:
        return {
            "status": "helper_unavailable",
            "code": None,
            "failure_code": None,
        }
    try:
        result = _extract_humaneval_code(decoder_generation)
    except Exception as exc:  # Inspection remains usable across overlay drift.
        return {
            "status": "helper_error",
            "code": None,
            "failure_code": type(exc).__name__,
        }
    return {
        "status": "accepted" if result.succeeded else "rejected",
        "code": result.accepted_code,
        "failure_code": result.failure_code,
    }


def _arm_rows(
    treatment_id: str,
    transcript: BaselinePreviewTranscript,
    arm_name: str,
    arm: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blend_config = ed1_blend_config_from_metadata(transcript.metadata)
    task_model = ed1_task_model_from_metadata(transcript.metadata)
    outputs = {
        (row.task_identity, row.repeat): row for row in arm.outputs.outputs
    }
    traces = {
        (row.task_identity, row.repeat): row
        for row in arm.component_traces.rows
    }
    if set(outputs) != set(traces):
        raise InspectionError(
            f"{treatment_id}/{arm_name} output and trace coordinates differ"
        )
    expected = len(arm.evidence.task_identities) * arm.evidence.repeat_count
    accounting = arm.evidence.row_accounting
    counted = (
        accounting.present
        + accounting.missing
        + accounting.failed
        + accounting.invalid
    )
    if (
        accounting.planned != expected
        or counted != expected
        or len(outputs) != expected
    ):
        raise InspectionError(
            f"{treatment_id}/{arm_name} row accounting does not cover "
            "its exact plan"
        )

    rows: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in outputs:
        output = outputs[key]
        values = _trace_values(traces[key])
        input_code = _input_code(values["encoder_prompt"])
        encoder_generation = values["encoder_generation"]
        compression_ratio: float | None = None
        if (
            output.score is not None
            and input_code is not None
            and encoder_generation is not None
        ):
            denominator = len(input_code.encode("utf-8"))
            if denominator:
                compression_ratio = (
                    zstd_compressed_utf8_byte_length(encoder_generation)
                    / denominator
                )
        compression_value = (
            None
            if compression_ratio is None
            else compression_score(compression_ratio, blend_config)
        )
        row_blended = (
            None
            if output.score is None
            else blended_reward(
                primary_score=float(output.score),
                compression_ratio=compression_ratio,
                config=blend_config,
            )
        )
        derived_over_budget = (
            None
            if output.max_budget is None or encoder_generation is None
            else len(encoder_generation) > output.max_budget
        )
        if (
            output.over_budget is not None
            and output.over_budget != derived_over_budget
        ):
            raise InspectionError(
                f"{treatment_id}/{arm_name}/{key} budget evidence disagrees"
            )
        extracted = _extract_code(values["decoder_generation"])
        item: dict[str, Any] = {
            "treatment_id": treatment_id,
            "model": task_model.model,
            "budget_ratio": transcript.budget_ratio,
            "arm": arm_name,
            "candidate_id": output.candidate_id,
            "task_identity": output.task_identity,
            "repeat": output.repeat,
            "row_state": traces[key].executed_component_trace.row_state.value,
            "score": output.score,
            "failed": output.failed,
            "missing": output.missing,
            "invalid": output.invalid,
            "failure_code": output.failure_code,
            "finish_reason": output.finish_reason,
            "provider_error": None
            if output.provider_error is None
            else output.provider_error.to_json(),
            "max_budget": output.max_budget,
            "encoder_length": None
            if encoder_generation is None
            else len(encoder_generation),
            "over_budget": derived_over_budget,
            "input_code": input_code,
            **values,
            "extracted_decoder_code_status": extracted["status"],
            "extracted_decoder_code": extracted["code"],
            "extraction_failure_code": extracted["failure_code"],
            "correctness_reward": output.score,
            "compression_ratio": compression_ratio,
            "compression_reward": compression_value,
            "row_blended_reward": row_blended,
            "encoder_provider_cache_key": None,
            "decoder_provider_cache_key": None,
        }
        rows.append(item)
        by_task[output.task_identity].append(item)

    per_task: list[dict[str, Any]] = []
    stored = dict(
        zip(
            arm.evidence.task_identities,
            arm.evidence.per_task_values,
            strict=True,
        )
    )
    counts = dict(
        zip(
            arm.evidence.task_identities,
            arm.evidence.per_task_counts,
            strict=True,
        )
    )
    for task_identity in arm.evidence.task_identities:
        task_rows = by_task[task_identity]
        if len(task_rows) != arm.evidence.repeat_count:
            raise InspectionError(
                f"{treatment_id}/{arm_name}/{task_identity} repeat "
                "count differs"
            )
        primary = sum(float(row["score"] or 0.0) for row in task_rows) / len(
            task_rows
        )
        compression_values = [
            float(row["compression_ratio"])
            for row in task_rows
            if row["compression_ratio"] is not None
        ]
        compression = (
            sum(compression_values) / len(compression_values)
            if compression_values
            else None
        )
        blend = blended_reward(
            primary_score=primary,
            compression_ratio=compression,
            config=blend_config,
        )
        if counts[
            task_identity
        ] != arm.evidence.repeat_count or not math.isclose(
            blend, stored[task_identity], rel_tol=0.0, abs_tol=1e-12
        ):
            raise InspectionError(
                f"{treatment_id}/{arm_name}/{task_identity} derived "
                "per-task blend "
                "does not equal stored per_task evidence"
            )
        per_task.append(
            {
                "treatment_id": treatment_id,
                "model": task_model.model,
                "budget_ratio": transcript.budget_ratio,
                "arm": arm_name,
                "task_identity": task_identity,
                "repeat_count": len(task_rows),
                "correctness_reward": primary,
                "compression_ratio": compression,
                "compression_reward": (
                    None
                    if compression is None
                    else compression_score(compression, blend_config)
                ),
                "blended_reward": blend,
                "stored_per_task_value": stored[task_identity],
            }
        )
    return rows, per_task


def _provider_metadata(response_body: object) -> dict[str, object] | None:
    found: dict[str, object] = {}

    def visit(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if not isinstance(key, str):
                    continue
                child = f"{path}.{key}" if path else key
                if "provider" in key.lower() and isinstance(
                    nested, (str, int, float, bool, type(None))
                ):
                    found[child] = nested
                visit(nested, child)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")

    visit(response_body, "")
    return found or None


def _config_identity(request_identity: Mapping[str, Any]) -> object | None:
    for key in (
        "config_identity_hash",
        "provider_call_config",
        "provider_call_config_ref",
        "config",
        "config_ref",
        "config_identity",
    ):
        if key in request_identity:
            return request_identity[key]
    return None


def _provider_calls(
    output_dir: Path,
    treatment_roots: Mapping[str, Path],
    candidate_arms: Mapping[str, str],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for treatment_id, root in treatment_roots.items():
        for path in sorted(root.rglob("*.json")):
            if (
                "prompt-cache" not in path.parts
                and "provider-cache" not in path.parts
            ):
                continue
            raw = _read_json(path)
            if not isinstance(raw, dict) or raw.get("schema") != _CACHE_SCHEMA:
                continue
            key = raw.get("key")
            if (
                not isinstance(key, str)
                or _CACHE_KEY.fullmatch(key) is None
                or path.name != f"{key}.json"
            ):
                raise InspectionError(f"invalid prompt-cache key at {path}")
            scoped_key = (treatment_id, key)
            if scoped_key in seen_keys:
                raise InspectionError(
                    f"duplicate prompt-cache key {key} in {treatment_id}"
                )
            seen_keys.add(scoped_key)
            try:
                result = ProviderCallResult.model_validate(raw.get("result"))
            except ValidationError as exc:
                raise InspectionError(
                    f"invalid ProviderCallResult at {path}: {exc}"
                ) from exc
            if raw.get("request_identity") != result.request_identity:
                raise InspectionError(
                    f"cache/result request identity mismatch at {path}"
                )
            if (
                raw.get("execution_policy_hash")
                != result.execution_policy_hash
            ):
                raise InspectionError(
                    f"cache/result policy identity mismatch at {path}"
                )
            match = _LOGICAL_CALL_ID.fullmatch(result.logical_call_id)
            candidate = match.group("candidate") if match else None
            terminal_evidence = result.attempts[-1].evidence
            response = terminal_evidence.response
            response_body = (
                None if response is None else response.response_body
            )
            telemetry = asdict(call_telemetry(result))
            config_identity = _config_identity(result.request_identity)
            item = {
                "treatment_id": treatment_id,
                "cache_key": key,
                "relative_path": str(path.relative_to(output_dir)),
                "logical_call_id": result.logical_call_id,
                "candidate_id": candidate,
                "arm": candidate_arms.get(candidate or "", "unavailable"),
                "task_identity": match.group("task") if match else None,
                "repeat": int(match.group("repeat")) if match else None,
                "leg": match.group("leg") if match else None,
                "succeeded": result.succeeded,
                "attempt_count": result.attempt_count,
                "requested_config_identity": config_identity,
                "requested_config_identity_availability": (
                    "available"
                    if config_identity is not None
                    else "unavailable"
                ),
                "returned_model": None if response is None else response.model,
                "returned_model_availability": (
                    "available"
                    if response is not None and response.model is not None
                    else "unavailable"
                ),
                "response_body": response_body,
                "raw_response_availability": (
                    "available" if response_body else "unavailable"
                ),
                "upstream_provider_metadata": _provider_metadata(
                    response_body
                ),
                "upstream_provider_availability": (
                    "available"
                    if _provider_metadata(response_body) is not None
                    else "unavailable"
                ),
                "telemetry": telemetry,
                "semantic_failure": (
                    None
                    if result.semantic_failure is None
                    else result.semantic_failure.model_dump(mode="json")
                ),
            }
            calls.append(item)
    return calls


def _partials(
    output_dir: Path, treatment_roots: Mapping[str, Path]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    entry_name = re.compile(r"[0-9a-f]{64}\.json")
    for treatment_id, root in treatment_roots.items():
        candidates: set[Path] = set()
        for name in ("partial-log", "partials"):
            base = root / name
            if base.is_dir():
                if any(
                    entry_name.fullmatch(path.name) for path in base.iterdir()
                ):
                    candidates.add(base)
                candidates.update(
                    path.parent
                    for path in base.rglob("*.json")
                    if entry_name.fullmatch(path.name)
                )
        for directory in sorted(candidates):
            try:
                records = PartialLog(directory).load()
            except (OSError, ValueError) as exc:
                raise InspectionError(
                    f"invalid PartialLog at {directory}: {exc}"
                ) from exc
            for record in records:
                output.append(
                    {
                        "treatment_id": treatment_id,
                        "relative_log": str(directory.relative_to(output_dir)),
                        **record.as_dict(),
                    }
                )
    return output


def _sqlite_inventory(path: Path) -> dict[str, object]:
    tables: dict[str, int] = {}
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            names = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "ORDER BY name"
            ).fetchall()
            for (name,) in names:
                escaped = str(name).replace('"', '""')
                count = connection.execute(
                    f'SELECT COUNT(*) FROM "{escaped}"'
                ).fetchone()
                tables[str(name)] = int(count[0])
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return {"relative_path": str(path), "error": str(exc), "tables": {}}
    return {"relative_path": str(path), "error": None, "tables": tables}


def _execution_inventory(output_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("record.json")):
        if "inspection" in path.relative_to(output_dir).parts:
            continue
        raw = _read_json(path)
        if not isinstance(raw, dict):
            raise InspectionError(
                f"execution record must be an object: {path}"
            )
        raw_result = raw.get("result")
        result: dict[str, Any] = (
            raw_result if isinstance(raw_result, dict) else {}
        )
        raw_outcome = result.get("outcome")
        outcome: dict[str, Any] = (
            raw_outcome if isinstance(raw_outcome, dict) else {}
        )
        raw_measurements = result.get("measurements")
        measurements: dict[str, Any] = (
            raw_measurements if isinstance(raw_measurements, dict) else {}
        )
        records.append(
            {
                "relative_path": str(path.relative_to(output_dir)),
                "state": raw.get("state", "unavailable"),
                "outcome_kind": outcome.get("kind", "unavailable"),
                "exit_code": outcome.get("exit_code"),
                "duration_ns": measurements.get("duration_ns"),
                "job_id": (
                    result.get("execution_id", {}).get("job_id")
                    if isinstance(result.get("execution_id"), dict)
                    else None
                ),
            }
        )
    sqlite_files = [
        path
        for path in sorted(output_dir.rglob("*.sqlite3"))
        if "inspection" not in path.relative_to(output_dir).parts
    ]
    databases = []
    for path in sqlite_files:
        item = _sqlite_inventory(path)
        item["relative_path"] = str(path.relative_to(output_dir))
        item["size_bytes"] = path.stat().st_size
        databases.append(item)
    outcomes = Counter(str(record["outcome_kind"]) for record in records)
    return {
        "record_count": len(records),
        "outcomes": dict(sorted(outcomes.items())),
        "records": records,
        "databases": databases,
    }


def _mean(values: Iterable[object]) -> float | None:
    numeric = [
        float(value) for value in values if isinstance(value, int | float)
    ]
    return sum(numeric) / len(numeric) if numeric else None


def _treatment_summary(
    treatment_id: str,
    transcript: BaselinePreviewTranscript,
    rows: Sequence[Mapping[str, Any]],
    per_task: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    task_model = ed1_task_model_from_metadata(transcript.metadata)
    accounting: dict[str, dict[str, int]] = {}
    for name, arm in (
        ("BASELINE", transcript.baseline),
        ("HUMAN_BEST", transcript.ceiling),
    ):
        accounting[name] = arm.evidence.row_accounting.model_dump(mode="json")
    failures = Counter(
        str(row["failure_code"] or "unspecified")
        for row in rows
        if row["failed"] or row["missing"] or row["invalid"]
    )
    budgeted = [row for row in rows if row["max_budget"] is not None]
    available_budget = [
        row for row in budgeted if row["over_budget"] is not None
    ]
    return {
        "treatment_id": treatment_id,
        "model": task_model.model,
        "budget_ratio": transcript.budget_ratio,
        "row_accounting": accounting,
        "integrity": "validated",
        "budget": {
            "budgeted_rows": len(budgeted),
            "compliance_available": len(available_budget),
            "compliant": sum(
                row["over_budget"] is False for row in available_budget
            ),
            "violations": sum(
                row["over_budget"] is True for row in available_budget
            ),
            "unavailable": len(budgeted) - len(available_budget),
        },
        "failures": dict(sorted(failures.items())),
        "metrics": {
            "correctness_mean_present": _mean(
                row["correctness_reward"] for row in rows
            ),
            "compression_ratio_mean_available": _mean(
                row["compression_ratio"] for row in rows
            ),
            "compression_reward_mean_available": _mean(
                row["compression_reward"] for row in rows
            ),
            "blended_reward_mean_per_task": _mean(
                row["blended_reward"] for row in per_task
            ),
        },
        "paired_delta_ci": asdict(transcript.paired_delta_ci),
        "power": {
            "certified_headroom": transcript.power.certified_headroom,
            "decomposition": asdict(transcript.power.decomposition),
            "recommendation": asdict(transcript.power.recommendation),
        },
    }


def _paired_deltas(
    per_task: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (
            str(row["treatment_id"]),
            str(row["task_identity"]),
            str(row["arm"]),
        ): row
        for row in per_task
    }
    deltas: list[dict[str, Any]] = []
    coordinates = sorted({(key[0], key[1]) for key in indexed})
    for treatment_id, task_identity in coordinates:
        baseline = indexed.get((treatment_id, task_identity, "BASELINE"))
        human = indexed.get((treatment_id, task_identity, "HUMAN_BEST"))
        if baseline is None or human is None:
            raise InspectionError(
                "paired arms are not aligned for "
                f"{treatment_id}/{task_identity}"
            )

        def delta(
            field: str,
            baseline_row: Mapping[str, Any] = baseline,
            human_row: Mapping[str, Any] = human,
        ) -> float | None:
            left, right = baseline_row[field], human_row[field]
            if not isinstance(left, int | float) or not isinstance(
                right, int | float
            ):
                return None
            return float(right) - float(left)

        deltas.append(
            {
                "treatment_id": treatment_id,
                "model": human["model"],
                "budget_ratio": human["budget_ratio"],
                "task_identity": task_identity,
                "direction": "HUMAN_BEST-BASELINE",
                "correctness_delta": delta("correctness_reward"),
                "compression_ratio_delta": delta("compression_ratio"),
                "compression_reward_delta": delta("compression_reward"),
                "blended_reward_delta": delta("blended_reward"),
            }
        )
    return deltas


def _atomic_text(path: Path, body: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    body = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in rows
    )
    _atomic_text(path, body)


def _write_reports(output_dir: Path, report: Mapping[str, Any]) -> Path:
    inspection = output_dir / "inspection"
    inspection.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        inspection / "report.json",
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
    )
    _write_jsonl(inspection / "treatments.jsonl", report["treatments"])
    _write_jsonl(inspection / "rows.jsonl", report["rows"])
    _write_jsonl(inspection / "provider_calls.jsonl", report["provider_calls"])
    _write_jsonl(inspection / "partials.jsonl", report["partials"])
    _write_jsonl(inspection / "paired_deltas.jsonl", report["paired_deltas"])
    _write_jsonl(
        inspection / "execution_records.jsonl",
        report["execution_inventory"]["records"],
    )
    rows = list(report["rows"])
    fields = (
        tuple(rows[0])
        if rows
        else (
            "treatment_id",
            "arm",
            "task_identity",
            "repeat",
        )
    )
    temporary = inspection / ".rows.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(value, sort_keys=True, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for field, value in row.items()
                }
            )
    temporary.replace(inspection / "rows.csv")
    return inspection


def inspect_matrix(output_dir: Path) -> dict[str, Any]:
    """Validate and inspect one matrix, then emit stable machine reports."""
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / "run-manifest.json"
    raw_manifest = _read_json(manifest_path)
    if not isinstance(raw_manifest, dict):
        raise InspectionError("run-manifest.json must contain an object")
    if raw_manifest.get("schema_version") not in (None, 1):
        raise InspectionError("unsupported run-manifest schema_version")
    treatments, run_completed = _manifest_treatments(output_dir, raw_manifest)
    completed = [item for item in treatments if item["completed"]]
    existing = [item for item in treatments if item["result_path"].is_file()]
    if run_completed and len(completed) != len(treatments):
        raise InspectionError(
            "completed run does not mark every treatment completed"
        )
    if len(existing) != len(completed):
        raise InspectionError(
            f"manifest records {len(completed)} completed results but "
            f"{len(existing)} exist"
        )
    discovered = {
        path.resolve()
        for path in output_dir.rglob("result.json")
        if "inspection" not in path.relative_to(output_dir).parts
    }
    planned_existing = {item["result_path"] for item in existing}
    if discovered != planned_existing:
        raise InspectionError(
            "result.json inventory does not exactly match the manifest"
        )
    declared_count = raw_manifest.get(
        "result_count", raw_manifest.get("completed_treatment_count")
    )
    if declared_count is not None and declared_count != len(existing):
        raise InspectionError(
            "manifest declared result count does not match files"
        )

    all_rows: list[dict[str, Any]] = []
    all_per_task: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    transcripts: dict[str, BaselinePreviewTranscript] = {}
    roots: dict[str, Path] = {}
    candidate_arms: dict[str, str] = {}
    for item in existing:
        try:
            transcript = BaselinePreviewTranscript.model_validate_json(
                item["result_path"].read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise InspectionError(
                "invalid BaselinePreviewTranscript at "
                f"{item['result_path']}: {exc}"
            ) from exc
        _validate_plan(transcript, item, raw_manifest)
        treatment_id = item["treatment_id"]
        transcripts[treatment_id] = transcript
        roots[treatment_id] = item["result_path"].parent
        candidate_arms[
            transcript.baseline.evidence.candidate.record.candidate_id
        ] = "BASELINE"
        candidate_arms[
            transcript.ceiling.evidence.candidate.record.candidate_id
        ] = "HUMAN_BEST"
        treatment_rows: list[dict[str, Any]] = []
        treatment_per_task: list[dict[str, Any]] = []
        for arm_name, arm in (
            ("BASELINE", transcript.baseline),
            ("HUMAN_BEST", transcript.ceiling),
        ):
            rows, per_task = _arm_rows(treatment_id, transcript, arm_name, arm)
            treatment_rows.extend(rows)
            treatment_per_task.extend(per_task)
        all_rows.extend(treatment_rows)
        all_per_task.extend(treatment_per_task)
        summaries.append(
            _treatment_summary(
                treatment_id, transcript, treatment_rows, treatment_per_task
            )
        )

    provider_calls = _provider_calls(output_dir, roots, candidate_arms)
    call_index: dict[tuple[str, str, str, int, str], str] = {}
    for call in provider_calls:
        if (
            call["arm"] != "unavailable"
            and call["task_identity"] is not None
            and call["repeat"] is not None
            and call["leg"] is not None
        ):
            key = (
                str(call["treatment_id"]),
                str(call["arm"]),
                str(call["task_identity"]),
                int(call["repeat"]),
                str(call["leg"]),
            )
            if key in call_index:
                raise InspectionError(
                    f"multiple provider cache entries for {key}"
                )
            call_index[key] = str(call["cache_key"])
    for row in all_rows:
        coordinate = (
            str(row["treatment_id"]),
            str(row["arm"]),
            str(row["task_identity"]),
            int(row["repeat"]),
        )
        row["encoder_provider_cache_key"] = call_index.get(
            (*coordinate, "enc")
        )
        row["decoder_provider_cache_key"] = call_index.get(
            (*coordinate, "dec")
        )

    partials = _partials(output_dir, roots)
    paired = _paired_deltas(all_per_task)
    inventory = _execution_inventory(output_dir)
    availability = {
        field: dict(
            sorted(
                Counter(str(call[field]) for call in provider_calls).items()
            )
        )
        for field in (
            "requested_config_identity_availability",
            "returned_model_availability",
            "raw_response_availability",
            "upstream_provider_availability",
        )
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "output_dir": str(output_dir),
        "manifest": {
            "mode": raw_manifest.get("mode", "unavailable"),
            "treatment_count": len(treatments),
            "completed_result_count": len(existing),
            "run_completed": run_completed,
            "integrity": "validated",
        },
        "treatments": summaries,
        "rows": all_rows,
        "per_task": all_per_task,
        "paired_deltas": paired,
        "provider_calls": provider_calls,
        "provider_evidence_availability": availability,
        "partials": partials,
        "execution_inventory": inventory,
        "limitations": [
            "Returned model, upstream-provider metadata, and decoded raw "
            "response are reported as unavailable when "
            "ProviderInvocationEvidence does not contain them; absence is "
            "never converted to zero.",
            "The HumanEval extraction projection is unavailable when the "
            "editable dr-code overlay does not export "
            "extract_humaneval_code; this does not invalidate the persisted "
            "matrix summary.",
            "This diagnostic matrix is descriptive and does not pool "
            "effects across models or budget ratios.",
        ],
    }
    inspection = _write_reports(output_dir, report)
    report["inspection_dir"] = str(inspection)
    return report


def _render(report: Mapping[str, Any], command: str, console: Console) -> None:
    manifest = report["manifest"]
    console.print(
        Panel.fit(
            f"{manifest['completed_result_count']}/"
            f"{manifest['treatment_count']} "
            f"typed treatment results · {len(report['rows'])} rows · "
            f"integrity {manifest['integrity']}",
            title="ED1 baseline behavior matrix inspection",
            border_style="green",
        )
    )
    if command in ("summary", "rows", "failures"):
        table = Table(
            "treatment",
            "rows",
            "failed",
            "budget violations",
            "correctness",
            "blend",
        )
        for item in report["treatments"]:
            accounting = item["row_accounting"]
            rows = sum(arm["planned"] for arm in accounting.values())
            failed = sum(
                arm["missing"] + arm["failed"] + arm["invalid"]
                for arm in accounting.values()
            )
            metrics = item["metrics"]
            table.add_row(
                str(item["treatment_id"]),
                str(rows),
                str(failed),
                str(item["budget"]["violations"]),
                "unavailable"
                if metrics["correctness_mean_present"] is None
                else f"{metrics['correctness_mean_present']:.3f}",
                "unavailable"
                if metrics["blended_reward_mean_per_task"] is None
                else f"{metrics['blended_reward_mean_per_task']:.3f}",
            )
        console.print(table)
    if command in ("summary", "providers"):
        console.print(
            f"Provider cache calls: {len(report['provider_calls'])}; "
            f"partial records: {len(report['partials'])}. "
            "Missing returned/upstream/raw evidence is explicitly unavailable."
        )
    if command in ("summary", "power"):
        table = Table(
            "treatment",
            "paired blend delta [95% CI]",
            "powered target",
            "tasks x repeats",
            "achieved MDD",
        )
        for item in report["treatments"]:
            ci = item["paired_delta_ci"]
            recommendation = item["power"]["recommendation"]
            table.add_row(
                str(item["treatment_id"]),
                f"{ci['point']:+.3f} [{ci['low']:+.3f}, {ci['high']:+.3f}]",
                "yes" if recommendation["achievable"] else "no",
                f"{recommendation['recommended_n_tasks']} x "
                f"{recommendation['recommended_repeats']}",
                f"{recommendation['achieved_mdd']:.3f}",
            )
        console.print(table)
    if command in ("summary", "inventory"):
        inventory = report["execution_inventory"]
        console.print(
            f"Execution records: {inventory['record_count']} "
            f"({inventory['outcomes']}); SQLite artifacts: "
            f"{len(inventory['databases'])}."
        )
    console.print(f"Machine reports: {report['inspection_dir']}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "summary",
            "rows",
            "failures",
            "providers",
            "inventory",
            "power",
        ),
        default="summary",
    )
    parser.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "summary",
        "rows",
        "failures",
        "providers",
        "inventory",
        "power",
    }
    if supplied and supplied[0] not in commands:
        supplied.insert(0, "summary")
    args = _parser().parse_args(supplied)
    try:
        report = inspect_matrix(args.output_dir)
    except InspectionError as exc:
        Console(stderr=True).print(f"[bold red]Inspection failed:[/] {exc}")
        return 2
    _render(report, args.command, Console())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
