from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from dr_dspy.records import PredictionSpecRecord


@dataclass(frozen=True, slots=True)
class JsonlSpecRef:
    fair_order_key: str
    prediction_id: str
    byte_offset: int


def index_jsonl_prediction_specs(
    path: Path,
    *,
    experiment_name: str,
) -> tuple[JsonlSpecRef, ...]:
    refs: list[JsonlSpecRef] = []
    seen_prediction_ids: set[str] = set()
    with path.open("rb") as file:
        for line_number, line in _iter_nonempty_jsonl_lines(file):
            payload = _parse_jsonl_index_payload(line, line_number=line_number)
            spec_experiment_name = _required_string_field(
                payload,
                "experiment_name",
                line_number=line_number,
            )
            if spec_experiment_name != experiment_name:
                raise ValueError(
                    "prediction spec experiment_name must match "
                    "submit operation"
                )
            prediction_id = _required_string_field(
                payload,
                "prediction_id",
                line_number=line_number,
            )
            if prediction_id in seen_prediction_ids:
                raise ValueError(
                    "duplicate prediction_id in submit operation: "
                    f"{prediction_id}"
                )
            seen_prediction_ids.add(prediction_id)
            fair_order_key = _required_string_field(
                payload,
                "fair_order_key",
                line_number=line_number,
            )
            refs.append(
                JsonlSpecRef(
                    fair_order_key=fair_order_key,
                    prediction_id=prediction_id,
                    byte_offset=line.byte_offset,
                )
            )
    return tuple(refs)


def load_jsonl_prediction_specs(
    path: Path,
    refs: Sequence[JsonlSpecRef],
) -> tuple[PredictionSpecRecord, ...]:
    if not refs:
        return ()
    specs_by_prediction_id: dict[str, PredictionSpecRecord] = {}
    refs_by_offset = sorted(refs, key=lambda ref: ref.byte_offset)
    with path.open("rb") as file:
        for ref in refs_by_offset:
            file.seek(ref.byte_offset)
            line = file.readline()
            try:
                spec = _validate_fair_order_spec(
                    PredictionSpecRecord.model_validate_json(
                        line.decode("utf-8")
                    )
                )
            except ValueError as error:
                raise ValueError(
                    f"invalid prediction spec JSON at byte offset "
                    f"{ref.byte_offset}"
                ) from error
            specs_by_prediction_id[ref.prediction_id] = spec
    return tuple(specs_by_prediction_id[ref.prediction_id] for ref in refs)


@dataclass(frozen=True, slots=True)
class _JsonlLine:
    byte_offset: int
    content: bytes


def _iter_nonempty_jsonl_lines(
    file: BinaryIO,
) -> Iterator[tuple[int, _JsonlLine]]:
    line_number = 0
    while True:
        byte_offset = file.tell()
        line = file.readline()
        if not line:
            break
        if not line.strip():
            continue
        line_number += 1
        yield line_number, _JsonlLine(byte_offset=byte_offset, content=line)


def _parse_jsonl_index_payload(
    line: _JsonlLine,
    *,
    line_number: int,
) -> dict[str, object]:
    try:
        payload = json.loads(line.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid prediction spec JSON on line {line_number}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(
            f"invalid prediction spec JSON on line {line_number}"
        )
    return payload


def _required_string_field(
    payload: dict[str, object],
    field_name: str,
    *,
    line_number: int,
) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise ValueError(
            f"invalid prediction spec JSON on line {line_number}"
        )
    return value


def _validate_fair_order_spec(
    spec: PredictionSpecRecord,
) -> PredictionSpecRecord:
    return PredictionSpecRecord.model_validate(spec.model_dump(mode="json"))
