from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from whetstone.platform.jsonl_specs import JsonlSpecRef
from whetstone.records import PredictionSpecRecord


def fair_ordered_specs(
    specs: Iterable[PredictionSpecRecord],
) -> tuple[PredictionSpecRecord, ...]:
    return fair_order_specs(
        tuple(validate_fair_order_spec(spec) for spec in specs)
    )


def fair_ordered_spec_ref_windows(
    refs: Sequence[JsonlSpecRef],
    *,
    window_size: int,
) -> Iterator[tuple[JsonlSpecRef, ...]]:
    if window_size < 1:
        raise ValueError("window_size must be positive")
    ordered = tuple(
        sorted(
            refs,
            key=lambda ref: (ref.fair_order_key, ref.prediction_id),
        )
    )
    for index in range(0, len(ordered), window_size):
        yield ordered[index:index + window_size]


def fair_ordered_spec_windows(
    specs: Iterable[PredictionSpecRecord],
    *,
    window_size: int,
) -> Iterator[tuple[PredictionSpecRecord, ...]]:
    if window_size < 1:
        raise ValueError("window_size must be positive")
    ordered = fair_order_specs(
        tuple(validate_fair_order_spec(spec) for spec in specs)
    )
    for index in range(0, len(ordered), window_size):
        yield ordered[index:index + window_size]


def fair_order_specs(
    specs: Sequence[PredictionSpecRecord],
) -> tuple[PredictionSpecRecord, ...]:
    return tuple(
        sorted(
            specs,
            key=lambda spec: (
                spec.fair_order_key,
                spec.prediction_id,
            ),
        )
    )


def validate_fair_order_spec(
    spec: PredictionSpecRecord,
) -> PredictionSpecRecord:
    return PredictionSpecRecord.model_validate(spec.model_dump(mode="json"))
