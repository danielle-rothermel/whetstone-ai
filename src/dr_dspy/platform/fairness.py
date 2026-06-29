from __future__ import annotations

from collections.abc import Iterable

from dr_dspy.records import PredictionSpecRecord


def fair_ordered_specs(
    specs: Iterable[PredictionSpecRecord],
) -> tuple[PredictionSpecRecord, ...]:
    validated = tuple(validate_fair_order_spec(spec) for spec in specs)
    return tuple(
        sorted(
            validated,
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
