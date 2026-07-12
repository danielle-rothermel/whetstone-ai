"""Whetstone's immutable Analysis and Detail publication contracts.

The platform owns pointer promotion.  This module owns only the application
members that may appear behind those pointers and the pinned read boundary
used by COPRO.  Operational Postgres rows never escape through this API.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import duckdb
from dr_platform import PinnedBundle, ProjectionSpec, resolve_local_pin
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

ANALYSIS_BUNDLE_KEY = "whetstone-analysis"
DETAIL_BUNDLE_KEY = "whetstone-detail"
ANALYSIS_MEMBERS = (
    "experiments",
    "predictions",
    "generation_runs",
    "score_attempts",
    "sweep_metrics",
    "failure_metrics",
)
DETAIL_MEMBERS = (
    "detail_predictions",
    "detail_prediction_payloads",
    "detail_generation_runs",
    "detail_node_attempts",
    "detail_score_attempts",
    "detail_score_harness_failures",
    "detail_platform_attempts",
)


class BundleRow(BaseModel):
    """A row constructed from one application snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: StrictStr
    snapshot_seq: StrictInt


def analysis_projection_specs() -> tuple[ProjectionSpec, ...]:
    """Return the frozen six-member Analysis Bundle inventory."""

    return (
        ProjectionSpec(
            member="experiments",
            columns=("experiment_name", "bundle_id", "snapshot_seq"),
            unique_key=("experiment_name",),
        ),
        ProjectionSpec(
            member="predictions",
            columns=(
                "prediction_id",
                "experiment_name",
                "bundle_id",
                "snapshot_seq",
            ),
            unique_key=("prediction_id",),
            references=(
                ("experiment_name", "experiments", "experiment_name"),
            ),
        ),
        ProjectionSpec(
            member="generation_runs",
            columns=(
                "generation_run_id",
                "prediction_id",
                "status",
                "platform_attempt",
                "bundle_id",
                "snapshot_seq",
            ),
            unique_key=("generation_run_id",),
            references=(("prediction_id", "predictions", "prediction_id"),),
        ),
        ProjectionSpec(
            member="score_attempts",
            columns=(
                "score_attempt_id",
                "prediction_id",
                "generation_run_id",
                "status",
                "score",
                "platform_attempt",
                "bundle_id",
                "snapshot_seq",
            ),
            unique_key=("score_attempt_id",),
            references=(
                ("prediction_id", "predictions", "prediction_id"),
                ("generation_run_id", "generation_runs", "generation_run_id"),
            ),
        ),
        ProjectionSpec(
            member="sweep_metrics",
            columns=(
                "experiment_name",
                "metric_key",
                "metric_value",
                "bundle_id",
                "snapshot_seq",
            ),
            unique_key=("experiment_name", "metric_key"),
            references=(
                ("experiment_name", "experiments", "experiment_name"),
            ),
        ),
        ProjectionSpec(
            member="failure_metrics",
            columns=(
                "experiment_name",
                "failure_class",
                "failure_count",
                "bundle_id",
                "snapshot_seq",
            ),
            unique_key=("experiment_name", "failure_class"),
            references=(
                ("experiment_name", "experiments", "experiment_name"),
            ),
        ),
    )


def detail_projection_specs() -> tuple[ProjectionSpec, ...]:
    """Return the root-cascaded Detail Bundle inventory."""

    root = ("prediction_id", "detail_predictions", "prediction_id")
    return (
        ProjectionSpec(
            member="detail_predictions",
            columns=("prediction_id", "bundle_id", "snapshot_seq"),
            unique_key=("prediction_id",),
        ),
        *(
            ProjectionSpec(
                member=member,
                columns=("prediction_id", "bundle_id", "snapshot_seq"),
                unique_key=("prediction_id",),
                references=(root,),
            )
            for member in DETAIL_MEMBERS[1:]
        ),
    )


def validate_projection_specs(specs: Iterable[ProjectionSpec]) -> None:
    """Reject incomplete inventories and invalid member references early."""

    declared = tuple(specs)
    names = {spec.member for spec in declared}
    if len(names) != len(declared):
        raise ValueError("publication members must be unique")
    for spec in declared:
        if not spec.unique_key or not set(spec.unique_key).issubset(
            spec.columns
        ):
            raise ValueError(f"{spec.member} has an invalid member key")
        for local, target, target_column in spec.references:
            if local not in spec.columns or target not in names:
                raise ValueError(
                    f"{spec.member} has an invalid member reference"
                )
            target_spec = next(
                item for item in declared if item.member == target
            )
            if target_column not in target_spec.columns:
                raise ValueError(f"{spec.member} references an unknown column")


class AnalysisBundleReader:
    """Read only tables named by a still-valid Analysis Bundle pin."""

    def __init__(
        self, database_path: str | Path, pinned: PinnedBundle
    ) -> None:
        self._database_path = Path(database_path)
        self._pinned = pinned
        if set(pinned.members) != set(ANALYSIS_MEMBERS):
            raise ValueError(
                "pinned bundle is not a complete Whetstone Analysis Bundle"
            )

    @classmethod
    def from_pin(
        cls, database_path: str | Path, pin: Any
    ) -> AnalysisBundleReader:
        return cls(database_path, resolve_local_pin(database_path, pin))

    @property
    def snapshot_seq(self) -> int:
        return self._pinned.snapshot_seq

    def rows(
        self, member: str, *, where: str = "", params: tuple[Any, ...] = ()
    ) -> tuple[Mapping[str, Any], ...]:
        if member not in ANALYSIS_MEMBERS:
            raise ValueError("Analysis reader does not expose that member")
        table = self._pinned.members[member]
        clause = f" WHERE {where}" if where else ""
        with duckdb.connect(
            str(self._database_path), read_only=True
        ) as connection:
            result = connection.execute(
                f'SELECT * FROM "{table}"{clause}', params
            )
            columns = tuple(item[0] for item in result.description)
            return tuple(
                dict(zip(columns, row, strict=True))
                for row in result.fetchall()
            )


validate_projection_specs(analysis_projection_specs())
validate_projection_specs(detail_projection_specs())
