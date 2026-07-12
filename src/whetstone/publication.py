"""The frozen Whetstone Analysis and Detail publication surfaces."""
# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
from dr_platform import (
    ApplicationSnapshot,
    ExportOptions,
    ExportReconciliationDependencies,
    ExportResult,
    PinnedBundle,
    PlatformSchema,
    ProjectionColumn,
    ProjectionColumnType,
    ProjectionSpec,
    export,
    resolve_local_pin,
)
from dr_platform.publication import BundleIntegritySigner
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr
from sqlalchemy import Connection, Engine, text

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
PUBLISHED_COMPRESSION_METHOD = "gzip"
PUBLISHED_COMPRESSION_SEMANTICS = "ratio_to_ground_truth"
PLATFORM_SCHEMA = PlatformSchema(prefix="whetstone")
_TEXT = ProjectionColumnType.TEXT
_INTEGER = ProjectionColumnType.INTEGER
_NUMERIC = ProjectionColumnType.NUMERIC
_TIMESTAMP = ProjectionColumnType.TIMESTAMP
_JSON = ProjectionColumnType.JSON


def _column_schema(
    *columns: tuple[str, ProjectionColumnType],
) -> tuple[ProjectionColumn, ...]:
    return tuple(
        ProjectionColumn(name=name, type=type_) for name, type_ in columns
    )


class BundleRow(BaseModel):
    """The coordinate stamped into every application publication row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: StrictStr
    snapshot_seq: StrictInt


def _builder(
    sql: str,
) -> Callable[[Connection, ApplicationSnapshot], Sequence[Mapping[str, Any]]]:
    """Bind a member query to the one repeatable-read export snapshot."""

    def build(
        connection: Connection, snapshot: ApplicationSnapshot
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            {
                **dict(row),
                "bundle_id": f"whetstone_{snapshot.snapshot_seq}",
                "snapshot_seq": snapshot.snapshot_seq,
            }
            for row in connection.execute(text(sql)).mappings()
        )

    return build


# The accepted-current predicate is intentionally shared by every member.  A
# historical evaluation is never silently published merely because it is the
# newest row; the experiment pointer is the authority.
_ACCEPTED = """
WITH current_acceptances AS (
  SELECT e.*, a.acceptance_id
  FROM whetstone_experiments e
  JOIN whetstone_experiment_acceptance_evaluations a
    ON a.acceptance_id = e.current_acceptance_id
  WHERE e.current_acceptance_id IS NOT NULL
    AND a.acceptance_source_version = e.acceptance_source_version
    AND jsonb_typeof(a.platform_cut) = 'array'
    AND CASE
      WHEN jsonb_typeof(a.platform_cut) = 'array'
      THEN jsonb_array_length(a.platform_cut)
      ELSE 0
    END > 0
    AND (
      SELECT count(DISTINCT cut.value->>'operation_key')
      FROM jsonb_array_elements(
        CASE
          WHEN jsonb_typeof(a.platform_cut) = 'array'
          THEN a.platform_cut
          ELSE '[]'::jsonb
        END
      ) AS cut(value)
    ) = CASE
      WHEN jsonb_typeof(a.platform_cut) = 'array'
      THEN jsonb_array_length(a.platform_cut)
      ELSE 0
    END
    AND NOT EXISTS (
      SELECT 1
      FROM jsonb_array_elements(
        CASE
          WHEN jsonb_typeof(a.platform_cut) = 'array'
          THEN a.platform_cut
          ELSE '[]'::jsonb
        END
      ) AS cut(value)
      LEFT JOIN whetstone_operations operation
        ON operation.operation_key = cut.value->>'operation_key'
      WHERE jsonb_typeof(cut.value) IS DISTINCT FROM 'object'
        OR jsonb_typeof(cut.value->'operation_key') IS DISTINCT FROM 'string'
        OR jsonb_typeof(cut.value->'platform_cut_version') IS DISTINCT FROM 'number'
        OR operation.operation_key IS NULL
        OR operation.platform_cut_version::text
          IS DISTINCT FROM cut.value->>'platform_cut_version'
        OR operation.status NOT IN ('succeeded', 'partial', 'failed', 'cancelled')
    )
),
accepted_predictions AS (
  SELECT p.*, e.acceptance_id,
    gm.generation_run_id AS selected_generation_run_id
  FROM current_acceptances e
  JOIN whetstone_experiment_acceptance_generation_members gm
    ON gm.acceptance_id = e.acceptance_id
    AND gm.disposition = 'selected_success'
  JOIN whetstone_prediction_specs p
    ON p.prediction_id = gm.prediction_id
    AND p.experiment_name = e.experiment_name
),
selected_generation_runs AS (
  SELECT g.* FROM accepted_predictions p
  JOIN whetstone_generation_runs g
    ON g.generation_run_id = p.selected_generation_run_id
),
selected_score_attempts AS (
  SELECT s.* FROM accepted_predictions p
  JOIN whetstone_experiment_acceptance_scoring_members sm
    ON sm.acceptance_id = p.acceptance_id
    AND sm.prediction_id = p.prediction_id
    AND sm.disposition = 'accepted'
  JOIN whetstone_score_attempts s
    ON s.score_attempt_id = sm.score_attempt_id
    AND s.generation_run_id = p.selected_generation_run_id
)
"""


def analysis_projection_specs() -> tuple[ProjectionSpec, ...]:
    """Return the exact six-member Analysis contract and its builders."""

    return (
        ProjectionSpec(
            member="experiments",
            columns=(
                "experiment_id",
                "display_name",
                "experiment_kind",
                "source",
                "row_count",
                "pass_rate",
                "created_at",
                "updated_at",
                "config_json",
                "bundle_id",
                "snapshot_seq",
            ),
            column_schema=_column_schema(
                ("experiment_id", _TEXT),
                ("display_name", _TEXT),
                ("experiment_kind", _TEXT),
                ("source", _TEXT),
                ("row_count", _INTEGER),
                ("pass_rate", _NUMERIC),
                ("created_at", _TIMESTAMP),
                ("updated_at", _TIMESTAMP),
                ("config_json", _JSON),
                ("bundle_id", _TEXT),
                ("snapshot_seq", _INTEGER),
            ),
            unique_key=("experiment_id",),
            full_rebuild_builder=_builder(
                _ACCEPTED
                + """
              SELECT e.experiment_name AS experiment_id, e.experiment_name AS display_name,
                COALESCE(e.config_metadata->>'experiment_kind', 'whetstone') AS experiment_kind,
                'whetstone' AS source, count(p.prediction_id)::bigint AS row_count,
                avg(CASE WHEN s.score >= 1 THEN 1.0 ELSE 0.0 END) AS pass_rate,
                e.created_at, e.acceptance_updated_at AS updated_at, e.config_metadata AS config_json
              FROM current_acceptances e
              LEFT JOIN accepted_predictions p ON p.experiment_name = e.experiment_name
              LEFT JOIN LATERAL (SELECT score FROM selected_score_attempts x WHERE x.prediction_id = p.prediction_id ORDER BY x.scoring_profile_id, x.parser_profile_id, x.dataset_name, x.dataset_split LIMIT 1) s ON true
              GROUP BY e.experiment_name, e.config_metadata, e.created_at, e.acceptance_updated_at
            """
            ),
        ),
        ProjectionSpec(
            member="predictions",
            columns=(
                "prediction_id",
                "experiment_id",
                "candidate_id",
                "task_id",
                "sample_index",
                "experiment_kind",
                "source",
                "model",
                "result_state",
                "generation_status",
                "scoring_status",
                "score",
                "provider_cost",
                "latency_ms",
                "compression_ratio",
                "failure_class",
                "created_at",
                "updated_at",
                "summary_json",
                "bundle_id",
                "snapshot_seq",
            ),
            column_schema=_column_schema(
                ("prediction_id", _TEXT),
                ("experiment_id", _TEXT),
                ("candidate_id", _TEXT),
                ("task_id", _TEXT),
                ("sample_index", _INTEGER),
                ("experiment_kind", _TEXT),
                ("source", _TEXT),
                ("model", _TEXT),
                ("result_state", _TEXT),
                ("generation_status", _TEXT),
                ("scoring_status", _TEXT),
                ("score", _NUMERIC),
                ("provider_cost", _NUMERIC),
                ("latency_ms", _NUMERIC),
                ("compression_ratio", _NUMERIC),
                ("failure_class", _TEXT),
                ("created_at", _TIMESTAMP),
                ("updated_at", _TIMESTAMP),
                ("summary_json", _JSON),
                ("bundle_id", _TEXT),
                ("snapshot_seq", _INTEGER),
            ),
            unique_key=("prediction_id",),
            references=(("experiment_id", "experiments", "experiment_id"),),
            full_rebuild_builder=_builder(
                _ACCEPTED
                + f"""
              SELECT p.prediction_id, p.experiment_name AS experiment_id, p.prediction_id AS candidate_id,
                p.task_id, p.repetition_seed AS sample_index,
                COALESCE(e.config_metadata->>'experiment_kind', 'whetstone') AS experiment_kind,
                'whetstone' AS source, p.model,
                COALESCE(s.submission_outcome, g.status, 'MISSING') AS result_state,
                g.status AS generation_status, s.status AS scoring_status, s.score,
                costs.provider_cost,
                EXTRACT(EPOCH FROM (COALESCE(s.completed_at, g.completed_at) - COALESCE(s.started_at, g.started_at))) * 1000 AS latency_ms,
                NULLIF(s.metrics->'compression'->'{PUBLISHED_COMPRESSION_METHOD}'->>'{PUBLISHED_COMPRESSION_SEMANTICS}', '')::double precision AS compression_ratio,
                n.failure->>'class' AS failure_class,
                p.created_at, GREATEST(p.created_at, COALESCE(s.completed_at, g.completed_at, p.created_at)) AS updated_at,
                COALESCE(s.metrics, g.summary) AS summary_json
              FROM accepted_predictions p
              JOIN whetstone_experiments e ON e.experiment_name = p.experiment_name
              JOIN selected_generation_runs g ON g.generation_run_id=p.selected_generation_run_id
              LEFT JOIN LATERAL (SELECT * FROM selected_score_attempts x WHERE x.prediction_id=p.prediction_id ORDER BY x.scoring_profile_id, x.parser_profile_id, x.dataset_name, x.dataset_split LIMIT 1) s ON true
              LEFT JOIN LATERAL (SELECT sum(NULLIF(x.usage_cost->>'provider_cost', '')::double precision) AS provider_cost FROM whetstone_node_attempts x WHERE x.generation_run_id=p.selected_generation_run_id) costs ON true
              LEFT JOIN LATERAL (SELECT * FROM whetstone_node_attempts x WHERE x.generation_run_id=p.selected_generation_run_id AND x.failure IS NOT NULL ORDER BY x.completed_at DESC LIMIT 1) n ON true
            """
            ),
        ),
        ProjectionSpec(
            member="generation_runs",
            columns=(
                "generation_run_id",
                "prediction_id",
                "attempt_index",
                "execution_recipe_digest",
                "platform_item_id",
                "platform_attempt",
                "status",
                "terminal_node_id",
                "summary_json",
                "started_at",
                "completed_at",
                "bundle_id",
                "snapshot_seq",
            ),
            column_schema=_column_schema(
                ("generation_run_id", _TEXT),
                ("prediction_id", _TEXT),
                ("attempt_index", _INTEGER),
                ("execution_recipe_digest", _TEXT),
                ("platform_item_id", _TEXT),
                ("platform_attempt", _INTEGER),
                ("status", _TEXT),
                ("terminal_node_id", _TEXT),
                ("summary_json", _JSON),
                ("started_at", _TIMESTAMP),
                ("completed_at", _TIMESTAMP),
                ("bundle_id", _TEXT),
                ("snapshot_seq", _INTEGER),
            ),
            unique_key=("generation_run_id",),
            references=(("prediction_id", "predictions", "prediction_id"),),
            full_rebuild_builder=_builder(
                _ACCEPTED
                + """
              SELECT g.generation_run_id, g.prediction_id, g.attempt_index, g.execution_recipe_digest,
                g.platform_item_id, g.platform_attempt, g.status, g.terminal_node_id,
                g.summary AS summary_json, g.started_at, g.completed_at
              FROM selected_generation_runs g JOIN accepted_predictions p ON p.selected_generation_run_id=g.generation_run_id
            """
            ),
        ),
        ProjectionSpec(
            member="score_attempts",
            columns=(
                "score_attempt_id",
                "prediction_id",
                "generation_run_id",
                "attempt_index",
                "execution_recipe_digest",
                "platform_item_id",
                "platform_attempt",
                "scoring_profile_id",
                "parser_profile_id",
                "dataset_name",
                "dataset_split",
                "status",
                "submission_outcome",
                "score",
                "metrics_json",
                "started_at",
                "completed_at",
                "bundle_id",
                "snapshot_seq",
            ),
            column_schema=_column_schema(
                ("score_attempt_id", _TEXT),
                ("prediction_id", _TEXT),
                ("generation_run_id", _TEXT),
                ("attempt_index", _INTEGER),
                ("execution_recipe_digest", _TEXT),
                ("platform_item_id", _TEXT),
                ("platform_attempt", _INTEGER),
                ("scoring_profile_id", _TEXT),
                ("parser_profile_id", _TEXT),
                ("dataset_name", _TEXT),
                ("dataset_split", _TEXT),
                ("status", _TEXT),
                ("submission_outcome", _TEXT),
                ("score", _NUMERIC),
                ("metrics_json", _JSON),
                ("started_at", _TIMESTAMP),
                ("completed_at", _TIMESTAMP),
                ("bundle_id", _TEXT),
                ("snapshot_seq", _INTEGER),
            ),
            unique_key=("score_attempt_id",),
            references=(
                ("prediction_id", "predictions", "prediction_id"),
                ("generation_run_id", "generation_runs", "generation_run_id"),
            ),
            full_rebuild_builder=_builder(
                _ACCEPTED
                + """
              SELECT s.score_attempt_id, s.prediction_id, s.generation_run_id, s.attempt_index, s.execution_recipe_digest,
                s.platform_item_id, s.platform_attempt, s.scoring_profile_id, s.parser_profile_id,
                s.dataset_name, s.dataset_split, s.status, s.submission_outcome, s.score,
                s.metrics AS metrics_json, s.started_at, s.completed_at
              FROM selected_score_attempts s JOIN accepted_predictions p ON p.prediction_id=s.prediction_id
            """
            ),
        ),
        ProjectionSpec(
            member="sweep_metrics",
            columns=(
                "experiment_id",
                "metric_key",
                "metric_value",
                "bundle_id",
                "snapshot_seq",
            ),
            column_schema=_column_schema(
                ("experiment_id", _TEXT),
                ("metric_key", _TEXT),
                ("metric_value", _NUMERIC),
                ("bundle_id", _TEXT),
                ("snapshot_seq", _INTEGER),
            ),
            unique_key=("experiment_id", "metric_key"),
            references=(("experiment_id", "experiments", "experiment_id"),),
            full_rebuild_builder=_builder(
                _ACCEPTED
                + """
          SELECT experiment_name AS experiment_id, 'acceptance_source_version' AS metric_key, acceptance_source_version::double precision AS metric_value FROM current_acceptances
        """
            ),
        ),
        ProjectionSpec(
            member="failure_metrics",
            columns=(
                "experiment_id",
                "failure_class",
                "failure_count",
                "bundle_id",
                "snapshot_seq",
            ),
            column_schema=_column_schema(
                ("experiment_id", _TEXT),
                ("failure_class", _TEXT),
                ("failure_count", _INTEGER),
                ("bundle_id", _TEXT),
                ("snapshot_seq", _INTEGER),
            ),
            unique_key=("experiment_id", "failure_class"),
            references=(("experiment_id", "experiments", "experiment_id"),),
            full_rebuild_builder=_builder(
                _ACCEPTED
                + """
          SELECT p.experiment_name AS experiment_id, COALESCE(n.failure->>'class', 'NONE') AS failure_class, count(*)::bigint AS failure_count
          FROM accepted_predictions p LEFT JOIN whetstone_node_attempts n ON n.generation_run_id=p.selected_generation_run_id AND n.failure IS NOT NULL
          GROUP BY p.experiment_name, COALESCE(n.failure->>'class', 'NONE')
        """
            ),
        ),
    )


def detail_projection_specs() -> tuple[ProjectionSpec, ...]:
    """Return the root-cascaded seven-member Detail contract and builders."""

    root = ("prediction_id", "detail_predictions", "prediction_id")
    detail_predictions = _builder(
        _ACCEPTED
        + """
      SELECT p.prediction_id, p.experiment_name AS experiment_id, 'whetstone' AS source,
        COALESCE(e.config_metadata->>'experiment_kind', 'whetstone') AS experiment_kind, p.task_id,
        p.repetition_seed AS sample_index, p.model, COALESCE(s.submission_outcome, g.status, 'MISSING') AS result_state,
        g.status AS generation_status, s.status AS scoring_status, s.score, costs.provider_cost,
        p.created_at, GREATEST(p.created_at, COALESCE(s.completed_at, g.completed_at, p.created_at)) AS updated_at,
        COALESCE(s.metrics, g.summary) AS summary_json
      FROM accepted_predictions p JOIN whetstone_experiments e ON e.experiment_name=p.experiment_name
      JOIN selected_generation_runs g ON g.generation_run_id=p.selected_generation_run_id
      LEFT JOIN LATERAL (SELECT * FROM selected_score_attempts x WHERE x.prediction_id=p.prediction_id ORDER BY x.scoring_profile_id, x.parser_profile_id, x.dataset_name, x.dataset_split LIMIT 1) s ON true
      LEFT JOIN LATERAL (SELECT sum(NULLIF(x.usage_cost->>'provider_cost', '')::double precision) AS provider_cost FROM whetstone_node_attempts x WHERE x.generation_run_id=p.selected_generation_run_id) costs ON true
    """
    )
    return (
        ProjectionSpec(
            member="detail_predictions",
            columns=(
                "prediction_id",
                "experiment_id",
                "source",
                "experiment_kind",
                "task_id",
                "sample_index",
                "model",
                "result_state",
                "generation_status",
                "scoring_status",
                "score",
                "provider_cost",
                "created_at",
                "updated_at",
                "summary_json",
                "bundle_id",
                "snapshot_seq",
            ),
            column_schema=_column_schema(
                ("prediction_id", _TEXT),
                ("experiment_id", _TEXT),
                ("source", _TEXT),
                ("experiment_kind", _TEXT),
                ("task_id", _TEXT),
                ("sample_index", _INTEGER),
                ("model", _TEXT),
                ("result_state", _TEXT),
                ("generation_status", _TEXT),
                ("scoring_status", _TEXT),
                ("score", _NUMERIC),
                ("provider_cost", _NUMERIC),
                ("created_at", _TIMESTAMP),
                ("updated_at", _TIMESTAMP),
                ("summary_json", _JSON),
                ("bundle_id", _TEXT),
                ("snapshot_seq", _INTEGER),
            ),
            unique_key=("prediction_id",),
            full_rebuild_builder=detail_predictions,
        ),
        ProjectionSpec(
            member="detail_prediction_payloads",
            columns=(
                "prediction_id",
                "input_kind",
                "input_text",
                "output_kind",
                "output_text",
                "prompt_text",
                "code_text",
                "raw_generation",
                "metrics_json",
                "request_json",
                "response_json",
                "validation_json",
                "bundle_id",
                "snapshot_seq",
            ),
            column_schema=_column_schema(
                ("prediction_id", _TEXT),
                ("input_kind", _TEXT),
                ("input_text", _TEXT),
                ("output_kind", _TEXT),
                ("output_text", _TEXT),
                ("prompt_text", _TEXT),
                ("code_text", _TEXT),
                ("raw_generation", _JSON),
                ("metrics_json", _JSON),
                ("request_json", _JSON),
                ("response_json", _JSON),
                ("validation_json", _JSON),
                ("bundle_id", _TEXT),
                ("snapshot_seq", _INTEGER),
            ),
            unique_key=("prediction_id",),
            references=(root,),
            full_rebuild_builder=_builder(
                _ACCEPTED
                + """
          SELECT p.prediction_id, p.task_snapshot->>'kind' AS input_kind, p.task_snapshot->>'prompt' AS input_text,
            g.summary->>'terminal_output_kind' AS output_kind, g.summary->>'terminal_output' AS output_text,
            p.graph_snapshot->>'prompt' AS prompt_text, p.task_snapshot->>'code' AS code_text,
            g.summary AS raw_generation, s.metrics AS metrics_json, NULL::jsonb AS request_json,
            n.response_metadata AS response_json, NULL::jsonb AS validation_json
          FROM accepted_predictions p JOIN selected_generation_runs g ON g.generation_run_id=p.selected_generation_run_id
          LEFT JOIN LATERAL (SELECT * FROM selected_score_attempts x WHERE x.prediction_id=p.prediction_id ORDER BY x.scoring_profile_id, x.parser_profile_id, x.dataset_name, x.dataset_split LIMIT 1) s ON true
          LEFT JOIN LATERAL (SELECT * FROM whetstone_node_attempts x WHERE x.generation_run_id=p.selected_generation_run_id ORDER BY x.completed_at DESC LIMIT 1) n ON true
        """
            ),
        ),
        ProjectionSpec(
            member="detail_generation_runs",
            columns=(
                "prediction_id",
                "generation_run_id",
                "attempt_index",
                "execution_recipe_digest",
                "platform_item_id",
                "platform_attempt",
                "status",
                "terminal_node_id",
                "terminal_output_node_id",
                "summary_json",
                "started_at",
                "completed_at",
                "bundle_id",
                "snapshot_seq",
            ),
            column_schema=_column_schema(
                ("prediction_id", _TEXT),
                ("generation_run_id", _TEXT),
                ("attempt_index", _INTEGER),
                ("execution_recipe_digest", _TEXT),
                ("platform_item_id", _TEXT),
                ("platform_attempt", _INTEGER),
                ("status", _TEXT),
                ("terminal_node_id", _TEXT),
                ("terminal_output_node_id", _TEXT),
                ("summary_json", _JSON),
                ("started_at", _TIMESTAMP),
                ("completed_at", _TIMESTAMP),
                ("bundle_id", _TEXT),
                ("snapshot_seq", _INTEGER),
            ),
            unique_key=("generation_run_id",),
            references=(root,),
            full_rebuild_builder=_builder(
                _ACCEPTED
                + """
          SELECT g.prediction_id, g.generation_run_id, g.attempt_index, g.execution_recipe_digest, g.platform_item_id, g.platform_attempt, g.status, g.terminal_node_id, g.terminal_output_node_id, g.summary AS summary_json, g.started_at, g.completed_at FROM accepted_predictions p JOIN whetstone_experiment_acceptance_generation_candidates c ON c.acceptance_id=p.acceptance_id AND c.prediction_id=p.prediction_id JOIN whetstone_generation_runs g ON g.generation_run_id=c.generation_run_id
        """
            ),
        ),
        ProjectionSpec(
            member="detail_node_attempts",
            columns=(
                "prediction_id",
                "node_attempt_id",
                "generation_run_id",
                "node_id",
                "attempt_index",
                "status",
                "provider_kind",
                "endpoint_kind",
                "model",
                "provider_config_json",
                "output_json",
                "usage_cost_json",
                "response_metadata_json",
                "failure_json",
                "started_at",
                "completed_at",
                "bundle_id",
                "snapshot_seq",
            ),
            column_schema=_column_schema(
                ("prediction_id", _TEXT),
                ("node_attempt_id", _TEXT),
                ("generation_run_id", _TEXT),
                ("node_id", _TEXT),
                ("attempt_index", _INTEGER),
                ("status", _TEXT),
                ("provider_kind", _TEXT),
                ("endpoint_kind", _TEXT),
                ("model", _TEXT),
                ("provider_config_json", _JSON),
                ("output_json", _JSON),
                ("usage_cost_json", _JSON),
                ("response_metadata_json", _JSON),
                ("failure_json", _JSON),
                ("started_at", _TIMESTAMP),
                ("completed_at", _TIMESTAMP),
                ("bundle_id", _TEXT),
                ("snapshot_seq", _INTEGER),
            ),
            unique_key=("node_attempt_id",),
            references=(root,),
            full_rebuild_builder=_builder(
                _ACCEPTED
                + """
          SELECT n.prediction_id, n.node_attempt_id, n.generation_run_id, n.node_id, n.attempt_index, n.status, n.provider_kind, n.endpoint_kind, n.model, n.provider_config AS provider_config_json, n.output AS output_json, n.usage_cost AS usage_cost_json, n.response_metadata AS response_metadata_json, n.failure AS failure_json, n.started_at, n.completed_at FROM accepted_predictions p JOIN whetstone_experiment_acceptance_generation_candidates c ON c.acceptance_id=p.acceptance_id AND c.prediction_id=p.prediction_id JOIN whetstone_node_attempts n ON n.generation_run_id=c.generation_run_id
        """
            ),
        ),
        ProjectionSpec(
            member="detail_score_attempts",
            columns=(
                "prediction_id",
                "score_attempt_id",
                "generation_run_id",
                "attempt_index",
                "execution_recipe_digest",
                "platform_item_id",
                "platform_attempt",
                "scoring_profile_id",
                "scoring_profile_version",
                "parser_profile_id",
                "parser_version",
                "dataset_name",
                "dataset_split",
                "dataset_snapshot_json",
                "status",
                "submission_outcome",
                "score",
                "extracted_submission_json",
                "metrics_json",
                "per_test_results_json",
                "started_at",
                "completed_at",
                "bundle_id",
                "snapshot_seq",
            ),
            column_schema=_column_schema(
                ("prediction_id", _TEXT),
                ("score_attempt_id", _TEXT),
                ("generation_run_id", _TEXT),
                ("attempt_index", _INTEGER),
                ("execution_recipe_digest", _TEXT),
                ("platform_item_id", _TEXT),
                ("platform_attempt", _INTEGER),
                ("scoring_profile_id", _TEXT),
                ("scoring_profile_version", _TEXT),
                ("parser_profile_id", _TEXT),
                ("parser_version", _TEXT),
                ("dataset_name", _TEXT),
                ("dataset_split", _TEXT),
                ("dataset_snapshot_json", _JSON),
                ("status", _TEXT),
                ("submission_outcome", _TEXT),
                ("score", _NUMERIC),
                ("extracted_submission_json", _JSON),
                ("metrics_json", _JSON),
                ("per_test_results_json", _JSON),
                ("started_at", _TIMESTAMP),
                ("completed_at", _TIMESTAMP),
                ("bundle_id", _TEXT),
                ("snapshot_seq", _INTEGER),
            ),
            unique_key=("score_attempt_id",),
            references=(root,),
            full_rebuild_builder=_builder(
                _ACCEPTED
                + """
          SELECT s.prediction_id, s.score_attempt_id, s.generation_run_id, s.attempt_index, s.execution_recipe_digest, s.platform_item_id, s.platform_attempt, s.scoring_profile_id, s.scoring_profile_version, s.parser_profile_id, s.parser_version, s.dataset_name, s.dataset_split, s.dataset_snapshot AS dataset_snapshot_json, s.status, s.submission_outcome, s.score, s.extracted_submission AS extracted_submission_json, s.metrics AS metrics_json, s.per_test_results AS per_test_results_json, s.started_at, s.completed_at FROM accepted_predictions p JOIN whetstone_experiment_acceptance_scoring_candidates c ON c.acceptance_id=p.acceptance_id AND c.prediction_id=p.prediction_id AND c.candidate_kind='score_attempt' JOIN whetstone_score_attempts s ON s.score_attempt_id=c.score_attempt_id
        """
            ),
        ),
        ProjectionSpec(
            member="detail_score_harness_failures",
            columns=(
                "prediction_id",
                "score_harness_failure_id",
                "generation_run_id",
                "score_attempt_id",
                "attempt_index",
                "execution_recipe_digest",
                "platform_item_id",
                "platform_attempt",
                "scoring_profile_id",
                "parser_profile_id",
                "dataset_name",
                "dataset_split",
                "failure_json",
                "started_at",
                "completed_at",
                "bundle_id",
                "snapshot_seq",
            ),
            column_schema=_column_schema(
                ("prediction_id", _TEXT),
                ("score_harness_failure_id", _TEXT),
                ("generation_run_id", _TEXT),
                ("score_attempt_id", _TEXT),
                ("attempt_index", _INTEGER),
                ("execution_recipe_digest", _TEXT),
                ("platform_item_id", _TEXT),
                ("platform_attempt", _INTEGER),
                ("scoring_profile_id", _TEXT),
                ("parser_profile_id", _TEXT),
                ("dataset_name", _TEXT),
                ("dataset_split", _TEXT),
                ("failure_json", _JSON),
                ("started_at", _TIMESTAMP),
                ("completed_at", _TIMESTAMP),
                ("bundle_id", _TEXT),
                ("snapshot_seq", _INTEGER),
            ),
            unique_key=("score_harness_failure_id",),
            references=(root,),
            full_rebuild_builder=_builder(
                _ACCEPTED
                + """
          SELECT h.prediction_id, h.score_harness_failure_id, h.generation_run_id, h.score_attempt_id, h.attempt_index, h.execution_recipe_digest, h.platform_item_id, h.platform_attempt, h.scoring_profile_id, h.parser_profile_id, h.dataset_name, h.dataset_split, h.failure AS failure_json, h.started_at, h.completed_at FROM accepted_predictions p JOIN whetstone_experiment_acceptance_scoring_candidates c ON c.acceptance_id=p.acceptance_id AND c.prediction_id=p.prediction_id AND c.candidate_kind='score_harness_failure' JOIN whetstone_score_harness_failures h ON h.score_attempt_id=c.score_attempt_id
        """
            ),
        ),
        ProjectionSpec(
            member="detail_platform_attempts",
            columns=(
                "prediction_id",
                "platform_item_id",
                "platform_attempt",
                "workflow_role",
                "execution_key",
                "workflow_id",
                "execution_state",
                "dbos_status",
                "failure_json",
                "created_at",
                "enqueued_at",
                "terminal_at",
                "updated_at",
                "bundle_id",
                "snapshot_seq",
            ),
            column_schema=_column_schema(
                ("prediction_id", _TEXT),
                ("platform_item_id", _TEXT),
                ("platform_attempt", _INTEGER),
                ("workflow_role", _TEXT),
                ("execution_key", _TEXT),
                ("workflow_id", _TEXT),
                ("execution_state", _TEXT),
                ("dbos_status", _TEXT),
                ("failure_json", _JSON),
                ("created_at", _TIMESTAMP),
                ("enqueued_at", _TIMESTAMP),
                ("terminal_at", _TIMESTAMP),
                ("updated_at", _TIMESTAMP),
                ("bundle_id", _TEXT),
                ("snapshot_seq", _INTEGER),
            ),
            unique_key=("platform_item_id", "platform_attempt"),
            references=(root,),
            full_rebuild_builder=_builder(
                _ACCEPTED
                + """
          SELECT p.prediction_id, a.item_id AS platform_item_id, a.attempt AS platform_attempt, a.workflow_role, a.execution_key, a.workflow_id, a.execution_state, a.dbos_status, a.failure AS failure_json, a.created_at, a.enqueued_at, a.terminal_at, a.updated_at
          FROM accepted_predictions p JOIN whetstone_item_attempts a ON (a.item_id, a.attempt) IN (SELECT c.platform_item_id, c.platform_attempt FROM whetstone_experiment_acceptance_generation_candidates c WHERE c.acceptance_id=p.acceptance_id AND c.prediction_id=p.prediction_id UNION SELECT c.platform_item_id, c.platform_attempt FROM whetstone_experiment_acceptance_scoring_candidates c WHERE c.acceptance_id=p.acceptance_id AND c.prediction_id=p.prediction_id)
        """
            ),
        ),
    )


def export_whetstone(
    source: Engine,
    *,
    reconciliation: ExportReconciliationDependencies,
    integrity_signer: BundleIntegritySigner,
    destination_path: str | Path,
    detail_destination_path: str | Path | None = None,
    analysis_remote_destinations: Sequence[Any] = (),
    detail_remote_destinations: Sequence[Any] = (),
) -> tuple[ExportResult, ExportResult]:
    """Run the one public platform export verb for both Whetstone bundles."""

    analysis = export(
        source,
        ExportOptions(
            destination_path=str(destination_path),
            bundle_key=ANALYSIS_BUNDLE_KEY,
            full_rebuild=True,
            projections=analysis_projection_specs(),
            source_change_sequence="whetstone_change_seq",
            integrity_signer=integrity_signer,
        ),
        reconciliation=reconciliation,
        schema=PLATFORM_SCHEMA,
        remote_destinations=analysis_remote_destinations,
    )
    detail = export(
        source,
        ExportOptions(
            destination_path=str(detail_destination_path or destination_path),
            bundle_key=DETAIL_BUNDLE_KEY,
            full_rebuild=True,
            projections=detail_projection_specs(),
            source_change_sequence="whetstone_change_seq",
            integrity_signer=integrity_signer,
        ),
        reconciliation=reconciliation,
        schema=PLATFORM_SCHEMA,
        remote_destinations=detail_remote_destinations,
    )
    return analysis, detail


def validate_projection_specs(specs: Iterable[ProjectionSpec]) -> None:
    declared = tuple(specs)
    names = {spec.member for spec in declared}
    if len(names) != len(declared):
        raise ValueError("publication members must be unique")
    for spec in declared:
        if (
            spec.full_rebuild_builder is None
            or not spec.unique_key
            or not set(spec.unique_key).issubset(spec.columns)
            or tuple(column.name for column in spec.column_schema)
            != spec.columns
        ):
            raise ValueError(f"{spec.member} has an invalid rebuild contract")
        for local, target, target_column in spec.references:
            target_spec = next(
                (item for item in declared if item.member == target), None
            )
            if (
                local not in spec.columns
                or target_spec is None
                or target_column not in target_spec.columns
            ):
                raise ValueError(
                    f"{spec.member} has an invalid member reference"
                )


class AnalysisBundleReader:
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
        cls,
        database_path: str | Path,
        pin: Any,
        *,
        public_key_ring: Mapping[str, str],
    ) -> AnalysisBundleReader:
        return cls(
            database_path,
            resolve_local_pin(
                database_path, pin, public_key_ring=public_key_ring
            ),
        )

    @property
    def snapshot_seq(self) -> int:
        return self._pinned.snapshot_seq

    def rows(
        self, member: str, *, where: str = "", params: tuple[Any, ...] = ()
    ) -> tuple[Mapping[str, Any], ...]:
        if member not in ANALYSIS_MEMBERS:
            raise ValueError("Analysis reader does not expose that member")
        clause = f" WHERE {where}" if where else ""
        with duckdb.connect(
            str(self._database_path), read_only=True
        ) as connection:
            result = connection.execute(
                f'SELECT * FROM "{self._pinned.members[member]}"{clause}',
                params,
            )
            columns = tuple(item[0] for item in result.description)
            return tuple(
                dict(zip(columns, row, strict=True))
                for row in result.fetchall()
            )


validate_projection_specs(analysis_projection_specs())
validate_projection_specs(detail_projection_specs())
