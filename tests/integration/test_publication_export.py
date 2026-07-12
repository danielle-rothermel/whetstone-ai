# ruff: noqa: E501

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from dr_platform import pin_local_bundle, resolve_local_pin
from dr_platform.export import ApplicationSnapshot
from sqlalchemy import create_engine, text

from whetstone.publication import (
    ANALYSIS_BUNDLE_KEY,
    DETAIL_BUNDLE_KEY,
    AnalysisBundleReader,
    analysis_projection_specs,
    detail_projection_specs,
    export_whetstone,
)


@pytest.mark.integration
def test_export_builds_and_promotes_complete_pinned_bundles(
    app_postgres_schema, tmp_path
) -> None:
    """All application builders execute from one source snapshot per bundle."""

    engine = create_engine(app_postgres_schema.database_url)
    now = datetime.now(UTC)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO whetstone_experiments "
                "(experiment_name, config_metadata, acceptance_source_version, "
                "current_acceptance_id, acceptance_updated_at, created_at) VALUES "
                "('exp', '{}'::jsonb, 1, 'acceptance', :now, :now)"
            ),
            {"now": now},
        )
        connection.execute(
            text(
                "INSERT INTO whetstone_prediction_specs "
                "(prediction_id, experiment_name, task_id, repetition_seed, "
                "graph_digest, dimensions_digest, graph_layout, provider_kind, "
                "endpoint_kind, model, throttle_key, task_snapshot, graph_snapshot, "
                "dimensions, provider_configs, created_at) VALUES "
                "('prediction', 'exp', 'task', 0, 'graph', 'dimensions', 'layout', "
                "'provider', 'endpoint', 'model', 'throttle', '{}'::jsonb, "
                "'{}'::jsonb, '{}'::jsonb, '{}'::jsonb, :now)"
            ),
            {"now": now},
        )
        connection.execute(
            text(
                "INSERT INTO whetstone_experiment_acceptance_evaluations "
                "(acceptance_id, experiment_name, acceptance_source_version, status, "
                "generation_operation_key, generation_manifest_digest, "
                "scoring_relationships, scoring_relationships_digest, "
                "selected_scoring_candidates, selected_scoring_candidates_digest, "
                "domain_cut, domain_cut_digest, platform_cut, platform_cut_digest, "
                "required_profiles, required_profiles_digest, policy, policy_digest, "
                "observed_matrix, observed_matrix_digest, expected_count, accepted_count, "
                "missing_count, rejected_count, created_at) VALUES "
                "('acceptance', 'exp', 1, 'ACCEPTED', 'op', 'digest', "
                "'[]'::jsonb, 'digest', '[]'::jsonb, 'digest', '{}'::jsonb, "
                "'digest', '{}'::jsonb, 'digest', '[]'::jsonb, 'digest', "
                "'{}'::jsonb, 'digest', '{}'::jsonb, 'digest', 1, 1, 0, 0, :now)"
            ),
            {"now": now},
        )
        connection.execute(
            text(
                "INSERT INTO whetstone_generation_runs "
                "(generation_run_id, prediction_id, attempt_index, execution_recipe_digest, "
                "platform_item_id, platform_attempt, status, terminal_node_id, summary, started_at, completed_at) "
                "VALUES ('generation', 'prediction', 0, 'recipe', 'generation-item', 0, "
                "'success', 'terminal', '{}'::jsonb, :now, :now)"
            ),
            {"now": now},
        )
        connection.execute(
            text(
                "INSERT INTO whetstone_node_attempts "
                "(node_attempt_id, generation_run_id, prediction_id, node_id, attempt_index, status, "
                "usage_cost, response_metadata, started_at, completed_at) VALUES "
                "('node', 'generation', 'prediction', 'terminal', 0, 'success', "
                "CAST(:usage_cost AS jsonb), '{}'::jsonb, :now, :now)"
            ),
            {
                "now": now,
                "usage_cost": '{"usage_metadata":{},"provider_cost":0.125}',
            },
        )
        connection.execute(
            text(
                "INSERT INTO whetstone_score_attempts "
                "(score_attempt_id, prediction_id, generation_run_id, attempt_index, execution_recipe_digest, "
                "platform_item_id, platform_attempt, scoring_profile_id, scoring_profile_version, parser_profile_id, "
                "parser_version, dataset_name, dataset_split, dataset_snapshot, status, submission_outcome, score, "
                "extracted_submission, metrics, per_test_results, started_at, completed_at) VALUES "
                "('score', 'prediction', 'generation', 0, 'score-recipe', 'score-item', 0, 'score-profile', '1', "
                "'parser-profile', '1', 'dataset', 'test', '{}'::jsonb, 'success', 'passed', 1.0, '{}'::jsonb, "
                "CAST(:metrics AS jsonb), '[]'::jsonb, :now, :now)"
            ),
            {"now": now, "metrics": '{"realized_compression_ratio":0.5}'},
        )

    database = tmp_path / "analysis.duckdb"
    with engine.begin() as connection:
        snapshot = ApplicationSnapshot(
            source_database="test", captured_at=now, snapshot_seq=1
        )
        for spec in analysis_projection_specs():
            assert spec.full_rebuild_builder is not None
            spec.full_rebuild_builder(connection, snapshot)
        for spec in detail_projection_specs():
            assert spec.full_rebuild_builder is not None
            spec.full_rebuild_builder(connection, snapshot)
    analysis, detail = export_whetstone(
        engine,
        destination_path=database,
    )
    assert [item.status for item in analysis.destinations] == ["PROMOTED"], analysis
    assert [item.status for item in detail.destinations] == ["PROMOTED"]
    analysis_pin = pin_local_bundle(database, bundle_key=ANALYSIS_BUNDLE_KEY)
    detail_pin = pin_local_bundle(database, bundle_key=DETAIL_BUNDLE_KEY)
    assert set(resolve_local_pin(database, analysis_pin).members) == {
        "experiments",
        "predictions",
        "generation_runs",
        "score_attempts",
        "sweep_metrics",
        "failure_metrics",
    }
    assert set(resolve_local_pin(database, detail_pin).members) == {
        "detail_predictions",
        "detail_prediction_payloads",
        "detail_generation_runs",
        "detail_node_attempts",
        "detail_score_attempts",
        "detail_score_harness_failures",
        "detail_platform_attempts",
    }
    prediction = AnalysisBundleReader.from_pin(database, analysis_pin).rows(
        "predictions"
    )[0]
    assert prediction["provider_cost"] == "0.125"
    assert prediction["compression_ratio"] == "0.5"
    engine.dispose()
