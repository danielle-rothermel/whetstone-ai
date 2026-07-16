# ruff: noqa: E501

from __future__ import annotations

import base64
import json
import subprocess
from datetime import UTC, datetime
from decimal import Decimal

import duckdb
import pytest
from dr_code.humaneval import (
    CompletedScore,
    HumanEvalTask,
    SubmissionOutcome,
    extract_code_with_profile,
    resolve_humaneval_scoring_profile,
)
from dr_platform import (
    ExportReconciliationDependencies,
    PlatformSchema,
    SubmitOptions,
    pin_local_bundle,
    resolve_local_pin,
)
from dr_platform.export import ApplicationSnapshot
from dr_platform.publication import OpenSslEd25519Signer
from dr_platform.reconciliation_runtime import ReconcileOptions
from sqlalchemy import create_engine, insert, text

from whetstone.platform.scoring import score_metrics_payload
from whetstone.platform.targets import target_registry
from whetstone.publication import (
    ANALYSIS_BUNDLE_KEY,
    DETAIL_BUNDLE_KEY,
    AnalysisBundleReader,
    analysis_projection_specs,
    detail_projection_specs,
    export_whetstone,
)


class _UnusedQueueLookup:
    def retrieve_queue(self, name: str) -> object | None:
        raise AssertionError(f"unexpected queue lookup: {name}")


class _UnusedLifecycleReader:
    def observe(self, *, workflow_id: str):
        raise AssertionError(f"unexpected lifecycle read: {workflow_id}")

    def read_step_history(self, *, workflow_id: str, limit: int = 100):
        raise AssertionError(f"unexpected step-history read: {workflow_id}")


def _integrity_signer(tmp_path):
    private_key = tmp_path / "integrity-private.pem"
    subprocess.run(
        [
            "openssl",
            "genpkey",
            "-algorithm",
            "ED25519",
            "-out",
            str(private_key),
        ],
        check=True,
        capture_output=True,
    )
    public_key = subprocess.run(
        [
            "openssl",
            "pkey",
            "-in",
            str(private_key),
            "-pubout",
            "-outform",
            "DER",
        ],
        check=True,
        capture_output=True,
    ).stdout
    signer = OpenSslEd25519Signer(key_id="test", private_key_path=private_key)
    return signer, {signer.key_id: base64.b64encode(public_key).decode()}


@pytest.mark.integration
def test_export_builds_and_promotes_complete_pinned_bundles(
    app_postgres_schema, tmp_path
) -> None:
    """All application builders execute from one source snapshot per bundle."""

    engine = create_engine(app_postgres_schema.database_url)
    now = datetime.now(UTC)
    task = HumanEvalTask(
        task_id="task",
        prompt="def answer(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="answer",
        test=(
            "def check(candidate):\n"
            "    inputs = [(1,)]\n"
            "    results = [2]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    )
    scoring_profile = resolve_humaneval_scoring_profile(
        scoring_profile_id="humaneval",
        scoring_profile_version="v1",
    )
    raw_submission = "def answer(x):\n    return x + 1\n"
    completed_score = CompletedScore(
        raw_submission=raw_submission,
        extraction=extract_code_with_profile(
            raw_submission, profile=scoring_profile.parser_profile
        ),
        outcome=SubmissionOutcome.PASSED,
        score=1.0,
    )
    metrics = score_metrics_payload(
        task=task,
        node_attempts=(),
        scoring_profile=scoring_profile,
        completed_score=completed_score,
    )
    expected_compression_ratio = metrics.compression["gzip"][
        "ratio_to_ground_truth"
    ]
    with engine.begin() as connection:
        platform = PlatformSchema(prefix="whetstone")
        connection.execute(
            insert(platform.operations).values(
                operation_key="op",
                group_key="exp",
                workflow_role="generation",
                status="succeeded",
                requested_count=1,
                manifest_version=3,
                manifest_digest="digest",
                manifest_page_size=1,
                manifest_page_count=1,
                operation_execution_recipe_digest="recipe",
                target_key="target",
                target_version=1,
                target_contract_digest="target-contract",
                platform_cut_version=1,
                registration_cursor=1,
                retry_policy=SubmitOptions().retry_policy.model_dump(
                    mode="json"
                ),
                inserted_count=1,
                already_present_count=0,
                enqueued_count=1,
                workflow_already_present_count=0,
                enqueue_failed_count=0,
                active_count=0,
                succeeded_count=1,
                terminal_failed_count=0,
                cancelled_count=0,
                spec={},
                metadata={},
                created_at=now,
                registration_completed_at=now,
                updated_at=now,
                completed_at=now,
                change_seq=1,
            )
        )
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
                "'{}'::jsonb, "
                "'{}'::jsonb, "
                "'{}'::jsonb, :now)"
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
                "'digest', CAST(:platform_cut AS jsonb), 'digest', '[]'::jsonb, 'digest', "
                "'{}'::jsonb, 'digest', '{}'::jsonb, 'digest', 1, 1, 0, 0, :now)"
            ),
            {
                "now": now,
                "platform_cut": json.dumps(
                    [
                        {
                            "operation_key": "op",
                            "platform_cut_version": 1,
                        }
                    ]
                ),
            },
        )
        connection.execute(
            text(
                "INSERT INTO whetstone_generation_runs "
                "(generation_run_id, prediction_id, attempt_index, execution_recipe_digest, "
                "platform_item_id, platform_attempt, status, terminal_node_id, summary, started_at, completed_at) "
                "VALUES ('generation-selected', 'prediction', 0, 'recipe-0', 'generation-item', 0, "
                "'success', 'terminal', '{}'::jsonb, :now, :now), "
                "('generation-rejected', 'prediction', 1, 'recipe-1', 'generation-item', 1, "
                "'error', 'terminal', '{}'::jsonb, :now, :now)"
            ),
            {"now": now},
        )
        connection.execute(
            text(
                "INSERT INTO whetstone_node_attempts "
                "(node_attempt_id, generation_run_id, prediction_id, node_id, attempt_index, status, "
                "usage_cost, response_metadata, started_at, completed_at) VALUES "
                "('node-selected', 'generation-selected', 'prediction', 'terminal', 0, 'success', "
                "CAST(:selected_cost AS jsonb), '{}'::jsonb, :now, :now), "
                "('node-rejected', 'generation-rejected', 'prediction', 'terminal', 0, 'error', "
                "CAST(:rejected_cost AS jsonb), '{}'::jsonb, :now, :now)"
            ),
            {
                "now": now,
                "selected_cost": '{"usage_metadata":{},"provider_cost":0.125}',
                "rejected_cost": '{"usage_metadata":{},"provider_cost":9.0}',
            },
        )
        connection.execute(
            text(
                "INSERT INTO whetstone_score_attempts "
                "(score_attempt_id, prediction_id, generation_run_id, attempt_index, execution_recipe_digest, "
                "platform_item_id, platform_attempt, scoring_profile_id, scoring_profile_version, parser_profile_id, "
                "parser_version, dataset_name, dataset_split, dataset_snapshot, status, submission_outcome, score, "
                "extracted_submission, metrics, per_test_results, started_at, completed_at) VALUES "
                "('score-selected', 'prediction', 'generation-selected', 0, 'score-recipe-0', 'score-item-0', 0, 'humaneval', 'v1', "
                "'humaneval-best-effort', 'v1', 'dataset', 'test', '{}'::jsonb, 'success', 'passed', 1.0, '{}'::jsonb, "
                "CAST(:metrics AS jsonb), '[]'::jsonb, :now, :now), "
                "('score-cross-run', 'prediction', 'generation-rejected', 0, 'score-recipe-1', 'score-item-1', 0, 'humaneval', 'v1', "
                "'humaneval-best-effort', 'v1', 'dataset', 'test', '{}'::jsonb, 'success', 'tests_failed', 0.0, '{}'::jsonb, "
                "'{}'::jsonb, '[]'::jsonb, :now, :now)"
            ),
            {"now": now, "metrics": metrics.model_dump_json()},
        )
        connection.execute(
            text(
                "INSERT INTO whetstone_experiment_acceptance_generation_members "
                "(acceptance_id, prediction_id, disposition, generation_run_id, generation_operation_key, platform_item_id, platform_attempt) VALUES "
                "('acceptance', 'prediction', 'selected_success', 'generation-selected', 'generation-operation', 'generation-item', 0)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO whetstone_experiment_acceptance_generation_candidates "
                "(acceptance_id, prediction_id, generation_run_id, disposition, generation_operation_key, platform_item_id, platform_attempt, status) VALUES "
                "('acceptance', 'prediction', 'generation-selected', 'selected', 'generation-operation', 'generation-item', 0, 'success'), "
                "('acceptance', 'prediction', 'generation-rejected', 'rejected', 'generation-operation', 'generation-item', 1, 'error')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO whetstone_experiment_acceptance_scoring_members "
                "(acceptance_id, prediction_id, scoring_profile_id, scoring_profile_version, parser_profile_id, parser_version, dataset_name, dataset_split, disposition, generation_run_id, score_attempt_id, accepted_scoring_ordinal, scoring_operation_key, platform_item_id, platform_attempt, manifest_digest) VALUES "
                "('acceptance', 'prediction', 'humaneval', 'v1', 'humaneval-best-effort', 'v1', 'dataset', 'test', 'accepted', 'generation-selected', 'score-selected', 1, 'scoring-operation', 'score-item-0', 0, 'manifest')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO whetstone_experiment_acceptance_scoring_candidates "
                "(acceptance_id, prediction_id, scoring_profile_id, scoring_profile_version, parser_profile_id, parser_version, dataset_name, dataset_split, accepted_scoring_ordinal, score_attempt_id, generation_run_id, disposition, operation_key, manifest_digest, selection_digest, platform_item_id, platform_attempt, status, candidate_kind) VALUES "
                "('acceptance', 'prediction', 'humaneval', 'v1', 'humaneval-best-effort', 'v1', 'dataset', 'test', 1, 'score-selected', 'generation-selected', 'selected', 'scoring-operation', 'manifest', 'selection', 'score-item-0', 0, 'success', 'score_attempt'), "
                "('acceptance', 'prediction', 'humaneval', 'v1', 'humaneval-best-effort', 'v1', 'dataset', 'test', 1, 'score-cross-run', 'generation-rejected', 'superseded_generation', 'scoring-operation', 'manifest', 'selection', 'score-item-1', 0, 'success', 'score_attempt')"
            )
        )

    database = tmp_path / "analysis.duckdb"
    integrity_signer, public_key_ring = _integrity_signer(tmp_path)
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
        reconciliation=ExportReconciliationDependencies(
            resolver=target_registry(),
            queue_lookup=_UnusedQueueLookup(),
            reader=_UnusedLifecycleReader(),
            dbos_engine=engine,
            options=ReconcileOptions(page_size=100),
            max_cycles=2,
        ),
        integrity_signer=integrity_signer,
        destination_path=database,
    )
    assert [item.status for item in analysis.destinations] == ["PROMOTED"], (
        analysis
    )
    assert [item.status for item in detail.destinations] == ["PROMOTED"]
    analysis_pin = pin_local_bundle(database, bundle_key=ANALYSIS_BUNDLE_KEY)
    detail_pin = pin_local_bundle(database, bundle_key=DETAIL_BUNDLE_KEY)
    assert set(
        resolve_local_pin(
            database, analysis_pin, public_key_ring=public_key_ring
        ).members
    ) == {
        "experiments",
        "predictions",
        "generation_runs",
        "score_attempts",
        "sweep_metrics",
        "failure_metrics",
    }
    assert set(
        resolve_local_pin(
            database, detail_pin, public_key_ring=public_key_ring
        ).members
    ) == {
        "detail_predictions",
        "detail_prediction_payloads",
        "detail_generation_runs",
        "detail_node_attempts",
        "detail_score_attempts",
        "detail_score_harness_failures",
        "detail_platform_attempts",
    }
    reader = AnalysisBundleReader.from_pin(
        database, analysis_pin, public_key_ring=public_key_ring
    )
    prediction = reader.rows("predictions")[0]
    assert prediction["candidate_id"] == "prediction"
    assert prediction["provider_cost"] == Decimal("0.125")
    assert float(prediction["compression_ratio"]) == pytest.approx(
        expected_compression_ratio
    )
    assert [
        row["generation_run_id"] for row in reader.rows("generation_runs")
    ] == ["generation-selected"]
    assert [
        row["score_attempt_id"] for row in reader.rows("score_attempts")
    ] == ["score-selected"]
    detail_bundle = resolve_local_pin(
        database, detail_pin, public_key_ring=public_key_ring
    )
    with duckdb.connect(str(database), read_only=True) as connection:
        detail_generation_ids = {
            row[0]
            for row in connection.execute(
                f'SELECT generation_run_id FROM "{detail_bundle.members["detail_generation_runs"]}"'
            ).fetchall()
        }
        detail_score_ids = {
            row[0]
            for row in connection.execute(
                f'SELECT score_attempt_id FROM "{detail_bundle.members["detail_score_attempts"]}"'
            ).fetchall()
        }
    assert detail_generation_ids == {
        "generation-selected",
        "generation-rejected",
    }
    assert detail_score_ids == {"score-selected", "score-cross-run"}
    engine.dispose()
