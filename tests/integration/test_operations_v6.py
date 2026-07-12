"""Fresh-Postgres operator retry regressions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from dr_platform import PlatformSchema, request_next_attempt
from sqlalchemy import insert, select, update

from tests.integration.test_acceptance_v6 import (
    _submit_generation,
    _submit_scoring,
    _target,
)
from whetstone.db import schema
from whetstone.platform import operations
from whetstone.platform.spec_builder import direct_graph, prediction_spec
from whetstone.platform.targets import target_registry


def _succeed_platform_attempt(
    connection: Any, *, item_id: str, operation_key: str
) -> None:
    platform = PlatformSchema(prefix="whetstone")
    now = datetime.now(UTC)
    connection.execute(
        update(platform.item_attempts)
        .where(platform.item_attempts.c.item_id == item_id)
        .values(
            execution_state="succeeded",
            dbos_status="SUCCESS",
            terminal_at=now,
            updated_at=now,
        )
    )
    connection.execute(
        update(platform.operations)
        .where(platform.operations.c.operation_key == operation_key)
        .values(
            status="succeeded",
            active_count=0,
            succeeded_count=1,
            completed_at=now,
            updated_at=now,
            platform_cut_version=platform.operations.c.platform_cut_version
            + 1,
        )
    )


@pytest.mark.integration
def test_authoritative_generation_and_scoring_domain_outcomes_create_retries(
    app_postgres_schema: Any,
) -> None:
    engine = app_postgres_schema.engine
    platform = PlatformSchema(prefix="whetstone")
    spec = prediction_spec(direct_graph(), experiment_name="exp")
    _submit_generation(engine, operation_key="generation", specs=(spec,))
    now = datetime.now(UTC)
    with engine.begin() as connection:
        generation_item = connection.execute(
            select(platform.items.c.item_id).where(
                platform.items.c.operation_key == "generation"
            )
        ).scalar_one()
        _succeed_platform_attempt(
            connection,
            item_id=generation_item,
            operation_key="generation",
        )
        connection.execute(
            insert(schema.generation_runs).values(
                generation_run_id="failed-run",
                prediction_id=spec.prediction_id,
                attempt_index=0,
                execution_recipe_digest="generation-recipe",
                platform_item_id=generation_item,
                platform_attempt=0,
                status="error",
                terminal_node_id="terminal",
                terminal_output_node_id=None,
                summary={
                    "execution_order": ["terminal"],
                    "terminal_node_id": "terminal",
                    "terminal_error": {
                        "node_id": "terminal",
                        "status": "error",
                        "failure": {
                            "failure_class": "transient",
                            "error_type": "TransientError",
                            "message": "retry generation",
                        },
                    },
                },
                started_at=now,
                completed_at=now,
            )
        )

    generation_request = operations._domain_outcome_request(
        engine=engine,
        kind="generation_run",
        record_id="failed-run",
        requested_by="test",
    )
    generation_result = request_next_attempt(
        generation_request,
        engine=engine,
        resolver=target_registry(),
        schema=operations.PLATFORM_SCHEMA,
    )
    assert generation_result.created_attempt == 1
    assert generation_request.reason.value == "domain_outcome"
    assert generation_request.operator_confirmed_at is None
    assert (
        request_next_attempt(
            generation_request,
            engine=engine,
            resolver=target_registry(),
            schema=operations.PLATFORM_SCHEMA,
        )
        == generation_result
    )

    target = _target(spec.prediction_id, "failed-run")
    _submit_scoring(engine, operation_key="scoring", targets=(target,))
    with engine.begin() as connection:
        scoring_item = connection.execute(
            select(platform.items.c.item_id).where(
                platform.items.c.operation_key == "scoring"
            )
        ).scalar_one()
        _succeed_platform_attempt(
            connection, item_id=scoring_item, operation_key="scoring"
        )
        connection.execute(
            insert(schema.score_harness_failures).values(
                score_harness_failure_id="harness-failure",
                prediction_id=spec.prediction_id,
                generation_run_id="failed-run",
                attempt_index=0,
                execution_recipe_digest="scoring-recipe",
                platform_item_id=scoring_item,
                platform_attempt=0,
                score_attempt_id="score-attempt",
                scoring_profile_id=target.scoring_profile_id,
                scoring_profile_version=target.scoring_profile_version,
                parser_profile_id=target.parser_profile_id,
                parser_version=target.parser_version,
                dataset_name=target.dataset_name,
                dataset_split=target.dataset_split,
                failure={
                    "kind": "harness_failure",
                    "raw_submission": "bad",
                    "extracted_submission": None,
                    "cause": {
                        "exception_type": "HarnessError",
                        "message": "retry scoring",
                    },
                    "failure_class": "transient",
                    "dataset_snapshot": target.dataset_snapshot.model_dump(
                        mode="json"
                    ),
                },
                started_at=now,
                completed_at=now,
            )
        )

    scoring_request = operations._domain_outcome_request(
        engine=engine,
        kind="score_harness_failure",
        record_id="harness-failure",
        requested_by="test",
    )
    scoring_result = request_next_attempt(
        scoring_request,
        engine=engine,
        resolver=target_registry(),
        schema=operations.PLATFORM_SCHEMA,
    )
    assert scoring_result.created_attempt == 1
    assert scoring_request.reason.value == "domain_outcome"
    assert scoring_request.operator_confirmed_at is None
    assert (
        request_next_attempt(
            scoring_request,
            engine=engine,
            resolver=target_registry(),
            schema=operations.PLATFORM_SCHEMA,
        )
        == scoring_result
    )
