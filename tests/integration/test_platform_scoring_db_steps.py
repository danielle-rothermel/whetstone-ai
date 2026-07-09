from __future__ import annotations

import uuid
from typing import Any

import pytest
from dr_code.humaneval import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
)
from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    PARSER_PROFILE_VERSION,
)
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from tests.integration.dbos_test_workflows import (
    integration_load_scoring_target_workflow,
    integration_persist_score_workflow,
)
from tests.support.platform_integration_helpers import (
    count_score_attempts,
    seed_scoring_target,
)
from tests.support.platform_scoring_fixtures import (
    scoring_task,
    seeded_scoring_target,
)
from tests.support.postgres_fixtures import start_test_workflow
from whetstone.platform.persistence import persist_score_attempt
from whetstone.platform.scoring import score_submission_run
from whetstone.records import ScoreAttemptRecord, stable_score_attempt_id

pytestmark = pytest.mark.integration


def _completed_score_submission_run(**kwargs: Any) -> ScoreAttemptRecord:
    record = score_submission_run(**kwargs)
    assert isinstance(record, ScoreAttemptRecord)
    return record


def test_load_scoring_target_step_round_trips_through_dbos(
    app_postgres_schema,
    reset_dbos,
) -> None:
    spec, run = seeded_scoring_target()
    seed_scoring_target(
        app_postgres_schema.database_url,
        spec=spec,
        generation_run=run,
    )

    workflow_id = f"test-load-scoring-target:{uuid.uuid4().hex}"
    payload = start_test_workflow(
        integration_load_scoring_target_workflow,
        workflow_id,
        app_postgres_schema.database_url,
        run.generation_run_id,
    )

    assert payload["spec"]["prediction_id"] == spec.prediction_id
    assert (
        payload["generation_run"]["generation_run_id"] == run.generation_run_id
    )
    assert payload["node_attempts"] == []


def test_persist_score_result_step_writes_rows_and_is_idempotent(
    app_postgres_schema,
    reset_dbos,
) -> None:
    spec, run = seeded_scoring_target()
    seed_scoring_target(
        app_postgres_schema.database_url,
        spec=spec,
        generation_run=run,
    )
    score = _completed_score_submission_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=scoring_task(),
        started_at=run.started_at,
        completed_at=run.completed_at,
    )
    payload = score.model_dump(mode="json")

    for attempt in range(2):
        workflow_id = f"test-persist-score:{uuid.uuid4().hex}:{attempt}"
        result = start_test_workflow(
            integration_persist_score_workflow,
            workflow_id,
            app_postgres_schema.database_url,
            payload,
        )
        if attempt == 0:
            assert result["status"] == "inserted"
        else:
            assert result["status"] == "already_present"

    assert (
        count_score_attempts(
            app_postgres_schema.database_url,
            score.score_attempt_id,
        )
        == 1
    )


def test_score_attempt_profile_unique_constraint_rejects_duplicate_profile(
    app_postgres_schema,
) -> None:
    spec, run = seeded_scoring_target()
    seed_scoring_target(
        app_postgres_schema.database_url,
        spec=spec,
        generation_run=run,
    )
    score = _completed_score_submission_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=scoring_task(),
        started_at=run.started_at,
        completed_at=run.completed_at,
    )
    duplicate_score = score.model_copy(
        update={"score_attempt_id": "duplicate-score-attempt-id"}
    )

    engine = create_engine(app_postgres_schema.database_url)
    try:
        with engine.begin() as connection:
            persist_score_attempt(connection, score_attempt=score)
        with pytest.raises(IntegrityError), engine.begin() as connection:
            persist_score_attempt(
                connection,
                score_attempt=duplicate_score,
            )
    finally:
        engine.dispose()

    score_attempt_id = stable_score_attempt_id(
        generation_run_id=run.generation_run_id,
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        attempt_index=0,
    )
    engine = create_engine(app_postgres_schema.database_url)
    try:
        with engine.connect() as connection:
            count = connection.execute(
                text(
                    "SELECT COUNT(*) FROM dr_dspy_score_attempts "
                    "WHERE score_attempt_id = :score_attempt_id"
                ),
                {"score_attempt_id": score_attempt_id},
            ).scalar_one()
    finally:
        engine.dispose()
    assert int(count) == 1
