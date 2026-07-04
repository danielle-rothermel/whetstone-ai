from __future__ import annotations

from typing import Any

import pytest

from tests.support.platform_integration_helpers import (
    count_score_attempts,
    fetch_score_attempt_snapshot,
    seed_scoring_target,
)
from tests.support.platform_scoring_fixtures import (
    scoring_task,
    seeded_scoring_target,
)
from tests.support.postgres_fixtures import start_test_workflow
from whetstone.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    PARSER_PROFILE_VERSION,
)
from whetstone.humaneval.profiles import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
)
from whetstone.platform import scoring_workflow
from whetstone.platform.scoring_workflow import (
    run_score_generation_workflow,
    run_score_generation_workflow_once,
)
from whetstone.platform.scoring_workflow_state import (
    ScoringWorkflowPresence,
    classify_scoring_workflow_presence,
)
from whetstone.records import (
    DEFAULT_SCORE_DATASET_NAME,
    DEFAULT_SCORE_DATASET_SPLIT,
    stable_score_attempt_id,
)

pytestmark = pytest.mark.integration


def _mock_humaneval_task_step(monkeypatch: pytest.MonkeyPatch) -> None:
    task_payload = scoring_task().model_dump(
        mode="json",
        exclude={
            "ground_truth_code",
            "ground_truth_code_without_comments",
        },
    )

    def fake_load_humaneval_task_step(
        dataset_name: str,
        dataset_split: str,
        task_id: str,
    ) -> dict[str, Any]:
        assert task_id == "HumanEval/fixture"
        return task_payload

    monkeypatch.setattr(
        scoring_workflow,
        "load_humaneval_task_step",
        fake_load_humaneval_task_step,
    )


def test_run_score_generation_workflow_once_persists_success(
    app_postgres_schema,
    reset_dbos,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, run = seeded_scoring_target()
    seed_scoring_target(
        app_postgres_schema.database_url,
        spec=spec,
        generation_run=run,
    )
    _mock_humaneval_task_step(monkeypatch)
    scoring_workflow.load_humaneval_task_map.cache_clear()

    result = run_score_generation_workflow_once(
        app_postgres_schema.database_url,
        run.generation_run_id,
    )

    expected_score_id = stable_score_attempt_id(
        generation_run_id=run.generation_run_id,
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        attempt_index=0,
    )
    assert result.score_attempt_id == expected_score_id
    assert result.insert_status == "inserted"
    snapshot = fetch_score_attempt_snapshot(
        app_postgres_schema.database_url,
        expected_score_id,
    )
    assert snapshot is not None
    assert snapshot.status == "success"
    assert snapshot.score == 1.0
    assert snapshot.insert_count == 1


def test_run_score_generation_workflow_once_is_idempotent_on_replay(
    app_postgres_schema,
    reset_dbos,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, run = seeded_scoring_target()
    seed_scoring_target(
        app_postgres_schema.database_url,
        spec=spec,
        generation_run=run,
    )
    _mock_humaneval_task_step(monkeypatch)
    scoring_workflow.load_humaneval_task_map.cache_clear()

    first = run_score_generation_workflow_once(
        app_postgres_schema.database_url,
        run.generation_run_id,
    )
    second = run_score_generation_workflow_once(
        app_postgres_schema.database_url,
        run.generation_run_id,
    )

    assert first.score_attempt_id == second.score_attempt_id
    assert (
        count_score_attempts(
            app_postgres_schema.database_url,
            first.score_attempt_id,
        )
        == 1
    )
    assert first.insert_status == "inserted"
    assert second.insert_status == "inserted"


def test_scoring_task_loader_runs_once_across_workflow_replay(
    app_postgres_schema,
    reset_dbos,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, run = seeded_scoring_target()
    seed_scoring_target(
        app_postgres_schema.database_url,
        spec=spec,
        generation_run=run,
    )
    load_calls: list[tuple[str, str, str]] = []
    task_payload = scoring_task().model_dump(
        mode="json",
        exclude={
            "ground_truth_code",
            "ground_truth_code_without_comments",
        },
    )

    def counting_load_humaneval_task_step(
        dataset_name: str,
        dataset_split: str,
        task_id: str,
    ) -> dict[str, Any]:
        load_calls.append((dataset_name, dataset_split, task_id))
        return task_payload

    monkeypatch.setattr(
        scoring_workflow,
        "load_humaneval_task_step",
        counting_load_humaneval_task_step,
    )
    scoring_workflow.load_humaneval_task_map.cache_clear()

    run_score_generation_workflow_once(
        app_postgres_schema.database_url,
        run.generation_run_id,
    )
    run_score_generation_workflow_once(
        app_postgres_schema.database_url,
        run.generation_run_id,
    )

    assert load_calls == [
        (
            DEFAULT_SCORE_DATASET_NAME,
            DEFAULT_SCORE_DATASET_SPLIT,
            "HumanEval/fixture",
        ),
    ]


def test_failed_scoring_workflow_is_classified_as_orphan(
    app_postgres_schema,
    reset_dbos,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, run = seeded_scoring_target()
    seed_scoring_target(
        app_postgres_schema.database_url,
        spec=spec,
        generation_run=run,
    )
    _mock_humaneval_task_step(monkeypatch)
    scoring_workflow.load_humaneval_task_map.cache_clear()

    persist_impl = getattr(
        scoring_workflow.persist_score_attempt_step,
        "__wrapped__",
        scoring_workflow.persist_score_attempt_step,
    )
    persist_state = {"fail": True}

    def persist_stub(
        database_url: str,
        score_attempt_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if persist_state["fail"]:
            raise RuntimeError("simulated crash before score persist")
        return persist_impl(database_url, score_attempt_payload)

    monkeypatch.setattr(
        scoring_workflow,
        "persist_score_attempt_step",
        persist_stub,
    )

    score_attempt_id = stable_score_attempt_id(
        generation_run_id=run.generation_run_id,
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        attempt_index=0,
    )
    workflow_id = scoring_workflow.platform_scoring_workflow_id(
        score_attempt_id
    )

    with pytest.raises(
        RuntimeError,
        match="simulated crash before score persist",
    ):
        start_test_workflow(
            run_score_generation_workflow,
            workflow_id,
            app_postgres_schema.database_url,
            run.generation_run_id,
        )

    persist_state["fail"] = False

    assert (
        count_score_attempts(
            app_postgres_schema.database_url,
            score_attempt_id,
        )
        == 0
    )
    presence = classify_scoring_workflow_presence(
        database_url=app_postgres_schema.database_url,
        score_attempt_id=score_attempt_id,
        workflow_id=workflow_id,
    )
    assert presence is ScoringWorkflowPresence.ORPHAN

    scheduled = scoring_workflow.schedule_score_generation_workflow(
        app_postgres_schema.database_url,
        run.generation_run_id,
        recover_orphans=False,
    )
    assert scheduled.scheduled is False
    assert scheduled.recovered is False

    recovered = scoring_workflow.schedule_score_generation_workflow(
        app_postgres_schema.database_url,
        run.generation_run_id,
        recover_orphans=True,
    )
    assert recovered.scheduled is True
    assert recovered.recovered is True
    assert (
        count_score_attempts(
            app_postgres_schema.database_url,
            score_attempt_id,
        )
        == 1
    )
    snapshot = fetch_score_attempt_snapshot(
        app_postgres_schema.database_url,
        score_attempt_id,
    )
    assert snapshot is not None
    assert snapshot.status == "success"
