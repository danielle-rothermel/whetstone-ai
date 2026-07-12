"""Fresh-Postgres acceptance and scoring-selection regressions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from dr_platform import PlatformSchema, SubmitOptions, submit
from dr_platform.enqueue_runtime import (
    PhysicalEnqueueDisposition,
    PhysicalEnqueueOutcome,
)
from sqlalchemy import create_engine, insert, select, update
from typer.testing import CliRunner

from whetstone.db import schema
from whetstone.platform import operations
from whetstone.platform.acceptance import (
    AcceptanceDisposition,
    CurrentAcceptanceDisposition,
    GenerationMembershipConflictError,
    RequiredScoringProfile,
    evaluate_strict_acceptance,
    load_acceptance,
    load_current_acceptance,
)
from whetstone.platform.spec_builder import direct_graph, prediction_spec
from whetstone.platform.submission import (
    ScoringTargetSpec,
    prepare_generation_manifest,
    prepare_scoring_manifest,
    select_populated_scoring_generation_runs,
)
from whetstone.platform.targets import target_registry
from whetstone.records import DatasetSnapshotIdentityPayload


class _QueueLookup:
    def retrieve_queue(self, name: str) -> object:
        del name
        return type(
            "QueueConfiguration",
            (),
            {"database_backed_queue": True, "priority_enabled": True},
        )()


class _EnqueueAdapter:
    def enqueue(self, call: Any) -> PhysicalEnqueueOutcome:
        return PhysicalEnqueueOutcome(
            workflow_id=call.workflow_id,
            disposition=PhysicalEnqueueDisposition.ENQUEUED,
            effective_service_priority=call.service_priority,
        )


def _submit_generation(
    engine: Any, *, operation_key: str, specs: tuple[Any, ...]
) -> None:
    manifest, source = prepare_generation_manifest(
        operation_key=operation_key,
        experiment_name="exp",
        specs=specs,
        options=SubmitOptions(page_size=1),
    )
    submit(
        manifest,
        source,
        engine=engine,
        resolver=target_registry(),
        options=SubmitOptions(page_size=1),
        schema=PlatformSchema(prefix="whetstone"),
        queue_lookup=_QueueLookup(),
        enqueue_adapter=_EnqueueAdapter(),
    )


def _target(
    prediction_id: str,
    generation_run_id: str,
    dataset_snapshot: dict[str, Any],
) -> ScoringTargetSpec:
    return ScoringTargetSpec(
        prediction_id=prediction_id,
        generation_run_id=generation_run_id,
        scoring_profile_id="humaneval",
        scoring_profile_version="v1",
        parser_profile_id="humaneval-best-effort",
        parser_version="v1",
        dataset_name=str(dataset_snapshot["header"]["dataset_id"]),
        dataset_split="test",
        dataset_snapshot=DatasetSnapshotIdentityPayload.model_validate(
            dataset_snapshot
        ),
    )


def _submit_scoring(
    engine: Any,
    *,
    operation_key: str,
    targets: tuple[ScoringTargetSpec, ...],
) -> None:
    manifest, source, selection_digest = prepare_scoring_manifest(
        operation_key=operation_key,
        experiment_name="exp",
        targets=targets,
    )
    submit(
        manifest,
        source,
        engine=engine,
        resolver=target_registry(),
        spec={
            "experiment_name": "exp",
            "source_generation_operation_key": "generation",
            "selection_digest": selection_digest,
        },
        schema=PlatformSchema(prefix="whetstone"),
        queue_lookup=_QueueLookup(),
        enqueue_adapter=_EnqueueAdapter(),
    )


def _terminalize_operation(connection: Any, operation_key: str) -> None:
    now = datetime.now(UTC)
    platform = PlatformSchema(prefix="whetstone")
    item_ids = list(
        connection.execute(
            select(platform.items.c.item_id).where(
                platform.items.c.operation_key == operation_key
            )
        ).scalars()
    )
    connection.execute(
        update(platform.item_attempts)
        .where(platform.item_attempts.c.item_id.in_(item_ids))
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
            succeeded_count=len(item_ids),
            completed_at=now,
            updated_at=now,
            platform_cut_version=platform.operations.c.platform_cut_version
            + 1,
        )
    )


def _insert_generation_run(
    connection: Any,
    *,
    prediction_id: str,
    generation_run_id: str,
    platform_item_id: str,
    platform_attempt: int,
) -> None:
    now = datetime.now(UTC)
    connection.execute(
        insert(schema.generation_runs).values(
            generation_run_id=generation_run_id,
            prediction_id=prediction_id,
            attempt_index=platform_attempt,
            execution_recipe_digest=f"recipe-{platform_attempt}",
            platform_item_id=platform_item_id,
            platform_attempt=platform_attempt,
            status="success",
            terminal_node_id="terminal",
            terminal_output_node_id="terminal",
            summary={
                "execution_order": ["terminal"],
                "terminal_node_id": "terminal",
                "terminal_submission_text": "return 1",
            },
            started_at=now,
            completed_at=now,
        )
    )


def _insert_score_attempt(
    connection: Any,
    *,
    target: ScoringTargetSpec,
    platform_item_id: str,
) -> None:
    now = datetime.now(UTC)
    connection.execute(
        insert(schema.score_attempts).values(
            score_attempt_id=f"score-{target.generation_run_id}",
            prediction_id=target.prediction_id,
            generation_run_id=target.generation_run_id,
            attempt_index=0,
            execution_recipe_digest=f"recipe-{target.generation_run_id}",
            platform_item_id=platform_item_id,
            platform_attempt=0,
            scoring_profile_id=target.scoring_profile_id,
            scoring_profile_version=target.scoring_profile_version,
            parser_profile_id=target.parser_profile_id,
            parser_version=target.parser_version,
            dataset_name=target.dataset_name,
            dataset_split=target.dataset_split,
            dataset_snapshot=target.dataset_snapshot.model_dump(mode="json"),
            status="success",
            submission_outcome="passed",
            score=1.0,
            extracted_submission={},
            metrics={},
            per_test_results=[],
            started_at=now,
            completed_at=now,
        )
    )


@pytest.mark.integration
def test_conflicting_generation_registration_rolls_back_specs(
    app_postgres_schema: Any,
) -> None:
    first = prediction_spec(direct_graph(), experiment_name="exp")
    second = prediction_spec(
        direct_graph(), experiment_name="exp", task_id="HumanEval/1"
    )
    _submit_generation(
        app_postgres_schema.engine,
        operation_key="generation",
        specs=(first,),
    )
    with app_postgres_schema.engine.connect() as connection:
        source_version = connection.execute(
            select(schema.experiments.c.acceptance_source_version)
        ).scalar_one()

    with pytest.raises(GenerationMembershipConflictError):
        _submit_generation(
            app_postgres_schema.engine,
            operation_key="generation-conflict",
            specs=(second,),
        )

    with app_postgres_schema.engine.connect() as connection:
        assert (
            connection.execute(
                select(schema.prediction_specs.c.prediction_id).where(
                    schema.prediction_specs.c.prediction_id
                    == second.prediction_id
                )
            ).scalar_one_or_none()
            is None
        )
        assert (
            connection.execute(
                select(schema.experiments.c.acceptance_source_version)
            ).scalar_one()
            == source_version
        )


@pytest.mark.integration
def test_scoring_registration_rejects_forged_recipe_axes(
    app_postgres_schema: Any,
) -> None:
    spec = prediction_spec(direct_graph(), experiment_name="exp")
    snapshot = {
        "sha256": "a" * 64,
        "header": {
            "schema_version": 1,
            "dataset_id": "evalplus/humanevalplus",
            "hf_revision": "frozen",
            "overrides_digest": "b" * 64,
        },
    }
    spec.task.metadata["dataset_snapshot"] = snapshot
    _submit_generation(
        app_postgres_schema.engine,
        operation_key="generation",
        specs=(spec,),
    )
    platform = PlatformSchema(prefix="whetstone")
    with app_postgres_schema.engine.begin() as connection:
        generation_item = connection.execute(
            select(platform.items.c.item_id).where(
                platform.items.c.operation_key == "generation"
            )
        ).scalar_one()
        _insert_generation_run(
            connection,
            prediction_id=spec.prediction_id,
            generation_run_id="run",
            platform_item_id=generation_item,
            platform_attempt=0,
        )
    forged = _target(spec.prediction_id, "run", snapshot).model_copy(
        update={"parser_profile_id": "forged-parser"}
    )
    with pytest.raises(
        ValueError,
        match="profile/parser axes do not match",
    ):
        _submit_scoring(
            app_postgres_schema.engine,
            operation_key="scoring-forged",
            targets=(forged,),
        )


@pytest.mark.integration
def test_selection_candidates_cut_current_read_and_cli(
    app_postgres_schema: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = prediction_spec(direct_graph(), experiment_name="exp")
    spec.task.metadata["dataset_snapshot"] = {
        "sha256": "a" * 64,
        "header": {
            "schema_version": 1,
            "dataset_id": "evalplus/humanevalplus",
            "hf_revision": "frozen",
            "overrides_digest": "b" * 64,
        },
    }
    _submit_generation(
        app_postgres_schema.engine,
        operation_key="generation",
        specs=(spec,),
    )
    platform = PlatformSchema(prefix="whetstone")
    with app_postgres_schema.engine.begin() as connection:
        generation_item = connection.execute(
            select(platform.items.c.item_id).where(
                platform.items.c.operation_key == "generation"
            )
        ).scalar_one()
        _terminalize_operation(connection, "generation")
        _insert_generation_run(
            connection,
            prediction_id=spec.prediction_id,
            generation_run_id="run-0",
            platform_item_id=generation_item,
            platform_attempt=0,
        )
        attempt_zero = dict(
            connection.execute(
                select(platform.item_attempts).where(
                    platform.item_attempts.c.item_id == generation_item
                )
            )
            .mappings()
            .one()
        )
        attempt_zero.pop("change_seq")
        retry_time = datetime.now(UTC)
        attempt_one = {
            **attempt_zero,
            "attempt": 1,
            "execution_key": "generation-execution-1",
            "workflow_id": "generation-workflow-1",
            "source_attempt": 0,
            "source_workflow_id": attempt_zero["workflow_id"],
            "retry_reason": "domain_outcome",
            "created_at": retry_time,
            "enqueued_at": retry_time,
            "updated_at": retry_time,
            "terminal_at": retry_time,
        }
        connection.execute(insert(platform.item_attempts).values(attempt_one))
        connection.execute(
            update(platform.items)
            .where(platform.items.c.item_id == generation_item)
            .values(current_attempt=1)
        )
        connection.execute(
            update(platform.operations)
            .where(platform.operations.c.operation_key == "generation")
            .values(
                platform_cut_version=platform.operations.c.platform_cut_version
                + 1
            )
        )
        _insert_generation_run(
            connection,
            prediction_id=spec.prediction_id,
            generation_run_id="run-1",
            platform_item_id=generation_item,
            platform_attempt=1,
        )

    with app_postgres_schema.engine.connect() as connection:
        selected = select_populated_scoring_generation_runs(
            connection, experiment_name="exp"
        )
    assert [run.generation_run_id for run in selected] == ["run-1"]

    targets = (
        _target(
            spec.prediction_id,
            "run-0",
            spec.task.metadata["dataset_snapshot"],
        ),
        _target(
            spec.prediction_id,
            "run-1",
            spec.task.metadata["dataset_snapshot"],
        ),
    )
    _submit_scoring(
        app_postgres_schema.engine,
        operation_key="scoring",
        targets=targets,
    )
    profile = RequiredScoringProfile(
        scoring_profile_id=targets[0].scoring_profile_id,
        scoring_profile_version=targets[0].scoring_profile_version,
        parser_profile_id=targets[0].parser_profile_id,
        parser_version=targets[0].parser_version,
        dataset_name=targets[0].dataset_name,
        dataset_split=targets[0].dataset_split,
    )
    with app_postgres_schema.engine.begin() as connection:
        scoring_items = {
            row["item_key"]: row["item_id"]
            for row in connection.execute(
                select(platform.items).where(
                    platform.items.c.operation_key == "scoring"
                )
            ).mappings()
        }
        for target in targets:
            _insert_score_attempt(
                connection,
                target=target,
                platform_item_id=scoring_items[target.item_key],
            )
        nonterminal = evaluate_strict_acceptance(
            connection,
            experiment_name="exp",
            required_profiles=(profile,),
        )
    assert (
        nonterminal.disposition is AcceptanceDisposition.EXECUTION_NOT_TERMINAL
    )

    with app_postgres_schema.engine.begin() as connection:
        _terminalize_operation(connection, "scoring")
        accepted = evaluate_strict_acceptance(
            connection,
            experiment_name="exp",
            required_profiles=(profile,),
        )
        assert accepted.disposition is AcceptanceDisposition.PROMOTED
        assert accepted.acceptance_id is not None
        generation_candidates = (
            connection.execute(
                select(
                    schema.experiment_acceptance_generation_candidates.c.disposition
                ).where(
                    schema.experiment_acceptance_generation_candidates.c.acceptance_id
                    == accepted.acceptance_id
                )
            )
            .scalars()
            .all()
        )
        scoring_candidates = (
            connection.execute(
                select(
                    schema.experiment_acceptance_scoring_candidates.c.disposition
                ).where(
                    schema.experiment_acceptance_scoring_candidates.c.acceptance_id
                    == accepted.acceptance_id
                )
            )
            .scalars()
            .all()
        )
        historical = load_acceptance(
            connection, acceptance_id=accepted.acceptance_id
        )
        current = load_current_acceptance(connection, experiment_name="exp")
    assert sorted(generation_candidates) == ["selected", "superseded_success"]
    assert sorted(scoring_candidates) == ["selected", "superseded_generation"]
    assert historical.platform_cut
    assert current.disposition is CurrentAcceptanceDisposition.CURRENT

    monkeypatch.setattr(
        operations,
        "_engine",
        lambda: create_engine(app_postgres_schema.database_url),
    )
    runner = CliRunner()
    shown = runner.invoke(
        operations.APP,
        ["show-current", "exp", "--json"],
    )
    assert shown.exit_code == 0, shown.output
    assert '"disposition":"current"' in shown.output
    history = runner.invoke(
        operations.APP,
        ["show-acceptance", accepted.acceptance_id, "--json"],
    )
    assert history.exit_code == 0, history.output
    evaluated = runner.invoke(
        operations.APP,
        [
            "evaluate",
            "exp",
            "humaneval",
            "v1",
            "humaneval-best-effort",
            "v1",
            targets[0].dataset_name,
            "test",
            "--json",
        ],
    )
    assert evaluated.exit_code == 0, evaluated.output

    with app_postgres_schema.engine.begin() as connection:
        connection.execute(
            update(platform.operations)
            .where(platform.operations.c.operation_key == "scoring")
            .values(
                platform_cut_version=platform.operations.c.platform_cut_version
                + 1
            )
        )
        stale = load_current_acceptance(connection, experiment_name="exp")
    assert stale.disposition is CurrentAcceptanceDisposition.STALE_PLATFORM_CUT
