"""Runner cell pathway: lifecycle, viewer projection, and CLI smoke."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from tests.optimization.support import candidate, make_store
from tests.provider import support as provider_support
from tests.runner.support import (
    BASELINE_TEMPLATE,
    cell_config,
    official_binding,
    official_engine,
)
from whetstone.coordination import proposal_provider, run_workflow
from whetstone.envs.generation_graph import PromptInputError
from whetstone.optimization.contracts import StepStatus
from whetstone.optimization.gepa import runner as gepa_runner
from whetstone.optimization.proposal.proposer import (
    ProviderProposerTransport,
)
from whetstone.runner.budget import (
    BudgetGuard,
    CreditsSnapshot,
    ReserveError,
    StopLossError,
)
from whetstone.runner.cell import (
    CELL_RUN_CONTROL_SCHEMA,
    OFFICIAL_ARM_BINDING_SCHEMA,
    CellError,
    _check_cell_start,
    bind_cell_launch,
    prepare_cell_launch,
    run_cell,
)
from whetstone.runner.cli import (
    RunnerLaunch,
    app,
    run_cell_command,
)
from whetstone.runner.events import EventStream
from whetstone.runner.ledger import (
    COMPLETED_CELL_STATUSES,
    Ledger,
    SpendRecord,
)
from whetstone.runner.viewer_projection import (
    VIEWER_GENERATION_ROW_SCHEMA,
    VIEWER_PROJECTION_SCHEMA,
    ViewerCellProjection,
    build_viewer_cell_projection,
)


@pytest.fixture(autouse=True)
def _isolated_registries() -> Iterator[None]:
    transport_registry = proposal_provider._TRANSPORT_REGISTRY
    with transport_registry._lock:
        transports = dict(transport_registry._transports)
    controllers = dict(run_workflow._CONTROLLERS)
    factories = dict(gepa_runner._GEPA_FACTORIES)
    with transport_registry._lock:
        transport_registry._transports.clear()
    run_workflow._CONTROLLERS.clear()
    gepa_runner._GEPA_FACTORIES.clear()
    yield
    with transport_registry._lock:
        transport_registry._transports.clear()
        transport_registry._transports.update(transports)
    run_workflow._CONTROLLERS.clear()
    run_workflow._CONTROLLERS.update(controllers)
    gepa_runner._GEPA_FACTORIES.clear()
    gepa_runner._GEPA_FACTORIES.update(factories)


def _transport() -> ProviderProposerTransport:
    provider_config = provider_support.openrouter_chat_config(
        model="proposal-model"
    )
    transport_policy = provider_support.build_transport_policy()
    return ProviderProposerTransport(
        resolve_provider_call_config=lambda _ref: provider_config,
        transport=provider_support.RecordingTransport(
            request=provider_support.build_request(),
            transport_policy=transport_policy,
            outcomes=[],
        ),
        execution_policy=provider_support.build_execution_policy(
            max_attempts=1, transport_policy=transport_policy
        ),
        clock=provider_support.FakeClock(),
        sleep=provider_support.SleepRecorder(),
    )


def _published(tmp_path: Path):
    outcome = run_cell(cell_config(tmp_path))
    publication = outcome.record.artifacts.viewer_publication
    assert publication is not None
    return outcome, publication, tmp_path / "ledger"


def test_completed_cell_is_skipped_without_rerunning(tmp_path: Path) -> None:
    first = run_cell(cell_config(tmp_path))
    assert not first.skipped

    stream = EventStream(tmp_path / "run")
    second_config = cell_config(tmp_path, event_stream=stream)
    second = run_cell(second_config)

    assert second.skipped
    assert second.result is None
    assert second.record == first.record
    events = [
        json.loads(line)
        for line in stream.path.read_text().splitlines()
        if line
    ]
    assert [event["event"] for event in events] == ["attempt_skipped"]


def test_official_arms_bind_before_they_are_paid_for(tmp_path: Path) -> None:
    config = cell_config(tmp_path)
    bind_cell_launch(config)

    for arm in ("baseline", "ceiling"):
        bound = config.store.resolve(
            f"{OFFICIAL_ARM_BINDING_SCHEMA}:{config.cell_id}#{arm}"
        )
        assert bound is not None
    assert (
        config.store.resolve(f"{CELL_RUN_CONTROL_SCHEMA}:{config.cell_id}")
        is not None
    )


def test_rebinding_a_changed_official_arm_is_refused(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ledger = Ledger(tmp_path / "ledger")
    first = cell_config(tmp_path, store=store, ledger=ledger)
    bind_cell_launch(first)

    engine = official_engine(store)
    changed = cell_config(
        tmp_path,
        store=store,
        ledger=ledger,
        official_engine=engine,
        official_evaluation_binding=official_binding(engine),
        baseline=candidate("baseline", text=BASELINE_TEMPLATE + " now"),
    )
    with pytest.raises(CellError, match="already bound"):
        bind_cell_launch(changed)


def test_cell_identity_must_match_its_run_control(tmp_path: Path) -> None:
    with pytest.raises(CellError, match="exact cell identity"):
        cell_config(tmp_path, env="other-env")


def test_spend_checkpoints_precede_every_paid_boundary(
    tmp_path: Path,
) -> None:
    snapshots = iter(
        [
            CreditsSnapshot(total_credits=100.0, total_usage=0.0),
            CreditsSnapshot(total_credits=100.0, total_usage=1.0),
            CreditsSnapshot(total_credits=100.0, total_usage=2.0),
            CreditsSnapshot(total_credits=100.0, total_usage=3.0),
            CreditsSnapshot(total_credits=100.0, total_usage=4.0),
            CreditsSnapshot(total_credits=100.0, total_usage=5.0),
        ]
    )
    outcome = run_cell(
        cell_config(
            tmp_path,
            credits_fetcher=lambda: next(snapshots),
            budget_guard=BudgetGuard(
                reserve_usd=0.0, expected_cell_usd=1000.0
            ),
        )
    )
    ledger = Ledger(tmp_path / "ledger")
    phases = [record.phase for record in ledger.spend_records()]

    assert phases[0] == "before"
    assert phases[-1] == "after"
    for arm in ("baseline", "ceiling", "best"):
        assert f"checkpoint:official:{arm}" in phases
    assert "checkpoint:optimization" in phases
    assert phases.index("checkpoint:official:baseline") < phases.index(
        "checkpoint:official:best"
    )
    assert outcome.record.spend_usd == pytest.approx(5.0)


def test_stop_loss_halts_before_the_next_paid_arm(tmp_path: Path) -> None:
    snapshots = iter(
        [
            CreditsSnapshot(total_credits=100.0, total_usage=0.0),
            CreditsSnapshot(total_credits=100.0, total_usage=0.0),
            CreditsSnapshot(total_credits=100.0, total_usage=99.0),
            CreditsSnapshot(total_credits=100.0, total_usage=99.0),
        ]
    )
    config = cell_config(
        tmp_path,
        credits_fetcher=lambda: next(snapshots),
        budget_guard=BudgetGuard(reserve_usd=0.0, expected_cell_usd=1.0),
    )
    with pytest.raises(StopLossError):
        run_cell(config)

    assert Ledger(tmp_path / "ledger").load() == []


def test_a_stop_loss_closes_its_spend_pair(tmp_path: Path) -> None:
    snapshots = iter(
        [
            CreditsSnapshot(total_credits=100.0, total_usage=0.0),
            CreditsSnapshot(total_credits=100.0, total_usage=0.0),
            CreditsSnapshot(total_credits=100.0, total_usage=99.0),
            CreditsSnapshot(total_credits=100.0, total_usage=99.0),
        ]
    )
    config = cell_config(
        tmp_path,
        credits_fetcher=lambda: next(snapshots),
        budget_guard=BudgetGuard(reserve_usd=0.0, expected_cell_usd=1.0),
    )
    with pytest.raises(StopLossError):
        run_cell(config)

    phases = [
        record.phase for record in Ledger(tmp_path / "ledger").spend_records()
    ]
    assert phases[0] == "before"
    assert phases[-1] == "after"
    total, gaps = Ledger(tmp_path / "ledger").spend_for_cell(config.cell_id)
    assert total == pytest.approx(99.0)
    assert gaps == []


def test_a_stopped_out_cell_can_run_again(tmp_path: Path) -> None:
    tripping = iter(
        [
            CreditsSnapshot(total_credits=100.0, total_usage=0.0),
            CreditsSnapshot(total_credits=100.0, total_usage=0.0),
            CreditsSnapshot(total_credits=100.0, total_usage=99.0),
            CreditsSnapshot(total_credits=100.0, total_usage=99.0),
        ]
    )
    first = cell_config(
        tmp_path,
        credits_fetcher=lambda: next(tripping),
        budget_guard=BudgetGuard(reserve_usd=0.0, expected_cell_usd=1.0),
    )
    with pytest.raises(StopLossError):
        run_cell(first)
    assert Ledger(tmp_path / "ledger").load() == []

    healthy = CreditsSnapshot(total_credits=100.0, total_usage=99.0)
    second = run_cell(
        cell_config(
            tmp_path,
            store=first.store,
            ledger=Ledger(tmp_path / "ledger"),
            credits_fetcher=lambda: healthy,
            budget_guard=BudgetGuard(reserve_usd=0.0, expected_cell_usd=1.0),
        )
    )

    assert not second.skipped
    assert second.record.status in COMPLETED_CELL_STATUSES


def test_a_later_attempt_is_not_a_rerun_for_the_reserve_guard(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    run_cell(cell_config(tmp_path, store=store, ledger=Ledger(tmp_path / "l")))

    low = CreditsSnapshot(total_credits=10.0, total_usage=0.0)
    with pytest.raises(ReserveError, match="reserve"):
        run_cell(
            cell_config(
                tmp_path,
                attempt=1,
                canonical=True,
                store=store,
                ledger=Ledger(tmp_path / "l"),
                credits_fetcher=lambda: low,
                budget_guard=BudgetGuard(reserve_usd=50.0),
            )
        )


def test_a_rerun_of_this_exact_attempt_escapes_the_reserve(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    run_cell(cell_config(tmp_path, store=store, ledger=Ledger(tmp_path / "l")))

    low = CreditsSnapshot(total_credits=10.0, total_usage=0.0)
    rerun = cell_config(
        tmp_path,
        attempt=0,
        canonical=True,
        store=store,
        ledger=Ledger(tmp_path / "l"),
        credits_fetcher=lambda: low,
        budget_guard=BudgetGuard(reserve_usd=50.0),
    )
    initial = SpendRecord(
        cell_id=rerun.cell_id,
        phase="before",
        lane=rerun.lane,
        remaining_usd=10.0,
    )

    _check_cell_start(rerun, initial)


def test_a_completed_cell_with_changed_controls_is_refused(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    first = cell_config(tmp_path, store=store, ledger=Ledger(tmp_path / "l"))
    run_cell(first)

    engine = official_engine(store)
    changed = cell_config(
        tmp_path,
        store=store,
        ledger=Ledger(tmp_path / "l"),
        official_engine=engine,
        official_evaluation_binding=official_binding(engine),
        baseline=candidate("baseline", text=BASELINE_TEMPLATE + " changed"),
    )

    with pytest.raises(CellError, match="changed controls"):
        prepare_cell_launch(changed)


def test_a_completed_cell_with_unchanged_controls_still_skips(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    first = cell_config(tmp_path, store=store, ledger=Ledger(tmp_path / "l"))
    run_cell(first)

    engine = official_engine(store)
    same = cell_config(
        tmp_path,
        store=store,
        ledger=Ledger(tmp_path / "l"),
        official_engine=engine,
        official_evaluation_binding=official_binding(engine),
    )
    outcome = prepare_cell_launch(same)

    assert outcome is not None
    assert outcome.skipped


def test_preflight_rejects_a_candidate_the_env_cannot_render(
    tmp_path: Path,
) -> None:
    config = cell_config(
        tmp_path, baseline=candidate("baseline", text="{nonexistent}")
    )
    with pytest.raises(PromptInputError):
        prepare_cell_launch(config)


def test_cell_without_a_ceiling_reports_no_headroom(tmp_path: Path) -> None:
    outcome = run_cell(cell_config(tmp_path, ceiling=False))
    record = outcome.record

    assert record.ceiling_official is None
    assert record.ceiling_ci95 is None
    assert record.headroom_delta is None
    assert record.headroom_ci95 is None
    assert record.status in COMPLETED_CELL_STATUSES
    assert not (tmp_path / "ledger" / "official_anchors").exists()


def test_official_anchor_projects_both_reference_arms(tmp_path: Path) -> None:
    outcome = run_cell(cell_config(tmp_path))
    anchor_path = Ledger(tmp_path / "ledger").official_anchor_path(
        outcome.record.cell_id
    )
    anchor = json.loads(anchor_path.read_text())

    assert anchor["cell_id"] == outcome.record.cell_id
    assert anchor["baseline_official"] == outcome.record.baseline_official
    assert anchor["ceiling_official"] == outcome.record.ceiling_official
    assert len(anchor["official_task_ids"]) == len(anchor["baseline_per_task"])


def test_a_completed_line_always_cites_committed_viewer_files(
    tmp_path: Path,
) -> None:
    outcome = run_cell(cell_config(tmp_path))
    record = outcome.record
    assert record.is_completed()
    publication = record.artifacts.viewer_publication
    assert publication is not None

    root = tmp_path / "ledger"
    for reference in (publication.projection, publication.generation_outputs):
        assert (root / reference.relative_path).is_file()


def test_projection_bytes_match_their_recorded_hashes(tmp_path: Path) -> None:
    _outcome, publication, root = _published(tmp_path)

    for reference in (publication.projection, publication.generation_outputs):
        body = (root / reference.relative_path).read_bytes()
        assert hashlib.sha256(body).hexdigest() == reference.sha256


def test_projection_reports_every_official_arm(tmp_path: Path) -> None:
    outcome, publication, root = _published(tmp_path)
    projection = json.loads(
        (root / publication.projection.relative_path).read_text()
    )
    lines = [
        json.loads(line)
        for line in (root / publication.generation_outputs.relative_path)
        .read_text()
        .splitlines()
        if line
    ]

    assert projection["schema"] == VIEWER_PROJECTION_SCHEMA
    assert projection["cell_id"] == outcome.record.cell_id
    assert len(projection["evidence_summaries"]) == 3
    assert projection["generation_row_count"] == len(lines)
    assert {
        summary["purpose"] for summary in projection["evidence_summaries"]
    } == {
        "official_baseline",
        "official_ceiling",
        "official_best",
    }
    for row in lines:
        assert row["schema"] == VIEWER_GENERATION_ROW_SCHEMA
        assert row["cell_id"] == outcome.record.cell_id
        assert row["task_hash"]


def test_projection_composes_every_ordered_step(tmp_path: Path) -> None:
    outcome, publication, root = _published(tmp_path)
    projection = json.loads(
        (root / publication.projection.relative_path).read_text()
    )

    indices = [step["step_index"] for step in projection["steps"]]
    assert indices == list(range(len(indices)))
    assert projection["terminal_status"] == projection["steps"][-1]["status"]
    assert projection["optimization_result_ref"] == json.loads(
        outcome.record.artifacts.optimization_result_ref.model_dump_json()
    )


def test_projection_serialization_is_canonical_and_stable(
    tmp_path: Path,
) -> None:
    outcome, publication, root = _published(tmp_path)
    body = (root / publication.projection.relative_path).read_bytes()
    reparsed = ViewerCellProjection.model_validate_json(body)

    assert reparsed.to_bytes() == body
    assert body.endswith(b"\n")
    assert b", " not in body
    assert reparsed.cell_id == outcome.record.cell_id


def test_projection_refuses_a_misaligned_cell_identity(
    tmp_path: Path,
) -> None:
    _outcome, publication, root = _published(tmp_path)
    payload = json.loads(
        (root / publication.projection.relative_path).read_text()
    )
    payload["attempt"] = payload["attempt"] + 1

    with pytest.raises(ValidationError, match="identity fields do not align"):
        ViewerCellProjection.model_validate(payload)


def test_projection_refuses_a_terminal_status_that_contradicts_its_steps(
    tmp_path: Path,
) -> None:
    _outcome, publication, root = _published(tmp_path)
    payload = json.loads(
        (root / publication.projection.relative_path).read_text()
    )
    payload["terminal_status"] = StepStatus.FAILED.value

    with pytest.raises(ValidationError, match="terminal_status must match"):
        ViewerCellProjection.model_validate(payload)


def test_projection_refuses_a_result_from_another_cell(
    tmp_path: Path,
) -> None:
    config = cell_config(tmp_path)
    outcome = run_cell(config)
    assert outcome.result is not None
    result_ref = outcome.record.artifacts.optimization_result_ref
    assert result_ref is not None

    with pytest.raises(ValueError, match="does not belong to this cell"):
        build_viewer_cell_projection(
            cell_id="identity:c18:a7",
            optimizer="identity",
            env="c18",
            attempt=7,
            result=outcome.result,
            result_ref=result_ref,
            store=config.store,
            official_evaluations=(),
        )


def test_a_completed_cell_never_constructs_a_dbos_runtime(
    tmp_path: Path,
) -> None:
    first = run_cell(cell_config(tmp_path))
    assert first.record.is_completed()

    launch = RunnerLaunch(cell=cell_config(tmp_path), transport=_transport())
    outcome = run_cell_command(
        launch,
        system_database_url="postgresql://invalid:1/nonexistent",
        environ={},
    )

    assert outcome.skipped
    assert outcome.record == first.record


def test_status_prints_every_validated_ledger_line(tmp_path: Path) -> None:
    outcome = run_cell(cell_config(tmp_path))
    result = CliRunner().invoke(
        app, ["status", "--root", str(tmp_path / "ledger")]
    )

    assert result.exit_code == 0
    records = json.loads(result.stdout)
    assert [record["cell_id"] for record in records] == [
        outcome.record.cell_id
    ]


def test_refinalize_reports_an_unchanged_correct_line(tmp_path: Path) -> None:
    outcome = run_cell(cell_config(tmp_path))
    result = CliRunner().invoke(
        app,
        [
            "refinalize",
            "--root",
            str(tmp_path / "ledger"),
            "--optimizer",
            "identity",
            "--env",
            outcome.record.env,
            "--attempt",
            "0",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["changed"] is False
    assert payload["record"]["cell_id"] == outcome.record.cell_id
    assert Ledger(tmp_path / "ledger").load()[-1].status == (
        outcome.record.status
    )
