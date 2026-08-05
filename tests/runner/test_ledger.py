"""Ledger schema, durability, and spend-pairing tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from whetstone.core.identity import TypedRef
from whetstone.evaluation.schema_names import EVALUATION_EVIDENCE_SCHEMA
from whetstone.runner.ledger import (
    CELL_STATUSES,
    CELLS_SCHEMA,
    COMPLETED_CELL_STATUSES,
    OFFICIAL_ANCHOR_SCHEMA,
    SPEND_SCHEMA,
    CellArtifacts,
    CellModels,
    CellRecord,
    Ledger,
    OfficialAnchorRecord,
    SpendRecord,
    ViewerCellPublicationRef,
    ViewerPublishedFileRef,
    cell_key,
)

_HASH = "a" * 64
_OTHER_HASH = "b" * 64


def _publication(cell_id: str = "copro__c18__a0") -> ViewerCellPublicationRef:
    return ViewerCellPublicationRef(
        projection=ViewerPublishedFileRef(
            relative_path=f"viewer_cells/{cell_id}/projection.json",
            sha256=_HASH,
        ),
        rollout_outputs=ViewerPublishedFileRef(
            relative_path=f"viewer_cells/{cell_id}/rollout_outputs.jsonl",
            sha256=_OTHER_HASH,
        ),
    )


def _cell(
    *,
    status: str = "no-improvement",
    cell_id: str = "copro:c18:a0",
    optimizer: str = "copro",
    env: str = "c18",
    attempt: int = 0,
    **overrides: object,
) -> CellRecord:
    """Build a valid cell record, then re-validate it with any overrides.

    Overrides are applied to the serialized form and revalidated, so a helper
    override exercises exactly the wire-level validation a real line does.
    """
    base = CellRecord(
        cell_id=cell_id,
        optimizer=optimizer,
        env=env,
        attempt=attempt,
        canonical=True,
        models=CellModels(task="t", proposer="p"),
        baseline_official=0.5,
        ceiling_official=0.9,
        best_official=0.5,
        delta=0.0,
        internal_evals_count=4,
        optimizer_steps=2,
        spend_usd=0.25,
        wall_s=12.5,
        lane="openrouter",
        status=status,
        artifacts=(
            CellArtifacts(
                viewer_publication=_publication(
                    f"{optimizer}__{env}__a{attempt}"
                )
            )
            if status in COMPLETED_CELL_STATUSES
            else CellArtifacts()
        ),
    )
    if not overrides:
        return base
    payload = base.model_dump(mode="json", by_alias=True)
    for name, value in overrides.items():
        payload[name] = (
            value.model_dump(mode="json")
            if isinstance(value, CellArtifacts)
            else value
        )
    return CellRecord.model_validate(payload)


def _evidence_ref(content_hash: str = _HASH) -> TypedRef:
    return TypedRef(
        schema_name=EVALUATION_EVIDENCE_SCHEMA, content_hash=content_hash
    )


def _anchor(**overrides: object) -> OfficialAnchorRecord:
    """Build a valid anchor, then re-validate it with any overrides."""
    base = OfficialAnchorRecord(
        cell_id="copro:c18:a0",
        env="c18",
        task_model="openai/gpt-5-nano",
        graph_hash=_HASH,
        eval_config_hash=_OTHER_HASH,
        official_instance_ids=("i0", "i1"),
        official_task_identities=("c" * 64, "d" * 64),
        baseline_evidence_ref=_evidence_ref(),
        ceiling_evidence_ref=_evidence_ref(_OTHER_HASH),
        baseline_official=0.5,
        ceiling_official=0.9,
        baseline_per_task=(0.4, 0.6),
        ceiling_per_task=(0.8, 1.0),
        baseline_per_task_counts=(2, 2),
        ceiling_per_task_counts=(2, 2),
        official_repeats_used=2,
    )
    if not overrides:
        return base
    payload = base.model_dump(mode="json", by_alias=True)
    for name, value in overrides.items():
        payload[name] = (
            value.model_dump(mode="json")
            if isinstance(value, TypedRef)
            else value
        )
    return OfficialAnchorRecord.model_validate(payload)


# --------------------------------------------------------------------------
# Golden literal pins (no-magic-strings rule): these exact wire strings are
# the persisted format. A rename here silently orphans every recorded line.
# --------------------------------------------------------------------------


def test_schema_literals_are_pinned() -> None:
    assert CELLS_SCHEMA == "whetstone.runner.cells/v1"
    assert SPEND_SCHEMA == "whetstone.runner.spend/v1"
    assert OFFICIAL_ANCHOR_SCHEMA == "whetstone.runner.official_anchor/v1"


def test_cell_statuses_are_pinned() -> None:
    assert CELL_STATUSES == frozenset(
        {
            "improved",
            "inconclusive",
            "no-improvement",
            "plumbing-retry",
            "halted",
            "incomplete-arm",
            "proposer-failure",
        }
    )
    assert COMPLETED_CELL_STATUSES == frozenset(
        {"improved", "inconclusive", "no-improvement", "halted"}
    )
    assert COMPLETED_CELL_STATUSES < CELL_STATUSES


def test_cell_record_wire_keys_are_pinned() -> None:
    payload = json.loads(_cell().to_line())

    assert set(payload) == {
        "schema",
        "cell_id",
        "optimizer",
        "env",
        "attempt",
        "canonical",
        "models",
        "baseline_official",
        "ceiling_official",
        "best_official",
        "delta",
        "delta_ci95",
        "naive_ci95",
        "ceiling_ci95",
        "headroom_delta",
        "headroom_ci95",
        "official_repeats_used",
        "escalated",
        "escalation_note",
        "pooled_observation_counts",
        "internal_evals_count",
        "optimizer_steps",
        "spend_usd",
        "wall_s",
        "lane",
        "window_notes",
        "status",
        "artifacts",
        "sampling",
        "telemetry",
        "controls",
        "graph_hash",
        "eval_config_hash",
        "started_at",
        "finished_at",
    }
    assert payload["schema"] == CELLS_SCHEMA
    assert set(payload["models"]) == {"task", "proposer"}
    assert set(payload["sampling"]) == {"official_n", "official_repeats"}
    assert set(payload["controls"]) == {
        "temperature",
        "reasoning_effort",
        "prompt_cache",
    }
    assert set(payload["artifacts"]) == {
        "optimization_result_ref",
        "optimization_trace_ref",
        "best_candidate_id",
        "official_record_before",
        "official_record_after",
        "viewer_publication",
    }
    assert set(payload["telemetry"]) == {
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "total_reasoning_tokens",
        "total_latency_s",
        "mean_latency_s",
        "token_coverage",
        "reasoning_coverage",
        "latency_coverage",
    }


def test_spend_record_wire_keys_are_pinned() -> None:
    payload = json.loads(
        SpendRecord(
            cell_id="copro:c18:a0", phase="before", lane="openrouter"
        ).to_line()
    )

    assert set(payload) == {
        "schema",
        "event_id",
        "cell_id",
        "phase",
        "lane",
        "total_credits",
        "total_usage",
        "remaining_usd",
        "at",
    }
    assert payload["schema"] == SPEND_SCHEMA


def test_official_anchor_wire_keys_are_pinned() -> None:
    payload = json.loads(_anchor().to_json())

    assert set(payload) == {
        "schema",
        "cell_id",
        "env",
        "task_model",
        "graph_hash",
        "eval_config_hash",
        "official_instance_ids",
        "official_task_identities",
        "baseline_evidence_ref",
        "ceiling_evidence_ref",
        "baseline_official",
        "ceiling_official",
        "baseline_per_task",
        "ceiling_per_task",
        "baseline_per_task_counts",
        "ceiling_per_task_counts",
        "official_repeats_used",
    }
    assert payload["schema"] == OFFICIAL_ANCHOR_SCHEMA


# --------------------------------------------------------------------------
# CellRecord validation
# --------------------------------------------------------------------------


def test_cell_round_trips_through_its_line() -> None:
    record = _cell()

    assert CellRecord.from_line(record.to_line()) == record


def test_unknown_status_is_refused() -> None:
    with pytest.raises(ValidationError, match="status must be one of"):
        _cell(status="nonsense")


def test_unknown_schema_stamp_is_refused() -> None:
    payload = json.loads(_cell().to_line())
    payload["schema"] = "whetstone.runner.cells/v0"

    with pytest.raises(ValidationError, match="schema must be exactly"):
        CellRecord.from_line(json.dumps(payload))


def test_identity_fields_must_align_with_the_cell_id() -> None:
    with pytest.raises(ValidationError, match="identity fields do not align"):
        _cell(cell_id="gepa:c18:a0")
    with pytest.raises(ValidationError, match="identity fields do not align"):
        _cell(cell_id="copro:c18:a1")


def test_the_cell_id_env_segment_must_equal_env() -> None:
    with pytest.raises(ValidationError, match="env segment must equal env"):
        _cell(cell_id="copro:c22:a0")


def test_malformed_cell_id_is_refused() -> None:
    with pytest.raises(ValidationError, match="cell_id must be exactly"):
        _cell(cell_id="copro:c18:0", attempt=0)


def test_unordered_interval_bounds_are_refused() -> None:
    with pytest.raises(ValidationError, match="bounds must be ordered"):
        _cell(delta_ci95=(0.4, 0.1))


def test_interval_must_be_a_pair() -> None:
    with pytest.raises(ValidationError):
        _cell(naive_ci95=(0.1, 0.2, 0.3))


def test_completed_cell_requires_a_viewer_publication() -> None:
    with pytest.raises(ValidationError, match="requires viewer publication"):
        _cell(status="improved", artifacts=CellArtifacts())


@pytest.mark.parametrize(
    "status", ["plumbing-retry", "incomplete-arm", "proposer-failure"]
)
def test_non_terminal_cells_publish_nothing(status: str) -> None:
    record = _cell(status=status)

    assert not record.is_completed()
    assert record.artifacts.viewer_publication is None


def test_viewer_publication_path_must_match_the_cell_identity() -> None:
    with pytest.raises(ValidationError, match="must match cell identity"):
        _cell(
            status="improved",
            artifacts=CellArtifacts(
                viewer_publication=_publication("copro__c22__a0")
            ),
        )


def test_extra_wire_keys_are_refused() -> None:
    payload = json.loads(_cell().to_line())
    payload["surprise"] = 1

    with pytest.raises(ValidationError):
        CellRecord.from_line(json.dumps(payload))


def test_negative_pooled_counts_are_refused() -> None:
    with pytest.raises(ValidationError, match="cannot be negative"):
        _cell(pooled_observation_counts={"naive": -1})


def test_cell_key_matches_the_record_key() -> None:
    assert _cell().key() == cell_key("copro", "c18", 0)


# --------------------------------------------------------------------------
# SpendRecord validation
# --------------------------------------------------------------------------


def test_spend_phase_is_a_closed_set() -> None:
    with pytest.raises(ValidationError, match="phase must be"):
        SpendRecord(cell_id="copro:c18:a0", phase="during", lane="l")
    with pytest.raises(ValidationError, match="phase must be"):
        SpendRecord(cell_id="copro:c18:a0", phase="checkpoint:", lane="l")


def test_spend_phase_admits_named_paid_checkpoints() -> None:
    record = SpendRecord(
        cell_id="copro:c18:a0", phase="checkpoint:official:best", lane="l"
    )

    assert SpendRecord.from_line(record.to_line()) == record


def test_spend_at_is_null_when_never_captured() -> None:
    record = SpendRecord(cell_id="copro:c18:a0", phase="before", lane="l")

    assert record.at is None
    assert SpendRecord.from_line(record.to_line()) == record


# --------------------------------------------------------------------------
# OfficialAnchorRecord validation
# --------------------------------------------------------------------------


def test_anchor_round_trips_through_its_json() -> None:
    record = _anchor()

    assert OfficialAnchorRecord.model_validate_json(record.to_json()) == record


def test_anchor_per_task_vectors_must_align() -> None:
    with pytest.raises(ValidationError, match="length must match"):
        _anchor(baseline_per_task=(0.4,))


def test_anchor_counts_cannot_exceed_repeats_used() -> None:
    with pytest.raises(ValidationError, match="must be between 0 and"):
        _anchor(baseline_per_task_counts=(3, 2))


def test_anchor_requires_evaluation_evidence_refs() -> None:
    with pytest.raises(ValidationError, match="must reference"):
        _anchor(
            baseline_evidence_ref=TypedRef(
                schema_name="whetstone.other", content_hash=_HASH
            )
        )


def test_anchor_instance_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="unique IDs"):
        _anchor(official_instance_ids=("i0", "i0"))


def test_anchor_cell_id_env_must_agree() -> None:
    with pytest.raises(ValidationError, match="env segment must equal env"):
        _anchor(env="c22")


# --------------------------------------------------------------------------
# Ledger durability and resumability
# --------------------------------------------------------------------------


def test_append_and_load_round_trip(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    record = _cell()

    ledger.append_cell(record)

    assert ledger.cells() == [record]
    assert ledger.cells_path.exists()


def test_append_is_a_complete_fsynced_line(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell())
    ledger.append_cell(_cell(attempt=1, cell_id="copro:c18:a1"))

    body = ledger.cells_path.read_text()

    assert body.endswith("\n")
    assert len(body.splitlines()) == 2


def test_completed_keys_drive_resumability(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell(status="improved"))
    ledger.append_cell(
        _cell(status="plumbing-retry", attempt=1, cell_id="copro:c18:a1")
    )

    assert ledger.completed_keys() == {("copro", "c18", 0)}
    assert ledger.is_completed("copro", "c18", 0)
    assert not ledger.is_completed("copro", "c18", 1)


def test_a_superseding_incomplete_line_reopens_the_key(
    tmp_path: Path,
) -> None:
    """The latest line decides completion, so a correction reopens the cell.

    A ``refinalize`` correction that demotes a certified line to a
    non-terminal status exists precisely so the cell can be re-run and
    repaired. Reading completion from any line rather than the latest would
    keep the key skipped forever on the strength of the line just superseded.
    """
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell(status="improved"))
    ledger.append_cell(_cell(status="incomplete-arm"))

    assert ledger.completed_keys() == set()
    assert not ledger.is_completed("copro", "c18", 0)
    latest = ledger.for_attempt("copro", "c18", 0)
    assert latest is not None
    assert latest.status == "incomplete-arm"


def test_a_superseding_completed_line_closes_the_key(tmp_path: Path) -> None:
    """The latest line decides completion in the other direction too."""
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell(status="plumbing-retry"))
    ledger.append_cell(_cell(status="improved"))

    assert ledger.completed_keys() == {("copro", "c18", 0)}
    assert ledger.is_completed("copro", "c18", 0)


def test_for_attempt_returns_the_latest_record(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell(status="plumbing-retry"))
    ledger.append_cell(_cell(status="improved"))

    latest = ledger.for_attempt("copro", "c18", 0)

    assert latest is not None
    assert latest.status == "improved"
    assert ledger.for_attempt("copro", "c18", 7) is None


def test_latest_for_ignores_other_optimizers(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell())
    ledger.append_cell(
        _cell(optimizer="miprov2", cell_id="miprov2:c18:a0", status="improved")
    )

    latest = ledger.latest_for("copro", "c18")

    assert latest is not None
    assert latest.optimizer == "copro"
    assert ledger.latest_for("gepa", "c18") is None


def test_a_malformed_line_is_refused_on_load(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell())
    with ledger.cells_path.open("a") as handle:
        handle.write('{"schema": "whetstone.runner.cells/v1"}\n')

    with pytest.raises(ValidationError):
        ledger.load()


def test_append_refuses_a_symlinked_ledger_file(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (tmp_path / "elsewhere.jsonl").write_text("")
    (root / "cells.jsonl").symlink_to(tmp_path / "elsewhere.jsonl")
    ledger = Ledger(root)

    with pytest.raises(OSError):
        ledger.append_cell(_cell())


def test_total_spend_sums_recorded_cells(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell(spend_usd=0.25))
    ledger.append_cell(
        _cell(attempt=1, cell_id="copro:c18:a1", spend_usd=0.75)
    )

    assert ledger.total_spend_usd() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Official-anchor publication
# --------------------------------------------------------------------------


def test_official_anchor_publishes_atomically(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    record = _anchor()

    path = ledger.write_official_anchor(record)

    assert path == ledger.official_anchor_path("copro:c18:a0")
    assert OfficialAnchorRecord.model_validate_json(path.read_text()) == record
    # No temporary file survives a successful publication.
    assert [entry.name for entry in path.parent.iterdir()] == [path.name]


def test_republishing_an_identical_anchor_is_idempotent(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")
    record = _anchor()

    first = ledger.write_official_anchor(record)
    second = ledger.write_official_anchor(record)

    assert first == second


def test_reassigning_anchor_evidence_is_refused(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.write_official_anchor(_anchor())

    with pytest.raises(RuntimeError, match="conflicts at"):
        ledger.write_official_anchor(_anchor(baseline_official=0.6))


def test_anchor_refuses_a_symlinked_target(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.official_anchors_dir.mkdir(parents=True)
    target = tmp_path / "evil.json"
    target.write_text("{}")
    ledger.official_anchor_path("copro:c18:a0").symlink_to(target)

    with pytest.raises(RuntimeError, match="unsafe at"):
        ledger.write_official_anchor(_anchor())


def test_anchor_refuses_a_symlinked_directory(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / "official_anchors").symlink_to(elsewhere)
    ledger = Ledger(root)

    with pytest.raises(RuntimeError, match="must be a real directory"):
        ledger.write_official_anchor(_anchor())


# --------------------------------------------------------------------------
# Viewer publication
# --------------------------------------------------------------------------


def test_viewer_publication_commits_by_one_atomic_rename(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")
    body = b'{"schema":"x"}'
    lines = ['{"row":0}\n', '{"row":1}\n']

    ref = ledger.write_viewer_publication(
        cell_id="copro:c18:a0",
        env="c18",
        projection_body=body,
        rollout_lines=lines,
    )

    directory = ledger.viewer_cell_dir("copro:c18:a0")
    assert (directory / "projection.json").read_bytes() == body
    assert (directory / "rollout_outputs.jsonl").read_text() == "".join(lines)
    assert ref.projection.relative_path == (
        "viewer_cells/copro__c18__a0/projection.json"
    )
    # No temp directory survives a successful rename.
    assert {entry.name for entry in directory.parent.iterdir()} == {
        "copro__c18__a0"
    }


def test_viewer_publication_hashes_the_exact_committed_bytes(
    tmp_path: Path,
) -> None:
    import hashlib

    ledger = Ledger(tmp_path / "run")
    body = b'{"schema":"x"}'
    lines = ['{"row":0}\n']

    ref = ledger.write_viewer_publication(
        cell_id="copro:c18:a0",
        env="c18",
        projection_body=body,
        rollout_lines=lines,
    )

    directory = ledger.viewer_cell_dir("copro:c18:a0")
    assert (
        ref.projection.sha256
        == hashlib.sha256(
            (directory / "projection.json").read_bytes()
        ).hexdigest()
    )
    assert (
        ref.rollout_outputs.sha256
        == hashlib.sha256(
            (directory / "rollout_outputs.jsonl").read_bytes()
        ).hexdigest()
    )


def test_republishing_identical_viewer_bytes_is_idempotent(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")

    first = ledger.write_viewer_publication(
        cell_id="copro:c18:a0",
        env="c18",
        projection_body=b"{}",
        rollout_lines=["{}\n"],
    )
    second = ledger.write_viewer_publication(
        cell_id="copro:c18:a0",
        env="c18",
        projection_body=b"{}",
        rollout_lines=["{}\n"],
    )

    assert first == second


def test_republishing_different_viewer_bytes_is_refused(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.write_viewer_publication(
        cell_id="copro:c18:a0",
        env="c18",
        projection_body=b"{}",
        rollout_lines=["{}\n"],
    )

    with pytest.raises(RuntimeError, match="conflicts at"):
        ledger.write_viewer_publication(
            cell_id="copro:c18:a0",
            env="c18",
            projection_body=b'{"changed":1}',
            rollout_lines=["{}\n"],
        )


def test_viewer_publication_refuses_a_mismatched_env(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")

    with pytest.raises(ValueError, match="env segment must equal env"):
        ledger.write_viewer_publication(
            cell_id="copro:c18:a0",
            env="c22",
            projection_body=b"{}",
            rollout_lines=[],
        )


def test_viewer_publication_ref_requires_the_canonical_file_pair() -> None:
    with pytest.raises(ValidationError, match="one canonical cell directory"):
        ViewerCellPublicationRef(
            projection=ViewerPublishedFileRef(
                relative_path="viewer_cells/copro__c18__a0/projection.json",
                sha256=_HASH,
            ),
            rollout_outputs=ViewerPublishedFileRef(
                relative_path="viewer_cells/copro__c22__a0/"
                "rollout_outputs.jsonl",
                sha256=_OTHER_HASH,
            ),
        )


def test_published_paths_must_be_relative_and_canonical() -> None:
    with pytest.raises(ValidationError, match="canonical and relative"):
        ViewerPublishedFileRef(
            relative_path="../escape/projection.json", sha256=_HASH
        )


# --------------------------------------------------------------------------
# Optimization trace artifacts
# --------------------------------------------------------------------------


def test_optimization_trace_is_written_under_a_safe_name(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")

    path = ledger.write_optimization_trace("copro:c18:a0", {"steps": []})

    assert path.name == "copro__c18__a0.json"
    assert json.loads(path.read_text()) == {"steps": []}


def test_rewriting_a_trace_supersedes_the_prior_one(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.write_optimization_trace("copro:c18:a0", {"steps": [1]})

    path = ledger.write_optimization_trace("copro:c18:a0", {"steps": [1, 2]})

    assert json.loads(path.read_text()) == {"steps": [1, 2]}


# --------------------------------------------------------------------------
# Spend pairing
# --------------------------------------------------------------------------


def _spend(cell_id: str, phase: str, remaining: float | None) -> SpendRecord:
    return SpendRecord(
        cell_id=cell_id,
        phase=phase,
        lane="openrouter",
        remaining_usd=remaining,
    )


def test_spend_pairs_a_clean_before_after(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_spend(_spend("copro:c18:a0", "before", 100.0))
    ledger.append_spend(_spend("copro:c18:a0", "after", 98.0))

    total, gaps = ledger.spend_for_cell("copro:c18:a0")

    assert total == pytest.approx(2.0)
    assert gaps == []


def test_spend_pairs_by_cell_id_under_interleaving(tmp_path: Path) -> None:
    # Another cell's snapshots land between this cell's before and after; the
    # "next record" reading would mis-bound the spend, so pairing is by id.
    ledger = Ledger(tmp_path / "run")
    ledger.append_spend(_spend("copro:c18:a0", "before", 100.0))
    ledger.append_spend(_spend("gepa:c22:a0", "before", 100.0))
    ledger.append_spend(_spend("gepa:c22:a0", "after", 95.0))
    ledger.append_spend(_spend("copro:c18:a0", "after", 90.0))

    total, gaps = ledger.spend_for_cell("copro:c18:a0")

    assert total == pytest.approx(10.0)
    assert gaps == []


def test_spend_spans_the_cells_own_checkpoints_to_its_after(
    tmp_path: Path,
) -> None:
    """Checkpoints sit inside the pair, so they never close it.

    This is the shape every real cell writes: a ``before``, one
    ``checkpoint:<boundary>`` per paid boundary, then the closing ``after``.
    Reading the first checkpoint as the closing snapshot would stop the total
    at the first boundary and silently omit every later one.
    """
    ledger = Ledger(tmp_path / "run")
    ledger.append_spend(_spend("copro:c18:a0", "before", 100.0))
    ledger.append_spend(
        _spend("copro:c18:a0", "checkpoint:official:baseline", 99.0)
    )
    ledger.append_spend(
        _spend("copro:c18:a0", "checkpoint:official:best", 95.0)
    )
    ledger.append_spend(_spend("copro:c18:a0", "after", 90.0))

    total, gaps = ledger.spend_for_cell("copro:c18:a0")

    assert total == pytest.approx(10.0)
    assert gaps == []


def test_a_crashed_attempt_with_checkpoints_reports_a_bounded_gap(
    tmp_path: Path,
) -> None:
    """A crash with no ``after`` is a reported gap, not a silent total.

    The crashed branch bounds by the next snapshot in file order, which for a
    cell that wrote checkpoints is its own first checkpoint. That is a
    documented lower bound, and it is reported as a gap rather than presented
    as a settled total -- unlike the clean-pair path, which must be exact.
    """
    ledger = Ledger(tmp_path / "run")
    ledger.append_spend(_spend("copro:c18:a0", "before", 100.0))
    ledger.append_spend(
        _spend("copro:c18:a0", "checkpoint:official:baseline", 96.0)
    )
    ledger.append_spend(_spend("gepa:c22:a0", "before", 94.0))

    total, gaps = ledger.spend_for_cell("copro:c18:a0")

    assert total == pytest.approx(4.0)
    assert len(gaps) == 1
    assert "crashed" in gaps[0]


def test_a_crashed_attempt_is_bounded_by_the_next_snapshot(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_spend(_spend("copro:c18:a0", "before", 100.0))
    ledger.append_spend(_spend("gepa:c22:a0", "before", 97.0))
    ledger.append_spend(_spend("copro:c18:a0", "before", 97.0))
    ledger.append_spend(_spend("copro:c18:a0", "after", 96.0))

    total, gaps = ledger.spend_for_cell("copro:c18:a0")

    # The crashed first attempt burned 3.0 before dying; the second cleanly
    # spent 1.0.
    assert total == pytest.approx(4.0)
    assert len(gaps) == 1
    assert "crashed" in gaps[0]


def test_a_trailing_before_is_reported_as_an_unbounded_gap(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_spend(_spend("copro:c18:a0", "before", 100.0))

    total, gaps = ledger.spend_for_cell("copro:c18:a0")

    assert total == pytest.approx(0.0)
    assert len(gaps) == 1
    assert "consumption unaccounted" in gaps[0]


def test_non_monotonic_credits_are_skipped_not_summed(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_spend(_spend("copro:c18:a0", "before", 90.0))
    ledger.append_spend(_spend("copro:c18:a0", "after", 95.0))

    total, gaps = ledger.spend_for_cell("copro:c18:a0")

    assert total == pytest.approx(0.0)
    assert len(gaps) == 1
    assert "non-monotonic" in gaps[0]


def test_snapshots_without_remaining_are_ignored(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_spend(_spend("copro:c18:a0", "before", None))
    ledger.append_spend(_spend("copro:c18:a0", "after", None))

    total, gaps = ledger.spend_for_cell("copro:c18:a0")

    assert total == pytest.approx(0.0)
    assert gaps == []


def test_spend_records_round_trip(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    records = [
        _spend("copro:c18:a0", "before", 100.0),
        _spend("copro:c18:a0", "after", 98.0),
    ]
    for record in records:
        ledger.append_spend(record)

    assert ledger.spend_records() == records


def test_reading_an_absent_ledger_yields_nothing(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "never-created")

    assert ledger.cells() == []
    assert ledger.spend_records() == []
    assert ledger.completed_keys() == set()


def test_concurrent_appends_never_interleave_a_line(tmp_path: Path) -> None:
    # Two independent Ledger handles over one root append under the same file
    # lock; every line stays complete and parseable.
    root = tmp_path / "run"
    first = Ledger(root)
    second = Ledger(root)
    first.append_cell(_cell())
    second.append_cell(_cell(attempt=1, cell_id="copro:c18:a1"))
    first.append_cell(_cell(attempt=2, cell_id="copro:c18:a2"))

    assert {cell.attempt for cell in Ledger(root).cells()} == {0, 1, 2}


def test_the_viewer_source_lock_is_created_beside_the_ledger(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell())

    lock = tmp_path / "run" / ".viewer-sources.lock"
    assert lock.exists()
    assert stat_mode(lock) == 0o600


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777
