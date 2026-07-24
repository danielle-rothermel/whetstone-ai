"""CLI surface tests: parser wiring + the no-live-calls refusal.

The CLI refuses to run without ``--live``. These tests never pass ``--live``
(no live paid call), so they exercise only the parser + refusal path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whetstone.runner.cli import _Heartbeat, build_parser, main
from whetstone.runner.ledger import (
    CellArtifacts,
    CellModels,
    CellRecord,
    Ledger,
)
from whetstone.runner.optimizers import OPTIMIZERS


def test_parser_has_pilot_and_cell_subcommands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "pilot" in help_text
    assert "cell" in help_text
    assert "refinalize" in help_text
    # Scaling documentation appears in the top-level help.
    assert "scaling" in help_text.lower()


def test_cell_optimizer_choices_are_the_five() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["cell", "--optimizer", "gepa", "--env", "c11"]
    )
    assert args.optimizer == "gepa"
    assert args.env == "c11"
    assert set(OPTIMIZERS) == {"eval", "copro", "miprov2", "gepa", "codex"}


def test_pilot_accepts_task_model_override() -> None:
    # The pilot subcommand exposes --task-model (parity with the cell path)
    # so a headroom pilot can be run under a specific task model.
    parser = build_parser()
    args = parser.parse_args(
        [
            "pilot",
            "--env",
            "c22h",
            "--task-model",
            "openai/gpt-5-nano",
        ]
    )
    assert args.env == "c22h"
    assert args.task_model == "openai/gpt-5-nano"


def test_pilot_task_model_defaults_to_none() -> None:
    # Absent --task-model, the flag is None and the per-env matrix default
    # (task_model_for_env) applies at run time.
    parser = build_parser()
    args = parser.parse_args(["pilot", "--env", "c22h"])
    assert args.task_model is None


def test_lane_choices_include_openrouter_and_plan_lanes() -> None:
    parser = build_parser()
    for lane in ("openrouter", "kimi", "glm", "minimax", "stepfun"):
        args = parser.parse_args(
            ["cell", "--optimizer", "copro", "--env", "c11", "--lane", lane]
        )
        assert args.lane == lane


def test_pilot_without_live_refuses() -> None:
    with pytest.raises(SystemExit, match="NO live"):
        main(["pilot", "--env", "c11"])


def test_cell_without_live_refuses() -> None:
    with pytest.raises(SystemExit, match="NO live"):
        main(["cell", "--optimizer", "copro", "--env", "c11"])


def test_invalid_optimizer_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit):
        main(["cell", "--optimizer", "nonsense", "--env", "c11"])


def test_heartbeat_emits_progress_lines_with_spend_estimate() -> None:
    import time

    lines: list[str] = []
    spends = iter([0.0, 0.5, 1.25])

    with _Heartbeat(
        label="cell eval:c11:a0",
        spend_estimate=lambda: next(spends, 1.25),
        interval=0.01,
        sink=lines.append,
    ):
        time.sleep(0.06)
    assert lines, "heartbeat should emit at least one progress line"
    assert all("[heartbeat] cell eval:c11:a0" in line for line in lines)
    assert any("spend~=$" in line for line in lines)
    assert all("elapsed=" in line for line in lines)


def test_refinalize_cli_corrects_a_halted_but_completed_cell(
    tmp_path: Path,
) -> None:
    # refinalize needs NO --live: it recomputes from persisted evidence only.
    ledger = Ledger(root=tmp_path)
    ledger.append_cell(
        CellRecord(
            cell_id="eval:c11:a0", optimizer="eval", env="c11", attempt=0,
            canonical=True,
            models=CellModels(task="openai/gpt-5-nano", proposer="p"),
            baseline_official=0.667, ceiling_official=0.981,
            best_official=0.667, delta=0.0, ci95=(0.0, 0.0),
            delta_ci95=(0.0, 0.0),
            internal_evals_count=1, optimizer_steps=0, spend_usd=1.5,
            wall_s=3870.0, lane="openrouter", status="halted",
            escalation_note="halted: whole-cell wall deadline 3600s reached",
            artifacts=CellArtifacts(),
        )
    )
    code = main(
        ["--root", str(tmp_path), "refinalize", "--optimizer", "eval",
         "--env", "c11"]
    )
    assert code == 0
    lines = Ledger(root=tmp_path).load()
    assert len(lines) == 2
    assert lines[-1].status == "no-improvement"
    assert "refinalized" in lines[-1].escalation_note
