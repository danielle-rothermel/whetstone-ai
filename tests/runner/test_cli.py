"""CLI surface tests: parser wiring + the no-live-calls refusal.

The CLI refuses to run without ``--live``. These tests never pass ``--live``
(no live paid call), so they exercise only the parser + refusal path.
"""

from __future__ import annotations

import pytest

from whetstone.runner.cli import build_parser, main
from whetstone.runner.optimizers import OPTIMIZERS


def test_parser_has_pilot_and_cell_subcommands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "pilot" in help_text
    assert "cell" in help_text
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
