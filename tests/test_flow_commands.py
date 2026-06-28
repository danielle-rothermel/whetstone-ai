from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from dr_dspy import humaneval_dbos_flow as flow
from dr_dspy.dbos_runtime import EvalDbosConfig
from dr_dspy.eval_repair import RepairPlan


def _config() -> EvalDbosConfig:
    return EvalDbosConfig(
        database_url="postgresql://x",
        dbos_system_database_url="postgresql://y",
        generation_concurrency=1,
        scoring_concurrency=1,
    )


def test_run_repair_command_dry_run_does_not_apply() -> None:
    backend = MagicMock()
    backend.build_repair_plan.return_value = RepairPlan()
    flow.run_repair_command(
        backend,
        config=_config(),
        experiment_name="exp",
        generation_limit=10,
        scoring_limit=10,
        score_timeout=10.0,
    )
    backend.create_schema.assert_called_once_with("postgresql://x")
    backend.build_repair_plan.assert_called_once()
    assert backend.method_calls.index(call.create_schema("postgresql://x")) < (
        backend.method_calls.index(
            call.build_repair_plan(
                "postgresql://x",
                dbos_system_database_url="postgresql://y",
                experiment_name="exp",
                generation_limit=10,
                scoring_limit=10,
            )
        )
    )
    backend.apply_repair.assert_not_called()


def test_repair_plan_line_omits_legacy_retry_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines: list[str] = []
    backend = MagicMock()
    backend.build_repair_plan.return_value = RepairPlan()
    monkeypatch.setattr(
        flow, "operator_log", lambda line, *, style=None: lines.append(line)
    )

    flow.run_repair_command(
        backend,
        config=_config(),
        experiment_name="exp",
        generation_limit=10,
        scoring_limit=10,
        score_timeout=10.0,
    )

    assert "legacy=" not in lines[0]
    assert "gen_retry=" in lines[0]
    assert "score_retry=" in lines[0]
