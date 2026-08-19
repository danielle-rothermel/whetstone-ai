from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from whetstone.eval.protocol import EvalRequest
from whetstone.experiment.reward import apply_reward_policy
from whetstone.optim.gepa.step_engine import GepaStepCheckpoint, run_one_gepa_iteration
from whetstone.testing.toy.experiment import build_toy_experiment


def test_gepa_zero_metric_budget_returns_terminal_checkpoint(tmp_path) -> None:
    from tests.test_gepa_harness_adapter import _toy_gepa_control

    control = _toy_gepa_control(
        max_metric_calls=0,
        sqlite_path=str(tmp_path / "gepa-zero.sqlite"),
    )
    detailed, checkpoint = run_one_gepa_iteration(
        control=control,
        seed_candidate={"generate": "seed"},
        trainset=(),
        valset=None,
        adapter=MagicMock(),
        checkpoint=GepaStepCheckpoint(),
    )
    assert checkpoint.terminal is True
    assert checkpoint.metric_calls_consumed == 0
    assert detailed.total_metric_calls == 0
    assert detailed.best_idx == 0


def test_internal_reward_fallback_includes_supplemental_aggregates(
    sqlite_store,
) -> None:
    from whetstone.eval.aggregate import Aggregate
    from whetstone.eval import AggregationOutput, AggregationStatus
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.eval.drivers.eval_result import InternalEvalResult
    from whetstone.experiment.candidate import candidate_reference

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(sqlite_store)
    ok_output = AggregationOutput(
        value=0.5,
        status=AggregationStatus.OK,
        count_total=1,
        count_applicable=1,
        count_present=1,
    )
    primary = Aggregate(
        name="score",
        graph_hash=experiment.rollout_graph.graph_hash,
        eval_config_hash=engine.eval_config_ref.config_hash,
        task_count=1,
        num_seeds=1,
        aggregation_output=ok_output,
        rows_present=1,
        rows_missing=0,
        rows_failed=0,
        rows_invalid=0,
    )
    supplemental = Aggregate(
        name="supplemental",
        graph_hash=experiment.rollout_graph.graph_hash,
        eval_config_hash=engine.eval_config_ref.config_hash,
        task_count=1,
        num_seeds=1,
        aggregation_output=AggregationOutput(
            value=0.25,
            status=AggregationStatus.OK,
            count_total=1,
            count_applicable=1,
            count_present=1,
        ),
        rows_present=1,
        rows_missing=0,
        rows_failed=0,
        rows_invalid=0,
    )
    request = EvalRequest(
        request_id="reward-fallback",
        candidate=experiment.initial_candidate,
    )
    internal = InternalEvalResult(
        aggregate=primary,
        reward=None,
        per_task_scores=(0.5,),
        per_task_counts=(1,),
        outputs=(),
        supplemental_aggregates=(supplemental,),
    )
    traces = MagicMock()
    traces.record_content.return_value = {"schema_version": 2}
    output_rows = MagicMock()
    with patch.object(
        engine,
        "_evaluation_records",
        return_value=(traces, output_rows),
    ), patch.object(
        engine,
        "_evaluation_outputs_record",
        return_value=MagicMock(record_content=lambda: {"schema_version": 4}),
    ), patch(
        "whetstone.eval.runtime_engine.apply_reward_policy",
        wraps=apply_reward_policy,
    ) as reward_policy:
        engine._persist_success(request, internal)  # noqa: SLF001
    reward_policy.assert_called_once()
    kwargs = reward_policy.call_args.kwargs
    assert kwargs["aggregates"] == {
        "score": 0.5,
        "supplemental": 0.25,
    }
    assert len(kwargs["evidence_refs"]) == 2
    _ = candidate_reference
