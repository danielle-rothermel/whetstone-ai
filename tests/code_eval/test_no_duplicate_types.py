"""No duplicate dr-code types in the Whetstone code-eval adapters.

The adapters consume the released dr-code kernel and add only Whetstone policy
and boundary roles. They must not redefine dr-code's evaluation-kernel
contracts (TextArtifact, Code Artifact, candidates, MetricFact, Score,
compression key/artifact/resolver, Aggregation Output).
"""

from __future__ import annotations

from dataclasses import is_dataclass

import dr_code.eval as dr_eval
import dr_code.trace as dr_trace

from whetstone.code_eval import (
    BootstrapCI,
    CandidateEvaluation,
    CandidateVerdict,
    CompletenessPolicy,
    CorrectnessOutcome,
    CorrectnessResult,
    PowerConfig,
    PowerRecommendation,
    PowerResult,
    PowerSurfacePoint,
    RolloutAggregate,
    RowValue,
    TaskRows,
    VarianceDecomposition,
    binary_test_pass_score,
    compressed_description_length_fact,
    compression_ratio_score,
    compression_selection,
    scoring,
    select_compression_reference,
    submission,
    submission_text_artifact,
)
from whetstone.code_eval import aggregate as agg_module

from .support import generation, operator_lineage


def test_boundary_reuses_dr_code_text_artifact() -> None:
    assert submission.TextArtifact is dr_trace.TextArtifact


def test_scoring_returns_dr_code_score_and_fact_types() -> None:
    score = binary_test_pass_score(
        _passed(), evaluation_procedure_config_hash="0" * 64
    )
    assert type(score) is dr_eval.Score

    fact = compressed_description_length_fact(
        "code", lineage=operator_lineage()
    )
    assert type(fact) is dr_eval.MetricFact

    ratio_score = compression_ratio_score(
        compressed_description_length=1,
        reference=dr_eval.CompressionReferenceArtifact(content=b"abcd"),
        evaluation_procedure_config_hash="0" * 64,
    )
    assert type(ratio_score) is dr_eval.Score


def test_compression_selection_returns_generic_types() -> None:
    from pydantic import BaseModel

    class _Task(BaseModel):
        gt_code_wo_comments: str

    artifact = select_compression_reference(_Task(gt_code_wo_comments="x"))
    assert type(artifact) is dr_eval.CompressionReferenceArtifact


def test_no_module_redefines_a_dr_code_kernel_type() -> None:
    dr_code_names = set(dr_eval.__all__) | {"TextArtifact", "CodeArtifact"}
    for module in (
        submission,
        scoring,
        compression_selection,
        agg_module,
    ):
        for name in getattr(module, "__all__", []):
            # A Whetstone export may share a *concept* but must not shadow a
            # dr-code kernel type name with a new class.
            if name in dr_code_names:
                exported = getattr(module, name)
                # If the name collides, it must be the dr-code object itself
                # (re-exported), not a Whetstone redefinition.
                assert getattr(exported, "__module__", "").startswith(
                    "dr_code"
                ), f"{module.__name__}.{name} shadows a dr-code type"


def _passed():
    from whetstone.code_eval import (
        CandidateVerdict,
        evaluate_candidate_correctness,
    )

    cs = dr_eval.CodeCandidateSet.from_sources(
        ("def f():\n    return 1\n",), origin="t"
    )
    return evaluate_candidate_correctness(
        cs, runner=lambda _a: CandidateVerdict.PASSED
    )


def test_submission_generation_is_whetstone_owned() -> None:
    # Code Generation stays a Whetstone Generation; dr-code never learns it.
    gen = generation(text="x = 1\n")
    assert gen.__class__.__module__.startswith("whetstone")
    artifact = submission_text_artifact(gen)
    assert artifact.__class__.__module__.startswith("dr_code")


def test_internal_value_objects_are_frozen_slotted_dataclasses() -> None:
    value_objects = (
        CandidateEvaluation(position=0, verdict=CandidateVerdict.PASSED),
        CorrectnessResult(outcome=CorrectnessOutcome.PASSED),
        CompletenessPolicy(),
        RowValue(value=1.0),
        TaskRows(
            task_identity="task",
            expected_repeats=1,
            rows=(RowValue(value=1.0),),
        ),
        BootstrapCI(point=0.5, low=0.25, high=0.75, level=0.95, resamples=10),
        PowerConfig(),
        VarianceDecomposition(
            base_rate=0.5,
            within_repeat_var=0.25,
            interaction_var=0.1,
            between_task_var=0.2,
            anchor_repeats=3,
            n_tasks_observed=2,
        ),
        PowerRecommendation(
            target_gap=0.1,
            achievable=True,
            recommended_n_tasks=2,
            recommended_repeats=1,
            achieved_mdd=0.1,
            recommended_calls=2,
            recommended_usd=None,
            best_achievable_mdd=0.1,
            best_n_tasks=2,
            best_repeats=1,
            repeat_plateau=None,
            pool_limited=False,
        ),
        PowerSurfacePoint(
            n_tasks=2,
            repeats=1,
            calls=2,
            mdd_at_target=0.1,
            simulated_rank_probability=0.8,
        ),
    )
    for value in value_objects:
        assert is_dataclass(value)
        assert hasattr(type(value), "__slots__")
        assert type(value).__dataclass_params__.frozen

    # Aggregate/result containers obey the same internal-value-object rule.
    output = dr_eval.AggregationOutput(
        status=dr_eval.AggregationStatus.NOT_APPLICABLE,
        value=None,
        count_total=0,
        count_applicable=0,
        count_present=0,
    )
    aggregate = RolloutAggregate(
        name="x",
        graph_hash="0" * 64,
        eval_config_hash="1" * 64,
        evaluation_context_id="2" * 64,
        task_count=0,
        repeat_count=1,
        aggregation_output=output,
        rows_present=0,
        rows_missing=0,
        rows_failed=0,
        rows_invalid=0,
    )
    result = PowerResult(
        config=PowerConfig(),
        certified_headroom=0.0,
        naive_mean=0.5,
        ceiling_mean=0.5,
        pool_ceiling=1,
        decomposition=value_objects[7],
        recommendation=value_objects[8],
    )
    for value in (aggregate, result):
        assert is_dataclass(value)
        assert hasattr(type(value), "__slots__")
        assert type(value).__dataclass_params__.frozen
