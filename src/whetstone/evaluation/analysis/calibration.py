from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.analysis.power import (
    PowerConfig,
    PowerResult,
    analyze_power,
)
from whetstone.evaluation.analysis.statistics import (
    DEFAULT_RESAMPLES,
    BootstrapCI,
    bootstrap_paired_delta_ci,
)
from whetstone.evaluation.engine import (
    EngineEvaluation,
    EvaluationEngine,
    EvaluationRequest,
)
from whetstone.experiment.binding import EvaluationBinding
from whetstone.experiment.candidate import Candidate

__all__ = [
    "AnchorCalibrationResult",
    "run_anchor_calibration",
]


@dataclass(frozen=True, slots=True)
class AnchorCalibrationResult:
    """Persisted anchor evaluations and their paired planning result."""

    evaluation_binding: EvaluationBinding
    baseline: EngineEvaluation
    ceiling: EngineEvaluation
    paired_delta_ci: BootstrapCI
    power: PowerResult


def _subset_binding(
    binding: EvaluationBinding,
    engine: EvaluationEngine,
) -> EvaluationBinding:
    return EvaluationBinding.model_validate(
        {
            **binding.model_dump(mode="json"),
            "eval_config": engine.eval_config_ref.model_dump(mode="json"),
        }
    )


def _validate_anchor_evidence(
    *,
    evaluated: EngineEvaluation,
    expected_binding: EvaluationBinding,
    expected_task_ids: tuple[str, ...],
    expected_repeats: int,
    expected_reward_policy_hash: str,
) -> None:
    evidence = evaluated.evidence
    if evidence.evaluation_binding != expected_binding:
        raise ValueError("calibration evidence changed its Evaluation Binding")
    if evidence.task_hashes != expected_task_ids:
        raise ValueError("calibration evidence changed task identity order")
    if evidence.num_samples != expected_repeats:
        raise ValueError("calibration evidence changed repeat count")
    if len(evidence.per_task_values) != len(expected_task_ids):
        raise ValueError("calibration evidence has incomplete per-task values")
    if evidence.per_task_counts != (expected_repeats,) * len(
        expected_task_ids
    ):
        raise ValueError("calibration evidence changed per-task repeat counts")
    if evidence.row_accounting.planned != (
        len(expected_task_ids) * expected_repeats
    ):
        raise ValueError("calibration evidence changed planned row accounting")
    if evidence.reward_ref is None:
        raise ValueError("calibration requires internal reward evidence")
    observed_policy_hash = (
        evidence.reward_ref.record.reward_policy.identity_hash()
    )
    if observed_policy_hash != expected_reward_policy_hash:
        raise ValueError("calibration evidence changed its Reward Policy")


def run_anchor_calibration(
    *,
    engine: EvaluationEngine,
    evaluation_binding: EvaluationBinding,
    baseline_candidate: Candidate,
    ceiling_candidate: Candidate,
    baseline_purpose: str,
    ceiling_purpose: str,
    task_ids: tuple[str, ...],
    pool_ceiling: int,
    power_config: PowerConfig | None = None,
    bootstrap_level: float = 0.95,
    bootstrap_resamples: int = DEFAULT_RESAMPLES,
    bootstrap_seed: int = 0,
    baseline_log_label: str = "baseline anchor",
    ceiling_log_label: str = "comparison anchor",
    log: Callable[[str], None] | None = None,
) -> AnchorCalibrationResult:
    """Evaluate both anchors on one exact task/sample binding.

    Callers must supply per-task values suitable for :func:`analyze_power`
    (typically bounded observations in ``[0, 1]``). The returned paired
    bootstrap is empirical over the aligned per-task vectors from both arms.
    """
    experiment = engine.experiment
    if evaluation_binding.role is not EvaluationRole.INTERNAL:
        raise ValueError("anchor calibration requires an internal binding")
    if evaluation_binding.eval_config != engine.eval_config_ref:
        raise ValueError(
            "calibration binding must name the engine's exact Eval Config"
        )
    if (
        evaluation_binding.provider_execution_policy_ref
        != engine.provider_execution_policy_ref
    ):
        raise ValueError(
            "calibration binding must name the engine's exact Provider "
            "Execution Policy"
        )
    if (
        engine.sampling.split_role
        != experiment.eval_configs.internal.split_role
    ):
        raise ValueError(
            "anchor calibration requires the internal sampling split"
        )
    if pool_ceiling < len(task_ids):
        raise ValueError(
            "pool_ceiling cannot be smaller than calibrated tasks"
        )
    if not 0.0 < bootstrap_level < 1.0:
        raise ValueError("bootstrap_level must be in (0, 1)")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be at least 1")

    from whetstone.evaluation.preview.anchor import _calibration_task_hashes

    calibration_task_ids = _calibration_task_hashes(engine, task_ids)
    subset_engine = engine.for_task_ids(calibration_task_ids)
    subset_binding = _subset_binding(evaluation_binding, subset_engine)
    baseline_request = EvaluationRequest(
        candidate=baseline_candidate,
        evaluation_binding=subset_binding,
        purpose=baseline_purpose,
    )
    ceiling_request = EvaluationRequest(
        candidate=ceiling_candidate,
        evaluation_binding=subset_binding,
        purpose=ceiling_purpose,
    )

    # Validate both anchors before the first paid evaluation starts.
    subset_engine.validate_request(baseline_request)
    subset_engine.validate_request(ceiling_request)

    planned_rows = (
        len(task_ids) * subset_engine.sampling.sample_plan.num_samples
    )
    if log is not None:
        log(f"Starting {baseline_log_label} evaluation ({planned_rows} rows)")
    baseline = subset_engine.evaluate(baseline_request)
    if log is not None:
        accounting = baseline.evidence.row_accounting
        log(
            f"Completed {baseline_log_label} evaluation "
            f"(present={accounting.present}/{accounting.planned}, "
            f"missing={accounting.missing}, failed={accounting.failed}, "
            f"invalid={accounting.invalid})"
        )
        log(f"Starting {ceiling_log_label} evaluation ({planned_rows} rows)")
    ceiling = subset_engine.evaluate(ceiling_request)
    if log is not None:
        accounting = ceiling.evidence.row_accounting
        log(
            f"Completed {ceiling_log_label} evaluation "
            f"(present={accounting.present}/{accounting.planned}, "
            f"missing={accounting.missing}, failed={accounting.failed}, "
            f"invalid={accounting.invalid})"
        )
    repeats = subset_engine.sampling.sample_plan.num_samples
    reward_policy_hash = subset_engine.reward_policy_identity_hash
    for evaluated in (baseline, ceiling):
        _validate_anchor_evidence(
            evaluated=evaluated,
            expected_binding=subset_binding,
            expected_task_ids=calibration_task_ids,
            expected_repeats=repeats,
            expected_reward_policy_hash=reward_policy_hash,
        )
    if baseline.evidence.graph_hash != ceiling.evidence.graph_hash:
        raise ValueError("calibration anchors changed graph identity")

    paired_delta_ci = bootstrap_paired_delta_ci(
        baseline.evidence.per_task_values,
        ceiling.evidence.per_task_values,
        level=bootstrap_level,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    power = analyze_power(
        naive_per_task=baseline.evidence.per_task_values,
        ceiling_per_task=ceiling.evidence.per_task_values,
        pool_ceiling=pool_ceiling,
        anchor_samples=repeats,
        config=power_config,
    )
    return AnchorCalibrationResult(
        evaluation_binding=subset_binding,
        baseline=baseline,
        ceiling=ceiling,
        paired_delta_ci=paired_delta_ci,
        power=power,
    )
