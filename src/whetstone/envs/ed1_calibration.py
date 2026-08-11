from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from whetstone.core.roles import EvaluationRole
from whetstone.envs.ed1 import ED1_ENV_NAME, Ed1Experiment
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

ED1_CALIBRATION_BASELINE_PURPOSE = "ed1-calibration-baseline"
ED1_CALIBRATION_CEILING_PURPOSE = "ed1-calibration-ceiling"


@dataclass(frozen=True, slots=True)
class Ed1CalibrationResult:
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
    if evidence.task_identities != expected_task_ids:
        raise ValueError("calibration evidence changed task identity order")
    if evidence.repeat_count != expected_repeats:
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


def run_ed1_calibration(
    *,
    engine: EvaluationEngine,
    evaluation_binding: EvaluationBinding,
    task_ids: tuple[str, ...],
    pool_ceiling: int,
    power_config: PowerConfig | None = None,
    bootstrap_level: float = 0.95,
    bootstrap_resamples: int = DEFAULT_RESAMPLES,
    bootstrap_seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> Ed1CalibrationResult:
    """Evaluate both ED1 anchors on one exact task/repeat binding.

    The returned paired bootstrap is empirical over the aligned per-task
    blended rewards. The current power model treats those bounded rewards as
    pass-rate-like observations, so its recommendation is an approximate
    planning estimate rather than a certification result.
    """
    experiment = engine.experiment
    if not isinstance(experiment, Ed1Experiment) or (
        experiment.env_name != ED1_ENV_NAME
    ):
        raise ValueError("ED1 calibration requires an ED1 EvaluationEngine")
    if evaluation_binding.role is not EvaluationRole.INTERNAL:
        raise ValueError("ED1 calibration requires an internal binding")
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
            "ED1 calibration requires the internal sampling split"
        )
    if pool_ceiling < len(task_ids):
        raise ValueError(
            "pool_ceiling cannot be smaller than calibrated tasks"
        )
    if not 0.0 < bootstrap_level < 1.0:
        raise ValueError("bootstrap_level must be in (0, 1)")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be at least 1")

    subset_engine = engine.for_task_ids(task_ids)
    subset_binding = _subset_binding(evaluation_binding, subset_engine)
    baseline_request = EvaluationRequest(
        candidate=experiment.initial_candidate,
        evaluation_binding=subset_binding,
        purpose=ED1_CALIBRATION_BASELINE_PURPOSE,
    )
    ceiling_request = EvaluationRequest(
        candidate=experiment.ceiling_candidate,
        evaluation_binding=subset_binding,
        purpose=ED1_CALIBRATION_CEILING_PURPOSE,
    )

    # Validate both anchors before the first paid evaluation starts.
    subset_engine.validate_request(baseline_request)
    subset_engine.validate_request(ceiling_request)

    planned_rows = (
        len(task_ids) * subset_engine.sampling.repeat_plan.repeat_count
    )
    if log is not None:
        log(
            "Starting hand-engineered baseline evaluation "
            f"({planned_rows} rows)"
        )
    baseline = subset_engine.evaluate(baseline_request)
    if log is not None:
        accounting = baseline.evidence.row_accounting
        log(
            "Completed hand-engineered baseline evaluation "
            f"(present={accounting.present}/{accounting.planned}, "
            f"missing={accounting.missing}, failed={accounting.failed}, "
            f"invalid={accounting.invalid})"
        )
        log(
            "Starting hand-engineered comparison evaluation "
            f"({planned_rows} rows)"
        )
    ceiling = subset_engine.evaluate(ceiling_request)
    if log is not None:
        accounting = ceiling.evidence.row_accounting
        log(
            "Completed hand-engineered comparison evaluation "
            f"(present={accounting.present}/{accounting.planned}, "
            f"missing={accounting.missing}, failed={accounting.failed}, "
            f"invalid={accounting.invalid})"
        )
    repeats = subset_engine.sampling.repeat_plan.repeat_count
    reward_policy_hash = subset_engine.reward_policy_identity_hash
    for evaluated in (baseline, ceiling):
        _validate_anchor_evidence(
            evaluated=evaluated,
            expected_binding=subset_binding,
            expected_task_ids=task_ids,
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
        anchor_repeats=repeats,
        config=power_config,
    )
    return Ed1CalibrationResult(
        evaluation_binding=subset_binding,
        baseline=baseline,
        ceiling=ceiling,
        paired_delta_ci=paired_delta_ci,
        power=power,
    )


__all__ = [
    "ED1_CALIBRATION_BASELINE_PURPOSE",
    "ED1_CALIBRATION_CEILING_PURPOSE",
    "Ed1CalibrationResult",
    "run_ed1_calibration",
]
