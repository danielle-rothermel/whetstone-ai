from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from whetstone.core.roles import EvalRole
from whetstone.eval.analysis.power import (
    PowerConfig,
    PowerResult,
    analyze_power,
)
from whetstone.eval.analysis.statistics import (
    DEFAULT_RESAMPLES,
    BootstrapCI,
    bootstrap_paired_delta_ci,
)
from whetstone.eval.metadata import metadata_with_purpose
from whetstone.eval.protocol import (
    EvalEvidenceWithRef,
    EvalRequest,
    EvalResult,
    EvalEngine,
    eval_is_rejected,
    eval_is_success,
)
from whetstone.eval.schema import EvalEvidence, EvalFailureEvidence
from whetstone.experiment.binding import EvalConfigRef
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.sampling import evaluation_role_for_split

__all__ = [
    "AnchorCalibrationResult",
    "run_anchor_calibration",
]


@dataclass(frozen=True, slots=True)
class AnchorCalibrationResult:
    """One naive/ceiling anchor pair calibrated on a single evaluation role.

    ``eval_role`` and ``split_role`` record which split the anchors were
    measured on. Anchors calibrate on any role — the Stage-0 power analysis
    inverts the MDE on the held-out split, which is the split the study
    reports from — and both anchors of a result always share one role.
    """

    eval_config_ref: EvalConfigRef
    eval_role: EvalRole
    split_role: str
    baseline: EvalEvidenceWithRef
    ceiling: EvalEvidenceWithRef
    paired_delta_ci: BootstrapCI
    power: PowerResult


def _require_success_eval(
    result: EvalResult,
    *,
    label: str,
) -> EvalEvidenceWithRef:
    if eval_is_rejected(result):
        raise ValueError(f"{label} rejected: {result.detail.message}")
    if isinstance(result, EvalEvidenceWithRef) and isinstance(
        result.evidence, EvalFailureEvidence
    ):
        raise ValueError(f"{label} failed: {result.evidence.message}")
    if not eval_is_success(result):
        raise TypeError(f"unexpected evaluation result for {label}: {result!r}")
    return result


def _validate_anchor_evidence(
    *,
    evaluated: EvalEvidenceWithRef,
    expected_eval_config_ref: EvalConfigRef,
    expected_eval_role: EvalRole,
    expected_task_hashes: tuple[str, ...],
    expected_samples: int,
    expected_reward_policy_hash: str | None,
) -> None:
    if not isinstance(evaluated.evidence, EvalEvidence):
        raise ValueError("calibration requires successful evaluation evidence")
    evidence = evaluated.evidence
    if evidence.eval_config_ref != expected_eval_config_ref:
        raise ValueError("calibration evidence changed its Eval Config")
    if evidence.eval_role is not expected_eval_role:
        raise ValueError("calibration evidence changed its Evaluation Role")
    if evidence.task_hashes != expected_task_hashes:
        raise ValueError("calibration evidence changed task identity order")
    if evidence.num_seeds != expected_samples:
        raise ValueError("calibration evidence changed sample count")
    if len(evidence.per_task_values) != len(expected_task_hashes):
        raise ValueError("calibration evidence has incomplete per-task values")
    if evidence.per_task_counts != (expected_samples,) * len(
        expected_task_hashes
    ):
        raise ValueError("calibration evidence changed per-task sample counts")
    if evidence.row_accounting.planned != (
        len(expected_task_hashes) * expected_samples
    ):
        raise ValueError("calibration evidence changed planned row accounting")
    if expected_reward_policy_hash is None:
        # Reward evidence is minted only on the internal role, so a non-internal
        # anchor must carry no reward reference. One that does means the
        # producer's role gate and this consumer's have drifted apart, and the
        # reward would be attributed to a role that never earned it.
        if evidence.reward_ref is not None:
            raise ValueError(
                "calibration evidence carries reward evidence on a "
                "non-internal role"
            )
        return
    if evidence.reward_ref is None:
        raise ValueError("calibration requires internal reward evidence")
    observed_policy_hash = (
        evidence.reward_ref.record.reward_policy.identity_hash()
    )
    if observed_policy_hash != expected_reward_policy_hash:
        raise ValueError("calibration evidence changed its Reward Policy")


def run_anchor_calibration(
    *,
    engine: EvalEngine,
    baseline_candidate: Candidate,
    ceiling_candidate: Candidate,
    baseline_purpose: str,
    ceiling_purpose: str,
    task_ids: tuple[str, ...],
    pool_ceiling: int,
    eval_role: EvalRole | None = None,
    power_config: PowerConfig | None = None,
    bootstrap_level: float = 0.95,
    bootstrap_resamples: int = DEFAULT_RESAMPLES,
    bootstrap_seed: int = 0,
    baseline_log_label: str = "baseline anchor",
    ceiling_log_label: str = "comparison anchor",
    log: Callable[[str], None] | None = None,
) -> AnchorCalibrationResult:
    """Evaluate a naive/ceiling anchor pair on one split and analyze its power.

    The anchors calibrate on whichever split ``engine`` is bound to. Pass
    ``eval_role`` to state the expected role explicitly; it must be the role
    the split owns, which makes an accidentally mis-bound engine a loud error
    rather than a silently relabelled calibration. Both anchors are validated
    against the same role, Eval Config, task identity, and sample count.
    """
    split_role = engine.sampling.split_role
    expected_eval_role = evaluation_role_for_split(split_role)
    if eval_role is not None and eval_role is not expected_eval_role:
        raise ValueError(
            f"anchor calibration was asked for evaluation role "
            f"{eval_role.value!r} but the engine is bound to split "
            f"{split_role!r} ({expected_eval_role.value!r})"
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
    calibrated_task_hashes = subset_engine.sampling.task_hashes
    baseline_request = EvalRequest(
        request_id=f"calibration:{baseline_purpose}",
        candidate=baseline_candidate,
        metadata=metadata_with_purpose(baseline_purpose),
    )
    ceiling_request = EvalRequest(
        request_id=f"calibration:{ceiling_purpose}",
        candidate=ceiling_candidate,
        metadata=metadata_with_purpose(ceiling_purpose),
    )

    planned_rows = (
        len(task_ids) * subset_engine.sampling.num_seeds
    )
    if log is not None:
        log(f"Starting {baseline_log_label} evaluation ({planned_rows} rows)")
    baseline = _require_success_eval(
        subset_engine.evaluate(baseline_request),
        label=baseline_log_label,
    )
    if log is not None:
        accounting = baseline.evidence.row_accounting
        log(
            f"Completed {baseline_log_label} evaluation "
            f"(present={accounting.present}/{accounting.planned}, "
            f"missing={accounting.missing}, failed={accounting.failed}, "
            f"invalid={accounting.invalid})"
        )
        log(f"Starting {ceiling_log_label} evaluation ({planned_rows} rows)")
    ceiling = _require_success_eval(
        subset_engine.evaluate(ceiling_request),
        label=ceiling_log_label,
    )
    if log is not None:
        accounting = ceiling.evidence.row_accounting
        log(
            f"Completed {ceiling_log_label} evaluation "
            f"(present={accounting.present}/{accounting.planned}, "
            f"missing={accounting.missing}, failed={accounting.failed}, "
            f"invalid={accounting.invalid})"
        )
    samples = subset_engine.sampling.num_seeds
    reward_policy_hash = (
        subset_engine.reward_policy_identity_hash()
        if expected_eval_role is EvalRole.INTERNAL
        else None
    )
    expected_eval_config_ref = subset_engine.eval_config_ref
    for evaluated in (baseline, ceiling):
        _validate_anchor_evidence(
            evaluated=evaluated,
            expected_eval_config_ref=expected_eval_config_ref,
            expected_eval_role=expected_eval_role,
            expected_task_hashes=calibrated_task_hashes,
            expected_samples=samples,
            expected_reward_policy_hash=reward_policy_hash,
        )
    baseline_evidence = baseline.evidence
    ceiling_evidence = ceiling.evidence
    assert isinstance(baseline_evidence, EvalEvidence)
    assert isinstance(ceiling_evidence, EvalEvidence)
    if baseline_evidence.graph_hash != ceiling_evidence.graph_hash:
        raise ValueError("calibration anchors changed graph identity")

    paired_delta_ci = bootstrap_paired_delta_ci(
        baseline_evidence.per_task_values,
        ceiling_evidence.per_task_values,
        level=bootstrap_level,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    power = analyze_power(
        naive_per_task=baseline_evidence.per_task_values,
        ceiling_per_task=ceiling_evidence.per_task_values,
        pool_ceiling=pool_ceiling,
        anchor_samples=samples,
        config=power_config,
    )
    return AnchorCalibrationResult(
        eval_config_ref=expected_eval_config_ref,
        eval_role=expected_eval_role,
        split_role=split_role,
        baseline=baseline,
        ceiling=ceiling,
        paired_delta_ci=paired_delta_ci,
        power=power,
    )
