from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dr_store import ObjectStore

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
from whetstone.eval.preview.persisted import load_evaluation_outputs
from whetstone.eval.schema import (
    EvalEvidence,
    EvalFailureEvidence,
    EvalOutputsRecord,
)
from whetstone.eval.config_ref import EvalConfigRef
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.sampling import evaluation_role_for_split

__all__ = [
    "MIN_ANCHOR_PRESENCE_FRACTION",
    "AnchorCalibrationResult",
    "BalancedAnchorSubset",
    "anchor_samples_per_task",
    "balanced_anchor_subset",
    "run_anchor_calibration",
]


#: Lowest overall present-row fraction an anchor evaluation may carry and
#: still calibrate. Real infrastructure loses rows; demanding perfection from
#: it makes a whole calibration hostage to a single lost sample. This is the
#: repo's completeness-floor convention (0.9), applied here as "expect a high
#: presence percentage, then balance", not "require every planned row".
MIN_ANCHOR_PRESENCE_FRACTION = 0.9


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
    #: Samples per task the calibration actually reduced to. It equals the
    #: planned ``num_seeds`` when nothing was lost, and drops to the worst
    #: task's present count when infrastructure lost rows. Both anchors are
    #: balanced to the same ``k`` so the paired delta stays paired.
    samples_per_task: int


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
    if len(evidence.per_task_counts) != len(expected_task_hashes):
        raise ValueError("calibration evidence has incomplete per-task counts")
    if any(count > expected_samples for count in evidence.per_task_counts):
        raise ValueError(
            "calibration evidence reports more present rows than planned "
            "samples"
        )
    # Anchors calibrate the achievable range, so presence must be high --
    # but not perfect. Infrastructure loses rows, and refusing the whole
    # calibration over one lost sample is a bar real runs cannot clear. The
    # floor keeps the anchor honest; balanced subsetting downstream keeps
    # every task contributing the same number of samples.
    planned = expected_samples * len(expected_task_hashes)
    present = sum(evidence.per_task_counts)
    presence = present / planned if planned else 0.0
    if presence < MIN_ANCHOR_PRESENCE_FRACTION:
        raise ValueError(
            "calibration evidence presence "
            f"{presence:.3f} ({present}/{planned} rows) is below the "
            f"required floor of {MIN_ANCHOR_PRESENCE_FRACTION:.2f}"
        )
    # A task with no present row cannot be balanced against the others at
    # any k: it would enter the variance decomposition as a *measured* 0.0
    # and mint a false hard task, biasing the anchor gap and every MDD on
    # the surface. It is rejected here rather than coerced downstream.
    unobserved = tuple(
        task_hash
        for task_hash, count in zip(
            expected_task_hashes, evidence.per_task_counts, strict=True
        )
        if count == 0
    )
    if unobserved:
        raise ValueError(
            "calibration evidence carries an unobserved task with zero "
            f"present rows: {unobserved!r}"
        )
    if any(value is None for value in evidence.per_task_values):
        raise ValueError(
            "calibration evidence carries an unobserved per-task value"
        )
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


@dataclass(frozen=True, slots=True)
class BalancedAnchorSubset:
    """One anchor's per-task values reduced to a common sample count.

    ``samples_per_task`` is the ``k`` every task contributes. ``per_task``
    holds each task's mean over exactly its first ``k`` present rows, in
    ``task_hashes`` order.
    """

    samples_per_task: int
    per_task: tuple[float, ...]


def _present_rows_by_task(
    outputs: EvalOutputsRecord,
    *,
    task_hashes: tuple[str, ...],
) -> tuple[tuple[tuple[int, float], ...], ...]:
    """Group present rows per task as ``(seed_index, score)``, seed-ordered.

    A row is present exactly when it carries a score. A blank generation now
    scores as the failing sample it is, so it is present here and never
    reduces a task's usable count; only genuine infrastructure loss does.
    """
    by_task: dict[str, list[tuple[int, float]]] = {
        task_hash: [] for task_hash in task_hashes
    }
    for row in outputs.outputs:
        if row.score is None:
            continue
        bucket = by_task.get(row.task_hash)
        if bucket is None:
            raise ValueError(
                "calibration outputs carry a row outside the anchor task set"
            )
        bucket.append((row.seed_index, float(row.score)))
    return tuple(
        tuple(sorted(by_task[task_hash])) for task_hash in task_hashes
    )


def anchor_samples_per_task(
    outputs: EvalOutputsRecord,
    *,
    task_hashes: tuple[str, ...],
) -> int:
    """The largest ``k`` every task in this anchor can supply.

    That is the smallest present-row count across tasks: any larger ``k``
    would leave some task short, and balancing exists precisely so no task
    is weighted more heavily merely because infrastructure happened to lose
    fewer of its rows.
    """
    present = _present_rows_by_task(outputs, task_hashes=task_hashes)
    counts = tuple(len(rows) for rows in present)
    if not counts or min(counts) < 1:
        raise ValueError(
            "balanced anchor subsetting requires every task to have at "
            "least one present row"
        )
    return min(counts)


def balanced_anchor_subset(
    outputs: EvalOutputsRecord,
    *,
    task_hashes: tuple[str, ...],
    samples_per_task: int,
) -> BalancedAnchorSubset:
    """Reduce every task to exactly ``samples_per_task`` present samples.

    Which rows are kept is decided by **lowest seed index among present
    rows** -- deterministic and outcome-blind. Selecting on score would let
    the subset choose its own answer: dropping the worst rows would inflate
    an anchor and bias every downstream MDD. Seed index is fixed before any
    generation exists, so the same evidence always yields the same subset.
    """
    if samples_per_task < 1:
        raise ValueError("samples_per_task must be at least 1")
    present = _present_rows_by_task(outputs, task_hashes=task_hashes)
    short = tuple(
        task_hash
        for task_hash, rows in zip(task_hashes, present, strict=True)
        if len(rows) < samples_per_task
    )
    if short:
        raise ValueError(
            f"tasks {short!r} carry fewer than {samples_per_task} present "
            "rows and cannot be balanced at that sample count"
        )
    per_task = tuple(
        sum(score for _seed, score in rows[:samples_per_task])
        / samples_per_task
        for rows in present
    )
    return BalancedAnchorSubset(
        samples_per_task=samples_per_task,
        per_task=per_task,
    )


def run_anchor_calibration(
    *,
    engine: EvalEngine,
    store: ObjectStore,
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

    Presence is required to be high but not perfect: each anchor must clear
    :data:`MIN_ANCHOR_PRESENCE_FRACTION` overall with no wholly unobserved
    task. Both anchors are then reduced to one common samples-per-task ``k``
    so every task contributes equally, and the delta stays paired.
    ``store`` is read for the row-level outputs that balancing needs; the
    aggregated ``per_task_values`` are means over *all* present rows and
    cannot be re-balanced after the fact.
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

    # ``_validate_anchor_evidence`` has cleared the presence floor and
    # rejected any wholly unobserved task, so every task here has at least
    # one present row to balance on.
    #
    # Balance both anchors to one common k. Each anchor's own worst task
    # sets its k, and the pair takes the smaller of the two: the delta is
    # paired across arms, so an arm carrying more samples than its partner
    # would compare unequal amounts of evidence per task.
    baseline_outputs = load_evaluation_outputs(store, baseline_evidence)
    ceiling_outputs = load_evaluation_outputs(store, ceiling_evidence)
    samples_per_task = min(
        anchor_samples_per_task(
            baseline_outputs, task_hashes=calibrated_task_hashes
        ),
        anchor_samples_per_task(
            ceiling_outputs, task_hashes=calibrated_task_hashes
        ),
    )
    baseline_subset = balanced_anchor_subset(
        baseline_outputs,
        task_hashes=calibrated_task_hashes,
        samples_per_task=samples_per_task,
    )
    ceiling_subset = balanced_anchor_subset(
        ceiling_outputs,
        task_hashes=calibrated_task_hashes,
        samples_per_task=samples_per_task,
    )
    if log is not None and samples_per_task != samples:
        log(
            "Balanced anchors to "
            f"{samples_per_task} sample(s) per task "
            f"(planned {samples}); infrastructure lost rows"
        )
    baseline_per_task = baseline_subset.per_task
    ceiling_per_task = ceiling_subset.per_task
    paired_delta_ci = bootstrap_paired_delta_ci(
        baseline_per_task,
        ceiling_per_task,
        level=bootstrap_level,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    power = analyze_power(
        naive_per_task=baseline_per_task,
        ceiling_per_task=ceiling_per_task,
        pool_ceiling=pool_ceiling,
        anchor_samples=samples_per_task,
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
        samples_per_task=samples_per_task,
    )
