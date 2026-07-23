"""Sampling Configs, Task Sets, and the composite internal/official Eval
Configs from a TaskPool's splits.

The env's committed pool split (``EnvSpec.default_split_sizes``) carves the
pool into three ordered, disjoint subsets: ``internal_eval`` (optimizer
feedback), ``official`` (before/after comparison), and ``held_out``
(untouched). This module maps the two *used* splits onto dr-code Task Sets +
Repeat Plans + Sampling Configs, and assembles the two composite Eval Configs
that share the **exact same** Evaluation Procedure Config identity.

Guarantees, per the validation-plan cell definition:

* The internal and official Task Sets are **ordered** (ordering is
  identity-bearing) and **disjoint** (their task identities never overlap --
  the pool split already asserts disjointness by instance id, and this module
  re-asserts it over task identities).
* ``held_out`` is never referenced by any Sampling Config built here (proved
  by a test that no built config's task identities intersect the held-out
  set).
* Both Eval Configs fold in **one** Evaluation Procedure Config identity, so
  ``graph_hash`` is unchanged across the two while ``eval_config_hash``
  differs (their Sampling Configs differ).
* The Aggregation Config is ``mean`` with an explicit completeness policy
  (``missing_data`` propagate/skip, ``zero_denominator`` not_applicable).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from dr_code.eval import (
    AggregationConfig,
    EvalConfig,
    EvalDefinition,
    EvaluationProcedureConfig,
    RepeatPlan,
    SamplingConfig,
    SamplingDefinition,
    TaskSet,
)
from whetstone_envs.core import Instance, PoolSplit, TaskPool

from whetstone.code_eval.aggregate import (
    CompletenessPolicy,
    RowPolicy,
    aggregation_definition,
)
from whetstone.envs.registry import DEFAULT_REPEATS, EnvSpec
from whetstone.envs.task import EnvTask

_DEFINITION_VERSION = "1"

#: The split roles this adapter samples from. ``held_out`` is deliberately
#: absent: no Sampling Config references it.
INTERNAL_EVAL = "internal_eval"
OFFICIAL = "official"


class Completeness(StrEnum):
    """The Aggregation Config completeness policy over planned rows.

    ``PROPAGATE`` (default): any missing/failed row makes the aggregate
    incomplete (the mean is not reported over an incomplete matrix).
    ``SKIP``: excluded rows are dropped from the reduction but their
    exclusion is still counted in provenance -- bounded by a declared
    ``max_skip_fraction`` completeness tolerance (see
    :func:`build_aggregation_config`): beyond the bound the aggregate is
    forced incomplete rather than certified over an out-of-tolerance matrix.
    """

    PROPAGATE = "propagate"
    SKIP = "skip"

    def to_policy(
        self, *, max_skip_fraction: float = 0.0
    ) -> CompletenessPolicy:
        """The :class:`CompletenessPolicy` this enum + tolerance denotes."""
        row_policy = (
            RowPolicy.PROPAGATE
            if self is Completeness.PROPAGATE
            else RowPolicy.SKIP
        )
        return CompletenessPolicy(
            row_policy=row_policy, max_skip_fraction=max_skip_fraction
        )


def _dataset_revision(env: EnvSpec) -> str:
    """The env pool's generator version -- the Task Set dataset revision.

    Reads :attr:`EnvSpec.generator_version` (not the module's constant) so a
    preset variant (c22h -> ``c22-generate-1+hard``) records a DISTINCT dataset
    revision from its base env even though both load the same module.
    """
    return str(env.generator_version)


def task_identities(env: EnvSpec, instances: tuple[Instance, ...]) -> tuple[
    str, ...
]:
    """The ordered task identities for ``instances`` (pool order preserved)."""
    return tuple(
        EnvTask.from_instance(env.name, inst).task_identity()
        for inst in instances
    )


def build_task_set(
    env: EnvSpec,
    *,
    split_role: str,
    instances: tuple[Instance, ...],
) -> TaskSet:
    """A versioned, ordered Task Set manifest for one split's instances.

    The ordering (pool order) and the dataset revision (the env generator
    version) are identity-bearing, so the internal and official Task Sets are
    distinct identities even though they draw from one pool.
    """
    return TaskSet(
        manifest_id=f"whetstone.env.{env.name}.{split_role}",
        version=_DEFINITION_VERSION,
        dataset_revision=_dataset_revision(env),
        task_identities=task_identities(env, instances),
    )


def build_repeat_plan(
    env: EnvSpec,
    *,
    split_role: str,
    task_set: TaskSet,
    repeats: int = DEFAULT_REPEATS,
) -> RepeatPlan:
    """The Repeat Plan: ``repeats`` ordered slots per task in the Task Set.

    Per-slot RNG seeds are slot data (excluded from Repeat Plan identity);
    the spec-default repeat count is :data:`DEFAULT_REPEATS`.
    """
    return RepeatPlan(
        plan_id=f"whetstone.env.{env.name}.{split_role}",
        version=_DEFINITION_VERSION,
        task_identities=task_set.task_identities,
        repeat_count=repeats,
    )


def build_sampling_config(
    env: EnvSpec,
    *,
    split_role: str,
    task_set: TaskSet,
    repeat_plan: RepeatPlan,
) -> SamplingConfig:
    """The Sampling Config binding this split's Task Set + Repeat Plan."""
    definition = SamplingDefinition(
        definition_id=f"whetstone.env.{env.name}.{split_role}.sampling",
        version=_DEFINITION_VERSION,
    )
    return definition.materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "repeat_plan_hash": repeat_plan.identity_hash(),
        }
    )


def build_aggregation_config(
    env: EnvSpec,
    *,
    completeness: Completeness = Completeness.PROPAGATE,
    max_skip_fraction: float = 0.0,
) -> AggregationConfig:
    """The ``mean`` Aggregation Config with an explicit completeness policy.

    ``reduction=mean`` with ``missing_data`` set from ``completeness``,
    ``zero_denominator=not_applicable`` (an empty reduction is an explicit
    non-OK status, never a fabricated value), and an identity-bearing
    ``max_skip_fraction`` completeness tolerance. The tolerance folds into the
    config identity, so a tolerant SKIP config has a DISTINCT
    ``eval_config_hash`` from an untolerant one (or from PROPAGATE). Under
    ``PROPAGATE`` the bound is inert but still declared (and defaults ``0.0``,
    preserving the legacy identity of untolerant configs).
    """
    policy = completeness.to_policy(max_skip_fraction=max_skip_fraction)
    return aggregation_definition(
        f"whetstone.env.{env.name}.aggregation"
    ).materialize(
        {
            "reduction": "mean",
            "missing_data": policy.missing_data,
            "zero_denominator": "not_applicable",
            "max_skip_fraction": policy.skip_fraction_token(),
        }
    )


def build_eval_config(
    env: EnvSpec,
    *,
    split_role: str,
    sampling: SamplingConfig,
    procedure: EvaluationProcedureConfig,
    aggregation: AggregationConfig,
) -> EvalConfig:
    """Compose one split's Eval Config from its three component Configs.

    The Evaluation Procedure Config is shared across the internal and
    official Eval Configs (same identity); only the Sampling Config differs,
    so ``graph_hash`` is unchanged while ``eval_config_hash`` differs.
    """
    definition = EvalDefinition(
        definition_id=f"whetstone.env.{env.name}.eval",
        version=_DEFINITION_VERSION,
    )
    return definition.materialize(
        sampling=sampling,
        evaluation_procedure=procedure,
        aggregation=aggregation,
    )


@dataclass(frozen=True, slots=True)
class SamplingOverrides:
    """Reduced-sampling overrides for the OFFICIAL split's Sampling Config.

    Both are identity-bearing per the Sampling Config contract, so a cell that
    passes either gets a DIFFERENT official ``eval_config_hash`` than the
    spec-default (full) config -- the composite Eval Config Identity Hash folds
    in the official Task Set (ordering + membership) and Repeat Plan repeat
    count.

    * ``official_n`` selects a deterministic ordered subset of the official
      Task Set: the FIRST-N instances in pool (Task Set) order. First-N is the
      selection rule so the subset is reproducible and identity-bearing (a
      different N -> a different Task Set identity -> a different
      ``eval_config_hash``). ``None`` keeps the full official split.
    * ``official_repeats`` overrides the official Repeat Plan's repeat count
      (the internal split is untouched). ``None`` keeps the spec-default.
    """

    official_n: int | None = None
    official_repeats: int | None = None

    def is_noop(self) -> bool:
        """True when neither override is set (spec-default sampling)."""
        return self.official_n is None and self.official_repeats is None


@dataclass(frozen=True, slots=True)
class EnvSplitSampling:
    """The sampling artifacts for one split (internal_eval or official)."""

    split_role: str
    instances: tuple[Instance, ...]
    task_set: TaskSet
    repeat_plan: RepeatPlan
    sampling_config: SamplingConfig
    eval_config: EvalConfig


@dataclass(frozen=True, slots=True)
class EnvEvalConfigs:
    """The internal + official Eval Configs and their shared Procedure.

    ``procedure_config_hash`` is the single Evaluation Procedure Config
    identity both Eval Configs fold in. ``held_out`` instances are retained
    for the untouched-held-out proof but never sampled.
    """

    env_name: str
    procedure_config_hash: str
    internal: EnvSplitSampling
    official: EnvSplitSampling
    held_out_task_identities: tuple[str, ...]

    def eval_config_for(self, split_role: str) -> EvalConfig:
        if split_role == INTERNAL_EVAL:
            return self.internal.eval_config
        if split_role == OFFICIAL:
            return self.official.eval_config
        raise KeyError(f"no eval config for split role {split_role!r}")


def _per_stratum_quota(total: int, n_strata: int) -> list[int]:
    """Distribute ``total`` picks across ``n_strata`` as evenly as possible.

    Each stratum gets ``total // n_strata``; the first ``total % n_strata``
    strata (in pool order) get one extra. The result sums to ``total`` and no
    two quotas differ by more than one, so a blocked pool is sampled evenly
    across strata rather than skewed toward the leading blocks.
    """
    if n_strata <= 0:  # pragma: no cover - a pool always has >= 1 stratum
        return []
    base, remainder = divmod(total, n_strata)
    return [base + (1 if i < remainder else 0) for i in range(n_strata)]


def stratified_split(
    pool: TaskPool,
    internal_n: int,
    official_n: int,
    held_out_n: int,
) -> PoolSplit:
    """A per-stratum (stratified) analogue of ``TaskPool.split``.

    ``TaskPool.split`` takes three *contiguous* slices in pool order, which is
    only balanced when the pool interleaves its strata. c22's pool is
    **blocked** (all ``n3_easy`` instances first, then all ``n3_mixed``, ...),
    so a contiguous split would put the whole internal_eval slice in the single
    easiest stratum and drop the hardest strata into the unused remainder tail
    (build-report judgment call #2's balance claim fails for c22).

    This builds the same three disjoint subsets but samples each stratum
    independently: for every stratum (in first-seen pool order) it takes the
    first ``internal_per`` instances into internal_eval, the next
    ``official_per`` into official, and the next ``held_out_per`` into
    held_out, where the per-stratum quotas distribute the requested totals
    evenly across strata (:func:`_per_stratum_quota`). The result is
    per-stratum balanced for internal / official / held_out, and held_out stays
    disjoint from the two sampled splits.

    Assumes each instance carries exactly one stratum label (true for c22);
    the totals must not exceed what the per-stratum quotas can draw.
    """
    strata = pool.strata
    n_strata = len(strata)
    internal_q = _per_stratum_quota(internal_n, n_strata)
    official_q = _per_stratum_quota(official_n, n_strata)
    held_out_q = _per_stratum_quota(held_out_n, n_strata)

    internal: list[Instance] = []
    official: list[Instance] = []
    held_out: list[Instance] = []
    for i, label in enumerate(strata):
        members = pool.in_stratum(label)
        need = internal_q[i] + official_q[i] + held_out_q[i]
        if need > len(members):
            msg = (
                f"stratified split needs {need} instances from stratum "
                f"{label!r} but it has only {len(members)}"
            )
            raise ValueError(msg)
        cut1 = internal_q[i]
        cut2 = cut1 + official_q[i]
        cut3 = cut2 + held_out_q[i]
        internal.extend(members[:cut1])
        official.extend(members[cut1:cut2])
        held_out.extend(members[cut2:cut3])
    return PoolSplit(
        internal_eval=tuple(internal),
        official=tuple(official),
        held_out=tuple(held_out),
    )


def _split(
    env: EnvSpec,
    pool: TaskPool,
    split_sizes: tuple[int, int, int] | None,
) -> PoolSplit:
    if split_sizes is None:
        internal_n, official_n, held_out_n = env.default_split_sizes(pool)
    else:
        internal_n, official_n, held_out_n = split_sizes
    if env.stratified_split:
        return stratified_split(pool, internal_n, official_n, held_out_n)
    return pool.split(internal_n, official_n, held_out_n)


def _build_split_sampling(
    env: EnvSpec,
    *,
    split_role: str,
    instances: tuple[Instance, ...],
    procedure: EvaluationProcedureConfig,
    aggregation: AggregationConfig,
    repeats: int,
) -> EnvSplitSampling:
    task_set = build_task_set(
        env, split_role=split_role, instances=instances
    )
    repeat_plan = build_repeat_plan(
        env, split_role=split_role, task_set=task_set, repeats=repeats
    )
    sampling = build_sampling_config(
        env,
        split_role=split_role,
        task_set=task_set,
        repeat_plan=repeat_plan,
    )
    eval_config = build_eval_config(
        env,
        split_role=split_role,
        sampling=sampling,
        procedure=procedure,
        aggregation=aggregation,
    )
    return EnvSplitSampling(
        split_role=split_role,
        instances=instances,
        task_set=task_set,
        repeat_plan=repeat_plan,
        sampling_config=sampling,
        eval_config=eval_config,
    )


class HeldOutReferencedError(AssertionError):
    """A Sampling Config referenced a held-out task identity."""


class SplitOverlapError(AssertionError):
    """The internal and official Task Sets share a task identity."""


def build_eval_configs(
    env: EnvSpec,
    *,
    pool: TaskPool,
    procedure: EvaluationProcedureConfig,
    completeness: Completeness = Completeness.PROPAGATE,
    max_skip_fraction: float = 0.0,
    repeats: int = DEFAULT_REPEATS,
    split_sizes: tuple[int, int, int] | None = None,
    overrides: SamplingOverrides | None = None,
) -> EnvEvalConfigs:
    """Build the internal + official Eval Configs from ``pool``'s splits.

    Both Eval Configs share ``procedure``'s identity; their Sampling Configs
    differ. Asserts the internal and official Task Sets are disjoint and that
    neither references a held-out task identity (held-out stays untouched).

    ``split_sizes`` defaults to the env's committed spec-default split
    (:meth:`EnvSpec.default_split_sizes`); tests pass an explicit tiny
    ``(internal, official, held_out)`` split for a small pool.

    ``overrides`` (a :class:`SamplingOverrides`) reduces the OFFICIAL split's
    sampling only: ``official_n`` takes the first-N of the ordered official
    Task Set (identity-bearing subset), ``official_repeats`` overrides the
    official Repeat Plan count. Both change the official ``eval_config_hash``;
    the internal split is never affected. ``None`` keeps the spec-default.
    """
    split = _split(env, pool, split_sizes)
    aggregation = build_aggregation_config(
        env, completeness=completeness, max_skip_fraction=max_skip_fraction
    )

    ov = overrides or SamplingOverrides()
    official_instances = split.official
    if ov.official_n is not None:
        if ov.official_n < 1:
            raise ValueError(
                f"official_n override must be >= 1; got {ov.official_n}"
            )
        if ov.official_n > len(official_instances):
            raise ValueError(
                f"official_n override {ov.official_n} exceeds the official "
                f"split size {len(official_instances)}"
            )
        # Deterministic ordered-subset selection: first-N in Task Set order.
        official_instances = official_instances[: ov.official_n]
    official_repeats = (
        ov.official_repeats if ov.official_repeats is not None else repeats
    )
    if official_repeats < 1:
        raise ValueError(
            f"official_repeats override must be >= 1; got {official_repeats}"
        )

    internal = _build_split_sampling(
        env,
        split_role=INTERNAL_EVAL,
        instances=split.internal_eval,
        procedure=procedure,
        aggregation=aggregation,
        repeats=repeats,
    )
    official = _build_split_sampling(
        env,
        split_role=OFFICIAL,
        instances=official_instances,
        procedure=procedure,
        aggregation=aggregation,
        repeats=official_repeats,
    )

    internal_ids = set(internal.task_set.task_identities)
    official_ids = set(official.task_set.task_identities)
    if internal_ids & official_ids:
        raise SplitOverlapError(
            "internal and official Task Sets share a task identity"
        )

    held_out_ids = task_identities(env, split.held_out)
    held_out_set = set(held_out_ids)
    if (internal_ids | official_ids) & held_out_set:
        raise HeldOutReferencedError(
            "a Sampling Config references a held-out task identity"
        )

    return EnvEvalConfigs(
        env_name=env.name,
        procedure_config_hash=procedure.config_identity_hash,
        internal=internal,
        official=official,
        held_out_task_identities=held_out_ids,
    )


__all__ = [
    "INTERNAL_EVAL",
    "OFFICIAL",
    "Completeness",
    "EnvEvalConfigs",
    "EnvSplitSampling",
    "HeldOutReferencedError",
    "SamplingOverrides",
    "SplitOverlapError",
    "build_aggregation_config",
    "build_eval_config",
    "build_eval_configs",
    "build_repeat_plan",
    "build_sampling_config",
    "build_task_set",
    "task_identities",
]
