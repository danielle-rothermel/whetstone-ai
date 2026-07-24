"""Checks for the c18h hard-mode env variant.

c18h is a distinct env id that reuses the c18 modules but generates its pool
from c18's ``HARD_PRESET`` -- the hardest upstream-PrOntoQA configuration along
c18's depth + distractor axes (deeper chains D5/D8/D10, distractors ON where
the pinned upstream can honor them; no hidden-information change). These checks
prove: it resolves through the registry to the c18 modules with the hard
preset; its 3 x 20 pool (depths D5/D8/D10) splits to (internal 6, official 18,
held_out 36) with disjoint splits; and its Eval Config / Task Set / dataset
identities are DISTINCT from base c18's, so a c18h cell never collides with a
c18 cell in the ledger.

The c18 pool is reseeded from the vendored PrOntoQA generator (a subprocess
per depth), so a c18h pool build is slower than the in-process envs; the tests
that build the full pool are marked accordingly but kept minimal.
"""

from __future__ import annotations

import pytest

from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.envs.registry import (
    ENV_NAMES,
    ESTIMATE_INHERITED_PENDING,
    env_spec,
)

_MODEL = "deepseek/deepseek-v4-flash"

# The c18h pool reseeds the vendored PrOntoQA generator through a subprocess
# per depth, so a FULL-N (20/stratum) pool costs ~18s. Every property these
# tests assert -- registry resolution, split disjointness, identity hashing,
# procedure partition -- is N-INDEPENDENT, so they build a tiny pool (shared
# once per session via the fixtures below). The one test that pins the
# FULL-N pool shape (60 instances, 20 per depth) regenerates the real pool
# and carries @pytest.mark.slow (deselected by default; run with -m slow).
_TINY_N = 2
# A proportional (1, 2, 3) split of the tiny 3-stratum pool (6 instances):
# mirrors the full (6, 18, 36) split's ratios with no leftovers, so
# disjointness/ordering (N-independent) is exercised without full generation.
_TINY_SPLIT = (1, 2, 3)


@pytest.fixture(scope="session")
def c18h_tiny() -> EnvExperiment:
    """A tiny-N c18h experiment, built once and shared across the session."""
    return build_env_experiment(
        "c18h",
        model=_MODEL,
        pool_n_per_stratum=_TINY_N,
        split_sizes=_TINY_SPLIT,
    )


@pytest.fixture(scope="session")
def c18_tiny() -> EnvExperiment:
    """A tiny-N base-c18 experiment (for the variant-vs-base comparisons)."""
    return build_env_experiment(
        "c18",
        model=_MODEL,
        pool_n_per_stratum=_TINY_N,
        split_sizes=_TINY_SPLIT,
    )


def test_c18h_is_a_bound_env_that_resolves() -> None:
    assert "c18h" in ENV_NAMES
    env = env_spec("c18h")
    assert env.name == "c18h"
    # Reuses the c18 modules (mapped module name), not a c18h package.
    assert env.generate.__name__ == "whetstone_envs.c18.generate"
    assert env.oracle.__name__ == "whetstone_envs.c18.oracle"
    assert env.oracle_qualname == "whetstone_envs.c18.oracle.score_gold"
    # c18's oracle is the usual (prediction, gold) order, and its pool
    # interleaves strata (not blocked) -- so c18h is neither gold-first nor
    # stratified, unlike c22h.
    assert env.gold_first is False
    assert env.stratified_split is False
    # Inherited token estimate pending its own pilot (c18 measured x1.5).
    assert env.token_estimate.estimate_source == ESTIMATE_INHERITED_PENDING
    assert env.token_estimate.naive == 1959
    assert env.token_estimate.ceiling == 3672


def test_c18h_pool_is_the_deep_hard_preset() -> None:
    # Composition (N-independent): three balanced D5/D8/D10 strata, every
    # gold a True/False entailment label (the c18 task shape). Exercised on a
    # tiny pool; the exact full-N counts are pinned by the slow test below.
    env = env_spec("c18h")
    pool = env.generate_pool(n_per_stratum=_TINY_N)
    assert pool.stratum_counts() == {
        "D5": _TINY_N,
        "D8": _TINY_N,
        "D10": _TINY_N,
    }
    for inst in pool.instances:
        assert inst.gold in {"True", "False"}


@pytest.mark.slow
def test_c18h_full_pool_is_sixty_across_three_deep_strata() -> None:
    # The FULL-N pool shape: 60 instances, 20 per deep stratum. Regenerates
    # the real (slow) hard pool -- the ONE test that pays full generation to
    # pin the committed pool size; every other property is N-independent.
    env = env_spec("c18h")
    pool = env.generate_pool()
    assert len(pool) == 60
    assert pool.stratum_counts() == {"D5": 20, "D8": 20, "D10": 20}
    for inst in pool.instances:
        assert inst.gold in {"True", "False"}


def test_c18h_dataset_revision_is_the_hard_generator_version() -> None:
    assert env_spec("c18h").generator_version == "c18-generate-1+hard"
    assert env_spec("c18").generator_version == "c18-generate-1"


def test_c18h_split_is_six_eighteen_thirtysix() -> None:
    # default_split_sizes scales the fixed per-stratum split (2, 6, 12) by the
    # stratum count (3) -- independent of N -- so a tiny pool yields the same
    # committed (6, 18, 36) whole-pool split as the full pool.
    env = env_spec("c18h")
    pool = env.generate_pool(n_per_stratum=_TINY_N)
    assert env.default_split_sizes(pool) == (6, 18, 36)


def test_c18h_splits_are_disjoint_and_holdout_untouched(
    c18h_tiny: EnvExperiment,
) -> None:
    # Disjointness of the three splits and a non-empty held-out set are
    # N-independent; the tiny (1, 2, 3) split exercises them without full
    # generation. (The full-N (6, 18, 36) sizes are pinned by the pool-shape
    # and split-arithmetic tests above.)
    exp = c18h_tiny
    internal = set(exp.eval_configs.internal.task_set.task_identities)
    official = set(exp.eval_configs.official.task_set.task_identities)
    held_out = set(exp.eval_configs.held_out_task_identities)
    assert len(internal) == _TINY_SPLIT[0]
    assert len(official) == _TINY_SPLIT[1]
    assert len(held_out) == _TINY_SPLIT[2]
    assert internal.isdisjoint(official)
    assert internal.isdisjoint(held_out)
    assert official.isdisjoint(held_out)


def test_c18h_eval_config_hash_differs_from_c18(
    c18_tiny: EnvExperiment,
    c18h_tiny: EnvExperiment,
) -> None:
    # The eval_config identity folds in the dataset revision + env name +
    # seed range, none of which is N -- so the c18-vs-c18h hash distinction
    # holds at any pool size. Exercised on the shared tiny experiments.
    for role in ("internal", "official"):
        c18_hash = getattr(
            c18_tiny.eval_configs, role
        ).eval_config.config_identity_hash
        c18h_hash = getattr(
            c18h_tiny.eval_configs, role
        ).eval_config.config_identity_hash
        assert c18_hash != c18h_hash, f"{role} eval_config_hash collides"


def test_c18h_task_identities_are_disjoint_from_c18(
    c18_tiny: EnvExperiment,
    c18h_tiny: EnvExperiment,
) -> None:
    # Distinct env name + distinct seed range (2_000_000_000 vs the base
    # 1_000_000_000 window) => no shared task identity, so a c18h row never
    # aliases a c18 row in the ledger. The seed windows are disjoint at any N,
    # so the shared tiny experiments prove it.
    c18_ids = set(
        c18_tiny.eval_configs.official.task_set.task_identities
    ) | set(c18_tiny.eval_configs.internal.task_set.task_identities)
    c18h_ids = set(
        c18h_tiny.eval_configs.official.task_set.task_identities
    ) | set(c18h_tiny.eval_configs.internal.task_set.task_identities)
    assert c18_ids.isdisjoint(c18h_ids)


def test_c18h_procedure_identity_partition_holds(
    c18h_tiny: EnvExperiment,
) -> None:
    # The Rollout Definition and both Eval Configs share one Procedure id --
    # a construction-time partition independent of pool size.
    assert (
        c18h_tiny.rollout_definition.procedure_config_hash
        == c18h_tiny.eval_configs.procedure_config_hash
    )


@pytest.mark.parametrize("model", ["openai/gpt-5-nano", _MODEL])
def test_c18h_builds_under_both_task_models(model: str) -> None:
    # The pilot exercises both task models; the cell build must succeed under
    # either (the task model folds into the route's Provider Call Config).
    # The build succeeds regardless of pool size, so a tiny pool suffices.
    exp = build_env_experiment(
        "c18h",
        model=model,
        pool_n_per_stratum=_TINY_N,
        split_sizes=_TINY_SPLIT,
    )
    assert exp.env_name == "c18h"
