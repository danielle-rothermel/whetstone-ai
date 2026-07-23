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

from whetstone.envs.factory import build_env_experiment
from whetstone.envs.registry import (
    ENV_NAMES,
    ESTIMATE_INHERITED_PENDING,
    env_spec,
)

_MODEL = "deepseek/deepseek-v4-flash"


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
    env = env_spec("c18h")
    pool = env.generate_pool()
    assert len(pool) == 60
    assert pool.stratum_counts() == {"D5": 20, "D8": 20, "D10": 20}
    # Every gold is a True/False entailment label (the c18 task shape).
    for inst in pool.instances:
        assert inst.gold in {"True", "False"}


def test_c18h_dataset_revision_is_the_hard_generator_version() -> None:
    assert env_spec("c18h").generator_version == "c18-generate-1+hard"
    assert env_spec("c18").generator_version == "c18-generate-1"


def test_c18h_split_is_six_eighteen_thirtysix() -> None:
    env = env_spec("c18h")
    pool = env.generate_pool()
    assert env.default_split_sizes(pool) == (6, 18, 36)


def test_c18h_splits_are_disjoint_and_holdout_untouched() -> None:
    exp = build_env_experiment("c18h", model=_MODEL)
    internal = set(exp.eval_configs.internal.task_set.task_identities)
    official = set(exp.eval_configs.official.task_set.task_identities)
    held_out = set(exp.eval_configs.held_out_task_identities)
    assert len(internal) == 6
    assert len(official) == 18
    assert len(held_out) == 36
    assert internal.isdisjoint(official)
    assert internal.isdisjoint(held_out)
    assert official.isdisjoint(held_out)


def test_c18h_eval_config_hash_differs_from_c18() -> None:
    c18 = build_env_experiment("c18", model=_MODEL)
    c18h = build_env_experiment("c18h", model=_MODEL)
    for role in ("internal", "official"):
        c18_hash = getattr(
            c18.eval_configs, role
        ).eval_config.config_identity_hash
        c18h_hash = getattr(
            c18h.eval_configs, role
        ).eval_config.config_identity_hash
        assert c18_hash != c18h_hash, f"{role} eval_config_hash collides"


def test_c18h_task_identities_are_disjoint_from_c18() -> None:
    # Distinct env name + distinct seed range (2_000_000_000 vs the base
    # 1_000_000_000 window) => no shared task identity, so a c18h row never
    # aliases a c18 row in the ledger.
    c18 = build_env_experiment("c18", model=_MODEL)
    c18h = build_env_experiment("c18h", model=_MODEL)
    c18_ids = set(c18.eval_configs.official.task_set.task_identities) | set(
        c18.eval_configs.internal.task_set.task_identities
    )
    c18h_ids = set(c18h.eval_configs.official.task_set.task_identities) | set(
        c18h.eval_configs.internal.task_set.task_identities
    )
    assert c18_ids.isdisjoint(c18h_ids)


def test_c18h_procedure_identity_partition_holds() -> None:
    exp = build_env_experiment("c18h", model=_MODEL)
    assert (
        exp.rollout_definition.procedure_config_hash
        == exp.eval_configs.procedure_config_hash
    )


@pytest.mark.parametrize("model", ["openai/gpt-5-nano", _MODEL])
def test_c18h_builds_under_both_task_models(model: str) -> None:
    # The pilot exercises both task models; the cell build must succeed under
    # either (the task model folds into the route's Provider Call Config).
    exp = build_env_experiment("c18h", model=model)
    assert exp.env_name == "c18h"
