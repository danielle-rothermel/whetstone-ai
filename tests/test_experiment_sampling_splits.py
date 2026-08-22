"""The three-role eval-split contract and its mechanical leakage check.

Splits are derived from an explicitly seeded, strata-shaped task pool so
every assertion here is a property of the split derivation rather than of a
particular fixture: the same pool always yields the same task hashes, and a
split's identity depends only on the tasks it was given.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest
from pydantic import ValidationError

from whetstone.core.roles import EvalRole
from whetstone.eval import (
    EvalProcedureConfig,
    EvalProcedureDefinition,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
)
from whetstone.experiment.sampling import (
    HELD_OUT,
    INTERNAL_EVAL,
    OFFICIAL,
    SPLIT_ROLES,
    EvalConfigs,
    EvalSplit,
    HeldOutReferencedError,
    SplitOverlapError,
    SplitProcedureMismatchError,
    assert_split_disjointness,
    derive_eval_split,
    evaluation_role_for_split,
    validate_evaluation_role_for_split,
)
from whetstone.testing.toy.experiment import (
    ToyTask,
    build_toy_experiment,
)

NAMESPACE = "whetstone.split-test"
DATASET_REVISION = "split-test/v1"

#: Deterministic strata-balanced pool: 4 strata x 12 tasks. The split helper
#: below deals tasks round-robin over strata, so any prefix of the pool is
#: itself strata-balanced.
STRATA = ("alpha", "beta", "gamma", "delta")
PER_STRATUM = 12


def _pool() -> tuple[ToyTask, ...]:
    """A strata-interleaved pool: index i belongs to stratum i % len(STRATA)."""
    tasks: list[ToyTask] = []
    for offset in range(PER_STRATUM):
        for stratum in STRATA:
            tasks.append(
                ToyTask(
                    task_id=f"{stratum}-{offset:02d}",
                    prompt_inputs={"prompt": f"{stratum} #{offset}"},
                    gold=stratum.upper(),
                )
            )
    return tuple(tasks)


def _task_hash(task: ToyTask) -> str:
    """A content hash over the task's identity, as the envs adapters do."""
    payload = json.dumps(
        {
            "task_id": task.task_id,
            "prompt_inputs": dict(task.prompt_inputs),
            "gold": task.gold,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _procedure_and_aggregation() -> tuple[EvalProcedureConfig, object]:
    experiment = build_toy_experiment()
    internal = experiment.eval_configs.internal
    return internal.procedure_config, internal.aggregation_config


def _procedure_hash() -> str:
    """The shared procedure identity every split below is derived with."""
    procedure, _ = _procedure_and_aggregation()
    return procedure.config_hash


def _split(
    tasks: tuple[ToyTask, ...], *, split_role: str, num_seeds: int = 3
) -> EvalSplit:
    procedure, aggregation = _procedure_and_aggregation()
    return derive_eval_split(
        namespace=NAMESPACE,
        dataset_revision=DATASET_REVISION,
        split_role=split_role,
        tasks=tasks,
        task_hash_of=_task_hash,
        procedure=procedure,
        aggregation=aggregation,
        num_seeds=num_seeds,
    )


def _configs(
    *,
    internal_size: int,
    official_size: int,
    held_out_size: int | None,
    num_seeds: int = 3,
) -> EvalConfigs:
    """Deal the pool into consecutive, disjoint internal/official/held-out."""
    pool = _pool()
    cut_a = internal_size
    cut_b = cut_a + official_size
    held_out = None
    if held_out_size is not None:
        held_out = _split(
            pool[cut_b : cut_b + held_out_size],
            split_role=HELD_OUT,
            num_seeds=num_seeds,
        )
    return EvalConfigs(
        env_name=NAMESPACE,
        procedure_config_hash=_procedure_hash(),
        internal=_split(
            pool[:cut_a], split_role=INTERNAL_EVAL, num_seeds=num_seeds
        ),
        official=_split(
            pool[cut_a:cut_b], split_role=OFFICIAL, num_seeds=num_seeds
        ),
        held_out=held_out,
    )


# --- role <-> split-role mapping -------------------------------------------


@pytest.mark.parametrize(
    ("split_role", "eval_role"),
    [
        (INTERNAL_EVAL, EvalRole.INTERNAL),
        (OFFICIAL, EvalRole.OFFICIAL),
        (HELD_OUT, EvalRole.HELD_OUT),
    ],
)
def test_every_split_role_owns_exactly_one_evaluation_role(
    split_role: str, eval_role: EvalRole
) -> None:
    assert evaluation_role_for_split(split_role) is eval_role
    validate_evaluation_role_for_split(
        split_role=split_role, evaluation_role=eval_role
    )


def test_split_roles_cover_every_evaluation_role_exactly_once() -> None:
    # The two enumerations must stay in lockstep: a new EvalRole with no
    # split role (or the reverse) would leave a split unaddressable.
    assert len(SPLIT_ROLES) == len(tuple(EvalRole))
    assert {evaluation_role_for_split(role) for role in SPLIT_ROLES} == set(
        EvalRole
    )


def test_an_unknown_split_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown evaluation split role"):
        evaluation_role_for_split("shadow")


def test_a_mismatched_evaluation_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match split role"):
        validate_evaluation_role_for_split(
            split_role=HELD_OUT, evaluation_role=EvalRole.OFFICIAL
        )


# --- three-split experiments ------------------------------------------------


def test_three_disjoint_splits_pass_the_leakage_check() -> None:
    configs = _configs(internal_size=8, official_size=12, held_out_size=20)

    covered = assert_split_disjointness(configs)

    assert sorted(configs.splits()) == sorted(SPLIT_ROLES)
    assert len(covered) == 8 + 12 + 20
    assert len(configs.held_out_task_hashes) == 20


def test_each_split_carries_its_own_role_and_eval_config() -> None:
    configs = _configs(internal_size=8, official_size=12, held_out_size=20)

    for split_role, split in configs.splits().items():
        assert split.split_role == split_role
        assert configs.eval_config_for(split_role) == split.eval_config
        assert configs.split_for(split_role) is split

    # Held-out is derived by its own derive_eval_split call, so its task set
    # and seed plan are distinct identities from the official split's.
    assert (
        configs.held_out.task_set.identity_hash()
        != configs.official.task_set.identity_hash()
    )
    assert (
        configs.held_out.seed_plan.identity_hash()
        != configs.official.seed_plan.identity_hash()
    )


def test_held_out_task_hashes_are_the_held_out_split_identity() -> None:
    configs = _configs(internal_size=8, official_size=12, held_out_size=20)

    assert (
        configs.held_out_task_hashes
        == configs.held_out.task_set.task_hashes
    )


# --- two-split experiments stay valid ---------------------------------------


def test_two_split_experiments_remain_valid() -> None:
    configs = _configs(internal_size=8, official_size=12, held_out_size=None)

    assert configs.held_out is None
    assert configs.held_out_task_hashes == ()
    assert sorted(configs.splits()) == sorted([INTERNAL_EVAL, OFFICIAL])
    assert len(assert_split_disjointness(configs)) == 20

    with pytest.raises(KeyError, match="no eval split for split role"):
        configs.eval_config_for(HELD_OUT)


def test_the_default_toy_experiment_has_no_held_out_split() -> None:
    # Existing toy and envs experiments define two roles and must keep
    # working unchanged.
    experiment = build_toy_experiment()

    assert experiment.eval_configs.held_out is None
    assert experiment.eval_configs.held_out_task_hashes == ()
    assert_split_disjointness(experiment.eval_configs)


def test_a_toy_experiment_can_define_a_held_out_split() -> None:
    experiment = build_toy_experiment(
        internal_tasks=(ToyTask(task_id="i", prompt_inputs={"prompt": "i"}),),
        official_tasks=(ToyTask(task_id="o", prompt_inputs={"prompt": "o"}),),
        held_out_tasks=(ToyTask(task_id="h", prompt_inputs={"prompt": "h"}),),
    )
    configs = experiment.eval_configs

    assert configs.held_out is not None
    assert configs.held_out.split_role == HELD_OUT
    assert len(assert_split_disjointness(configs)) == 3


# --- leakage detection ------------------------------------------------------


def test_a_held_out_task_reaching_the_internal_split_is_a_leak() -> None:
    pool = _pool()

    with pytest.raises(HeldOutReferencedError, match="share 1 held-out"):
        EvalConfigs(
            env_name=NAMESPACE,
            procedure_config_hash=_procedure_hash(),
            internal=_split(pool[:8], split_role=INTERNAL_EVAL),
            official=_split(pool[8:20], split_role=OFFICIAL),
            # Overlaps the internal split by one task.
            held_out=_split(pool[7:20], split_role=HELD_OUT),
        )


def test_a_held_out_task_reaching_the_official_split_is_a_leak() -> None:
    pool = _pool()

    with pytest.raises(HeldOutReferencedError, match="held_out"):
        EvalConfigs(
            env_name=NAMESPACE,
            procedure_config_hash=_procedure_hash(),
            internal=_split(pool[:8], split_role=INTERNAL_EVAL),
            official=_split(pool[8:20], split_role=OFFICIAL),
            held_out=_split(pool[19:30], split_role=HELD_OUT),
        )


def test_internal_and_official_overlap_is_a_plain_split_overlap() -> None:
    pool = _pool()

    with pytest.raises(SplitOverlapError, match="share 2 task identities"):
        EvalConfigs(
            env_name=NAMESPACE,
            procedure_config_hash=_procedure_hash(),
            internal=_split(pool[:8], split_role=INTERNAL_EVAL),
            official=_split(pool[6:20], split_role=OFFICIAL),
        )


def test_a_leaking_experiment_cannot_be_built_at_all() -> None:
    # The leakage check runs in EvalConfigs.__post_init__, so an experiment
    # builder that hands the same tasks to two splits fails at construction
    # rather than producing a leaking Experiment that a caller must
    # remember to audit.
    internal = (
        ToyTask(task_id="i", prompt_inputs={"prompt": "i"}),
        ToyTask(task_id="j", prompt_inputs={"prompt": "j"}),
    )

    with pytest.raises(HeldOutReferencedError, match="share 2 held-out"):
        build_toy_experiment(
            internal_tasks=internal,
            official_tasks=(
                ToyTask(task_id="o", prompt_inputs={"prompt": "o"}),
            ),
            held_out_tasks=internal,
        )


def test_a_split_cannot_be_derived_with_a_repeated_task() -> None:
    # Within-split uniqueness is a TaskSet validation, so a split can never
    # reach the leakage check carrying a duplicate.
    pool = _pool()

    with pytest.raises(ValidationError, match="task_hashes must be unique"):
        _split((pool[0], pool[1], pool[0]), split_role=INTERNAL_EVAL)


def _other_procedure() -> EvalProcedureConfig:
    """A second, genuinely different evaluation procedure config."""
    procedure, _ = _procedure_and_aggregation()
    definition = EvalProcedureDefinition(
        definition_id=f"{NAMESPACE}.other_evaluation_procedure",
        version="1",
    )
    preprocessing = PreprocessingDefinition(
        definition_id=f"{NAMESPACE}.preprocessing",
        version="1",
        steps=(),
    ).materialize()
    metric_extraction = MetricExtractionDefinition(
        definition_id=f"{NAMESPACE}.metric_extraction",
        version="1",
        questions=(MetricQuestionBinding(metric="score", on="submission"),),
    ).materialize(resolved_operators=(("score", "1"),))
    other = definition.materialize(
        preprocessing=preprocessing,
        metric_extraction=metric_extraction,
        assignment={"zero_denominator": "not_applicable"},
    )
    assert other.config_hash != procedure.config_hash
    return other


def _split_with(
    tasks: tuple[ToyTask, ...],
    *,
    split_role: str,
    procedure: EvalProcedureConfig,
) -> EvalSplit:
    _, aggregation = _procedure_and_aggregation()
    return derive_eval_split(
        namespace=NAMESPACE,
        dataset_revision=DATASET_REVISION,
        split_role=split_role,
        tasks=tasks,
        task_hash_of=_task_hash,
        procedure=procedure,
        aggregation=aggregation,
        num_seeds=3,
    )


def test_a_held_out_split_on_a_foreign_procedure_is_rejected() -> None:
    # The runtime executes the experiment's shared rollout graph but persists
    # the held-out split's own eval_config_ref, so a held-out split derived
    # with a different procedure would publish a procedure identity that was
    # never run. Construction refuses it.
    pool = _pool()

    with pytest.raises(SplitProcedureMismatchError, match="'held_out'"):
        EvalConfigs(
            env_name=NAMESPACE,
            procedure_config_hash=_procedure_hash(),
            internal=_split(pool[:8], split_role=INTERNAL_EVAL),
            official=_split(pool[8:20], split_role=OFFICIAL),
            held_out=_split_with(
                pool[20:30],
                split_role=HELD_OUT,
                procedure=_other_procedure(),
            ),
        )


def test_an_internal_split_on_a_foreign_procedure_is_rejected() -> None:
    # The same rule for the split the optimizer searches against.
    pool = _pool()

    with pytest.raises(SplitProcedureMismatchError, match="'internal_eval'"):
        EvalConfigs(
            env_name=NAMESPACE,
            procedure_config_hash=_procedure_hash(),
            internal=_split_with(
                pool[:8],
                split_role=INTERNAL_EVAL,
                procedure=_other_procedure(),
            ),
            official=_split(pool[8:20], split_role=OFFICIAL),
        )


def test_a_split_whose_persisted_eval_config_names_a_foreign_procedure(
) -> None:
    # ``EvalSplit`` is a plain dataclass, so its ``procedure_config`` and the
    # procedure recorded in the Eval Config it persists are not validated
    # against each other. A split carrying the shared procedure in the
    # redundant field while persisting an Eval Config built for another
    # procedure would run the experiment graph and then record the foreign
    # procedure identity, so construction checks the persisted config too.
    pool = _pool()
    foreign = _split_with(
        pool[20:30], split_role=HELD_OUT, procedure=_other_procedure()
    )
    procedure, _ = _procedure_and_aggregation()
    tampered = dataclasses.replace(foreign, procedure_config=procedure)
    assert tampered.procedure_config.config_hash == _procedure_hash()
    assert (
        tampered.eval_config.evaluation_procedure_config_hash
        != _procedure_hash()
    )

    with pytest.raises(
        SplitProcedureMismatchError, match="persists an Eval Config recording"
    ):
        EvalConfigs(
            env_name=NAMESPACE,
            procedure_config_hash=_procedure_hash(),
            internal=_split(pool[:8], split_role=INTERNAL_EVAL),
            official=_split(pool[8:20], split_role=OFFICIAL),
            held_out=tampered,
        )


def test_a_split_filed_under_the_wrong_role_is_rejected() -> None:
    pool = _pool()

    with pytest.raises(ValueError, match="carries split role"):
        EvalConfigs(
            env_name=NAMESPACE,
            procedure_config_hash=_procedure_hash(),
            internal=_split(pool[:8], split_role=INTERNAL_EVAL),
            official=_split(pool[8:20], split_role=OFFICIAL),
            # An official split filed in the held-out slot.
            held_out=_split(pool[20:40], split_role=OFFICIAL),
        )


# --- split stability under a growing held-out split -------------------------


def test_growing_held_out_leaves_internal_and_official_unchanged() -> None:
    # The Stage-0 gate defers the held-out size decision, which is only safe
    # if enlarging held-out strands no already-spent internal/official
    # evaluation. Split identity depends solely on the tasks a split is
    # given, so a larger held-out slice is a superset and the other two
    # splits are byte-identical.
    small = _configs(internal_size=8, official_size=12, held_out_size=20)
    large = _configs(internal_size=8, official_size=12, held_out_size=28)

    assert small.internal == large.internal
    assert small.official == large.official
    assert (
        small.internal.eval_config.config_hash
        == large.internal.eval_config.config_hash
    )
    assert (
        small.official.eval_config.config_hash
        == large.official.eval_config.config_hash
    )

    assert set(small.held_out_task_hashes) < set(large.held_out_task_hashes)
    assert (
        large.held_out_task_hashes[: len(small.held_out_task_hashes)]
        == small.held_out_task_hashes
    )

    assert_split_disjointness(small)
    assert_split_disjointness(large)


def test_growing_held_out_changes_only_the_held_out_identity() -> None:
    small = _configs(internal_size=8, official_size=12, held_out_size=20)
    large = _configs(internal_size=8, official_size=12, held_out_size=28)

    assert (
        small.held_out.eval_config.config_hash
        != large.held_out.eval_config.config_hash
    )
    assert small.held_out.seed_plan.num_seeds == (
        large.held_out.seed_plan.num_seeds
    )
