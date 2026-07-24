"""The cross-env factory + the transport-injected internal-eval loop.

The load-bearing proof: a full internal-eval of the naive candidate on a
tiny pool, driven by a FAKE transport, produces a valid Rollout Aggregate +
Reward -- per env, with no live paid call.
"""

from __future__ import annotations

import pytest
from dr_code.eval import AggregationStatus

from tests.envs.support import (
    FakeTransport,
    ReplyFn,
    constant_reply,
    execution_policy,
)
from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.envs.internal_eval import run_internal_eval
from whetstone.envs.registry import ENV_NAMES, env_spec
from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.reward import Reward

_MODEL = "openai/gpt-5-nano"
_SPLIT = (2, 2, 2)


def _tiny_experiment(env_name: str) -> EnvExperiment:
    # n_per_stratum=1 gives >= 4 instances (all envs have >= 4 strata except
    # c18 which has 4); a (2,2,2) split needs >= 6, so grow the pool until it
    # is large enough. For a stratified-split env (c22, whose pool is blocked)
    # each stratum must independently hold its per-stratum quota, so grow until
    # the stratified split is satisfiable rather than only until the total
    # instance count clears sum(_SPLIT).
    env = env_spec(env_name)
    n = 1
    while not _split_fits(env, n):
        n += 1
    return build_env_experiment(
        env_name,
        model=_MODEL,
        pool_n_per_stratum=n,
        split_sizes=_SPLIT,
    )


def _split_fits(env, n: int) -> bool:
    """True once a pool at ``n_per_stratum=n`` can serve the ``_SPLIT`` totals.

    For a contiguous-split env the whole pool need only exceed ``sum(_SPLIT)``;
    for a stratified-split env each stratum must hold its per-stratum quota, so
    ``n`` must clear the largest single-stratum draw.
    """
    pool = env.generate_pool(n_per_stratum=n)
    if not env.stratified_split:
        return len(pool) >= sum(_SPLIT)
    n_strata = len(pool.strata)
    per_stratum_max = sum(
        -(-part // n_strata) for part in _SPLIT  # ceil division per split part
    )
    return n >= per_stratum_max


def _correct_reply(env_name: str, instances) -> ReplyFn:
    """A reply fn that returns the correct answer for the matching task.

    The env oracle grades the generation against each task's gold; the fake
    returns each task's own correct answer keyed off its rendered prompt so
    the internal eval scores a clean 1.0.
    """
    env = env_spec(env_name)
    from whetstone.envs.rollout_definition import (
        initial_candidate,
        render_prompt,
    )

    naive = initial_candidate(env)
    # Map rendered-naive-prompt -> the correct generation for that instance.
    correct_by_prompt: dict[str, str] = {}
    for inst in instances:
        prompt = render_prompt(env, naive, inst)
        correct_by_prompt[prompt] = _correct_generation(env, inst)

    def reply(prompt: str) -> str:
        return correct_by_prompt.get(prompt, "")

    return reply


def _correct_generation(env, instance) -> str:
    """The known-correct generation for an instance (per env)."""
    if env.name == "c22":
        # A response satisfying whatever stack the instance carries is
        # instance-specific; the c22 internal-eval test uses a hand-built
        # single-instance fixture instead (see the c22-specific test below).
        return instance.gold
    # For the re-derive envs the gold IS the correct answer.
    return instance.gold


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_build_env_experiment_returns_all_five_deliverables(
    env_name: str,
) -> None:
    exp = build_env_experiment(env_name, model=_MODEL)
    d = exp.as_dict()
    assert set(d) == {
        "rollout_definition",
        "initial_candidate",
        "ceiling_candidate",
        "eval_configs",
        "reward_policy",
    }
    # The Rollout Definition and both Eval Configs share one Procedure id.
    assert (
        exp.rollout_definition.procedure_config_hash
        == exp.eval_configs.procedure_config_hash
    )


@pytest.mark.parametrize("env_name", ["c11", "c19", "c18", "c23"])
def test_internal_eval_naive_candidate_clean_pass(env_name: str) -> None:
    exp = _tiny_experiment(env_name)
    internal_insts = exp.eval_configs.internal.instances
    transport = FakeTransport(reply=_correct_reply(env_name, internal_insts))

    result = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        instances=internal_insts,
        execution_policy=execution_policy(),
        transport=transport,
        repeats=2,
    )
    agg = result.aggregate
    assert agg.name == "env_exact_match"
    assert agg.aggregation_output.status is AggregationStatus.OK
    assert agg.aggregation_output.value == pytest.approx(1.0)
    # Complete matrix accounting: every planned row is present, none dropped.
    planned = agg.task_count * agg.repeat_count
    assert agg.rows_present == planned
    assert agg.rows_missing == agg.rows_failed == agg.rows_invalid == 0
    # A valid internal-role Reward maps the aggregate.
    assert isinstance(result.reward, Reward)
    assert result.reward.evidence_role is EvaluationRole.INTERNAL
    assert result.reward.value == pytest.approx(1.0)


def test_c22_internal_eval_produces_valid_aggregate_and_reward() -> None:
    # c22 correct responses are constraint-stack-specific (proven at score 1
    # against a hand-built fixture in test_oracle_operator); here the full
    # internal-eval loop is exercised end to end through the c22 gold-first
    # oracle, producing a VALID Rollout Aggregate + Reward. A response that
    # satisfies no stack scores 0 across the split.
    exp = _tiny_experiment("c22")
    internal_insts = exp.eval_configs.internal.instances
    transport = FakeTransport(reply=constant_reply("plain, comma-laden text"))
    result = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        instances=internal_insts,
        execution_policy=execution_policy(),
        transport=transport,
        repeats=2,
    )
    agg = result.aggregate
    assert agg.name == "env_exact_match"
    assert agg.aggregation_output.status is AggregationStatus.OK
    assert agg.aggregation_output.value == pytest.approx(0.0)
    planned = agg.task_count * agg.repeat_count
    assert agg.rows_present == planned
    assert result.reward.evidence_role is EvaluationRole.INTERNAL
    assert result.reward.value == pytest.approx(0.0)


def test_internal_eval_wrong_answers_score_zero() -> None:
    exp = _tiny_experiment("c18")
    internal_insts = exp.eval_configs.internal.instances
    # Always answer the opposite label so every task scores 0.
    transport = FakeTransport(reply=constant_reply("definitely-not-a-label"))
    result = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        instances=internal_insts,
        execution_policy=execution_policy(),
        transport=transport,
        repeats=2,
    )
    assert result.aggregate.aggregation_output.value == pytest.approx(0.0)
    assert result.reward.value == pytest.approx(0.0)


def test_internal_eval_is_deterministic() -> None:
    exp = _tiny_experiment("c18")
    internal_insts = exp.eval_configs.internal.instances
    reply = _correct_reply("c18", internal_insts)
    a = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        instances=internal_insts,
        execution_policy=execution_policy(),
        transport=FakeTransport(reply=reply),
        repeats=2,
    )
    b = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        instances=internal_insts,
        execution_policy=execution_policy(),
        transport=FakeTransport(reply=reply),
        repeats=2,
    )
    assert a.aggregate.aggregation_output.value == (
        b.aggregate.aggregation_output.value
    )
    assert a.reward.value == b.reward.value
    assert a.aggregate.graph_hash == b.aggregate.graph_hash


def test_blank_generation_is_a_failed_row_not_a_silent_zero() -> None:
    # A blank generation is not an accepted Generation (a provider semantic
    # failure); the internal-eval marks it a FAILED row. Under the default
    # PROPAGATE policy that makes the aggregate visibly incomplete (value
    # None) -- never a silent 0 -- so the FAIL Reward Policy refuses to
    # compute a Reward over incomplete internal evidence.
    exp = _tiny_experiment("c18")
    internal_insts = exp.eval_configs.internal.instances
    transport = FakeTransport(reply=constant_reply("   "))
    with pytest.raises(ValueError, match="missing"):
        run_internal_eval(
            exp,
            candidate=exp.initial_candidate,
            instances=internal_insts,
            execution_policy=execution_policy(),
            transport=transport,
            repeats=2,
        )


def test_failed_rows_under_skip_still_visible_in_provenance() -> None:
    # Under SKIP, all-failed rows leave the reduction empty -> a non-OK
    # status (never a fabricated zero); the failed rows remain counted in the
    # aggregate provenance.
    from whetstone.code_eval.aggregate import RowPolicy, RowValue, TaskRows
    from whetstone.envs.internal_eval import _env_exact_match_aggregate

    task_rows = (
        TaskRows(
            task_identity="t0",
            expected_repeats=2,
            rows=(RowValue(failed=True), RowValue(failed=True)),
        ),
    )
    agg = _env_exact_match_aggregate(
        graph_hash="g" * 64,
        eval_config_hash="e" * 64,
        evaluation_context_id="c" * 64,
        task_rows=task_rows,
        repeat_count=2,
        policy=RowPolicy.SKIP,
    )
    assert agg.rows_failed == 2
    assert agg.rows_present == 0
    assert agg.aggregation_output.status is not AggregationStatus.OK


def test_unknown_env_rejected() -> None:
    from whetstone.envs.registry import UnknownEnvError

    with pytest.raises(UnknownEnvError):
        build_env_experiment("c99", model=_MODEL)
