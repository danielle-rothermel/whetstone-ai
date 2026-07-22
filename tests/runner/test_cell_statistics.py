"""Cell-level statistical-confidence behaviour (fake transports, no calls).

Covers the parts of the "Statistical confidence" upgrade that live in
``run_cell``: the sharpened status boundaries (positive delta whose paired CI
spans 0 -> ``inconclusive``), the escalation that pools additional repeats and
can flip ``inconclusive`` -> ``improved``, the pooling math, the Eval-row
headroom gate (both directions), and the schema fields landing on the ledger.
Every transport is a scripted fake; no test makes a live paid LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from dr_providers import (
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    RawHttpRequest,
)

from tests.envs.support import transport_policy
from whetstone.envs.factory import build_env_experiment
from whetstone.envs.registry import EnvSpec, env_spec
from whetstone.envs.rollout_definition import (
    ceiling_candidate,
    initial_candidate,
    render_prompt,
)
from whetstone.optimization.mutation import MUTATION_FIELD
from whetstone.optimization.schema import Candidate
from whetstone.runner.budget import BudgetGuard
from whetstone.runner.cell import CellConfig, _pool_per_task, run_cell
from whetstone.runner.execution_mode import ExecutionMode
from whetstone.runner.ledger import Ledger

from .support import (
    PROPOSER_MODEL,
    TASK_MODEL,
    ScriptedProposer,
    credits_fetcher,
    proposer_config,
    runner_execution_policy,
)

MISS = "definitely-not-a-label"
WIN = "WIN_TEMPLATE {input}"
#: A split with SIX official tasks so a paired task-bootstrap has real width.
SPLIT = (2, 6, 2)


def _pool_n(env_name: str) -> int:
    env = env_spec(env_name)
    n = 1
    while n <= 60:
        try:
            build_env_experiment(
                env.name, model=TASK_MODEL, pool_n_per_stratum=n,
                split_sizes=SPLIT,
            )
        except Exception:
            n += 1
            continue
        return n
    msg = f"could not size a pool for {env_name}"
    raise RuntimeError(msg)  # pragma: no cover


def _experiment(env_name: str):
    return build_env_experiment(
        env_name, model=TASK_MODEL, pool_n_per_stratum=_pool_n(env_name),
        split_sizes=SPLIT,
    )


def _prompt_of(request: ProviderCallRequest) -> str:
    messages = request.transcript.messages
    return messages[-1].content if messages else ""


def _response(text: str) -> ProviderTransportResponse:
    return ProviderTransportResponse(
        text=text,
        raw_body={"choices": [{"message": {"content": text}}]},
        response_id="resp-1",
        model="test-model",
        finish_reason="stop",
    )


@dataclass
class PhasedTransport:
    """A stateful fake: the winner resolves MORE official tasks on escalation.

    ``reply`` maps a rendered prompt to gold/miss, but for the WINNER's
    official prompts it counts how many such calls it has served: the first
    pass (before escalation) only answers the FIRST official task's gold; once
    the served count crosses ``first_pass_calls`` (the initial best-eval pass),
    every official task gets gold. So the initial paired delta is a small
    positive whose CI spans 0 (``inconclusive``), and the pooled
    post-escalation delta is a clean positive (``improved``). Naive always
    misses. No network.
    """

    winner_gold_first: dict[str, str]
    winner_gold_rest: dict[str, str]
    other_gold: dict[str, str]
    first_pass_calls: int
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)
    _winner_calls: int = 0
    served: list[ProviderCallRequest] = field(default_factory=list)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        self.served.append(request)
        prompt = _prompt_of(request)
        if prompt in self.winner_gold_first or prompt in self.winner_gold_rest:
            self._winner_calls += 1
            escalated = self._winner_calls > self.first_pass_calls
            if prompt in self.winner_gold_first:
                text = self.winner_gold_first[prompt]
            elif escalated:
                text = self.winner_gold_rest[prompt]
            else:
                text = MISS
        else:
            text = self.other_gold.get(prompt, MISS)
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"Authorization": "Bearer k", "content-type": "json"},
            body={"model": "test-model"},
        )
        return ProviderInvocationEvidence.build(
            request=request, policy=self.policy,
            raw_request=raw_request, outcome=_response(text),
        )


def _winner_candidate(env):
    naive = initial_candidate(env)
    return Candidate(
        candidate_id="winner",
        base_ref=naive.base_ref,
        payload={MUTATION_FIELD: WIN},
    )


def _official(experiment):
    return tuple(experiment.eval_configs.official.instances)


def _internal(experiment):
    return tuple(experiment.eval_configs.internal.instances)


def _winner_internal_gold(experiment) -> dict[str, str]:
    """Winner golds every INTERNAL instance so the optimizer selects it.

    Without this the winner ties the naive baseline on the internal split and
    the optimizer keeps naive -> the best-eval would then re-drive naive.
    """
    env = env_spec(experiment.env_name)
    winner = _winner_candidate(env)
    return {
        render_prompt(env, winner, inst): inst.gold
        for inst in _internal(experiment)
    }


def _config(
    env_name, *, optimizer, rollout, proposer, official_repeats=5,
    canonical=True, task_model=TASK_MODEL,
):
    return CellConfig(
        optimizer=optimizer,
        env=env_name,
        lane="openrouter",
        attempt=0,
        task_model=task_model,
        proposer_model=PROPOSER_MODEL,
        canonical=canonical,
        proposer_config=proposer_config(),
        proposer_transport=proposer,
        rollout_transport=rollout,
        execution_policy=runner_execution_policy(),
        repeats=3,
        official_repeats=official_repeats,
        pool_n_per_stratum=_pool_n(env_name),
        split_sizes=SPLIT,
        execution_mode=ExecutionMode.IN_PROCESS,
    )


def _phased_transport(experiment, *, official_repeats):
    """Winner golds task 0 always; the rest only after the escalation pass."""
    env = env_spec(experiment.env_name)
    winner = _winner_candidate(env)
    official = _official(experiment)
    winner_first: dict[str, str] = {}
    winner_rest: dict[str, str] = {}
    for i, inst in enumerate(official):
        prompt = render_prompt(env, winner, inst)
        if i == 0:
            winner_first[prompt] = inst.gold
        else:
            winner_rest[prompt] = inst.gold
    first_pass_calls = len(official) * official_repeats
    return PhasedTransport(
        winner_gold_first=winner_first,
        winner_gold_rest=winner_rest,
        # The winner golds the internal split so the optimizer selects it (its
        # official prompts are phase-gated separately above).
        other_gold=_winner_internal_gold(experiment),
        first_pass_calls=first_pass_calls,
    )


def test_positive_delta_with_ci_spanning_zero_is_inconclusive(
    tmp_path: Path,
) -> None:
    env_name = "c11"
    exp = _experiment(env_name)
    rollout = _phased_transport(exp, official_repeats=5)
    # A non-canonical cell clears the start reserve guard even below reserve;
    # escalation is then skipped because remaining ($10) < reserve ($18.60).
    cfg = _config(
        env_name, optimizer="copro", rollout=rollout,
        proposer=ScriptedProposer((WIN,)), canonical=False,
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg, ledger=ledger, budget=BudgetGuard(),
        credits_fetcher=credits_fetcher([(700.0, 690.0), (700.0, 690.0)]),
    )
    r = outcome.record
    # Delta is positive (winner golds one official task) but the paired CI over
    # 6 tasks spans 0 -> inconclusive. Escalation is skipped (below reserve).
    assert r.delta is not None and r.delta > 0
    assert r.delta_ci95 is not None and r.delta_ci95[0] <= 0.0
    assert r.status == "inconclusive"
    assert r.escalated is False
    assert "escalation skipped" in r.escalation_note


def test_escalation_flips_inconclusive_to_improved(tmp_path: Path) -> None:
    env_name = "c11"
    exp = _experiment(env_name)
    rollout = _phased_transport(exp, official_repeats=5)
    cfg = _config(
        env_name, optimizer="copro", rollout=rollout,
        proposer=ScriptedProposer((WIN,)),
    )
    ledger = Ledger(root=tmp_path)
    # Plenty of remaining credits -> escalation permitted. The extra repeats
    # resolve every official task (phase flip), so the pooled paired CI
    # excludes zero and the status flips inconclusive -> improved.
    outcome = run_cell(
        cfg, ledger=ledger, budget=BudgetGuard(),
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    assert r.escalated is True
    assert r.official_repeats_used == 10  # doubled from 5
    assert r.status == "improved"
    assert r.delta_ci95 is not None and r.delta_ci95[0] > 0.0
    # Pooling kept every observation: naive/best pooled over both passes.
    assert r.pooled_observation_counts["naive"] == 6 * 10
    assert r.pooled_observation_counts["best"] == 6 * 10


def test_eval_row_headroom_gate_present_when_ceiling_beats_naive(
    tmp_path: Path,
) -> None:
    env_name = "c11"
    exp = _experiment(env_name)
    env = env_spec(env_name)
    ceiling = ceiling_candidate(env)
    official = _official(exp)
    # Ceiling golds EVERY official task; naive misses everything -> a real,
    # demonstrable headroom gap (paired ceiling-naive CI excludes 0).
    ceiling_gold = {
        render_prompt(env, ceiling, inst): inst.gold for inst in official
    }
    rollout = PhasedTransport(
        winner_gold_first={}, winner_gold_rest={},
        other_gold=ceiling_gold, first_pass_calls=0,
    )
    cfg = _config(
        env_name, optimizer="eval", rollout=rollout,
        proposer=ScriptedProposer(()),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(cfg, ledger=ledger)
    r = outcome.record
    assert r.headroom_delta is not None and r.headroom_delta > 0
    assert r.headroom_ci95 is not None and r.headroom_ci95[0] > 0.0
    assert r.no_demonstrable_headroom is False
    # The Eval row established the per-env cache (naive + ceiling vectors).
    cache = ledger.env_cache_for(env_name)
    assert cache is not None
    assert len(cache.naive_per_task) == len(official)
    assert len(cache.ceiling_per_task) == len(official)


def test_env_cache_keyed_by_task_model_deepseek_misses_nano(
    tmp_path: Path,
) -> None:
    # FIX 7: EnvOfficialCache keys by (env, task-model). An Eval row under the
    # NANO task model caches nano vectors; a later NON-eval cell under the
    # DEEPSEEK task model must NOT pair against those cached nano vectors -- it
    # gets a cache MISS and drives its own naive/ceiling arms. The task model
    # is recorded on both the cell line (models.task) and the cache line.
    env_name = "c11"
    exp = _experiment(env_name)
    env = env_spec(env_name)
    ceiling = ceiling_candidate(env)
    official = _official(exp)
    ceiling_gold = {
        render_prompt(env, ceiling, inst): inst.gold for inst in official
    }
    ledger = Ledger(root=tmp_path)

    # 1. Eval row under NANO: establishes the (c11, nano) cache.
    nano_rollout = PhasedTransport(
        winner_gold_first={}, winner_gold_rest={},
        other_gold=ceiling_gold, first_pass_calls=0,
    )
    nano_eval = _config(
        env_name, optimizer="eval", rollout=nano_rollout,
        proposer=ScriptedProposer(()), task_model="openai/gpt-5-nano",
    )
    run_cell(nano_eval, ledger=ledger)
    assert ledger.env_cache_for(env_name, task_model="openai/gpt-5-nano")
    # A deepseek-keyed lookup MISSES the nano cache.
    assert (
        ledger.env_cache_for(
            env_name, task_model="deepseek/deepseek-v4-flash"
        )
        is None
    )

    # 2. A copro cell under DEEPSEEK: the cache is nano-only, so the deepseek
    # cell must NOT reuse nano vectors -- it drives its own naive baseline arm.
    # We prove the drive happened by counting naive-candidate official calls on
    # a fresh transport (a cache HIT would drive ZERO naive calls).
    counting = _NaiveCallCountingTransport(env=env, other_gold=ceiling_gold)
    deepseek_cell = _config(
        env_name, optimizer="copro", rollout=counting,
        proposer=ScriptedProposer((WIN,)),
        task_model="deepseek/deepseek-v4-flash",
    )
    outcome = run_cell(deepseek_cell, ledger=ledger)
    assert outcome.record.models.task == "deepseek/deepseek-v4-flash"
    # The deepseek cell drove the naive baseline itself (cache miss): >0 naive
    # official calls. A nano-cache reuse would have driven zero.
    assert counting.naive_official_calls > 0


@dataclass
class _NaiveCallCountingTransport:
    """Counts official-split calls for the NAIVE candidate's rendered prompts.

    A cache HIT would skip driving the naive baseline entirely (zero naive
    official calls); a MISS drives it (>0). Replies gold for the ceiling probe
    so the ceiling arm computes; naive always misses.
    """

    env: EnvSpec
    other_gold: dict[str, str]
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)
    naive_official_calls: int = 0
    _naive_prompts: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        naive = initial_candidate(self.env)
        exp = _experiment(self.env.name)
        self._naive_prompts = frozenset(
            render_prompt(self.env, naive, inst)
            for inst in _official(exp)
        )

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        prompt = _prompt_of(request)
        if prompt in self._naive_prompts:
            self.naive_official_calls += 1
        text = self.other_gold.get(prompt, MISS)
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"content-type": "json"}, body={"model": "test-model"},
        )
        return ProviderInvocationEvidence.build(
            request=request, policy=self.policy, raw_request=raw_request,
            outcome=_response(text),
        )


def test_eval_row_no_demonstrable_headroom_when_ceiling_ties_naive(
    tmp_path: Path,
) -> None:
    env_name = "c11"
    # Nobody golds anything: naive == ceiling == 0 -> headroom CI includes 0.
    rollout = PhasedTransport(
        winner_gold_first={}, winner_gold_rest={},
        other_gold={}, first_pass_calls=0,
    )
    cfg = _config(
        env_name, optimizer="eval", rollout=rollout,
        proposer=ScriptedProposer(()),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(cfg, ledger=ledger)
    r = outcome.record
    assert r.headroom_delta == pytest.approx(0.0)
    assert r.no_demonstrable_headroom is True


def test_pool_per_task_is_count_weighted_and_discards_nothing() -> None:
    # Pass 1: task means over 5 repeats; pass 2: means over 3 repeats. The
    # pooled mean is the count-weighted mean; the pooled count is the sum.
    scores_a = (1.0, 0.0, 0.4)
    counts_a = (5, 5, 5)
    scores_b = (0.0, 1.0, 1.0)
    counts_b = (3, 3, 3)
    pooled_scores, pooled_counts = _pool_per_task(
        scores_a, counts_a, scores_b, counts_b
    )
    assert pooled_counts == (8, 8, 8)
    # task0: (1.0*5 + 0.0*3)/8 = 0.625
    assert pooled_scores[0] == pytest.approx(0.625)
    # task1: (0.0*5 + 1.0*3)/8 = 0.375
    assert pooled_scores[1] == pytest.approx(0.375)
    # task2: (0.4*5 + 1.0*3)/8 = (2.0 + 3.0)/8 = 0.625
    assert pooled_scores[2] == pytest.approx(0.625)


def test_pool_per_task_zero_total_is_zero() -> None:
    pooled_scores, pooled_counts = _pool_per_task(
        (0.0,), (0,), (0.0,), (0,)
    )
    assert pooled_scores == (0.0,)
    assert pooled_counts == (0,)


def test_pool_per_task_rejects_misaligned() -> None:
    with pytest.raises(ValueError, match="aligned"):
        _pool_per_task((0.0, 1.0), (5, 5), (1.0,), (5,))


def test_non_eval_cell_reuses_cached_naive_vectors(tmp_path: Path) -> None:
    env_name = "c11"
    exp = _experiment(env_name)
    env = env_spec(env_name)
    official = _official(exp)
    ceiling = ceiling_candidate(env)
    ceiling_gold = {
        render_prompt(env, ceiling, inst): inst.gold for inst in official
    }
    ledger = Ledger(root=tmp_path)
    # 1. Eval row establishes the cache.
    eval_rollout = PhasedTransport(
        winner_gold_first={}, winner_gold_rest={},
        other_gold=ceiling_gold, first_pass_calls=0,
    )
    run_cell(
        _config(env_name, optimizer="eval", rollout=eval_rollout,
                proposer=ScriptedProposer(())),
        ledger=ledger,
    )
    # 2. A later optimizer cell must NOT re-drive the naive baseline: its
    # rollout never sees a naive or ceiling official prompt (only winner's).
    naive = initial_candidate(env)
    naive_official = {render_prompt(env, naive, inst) for inst in official}
    ceiling_official = set(ceiling_gold)
    winner = _winner_candidate(env)
    winner_gold = {
        render_prompt(env, winner, inst): inst.gold for inst in official
    }
    opt_rollout = PhasedTransport(
        winner_gold_first=winner_gold, winner_gold_rest={},
        # Winner golds the internal split so the optimizer selects it.
        other_gold=_winner_internal_gold(exp), first_pass_calls=10_000,
    )
    run_cell(
        _config(env_name, optimizer="copro", rollout=opt_rollout,
                proposer=ScriptedProposer((WIN,))),
        ledger=ledger,
    )
    served_prompts = {_prompt_of(req) for req in opt_rollout.served}
    # The cached naive/ceiling official arms were NOT re-driven.
    assert served_prompts.isdisjoint(naive_official)
    assert served_prompts.isdisjoint(ceiling_official)
