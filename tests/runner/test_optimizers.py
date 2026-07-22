"""Optimizer hyperparameter scaling + internal-split optimize-loop tests."""

from __future__ import annotations

import pytest

from whetstone.optimization.mutation import MUTATION_FIELD
from whetstone.runner.optimizers import (
    OPTIMIZERS,
    hyperparameters_for,
    run_optimize,
    scaled_hyperparameters,
    scaling_help,
)

from .support import (
    FakeTransport,
    ScriptedProposer,
    correct_reply,
    improvement_reply,
    no_improvement_reply,
    proposer_config,
    runner_execution_policy,
    tiny_experiment,
)

WIN = "WIN_TEMPLATE {input}"


def test_all_optimizers_have_brief_hyperparameters() -> None:
    for opt in OPTIMIZERS:
        hyper = hyperparameters_for(opt)
        assert "internal_task_count" in hyper


def test_copro_brief_pins_breadth_depth() -> None:
    hyper = hyperparameters_for("copro")
    assert hyper["breadth"] == 4
    assert hyper["depth"] == 2
    assert hyper["copro_variant"] == "whetstone_multi_seed/v1"


def test_scaling_clamps_internal_task_count() -> None:
    # COPRO's brief internal_task_count is 20; a pool of 3 clamps it to 3.
    scaled = scaled_hyperparameters("copro", internal_pool_size=3)
    assert scaled["internal_task_count_scaled"] == 3
    assert "clamped" in scaled["scaling_note"]


def test_scaling_leaves_small_counts_unchanged() -> None:
    # MIPROv2 minibatch is 8; a pool of 8 needs no clamp.
    scaled = scaled_hyperparameters("miprov2", internal_pool_size=8)
    assert scaled["internal_task_count_scaled"] == 8
    assert scaled["full_eval_task_count_scaled"] == 8  # 35 clamped to 8


def test_scaling_help_mentions_pool_sizes() -> None:
    help_text = scaling_help()
    assert "internal_pool_size" in help_text
    assert "clamp" in help_text.lower()


def test_run_optimize_finds_winning_proposal() -> None:
    exp = tiny_experiment("c11")
    result = run_optimize(
        exp,
        optimizer="copro",
        proposer_config=proposer_config(),
        proposer_transport=ScriptedProposer((WIN,)),
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        execution_policy=runner_execution_policy(),
        internal_instances=exp.eval_configs.internal.instances,
        repeats=3,
    )
    assert result.best_candidate.payload[MUTATION_FIELD] == WIN
    assert result.best_internal_score == pytest.approx(1.0)
    assert result.optimizer_steps > 0


def test_run_optimize_keeps_naive_when_no_proposal_wins() -> None:
    exp = tiny_experiment("c11")
    result = run_optimize(
        exp,
        optimizer="copro",
        proposer_config=proposer_config(),
        proposer_transport=ScriptedProposer(("loser {input}",)),
        rollout_transport=FakeTransport(reply=no_improvement_reply(exp)),
        execution_policy=runner_execution_policy(),
        internal_instances=exp.eval_configs.internal.instances,
        repeats=3,
    )
    # No proposal beats the (equally-scoring) naive candidate.
    assert (
        result.best_candidate.candidate_id
        == exp.initial_candidate.candidate_id
    )


def test_eval_identity_makes_no_proposals() -> None:
    exp = tiny_experiment("c11")
    result = run_optimize(
        exp,
        optimizer="eval",
        proposer_config=proposer_config(),
        proposer_transport=ScriptedProposer(()),
        rollout_transport=FakeTransport(reply=correct_reply(exp)),
        execution_policy=runner_execution_policy(),
        internal_instances=exp.eval_configs.internal.instances,
        repeats=3,
    )
    assert result.optimizer_steps == 0
    assert (
        result.best_candidate.candidate_id
        == exp.initial_candidate.candidate_id
    )


def test_proposer_route_identity_distinct_from_graph() -> None:
    exp = tiny_experiment("c11")
    proposer = ScriptedProposer((WIN,))
    run_optimize(
        exp,
        optimizer="copro",
        proposer_config=proposer_config(),
        proposer_transport=proposer,
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        execution_policy=runner_execution_policy(),
        internal_instances=exp.eval_configs.internal.instances,
        repeats=3,
    )
    # The proposer transport was called through the proposer route's identity
    # hash, distinct from the graph's provider-call-config hash.
    assert proposer.calls
    used_hash = proposer.calls[0][0]
    graph_hash = exp.rollout_definition.provider_call_config.identity_hash
    assert used_hash != graph_hash


def test_unknown_optimizer_rejected() -> None:
    with pytest.raises(ValueError, match="unknown optimizer"):
        hyperparameters_for("nope")
