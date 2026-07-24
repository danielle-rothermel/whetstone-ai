"""Optimizer hyperparameter scaling + internal-split optimize-loop tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from dr_providers import (
    FailureClass,
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderTransportFailure,
    ProviderTransportPolicy,
    RawHttpRequest,
)

from tests.envs.support import (
    ReplyFn,
    _prompt_of,
    _response,
    transport_policy,
)
from whetstone.optimization.mutation import MUTATION_FIELD
from whetstone.optimization.proposer import (
    FakeProposerTransport,
    ProposalDraft,
    ProposalRequest,
    ProposerConfig,
)
from whetstone.runner.optimizers import (
    INVALID_TEMPLATE_PLACEHOLDERS,
    OPTIMIZERS,
    PROPOSER_DRAFT_FAILED,
    UNSCORABLE_CANDIDATE,
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

# The live c22 crash shape, reproduced on c11 (whose fixtures score a winner):
# an untrusted proposer draft carries a placeholder ({question}) that is NOT
# one of the env's prompt_inputs keys (c11's only key is {input}), exactly as
# c22's {question} was not among its {constraints_block} inputs. Rendering it
# would raise the probe surface's loud KeyError and kill the cell -- so it must
# be rejected at intake, before any eval spend.
BAD = "Question: {question}\n\nAnswer:"


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


def test_run_optimize_rejects_bad_placeholder_without_eval_spend() -> None:
    # The c22 crash shape: the proposer emits an unknown-placeholder draft
    # ({question}) AND a valid winning draft. The bad draft is REJECTED at
    # intake (no eval spend), recorded with the typed reason + offending field,
    # and the valid candidate is selected as best -- the run completes.
    exp = tiny_experiment("c11")
    transport = FakeTransport(reply=improvement_reply(exp, WIN))
    result = run_optimize(
        exp,
        optimizer="copro",
        proposer_config=proposer_config(),
        # breadth is 4; the first two drafts are scripted, the rest padded.
        proposer_transport=ScriptedProposer((BAD, WIN)),
        rollout_transport=transport,
        execution_policy=runner_execution_policy(),
        internal_instances=exp.eval_configs.internal.instances,
        repeats=3,
    )
    rejected = [s for s in result.steps if s.rejected]
    assert rejected, "the bad-placeholder draft must be recorded as rejected"
    bad = next(s for s in rejected if "question" in s.rejected_fields)
    assert bad.rejected_reason == INVALID_TEMPLATE_PLACEHOLDERS
    assert bad.rejected_fields == ("question",)
    assert bad.evaluation is None and bad.internal_score is None
    # The rejected candidate is never selected as best.
    assert result.best_candidate.payload[MUTATION_FIELD] != BAD
    assert result.best_candidate.payload[MUTATION_FIELD] == WIN
    assert result.best_internal_score == pytest.approx(1.0)
    assert result.rejected_candidate_count == len(rejected)
    # A rejected candidate spent NO eval call: internal_evals_count skips it.
    evaluated = [s for s in result.steps if not s.rejected]
    assert result.internal_evals_count == 1 + len(evaluated)


def test_run_optimize_all_drafts_bad_keeps_naive_best() -> None:
    # Every proposed template is unknown-placeholder junk: all are rejected and
    # the naive Initial Candidate remains best (never crashes, never selects a
    # rejected candidate).
    exp = tiny_experiment("c11")
    result = run_optimize(
        exp,
        optimizer="copro",
        proposer_config=proposer_config(),
        proposer_transport=ScriptedProposer((BAD, "{another_unknown}")),
        rollout_transport=FakeTransport(reply=no_improvement_reply(exp)),
        execution_policy=runner_execution_policy(),
        internal_instances=exp.eval_configs.internal.instances,
        repeats=3,
    )
    assert result.rejected_candidate_count >= 1
    assert (
        result.best_candidate.candidate_id
        == exp.initial_candidate.candidate_id
    )


# A round-1 winner and a round-2 poison template whose internal rollouts all
# fail transiently -- the (optimizer x postgres) live shape where a depth>=2
# optimizer's second-round candidate could not be scored.
WIN_R1 = "R1_WINNER {input}"
POISON_R2 = "R2_POISON {input}"


@dataclass
class _PoisonInternalTransport:
    """Fail (PERMANENT) every call whose prompt renders the poison template.

    Models a transient internal-rollout wipeout scoped to ONE candidate: the
    poison template's rendered prompt fails every repeat, so its internal
    aggregate is missing (None) under the FAIL Reward policy, while every other
    candidate (the round-1 winner, the naive baseline) scores cleanly.
    """

    poison_marker: str
    reply: ReplyFn
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)
    served: int = 0

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        self.served += 1
        prompt = _prompt_of(request)
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"content-type": "json"}, body={"model": "test-model"},
        )
        if self.poison_marker in prompt:
            failure = ProviderTransportFailure(
                failure_class=FailureClass.PERMANENT,
                code="http_status_429",
                message="scripted permanent failure (poison candidate)",
                retryable=False,
            )
            return ProviderInvocationEvidence.build(
                request=request, policy=self.policy,
                raw_request=raw_request, outcome=failure,
            )
        return ProviderInvocationEvidence.build(
            request=request, policy=self.policy, raw_request=raw_request,
            outcome=_response(self.reply(prompt)),
        )


def test_unscorable_round2_candidate_isolated_not_whole_run() -> None:
    # The (optimizer x postgres) live defect: a depth>=2 optimizer's round-2
    # candidate whose internal eval could not be scored (transient wipeout)
    # used to raise CandidateEvaluationFailure out of run_optimize, DISCARDING
    # every already-scored round-1 step so the cell finalized incomplete-arm
    # with optimizer_steps=0. Now the failure is ISOLATED to that candidate: it
    # is recorded as a typed rejected step, round-1 progress survives, and the
    # winning round-1 candidate stays best.
    exp = tiny_experiment("c11")
    # copro is depth=2: round 0 = seed_proposal, round 1 = history_proposal.
    proposer = FakeProposerTransport(
        script={
            ("seed_proposal", 0): (WIN_R1,),
            ("history_proposal", 1): (POISON_R2,),
        },
        default=("neutral {input}",),
    )
    transport = _PoisonInternalTransport(
        poison_marker="R2_POISON",
        reply=improvement_reply(exp, WIN_R1),
    )
    result = run_optimize(
        exp,
        optimizer="copro",
        proposer_config=proposer_config(),
        proposer_transport=proposer,
        rollout_transport=transport,
        execution_policy=runner_execution_policy(),
        internal_instances=exp.eval_configs.internal.instances,
        repeats=3,
    )
    # The run COMPLETED (no exception escaped) and kept the round-1 winner.
    assert result.best_candidate.payload[MUTATION_FIELD] == WIN_R1
    assert result.best_internal_score == pytest.approx(1.0)
    # The poison round-2 candidate is recorded as a typed unscorable rejection,
    # never selected as best, and counted -- not silently dropped.
    poisoned = [
        s for s in result.steps
        if s.rejected and s.rejected_reason == UNSCORABLE_CANDIDATE
    ]
    assert poisoned, "the unscorable round-2 candidate must be a rejected step"
    assert poisoned[0].template == POISON_R2
    assert poisoned[0].evaluation is None
    assert poisoned[0].rejected_detail  # a human-readable cause is retained
    # Round-1 scored steps SURVIVED (pre-fix they were discarded).
    scored = [s for s in result.steps if not s.rejected]
    assert any(
        s.template == WIN_R1 for s in scored
    ), "round-1 winner step must survive the round-2 failure"
    assert result.rejected_candidate_count == len(poisoned)


class _FailingDraftProposer:
    """A proposer transport that emits N failed draft slots then real ones.

    ``fail_count`` failed slots (typed failures, NO template) come first, then
    the remaining slots draft ``winning_template``. Models the codex-CLI /
    HTTP proposer failure path: a draft the route could not produce is a TYPED
    FAILURE, never a base-template echo.
    """

    def __init__(self, *, fail_count: int, winning_template: str) -> None:
        self._fail_count = fail_count
        self._winning = winning_template
        self.calls = 0

    def draft(
        self, config: ProposerConfig, request: ProposalRequest, count: int
    ) -> tuple[ProposalDraft, ...]:
        drafts: list[ProposalDraft] = []
        for index in range(count):
            self.calls += 1
            if index < self._fail_count:
                drafts.append(
                    ProposalDraft.failure(
                        detail=f"scripted draft failure #{index}",
                        request_evidence={"draft_index": index},
                    )
                )
            else:
                drafts.append(ProposalDraft(template=self._winning))
        return tuple(drafts)


def test_failed_drafts_recorded_as_typed_slots_no_phantom_candidate() -> None:
    # A rejected-model / failed proposer draft is recorded as a TYPED failed
    # slot (no template, never scored, never best) -- never a phantom candidate
    # echoing the base template.
    exp = tiny_experiment("c11")
    # copro breadth=4 depth=2; fail the first 2 per round, win with the rest.
    proposer = _FailingDraftProposer(fail_count=2, winning_template=WIN)
    result = run_optimize(
        exp,
        optimizer="copro",
        proposer_config=proposer_config(),
        proposer_transport=proposer,
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        execution_policy=runner_execution_policy(),
        internal_instances=exp.eval_configs.internal.instances,
        repeats=3,
    )
    failed = [
        s for s in result.steps if s.rejected_reason == PROPOSER_DRAFT_FAILED
    ]
    assert failed, "failed drafts must be recorded as typed slots"
    for s in failed:
        assert s.template == ""  # NO fabricated candidate
        assert s.evaluation is None
        assert s.internal_score is None
        assert s.rejected_detail and "failure" in s.rejected_detail
    assert result.failed_draft_count == len(failed)
    # Real drafts still scored + selected: the winner beats the baseline.
    assert result.scored_candidate_count > 0
    assert result.best_candidate.payload[MUTATION_FIELD] == WIN
    assert result.best_internal_score == pytest.approx(1.0)
    # A failed slot is NEVER the base template -> never confusable with a real
    # candidate (the c11 naive template would be the fallback echo pre-fix).
    naive_template = exp.initial_candidate.payload[MUTATION_FIELD]
    assert all(s.template != naive_template for s in failed)
    assert result.all_drafts_failed is False


def test_all_drafts_failed_flagged_distinct_from_no_improvement() -> None:
    # EVERY draft fails -> all_drafts_failed True, zero scored candidates, and
    # best stays the naive Initial Candidate. This is a proposer OUTAGE, NOT an
    # honest no-improvement (where real candidates WERE scored).
    exp = tiny_experiment("c11")
    proposer = _FailingDraftProposer(fail_count=999, winning_template=WIN)
    result = run_optimize(
        exp,
        optimizer="copro",
        proposer_config=proposer_config(),
        proposer_transport=proposer,
        rollout_transport=FakeTransport(reply=no_improvement_reply(exp)),
        execution_policy=runner_execution_policy(),
        internal_instances=exp.eval_configs.internal.instances,
        repeats=3,
    )
    assert result.all_drafts_failed is True
    assert result.scored_candidate_count == 0
    assert result.failed_draft_count == result.optimizer_steps
    assert (
        result.best_candidate.candidate_id
        == exp.initial_candidate.candidate_id
    )


def test_honest_no_improvement_is_not_all_drafts_failed() -> None:
    # Contrast: real candidates ARE drafted + scored but none beats the naive
    # baseline -> all_drafts_failed False (an honest no-improvement, distinct
    # from a proposer outage).
    exp = tiny_experiment("c11")
    result = run_optimize(
        exp,
        optimizer="copro",
        proposer_config=proposer_config(),
        proposer_transport=ScriptedProposer(("also-loses {input}",)),
        rollout_transport=FakeTransport(reply=no_improvement_reply(exp)),
        execution_policy=runner_execution_policy(),
        internal_instances=exp.eval_configs.internal.instances,
        repeats=3,
    )
    assert result.scored_candidate_count > 0
    assert result.all_drafts_failed is False
