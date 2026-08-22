"""GEPA reports the reflection spend a Step caused, never the spend it replayed.

GEPA's step engine re-runs upstream ``optimize`` from the seed on every
harness step and relies on the durable effect cache to replay the prefix its
predecessors already paid for. Every replayed reflection therefore comes back
through the broker looking exactly like a fresh one. Recording those as spend
would bill a crash-and-resume twice and would grow each step's reported
proposer cost by the whole prefix before it.

These drive the real ``HarnessGepaEffectBroker`` over a real store, so the
replay under test is the durable cache the production path uses.
"""

from __future__ import annotations

import pytest
from dr_store.sync import open_sqlite

from whetstone.optim.gepa.contracts import (
    GepaCandidateComponent,
    GepaProposalEffectRequest,
    GepaProposalEffectResult,
)
from whetstone.optim.gepa.harness_broker import HarnessGepaEffectBroker
from whetstone.optim.cost import ProposerCallUsage
from whetstone.optim.cost_aggregation import aggregate_run_cost

from tests.test_gepa_reflection_retry import (
    COMPONENT,
    GOOD,
    SEED,
    _adapter,
    _dataset,
    _scripted_attempt_ref,
    _services,
)

PROMPT_TOKENS = 120
COMPLETION_TOKENS = 30
CALL_USD = 0.05


class _CountingProposalAuthority:
    """A proposal authority that answers, and counts what it was asked for."""

    runtime_hash = "9" * 64

    def __init__(self) -> None:
        self.calls = 0

    def propose(
        self, request: GepaProposalEffectRequest
    ) -> GepaProposalEffectResult:
        self.calls += 1
        parsed = _services().parse_replacement(request.component_name, GOOD)
        return GepaProposalEffectResult(
            request_hash=request.identity_hash(),
            raw_response=GOOD,
            parsed_components=(
                GepaCandidateComponent(
                    name=request.component_name, text=parsed
                ),
            ),
            request_evidence={"scripted": True},
            response_evidence={"scripted": True},
            provider_attempt_refs=(_scripted_attempt_ref(),),
            usage={
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
            },
            cost=CALL_USD,
        )


class _UnusedEvaluationAuthority:
    runtime_hash = "8" * 64

    def evaluate(self, request):  # pragma: no cover - proposals only
        raise AssertionError("this broker only serves proposals")

    def collect_replayed(self, result):  # pragma: no cover - proposals only
        raise AssertionError("this broker only serves proposals")


class _StepResult:
    def __init__(self, usage: tuple[ProposerCallUsage, ...]) -> None:
        self.resolved_intents = ()
        self.search_evidence = ()
        self.tool_evidence = ()
        self.proposer_usage = usage


def _drive_one_reflection(broker) -> tuple[ProposerCallUsage, ...]:
    """Run one reflection through an adapter bound to the shared broker."""
    adapter, _scripted = _adapter([GOOD])
    adapter._broker = broker  # noqa: SLF001 - swap in the durable broker
    replacements = adapter.propose_new_texts(
        {COMPONENT: SEED}, _dataset(), [COMPONENT]
    )
    assert replacements == {COMPONENT: GOOD}
    return adapter.proposer_usage


def test_a_replayed_reflection_is_not_recorded_as_spend(tmp_path) -> None:
    """The crash-and-resume case: re-driving the prefix costs nothing.

    The first drive pays for the reflection. The second re-drives the same
    request, which the durable effect cache answers -- so the provider is
    never asked again, and the second Step must report no proposer spend.
    """
    authority = _CountingProposalAuthority()
    with open_sqlite(str(tmp_path / "gepa.sqlite")) as store:
        broker = HarnessGepaEffectBroker(
            store,
            evaluation_authority=_UnusedEvaluationAuthority(),
            proposal_authority=authority,
        )
        first = _drive_one_reflection(broker)
        # The re-invoke a crashed Step performs: same request, cached answer.
        replayed = _drive_one_reflection(broker)

    assert authority.calls == 1, "the replay must not reach the provider"
    assert len(first) == 1
    assert first[0].prompt_tokens == PROMPT_TOKENS
    assert first[0].usd == pytest.approx(CALL_USD)
    assert replayed == (), "a replayed reflection is spend this Step did not cause"


def test_a_reinvoked_step_reports_the_uninterrupted_run_total(
    tmp_path,
) -> None:
    """A run interrupted mid-step reports the proposer cost of one that was not.

    Whichever way the run reached its Step Results, one reflection was paid
    for once, so the run total is that one call. The interrupted run reaches
    it differently: the crashed attempt's usage is on the Step Result it
    persisted, and the re-invoked attempt adds nothing because the effect
    cache served it. De-duplication by ``call_id`` then makes the two
    identical even if both attempts' Step Results survive.
    """
    with open_sqlite(str(tmp_path / "uninterrupted.sqlite")) as store:
        broker = HarnessGepaEffectBroker(
            store,
            evaluation_authority=_UnusedEvaluationAuthority(),
            proposal_authority=_CountingProposalAuthority(),
        )
        uninterrupted = aggregate_run_cost(
            store=store,
            step_results=(_StepResult(_drive_one_reflection(broker)),),
        )

    authority = _CountingProposalAuthority()
    with open_sqlite(str(tmp_path / "crashed.sqlite")) as store:
        broker = HarnessGepaEffectBroker(
            store,
            evaluation_authority=_UnusedEvaluationAuthority(),
            proposal_authority=authority,
        )
        # The Step drove its reflection and then died. The harness re-invoked
        # it, and it re-drove the same reflection from the durable cache.
        crashed_attempt = _drive_one_reflection(broker)
        reinvoked_attempt = _drive_one_reflection(broker)
        interrupted = aggregate_run_cost(
            store=store,
            step_results=(
                _StepResult(crashed_attempt),
                _StepResult(reinvoked_attempt),
            ),
        )

    assert authority.calls == 1, "the re-invoke must not reach the provider"
    assert uninterrupted.proposer.calls == 1
    assert uninterrupted.proposer.usd == pytest.approx(CALL_USD)
    assert interrupted.proposer == uninterrupted.proposer


def test_a_replayed_call_is_deduplicated_even_if_both_attempts_report_it(
    tmp_path,
) -> None:
    """Belt and braces: identity de-duplication backs up the replay skip.

    The append site skips replayed calls, so this cannot arise through the
    broker today. It is asserted directly because the two guards protect the
    same invariant from different sides, and a future append site that forgets
    the replay flag must still not double-bill.
    """
    with open_sqlite(str(tmp_path / "dedup.sqlite")) as store:
        broker = HarnessGepaEffectBroker(
            store,
            evaluation_authority=_UnusedEvaluationAuthority(),
            proposal_authority=_CountingProposalAuthority(),
        )
        paid = _drive_one_reflection(broker)
        assert len(paid) == 1
        report = aggregate_run_cost(
            store=store,
            step_results=(_StepResult(paid), _StepResult(paid)),
        )

    assert report.proposer.calls == 1
    assert report.proposer.usd == pytest.approx(CALL_USD)


def test_per_step_proposer_usage_is_not_cumulative(tmp_path) -> None:
    """Step N reports its own reflection, not reflections 1..N.

    GEPA replays its whole prefix each step, so a cumulative record would
    make a K-step run report roughly K(K+1)/2 reflections instead of K.
    """
    steps: list[tuple[ProposerCallUsage, ...]] = []
    with open_sqlite(str(tmp_path / "multi.sqlite")) as store:
        broker = HarnessGepaEffectBroker(
            store,
            evaluation_authority=_UnusedEvaluationAuthority(),
            proposal_authority=_CountingProposalAuthority(),
        )
        for _step in range(3):
            # Each step re-drives the prefix, then its own fresh reflection.
            # Here every request is identical, so after the first step every
            # drive is pure replay.
            steps.append(_drive_one_reflection(broker))
        report = aggregate_run_cost(
            store=store,
            step_results=tuple(_StepResult(usage) for usage in steps),
        )

    assert [len(usage) for usage in steps] == [1, 0, 0]
    # One reflection was actually paid for across the whole run.
    assert report.proposer.calls == 1
    assert report.proposer.input_tokens == PROMPT_TOKENS
    assert report.proposer.usd == pytest.approx(CALL_USD)
