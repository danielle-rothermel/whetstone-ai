"""Full cell runs against scripted responses: improvement + no-improvement."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
from whetstone.envs.registry import env_spec
from whetstone.optimization.codex_proposer import (
    CodexInvocation,
    CodexProposerTransport,
)
from whetstone.optimization.proposer import (
    FakeProposerTransport,
    ProposalDraft,
    ProposalRequest,
    ProposerConfig,
)
from whetstone.runner.budget import BudgetGuard, ReserveError
from whetstone.runner.cell import CellBaselineFailure, CellConfig, run_cell
from whetstone.runner.execution_mode import ExecutionMode
from whetstone.runner.ledger import Ledger

from .support import (
    PROPOSER_MODEL,
    SPLIT,
    TASK_MODEL,
    FailingTransport,
    FakeTransport,
    ScriptedProposer,
    _split_fits,
    ceiling_only_reply,
    correct_reply,
    credits_fetcher,
    improvement_reply,
    no_improvement_reply,
    proposer_config,
    runner_execution_policy,
    tiny_experiment,
)

WIN = "WIN_TEMPLATE {input}"

# The live c22 crash shape, reproduced on c11 (whose fixtures score a winner):
# an untrusted proposer draft with an unknown placeholder ({question}, not one
# of the env's prompt_inputs keys) beside a valid winning draft ({input}).
BAD = "Question: {question}\n\nAnswer:"


def _pool_n(env: str) -> int:
    n = 1
    while not _split_fits(env_spec(env), n):
        n += 1
    return n


def _config(
    env: str,
    *,
    optimizer: str,
    rollout_transport,
    proposer_transport,
    attempt: int = 0,
    canonical: bool = True,
    lane: str = "openrouter",
    power_config=None,
) -> CellConfig:
    return CellConfig(
        optimizer=optimizer,
        env=env,
        lane=lane,
        attempt=attempt,
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        canonical=canonical,
        proposer_config=proposer_config(),
        proposer_transport=proposer_transport,
        rollout_transport=rollout_transport,
        execution_policy=runner_execution_policy(),
        repeats=3,
        pool_n_per_stratum=_pool_n(env),
        split_sizes=SPLIT,
        execution_mode=ExecutionMode.IN_PROCESS,
        power_config=power_config,
    )


def test_cell_improvement_script(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env,
        optimizer="copro",
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        proposer_transport=ScriptedProposer((WIN,)),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg,
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    assert r.status == "improved"
    assert r.baseline_official == pytest.approx(0.0)
    assert r.best_official == pytest.approx(1.0)
    assert r.delta == pytest.approx(1.0)
    assert r.delta is not None
    assert r.ci95 is not None
    assert r.ci95[0] <= r.delta <= r.ci95[1]
    assert r.internal_evals_count >= 1
    assert r.spend_usd == pytest.approx(0.5)
    # The ledger line lands.
    assert len(ledger.cells()) == 1


def test_cell_records_task_side_latency_telemetry(tmp_path: Path) -> None:
    # (Task 20) A completed cell records per-cell task-side telemetry summed
    # from the partial log. The FakeTransport supplies no usage block, so token
    # coverage is 0 (coverage-honest -- never a fake 0 total), but the per-call
    # latency IS captured from the driver clock, so latency_coverage > 0.
    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env, optimizer="copro",
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        proposer_transport=ScriptedProposer((WIN,)),
    )
    ledger = Ledger(root=tmp_path)
    r = run_cell(
        cfg, ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    ).record
    tel = r.telemetry
    # Latency is captured for every driven call.
    assert tel.latency_coverage > 0
    assert tel.total_latency_s is not None
    assert tel.mean_latency_s is not None
    # No usage block -> token totals stay None (NOT a fake 0), coverage 0.
    assert tel.token_coverage == 0
    assert tel.total_tokens is None
    assert tel.total_reasoning_tokens is None


def test_cell_bad_placeholder_candidate_does_not_kill_cell(
    tmp_path: Path,
) -> None:
    # Regression for the live c22 crash: the proposer (openai/gpt-5.4-nano)
    # emitted a candidate with the unknown placeholder {question}, which is not
    # one of c22's prompt_inputs keys. Previously the render raised a KeyError
    # that killed the whole cell with no ledger line. Now the bad candidate is
    # REJECTED at intake (no eval spend), the run completes, and the valid
    # winning candidate is selected as best -- a clean ledger line lands.
    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env,
        optimizer="copro",
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        proposer_transport=ScriptedProposer((BAD, WIN)),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg,
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    # The cell finalized (no crash) and the valid winner beat the baseline.
    assert r.status == "improved"
    assert r.best_official == pytest.approx(1.0)
    # A ledger line landed (the crash previously produced none).
    assert len(ledger.cells()) == 1


@dataclass
class _PoisonRoundTwoTransport:
    """Fail every internal call whose prompt renders the poison template.

    Models the (optimizer x postgres) live shape: a depth>=2 optimizer's
    round-2 candidate whose internal rollouts all fail transiently, leaving its
    aggregate missing. Every other candidate (round-1 winner, naive, ceiling)
    scores cleanly. Official arms never render the poison template, so they are
    unaffected.
    """

    poison_marker: str
    reply: ReplyFn
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
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


def test_cell_unscorable_round2_candidate_completes_loud_not_silent(
    tmp_path: Path,
) -> None:
    # The (optimizer x postgres) live defect: a depth>=2 optimizer's round-2
    # candidate that could not be scored used to abort the whole optimize run,
    # discarding round-1 progress -> incomplete-arm, optimizer_steps=0, and NO
    # detail on the ledger line (the silence the coordinator flagged). Now the
    # failure is isolated to that candidate; the cell COMPLETES with a real
    # delta and nonzero optimizer_steps, and the drop is LOUD + typed on the
    # ledger note.
    env = "c11"
    exp = tiny_experiment(env)
    win_r1 = "R1_WINNER {input}"
    poison_r2 = "R2_POISON {input}"
    proposer = FakeProposerTransport(
        script={
            ("seed_proposal", 0): (win_r1,),
            ("history_proposal", 1): (poison_r2,),
        },
        default=("neutral {input}",),
    )
    cfg = _config(
        env,
        optimizer="copro",
        rollout_transport=_PoisonRoundTwoTransport(
            poison_marker="R2_POISON",
            reply=improvement_reply(exp, win_r1),
        ),
        proposer_transport=proposer,
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg,
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    # The cell COMPLETED with a real delta (pre-fix: incomplete-arm/None).
    assert r.status != "incomplete-arm"
    assert r.status == "improved"
    assert r.delta is not None
    assert r.best_official == pytest.approx(1.0)
    # Nonzero optimizer work survived the round-2 failure (pre-fix: 0).
    assert r.optimizer_steps > 0
    # LOUD + typed: the ledger note names the unscorable-candidate drop count.
    assert "unscorable_candidate_internal_eval" in r.escalation_note
    assert "not scored" in r.escalation_note
    assert len(ledger.cells()) == 1


@dataclass
class _SequencedCodexInvoker:
    """A codex-CLI invoker returning a scripted template per successive call.

    The codex OPTIMIZER drafts ``returned_proposal_count = 4`` candidates in
    one round, so ``draft`` calls the invoker 4 times; this returns
    ``templates[i]`` for call ``i`` (last value repeated if the script short).
    """

    templates: tuple[str, ...]
    calls: int = 0

    def __call__(self, *, prompt: str, model: str) -> CodexInvocation:
        idx = min(self.calls, len(self.templates) - 1)
        self.calls += 1
        return CodexInvocation(text=self.templates[idx], returncode=0)


def test_cell_codex_optimizer_completes_with_isolation_and_trace(
    tmp_path: Path,
) -> None:
    # The codex OPTIMIZER's live path drafts through the local codex CLI (its
    # proposer IS the codex CLI). Driven end-to-end with an injected
    # CodexProposerTransport (fake invoker, no network): a winning draft + a
    # poison draft whose internal rollouts fail. The cell COMPLETES (pre-fix a
    # live codex cell raised an unhandled RuntimeError from the placeholder
    # proposer), the poison candidate is ISOLATED (never best), and the
    # optimizer-search TRACE is persisted -- all free via the shared
    # run_optimize seam.
    import json

    env = "c11"
    exp = tiny_experiment(env)
    win = "CODEX_WIN {input}"
    poison = "CODEX_POISON {input}"
    invoker = _SequencedCodexInvoker(
        templates=(win, poison, "neutral-a {input}", "neutral-b {input}")
    )
    proposer = CodexProposerTransport(model="gpt-5.6", invoker=invoker)
    cfg = _config(
        env,
        optimizer="codex",
        rollout_transport=_PoisonRoundTwoTransport(
            poison_marker="CODEX_POISON",
            reply=improvement_reply(exp, win),
        ),
        proposer_transport=proposer,
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg,
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    # The codex cell COMPLETED (no unhandled RuntimeError); picked the winner.
    assert r.status == "improved"
    assert r.best_official == pytest.approx(1.0)
    assert r.optimizer_steps > 0
    # codex drafted 4 candidates through the codex CLI (one round of 4).
    assert invoker.calls == 4
    # Per-candidate isolation: the poison candidate is dropped, loud + typed.
    assert "unscorable_candidate_internal_eval" in r.escalation_note
    # Trace persistence: the codex cell's search trace lands on disk with the
    # accepted-candidate prompt text + per-step evidence.
    trace_path = ledger.optimization_trace_path(r.cell_id)
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text())
    assert trace["optimizer"] == "codex"
    assert trace["best_candidate_template"] == win
    poisoned = [
        s for s in trace["steps"]
        if s["rejected"]
        and s["rejected_reason"] == "unscorable_candidate_internal_eval"
    ]
    assert poisoned, "the poison codex candidate must be a rejected trace step"


class _AllFailedDraftProposer:
    """A proposer transport whose EVERY draft is a typed failure (no template).

    Models a proposer outage (e.g. a model the CLI rejects with HTTP 400):
    every slot is a typed failure, never a base-template echo.
    """

    def draft(
        self, config: ProposerConfig, request: ProposalRequest, count: int
    ) -> tuple[ProposalDraft, ...]:
        return tuple(
            ProposalDraft.failure(
                detail=f"scripted total proposer outage (slot {i})",
                request_evidence={"draft_index": i},
            )
            for i in range(count)
        )


def test_cell_all_drafts_failed_finalizes_loud_proposer_failure(
    tmp_path: Path,
) -> None:
    # A proposer OUTAGE (every draft a typed failure) must NOT be hidden behind
    # a naive-as-best "no-improvement": the cell finalizes with the loud typed
    # ``proposer-failure`` status naming the per-draft reasons, emits NO
    # best/delta/headroom, and is impossible to confuse with an honest
    # no-improvement in the ledger, trace, and log line.
    import json

    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env,
        optimizer="copro",
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        proposer_transport=_AllFailedDraftProposer(),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg,
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    # LOUD typed status, distinct from no-improvement.
    assert r.status == "proposer-failure"
    assert r.status != "no-improvement"
    # No best/delta/headroom off a run that explored zero real candidates.
    assert r.best_official is None
    assert r.delta is None
    assert r.headroom_delta is None
    # The note names the proposer outage + per-draft reasons.
    assert "proposer failure" in r.escalation_note.lower()
    assert "0 real candidates" in r.escalation_note
    # NOT a completed terminal status: a re-run supersedes it (resumable).
    assert not r.is_completed()
    # The trace records the failed-draft slots (no phantom candidates).
    trace_path = ledger.optimization_trace_path(r.cell_id)
    trace = json.loads(trace_path.read_text())
    assert trace["all_drafts_failed"] is True
    assert trace["scored_candidate_count"] == 0
    assert trace["failed_draft_count"] > 0
    for s in trace["steps"]:
        assert s["rejected_reason"] == "proposer_draft_failed"
        assert s["template"] == ""  # no fabricated candidate
    # The per-env official cache is NOT poisoned by a proposer-failure cell.
    assert ledger.env_cache_for(env, task_model=TASK_MODEL) is None


def test_cell_no_improvement_script(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env,
        optimizer="copro",
        rollout_transport=FakeTransport(reply=no_improvement_reply(exp)),
        proposer_transport=ScriptedProposer(("also-loses {input}",)),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg,
        ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.1)]),
    )
    r = outcome.record
    assert r.status == "no-improvement"
    assert r.baseline_official == pytest.approx(0.0)
    assert r.best_official == pytest.approx(0.0)
    assert r.delta == pytest.approx(0.0)


def test_cell_ceiling_cached_once_per_env(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    ledger = Ledger(root=tmp_path)
    # First cell computes and records the ceiling.
    run_cell(
        _config(
            env,
            optimizer="copro",
            rollout_transport=FakeTransport(reply=correct_reply(exp)),
            proposer_transport=ScriptedProposer((WIN,)),
        ),
        ledger=ledger,
    )
    ceiling = ledger.ceiling_for(env)
    assert ceiling is not None
    # A second (different optimizer) cell reuses the cached ceiling verbatim.
    out2 = run_cell(
        _config(
            env,
            optimizer="miprov2",
            rollout_transport=FakeTransport(reply=correct_reply(exp)),
            proposer_transport=ScriptedProposer((WIN,)),
        ),
        ledger=ledger,
    )
    assert out2.record.ceiling_official == ceiling


def test_cell_records_execution_mode(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env,
        optimizer="eval",
        rollout_transport=FakeTransport(reply=correct_reply(exp)),
        proposer_transport=ScriptedProposer(()),
    )
    ledger = Ledger(root=tmp_path)
    out = run_cell(cfg, ledger=ledger)
    # eval identity: no proposal steps (naive == best).
    assert out.record.optimizer_steps == 0
    # The persisted aggregate artifacts carry the execution mode.
    # (recorded on the SplitEvaluation; the cell status is terminal)
    assert out.record.status in ("improved", "no-improvement")


def test_reserve_guard_refuses_canonical_below_reserve(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env,
        optimizer="copro",
        rollout_transport=FakeTransport(reply=no_improvement_reply(exp)),
        proposer_transport=ScriptedProposer(("x {input}",)),
    )
    ledger = Ledger(root=tmp_path)
    # remaining $10 < reserve $18.60 -> refuse to start.
    with pytest.raises(ReserveError, match="reserve"):
        run_cell(
            cfg,
            ledger=ledger,
            budget=BudgetGuard(),
            credits_fetcher=credits_fetcher([(700.0, 690.0), (700.0, 690.0)]),
        )
    # No cell line was appended (the cell never started).
    assert ledger.cells() == []


def test_cell_zero_success_baseline_fails_loudly(tmp_path: Path) -> None:
    # Live round-1: when every baseline rollout fails (missing_base_url), the
    # cell must NOT record a null-scores line -- it raises CellBaselineFailure
    # so the CLI can exit non-zero. No ledger line is appended.
    env = "c11"
    tiny_experiment(env)
    cfg = _config(
        env,
        optimizer="copro",
        rollout_transport=FailingTransport(code="missing_base_url"),
        proposer_transport=ScriptedProposer((WIN,)),
    )
    ledger = Ledger(root=tmp_path)
    with pytest.raises(CellBaselineFailure, match="plumbing failure"):
        run_cell(cfg, ledger=ledger)
    # No cell line recorded (the failed baseline never becomes a result).
    assert ledger.cells() == []


def test_stop_loss_halts_cell(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env,
        optimizer="copro",
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        proposer_transport=ScriptedProposer((WIN,)),
    )
    ledger = Ledger(root=tmp_path)
    # A tiny expected cost so the observed $5 spend crosses 2x -> halted.
    budget = BudgetGuard(expected_cell_usd=1.0)  # stop-loss $2
    outcome = run_cell(
        cfg,
        ledger=ledger,
        budget=budget,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 621.0)]),
    )
    assert outcome.record.status == "halted"
    assert outcome.record.spend_usd == pytest.approx(5.0)


# --- Opt-in pre-run statistical-power stage --------------------------------


def _power_cfg(alpha: float = 0.25):
    from whetstone.runner.power import PowerConfig

    # A small repeat cap + trials keeps the fake-cell test fast + seeded.
    return PowerConfig(alpha=alpha, repeat_cap=6, trials=400, seed=99)


def test_cell_power_stage_off_is_byte_identical(tmp_path: Path) -> None:
    # The opt-in inertness guarantee: with power_config None (the default), the
    # cell record + optimization trace are IDENTICAL to a run that never knew
    # about the power stage. Only power_sizing/power_analysis_ref differ (both
    # null), and no power_analysis artifact is written.
    env = "c11"
    exp = tiny_experiment(env)

    def _run(root: Path, power_config):
        cfg = _config(
            env, optimizer="copro",
            rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
            proposer_transport=ScriptedProposer((WIN,)),
            power_config=power_config,
        )
        ledger = Ledger(root=root)
        out = run_cell(
            cfg, ledger=ledger,
            credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
        )
        return out.record, ledger

    off_record, off_ledger = _run(tmp_path / "off", None)
    # A second OFF run in a fresh root must produce an identical record (minus
    # the cell-id-independent bits, which are the same here).
    off2_record, _ = _run(tmp_path / "off2", None)

    off_dump = off_record.model_dump(mode="json")
    off2_dump = off2_record.model_dump(mode="json")
    # Wall time + per-call latency telemetry + the ISO wall-clock started_at/
    # finished_at (task 26) are the nondeterministic fields (real clock); drop
    # them before the byte-identity comparison.
    for key in ("wall_s", "telemetry", "started_at", "finished_at"):
        off_dump.pop(key)
        off2_dump.pop(key)
    assert off_dump == off2_dump
    # Power fields are inert (null) and NO artifact directory was created.
    assert off_record.power_sizing is None
    assert off_record.artifacts.power_analysis_ref is None
    assert not (tmp_path / "off" / "power_analysis").exists()
    # The optimizer trace records the brief-sourced internal task count.
    import json

    trace = json.loads(
        off_ledger.optimization_trace_path(off_record.cell_id).read_text()
    )
    assert trace["internal_task_count_source"] == "brief"


def test_cell_power_stage_on_writes_artifact_and_sets_sizes(
    tmp_path: Path,
) -> None:
    # With the power stage ON: a power_analysis artifact is written, the cell
    # line references it + records recommended-vs-used sizing, and the trace
    # trace records the power-sourced internal task count.
    import json

    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env, optimizer="copro",
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        proposer_transport=ScriptedProposer((WIN,)),
        power_config=_power_cfg(),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg, ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    # The cell still completed (the power stage only sizes, never blocks).
    assert r.status in {"improved", "inconclusive", "no-improvement"}
    # The power_analysis artifact was written and is referenced from the line.
    power_path = ledger.power_analysis_path(r.cell_id)
    assert power_path.exists()
    assert r.artifacts.power_analysis_ref == str(
        power_path.relative_to(tmp_path)
    )
    art = json.loads(power_path.read_text())
    assert art["schema"] == "whetstone.runner.power_analysis/v1"
    # Full n x r surface + variance decomposition + recommendation persisted.
    assert art["surface"]
    assert "variance_decomposition" in art
    assert "recommendation" in art
    assert art["pool_ceiling"] == len(exp.eval_configs.internal.instances)
    # The cell line records recommended-vs-used internal sizing.
    ps = r.power_sizing
    assert ps is not None
    assert ps.used_n_tasks <= ps.pool_ceiling  # clamped to pool
    assert ps.recommended_n_tasks >= 1
    # The trace records the power-sourced internal task count (recommended-vs-
    # brief both retained).
    trace = json.loads(
        ledger.optimization_trace_path(r.cell_id).read_text()
    )
    assert trace["internal_task_count_source"] == "power_stage"
    assert trace["internal_task_count_scaled"] == ps.used_n_tasks
    assert "internal_task_count_brief" in trace
    # The trace reference resolves + the power reference resolves.
    trace_ref = r.artifacts.optimization_result_ref
    power_ref = r.artifacts.power_analysis_ref
    assert trace_ref is not None and power_ref is not None
    assert (tmp_path / trace_ref).exists()
    assert (tmp_path / power_ref).exists()


def test_powered_cell_records_nonzero_spend_when_fetcher_reports_delta(
    tmp_path: Path,
) -> None:
    # Spend-attribution regression guard (tasks 6-9 heartbeat-$0 scare): a cell
    # run WITH the power stage on the openrouter lane must still attribute the
    # OpenRouter credits delta to the cell line. The fetcher reports a real
    # usage delta (616.0 -> 616.5 total_usage == $0.50 burned), and the
    # recorded spend_usd must reflect it -- not collapse to $0.
    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env, optimizer="copro",
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        proposer_transport=ScriptedProposer((WIN,)),
        power_config=_power_cfg(),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg, ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    # The openrouter completion path sums this cell's before/after credits
    # delta from spend.jsonl; the $0.50 burn must land on the line.
    assert r.spend_usd == pytest.approx(0.5)
    assert r.spend_usd > 0.0


# --- Rollout-output sidecar (qualitative prompt->output logging) -----------


def test_cell_writes_rollout_output_sidecar_with_full_coverage(
    tmp_path: Path,
) -> None:
    # Every internal candidate eval row AND every official arm row is captured
    # in the per-cell rollout-output sidecar with the FULL output text + score;
    # the trace references it and the reference resolves.
    import json

    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env, optimizer="copro",
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        proposer_transport=ScriptedProposer((WIN,)),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg, ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    sidecar = ledger.rollout_outputs_path(r.cell_id)
    assert sidecar.exists(), "the rollout-output sidecar must be written"
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert rows
    # Every row carries the required fields (full text, never truncated) plus
    # the task-26 provenance superset: a schema stamp, structured id components
    # (cell_id/env/optimizer/attempt/lane/model), and per-call finish_reason +
    # provider_error. (QA rows carry no ed1 budget, so max_budget is absent.)
    for row in rows:
        assert set(row) == {
            "schema", "cell_id", "env", "optimizer", "attempt", "lane",
            "model", "split_role", "candidate_id", "instance_id", "repeat",
            "output_text", "score", "failure_code",
            "finish_reason", "provider_error",
            # Task 28 item 3: an ISO-8601 UTC recording timestamp on every row.
            "at",
        }
        assert row["at"]  # non-empty on success rows (and error rows alike)
    roles = {row["split_role"] for row in rows}
    # Official arms AND internal candidate evals are ALL covered.
    assert "official_naive" in roles
    assert "official_ceiling" in roles
    assert "official_best" in roles
    assert "internal_naive" in roles
    assert "internal_candidate" in roles
    # The winning candidate's official rows carry its FULL gold output text.
    win_rows = [
        row for row in rows
        if row["split_role"] == "official_best" and row["output_text"]
    ]
    assert win_rows
    # The trace references the sidecar and the reference resolves.
    trace = json.loads(
        ledger.optimization_trace_path(r.cell_id).read_text()
    )
    assert trace["rollout_outputs_ref"] == str(
        sidecar.relative_to(tmp_path)
    )
    assert (tmp_path / trace["rollout_outputs_ref"]).exists()


def test_cell_rollout_sidecar_covers_eval_row_official_arms(
    tmp_path: Path,
) -> None:
    # The eval (identity) optimizer drafts nothing, but its OFFICIAL arms
    # (naive/ceiling) outputs are still captured -- the directive applies to
    # eval cells' official arms too.
    import json

    env = "c11"
    exp = tiny_experiment(env)
    cfg = _config(
        env, optimizer="eval",
        rollout_transport=FakeTransport(reply=correct_reply(exp)),
        proposer_transport=ScriptedProposer(()),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg, ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    sidecar = ledger.rollout_outputs_path(r.cell_id)
    assert sidecar.exists()
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    roles = {row["split_role"] for row in rows}
    assert "official_naive" in roles
    assert "official_ceiling" in roles


def _big_internal_config(optimizer: str, exp, power_config):
    # A c11 cell with a 12-task INTERNAL split so MIPROv2's 8-task minibatch is
    # below the pool (the analysis shape) and the power stage can recommend a
    # larger n than the brief. WIN scores gold under improvement_reply.
    return CellConfig(
        optimizer=optimizer, env="c11", lane="openrouter", attempt=0,
        task_model=TASK_MODEL, proposer_model=PROPOSER_MODEL, canonical=True,
        proposer_config=proposer_config(),
        proposer_transport=ScriptedProposer((WIN,)),
        rollout_transport=FakeTransport(reply=improvement_reply(exp, WIN)),
        execution_policy=runner_execution_policy(),
        repeats=3, pool_n_per_stratum=4, split_sizes=(12, 2, 2),
        execution_mode=ExecutionMode.IN_PROCESS,
        power_config=power_config,
    )


def test_power_stage_applies_recommended_repeats_to_internal_evals(
    tmp_path: Path,
) -> None:
    # The power recommendation is APPLIED (not just recorded): the internal
    # candidate evals actually run at used_repeats x used_n_tasks. Verified via
    # the rollout-output sidecar (task 8), whose internal rows reflect the
    # actual repeats/tasks driven.
    import json

    from whetstone.envs.factory import build_env_experiment
    from whetstone.runner.power import PowerConfig

    exp = build_env_experiment(
        "c11", model=TASK_MODEL, pool_n_per_stratum=4, split_sizes=(12, 2, 2),
    )
    cfg = _big_internal_config(
        "copro", exp,
        PowerConfig(alpha=0.25, repeat_cap=5, trials=300, seed=5),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg, ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    ps = r.power_sizing
    assert ps is not None
    # The internal candidate eval rows actually driven reflect the APPLIED
    # sizes: max repeat index == used_repeats - 1, distinct tasks == used_n.
    rows = [
        json.loads(line)
        for line in ledger.rollout_outputs_path(r.cell_id)
        .read_text().splitlines()
    ]
    internal = [
        row for row in rows if row["split_role"] == "internal_candidate"
    ]
    assert internal, "the power-sized internal candidate evals must have run"
    max_repeat = max(row["repeat"] for row in internal)
    distinct_tasks = {row["instance_id"] for row in internal}
    assert max_repeat + 1 == ps.used_repeats  # repeats APPLIED
    assert len(distinct_tasks) == ps.used_n_tasks  # task count APPLIED
    # The trace records the APPLIED repeat count (not config.repeats) + source.
    trace = json.loads(
        ledger.optimization_trace_path(r.cell_id).read_text()
    )
    assert trace["internal_task_count_source"] == "power_stage"
    assert trace["internal_repeat_count_as_run"] == ps.used_repeats
    assert trace["internal_task_count_scaled"] == ps.used_n_tasks


def test_power_stage_overrides_miprov2_minibatch(tmp_path: Path) -> None:
    # MIPROv2's internal_task_count is PINNED at the 8-task minibatch no matter
    # the pool (the analysis's weakest link). With the power stage on, the
    # recommended n_tasks (up to the pool) OVERRIDES that minibatch -- the used
    # task count comes from the recommendation, not the fixed 8.
    import json

    from whetstone.envs.factory import build_env_experiment
    from whetstone.runner.optimizers import scaled_hyperparameters
    from whetstone.runner.power import PowerConfig

    exp = build_env_experiment(
        "c11", model=TASK_MODEL, pool_n_per_stratum=4, split_sizes=(12, 2, 2),
    )
    pool = len(exp.eval_configs.internal.instances)
    assert pool == 12
    # Sanity: WITHOUT the power stage MIPROv2 would clamp to the 8-task
    # minibatch (below the pool of 12) -- the analysis's weak spot.
    brief = scaled_hyperparameters("miprov2", internal_pool_size=pool)
    assert brief["internal_task_count_scaled"] == 8

    cfg = _big_internal_config(
        "miprov2", exp,
        PowerConfig(alpha=0.25, repeat_cap=5, trials=300, seed=5),
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg, ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    ps = r.power_sizing
    assert ps is not None
    trace = json.loads(
        ledger.optimization_trace_path(r.cell_id).read_text()
    )
    # The used internal task count is the power recommendation (clamped to
    # pool), NOT the brief's fixed 8-task minibatch.
    assert trace["internal_task_count_source"] == "power_stage"
    assert trace["internal_task_count_scaled"] == ps.used_n_tasks
    # The brief minibatch (8) is retained for the recommended-vs-used record,
    # and the APPLIED value came from the power rec, not the fixed 8.
    assert trace["internal_task_count_brief"] == 8
    # The applied task count is what the internal evals actually ran on.
    rows = [
        json.loads(line)
        for line in ledger.rollout_outputs_path(r.cell_id)
        .read_text().splitlines()
    ]
    internal_tasks = {
        row["instance_id"] for row in rows
        if row["split_role"] in ("internal_candidate", "internal_naive")
    }
    assert len(internal_tasks) == ps.used_n_tasks


def test_power_stage_base_rate_zero_anchor_floors_at_brief_sizes(
    tmp_path: Path,
) -> None:
    # Boundary guard (the c11 delta-0 flaw): when the naive anchor sits at the
    # 0.0 base-rate floor, the anchor-arm variance decomposition is degenerate
    # and the raw power recommendation collapses (e.g. n_tasks=2) -- far below
    # the candidate-comparison variance once candidates move off the naive base
    # rate. The opt-in flag means "at least as powered as before", so the USED
    # sizes must be FLOORED at the as-run brief (n = brief scope clamped to
    # pool, r = config.repeats) -- the stage may only ADD power, never subtract
    # below it. Reproduced on miprov2 (brief internal scope = 8) with a 12-task
    # pool and a ceiling-only reply (naive base_rate == 0, ceiling == 1).
    import json

    from whetstone.envs.factory import build_env_experiment
    from whetstone.runner.optimizers import scaled_hyperparameters
    from whetstone.runner.power import PowerConfig

    exp = build_env_experiment(
        "c11", model=TASK_MODEL, pool_n_per_stratum=4, split_sizes=(12, 2, 2),
    )
    pool = len(exp.eval_configs.internal.instances)
    assert pool == 12
    brief_n = scaled_hyperparameters(
        "miprov2", internal_pool_size=pool
    )["internal_task_count_scaled"]
    assert brief_n == 8  # the minibatch scope, below the pool

    cfg = _big_internal_config(
        "miprov2", exp,
        # alpha=0.1 widens the target gap -> the degenerate rec would drop to a
        # tiny n_tasks (the a4 shape) absent the floor.
        PowerConfig(alpha=0.1, repeat_cap=5, trials=300, seed=5),
    )
    from dataclasses import replace

    cfg = replace(
        cfg, rollout_transport=FakeTransport(reply=ceiling_only_reply(exp))
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg, ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    ps = r.power_sizing
    assert ps is not None
    # The degenerate base-rate-0 anchor: the RAW recommendation is recorded and
    # may be below the brief (that's the boundary condition being guarded).
    art = json.loads(ledger.power_analysis_path(r.cell_id).read_text())
    assert art["variance_decomposition"]["base_rate"] == 0.0
    # The USED sizes are floored at the brief and never subtract below it.
    assert ps.used_n_tasks >= brief_n
    assert ps.used_n_tasks <= ps.pool_ceiling
    assert ps.used_repeats >= cfg.repeats
    # The recommendation is still visible (auditable) on the line + artifact.
    assert (
        ps.recommended_n_tasks
        == art["recommendation"]["recommended_n_tasks"]
    )
    # The internal evals actually RAN at the floored sizes.
    rows = [
        json.loads(line)
        for line in ledger.rollout_outputs_path(r.cell_id)
        .read_text().splitlines()
    ]
    internal = {
        row["instance_id"] for row in rows
        if row["split_role"] in ("internal_candidate", "internal_naive")
    }
    assert len(internal) == ps.used_n_tasks >= brief_n


# --- Task 28 item 1: attractor pull carried onto the CellRecord --------------


def _attractor_config(env: str) -> CellConfig:
    # A minimal CellConfig for the pure ``_ed1m_attractor`` helper (it reads
    # only ``config.env``); built directly so it does not require ``env`` to be
    # in the test registry.
    return CellConfig(
        optimizer="copro",
        env=env,
        lane="openrouter",
        attempt=0,
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        canonical=True,
        proposer_config=proposer_config(),
        proposer_transport=FakeProposerTransport(script={}, default=()),
        rollout_transport=FakeTransport(reply=lambda _p: "x"),
        execution_policy=runner_execution_policy(),
        repeats=3,
        pool_n_per_stratum=1,
        split_sizes=SPLIT,
        execution_mode=ExecutionMode.IN_PROCESS,
    )


def test_ed1m_attractor_helper_reports_best_arm_mean() -> None:
    # (Task 28 item 1) The ed1m attractor sub-object carries the BEST arm's
    # reported mean + per-task vector; the mean is over NON-null per-task
    # values (a null task -- no discriminating sample -- is never zeroed in).
    from whetstone.runner.cell import _ed1m_attractor

    cfg = _attractor_config("ed1m")
    by_role = {
        "official_best": (0.5, (1.0, None, 0.0)),
        "official_naive": (0.0, (0.0, 0.0)),
    }
    attractor = _ed1m_attractor(cfg, by_role)
    assert attractor is not None
    # Mean over the two non-null tasks: (1.0 + 0.0) / 2 = 0.5.
    assert attractor.mean == pytest.approx(0.5)
    assert attractor.per_task == (1.0, None, 0.0)
    assert attractor.sampled_task_count == 2


def test_ed1m_attractor_helper_null_when_no_sample() -> None:
    # ed1m ran but no task produced a discriminating sample -> mean None (a
    # non-null record with a null mean, never a fake 0).
    from whetstone.runner.cell import _ed1m_attractor

    cfg = _attractor_config("ed1m")
    attractor = _ed1m_attractor(cfg, {"official_best": (None, (None, None))})
    assert attractor is not None
    assert attractor.mean is None
    assert attractor.sampled_task_count == 0


def test_attractor_helper_none_for_non_ed1m_envs() -> None:
    # ed1 / QA / d1 do not measure attractor pull -> None (null-not-zero).
    from whetstone.runner.cell import _ed1m_attractor

    for env in ("ed1", "c11", "d1"):
        assert _ed1m_attractor(_attractor_config(env), {}) is None
