"""ed1 (enc-dec HumanEval compression) env tests -- no network, no Docker.

Covers the two gaps + the pilot support:
* the 3-node Encoder->Decoder->Eval graph (structure + ``budget_ratio`` folding
  into ``graph_hash``);
* the HumanEval code scoring wired via a real (offline snapshot) fixture task
  through the LOCAL subprocess runner (no container);
* a full fake-transport enc-dec cell end-to-end recording BOTH scores;
* budget derivation + the design's "guidance, not enforcement" (over-budget is
  NOT clipped/failed);
* the pinned dataset revision recorded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whetstone.envs.ed1 import (
    ED1_CANONICAL_MODEL,
    ED1_DATASET_REVISION,
    ED1_ENV_NAME,
    build_ed1_experiment,
    ed1_ceiling_candidate,
    ed1_initial_candidate,
    load_ed1_tasks,
)
from whetstone.envs.ed1_scoring import score_ed1_submission
from whetstone.envs.encdec_rollout import (
    DECODER_NODE_ID,
    ENCODER_NODE_ID,
    EVAL_NODE_ID,
    build_encdec_rollout_definition,
    encdec_graph_definition,
)
from whetstone.optimization.mutation import MUTATION_FIELD

# --- Gap 1: the 3-node graph + budget identity ------------------------------


def test_encdec_graph_is_three_nodes_terminal_eval() -> None:
    d = encdec_graph_definition()
    assert [n.node_id for n in d.nodes] == [
        ENCODER_NODE_ID, DECODER_NODE_ID, EVAL_NODE_ID,
    ]
    assert d.terminal_node_id == EVAL_NODE_ID


def test_budget_ratio_and_model_fold_into_graph_hash() -> None:
    base = build_encdec_rollout_definition(
        ED1_ENV_NAME, model=ED1_CANONICAL_MODEL,
        procedure_config_hash="a" * 64, budget_ratio=0.5,
    )
    other_ratio = build_encdec_rollout_definition(
        ED1_ENV_NAME, model=ED1_CANONICAL_MODEL,
        procedure_config_hash="a" * 64, budget_ratio=0.75,
    )
    other_model = build_encdec_rollout_definition(
        ED1_ENV_NAME, model="openai/gpt-5-nano",
        procedure_config_hash="a" * 64, budget_ratio=0.5,
    )
    # A distinct ratio is a distinct Rollout Variant; so is a distinct model.
    assert base.graph_hash != other_ratio.graph_hash
    assert base.graph_hash != other_model.graph_hash
    assert base.budget_rule.ratio == 0.5


def test_encoder_and_decoder_share_the_same_route() -> None:
    rd = build_encdec_rollout_definition(
        ED1_ENV_NAME, model=ED1_CANONICAL_MODEL,
        procedure_config_hash="a" * 64, budget_ratio=0.5,
    )
    # The same Provider Call Config (route) plays both encoder + decoder.
    route = rd.provider_call_config.definition.route
    assert route.model == ED1_CANONICAL_MODEL


# --- Gap 2: HumanEval scoring via a real offline fixture task ---------------


def _fixture_task():
    # A real HumanEval+ task from the committed offline snapshot (no network).
    return load_ed1_tasks(prefer_snapshot=True, limit=1)[0].humaneval_task


def test_humaneval_scoring_canonical_passes_wrong_fails() -> None:
    ht = _fixture_task()
    # The canonical solution passes the task's own test suite.
    good = score_ed1_submission(
        raw_submission=ht.ground_truth_code, task=ht, timeout_seconds=30.0
    )
    assert good.passed is True
    assert good.infrastructure_unknown is False
    # A wrong body definitively fails (score 0), NOT infrastructure-unknown.
    bad_body = f"def {ht.entry_point}(*a, **k):\n    return None\n"
    bad = score_ed1_submission(
        raw_submission=bad_body, task=ht, timeout_seconds=30.0
    )
    assert bad.passed is False
    assert bad.infrastructure_unknown is False


def test_dataset_revision_is_pinned_and_recorded() -> None:
    exp = build_ed1_experiment(
        tasks=load_ed1_tasks(prefer_snapshot=True, limit=3),
        internal_n=2, official_n=1,
    )
    assert exp.dataset_revision == ED1_DATASET_REVISION
    assert ED1_DATASET_REVISION  # a concrete pinned revision string


# --- Full fake-transport enc-dec cell end-to-end (both scores) --------------


def _fake_encdec_transport(tasks):
    from tests.envs.support import FakeTransport

    by_entry = {
        t.humaneval_task.entry_point: t.humaneval_task.ground_truth_code
        for t in tasks
    }

    def reply(prompt: str) -> str:
        if prompt.startswith("Provide") or prompt.startswith("Compress"):
            for ep in by_entry:
                if f"def {ep}(" in prompt:
                    return f"REBUILD:{ep}"
            return "REBUILD:x"
        for ep, gt in by_entry.items():
            if f"REBUILD:{ep}" in prompt:
                return gt
        return "def _x():\n    return None\n"

    return FakeTransport(reply=reply)


def test_ed1_cell_end_to_end_records_both_scores(tmp_path: Path) -> None:
    from tests.envs.support import execution_policy
    from tests.runner.support import credits_fetcher, proposer_config
    from whetstone.optimization.proposer import FakeProposerTransport
    from whetstone.runner.cell import CellConfig, run_cell
    from whetstone.runner.execution_mode import ExecutionMode
    from whetstone.runner.ledger import Ledger

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=3)
    transport = _fake_encdec_transport(tasks)
    cfg = CellConfig(
        optimizer="eval", env=ED1_ENV_NAME, lane="openrouter", attempt=0,
        task_model=ED1_CANONICAL_MODEL, proposer_model="none", canonical=True,
        proposer_config=proposer_config(),
        proposer_transport=FakeProposerTransport(script={}, default=()),
        rollout_transport=transport,
        execution_policy=execution_policy(max_attempts=1),
        repeats=1, official_repeats=1,
        execution_mode=ExecutionMode.IN_PROCESS,
        budget_ratio=0.5, ed1_task_limit=3,
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg, ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    # Both objectives land: the PRIMARY pass-rate score + the REPORTED
    # compression alongside (budget + pinned dataset provenance).
    assert r.best_official == pytest.approx(1.0)  # canonical solutions pass
    assert r.dual_scores is not None
    assert r.dual_scores.best_compression is not None
    assert 0.0 < r.dual_scores.best_compression < 1.0  # a real zstd ratio
    assert r.dual_scores.budget_ratio == 0.5
    assert r.dual_scores.dataset_revision == ED1_DATASET_REVISION
    # The rollout-output sidecar carries the per-task compression alongside the
    # enc/dec output text.
    import json

    rows = [
        json.loads(line)
        for line in ledger.rollout_outputs_path(r.cell_id)
        .read_text().splitlines()
    ]
    assert rows
    assert any("compression_ratio" in row for row in rows)
    assert any("ENCODER:" in (row["output_text"] or "") for row in rows)


# --- Budget derivation + guidance-not-enforcement ---------------------------


def test_max_budget_is_rounded_ratio_of_input_chars() -> None:
    from whetstone.envs.ed1_eval import _max_budget
    from whetstone.graph.character_budget import CharacterBudgetRule

    rule = CharacterBudgetRule(ratio=0.5)
    assert _max_budget("x" * 411, rule) == 206  # round(0.5 * 411)
    assert _max_budget("x" * 100, rule) == 50


def test_over_budget_encoder_output_is_scored_not_clipped(
    tmp_path: Path,
) -> None:
    # Design: "the budget steers, nothing clips". An over-budget encoder output
    # is NOT truncated or failed -- it just yields a larger compression ratio.
    from tests.envs.support import FakeTransport, execution_policy
    from whetstone.envs.ed1_eval import run_ed1_eval

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=1)
    exp = build_ed1_experiment(tasks=tasks, internal_n=1, official_n=1)
    ht = tasks[0].humaneval_task
    # The encoder deliberately blows the budget (a long description); the
    # decoder returns the canonical solution -> PASS, but a big compression.
    long_desc = "x" * (len(tasks[0].input_code) * 5)

    def reply(prompt: str) -> str:
        if prompt.startswith("Provide") or prompt.startswith("Compress"):
            return long_desc
        return ht.ground_truth_code

    template = ed1_initial_candidate().payload[MUTATION_FIELD]
    ed = run_ed1_eval(
        exp, candidate_template=template, candidate_id="ed1-naive",
        instances=exp.eval_configs.internal.instances,
        execution_policy=execution_policy(max_attempts=1),
        transport=FakeTransport(reply=reply), repeats=1, apply_reward=False,
    )
    # Not failed/clipped: the row scored (pass rate present), and the
    # over-budget
    # description gives a LARGER compression ratio than a within-budget one.
    assert ed.pass_aggregate.aggregation_output.value == pytest.approx(1.0)
    assert ed.compression_aggregate.aggregation_output.value is not None
    assert ed.compression_aggregate.aggregation_output.value > 0.0


# --- Templates: naive vs ceiling are distinct Mutation-Surface templates ----


def test_naive_and_ceiling_encoder_templates_are_distinct() -> None:
    naive = ed1_initial_candidate().payload[MUTATION_FIELD]
    ceiling = ed1_ceiling_candidate().payload[MUTATION_FIELD]
    assert naive != ceiling
    # Both carry the two encoder placeholders the render fills.
    for tmpl in (naive, ceiling):
        assert "{input_code}" in tmpl
        assert "{max_budget}" in tmpl


# --- The ed1 pilot: both probes, dual scores --------------------------------


def test_ed1_pilot_reports_both_probes_dual_scores(tmp_path: Path) -> None:
    from tests.envs.support import execution_policy
    from whetstone.runner.ed1_pilot import run_ed1_pilot

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=3)
    transport = _fake_encdec_transport(tasks)
    report = run_ed1_pilot(
        transport=transport, execution_policy=execution_policy(max_attempts=1),
        tasks=3, repeats=1, concurrency=2,
    )
    # BOTH encoder probes (naive A + ceiling B) measured, each with pass rate +
    # Mean Compression Ratio + the pinned dataset revision.
    assert report.naive.probe == "ed1-naive"
    assert report.ceiling.probe == "ed1-ceiling"
    assert report.naive.pass_rate == pytest.approx(1.0)
    assert report.ceiling.pass_rate == pytest.approx(1.0)
    assert report.naive.mean_compression is not None
    assert report.ceiling.mean_compression is not None
    assert report.dataset_revision == ED1_DATASET_REVISION
    # The report writes to <root>/pilots/ed1.json.
    path = report.write(tmp_path)
    assert path.exists() and path.name == "ed1.json"


def test_ed1_pilot_honors_budget_ratio_override() -> None:
    # (Task 13d) The pilot's --budget-ratio flows into the enc-dec graph: a
    # distinct ratio changes MAX_BUDGET and is recorded on the report.
    from tests.envs.support import execution_policy
    from whetstone.runner.ed1_pilot import run_ed1_pilot

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=3)
    transport = _fake_encdec_transport(tasks)
    report = run_ed1_pilot(
        transport=transport, execution_policy=execution_policy(max_attempts=1),
        tasks=3, repeats=1, concurrency=2, budget_ratio=0.25,
    )
    assert report.budget_ratio == 0.25


def test_ed1_pilot_subcommand_exposes_budget_ratio() -> None:
    # (Task 13d) The flag is on the PILOT subparser (was cell-only), defaulting
    # to the canonical ratio and overridable for a cheap ratio scan.
    from whetstone.envs.ed1 import ED1_DEFAULT_BUDGET_RATIO
    from whetstone.runner.cli import build_parser

    parser = build_parser()
    default_args = parser.parse_args(["pilot", "--env", "ed1"])
    assert default_args.budget_ratio == ED1_DEFAULT_BUDGET_RATIO
    scan_args = parser.parse_args(
        ["pilot", "--env", "ed1", "--budget-ratio", "0.25"]
    )
    assert scan_args.budget_ratio == 0.25


def test_ed1_cell_official_n_slices_pool_not_full_split(
    tmp_path: Path,
) -> None:
    # (Task 13a) --official-n (SamplingOverrides.official_n) must fold into the
    # ed1 pool slice so a reduced-anchor eval cell drives only N official tasks
    # -- NOT the full HumanEval pool (the killed cell drove ~82 tasks because
    # the override was dropped on the ed1 build path).
    from tests.envs.support import execution_policy
    from tests.runner.support import proposer_config
    from whetstone.envs.sampling import SamplingOverrides
    from whetstone.optimization.proposer import FakeProposerTransport
    from whetstone.runner.cell import CellConfig, _build_experiment
    from whetstone.runner.eval_run import official_instances
    from whetstone.runner.execution_mode import ExecutionMode

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=12)
    transport = _fake_encdec_transport(tasks)
    cfg = CellConfig(
        optimizer="eval", env=ED1_ENV_NAME, lane="openrouter", attempt=0,
        task_model=ED1_CANONICAL_MODEL, proposer_model="none", canonical=True,
        proposer_config=proposer_config(),
        proposer_transport=FakeProposerTransport(script={}, default=()),
        rollout_transport=transport,
        execution_policy=execution_policy(max_attempts=1),
        repeats=1, official_repeats=1,
        execution_mode=ExecutionMode.IN_PROCESS,
        budget_ratio=0.5, ed1_task_limit=12,
        sampling_overrides=SamplingOverrides(official_n=3),
    )
    exp = _build_experiment(cfg)
    # The official split is exactly the 3 requested tasks, not the full pool.
    assert len(official_instances(exp)) == 3


def test_ed1_eval_streams_partials_and_resume_skips_redrive(
    tmp_path: Path,
) -> None:
    # (Task 13b) The ed1 eval must stream partials incrementally (each row is
    # on disk the instant it completes) AND a resumed drive must skip re-drive
    # already-recorded rows -- so a crash/interrupt never loses finished rows.
    from tests.envs.support import execution_policy
    from whetstone.envs.ed1 import ed1_initial_candidate
    from whetstone.envs.ed1_eval import run_ed1_eval
    from whetstone.execution.partials import PartialLog
    from whetstone.optimization.mutation import MUTATION_FIELD

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=4)
    transport = _fake_encdec_transport(tasks)
    exp = build_ed1_experiment(
        tasks=tasks, internal_n=4, official_n=4, repeats=1,
    )
    instances = exp.eval_configs.official.instances
    cand = ed1_initial_candidate()
    template = str(cand.payload[MUTATION_FIELD])
    log = PartialLog(path=tmp_path / "ed1.partial.jsonl")

    first = run_ed1_eval(
        exp, candidate_template=template, candidate_id=cand.candidate_id,
        instances=instances, execution_policy=execution_policy(max_attempts=1),
        transport=transport, repeats=1, apply_reward=False,
        partial_log=log, split_role="official",
    )
    # Every driven row was appended to the partial log incrementally.
    recorded = [
        rec for rec in log.load()
        if rec.phase == "official" and rec.unit == cand.candidate_id
    ]
    assert len(recorded) == len(instances)
    # A record carries the dual payload (compression) for a lossless resume.
    import json as _json

    payloads = [_json.loads(rec.raw_response) for rec in recorded]
    assert all("compression_value" in p for p in payloads)

    # A resumed drive over a transport that RAISES if called proves the rows
    # were restored from disk (no re-drive, no re-pay).
    from tests.envs.support import FakeTransport

    def _boom(_prompt: str) -> str:
        raise AssertionError("resume must not re-drive recorded rows")

    resumed = run_ed1_eval(
        exp, candidate_template=template, candidate_id=cand.candidate_id,
        instances=instances, execution_policy=execution_policy(max_attempts=1),
        transport=FakeTransport(reply=_boom), repeats=1, apply_reward=False,
        partial_log=log, split_role="official",
    )
    # The resumed aggregate matches the first drive (restored losslessly).
    assert (
        resumed.pass_aggregate.aggregation_output.value
        == first.pass_aggregate.aggregation_output.value
    )
    assert (
        resumed.compression_aggregate.aggregation_output.value
        == first.compression_aggregate.aggregation_output.value
    )


# --- Task 14: pilot row-level diagnostics (arm-None explainability) ----------


def test_ed1_row_diags_carry_budget_and_failure_context() -> None:
    # (Task 14a) Every row exposes a diagnostic: the typed failure_code, the
    # per-task MAX_BUDGET, the actual encoder-output length, and the derived
    # over_budget flag -- so an arm-level None is explainable from disk.
    from tests.envs.support import FakeTransport, execution_policy
    from whetstone.envs.ed1_eval import run_ed1_eval

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=1)
    exp = build_ed1_experiment(
        tasks=tasks, internal_n=1, official_n=1, budget_ratio=0.5,
    )
    ht = tasks[0].humaneval_task
    input_chars = len(tasks[0].input_code)
    long_desc = "y" * (input_chars * 4)  # deliberately over budget

    def reply(prompt: str) -> str:
        if prompt.startswith("Provide") or prompt.startswith("Compress"):
            return long_desc
        return ht.ground_truth_code

    template = ed1_initial_candidate().payload[MUTATION_FIELD]
    ed = run_ed1_eval(
        exp, candidate_template=template, candidate_id="ed1-naive",
        instances=exp.eval_configs.internal.instances,
        execution_policy=execution_policy(max_attempts=1),
        transport=FakeTransport(reply=reply), repeats=1, apply_reward=False,
    )
    assert len(ed.row_diags) == 1
    d = ed.row_diags[0]
    # MAX_BUDGET is the per-task rounded ratio of input chars (0.5).
    assert d.max_budget == round(0.5 * input_chars)
    assert d.encoder_len == len(long_desc)
    # Over-budget is FLAGGED (encoder_len > max_budget) but NOT failed -- the
    # budget only steers; the row still scored (canonical -> pass).
    assert d.over_budget is True
    assert d.failed is False
    assert d.failure_code == ""
    assert d.passed == pytest.approx(1.0)
    # The dict form (what lands in pilots/ed1.json) carries all the fields.
    row = d.as_dict()
    assert set(row) >= {
        "instance_id", "repeat", "passed", "compression", "failed",
        "failure_code", "max_budget", "encoder_len", "over_budget",
    }


def test_ed1_row_diag_within_budget_flag_is_false() -> None:
    # (Task 14a/b) The over_budget flag is FALSE for a within-budget encoder
    # output -- the diagnosis rests on distinguishing within- vs over-budget
    # rows, so the flag must track encoder_len vs max_budget both directions.
    from tests.envs.support import FakeTransport, execution_policy
    from whetstone.envs.ed1_eval import run_ed1_eval

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=1)
    exp = build_ed1_experiment(
        tasks=tasks, internal_n=1, official_n=1, budget_ratio=0.5,
    )
    ht = tasks[0].humaneval_task

    def reply(prompt: str) -> str:
        if prompt.startswith("Provide") or prompt.startswith("Compress"):
            return "ok"  # 2 chars, well within any budget
        return ht.ground_truth_code

    template = ed1_initial_candidate().payload[MUTATION_FIELD]
    ed = run_ed1_eval(
        exp, candidate_template=template, candidate_id="ed1-naive",
        instances=exp.eval_configs.internal.instances,
        execution_policy=execution_policy(max_attempts=1),
        transport=FakeTransport(reply=reply), repeats=1, apply_reward=False,
    )
    d = ed.row_diags[0]
    assert d.encoder_len == 2
    assert d.max_budget is not None and d.max_budget >= 2
    assert d.over_budget is False


def test_ed1_pilot_zero_row_arm_is_loud_not_bare_none() -> None:
    # (Task 14b) An arm whose rows all fail must be LOUD: present_rows==0 with
    # a none_reason naming the dominant failure code + count, not a bare None.
    from tests.envs.support import execution_policy
    from whetstone.envs.ed1_scoring import CodeScore
    from whetstone.runner.ed1_pilot import run_ed1_pilot

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=3)
    transport = _fake_encdec_transport(tasks)

    # A scorer that always reports infrastructure-unknown -> every row FAILS
    # (never scored 0), so the pass aggregate is incomplete (arm-level None).
    def _infra_unknown(**_kwargs) -> CodeScore:
        return CodeScore(
            passed=False, infrastructure_unknown=True,
            outcome="HARNESS_FAILURE",
        )

    report = run_ed1_pilot(
        transport=transport, execution_policy=execution_policy(max_attempts=1),
        tasks=3, repeats=1, concurrency=2, scorer=_infra_unknown,
    )
    arm = report.naive
    # The aggregate pass rate is None (all rows failed) -- but the arm is LOUD.
    assert arm.pass_rate is None
    assert arm.present_rows == 0
    assert arm.failed_rows == 3
    assert arm.none_reason is not None
    assert "code_eval_infrastructure_unknown" in arm.none_reason
    assert "0 present rows" in arm.none_reason
    # The per-row records are on the report (persisted to pilots/ed1.json).
    d = arm.as_dict()
    assert d["present_rows"] == 0
    assert d["none_reason"] == arm.none_reason
    rows = list(arm.row_diags)  # already dict records on the summary
    assert len(rows) == 3
    codes = [r["failure_code"] for r in rows]
    assert codes == ["code_eval_infrastructure_unknown"] * 3


# --- Task 15: transport-bound guard sizing + timed-out-row re-drive ---------


def test_ed1_row_guard_budgets_both_wire_calls() -> None:
    # (Task 15a) An ed1 row makes TWO sequential wire calls (encoder THEN
    # decoder), so its runner guard must budget BOTH calls' transport caps --
    # otherwise the guard (sized for one call) trips mid-decoder the moment the
    # encoder used any time, which made eval:ed1:a1 hang to the guard at
    # cap+15s and look like a transport-bound regression.
    from tests.envs.support import execution_policy
    from whetstone.envs.ed1_eval import _ED1_WIRE_CALLS_PER_ROW, _deadline
    from whetstone.execution.call_support import (
        GUARD_MARGIN_SECONDS,
        guard_deadline_seconds,
    )

    policy = execution_policy(max_attempts=1)
    cap = policy.transport_policy.timeout_seconds
    # The ed1 row guard budgets both calls; the QA (1-call) guard budgets one.
    assert _ED1_WIRE_CALLS_PER_ROW == 2
    assert _deadline(policy) == cap * 2 + GUARD_MARGIN_SECONDS
    assert _deadline(policy) > guard_deadline_seconds(policy)
    # The default (QA) guard is unchanged -- byte-identical to before.
    assert guard_deadline_seconds(policy) == cap + GUARD_MARGIN_SECONDS


def test_ed1_transient_encoder_failure_is_redriven_to_success() -> None:
    # (Task 15b) A transient transport failure on an ed1 row must trigger ONE
    # bounded re-drive before it lands as a failed row -- a single flaky
    # observation must NOT fail the whole ed1 arm under the FAIL policy (the
    # eval:ed1:a1 kill). Here each encoder prompt fails transiently once, then
    # the re-drive succeeds, so the arm aggregate computes cleanly (no fails).
    from dataclasses import dataclass, field

    from dr_providers import (
        FailureClass,
        ProviderCallRequest,
        ProviderInvocationEvidence,
        ProviderTransportFailure,
        ProviderTransportPolicy,
        RawHttpRequest,
    )

    from tests.envs.support import (
        _prompt_of,
        _response,
        execution_policy,
        transport_policy,
    )
    from whetstone.envs.ed1_eval import run_ed1_eval

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=3)
    by_entry = {
        t.humaneval_task.entry_point: t.humaneval_task.ground_truth_code
        for t in tasks
    }

    def _reply(prompt: str) -> str:
        if prompt.startswith("Provide") or prompt.startswith("Compress"):
            for ep in by_entry:
                if f"def {ep}(" in prompt:
                    return f"REBUILD:{ep}"
            return "REBUILD:x"
        for ep, gt in by_entry.items():
            if f"REBUILD:{ep}" in prompt:
                return gt
        return "def _x():\n    return None\n"

    @dataclass
    class _TransientOnce:
        policy: ProviderTransportPolicy = field(
            default_factory=transport_policy
        )
        seen: set[str] = field(default_factory=set)
        fail_count: int = 0

        def __call__(
            self, request: ProviderCallRequest
        ) -> ProviderInvocationEvidence:
            prompt = _prompt_of(request)
            raw = RawHttpRequest.build(
                url="https://example.test/v1/chat/completions",
                headers={"content-type": "json"},
                body={"model": "test-model"},
            )
            # Fail ONLY the first ENCODER call of each task transiently (a
            # decoder prompt is derived from the encoder output, so failing it
            # too would exhaust the single re-drive on a 2-call row). The
            # re-drive then sees the encoder succeed and the decoder succeed ->
            # the row recovers, proving the row-level bounded re-drive works.
            is_encoder = prompt.startswith(("Provide", "Compress"))
            if is_encoder and prompt not in self.seen:
                self.seen.add(prompt)
                self.fail_count += 1
                failure = ProviderTransportFailure(
                    failure_class=FailureClass.TRANSIENT,
                    code="transport_error",
                    message="connection reset", retryable=True,
                )
                return ProviderInvocationEvidence.build(
                    request=request, policy=self.policy,
                    raw_request=raw, outcome=failure,
                )
            return ProviderInvocationEvidence.build(
                request=request, policy=self.policy, raw_request=raw,
                outcome=_response(_reply(prompt)),
            )

    exp = build_ed1_experiment(
        tasks=tasks, internal_n=3, official_n=3, budget_ratio=0.25,
    )
    transport = _TransientOnce()
    template = ed1_initial_candidate().payload[MUTATION_FIELD]
    ed = run_ed1_eval(
        exp, candidate_template=template,
        candidate_id="ed1-naive",
        instances=exp.eval_configs.internal.instances,
        execution_policy=execution_policy(max_attempts=1),
        transport=transport, repeats=1, apply_reward=False,
    )
    # Every distinct encoder prompt failed transiently on its first drive...
    assert transport.fail_count > 0
    # ...but the single bounded re-drive recovered them: NO failed rows, and
    # the arm aggregate is present (not None) -- the arm survived the flakes.
    assert all(not d.failed for d in ed.row_diags)
    assert ed.pass_aggregate.aggregation_output.value is not None
    assert ed.pass_aggregate.aggregation_output.value == pytest.approx(1.0)


def test_ed1_pilot_healthy_arm_has_no_none_reason() -> None:
    # The LOUD reason is present ONLY for a 0-row arm: a healthy arm records
    # the per-row diagnostics but none_reason stays None.
    from tests.envs.support import execution_policy
    from whetstone.runner.ed1_pilot import run_ed1_pilot

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=3)
    transport = _fake_encdec_transport(tasks)
    report = run_ed1_pilot(
        transport=transport, execution_policy=execution_policy(max_attempts=1),
        tasks=3, repeats=1, concurrency=2,
    )
    assert report.naive.present_rows == 3
    assert report.naive.none_reason is None
    assert len(report.naive.row_diags) == 3


# --- Task 16: SKIP tolerance for ed1 anchors (identical to c18's lane) -------


def test_ed1_skip_policy_certifies_arm_with_bounded_failed_rows() -> None:
    # (Task 16b) Under a SKIP completeness policy with tolerance, an ed1 arm
    # whose per-row failures stay within the tolerance still CERTIFIES (a
    # present aggregate) -- exactly as c18's deepseek lane does -- instead of
    # the whole arm dying on any failure under PROPAGATE. Here 1 of 4 tasks
    # fails scoring (25%, under a 30% tolerance) and the arm still aggregates.
    from tests.envs.support import execution_policy
    from whetstone.envs.ed1_eval import run_ed1_eval
    from whetstone.envs.ed1_scoring import CodeScore
    from whetstone.envs.sampling import Completeness

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=4)
    transport = _fake_encdec_transport(tasks)
    fail_task = str(tasks[0].instance.id)

    # A scorer that reports infra-unknown for ONE task (a hung-harness row),
    # PASSED for the rest -- the bounded-failure shape the anchor tolerates.
    def _scorer(*, raw_submission, task, **_kw) -> CodeScore:
        if task.entry_point == tasks[0].humaneval_task.entry_point:
            return CodeScore(
                passed=False, infrastructure_unknown=True,
                outcome="TIMED_OUT",
            )
        return CodeScore(
            passed=True, infrastructure_unknown=False, outcome="PASSED",
        )

    exp = build_ed1_experiment(
        tasks=tasks, internal_n=4, official_n=4, budget_ratio=0.5,
        completeness=Completeness.SKIP, max_skip_fraction=0.30,
    )
    template = ed1_initial_candidate().payload[MUTATION_FIELD]
    ed = run_ed1_eval(
        exp, candidate_template=template,
        candidate_id="ed1-naive",
        instances=exp.eval_configs.official.instances,
        execution_policy=execution_policy(max_attempts=1),
        transport=transport, repeats=1, scorer=_scorer, apply_reward=False,
        policy=exp.completeness_policy,
    )
    # The one failing task is dropped under tolerance; the arm CERTIFIES.
    assert ed.pass_aggregate.rows_failed == 1
    assert ed.pass_aggregate.aggregation_output.value is not None
    # The remaining 3 tasks all passed -> the certified pass rate is 1.0.
    assert ed.pass_aggregate.aggregation_output.value == pytest.approx(1.0)
    # Per-task weights are the PLANNED repeat count (QA-identical), so a task
    # with a failed row is not mis-weighted when escalation pools repeats.
    assert all(c == 1 for c in ed.per_task_counts)
    assert len(ed.per_task_counts) == 4
    _ = fail_task  # (the dropped task id, retained for clarity)


def test_ed1_per_task_count_matches_qa_planned_repeats_semantics() -> None:
    # (Task 16b) ed1's per_task_counts is the PLANNED repeat count (len rows),
    # matching the QA lane's _per_task_count (len(completed_rows)) -- NOT the
    # present-only count -- so the paired/pooled bootstrap weights ed1 tasks
    # identically to c18 even when some repeats failed.
    from tests.envs.support import execution_policy
    from whetstone.envs.ed1_eval import run_ed1_eval
    from whetstone.envs.ed1_scoring import CodeScore
    from whetstone.envs.sampling import Completeness

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=1)
    transport = _fake_encdec_transport(tasks)
    calls = {"n": 0}

    # Fail the FIRST repeat of the single task, pass the second (both planned).
    def _scorer(*, raw_submission, task, **_kw) -> CodeScore:
        calls["n"] += 1
        if calls["n"] == 1:
            return CodeScore(
                passed=False, infrastructure_unknown=True, outcome="TIMED_OUT",
            )
        return CodeScore(
            passed=True, infrastructure_unknown=False, outcome="PASSED",
        )

    exp = build_ed1_experiment(
        tasks=tasks, internal_n=1, official_n=1, budget_ratio=0.5,
        completeness=Completeness.SKIP, max_skip_fraction=0.60,
    )
    template = ed1_initial_candidate().payload[MUTATION_FIELD]
    ed = run_ed1_eval(
        exp, candidate_template=template,
        candidate_id="ed1-naive",
        instances=exp.eval_configs.official.instances,
        execution_policy=execution_policy(max_attempts=1),
        transport=transport, repeats=2, scorer=_scorer, apply_reward=False,
        policy=exp.completeness_policy,
    )
    # The weight is the PLANNED 2 repeats, not the 1 present row.
    assert ed.per_task_counts == (2,)
