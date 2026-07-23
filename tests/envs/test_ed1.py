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
