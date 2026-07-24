"""Tests for the d1 direct-optimization precursor env (task 23).

d1 is DIRECT generation (single LLM call) over a FROZEN input arm (reusing the
screen's direct-arm construction, incl. the renamed-canonical all-occurrence
scrub) with a MUTABLE surrounding wrapper (frame/body Mutation Surface) and a
PLAIN pass-rate reward (NOT blended). Every test drives a fake transport + a
local (no-container) code scorer -- no live paid call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whetstone.envs.d1 import (
    D1_CANONICAL_MODEL,
    D1_ENV_NAME,
    D1_INPUT_ARMS,
    D1_WRAPPER_BODY_CEILING,
    D1_WRAPPER_BODY_NAIVE,
    build_d1_experiment,
    d1_ceiling_candidate,
    d1_initial_candidate,
    render_d1_frame,
)
from whetstone.envs.ed1 import ED1_DATASET_REVISION, load_ed1_tasks
from whetstone.optimization.mutation import MUTATION_FIELD


def _tasks(limit: int = 3):
    return load_ed1_tasks(prefer_snapshot=True, limit=limit)


# --- Identity: the frozen input arm folds into graph + eval config ----------


def test_each_input_arm_yields_distinct_graph_and_eval_hash() -> None:
    tasks = _tasks(4)
    graphs: set[str] = set()
    evals: set[str] = set()
    for arm in D1_INPUT_ARMS:
        exp = build_d1_experiment(input_arm=arm, tasks=tasks)
        graphs.add(exp.rollout_definition.graph_hash)
        evals.add(
            exp.eval_configs.official.eval_config.config_identity_hash
        )
    # Five arms -> five DISTINCT graph hashes AND five distinct eval hashes:
    # the frozen arm is an output-affecting knob, folded into both identities.
    assert len(graphs) == len(D1_INPUT_ARMS)
    assert len(evals) == len(D1_INPUT_ARMS)


def test_task_model_folds_into_graph_hash() -> None:
    tasks = _tasks(3)
    a = build_d1_experiment(input_arm="original", model="m/a", tasks=tasks)
    b = build_d1_experiment(input_arm="original", model="m/b", tasks=tasks)
    assert a.rollout_definition.graph_hash != b.rollout_definition.graph_hash


def test_d1_graph_is_two_nodes_terminal_eval() -> None:
    from whetstone.envs.d1 import d1_graph_definition

    definition = d1_graph_definition()
    assert len(definition.nodes) == 2
    assert definition.terminal_node_id == "evaluate"


def test_dataset_revision_pinned_and_recorded() -> None:
    exp = build_d1_experiment(tasks=_tasks(3))
    assert exp.dataset_revision == ED1_DATASET_REVISION
    assert ED1_DATASET_REVISION


# --- The naive probe IS the screen wrapper (reproduces screen numbers) ------


def test_naive_prompt_is_byte_identical_to_the_screen_direct_prompt() -> None:
    """The load-bearing science claim: a d1 naive (eval-anchor) prompt on an
    arm is BYTE-IDENTICAL to the screen's ``_direct_prompt`` on that arm, so a
    d1 eval anchor reproduces the corresponding screen arm's pass numbers."""
    from whetstone.envs.d1_eval import _input_arm_text
    from whetstone.runner.task_screen import _direct_prompt, split_prompt

    tasks = _tasks(3)
    for arm in D1_INPUT_ARMS:
        exp = build_d1_experiment(input_arm=arm, tasks=tasks)
        for inst in exp.eval_configs.internal.instances:
            body, _ = _input_arm_text(exp, inst)
            d1_prompt = render_d1_frame(
                D1_WRAPPER_BODY_NAIVE, input_arm=body
            )
            ht = exp.humaneval_for(inst)
            parts = split_prompt(ht.prompt, ht.entry_point)
            screen_prompt = _direct_prompt(
                f"direct_{arm}", parts, rename_token=exp.rename_token
            )
            assert d1_prompt == screen_prompt


def test_naive_and_ceiling_bodies_are_distinct() -> None:
    naive = d1_initial_candidate()
    ceiling = d1_ceiling_candidate()
    assert naive.payload[MUTATION_FIELD] == D1_WRAPPER_BODY_NAIVE
    assert ceiling.payload[MUTATION_FIELD] == D1_WRAPPER_BODY_CEILING
    assert naive.payload[MUTATION_FIELD] != ceiling.payload[MUTATION_FIELD]


# --- The renamed arm scrubs the canonical name + scores against renamed -----


def test_renamed_arm_scrubs_canonical_name_and_scores_renamed() -> None:
    """The 'renamed' arm's rendered prompt carries the renamed entry point
    (the canonical name scrubbed everywhere) and its scoring task is the
    RENAMED task -- the amendment-2 causal ablation + scoring trap."""
    from whetstone.envs.d1_eval import _input_arm_text

    exp = build_d1_experiment(input_arm="renamed", tasks=_tasks(3))
    inst = exp.eval_configs.internal.instances[0]
    body, score_task = _input_arm_text(exp, inst)
    ht = exp.humaneval_for(inst)
    # The canonical entry point is GONE from the input arm; the rename token is
    # present; the scoring task's entry point is the renamed token.
    assert f"def {ht.entry_point}(" not in body
    assert exp.rename_token in body
    assert score_task.entry_point == exp.rename_token


# --- Pass-only reward (NOT blended); d1 excluded from the blend guard rail ---


def test_d1_reward_policy_is_pass_only_single_term() -> None:
    exp = build_d1_experiment(tasks=_tasks(3))
    policy = exp.reward_policy
    assert len(policy.terms) == 1
    assert policy.terms[0].name == "binary_test_pass"


def test_d1_optimizer_cell_is_not_subject_to_blend_guard(
    tmp_path: Path,
) -> None:
    """An ed1/ed1m optimizer cell REFUSES pass-only (task 22 guard); a d1
    optimizer cell does NOT -- d1 is pass-only by design, so a copro d1 cell
    builds without a blend config."""
    from tests.envs.support import execution_policy
    from tests.runner.support import proposer_config
    from whetstone.optimization.proposer import FakeProposerTransport
    from whetstone.runner.cell import CellConfig, _build_experiment
    from whetstone.runner.execution_mode import ExecutionMode

    cfg = CellConfig(
        optimizer="copro", env=D1_ENV_NAME, lane="openrouter", attempt=0,
        task_model=D1_CANONICAL_MODEL, proposer_model="none", canonical=False,
        proposer_config=proposer_config(),
        proposer_transport=FakeProposerTransport(script={}, default=()),
        rollout_transport=_fake_direct_transport(_tasks(3)),
        execution_policy=execution_policy(max_attempts=1),
        repeats=1, official_repeats=1,
        execution_mode=ExecutionMode.IN_PROCESS,
        ed1_task_limit=3, ed1_blend_config=None,  # NO blend -> must be allowed
        d1_input_arm="original",
    )
    exp = _build_experiment(cfg)  # does not raise
    assert exp.env_name == D1_ENV_NAME


# --- Full fake-transport direct cell end-to-end -----------------------------


def _fake_direct_transport(tasks):
    """A scripted direct transport: match each task on its docstring/name,
    reply the (possibly renamed) canonical solution so the sandbox scores PASS.
    """
    from tests.envs.support import FakeTransport
    from whetstone.runner.task_screen import rename_identifier, split_prompt

    def reply(prompt: str) -> str:
        for t in tasks:
            ht = t.humaneval_task
            ep = ht.entry_point
            parts = split_prompt(ht.prompt, ep)
            marker = (
                parts.docstring.splitlines()[0].strip()
                if parts.docstring else None
            )
            if (marker and marker in prompt) or f"def {ep}(" in prompt or (
                f"named {ep}." in prompt
            ):
                gt = ht.ground_truth_code
                # If the prompt renamed the entry point, rename the solution.
                if f"def {ep}(" not in prompt and "target_fxn" in prompt:
                    return rename_identifier(gt, ep, "target_fxn")
                return gt
        return "def _x():\n    return None\n"

    return FakeTransport(reply=reply)


@pytest.mark.parametrize("arm", ["original", "docstring", "renamed"])
def test_d1_cell_end_to_end_records_pass_rate(
    tmp_path: Path, arm: str
) -> None:
    from tests.envs.support import execution_policy
    from tests.runner.support import credits_fetcher, proposer_config
    from whetstone.optimization.proposer import FakeProposerTransport
    from whetstone.runner.cell import CellConfig, run_cell
    from whetstone.runner.execution_mode import ExecutionMode
    from whetstone.runner.ledger import Ledger

    tasks = _tasks(3)
    cfg = CellConfig(
        optimizer="eval", env=D1_ENV_NAME, lane="openrouter", attempt=0,
        task_model=D1_CANONICAL_MODEL, proposer_model="none", canonical=True,
        proposer_config=proposer_config(),
        proposer_transport=FakeProposerTransport(script={}, default=()),
        rollout_transport=_fake_direct_transport(tasks),
        execution_policy=execution_policy(max_attempts=1),
        repeats=1, official_repeats=1,
        execution_mode=ExecutionMode.IN_PROCESS,
        ed1_task_limit=3, d1_input_arm=arm,
    )
    ledger = Ledger(root=tmp_path)
    outcome = run_cell(
        cfg, ledger=ledger,
        credits_fetcher=credits_fetcher([(710.0, 616.0), (710.0, 616.5)]),
    )
    r = outcome.record
    # Canonical solutions pass -> pass rate 1.0. d1 records NO dual/compression
    # scores (pass-only, single objective).
    assert r.best_official == pytest.approx(1.0)
    assert r.dual_scores is None
    # The recorded task-model folds the deepseek default; the arm folds the
    # graph identity (a distinct arm is a distinct graph_hash, checked above).
    assert r.models.task == D1_CANONICAL_MODEL


# --- Intake: the d1 wrapper body validates like ed1 (no placeholders/fence) --


def test_d1_intake_rejects_body_with_placeholder_typed() -> None:
    from tests.envs.support import execution_policy
    from tests.runner.support import proposer_config
    from whetstone.envs.d1_eval import run_d1_eval  # noqa: F401 (import seam)
    from whetstone.envs.ed1 import ED1_INVALID_BODY
    from whetstone.optimization.proposer import FakeProposerTransport
    from whetstone.runner.optimizers import run_optimize

    tasks = _tasks(3)
    exp = build_d1_experiment(tasks=tasks, internal_n=2, official_n=1)
    # COPRO drafts one INVALID body (a {placeholder} the frame owns) + one
    # VALID wrapper body. The invalid one is a TYPED intake rejection
    # (ED1_INVALID_BODY), spending no eval; the clean one is accepted.
    result = run_optimize(
        exp, optimizer="copro",
        proposer_transport=FakeProposerTransport(
            script={},
            default=("Solve {input_arm} now.", "Solve it carefully."),
        ),
        proposer_config=proposer_config(),
        rollout_transport=_fake_direct_transport(tasks),
        execution_policy=execution_policy(max_attempts=1),
        internal_instances=exp.eval_configs.internal.instances,
        repeats=1,
    )
    rejected = [
        s for s in result.steps if s.rejected_reason == ED1_INVALID_BODY
    ]
    assert rejected, "the placeholder body must be a typed intake rejection"
    assert any("{input_arm}" in s.rejected_fields for s in rejected)
    for s in rejected:
        assert s.internal_score is None and s.evaluation is None


# --- Resume: a recorded d1 row is restored, not re-driven -------------------


def test_d1_eval_resume_skips_recorded_rows(tmp_path: Path) -> None:
    from tests.envs.support import execution_policy
    from whetstone.envs.d1_eval import run_d1_eval
    from whetstone.execution.partials import PartialLog

    tasks = _tasks(2)
    exp = build_d1_experiment(input_arm="original", tasks=tasks)
    instances = exp.eval_configs.official.instances
    log = PartialLog(path=tmp_path / "d1.partial.jsonl")

    class _Counting:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def __call__(self, request):
            self.calls += 1
            return self.inner(request)

    inner = _fake_direct_transport(tasks)
    counting = _Counting(inner)
    # First drive records every row.
    run_d1_eval(
        exp, candidate_body=D1_WRAPPER_BODY_NAIVE, candidate_id="d1-naive",
        instances=instances, execution_policy=execution_policy(max_attempts=1),
        transport=counting, repeats=2, apply_reward=False, partial_log=log,
        split_role="official",
    )
    first = counting.calls
    assert first == len(instances) * 2
    # A resumed drive over the same log re-drives ZERO rows.
    run_d1_eval(
        exp, candidate_body=D1_WRAPPER_BODY_NAIVE, candidate_id="d1-naive",
        instances=instances, execution_policy=execution_policy(max_attempts=1),
        transport=counting, repeats=2, apply_reward=False, partial_log=log,
        split_role="official",
    )
    assert counting.calls == first  # no new calls


# --- CLI + dry-run wiring ----------------------------------------------------


def test_d1_cell_subcommand_exposes_input_arm() -> None:
    from whetstone.runner.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["cell", "--optimizer", "eval", "--env", "d1",
         "--input-arm", "renamed"]
    )
    assert args.input_arm == "renamed"


def test_d1_dry_cell_runs_end_to_end(tmp_path: Path) -> None:
    from whetstone.runner.dryrun import run_dry_cell

    outcome = run_dry_cell(
        env="d1", optimizer="eval", root=tmp_path, input_arm="renamed"
    )
    r = outcome.record
    assert r.cell_id == "eval:d1:a0"
    assert r.baseline_official == pytest.approx(1.0)
    assert r.dual_scores is None  # d1 is pass-only
