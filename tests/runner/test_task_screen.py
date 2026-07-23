"""Tests for the ed1 task-informativeness screen + pool filter (task 17 P2).

No network, no Docker: fake transports + the offline HumanEval snapshot + the
LOCAL subprocess scorer.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.envs.support import FakeTransport, execution_policy
from whetstone.envs.ed1 import build_ed1_experiment, load_ed1_tasks
from whetstone.runner.task_screen import (
    DIRECT_ARMS,
    NAME_ONLY_WRAPPER,
    SCREEN_ARMS,
    SCREEN_SCHEMA,
    cross_model_summary,
    load_exclusion_ids,
    run_task_screen,
    split_prompt,
)


def _all_pass_reply(tasks, rename_token: str = "target_fxn"):
    # A transport that PASSES every arm, INCLUDING renamed arms. It keys the
    # task by a UNIQUE docstring fingerprint (present in every direct prompt +
    # carried through the encoder->REBUILD tag), then returns the canonical
    # solution renamed to whatever function name that arm scores against.
    from whetstone.runner.task_screen import (
        rename_identifier,
        split_prompt,
    )

    # docstring fingerprint -> (entry_point, gt): the docstring is unique per
    # task and appears in the DIRECT prompts (original/docstring/signature).
    fp: dict[str, tuple[str, str]] = {}
    # code fingerprint -> (entry_point, gt): the stripped-code body (after the
    # signature) is unique per task and survives renaming -- keys the ENCODER
    # probe + the renamed direct prompt.
    code_fp: dict[str, tuple[str, str]] = {}
    for t in tasks:
        ht = t.humaneval_task
        parts = split_prompt(ht.prompt, ht.entry_point)
        fp[parts.docstring[:40]] = (ht.entry_point, ht.ground_truth_code)
        # Register the code body under BOTH the canonical name and the renamed
        # form, so the fingerprint matches whether the input was renamed.
        for code in (
            t.input_code,
            rename_identifier(t.input_code, ht.entry_point, rename_token),
        ):
            body = code.split("):", 1)[-1].strip()[:40]
            if body:
                code_fp[body] = (ht.entry_point, ht.ground_truth_code)

    def _match(prompt: str) -> tuple[str, str] | None:
        for key, val in fp.items():
            if key and key in prompt:
                return val
        for key, val in code_fp.items():
            if key and key in prompt:
                return val
        return None

    def _renamed_gt(ep: str, gt: str, prompt: str) -> str:
        # If the prompt was renamed (canonical name gone, token present),
        # return the gt under the token; else under the canonical name.
        if rename_token in prompt and ep not in prompt:
            return rename_identifier(gt, ep, rename_token)
        return gt

    def reply(prompt: str) -> str:
        # Encoder probe -> REBUILD tag. The encoder input is the STRIPPED code
        # (no docstring); key on the unique code body (survives renaming) and
        # flag whether the input was renamed (the entry point def is gone).
        if prompt.startswith("Provide") or prompt.startswith("Compress"):
            match = _match(prompt)
            if match is None:
                return "REBUILD::x"
            ep, _gt = match
            renamed = "1" if f"def {rename_token}(" in prompt else "0"
            return f"REBUILD:{ep}:{renamed}"
        # Decoder REBUILD -> the canonical gt (renamed iff flagged).
        for t in tasks:
            ht = t.humaneval_task
            gt = ht.ground_truth_code
            if f"REBUILD:{ht.entry_point}:1" in prompt:
                return rename_identifier(gt, ht.entry_point, rename_token)
            if f"REBUILD:{ht.entry_point}:0" in prompt:
                return gt
        # DIRECT arms: match by docstring fingerprint, rename as the arm needs.
        match = _match(prompt)
        if match is not None:
            ep, gt = match
            return _renamed_gt(ep, gt, prompt)
        # name-only arm: match by the bare name in the wrapper.
        for t in tasks:
            ht = t.humaneval_task
            if ht.entry_point in prompt:
                return ht.ground_truth_code
        return tasks[0].humaneval_task.ground_truth_code

    return FakeTransport(reply=reply)


# --- prompt splitting (the four direct arms) ---------------------------------


def test_split_prompt_yields_four_clean_arms() -> None:
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=1)
    ht = tasks[0].humaneval_task
    parts = split_prompt(ht.prompt, ht.entry_point)
    # original is the verbatim prompt.
    assert parts.original == ht.prompt
    # signature includes the def line, NOT the docstring body.
    assert f"def {ht.entry_point}(" in parts.signature
    assert '"""' not in parts.signature
    # docstring is nonempty and free of the def line.
    assert parts.docstring
    assert f"def {ht.entry_point}(" not in parts.docstring
    # name-only is the recorded neutral wrapper around the bare name.
    assert parts.name_only == NAME_ONLY_WRAPPER.format(name=ht.entry_point)
    assert ht.entry_point in parts.name_only


def test_split_prompt_handles_multiline_signature() -> None:
    prompt = (
        "from typing import List\n\n"
        "def f(\n    a: int,\n    b: int,\n) -> int:\n"
        '    """ Add two ints.\n    >>> f(1, 2)\n    3\n    """\n'
    )
    parts = split_prompt(prompt, "f")
    # The whole multi-line signature is kept, ending at ') -> int:'.
    assert parts.signature.rstrip().endswith("-> int:")
    assert "b: int," in parts.signature
    assert '"""' not in parts.signature
    assert "Add two ints." in parts.docstring


# --- screen output schema ----------------------------------------------------


def test_screen_report_schema_and_verdicts(tmp_path: Path) -> None:
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=3)
    report = run_task_screen(
        model="qwen/qwen3-coder-flash",
        transport=_all_pass_reply(tasks),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=2, concurrency=4,
    )
    # Every task ran all seven arms; all pass -> always_pass -> excluded.
    assert len(report.rows) == 3
    for row in report.rows:
        assert set(row.pass_counts) == set(SCREEN_ARMS)
        assert all(row.pass_counts[a] == 2 for a in SCREEN_ARMS)
        assert row.always_pass is True
    assert len(report.excluded_task_ids) == 3
    # The written artifact carries the FULL config identity + summaries.
    path = report.write(tmp_path)
    assert path.name == "ed1_qwen_qwen3_coder_flash.json"
    data = json.loads(path.read_text())
    assert data["schema"] == SCREEN_SCHEMA
    assert data["arms"] == list(SCREEN_ARMS)
    assert data["name_only_wrapper"] == NAME_ONLY_WRAPPER
    assert data["rename_token"] == "target_fxn"
    assert data["dataset_revision"]  # the pinned revision string
    assert data["excluded_count"] == 3
    assert "science_intent" in data
    assert set(data["arm_summary"]) == set(SCREEN_ARMS)
    for arm in SCREEN_ARMS:
        assert data["arm_summary"][arm]["mean_pass_rate"] == 1.0
        assert data["arm_summary"][arm]["tasks_full_pass"] == 3
    # The canonical-minus-renamed deltas present (all pass -> delta 0 here).
    assert "direct_original_minus_direct_renamed" in data["rename_deltas"]
    assert "encdec_naive_minus_encdec_renamed" in data["rename_deltas"]
    for d in data["rename_deltas"].values():
        assert d["delta"] == 0.0  # canonical == renamed under all-pass fake


def test_screen_informative_task_is_not_excluded(tmp_path: Path) -> None:
    # A task the model gets WRONG on some arm is informative -> NOT excluded.
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=2)
    by_entry = {
        t.humaneval_task.entry_point: t.humaneval_task.ground_truth_code
        for t in tasks
    }
    wrong_entry = tasks[0].humaneval_task.entry_point

    def reply(prompt: str) -> str:
        if prompt.startswith("Provide") or prompt.startswith("Compress"):
            for ep in by_entry:
                if f"def {ep}(" in prompt:
                    return f"REBUILD:{ep}"
            return "REBUILD:x"
        for ep, gt in by_entry.items():
            if f"REBUILD:{ep}" in prompt:
                return gt
        # DIRECT: task[0] gets a WRONG body (informative); task[1] gets gt.
        for ep, gt in by_entry.items():
            if ep in prompt:
                if ep == wrong_entry:
                    return f"def {ep}(*a, **k):\n    return None\n"
                return gt
        return next(iter(by_entry.values()))

    report = run_task_screen(
        model="qwen/qwen3-coder-flash",
        transport=FakeTransport(reply=reply),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=2, concurrency=4,
    )
    rows = {r.task_id: r for r in report.rows}
    wrong_id = str(tasks[0].instance.id)
    # The task with a wrong direct arm is informative -> not always_pass.
    assert rows[wrong_id].always_pass is False
    assert wrong_id not in report.excluded_task_ids
    # Its direct arms failed (0/2) but encdec still passed -> mixed => keep.
    assert rows[wrong_id].pass_counts["direct_original"] == 0


def test_cross_model_summary_table(tmp_path: Path) -> None:
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=2)
    reports = [
        run_task_screen(
            model=m, transport=_all_pass_reply(tasks),
            execution_policy=execution_policy(max_attempts=1),
            tasks=tasks, repeats=1, concurrency=2,
        )
        for m in ("qwen/qwen3-coder-flash", "deepseek/deepseek-v4-flash")
    ]
    table = cross_model_summary(reports)
    assert table["arms"] == list(SCREEN_ARMS)
    # One row per (model, arm).
    rows = table["table"]
    assert isinstance(rows, list)
    assert len(rows) == 2 * len(SCREEN_ARMS)
    expected_models = {"qwen/qwen3-coder-flash", "deepseek/deepseek-v4-flash"}
    models: set[str] = set()
    for r in rows:
        assert isinstance(r, dict)
        models.add(str(r.get("model")))
    assert models == expected_models
    per_model = table["per_model_excluded"]
    assert isinstance(per_model, dict)
    assert set(per_model) == expected_models
    # The cross-model rename deltas (paper figure) are present per model.
    deltas_by_model = table["rename_deltas_by_model"]
    assert isinstance(deltas_by_model, dict)
    assert set(deltas_by_model) == expected_models


# --- pool filter: exclusion folds into split identity ------------------------


def test_pool_filter_excludes_tasks_and_changes_identity() -> None:
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=6)
    drop = frozenset({str(tasks[0].instance.id), str(tasks[1].instance.id)})

    full = build_ed1_experiment(
        tasks=tasks, internal_n=3, official_n=3, budget_ratio=0.25,
    )
    filtered = build_ed1_experiment(
        tasks=tasks, internal_n=3, official_n=3, budget_ratio=0.25,
        exclude_task_ids=drop,
    )
    full_ids = {
        str(i.id) for i in full.eval_configs.internal.instances
    } | {str(i.id) for i in full.eval_configs.official.instances}
    filt_ids = {
        str(i.id) for i in filtered.eval_configs.internal.instances
    } | {str(i.id) for i in filtered.eval_configs.official.instances}
    # The dropped tasks are gone from BOTH splits.
    assert drop & filt_ids == set()
    assert drop <= full_ids
    # The filter FOLDS into identity: each split's eval_config_hash differs.
    full_ihash = (
        full.eval_configs.internal.eval_config.config_identity_hash
    )
    filt_ihash = (
        filtered.eval_configs.internal.eval_config.config_identity_hash
    )
    assert full_ihash != filt_ihash
    full_ohash = (
        full.eval_configs.official.eval_config.config_identity_hash
    )
    filt_ohash = (
        filtered.eval_configs.official.eval_config.config_identity_hash
    )
    assert full_ohash != filt_ohash


def test_load_exclusion_ids_from_screen_file(tmp_path: Path) -> None:
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=2)
    report = run_task_screen(
        model="qwen/qwen3-coder-flash",
        transport=_all_pass_reply(tasks),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=1, concurrency=2,
    )
    path = report.write(tmp_path)
    ids = load_exclusion_ids(path)
    assert ids == frozenset(str(t.instance.id) for t in tasks)


def test_screen_arms_are_five_direct_plus_two_encdec() -> None:
    assert DIRECT_ARMS == (
        "direct_original", "direct_docstring",
        "direct_signature", "direct_name", "direct_renamed",
    )
    assert SCREEN_ARMS[-2:] == ("encdec_naive", "encdec_renamed")
    assert len(SCREEN_ARMS) == 7


# --- amendment 2: rename ablation (rename + scoring trap) ------------------


def test_rename_identifier_covers_signature_and_doctests() -> None:
    # The rename MUST cover every occurrence -- the signature AND doctest lines
    # (">>> old(...)") -- else a leaked canonical name voids the ablation.
    from whetstone.runner.task_screen import rename_identifier

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=1)
    ht = tasks[0].humaneval_task
    ep = ht.entry_point
    assert ht.prompt.count(ep) >= 2  # signature + at least one doctest
    renamed = rename_identifier(ht.prompt, ep, "target_fxn")
    # No canonical-name occurrence survives; the token appears the same count.
    assert ep not in renamed
    assert renamed.count("target_fxn") == ht.prompt.count(ep)
    # Whole-identifier only: a substring like "truncate_numbers" is untouched.
    assert rename_identifier("foo_bar and foobar", "foo_bar", "X") == (
        "X and foobar"
    )


def test_renamed_task_scores_against_renamed_entry_point() -> None:
    # (Amendment-2 trap b) A renamed submission scores against the RENAMED
    # entry point; the original-named canonical solution passes ONLY when
    # renamed to match.
    from whetstone.envs.ed1_scoring import score_ed1_submission
    from whetstone.runner.task_screen import (
        rename_identifier,
        renamed_task,
    )

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=1)
    ht = tasks[0].humaneval_task
    rt = renamed_task(ht, old=ht.entry_point, new="target_fxn")
    assert rt.entry_point == "target_fxn"
    # The gt renamed to the token passes the renamed task.
    gt_renamed = rename_identifier(
        ht.ground_truth_code, ht.entry_point, "target_fxn"
    )
    good = score_ed1_submission(
        raw_submission=gt_renamed, task=rt, timeout_seconds=30.0
    )
    assert good.passed is True


def test_rename_deltas_capture_canonical_minus_renamed(tmp_path: Path) -> None:
    # (Amendment-2 paper figure) The rename delta = canonical mean pass MINUS
    # renamed mean pass. Here the renamed arms FAIL (the fake only solves the
    # canonical name), so the delta is positive -> a memorization signal.
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=2)
    by_entry = {
        t.humaneval_task.entry_point: t.humaneval_task.ground_truth_code
        for t in tasks
    }

    def reply(prompt: str) -> str:
        # Encoder probe -> REBUILD only for the CANONICAL-named input; the
        # renamed input (def target_fxn) gets a non-reconstructable tag.
        if prompt.startswith("Provide") or prompt.startswith("Compress"):
            for ep in by_entry:
                if f"def {ep}(" in prompt:
                    return f"REBUILD:{ep}"
            return "REBUILD:renamed-unsolved"
        for ep, gt in by_entry.items():
            if f"REBUILD:{ep}" in prompt:
                return gt
        # DIRECT: solve ONLY when the canonical name is present (not renamed).
        for ep, gt in by_entry.items():
            if ep in prompt:
                return gt
        return "def _wrong():\n    return None\n"

    report = run_task_screen(
        model="deepseek/deepseek-v4-flash",
        transport=FakeTransport(reply=reply),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=2, concurrency=4,
    )
    deltas = report.rename_deltas()
    # Canonical arms pass, renamed arms fail -> positive delta on both pairs.
    assert deltas["direct_original_minus_direct_renamed"]["delta"] > 0.0
    assert deltas["encdec_naive_minus_encdec_renamed"]["delta"] > 0.0
    # The artifact carries the deltas for the paper.
    data = json.loads(report.write(tmp_path).read_text())
    assert data["rename_deltas"][
        "direct_original_minus_direct_renamed"
    ]["delta"] > 0.0


def test_screen_variants_subset_runs_only_selected_arms() -> None:
    # (Addendum) --variants selects a subset; only those arms are driven +
    # judged, and always_pass is evaluated over the measured arms.
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=2)
    report = run_task_screen(
        model="qwen/qwen3-coder-flash",
        transport=_all_pass_reply(tasks),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=1, concurrency=2,
        variants=("direct_name", "direct_renamed"),
    )
    assert report.arms == ("direct_name", "direct_renamed")
    for row in report.rows:
        assert set(row.pass_counts) == {"direct_name", "direct_renamed"}
    assert set(report.arm_summary()) == {"direct_name", "direct_renamed"}


def test_screen_resumes_from_partials_without_repaying() -> None:
    # (Addendum) A resumed screen restores recorded rows from the partial log
    # instead of re-driving -- a transport that RAISES proves no re-pay.
    from whetstone.execution.partials import PartialLog

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=2)
    log = PartialLog(path=Path("/tmp") / "screen_resume_test.partial.jsonl")
    if log.path.exists():
        log.path.unlink()
    first = run_task_screen(
        model="qwen/qwen3-coder-flash",
        transport=_all_pass_reply(tasks),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=1, concurrency=2, partial_log=log,
    )

    def _boom(_prompt: str) -> str:
        raise AssertionError("resume must not re-drive recorded rows")

    resumed = run_task_screen(
        model="qwen/qwen3-coder-flash",
        transport=FakeTransport(reply=_boom),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=1, concurrency=2, partial_log=log,
    )
    log.path.unlink()
    # The resumed report reproduces the first run's per-arm pass counts.
    first_counts = {r.task_id: r.pass_counts for r in first.rows}
    resumed_counts = {r.task_id: r.pass_counts for r in resumed.rows}
    assert resumed_counts == first_counts
