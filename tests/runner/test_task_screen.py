"""Tests for the ed1 task-informativeness screen + pool filter (task 17 P2).

No network, no Docker: fake transports + the offline HumanEval snapshot + the
LOCAL subprocess scorer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    # Filename folds the budget ratio (r=0.25 -> r025) so the two-ratio encdec
    # re-run never overwrites the compression screen.
    path = report.write(tmp_path)
    assert path.name == "ed1_qwen_qwen3_coder_flash_r025.json"
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
    # Table rows carry the task-20 telemetry columns + the config keys.
    for r in rows:
        assert isinstance(r, dict)
        assert "budget_ratio" in r
        assert "reasoning_effort" in r
        assert "config_key" in r
        assert "mean_latency_s" in r
        assert "latency_coverage" in r
        assert "total_reasoning_tokens" in r
    per_config = table["per_config_excluded"]
    assert isinstance(per_config, dict)
    # Keyed per (model, ratio, effort); both are default-effort r=0.25 ->
    # model@r025 (no effort suffix on the default).
    expected_keys = {f"{m}@r025" for m in expected_models}
    assert set(per_config) == expected_keys
    # The cross-model rename deltas (paper figure) present per config.
    deltas = table["rename_deltas_by_config"]
    assert isinstance(deltas, dict)
    assert set(deltas) == expected_keys
    # Default-effort reports have no reasoning-honored entries (no labels).
    assert table["reasoning_honored"] == {}


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


# --- Task 19 ext: per-(model, ratio) filenames + phase isolation -------------


def test_screen_stem_folds_ratio_into_filename() -> None:
    from whetstone.runner.task_screen import ratio_tag, screen_stem

    assert ratio_tag(0.25) == "r025"
    assert ratio_tag(1.0) == "r100"
    assert screen_stem("qwen/q3", 0.25) == "ed1_qwen_q3_r025"
    assert screen_stem("qwen/q3", 1.0) == "ed1_qwen_q3_r100"


def test_two_ratio_screens_write_distinct_files(tmp_path: Path) -> None:
    # (Task 19 ext) A r=0.25 full screen and a r=1.0 encdec-only re-run write
    # DISTINCT files -- the fair-channel run never overwrites the compression
    # screen.
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=2)
    full = run_task_screen(
        model="qwen/qwen3-coder-flash", transport=_all_pass_reply(tasks),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=1, concurrency=2, budget_ratio=0.25,
    )
    encdec = run_task_screen(
        model="qwen/qwen3-coder-flash", transport=_all_pass_reply(tasks),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=1, concurrency=2, budget_ratio=1.0,
        variants=("encdec_naive", "encdec_renamed"),
    )
    p_full = full.write(tmp_path)
    p_enc = encdec.write(tmp_path)
    assert p_full.name.endswith("_r025.json")
    assert p_enc.name.endswith("_r100.json")
    assert p_full != p_enc and p_full.exists() and p_enc.exists()
    # The encdec-only artifact has just the 2 encdec arms.
    data = json.loads(p_enc.read_text())
    assert data["arms"] == ["encdec_naive", "encdec_renamed"]
    assert data["budget_ratio"] == 1.0


def test_ratio_phase_isolation_no_cross_ratio_resume(tmp_path: Path) -> None:
    # (Task 19 ext) A r=1.0 encdec run must NOT restore-skip against r=0.25
    # encdec partials -- a different ratio is a different rollout, re-driven.
    from whetstone.execution.partials import PartialLog

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=2)
    log = PartialLog(path=tmp_path / "screen.partial.jsonl")
    # First: r=0.25 encdec only, records partials under the r025 phase.
    run_task_screen(
        model="qwen/qwen3-coder-flash", transport=_all_pass_reply(tasks),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=1, concurrency=2, budget_ratio=0.25,
        variants=("encdec_naive",), partial_log=log,
    )
    # Now r=1.0 encdec with a transport that RAISES if called: if the r=0.25
    # rows were wrongly restored, no call happens; but they must NOT be, so the
    # transport IS called -> the raise proves the rows were re-driven.
    calls = {"n": 0}

    def _reply(_prompt: str) -> str:
        calls["n"] += 1
        return "def x():\n    return None\n"

    run_task_screen(
        model="qwen/qwen3-coder-flash", transport=FakeTransport(reply=_reply),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=1, concurrency=2, budget_ratio=1.0,
        variants=("encdec_naive",), partial_log=log,
    )
    assert calls["n"] > 0, "r=1.0 must re-drive, not resume r=0.25 rows"


# --- Task 20: reasoning-token + latency telemetry (coverage-honest) ----------


def _telemetry_transport(tasks, *, reasoning: int | None):
    # A transport that returns the canonical solution AND a usage block
    # (prompt/completion/total + optional reasoning_tokens) so call_telemetry
    # reads tokens; latency comes from the driver's real monotonic clock.
    from dr_providers import (
        ProviderCallRequest,
        ProviderInvocationEvidence,
        ProviderTransportResponse,
        RawHttpRequest,
        token_usage_from_body,
    )

    from tests.envs.support import _prompt_of, transport_policy

    by_entry = {
        t.humaneval_task.entry_point: t.humaneval_task.ground_truth_code
        for t in tasks
    }

    class _T:
        policy = transport_policy()

        def __call__(self, request: ProviderCallRequest):
            prompt = _prompt_of(request)
            if prompt.startswith("Provide") or prompt.startswith("Compress"):
                for ep in by_entry:
                    if f"def {ep}(" in prompt:
                        text = f"REBUILD:{ep}"
                        break
                else:
                    text = "REBUILD:x"
            elif prompt.startswith("REBUILD") or "REBUILD:" in prompt:
                text = next(iter(by_entry.values()))
                for ep, gt in by_entry.items():
                    if f"REBUILD:{ep}" in prompt:
                        text = gt
                        break
            else:
                text = next(iter(by_entry.values()))
                for ep, gt in by_entry.items():
                    if ep in prompt:
                        text = gt
                        break
            usage_body = {
                "usage": {
                    "prompt_tokens": 11, "completion_tokens": 22,
                    "total_tokens": 33,
                    **({"completion_tokens_details": {
                        "reasoning_tokens": reasoning}} if reasoning
                       is not None else {}),
                }
            }
            raw = RawHttpRequest.build(
                url="https://example.test/v1/chat/completions",
                headers={"content-type": "json"},
                body={"model": "m"},
            )
            resp = ProviderTransportResponse(
                text=text,
                raw_body={"choices": [{"message": {"content": text}}]},
                usage=token_usage_from_body(usage_body),
                response_id="r", model="m", finish_reason="stop",
            )
            return ProviderInvocationEvidence.build(
                request=request, policy=self.policy, raw_request=raw,
                outcome=resp,
            )

    return _T()


def test_screen_captures_reasoning_and_latency_telemetry(
    tmp_path: Path,
) -> None:
    # (Task 20) Screen rows capture reasoning tokens + latency; the artifact's
    # arm_summary reports mean/median latency + total reasoning with coverage.
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=2)
    report = run_task_screen(
        model="gpt-5.4-nano",
        transport=_telemetry_transport(tasks, reasoning=5),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=2, concurrency=2,
    )
    summary = report.arm_summary()
    # Direct arm = 1 call (reasoning 5); encdec arm = 2 calls (reasoning 10).
    direct = summary["direct_original"]
    assert direct["reasoning_coverage"] == 4  # 2 tasks x 2 repeats
    assert direct["total_reasoning_tokens"] == 20  # 4 rows x 5
    assert direct["mean_reasoning_tokens"] == 5.0
    assert direct["latency_coverage"] == 4
    assert direct["mean_latency_s"] is not None  # a real (tiny) wall-clock
    encdec = summary["encdec_naive"]
    assert encdec["total_reasoning_tokens"] == 40  # 4 rows x (5+5)
    # The artifact carries the telemetry aggregates.
    data = json.loads(report.write(tmp_path).read_text())
    summ = data["arm_summary"]["direct_original"]
    assert summ["total_reasoning_tokens"] == 20


def test_screen_telemetry_coverage_honest_when_absent(tmp_path: Path) -> None:
    # (Task 20) A provider exposing NO reasoning detail -> reasoning_tokens
    # None (never 0-conflated); coverage counts the rows-with-field only.
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=2)
    report = run_task_screen(
        model="qwen/qwen3-coder-flash",
        transport=_telemetry_transport(tasks, reasoning=None),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=1, concurrency=2,
    )
    s = report.arm_summary()["direct_original"]
    # No reasoning reported anywhere -> total is None, coverage 0.
    assert s["total_reasoning_tokens"] is None
    assert s["reasoning_coverage"] == 0
    # Latency IS still captured (from the driver clock).
    assert s["latency_coverage"] == 2


def test_screen_telemetry_mixed_coverage_counts_not_conflated() -> None:
    # (Task 20 coverage honesty) When only SOME rows report reasoning, the
    # aggregate is over rows-with-field with a coverage count -- nulls NEVER
    # summed as zeros. Simulated by a report with mixed per-row samples.
    from whetstone.runner.task_screen import TaskScreenReport, TaskScreenRow

    rows = (
        TaskScreenRow(
            task_id="t1", entry_point="f", repeats=2,
            pass_counts={"direct_original": 2},
            fail_counts={"direct_original": 0},
            arm_latencies={"direct_original": [0.1, 0.2]},
            arm_reasoning={"direct_original": [5]},  # only 1 of 2 reported
        ),
    )
    report = TaskScreenReport(
        model="m", budget_ratio=0.25, repeats=2,
        arms=("direct_original",), rename_token="target_fxn",
        dataset_revision="rev", name_only_wrapper="w", rows=rows,
    )
    s = report.arm_summary()["direct_original"]
    assert s["reasoning_coverage"] == 1  # NOT 2 -- only rows with the field
    assert s["total_reasoning_tokens"] == 5
    assert s["mean_reasoning_tokens"] == 5.0
    assert s["latency_coverage"] == 2
    assert s["median_latency_s"] == pytest.approx(0.15)


# --- Task 28 item 2: honest screen aggregates (provider_error excluded) ------


def test_arm_summary_excludes_provider_error_rows_from_mean_pass() -> None:
    # A rate-limited provider_error row is pollution, not a content failure:
    # the headline mean_pass excludes it (non-error denominator), the polluted
    # all-rows number is kept for continuity, and rows_errored is surfaced.
    from whetstone.runner.task_screen import TaskScreenReport, TaskScreenRow

    # 4 tasks x 5 repeats. Task t1 has 3 error (rate-limited) rows; of its 2
    # non-error rows both pass -> honest full-pass. Its pass_counts=2.
    rows = (
        TaskScreenRow(
            task_id="t1", entry_point="f", repeats=5,
            pass_counts={"direct_original": 2},
            fail_counts={"direct_original": 3},
            error_counts={"direct_original": 3},
        ),
        TaskScreenRow(
            task_id="t2", entry_point="g", repeats=5,
            pass_counts={"direct_original": 5},
            fail_counts={"direct_original": 0},
            error_counts={"direct_original": 0},
        ),
        TaskScreenRow(
            task_id="t3", entry_point="h", repeats=5,
            pass_counts={"direct_original": 4},
            fail_counts={"direct_original": 1},  # a genuine content fail
            error_counts={"direct_original": 0},
        ),
        TaskScreenRow(
            task_id="t4", entry_point="i", repeats=5,
            pass_counts={"direct_original": 5},
            fail_counts={"direct_original": 0},
            error_counts={"direct_original": 0},
        ),
    )
    report = TaskScreenReport(
        model="m", budget_ratio=0.25, repeats=5,
        arms=("direct_original",), rename_token="target_fxn",
        dataset_revision="rev", name_only_wrapper="w", rows=rows,
    )
    s = report.arm_summary()["direct_original"]
    total_pass = 2 + 5 + 4 + 5  # 16
    # Honest denominator: 20 total rows - 3 error rows = 17.
    assert s["mean_pass_rate"] == pytest.approx(total_pass / 17)
    # Polluted all-rows number kept for continuity (denominator 20).
    assert s["mean_pass_all_rows"] == pytest.approx(total_pass / 20)
    # Pollution surfaced, not folded in.
    assert s["rows_errored"] == 3
    # Honest full-pass: t1 (2/2 non-error pass), t2, t4 -> 3; t3 has a real
    # content fail so it is NOT full-pass.
    assert s["tasks_full_pass"] == 3
    # The all-rows full-pass count is stricter (t1's 2 != 5) -> only t2, t4.
    assert s["tasks_full_pass_all_rows"] == 2
    # Report-level totals expose the pollution.
    assert report.rows_errored == 3
    assert report.rows_errored_by_arm() == {"direct_original": 3}
    data = report.as_dict()
    assert data["rows_errored"] == 3
    assert data["rows_errored_by_arm"] == {"direct_original": 3}


def test_is_provider_error_detects_payload_and_rate_limit_code() -> None:
    # An error row is one with a typed provider_error payload OR (for a
    # restored row that predates payload persistence) a rate-limit code.
    from whetstone.runner.task_screen import (
        _is_provider_error,
        _ScreenRowOutcome,
    )

    payload = _ScreenRowOutcome(
        passed=None, failed=True, failure_code="http_status_429",
        output_text=None, provider_error={"status": 429},
    )
    assert _is_provider_error(payload) is True
    # No payload but a rate-limit failure code (restored row) still counts.
    code_only = _ScreenRowOutcome(
        passed=None, failed=True, failure_code="http_status_429",
        output_text=None,
    )
    assert _is_provider_error(code_only) is True
    # A genuine content failure (no payload, non-rate-limit code) does NOT.
    content_fail = _ScreenRowOutcome(
        passed=False, failed=False, failure_code="", output_text="wrong",
    )
    assert _is_provider_error(content_fail) is False


def test_screen_sidecar_stamps_at_on_every_row(tmp_path: Path) -> None:
    # (Task 28 item 3) Every sidecar row -- success AND (would-be) error
    # rows -- carries an ISO-8601 UTC ``at`` stamp for failure-timing.
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=2)
    sidecar = tmp_path / "ed1_qwen.outputs.jsonl"
    run_task_screen(
        model="qwen/qwen3-coder-flash",
        transport=_all_pass_reply(tasks),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=1, concurrency=2,
        variants=("direct_original",),
        sidecar_path=sidecar,
    )
    lines = [
        json.loads(line)
        for line in sidecar.read_text().splitlines() if line.strip()
    ]
    assert lines  # rows were written
    for row in lines:
        assert row.get("at")  # non-empty ISO timestamp on every row
        assert row["at"].endswith("+00:00") or "T" in row["at"]


def test_arm_summary_fully_polluted_arm_is_defined_zero() -> None:
    # An arm whose every row is a provider_error has no honest signal: the
    # headline mean_pass is a defined 0.0 (never NaN) and rows_errored == all.
    from whetstone.runner.task_screen import TaskScreenReport, TaskScreenRow

    rows = (
        TaskScreenRow(
            task_id="t1", entry_point="f", repeats=3,
            pass_counts={"direct_original": 0},
            fail_counts={"direct_original": 3},
            error_counts={"direct_original": 3},
        ),
    )
    report = TaskScreenReport(
        model="m", budget_ratio=0.25, repeats=3,
        arms=("direct_original",), rename_token="target_fxn",
        dataset_revision="rev", name_only_wrapper="w", rows=rows,
    )
    s = report.arm_summary()["direct_original"]
    assert s["mean_pass_rate"] == 0.0
    assert s["rows_errored"] == 3
    # No non-error row -> the task carries no honest full-pass verdict.
    assert s["tasks_full_pass"] == 0


# --- Task 21.1: reasoning-effort keyed artifacts + honor-vs-ignore -----------


def test_screen_stem_folds_reasoning_effort() -> None:
    from whetstone.runner.task_screen import effort_suffix, screen_stem

    # Default effort keeps the original name (byte-compat w/ running runs).
    assert screen_stem("qwen/q3", 0.25, None) == "ed1_qwen_q3_r025"
    assert effort_suffix(None) == ""
    # A labeled effort adds _e<effort>, never colliding with default.
    assert screen_stem("qwen/q3", 0.25, "low") == "ed1_qwen_q3_r025_elow"
    assert screen_stem("qwen/q3", 0.25, "none") == "ed1_qwen_q3_r025_enone"


def test_screen_artifact_and_phase_keyed_by_effort(tmp_path: Path) -> None:
    # (Task 21.1) A low-effort round writes a DISTINCT artifact + phase
    # from the default round -- never overwrites or cross-resumes.
    from whetstone.execution.partials import PartialLog

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=2)
    log = PartialLog(path=tmp_path / "s.partial.jsonl")
    default = run_task_screen(
        model="gpt-5.4-nano", transport=_all_pass_reply(tasks),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=1, concurrency=2, variants=("direct_original",),
        partial_log=log,
    )
    low = run_task_screen(
        model="gpt-5.4-nano", transport=_all_pass_reply(tasks),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=1, concurrency=2, variants=("direct_original",),
        reasoning_effort="low", partial_log=log,
    )
    p_default = default.write(tmp_path)
    p_low = low.write(tmp_path)
    assert p_default.name == "ed1_gpt_5_4_nano_r025.json"
    assert p_low.name == "ed1_gpt_5_4_nano_r025_elow.json"
    assert p_default != p_low and p_default.exists() and p_low.exists()
    assert json.loads(p_low.read_text())["reasoning_effort"] == "low"
    # The two configs recorded DISTINCT partials phases (no cross-resume).
    phases = {rec.phase for rec in log.load()}
    assert any(":r025:" in p for p in phases)       # default
    assert any(":r025_elow:" in p for p in phases)  # low


def test_reasoning_honored_flags_detect_ignore() -> None:
    # (Task 21.3) A model that IGNORES the effort has ~equal reasoning tokens
    # across default and low -> honored=False, flagged as duplicate.
    from whetstone.runner.task_screen import (
        TaskScreenReport,
        TaskScreenRow,
        reasoning_honored_flags,
    )

    def _report(effort, reason_per_row):
        rows = (
            TaskScreenRow(
                task_id="t1", entry_point="f", repeats=1,
                pass_counts={"direct_original": 1},
                fail_counts={"direct_original": 0},
                arm_latencies={"direct_original": [0.1]},
                arm_reasoning={"direct_original": [reason_per_row]},
            ),
        )
        return TaskScreenReport(
            model="m", budget_ratio=0.25, repeats=1,
            arms=("direct_original",), rename_token="target_fxn",
            dataset_revision="rev", name_only_wrapper="w", rows=rows,
            reasoning_effort=effort,
        )

    # IGNORE case: default=100, low=100 -> not honored.
    flags = reasoning_honored_flags([
        _report(None, 100), _report("low", 100),
    ])
    key = "m@r025/low"
    assert flags[key]["honored"] is False
    assert "IGNORED" in str(flags[key]["note"])
    # HONOR case: default=100, low=10 -> honored (moved >5%).
    flags2 = reasoning_honored_flags([
        _report(None, 100), _report("low", 10),
    ])
    assert flags2[key]["honored"] is True
    assert flags2[key]["note"] is None


# --- Task 27: per-(model, effort) sidecar lock + atomic summary rewrite ----


def test_sidecar_lock_path_drops_ratio_keeps_model_effort(
    tmp_path: Path,
) -> None:
    # The lock guards the SIDECAR (keyed by model+effort, no ratio). It lives
    # beside the sidecar, named for the same (model, effort) key.
    from whetstone.runner.task_screen import sidecar_lock_path

    d = tmp_path / "task_screen"
    sidecar = d / "ed1_google_gemini_enone.outputs.jsonl"
    lock = sidecar_lock_path(sidecar)
    assert lock == d / "ed1_google_gemini_enone.lock"
    # No budget_ratio component anywhere in the lock name -- two ratios of the
    # same model+effort share one sidecar and MUST serialize on one lock.
    assert "r0" not in lock.name and "r1" not in lock.name


def test_screen_lock_second_start_refuses_typed(tmp_path: Path) -> None:
    # While one process holds the lock, a second start on the SAME key refuses
    # with the typed ScreenKeyLocked failure -- no wait, no retry.
    from whetstone.runner.task_screen import (
        ScreenKeyLocked,
        screen_key_lock,
    )

    lock = tmp_path / "ed1_m_enone.lock"
    with screen_key_lock(lock, screen_key="screen:m_enone"):
        with pytest.raises(ScreenKeyLocked) as excinfo:
            with screen_key_lock(lock, screen_key="screen:m_enone"):
                pass
    msg = str(excinfo.value)
    assert "screen:m_enone" in msg  # names the contested key
    assert "one writer per" in msg  # states the standing rule
    assert "wait for the holder" in msg  # suggests waiting


def test_screen_lock_released_on_normal_exit(tmp_path: Path) -> None:
    # After a clean block exit the lock is free -- a re-acquire succeeds.
    from whetstone.runner.task_screen import screen_key_lock

    lock = tmp_path / "ed1_m_enone.lock"
    with screen_key_lock(lock, screen_key="screen:m_enone"):
        pass
    # Re-acquiring must not raise (the lock was released).
    with screen_key_lock(lock, screen_key="screen:m_enone"):
        pass


def test_screen_lock_released_on_exception(tmp_path: Path) -> None:
    # An exception raised INSIDE the block still releases the lock (finally).
    from whetstone.runner.task_screen import screen_key_lock

    lock = tmp_path / "ed1_m_enone.lock"
    with pytest.raises(RuntimeError, match="boom"):
        with screen_key_lock(lock, screen_key="screen:m_enone"):
            raise RuntimeError("boom")
    # The lock is free despite the exception -- re-acquire succeeds.
    with screen_key_lock(lock, screen_key="screen:m_enone"):
        pass


def test_screen_lock_refusal_emits_marker_and_event(tmp_path: Path) -> None:
    # A refused start fires the loud SCREEN-KEY-LOCKED marker AND an
    # events.jsonl event when the event stream is available.
    from whetstone.runner.events import (
        EVENT_MARKERS,
        SCREEN_KEY_LOCKED,
        EventStream,
    )
    from whetstone.runner.task_screen import ScreenKeyLocked, screen_key_lock

    markers: list[str] = []
    events = EventStream(root=tmp_path, marker_sink=markers.append)
    lock = tmp_path / "ed1_m_enone.lock"
    with screen_key_lock(lock, screen_key="screen:m_enone"):
        with pytest.raises(ScreenKeyLocked):
            with screen_key_lock(
                lock, screen_key="screen:m_enone", events=events,
                marker_sink=markers.append,
            ):
                pass
    # The loud greppable marker fired.
    assert any(EVENT_MARKERS[SCREEN_KEY_LOCKED] in m for m in markers)
    # And a typed event landed on the stream.
    emitted = [e for e in events.load() if e.event == SCREEN_KEY_LOCKED]
    assert len(emitted) == 1
    assert emitted[0].fields["screen_key"] == "screen:m_enone"
    assert emitted[0].fields["lock_path"] == str(lock)


def test_screen_run_refuses_second_writer_on_shared_sidecar(
    tmp_path: Path,
) -> None:
    # Two same-(model, effort) screens sharing ONE sidecar: while one holds the
    # lock, a second run_task_screen refuses. Simulates the concurrency race by
    # holding the derived sidecar lock, then starting a second screen.
    from whetstone.runner.task_screen import (
        ScreenKeyLocked,
        screen_key_lock,
        sidecar_lock_path,
    )

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=1)
    sidecar = (
        tmp_path / "task_screen"
        / "ed1_qwen_qwen3_coder_flash.outputs.jsonl"
    )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    held = sidecar_lock_path(sidecar)
    with screen_key_lock(held, screen_key="held"):
        with pytest.raises(ScreenKeyLocked):
            run_task_screen(
                model="qwen/qwen3-coder-flash",
                transport=_all_pass_reply(tasks),
                execution_policy=execution_policy(max_attempts=1),
                tasks=tasks, repeats=1, concurrency=1,
                sidecar_path=sidecar,
            )


def test_screen_run_holds_then_releases_sidecar_lock(tmp_path: Path) -> None:
    # A normal screen run acquires and RELEASES the sidecar lock -- a second
    # run on the same key afterward succeeds (no leaked lock).
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=1)
    sidecar = (
        tmp_path / "task_screen"
        / "ed1_qwen_qwen3_coder_flash.outputs.jsonl"
    )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        run_task_screen(
            model="qwen/qwen3-coder-flash",
            transport=_all_pass_reply(tasks),
            execution_policy=execution_policy(max_attempts=1),
            tasks=tasks, repeats=1, concurrency=1, sidecar_path=sidecar,
        )
    # Both runs completed; the lock is not held after the loop.


def test_summary_rewrite_is_atomic_no_partial_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The summary write goes through a temp file + os.replace(): a crash AT the
    # replace leaves the PRIOR summary intact (never torn) and no temp behind.
    import json as _json

    import whetstone.runner.task_screen as ts_mod

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=1)
    report = run_task_screen(
        model="qwen/qwen3-coder-flash",
        transport=_all_pass_reply(tasks),
        execution_policy=execution_policy(max_attempts=1),
        tasks=tasks, repeats=1, concurrency=1,
    )
    # First write lands a valid summary.
    path = report.write(tmp_path)
    original = path.read_text()
    assert _json.loads(original)["schema"] == SCREEN_SCHEMA

    # Simulate a crash right at the atomic swap: the temp file is fully
    # written, then os.replace() raises. The final path must be untouched (the
    # prior valid summary) and the finally-cleanup removes the temp file.
    def _boom_replace(src: object, dst: object) -> None:
        raise OSError("crash at replace")

    monkeypatch.setattr(ts_mod.os, "replace", _boom_replace)
    with pytest.raises(OSError, match="crash at replace"):
        report.write(tmp_path)

    # The FINAL summary is still the prior valid one (never torn) and parses.
    assert path.read_text() == original
    _json.loads(path.read_text())
    # No torn temp file left behind (finally cleaned it up).
    assert list(path.parent.glob(".*.tmp.*")) == []
