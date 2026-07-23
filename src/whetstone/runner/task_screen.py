"""The ed1 task-informativeness screen (task 17 Part 2).

Screens the full HumanEval+ pool for tasks that carry NO information for a
given task model -- tasks a model solves 100% of the time regardless of how the
prompt is presented. The user's rule: a task 100% correct across the screen's
arms carries no signal and is DROPPED from the ed1 train/eval/test pools.

Per (task_model, task) the screen runs SEVEN arms x N repeats and scores each
submission through the SAME dr-code HumanEval sandbox the ed1 rollout uses:

  * FIVE DIRECT arms -- generate the function directly from a slice of the
    canonical HumanEval prompt, varying HOW MUCH of the prompt the model sees:
      1. ``direct_original``   -- the full canonical prompt, verbatim.
      2. ``direct_docstring``  -- the docstring/comment text ONLY.
      3. ``direct_signature``  -- the ``def`` signature line(s) ONLY.
      4. ``direct_name``       -- the bare function NAME in a neutral wrapper
         (:data:`NAME_ONLY_WRAPPER`) -- the memorization DISCRIMINATOR.
      5. ``direct_renamed``    -- the full prompt with EVERY canonical-name
         occurrence (signature + doctests) scrubbed to a neutral token -- the
         CAUSAL memorization ablation.
  * TWO ENCDEC arms -- the 3-node encoder->decoder->code-eval rollout with the
    NAIVE strategy body under the immutable frame (part 1) at the budget ratio
    (default 0.25): ``encdec_naive`` and ``encdec_renamed`` (the input code
    name-scrubbed, scored against the renamed entry point).

The screen runs per task model; the exclusion list is PER-MODEL (a task
uninformative for one model may be informative for another). Output:
``<root>/task_screen/ed1_<model_tag>.json`` -- per task the pass count/repeats
per arm + an ``always_pass`` verdict + canonical-minus-renamed rename deltas,
plus a cross-model summary table for the paper's contamination section.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from whetstone.envs.ed1 import (
    ENCODER_BODY_A,
    Ed1Instance,
    build_ed1_experiment,
)
from whetstone.envs.ed1_scoring import CodeScore, score_ed1_submission
from whetstone.execution.fanout import CallSpec, FanoutConfig, run_call_pool
from whetstone.execution.partials import PartialCallRecord, PartialLog
from whetstone.provider.driver import TransportCall, run_provider_call
from whetstone.provider.policy import ProviderExecutionPolicy

SCREEN_SCHEMA = "whetstone.runner.task_screen/v2"

#: The neutral name-only wrapper (recorded in the artifact for repro).
NAME_ONLY_WRAPPER = "Write a Python function named {name}."

#: The default rename token for the memorization-ablation arms (recorded).
DEFAULT_RENAME_TOKEN = "target_fxn"

#: The direct-arm ids (order stable for the artifact + summary table).
DIRECT_ARMS: tuple[str, ...] = (
    "direct_original",
    "direct_docstring",
    "direct_signature",
    "direct_name",
    "direct_renamed",
)
ENCDEC_ARMS: tuple[str, ...] = (
    "encdec_naive",
    "encdec_renamed",
)
ENCDEC_ARM = "encdec_naive"
SCREEN_ARMS: tuple[str, ...] = (*DIRECT_ARMS, *ENCDEC_ARMS)

#: The canonical-minus-renamed causal-memorization deltas (paper figure): the
#: canonical arm paired with its renamed ablation.
RENAME_DELTA_PAIRS: tuple[tuple[str, str], ...] = (
    ("direct_original", "direct_renamed"),
    ("encdec_naive", "encdec_renamed"),
)


def rename_identifier(text: str, old: str, new: str) -> str:
    """Replace EVERY whole-identifier occurrence of ``old`` with ``new``.

    Uses identifier boundaries (not ``\\b``, which treats ``_`` as a word char
    correctly but also matches inside dotted access) so the signature line AND
    every doctest example (``>>> old(...)``) is renamed -- a leaked canonical
    name in a doctest would otherwise void the memorization ablation.
    """
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(old) + r"(?![A-Za-z0-9_])"
    return re.sub(pattern, new, text)


def model_tag(model: str) -> str:
    """A filesystem-safe tag for a model slug (``a/b:c`` -> ``a_b_c``)."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", model).strip("_")


# --- Canonical-prompt splitting (the four direct arms) -----------------------


@dataclass(frozen=True, slots=True)
class PromptParts:
    """The canonical HumanEval prompt split into direct-arm inputs."""

    original: str
    docstring: str
    signature: str
    name_only: str
    entry_point: str


def split_prompt(prompt: str, entry_point: str) -> PromptParts:
    """Split a canonical HumanEval prompt into the four direct-arm inputs.

    * ``original`` -- the verbatim prompt (imports + signature + docstring).
    * ``docstring`` -- the triple-quoted docstring text ONLY (no signature).
    * ``signature`` -- everything up to and INCLUDING the entry-point ``def``
      line(s) (a multi-line signature is kept whole), no docstring.
    * ``name_only`` -- the bare function name in :data:`NAME_ONLY_WRAPPER`.
    """
    lines = prompt.split("\n")
    sig_end: int | None = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def ") and f"{entry_point}(" in line:
            depth = 0
            j = i
            while j < len(lines):
                depth += lines[j].count("(") - lines[j].count(")")
                if lines[j].rstrip().endswith(":") and depth <= 0:
                    sig_end = j
                    break
                j += 1
            break
    signature = (
        "\n".join(lines[: sig_end + 1]) if sig_end is not None else prompt
    )
    match = re.search(r"(\"\"\"|''')(.*?)(\1)", prompt, re.DOTALL)
    docstring = match.group(2).strip() if match else ""
    return PromptParts(
        original=prompt,
        docstring=docstring,
        signature=signature,
        name_only=NAME_ONLY_WRAPPER.format(name=entry_point),
        entry_point=entry_point,
    )


# --- Per-arm rollout drives --------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ScreenRowOutcome:
    """One (task, arm, repeat) screen row's result."""

    passed: bool | None
    failed: bool
    failure_code: str
    output_text: str | None


def _direct_body(arm: str, parts: PromptParts, *, rename_token: str) -> str:
    """The prompt-slice body for one DIRECT arm (renamed arm fully renamed)."""
    if arm == "direct_original":
        return parts.original
    if arm == "direct_docstring":
        return parts.docstring
    if arm == "direct_signature":
        return parts.signature
    if arm == "direct_name":
        return parts.name_only
    if arm == "direct_renamed":
        # The FULL prompt with EVERY canonical-name occurrence (signature +
        # doctest examples) replaced by the neutral rename token.
        return rename_identifier(
            parts.original, parts.entry_point, rename_token
        )
    raise ValueError(f"unknown direct arm {arm!r}")  # pragma: no cover


def _direct_prompt(arm: str, parts: PromptParts, *, rename_token: str) -> str:
    """The generation input for one DIRECT arm.

    A short instruction frames the prompt slice so the model returns a function
    (the sandbox scorer extracts fenced/prose code, so the exact wrapper is not
    load-bearing -- it just asks for an implementation).
    """
    body = _direct_body(arm, parts, rename_token=rename_token)
    return (
        "Write a complete, correct Python implementation for the following. "
        "Output only Python code.\n" + body
    )


def renamed_task(ht, *, old: str, new: str):
    """A HumanEvalTask with EVERY occurrence of ``old`` renamed to ``new``.

    Renames the prompt, canonical solution, test harness AND the entry_point
    field so the sandbox scores the submission against the RENAMED name, not
    silently against the canonical name (the amendment-2 scoring trap).
    """
    from dr_code.humaneval import HumanEvalTask

    def _rn(text: str | None) -> str:
        return rename_identifier(text, old, new) if text else (text or "")

    return HumanEvalTask(
        task_id=ht.task_id,
        prompt=_rn(ht.prompt),
        canonical_solution=_rn(ht.canonical_solution),
        entry_point=new,
        test=_rn(ht.test),
    )


def _score(scorer: Callable[..., CodeScore], *, code: str, ht) -> CodeScore:
    return scorer(raw_submission=code, task=ht)


def _run_direct_row(
    *,
    arm: str,
    instance: Ed1Instance,
    provider_call_config,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    scorer: Callable[..., CodeScore],
    logical_call_id: str,
    rename_token: str,
) -> _ScreenRowOutcome:
    """Drive one DIRECT-arm rollout: prompt -> generation -> code score."""
    from dr_providers import (
        MessageRole,
        PromptMessage,
        ProviderCallRequest,
        Transcript,
    )

    ht = instance.humaneval_task
    parts = split_prompt(ht.prompt, ht.entry_point)
    prompt = _direct_prompt(arm, parts, rename_token=rename_token)
    # The renamed arm scores against the RENAMED entry point (never the leaked
    # canonical name) -- the amendment-2 scoring trap.
    score_task = (
        renamed_task(ht, old=ht.entry_point, new=rename_token)
        if arm == "direct_renamed" else ht
    )
    request = ProviderCallRequest(
        config=provider_call_config,
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content=prompt),)
        ),
    )
    result = run_provider_call(
        request=request, policy=execution_policy, transport=transport,
        logical_call_id=logical_call_id,
    )
    if not result.succeeded or result.generation is None:
        from whetstone.execution.call_support import failure_code_of
        return _ScreenRowOutcome(
            passed=None, failed=True,
            failure_code=failure_code_of(result), output_text=None,
        )
    code = result.generation.text
    score = _score(scorer, code=code, ht=score_task)
    if score.infrastructure_unknown:
        return _ScreenRowOutcome(
            passed=None, failed=True,
            failure_code="code_eval_infrastructure_unknown",
            output_text=code,
        )
    return _ScreenRowOutcome(
        passed=score.passed, failed=False, failure_code="",
        output_text=code,
    )


def _renamed_instance(instance: Ed1Instance, *, new: str) -> Ed1Instance:
    """An Ed1Instance whose input_code + reconstructable task are renamed.

    Renames EVERY occurrence of the canonical entry point (in ``input_code``
    and the packed prompt/test/entry_point fields) to ``new`` so the
    encoder sees the renamed code AND the sandbox scores against the renamed
    name (the amendment-2 scoring trap). The gold + humaneval_task are renamed
    to match.
    """
    from whetstone_envs.core import Instance

    ht = instance.humaneval_task
    old = ht.entry_point
    pi = dict(instance.instance.prompt_inputs)
    pi["input_code"] = rename_identifier(pi["input_code"], old, new)
    pi["prompt"] = rename_identifier(pi.get("prompt", ""), old, new)
    pi["canonical_solution"] = rename_identifier(
        pi.get("canonical_solution", ""), old, new
    )
    pi["test"] = rename_identifier(pi.get("test", ""), old, new)
    pi["entry_point"] = new
    new_instance = Instance(
        id=instance.instance.id, seed=instance.instance.seed,
        strata=instance.instance.strata, prompt_inputs=pi,
        gold=rename_identifier(instance.instance.gold, old, new),
    )
    return Ed1Instance(
        instance=new_instance,
        humaneval_task=renamed_task(ht, old=old, new=new),
    )


def _run_encdec_row(
    *,
    instance: Ed1Instance,
    budget_ratio: float,
    provider_call_config,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    scorer: Callable[..., CodeScore],
    logical_call_id: str,
    rename_token: str | None = None,
) -> _ScreenRowOutcome:
    """Drive one ENCDEC rollout via the shared ed1 row driver.

    ``rename_token`` (the encdec_renamed arm) renames the input code BEFORE the
    encoder sees it and scores the decoder output against the renamed entry
    point -- the memorization ablation on the enc-dec channel.
    """
    from whetstone.envs.ed1_eval import _drive_row

    if rename_token is not None:
        instance = _renamed_instance(instance, new=rename_token)
    # A minimal experiment shim carrying the encdec rollout at this ratio, so
    # shared _drive_row composes the immutable frame around the naive body.
    exp = build_ed1_experiment(
        tasks=(instance,), internal_n=1, official_n=1,
        budget_ratio=budget_ratio,
    )
    outcome = _drive_row(
        experiment=exp,
        candidate_template=ENCODER_BODY_A,  # the naive strategy body
        instance=instance.instance,
        provider_call_config=provider_call_config,
        execution_policy=execution_policy,
        transport=transport,
        scorer=scorer,
        logical_call_id=logical_call_id,
    )
    text = None
    if outcome.encoder_text is not None or outcome.decoder_text is not None:
        text = (
            f"ENCODER:\n{outcome.encoder_text or ''}\n\n"
            f"DECODER:\n{outcome.decoder_text or ''}"
        )
    return _ScreenRowOutcome(
        passed=(None if outcome.pass_value is None
                else bool(outcome.pass_value)),
        failed=outcome.failed,
        failure_code=outcome.failure_code,
        output_text=text,
    )


# --- Screen output schema ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskScreenRow:
    """One task's per-arm pass counts + verdict for one model."""

    task_id: str
    entry_point: str
    pass_counts: dict[str, int]
    fail_counts: dict[str, int]
    repeats: int

    @property
    def always_pass(self) -> bool:
        """True when EVERY scored arm passed all repeats (no information).

        Evaluated over the arms THIS row actually ran (its ``pass_counts``
        keys), so a subset-``--variants`` screen judges on what it measured.
        """
        return bool(self.pass_counts) and all(
            count == self.repeats for count in self.pass_counts.values()
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "entry_point": self.entry_point,
            "repeats": self.repeats,
            "pass_counts": dict(self.pass_counts),
            "fail_counts": dict(self.fail_counts),
            "always_pass": self.always_pass,
        }


@dataclass(frozen=True, slots=True)
class TaskScreenReport:
    """The per-model screen report + the pool-exclusion list it induces.

    Carries the FULL config identity (schema version, model, arms, repeats,
    dataset revision, the verbatim name-only wrapper + rename token) so two
    screen runs are comparable-or-distinct on inspection -- no wall-clock is
    recorded, so a re-run of the same config produces a byte-comparable header.
    """

    model: str
    budget_ratio: float
    repeats: int
    arms: tuple[str, ...]
    rename_token: str
    dataset_revision: str
    name_only_wrapper: str
    rows: tuple[TaskScreenRow, ...]

    @property
    def excluded_task_ids(self) -> tuple[str, ...]:
        """The always-pass task ids (the model's pool-exclusion list)."""
        return tuple(r.task_id for r in self.rows if r.always_pass)

    def arm_summary(self) -> dict[str, dict[str, float]]:
        """Per-arm mean pass rate + n tasks at full pass (the paper table)."""
        out: dict[str, dict[str, float]] = {}
        n_tasks = len(self.rows) or 1
        for arm in self.arms:
            total_pass = sum(r.pass_counts.get(arm, 0) for r in self.rows)
            total_rows = self.repeats * len(self.rows) or 1
            full = sum(
                1 for r in self.rows
                if r.pass_counts.get(arm, 0) == self.repeats
            )
            out[arm] = {
                "mean_pass_rate": total_pass / total_rows,
                "tasks_full_pass": full,
                "tasks_full_pass_fraction": full / n_tasks,
            }
        return out

    def rename_deltas(self) -> dict[str, dict[str, float]]:
        """Canonical-minus-renamed mean-pass delta per pair (paper figure).

        The CAUSAL memorization signal: how much a model's pass rate DROPS when
        the canonical function name is scrubbed. Computed only for pairs whose
        BOTH arms were screened.
        """
        summary = self.arm_summary()
        out: dict[str, dict[str, float]] = {}
        for canon, renamed in RENAME_DELTA_PAIRS:
            if canon in summary and renamed in summary:
                c = summary[canon]["mean_pass_rate"]
                r = summary[renamed]["mean_pass_rate"]
                out[f"{canon}_minus_{renamed}"] = {
                    "canonical_mean_pass": c,
                    "renamed_mean_pass": r,
                    "delta": c - r,
                }
        return out

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCREEN_SCHEMA,
            "model": self.model,
            "budget_ratio": self.budget_ratio,
            "repeats": self.repeats,
            "arms": list(self.arms),
            "dataset_revision": self.dataset_revision,
            "name_only_wrapper": self.name_only_wrapper,
            "rename_token": self.rename_token,
            "science_intent": (
                "The direct_name arm is the memorization DISCRIMINATOR (solve "
                "from the function NAME alone = recall). The direct_renamed / "
                "encdec_renamed arms are the CAUSAL ablation: scrubbing the "
                "canonical name (signature AND doctests) drops pass rate iff "
                "the model was relying on task recognition. Prediction: "
                "deepseek (contaminated) shows a large canonical-vs-renamed "
                "delta; a cleaner model a smaller one. A task passing "
                "ALL screened arms carries no signal and is dropped from the "
                "ed1 pools for THIS model."
            ),
            "arm_summary": self.arm_summary(),
            "rename_deltas": self.rename_deltas(),
            "excluded_task_ids": list(self.excluded_task_ids),
            "excluded_count": len(self.excluded_task_ids),
            "tasks": [r.as_dict() for r in self.rows],
        }

    def write(self, root: Path) -> Path:
        out_dir = root / "task_screen"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"ed1_{model_tag(self.model)}.json"
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))
        return path


# --- The screen driver -------------------------------------------------------


def run_task_screen(
    *,
    model: str,
    transport: TransportCall,
    execution_policy: ProviderExecutionPolicy,
    budget_ratio: float = 0.25,
    repeats: int = 5,
    variants: Sequence[str] | None = None,
    rename_token: str = DEFAULT_RENAME_TOKEN,
    tasks: Sequence[Ed1Instance] | None = None,
    limit: int | None = None,
    prefer_snapshot: bool = True,
    concurrency: int = 8,
    scorer: Callable[..., CodeScore] | None = None,
    partial_log: PartialLog | None = None,
    sidecar_path: Path | None = None,
) -> TaskScreenReport:
    """Screen the HumanEval+ pool for one task model -> a per-model report.

    Drives the selected ``variants`` (default: all seven :data:`SCREEN_ARMS`) x
    ``repeats`` per task through the injected transport + code ``scorer`` (a
    fake in tests, the live route in a real run). DETERMINISTIC: tasks run in
    fixed ``task_id`` order and the report pins the dataset revision + the
    verbatim wrapper/rename texts, so two runs are comparable-or-distinct on
    inspection (no wall-clock in identity). RESUMABLE: each completed row
    appends to ``partial_log`` (+ its output to ``sidecar_path``), and a
    re-run restores recorded rows instead of re-paying for them.
    """
    from whetstone.envs.ed1 import ED1_DATASET_REVISION, load_ed1_tasks

    scorer = scorer or score_ed1_submission
    arms = tuple(variants) if variants is not None else SCREEN_ARMS
    unknown = [a for a in arms if a not in SCREEN_ARMS]
    if unknown:
        raise ValueError(f"unknown screen variants: {unknown}")
    pool = list(
        tasks if tasks is not None
        else load_ed1_tasks(prefer_snapshot=prefer_snapshot, limit=limit)
    )
    # Deterministic task ordering (stable across re-runs; no wall-clock).
    by_task = {
        str(t.instance.id): t
        for t in sorted(pool, key=lambda t: str(t.instance.id))
    }
    ordered_ids = list(by_task)
    fanout = FanoutConfig(concurrency=concurrency)

    # The shared enc/dec/direct route config (one route for the whole screen).
    rd = build_ed1_experiment(
        tasks=tuple(pool[:1]) or None, internal_n=1, official_n=1,
        model=model, budget_ratio=budget_ratio,
    ).encdec_rollout
    assert rd is not None
    pcc = rd.provider_call_config

    restored = _restore_screen(partial_log, model)
    encdec_arms = frozenset(ENCDEC_ARMS)

    def _spec(
        task_id: str, arm: str, index: int
    ) -> CallSpec[tuple[str, str, int], _ScreenRowOutcome]:
        inst = by_task[task_id]
        cid = f"{model_tag(model)}:{task_id}:{arm}#{index}"

        def _run(
            inst=inst, arm=arm, cid=cid, index=index, task_id=task_id
        ) -> _ScreenRowOutcome:
            if arm in encdec_arms:
                out = _run_encdec_row(
                    instance=inst, budget_ratio=budget_ratio,
                    provider_call_config=pcc,
                    execution_policy=execution_policy, transport=transport,
                    scorer=scorer, logical_call_id=cid,
                    rename_token=(
                        rename_token if arm == "encdec_renamed" else None
                    ),
                )
            else:
                out = _run_direct_row(
                    arm=arm, instance=inst, provider_call_config=pcc,
                    execution_policy=execution_policy, transport=transport,
                    scorer=scorer, logical_call_id=cid,
                    rename_token=rename_token,
                )
            _persist_row(
                partial_log, sidecar_path, model=model, task_id=task_id,
                arm=arm, index=index, outcome=out,
            )
            return out

        from whetstone.execution.call_support import guard_deadline_seconds
        return CallSpec(
            key=(task_id, arm, index),
            run=_run,
            deadline_seconds=guard_deadline_seconds(
                execution_policy,
                wire_calls_per_unit=2 if arm in encdec_arms else 1,
            ),
        )

    specs = [
        _spec(task_id, arm, index)
        for task_id in ordered_ids
        for arm in arms
        for index in range(repeats)
        if (task_id, arm, index) not in restored
    ]
    pool_out = run_call_pool(
        specs, concurrency=fanout.concurrency,
        is_rate_limited=lambda _o: False,
        max_wall_seconds=fanout.max_wall_seconds,
    )
    driven: dict[tuple[str, str, int], _ScreenRowOutcome] = dict(restored)
    for res in pool_out.results:
        if res.value is not None:
            driven[res.key] = res.value
        else:
            driven[res.key] = _ScreenRowOutcome(
                passed=None, failed=True, failure_code="runner_timeout",
                output_text=None,
            )

    rows: list[TaskScreenRow] = []
    for task_id in ordered_ids:
        inst = by_task[task_id]
        pass_counts: dict[str, int] = {}
        fail_counts: dict[str, int] = {}
        for arm in arms:
            p = f = 0
            for index in range(repeats):
                out = driven[(task_id, arm, index)]
                if out.passed:
                    p += 1
                elif out.failed or out.passed is None:
                    f += 1
            pass_counts[arm] = p
            fail_counts[arm] = f
        rows.append(TaskScreenRow(
            task_id=task_id,
            entry_point=inst.humaneval_task.entry_point,
            pass_counts=pass_counts, fail_counts=fail_counts,
            repeats=repeats,
        ))
    return TaskScreenReport(
        model=model, budget_ratio=budget_ratio, repeats=repeats,
        arms=tuple(arms), rename_token=rename_token,
        dataset_revision=ED1_DATASET_REVISION,
        name_only_wrapper=NAME_ONLY_WRAPPER, rows=tuple(rows),
    )


def _persist_row(
    partial_log: PartialLog | None,
    sidecar_path: Path | None,
    *,
    model: str,
    task_id: str,
    arm: str,
    index: int,
    outcome: _ScreenRowOutcome,
) -> None:
    """Append one completed screen row to the partial log + output sidecar."""
    if partial_log is not None:
        partial_log.append(PartialCallRecord(
            phase=f"screen:{model_tag(model)}:{arm}",
            instance_id=task_id, unit=arm, repeat_id=index,
            score=(None if outcome.passed is None else float(outcome.passed)),
            failed=outcome.failed, failure_code=outcome.failure_code,
        ))
    if sidecar_path is not None:
        with sidecar_path.open("a") as handle:
            handle.write(json.dumps({
                "model": model, "task_id": task_id, "arm": arm,
                "repeat": index, "passed": outcome.passed,
                "failure_code": outcome.failure_code,
                "output_text": outcome.output_text,
            }) + "\n")


def _restore_screen(
    partial_log: PartialLog | None, model: str
) -> dict[tuple[str, str, int], _ScreenRowOutcome]:
    """Rebuild screen rows already recorded (resume skip) by (task,arm,r)."""
    if partial_log is None:
        return {}
    tag = model_tag(model)
    restored: dict[tuple[str, str, int], _ScreenRowOutcome] = {}
    for rec in partial_log.load():
        if not rec.phase.startswith(f"screen:{tag}:"):
            continue
        restored[(rec.instance_id, rec.unit, rec.repeat_id)] = (
            _ScreenRowOutcome(
                passed=(None if rec.score is None else bool(rec.score)),
                failed=rec.failed, failure_code=rec.failure_code,
                output_text=None,
            )
        )
    return restored


def load_exclusion_ids(screen_path: Path) -> frozenset[str]:
    """The always-pass exclusion task ids from a written screen artifact.

    Reads ``excluded_task_ids`` from a ``task_screen/ed1_<model>.json`` file so
    a cell can pass the model's exclusion list to ``build_ed1_experiment``.
    The screen file is PER-MODEL, so the caller picks the file matching the
    model the ed1 cell actually runs.
    """
    data = json.loads(screen_path.read_text())
    ids = data.get("excluded_task_ids", [])
    return frozenset(str(x) for x in ids)


def cross_model_summary(
    reports: Sequence[TaskScreenReport],
) -> dict[str, object]:
    """A compact cross-model table (per variant: mean pass, n at full pass).

    Headed for the paper's contamination section: one row per (model, arm) with
    the mean pass rate and the count of tasks the model solved at FULL pass.
    """
    table: list[dict[str, object]] = []
    for report in reports:
        summary = report.arm_summary()
        for arm in report.arms:
            table.append({
                "model": report.model,
                "arm": arm,
                "mean_pass_rate": summary[arm]["mean_pass_rate"],
                "tasks_full_pass": summary[arm]["tasks_full_pass"],
            })
    return {
        "schema": f"{SCREEN_SCHEMA}.cross_model",
        "arms": list(SCREEN_ARMS),
        "table": table,
        # The paper's causal-memorization figure: canonical-minus-renamed pass
        # delta per model per arm-pair (a larger delta = more contamination).
        "rename_deltas_by_model": {
            r.model: r.rename_deltas() for r in reports
        },
        "per_model_excluded": {
            r.model: len(r.excluded_task_ids) for r in reports
        },
    }


__all__ = [
    "DEFAULT_RENAME_TOKEN",
    "DIRECT_ARMS",
    "ENCDEC_ARM",
    "ENCDEC_ARMS",
    "NAME_ONLY_WRAPPER",
    "RENAME_DELTA_PAIRS",
    "SCREEN_ARMS",
    "SCREEN_SCHEMA",
    "PromptParts",
    "TaskScreenReport",
    "TaskScreenRow",
    "cross_model_summary",
    "load_exclusion_ids",
    "model_tag",
    "rename_identifier",
    "renamed_task",
    "run_task_screen",
    "split_prompt",
]
