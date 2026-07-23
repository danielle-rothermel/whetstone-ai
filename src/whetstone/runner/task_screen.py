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

import contextlib
import fcntl
import json
import os
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
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
from whetstone.runner.events import (
    EventStream,
    EventUnit,
    is_rate_limit_code,
    latency_snapshot_event,
    rate_limit_pressure_event,
    screen_key_locked_event,
)

SCREEN_SCHEMA = "whetstone.runner.task_screen/v2"


class ScreenKeyLocked(RuntimeError):
    """A second writer refused to start on an already-held screen key.

    Screen artifacts share a per-(model, effort) mutable sidecar + partial log
    (the ``budget_ratio`` is a ROW field, not part of the sidecar name -- two
    ratios of the same model+effort legally append to ONE sidecar). A live
    process holds an advisory ``flock`` on ``<sidecar-stem>.lock`` for its
    whole lifetime; a second process that finds it held raises this rather than
    racing. Two writers on one sidecar race the summary rewrite (last-writer
    -wins can leave the summary describing a SUBSET of the sidecar) and resume
    off a stale snapshot -- the cross-resume incident. THE RULE: one writer per
    (model, effort) sidecar. There is no retry -- refusing loudly is correct;
    wait for the standing holder to exit, then re-run.
    """

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


def ratio_tag(budget_ratio: float) -> str:
    """A filesystem-safe tag for a budget ratio (``0.25`` -> ``r025``).

    Folds the ratio into the artifact/partials names so the compression point
    (r=0.25, full 7 arms) and the fair-channel point (r=1.0, encdec-only) are
    DISTINCT files -- a per-(model, ratio) config never overwrites another.
    """
    return "r" + f"{budget_ratio:.2f}".replace(".", "")


def effort_suffix(reasoning_effort: str | None) -> str:
    """The artifact/phase suffix for a reasoning effort.

    The DEFAULT (provider-default, ``None``) effort keeps NO suffix -- so the
    existing default screens' names/phases are unchanged (byte-compatible with
    the in-flight default runs). A LABELED effort adds ``_e<effort>`` (e.g.
    ``_elow`` / ``_enone``), OUTPUT-AFFECTING and never colliding with default.
    """
    return "" if reasoning_effort is None else f"_e{reasoning_effort}"


def screen_stem(
    model: str, budget_ratio: float, reasoning_effort: str | None = None
) -> str:
    """The per-(model, ratio, effort) artifact stem.

    ``ed1_<model>_<ratio>[_e<effort>]`` -- the reasoning effort folds into the
    name (and the partials phase) so the default / low / none rounds are
    DISTINCT files. Default effort keeps the original ``_r<ratio>`` name.
    """
    return (
        f"ed1_{model_tag(model)}_{ratio_tag(budget_ratio)}"
        f"{effort_suffix(reasoning_effort)}"
    )


def _mean(values: list[float]) -> float | None:
    """Mean over the sampled values, or ``None`` when none were sampled."""
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    """Median over the sampled values, or ``None`` when none were sampled."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


# --- Per-key advisory lock (one writer per sidecar) --------------------------


def sidecar_lock_path(sidecar_path: Path) -> Path:
    """The advisory-lock path guarding a screen ``sidecar_path``.

    The lock lives beside the sidecar it protects, named for the SAME
    per-(model, effort) key: ``ed1_<tag><suffix>.outputs.jsonl`` ->
    ``ed1_<tag><suffix>.lock``. The ``budget_ratio`` is deliberately NOT in the
    name -- two ratios of one model+effort share the sidecar (ratio is a row
    field), so they MUST serialize on the same lock.
    """
    name = sidecar_path.name
    stem = name[: -len(".outputs.jsonl")] if name.endswith(
        ".outputs.jsonl"
    ) else sidecar_path.stem
    return sidecar_path.with_name(f"{stem}.lock")


@contextlib.contextmanager
def screen_key_lock(
    lock_path: Path,
    *,
    screen_key: str,
    events: EventStream | None = None,
    unit: EventUnit | None = None,
    marker_sink: Callable[[str], None] | None = None,
) -> Iterator[None]:
    """Hold an advisory ``flock`` on ``lock_path`` for the block's lifetime.

    Acquires ``fcntl.flock(LOCK_EX | LOCK_NB)`` -- a NON-blocking exclusive
    lock. If a live process already holds it, this does NOT wait or retry: it
    emits the loud ``SCREEN-KEY-LOCKED`` stderr marker (+ a
    :func:`screen_key_locked_event` when an ``events`` stream is available) and
    raises :class:`ScreenKeyLocked` -- one writer per key is an invariant, not
    a convention. On normal exit AND on any exception raised inside the block
    the lock is released and the file handle closed (flock also drops when the
    process dies, so a crash never leaks it).
    """
    sink = marker_sink or (lambda line: sys.stderr.write(line + "\n"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            marker = (
                f"SCREEN-KEY-LOCKED {screen_key} lock_path={lock_path} "
                f"-- one writer per (model, effort) sidecar; another process "
                f"holds this key. Wait for it to exit, then re-run."
            )
            with contextlib.suppress(Exception):
                sink(marker)
            if events is not None:
                with contextlib.suppress(Exception):
                    events.emit(
                        screen_key_locked_event(
                            unit=unit or EventUnit(screen_id=screen_key),
                            screen_key=screen_key,
                            lock_path=str(lock_path),
                        )
                    )
            handle.close()
            raise ScreenKeyLocked(
                f"screen key {screen_key!r} is already held (lock "
                f"{lock_path}); one writer per (model, effort) sidecar -- "
                f"wait for the holder to exit, then re-run."
            ) from exc
        yield
    finally:
        with contextlib.suppress(Exception):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(Exception):
            handle.close()


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
    #: Summed generation token counts for the row (direct = 1 call; encdec =
    #: enc+dec). Persisted so cost is reconstructable per model -- the honest
    #: spend record for the OpenAI-direct lane (whose billing is NOT the
    #: OpenRouter credits API) and a sanity check for every lane.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    #: Task-20 telemetry: summed reasoning tokens + summed wall-clock latency
    #: for the row (enc+dec for encdec). ``None`` when the provider exposed no
    #: reasoning detail -- never 0-conflated.
    reasoning_tokens: int | None = None
    latency_s: float | None = None
    #: Task-26 per-call provenance (``None`` when unknown): the provider stop
    #: reason of the accepted Generation (encdec = the decoder call) + the FULL
    #: typed diagnostic of a failed call.
    finish_reason: str | None = None
    provider_error: dict[str, object] | None = None


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
        from whetstone.execution.call_support import (
            call_telemetry,
            failure_code_of,
        )
        return _ScreenRowOutcome(
            passed=None, failed=True,
            failure_code=failure_code_of(result), output_text=None,
            provider_error=call_telemetry(result).provider_error,
        )
    code = result.generation.text
    from whetstone.execution.call_support import call_telemetry
    tel = call_telemetry(result)
    score = _score(scorer, code=code, ht=score_task)
    if score.infrastructure_unknown:
        return _ScreenRowOutcome(
            passed=None, failed=True,
            failure_code="code_eval_infrastructure_unknown",
            output_text=code,
            prompt_tokens=tel.prompt_tokens,
            completion_tokens=tel.completion_tokens,
            total_tokens=tel.total_tokens,
            reasoning_tokens=tel.reasoning_tokens, latency_s=tel.latency_s,
            finish_reason=tel.finish_reason,
        )
    return _ScreenRowOutcome(
        passed=score.passed, failed=False, failure_code="",
        output_text=code,
        prompt_tokens=tel.prompt_tokens,
        completion_tokens=tel.completion_tokens,
        total_tokens=tel.total_tokens,
        reasoning_tokens=tel.reasoning_tokens, latency_s=tel.latency_s,
        finish_reason=tel.finish_reason,
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
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        total_tokens=outcome.total_tokens,
        reasoning_tokens=outcome.reasoning_tokens,
        latency_s=outcome.latency_s,
        finish_reason=outcome.finish_reason,
        provider_error=outcome.provider_error,
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
    #: Task-20 per-arm telemetry samples for THIS task's rows (one entry per
    #: repeat that reported the field). Kept as lists so the report can compute
    #: mean/median + coverage counts over rows-with-field (never 0-conflated).
    arm_latencies: dict[str, list[float]] = field(default_factory=dict)
    arm_reasoning: dict[str, list[int]] = field(default_factory=dict)

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
            # Per-arm telemetry samples (task 20): total reasoning tokens +
            # latency sample count for this task (coverage-honest).
            "reasoning_tokens_by_arm": {
                arm: (sum(v) if v else None)
                for arm, v in self.arm_reasoning.items()
            },
            "latency_sample_count_by_arm": {
                arm: len(v) for arm, v in self.arm_latencies.items()
            },
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
    #: The task-model reasoning effort this screen ran at (``None`` = the
    #: provider default). OUTPUT-AFFECTING: folds into the artifact stem +
    #: partials phase, and the realized reasoning tokens (arm_summary) show
    #: whether the model HONORED it.
    reasoning_effort: str | None = None

    @property
    def excluded_task_ids(self) -> tuple[str, ...]:
        """The always-pass task ids (the model's pool-exclusion list)."""
        return tuple(r.task_id for r in self.rows if r.always_pass)

    def arm_summary(self) -> dict[str, dict[str, float | None]]:
        """Per-arm pass rate + telemetry (task 20), coverage-honest.

        Adds per-arm mean/median wall-clock latency + total & mean reasoning
        tokens, each computed ONLY over rows that REPORTED the field, with a
        ``*_coverage`` count (rows-with-field) so a reader never mistakes a
        partial-coverage aggregate (mixed pre/post-telemetry rows) for a full
        one. A field absent everywhere yields ``None`` (never a fake 0).
        """
        out: dict[str, dict[str, float | None]] = {}
        n_tasks = len(self.rows) or 1
        total_rows_denom = self.repeats * len(self.rows) or 1
        for arm in self.arms:
            total_pass = sum(r.pass_counts.get(arm, 0) for r in self.rows)
            full = sum(
                1 for r in self.rows
                if r.pass_counts.get(arm, 0) == self.repeats
            )
            lat: list[float] = []
            reason: list[int] = []
            for r in self.rows:
                lat.extend(r.arm_latencies.get(arm, []))
                reason.extend(r.arm_reasoning.get(arm, []))
            out[arm] = {
                "mean_pass_rate": total_pass / total_rows_denom,
                "tasks_full_pass": full,
                "tasks_full_pass_fraction": full / n_tasks,
                "mean_latency_s": _mean(lat),
                "median_latency_s": _median(lat),
                "latency_coverage": len(lat),
                "total_reasoning_tokens": (sum(reason) if reason else None),
                "mean_reasoning_tokens": _mean(
                    [float(x) for x in reason]
                ),
                "reasoning_coverage": len(reason),
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
                c = summary[canon]["mean_pass_rate"] or 0.0
                r = summary[renamed]["mean_pass_rate"] or 0.0
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
            "reasoning_effort": self.reasoning_effort,
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
        # Per-(model, ratio, effort) filename so the compression / fair-channel
        # / low / none rounds never overwrite each other.
        out_dir = root / "task_screen"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = screen_stem(
            self.model, self.budget_ratio, self.reasoning_effort
        )
        path = out_dir / f"{stem}.json"
        # ATOMIC rewrite: write to a temp file in the SAME directory, then
        # os.replace() it onto the final path. os.replace is atomic within a
        # filesystem, so a reader (or a crashed writer) never observes a torn /
        # partial summary -- either the whole old file or the whole new one.
        body = json.dumps(self.as_dict(), indent=2, sort_keys=True)
        tmp = out_dir / f".{stem}.json.tmp.{os.getpid()}"
        try:
            tmp.write_text(body)
            os.replace(tmp, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
        return path


# --- The screen driver -------------------------------------------------------


def run_task_screen(
    *,
    model: str,
    transport: TransportCall,
    execution_policy: ProviderExecutionPolicy,
    budget_ratio: float = 0.25,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
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
    lock_path: Path | None = None,
    events: EventStream | None = None,
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

    CONCURRENCY-SAFE: when a ``sidecar_path`` (or explicit ``lock_path``) is
    given, the screen holds an advisory ``flock`` on ``<sidecar>.lock`` for its
    whole lifetime and REFUSES to start (raising :class:`ScreenKeyLocked`) if a
    second process already holds it -- one writer per (model, effort) sidecar.
    """
    from whetstone.envs.ed1 import ED1_DATASET_REVISION, load_ed1_tasks

    # One-writer-per-sidecar invariant: the sidecar + partial log are keyed by
    # (model, effort) ONLY (budget_ratio is a row field, so two ratios share
    # the file). Hold an advisory flock on <sidecar>.lock for the whole screen
    # so a concurrent same-(model, effort) writer refuses to start rather than
    # racing the summary rewrite / resume snapshot (the cross-resume incident).
    resolved_lock = lock_path or (
        sidecar_lock_path(sidecar_path) if sidecar_path is not None else None
    )
    eff = effort_suffix(reasoning_effort)
    with contextlib.ExitStack() as _lock_stack:
        if resolved_lock is not None:
            _lock_stack.enter_context(
                screen_key_lock(
                    resolved_lock,
                    screen_key=f"screen:{model_tag(model)}{eff}",
                    events=events,
                    unit=EventUnit(
                        screen_id=f"screen:{model}{eff}", model=model,
                    ),
                )
            )
        return _run_task_screen_locked(
            model=model,
            transport=transport,
            execution_policy=execution_policy,
            budget_ratio=budget_ratio,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            repeats=repeats,
            variants=variants,
            rename_token=rename_token,
            tasks=tasks,
            limit=limit,
            prefer_snapshot=prefer_snapshot,
            concurrency=concurrency,
            scorer=scorer,
            partial_log=partial_log,
            sidecar_path=sidecar_path,
            events=events,
            _load_ed1_tasks=load_ed1_tasks,
            _dataset_revision=ED1_DATASET_REVISION,
        )


def _run_task_screen_locked(
    *,
    model: str,
    transport: TransportCall,
    execution_policy: ProviderExecutionPolicy,
    budget_ratio: float,
    reasoning_effort: str | None,
    temperature: float | None,
    repeats: int,
    variants: Sequence[str] | None,
    rename_token: str,
    tasks: Sequence[Ed1Instance] | None,
    limit: int | None,
    prefer_snapshot: bool,
    concurrency: int,
    scorer: Callable[..., CodeScore] | None,
    partial_log: PartialLog | None,
    sidecar_path: Path | None,
    events: EventStream | None,
    _load_ed1_tasks: Callable[..., Sequence[Ed1Instance]],
    _dataset_revision: str,
) -> TaskScreenReport:
    """The screen body, run under the held per-(model, effort) lock.

    Byte-identical to the pre-lock single-process behavior -- the lock is a
    concurrency guard only, it does not touch keying / identity / artifacts.
    """
    load_ed1_tasks = _load_ed1_tasks
    ED1_DATASET_REVISION = _dataset_revision

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

    restored = _restore_screen(
        partial_log, model, budget_ratio, reasoning_effort
    )
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
                partial_log, sidecar_path, model=model,
                budget_ratio=budget_ratio,
                reasoning_effort=reasoning_effort, temperature=temperature,
                task_id=task_id, arm=arm, index=index, outcome=out,
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
        arm_lat: dict[str, list[float]] = {}
        arm_reason: dict[str, list[int]] = {}
        for arm in arms:
            p = f = 0
            lat: list[float] = []
            reason: list[int] = []
            for index in range(repeats):
                out = driven[(task_id, arm, index)]
                if out.passed:
                    p += 1
                elif out.failed or out.passed is None:
                    f += 1
                # Coverage-honest: sample only rows that reported the field.
                if out.latency_s is not None:
                    lat.append(out.latency_s)
                if out.reasoning_tokens is not None:
                    reason.append(out.reasoning_tokens)
            pass_counts[arm] = p
            fail_counts[arm] = f
            arm_lat[arm] = lat
            arm_reason[arm] = reason
        rows.append(TaskScreenRow(
            task_id=task_id,
            arm_latencies=arm_lat,
            arm_reasoning=arm_reason,
            entry_point=inst.humaneval_task.entry_point,
            pass_counts=pass_counts, fail_counts=fail_counts,
            repeats=repeats,
        ))
    # Push run telemetry (task 24) over the rows DRIVEN this pass (restored
    # rows are not re-driven, so they are not part of this window). A
    # rate_limit_pressure event fires when the window saw any 429/rate-limit
    # rows; a latency_snapshot pushes the window's median call latency (null
    # when no row reported one). Keyed by a screen-level id.
    if events is not None:
        newly = [out for key, out in driven.items() if key not in restored]
        rate_limit_rows = sum(
            1 for out in newly if is_rate_limit_code(out.failure_code)
        )
        guard_timeouts = sum(
            1 for out in newly if out.failure_code == "runner_timeout"
        )
        latencies = [
            out.latency_s for out in newly if out.latency_s is not None
        ]
        unit = EventUnit(
            screen_id=f"screen:{model}:r{budget_ratio}",
            lane=None, model=model,
        )
        if rate_limit_rows or guard_timeouts:
            events.emit(
                rate_limit_pressure_event(
                    unit=unit, rate_limit_rows=rate_limit_rows,
                    concurrency_halved=False, guard_timeouts=guard_timeouts,
                    window_label="screen",
                )
            )
        events.emit(
            latency_snapshot_event(
                unit=unit, median_latency_s=_median(latencies),
                coverage=len(latencies), window_label="screen",
            )
        )
    return TaskScreenReport(
        model=model, budget_ratio=budget_ratio, repeats=repeats,
        arms=tuple(arms), rename_token=rename_token,
        dataset_revision=ED1_DATASET_REVISION,
        name_only_wrapper=NAME_ONLY_WRAPPER, rows=tuple(rows),
        reasoning_effort=reasoning_effort,
    )


def _screen_phase(
    model: str, budget_ratio: float, arm: str,
    reasoning_effort: str | None = None,
) -> str:
    """The partials phase key: per (model, ratio, effort, arm).

    The RATIO and the reasoning EFFORT are in the key so a fair-channel r=1.0
    run (or a low/none-effort round) does NOT restore-skip against a different
    config's rows -- a different ratio OR effort is a different rollout that
    must be re-driven, not resumed. Default effort keeps the original key.
    """
    return (
        f"screen:{model_tag(model)}:{ratio_tag(budget_ratio)}"
        f"{effort_suffix(reasoning_effort)}:{arm}"
    )


def _persist_row(
    partial_log: PartialLog | None,
    sidecar_path: Path | None,
    *,
    model: str,
    budget_ratio: float,
    reasoning_effort: str | None,
    temperature: float | None = None,
    task_id: str,
    arm: str,
    index: int,
    outcome: _ScreenRowOutcome,
) -> None:
    """Append one completed screen row to the partial log + output sidecar."""
    if partial_log is not None:
        partial_log.append(PartialCallRecord(
            phase=_screen_phase(model, budget_ratio, arm, reasoning_effort),
            instance_id=task_id, unit=arm, repeat_id=index,
            score=(None if outcome.passed is None else float(outcome.passed)),
            failed=outcome.failed, failure_code=outcome.failure_code,
            split_role=arm,
            prompt_tokens=outcome.prompt_tokens,
            completion_tokens=outcome.completion_tokens,
            total_tokens=outcome.total_tokens,
            reasoning_tokens=outcome.reasoning_tokens,
            latency_s=outcome.latency_s,
            output_text=outcome.output_text,
            finish_reason=outcome.finish_reason,
            provider_error=outcome.provider_error,
        ))
    if sidecar_path is not None:
        with sidecar_path.open("a") as handle:
            handle.write(json.dumps({
                # Versioned schema stamp + structured id components (task 26):
                # a consumer branches on the version and joins on the id fields
                # (model / arm / budget_ratio / reasoning_effort) directly.
                "schema": SCREEN_SCHEMA,
                "model": model, "budget_ratio": budget_ratio,
                "reasoning_effort": reasoning_effort,
                "temperature": temperature,
                "task_id": task_id, "arm": arm,
                "repeat": index, "passed": outcome.passed,
                "failure_code": outcome.failure_code,
                "prompt_tokens": outcome.prompt_tokens,
                "completion_tokens": outcome.completion_tokens,
                "total_tokens": outcome.total_tokens,
                "reasoning_tokens": outcome.reasoning_tokens,
                "latency_s": outcome.latency_s,
                "output_text": outcome.output_text,
                # Per-call provenance (task 26): truncation vs clean stop, and
                # the full provider diagnostic on a failed row.
                "finish_reason": outcome.finish_reason,
                "provider_error": outcome.provider_error,
            }) + "\n")


def _restore_screen(
    partial_log: PartialLog | None, model: str, budget_ratio: float,
    reasoning_effort: str | None = None,
) -> dict[tuple[str, str, int], _ScreenRowOutcome]:
    """Rebuild screen rows already recorded (resume skip) by (task,arm,r).

    Matches ONLY this (model, ratio, effort) phase, so a fair-channel r=1.0 run
    (or a low/none-effort round) never restores another config's rows. Restored
    rows re-hydrate their token counts (for the cost sums); a pre-telemetry
    recorded row simply carries ``None`` tokens (coverage-honest, not 0).
    """
    if partial_log is None:
        return {}
    prefix = (
        f"screen:{model_tag(model)}:{ratio_tag(budget_ratio)}"
        f"{effort_suffix(reasoning_effort)}:"
    )
    restored: dict[tuple[str, str, int], _ScreenRowOutcome] = {}
    for rec in partial_log.load():
        if not rec.phase.startswith(prefix):
            continue
        restored[(rec.instance_id, rec.unit, rec.repeat_id)] = (
            _ScreenRowOutcome(
                passed=(None if rec.score is None else bool(rec.score)),
                failed=rec.failed, failure_code=rec.failure_code,
                output_text=None,
                prompt_tokens=rec.prompt_tokens,
                completion_tokens=rec.completion_tokens,
                total_tokens=rec.total_tokens,
                reasoning_tokens=rec.reasoning_tokens,
                latency_s=rec.latency_s,
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


def _config_key(report: TaskScreenReport) -> str:
    """The cross-model config key: ``<model>@<ratio>[/<effort>]``."""
    base = f"{report.model}@{ratio_tag(report.budget_ratio)}"
    if report.reasoning_effort is not None:
        base += f"/{report.reasoning_effort}"
    return base


def _mean_reasoning(report: TaskScreenReport) -> float | None:
    """The report's mean realized reasoning tokens over rows-with-field."""
    summary = report.arm_summary()
    vals: list[float] = []
    weights = 0
    for arm in report.arms:
        m = summary[arm]["mean_reasoning_tokens"]
        cov = summary[arm]["reasoning_coverage"] or 0
        if m is not None and cov:
            vals.append(m * cov)
            weights += int(cov)
    return (sum(vals) / weights) if weights else None


def reasoning_honored_flags(
    reports: Sequence[TaskScreenReport],
) -> dict[str, dict[str, object]]:
    """Per (model, ratio, effort) whether the model HONORED the effort setting.

    Compares a labeled-effort report's mean realized reasoning tokens against
    the SAME (model, ratio) DEFAULT report's. If a ``low``/``none`` round's
    reasoning tokens ~match the default's (within 5%), the model IGNORED the
    setting -- its labeled rows are DUPLICATES of default and must be flagged
    (not presented as a distinct condition). ``honored=None`` when there is no
    default to compare against, or neither reported reasoning tokens.
    """
    # Index the DEFAULT-effort report's mean reasoning per (model, ratio).
    default_reason: dict[tuple[str, float], float | None] = {}
    for r in reports:
        if r.reasoning_effort is None:
            default_reason[(r.model, r.budget_ratio)] = _mean_reasoning(r)
    out: dict[str, dict[str, object]] = {}
    for r in reports:
        if r.reasoning_effort is None:
            continue
        mine = _mean_reasoning(r)
        base = default_reason.get((r.model, r.budget_ratio))
        honored: bool | None = None
        if mine is not None and base is not None:
            # Honored iff the realized reasoning moved from default by >5%.
            denom = base if base else 1.0
            honored = abs(mine - base) / denom > 0.05
        out[_config_key(r)] = {
            "reasoning_effort": r.reasoning_effort,
            "mean_reasoning_tokens": mine,
            "default_mean_reasoning_tokens": base,
            "honored": honored,
            "note": (
                "IGNORED: labeled rows duplicate the default condition"
                if honored is False else None
            ),
        }
    return out


def cross_model_summary(
    reports: Sequence[TaskScreenReport],
) -> dict[str, object]:
    """A cross-model table, keyed per (model, arm, ratio, reasoning-effort).

    Headed for the paper's contamination + cost/latency + reasoning section.
    Each row is ONE (model, arm, ratio, effort) cell -- so the two-ratio encdec
    structure AND the reasoning-effort rounds (default / low / none) are all
    distinct rows, never merged. Columns: pass rate, tasks-full-pass, AND the
    task-20 telemetry (mean/median latency + reasoning tokens) + coverage
    counts. ``reasoning_honored`` flags labeled-effort configs whose realized
    reasoning tokens ~match the default (the model IGNORED the setting).
    """
    table: list[dict[str, object]] = []
    for report in reports:
        summary = report.arm_summary()
        for arm in report.arms:
            s = summary[arm]
            table.append({
                "model": report.model,
                "arm": arm,
                "budget_ratio": report.budget_ratio,
                "reasoning_effort": report.reasoning_effort,
                "config_key": _config_key(report),
                "mean_pass_rate": s["mean_pass_rate"],
                "tasks_full_pass": s["tasks_full_pass"],
                "mean_latency_s": s["mean_latency_s"],
                "median_latency_s": s["median_latency_s"],
                "latency_coverage": s["latency_coverage"],
                "total_reasoning_tokens": s["total_reasoning_tokens"],
                "reasoning_coverage": s["reasoning_coverage"],
            })
    return {
        "schema": f"{SCREEN_SCHEMA}.cross_model",
        "arms": list(SCREEN_ARMS),
        "table": table,
        # The paper's causal-memorization figure: canonical-minus-renamed pass
        # delta per config (model, ratio, effort) per arm-pair.
        "rename_deltas_by_config": {
            _config_key(r): r.rename_deltas() for r in reports
        },
        "per_config_excluded": {
            _config_key(r): len(r.excluded_task_ids) for r in reports
        },
        # Reasoning-effort honor-vs-ignore per labeled config (task 21.3).
        "reasoning_honored": reasoning_honored_flags(reports),
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
    "ScreenKeyLocked",
    "TaskScreenReport",
    "TaskScreenRow",
    "cross_model_summary",
    "effort_suffix",
    "load_exclusion_ids",
    "model_tag",
    "ratio_tag",
    "reasoning_honored_flags",
    "rename_identifier",
    "renamed_task",
    "run_task_screen",
    "screen_key_lock",
    "screen_stem",
    "sidecar_lock_path",
    "split_prompt",
]
