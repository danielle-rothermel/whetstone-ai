#!/usr/bin/env python3
"""Preview code-comp baselines with rich output.

Automated pytest coverage for the same flow lives in
``scripts/ci/preview-anchor-pathway.sh``.
"""

from __future__ import annotations

import argparse
import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from dr_providers import ProviderKind, policy_for
from dr_store import ObjectStore, SqliteBackend
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from whetstone.envs.code_comp.constants import CODE_COMP_CANONICAL_MODEL
from whetstone.envs.code_comp.dataset import CodeCompTaskInstance, load_tasks
from whetstone.envs.code_comp.modes.encdec import (
    EncDecTaskModelConfig,
    EncDecTaskModelKind,
    encdec_task_model_from_metadata,
)
from whetstone.envs.code_comp.preview import (
    run_code_comp_anchor_baseline_sweep,
)
from whetstone.envs.code_comp.registry import CodeCompMode
from whetstone.envs.code_comp.runtime import (
    EncDecScoringRuntimeSummary,
    build_code_comp_scoring_runtime,
)
from whetstone.envs.code_comp.scoring import (
    CODE_COMP_SCORING_PREFLIGHT_TASK_ID,
    CheckpointedCodeBatchScorer,
)
from whetstone.envs.task_pools import (
    select_lowest_historical_pass_rate_for_env,
    select_role_for_env,
)
from whetstone.evaluation.analysis.power import (
    DEFAULT_SAMPLE_CAP,
    PowerConfig,
)
from whetstone.evaluation.preview.anchor import (
    AnchorArmPreview,
    BaselinePreviewTranscript,
    BaselineSweepTranscript,
)
from whetstone.experiment.graph.nodes import PROVIDER_GENERATION_OUTPUT_FIELD
from whetstone.experiment.task_selection import (
    TaskRoleSelection,
    TaskSplitManifestError,
    TaskSplitRole,
    load_task_split_manifest,
)
from whetstone.provider.policy import ProviderExecutionPolicy
from whetstone.runner.routes import canonical_task_route

_DEFAULT_OUTPUT = Path("artifacts/ed1-baseline-preview")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the two hand-engineered ED1 encoder prompts on one "
            "shared binding, then show the paired bootstrap and power "
            "estimate."
        )
    )
    parser.add_argument("--evaluation-python", required=True, type=Path)
    parser.add_argument("--snapshot-path", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="explicit task ID; repeat to preserve an exact ordered selection",
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        help="frozen task-selection manifest",
    )
    parser.add_argument(
        "--task-role",
        choices=tuple(role.value for role in TaskSplitRole),
        help="ordered manifest role to evaluate",
    )
    parser.add_argument(
        "--worst-task-count",
        type=_positive_int,
        help=(
            "select this many lowest historical pass-rate tasks from the "
            "manifest role"
        ),
    )
    parser.add_argument(
        "--exclude-task-id",
        action="append",
        dest="excluded_task_ids",
        default=[],
        help="manifest task to exclude before worst-task ranking; repeatable",
    )
    parser.add_argument(
        "--task-count",
        type=_positive_int,
        default=2,
        help="temporary first-N preview when no --task-id is supplied",
    )
    parser.add_argument("--pool-ceiling", type=_positive_int)
    parser.add_argument("--repeats", type=_positive_int, default=2)
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=8,
        help="maximum task/sample rows evaluated concurrently",
    )
    parser.add_argument(
        "--budget-mode",
        choices=("both", "unbudgeted", "budgeted"),
        default="both",
    )
    parser.add_argument(
        "--budget-ratio",
        type=float,
        default=0.5,
        help="character-budget ratio used by the budgeted mode",
    )
    parser.add_argument(
        "--task-model",
        choices=("dummy", "provider"),
        default="dummy",
    )
    parser.add_argument("--provider-model", default=CODE_COMP_CANONICAL_MODEL)
    parser.add_argument(
        "--bootstrap-resamples",
        type=_positive_int,
        default=10_000,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--power-sample-cap",
        type=_positive_int,
        default=DEFAULT_SAMPLE_CAP,
    )
    return parser.parse_args()


def _tasks_by_id(
    pool: tuple[CodeCompTaskInstance, ...], task_ids: tuple[str, ...]
) -> tuple[CodeCompTaskInstance, ...]:
    if not task_ids:
        raise SystemExit("task selection must contain at least one task")
    by_id = {task.humaneval_task.task_id: task for task in pool}
    missing = tuple(task_id for task_id in task_ids if task_id not in by_id)
    if missing:
        raise SystemExit(
            f"selected task IDs are absent from snapshot: {missing}"
        )
    if len(set(task_ids)) != len(task_ids):
        raise SystemExit("selected task IDs must be unique")
    return tuple(by_id[task_id] for task_id in task_ids)


def _select_tasks(
    args: argparse.Namespace,
) -> tuple[
    tuple[CodeCompTaskInstance, ...],
    TaskRoleSelection | None,
    CodeCompTaskInstance,
]:
    has_manifest = args.task_manifest is not None
    has_role = args.task_role is not None
    if has_manifest != has_role:
        raise SystemExit(
            "--task-manifest and --task-role must be supplied together"
        )
    if has_manifest and args.task_ids:
        raise SystemExit("--task-manifest cannot be combined with --task-id")
    if args.worst_task_count is not None and not has_manifest:
        raise SystemExit(
            "--worst-task-count requires --task-manifest and --task-role"
        )
    if args.excluded_task_ids and args.worst_task_count is None:
        raise SystemExit("--exclude-task-id requires --worst-task-count")
    if has_manifest:
        try:
            manifest = load_task_split_manifest(args.task_manifest)
            role = TaskSplitRole(args.task_role)
            selection = (
                select_role_for_env(
                    manifest,
                    env="code_comp",
                    mode=CodeCompMode.ENCDEC,
                    role=role,
                )
                if args.worst_task_count is None
                else select_lowest_historical_pass_rate_for_env(
                    manifest,
                    env="code_comp",
                    mode=CodeCompMode.ENCDEC,
                    role=role,
                    count=args.worst_task_count,
                    excluded_task_ids=tuple(args.excluded_task_ids),
                )
            )
        except TaskSplitManifestError as exc:
            raise SystemExit(str(exc)) from None
        pool = load_tasks(snapshot_path=args.snapshot_path)
        return (
            _tasks_by_id(pool, selection.task_ids),
            selection,
            _tasks_by_id(pool, (CODE_COMP_SCORING_PREFLIGHT_TASK_ID,))[0],
        )
    if args.task_ids:
        pool = load_tasks(snapshot_path=args.snapshot_path)
        task_ids = tuple(args.task_ids)
        return (
            _tasks_by_id(pool, task_ids),
            None,
            _tasks_by_id(pool, (CODE_COMP_SCORING_PREFLIGHT_TASK_ID,))[0],
        )
    selected = load_tasks(
        snapshot_path=args.snapshot_path,
        limit=args.task_count,
    )
    return selected, None, selected[0]


def _task_model(
    args: argparse.Namespace,
) -> EncDecTaskModelConfig:
    kind = EncDecTaskModelKind(args.task_model)
    route = canonical_task_route(
        model=args.provider_model,
        temperature=None,
        max_attempts=1,
    )
    if kind is EncDecTaskModelKind.PROVIDER:
        return EncDecTaskModelConfig(
            kind=kind,
            provider_call_config=route.call_config,
            execution_policy=route.execution_policy,
        )
    return EncDecTaskModelConfig(
        kind=kind,
        provider_call_config=route.call_config,
        execution_policy=ProviderExecutionPolicy(
            transport_policy=policy_for(
                ProviderKind.OPENROUTER,
                api_key_env="WHETSTONE_DUMMY_PROVIDER_KEY",
                base_url="https://example.invalid/v1",
                native_retry_count=0,
            ),
            max_attempts=1,
        ),
    )


def _write_transcript(
    transcript: BaselineSweepTranscript,
    path: Path,
) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        transcript.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _step_text(step, field: str) -> str:
    value = step.inputs.get(field)
    if value is None:
        value = step.outputs.get(field)
    if not isinstance(value, str):
        raise ValueError(f"component step {field!r} is not text")
    return value


def _render_arm(console: Console, arm: AnchorArmPreview) -> None:
    console.rule(f"[bold blue]{arm.label}")
    console.print(Panel(Text(arm.instruction), title="Encoder instruction"))
    reward = arm.evidence.reward_ref
    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column("field", style="bold cyan")
    summary.add_column("value")
    summary.add_row(
        "mean blended reward",
        "None" if reward is None else f"{reward.record.value:.6f}",
    )
    summary.add_row("planned rows", str(arm.evidence.row_accounting.planned))
    summary.add_row("present rows", str(arm.evidence.row_accounting.present))
    console.print(summary)
    per_task = Table("task", "repeats", "blended reward", box=None)
    for task_id, count, value in zip(
        arm.evidence.task_hashes,
        arm.evidence.per_task_counts,
        arm.evidence.per_task_values,
        strict=True,
    ):
        per_task.add_row(task_id, str(count), f"{value:.6f}")
    console.print(Panel(per_task, title="Aligned per-task values"))

    first = arm.component_traces.rows[0].executed_component_trace
    for step in first.executed_component_steps:
        prompt = _step_text(step, "prompt")
        generation = _step_text(step, PROVIDER_GENERATION_OUTPUT_FIELD)
        console.print(
            Panel(Text(prompt), title=f"{step.component_id}: model input")
        )
        rendered = (
            Syntax(generation, "python", word_wrap=True)
            if step.component_id == "decode"
            else Text(generation)
        )
        console.print(
            Panel(rendered, title=f"{step.component_id}: model output")
        )


def _render_preview(
    console: Console,
    transcript: BaselinePreviewTranscript,
) -> None:
    task_model = encdec_task_model_from_metadata(transcript.metadata)
    budget = (
        "unbudgeted"
        if transcript.budget_ratio is None
        else f"budget ratio {transcript.budget_ratio}"
    )
    console.rule(f"[bold blue]{budget}")
    config = Table(show_header=False, box=None, pad_edge=False)
    config.add_column("field", style="bold cyan")
    config.add_column("value")
    config.add_row("task selection", ", ".join(transcript.task_ids))
    config.add_row("pool ceiling", str(transcript.pool_ceiling))
    config.add_row("row concurrency", str(transcript.concurrency))
    config.add_row("task model mode", task_model.kind.value)
    config.add_row("task model", task_model.model)
    config.add_row(
        "evaluation binding", transcript.evaluation_binding.identity_hash()
    )
    config.add_row(
        "preflight",
        f"{transcript.preflight.task_id}: {transcript.preflight.outcome}",
    )
    console.print(Panel(config, title="Exact preview configuration"))

    _render_arm(console, transcript.baseline)
    _render_arm(console, transcript.ceiling)

    ci = transcript.paired_delta_ci
    power = transcript.power
    best_mdd = min(point.mdd_at_target for point in power.surface)
    analysis = Table(show_header=False, box=None, pad_edge=False)
    analysis.add_column("field", style="bold magenta")
    analysis.add_column("value")
    analysis.add_row("comparison - baseline", f"{ci.point:.6f}")
    analysis.add_row(
        f"{ci.level:.0%} paired bootstrap CI",
        f"[{ci.low:.6f}, {ci.high:.6f}]",
    )
    analysis.add_row("bootstrap resamples", str(ci.resamples))
    analysis.add_row("certified headroom", f"{power.certified_headroom:.6f}")
    analysis.add_row("power target gap", f"{power.target_gap:.6f}")
    analysis.add_row("best achievable MDD on surface", f"{best_mdd:.6f}")
    analysis.add_row("noise verdict", power.decomposition.noise_verdict)
    console.print(Panel(analysis, title="Paired signal and power estimate"))


def render_transcript(
    console: Console,
    transcript: BaselineSweepTranscript,
    *,
    output_dir: Path,
) -> None:
    console.print(
        Panel.fit(
            "Two hand-engineered prompts → both budget modes → identical "
            "per-mode EvaluationEngine bindings → aligned per-task rewards "
            "→ paired bootstrap and power estimates",
            title="ED1 baseline calibration sweep",
            border_style="bright_blue",
        )
    )
    selection = Table(show_header=False, box=None, pad_edge=False)
    selection.add_column("field", style="bold cyan")
    selection.add_column("value")
    if transcript.task_selection is None:
        selection.add_row("selection source", "explicit IDs / preview prefix")
    else:
        selected = transcript.task_selection
        selection.add_row("manifest role", selected.role.value)
        selection.add_row("manifest pool", selected.pool_key)
        selection.add_row("manifest hash", selected.manifest_content_hash)
        selection.add_row("selection method", selected.selection_method.value)
        if selected.source_role_count is not None:
            selection.add_row(
                "source role count", str(selected.source_role_count)
            )
        if selected.eligible_pool_count is not None:
            selection.add_row(
                "eligible pool count", str(selected.eligible_pool_count)
            )
        if selected.excluded_task_ids:
            selection.add_row(
                "excluded tasks", ", ".join(selected.excluded_task_ids)
            )
    selection.add_row("task count", str(len(transcript.task_ids)))
    selection.add_row("task order", ", ".join(transcript.task_ids))
    console.print(Panel(selection, title="Frozen task selection"))
    if (
        transcript.task_selection is not None
        and transcript.task_selection.historical_pass_rates
    ):
        rates = Table("task", "historical pass rate", box=None)
        for task_id, pass_rate in zip(
            transcript.task_ids,
            transcript.task_selection.historical_pass_rates,
            strict=True,
        ):
            rates.add_row(task_id, f"{pass_rate:.6f}")
        console.print(Panel(rates, title="Difficulty-ranked probe tasks"))
    for preview in transcript.previews:
        _render_preview(console, preview)
    console.print(
        Panel(
            "\n".join(
                (
                    f"Transcript: {output_dir / 'transcript.json'}",
                    f"Object store: {output_dir / 'objects.sqlite3'}",
                    "Execution cache: "
                    f"{output_dir / 'execution-cache.sqlite3'}",
                    f"Execution records: {output_dir / 'execution-records'}",
                )
            ),
            title="Durable outputs",
            border_style="green",
        )
    )


def _budget_ratios(args: argparse.Namespace) -> tuple[float | None, ...]:
    if args.budget_mode == "unbudgeted":
        return (None,)
    if args.budget_mode == "budgeted":
        return (args.budget_ratio,)
    return (None, args.budget_ratio)


def _render_launch_plan(
    console: Console,
    *,
    tasks: tuple[CodeCompTaskInstance, ...],
    task_selection: TaskRoleSelection | None,
    task_model: EncDecTaskModelConfig,
    budget_ratios: tuple[float | None, ...],
    num_samples: int,
    concurrency: int,
    pool_ceiling: int,
    output_dir: Path,
) -> None:
    task_model_calls = len(tasks) * num_samples * 2 * len(budget_ratios) * 2
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("field", style="bold cyan")
    table.add_column("value")
    table.add_row("task-model route", task_model.kind.value)
    table.add_row("task model", task_model.model)
    table.add_row("selected tasks", str(len(tasks)))
    table.add_row("num samples", str(num_samples))
    table.add_row("row concurrency", str(concurrency))
    table.add_row("budget modes", str(len(budget_ratios)))
    table.add_row("baseline prompts", "2")
    table.add_row("planned task-model calls", str(task_model_calls))
    table.add_row("power-estimate pool ceiling", str(pool_ceiling))
    table.add_row("output directory", str(output_dir))
    if task_selection is not None:
        table.add_row("task IDs", ", ".join(task_selection.task_ids))
    console.print(
        Panel(
            table,
            title="Baseline power probe launch plan",
            border_style="yellow",
        )
    )


def main() -> None:
    launched_at = perf_counter()
    args = _parse_args()
    if os.environ.get("DR_CODE_DISPOSABLE_WORKER") != "1":
        raise SystemExit(
            "refusing to execute generated code: set "
            "DR_CODE_DISPOSABLE_WORKER=1 in a disposable worker environment"
        )
    output_dir = args.output_dir.expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, task_selection, preflight_task = _select_tasks(args)
    task_ids = tuple(task.humaneval_task.task_id for task in tasks)
    pool_ceiling = args.pool_ceiling or (
        task_selection.eligible_pool_count
        if task_selection is not None
        and task_selection.eligible_pool_count is not None
        else len(tasks)
    )
    task_model = _task_model(args)
    budget_ratios = _budget_ratios(args)
    console = Console()

    def log_progress(message: str) -> None:
        elapsed = perf_counter() - launched_at
        console.print(
            Text.assemble(
                (f"[{elapsed:7.1f}s]", "dim cyan"),
                " ",
                message,
            )
        )
        console.file.flush()

    _render_launch_plan(
        console,
        tasks=tasks,
        task_selection=task_selection,
        task_model=task_model,
        budget_ratios=budget_ratios,
        num_samples=args.repeats,
        concurrency=args.concurrency,
        pool_ceiling=pool_ceiling,
        output_dir=output_dir,
    )
    runtime = build_code_comp_scoring_runtime(
        runtime_executable=args.evaluation_python,
        record_root=output_dir / "execution-records",
    )
    runtime_summary = EncDecScoringRuntimeSummary(
        evaluation_python=runtime.probe.python_executable,
        dr_code_version=version("dr-code"),
        runtime_hash=runtime.runtime_hash,
        probe=runtime.probe,
    )
    store = ObjectStore(SqliteBackend(output_dir / "objects.sqlite3"))
    with CheckpointedCodeBatchScorer(
        output_dir / "execution-cache.sqlite3",
        runtime_document=runtime.runtime_document,
        executor=runtime.executor,
    ) as scorer:
        transcript = run_code_comp_anchor_baseline_sweep(
            store=store,
            tasks=tasks,
            task_ids=task_ids,
            task_selection=task_selection,
            preflight_task=preflight_task,
            pool_ceiling=pool_ceiling,
            task_model=task_model,
            batch_scorer=scorer,
            runtime=runtime_summary,
            budget_ratios=budget_ratios,
            concurrency=args.concurrency,
            num_samples=args.repeats,
            power_config=PowerConfig(
                sample_cap=args.power_sample_cap,
            ),
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            log=log_progress,
        )
    _write_transcript(transcript, output_dir / "transcript.json")
    render_transcript(console, transcript, output_dir=output_dir)


if __name__ == "__main__":
    main()
