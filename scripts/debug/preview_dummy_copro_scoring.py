#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import resource
import sys
from importlib.metadata import version
from pathlib import Path

from dr_exec import (
    DirectoryRunStore,
    IsolatedHostPythonRuntime,
    ProcessExecutor,
)
from dr_providers import ProviderKind, policy_for
from dr_store import ObjectStore, SqliteBackend
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from whetstone.envs.ed1 import (
    ED1_CANONICAL_MODEL,
    Ed1Instance,
    load_ed1_tasks,
)
from whetstone.envs.ed1_blended import compression_score
from whetstone.envs.ed1_runtime import build_ed1_scoring_runtime
from whetstone.envs.ed1_scoring import CheckpointedCodeBatchScorer
from whetstone.envs.task_selection import (
    TaskRoleSelection,
    TaskSplitManifestError,
    TaskSplitRole,
    load_task_split_manifest,
)
from whetstone.experiment.graph.nodes import GENERATION_OUTPUT_FIELD
from whetstone.optimization.codex.proposer import (
    CodexCliProposerConfig,
    CodexCliProposerTransport,
)
from whetstone.optimization.copro.ed1_dry_run import (
    DummyCoproProposerConfig,
    DummyCoproProposerTransport,
    Ed1CoproProposalCall,
    Ed1CoproRoundAttempt,
    Ed1CoproSweepRanges,
)
from whetstone.optimization.copro.ed1_scoring_preview import (
    ED1_SCORING_PREFLIGHT_TASK_ID,
    Ed1CoproCandidateProgress,
    Ed1CoproRoundFailure,
    Ed1CoproScoredRound,
    Ed1CoproScoringPoint,
    Ed1CoproScoringTranscript,
    Ed1ScoredCandidate,
    Ed1ScoringRuntimeSummary,
    run_ed1_copro_scoring_preview,
)
from whetstone.optimization.copro.ed1_scoring_preview_worker import (
    DUMMY_ALTERNATE_PASSING_BODY,
    DUMMY_FAILING_BODY,
    DUMMY_PASSING_BODY,
)
from whetstone.optimization.copro.ed1_task_model import (
    Ed1TaskModelConfig,
    Ed1TaskModelKind,
)
from whetstone.optimization.proposal.mutation import MUTATION_FIELD
from whetstone.optimization.proposal.prompts import (
    COPRO_INSTRUCTION_HISTORY_KEY,
)
from whetstone.optimization.proposal.proposer import (
    ProposerRouteConfig,
    ProposerTransport,
)
from whetstone.provider.policy import ProviderExecutionPolicy
from whetstone.runner.routes import canonical_task_route

_DEFAULT_OUTPUT = Path("artifacts/copro-scoring-preview")
_MINIMUM_OPEN_FILE_LIMIT = 4096
_OPEN_FILES_PER_WORKER = 64


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview dummy or Codex CLI COPRO proposals through deterministic "
            "or provider-backed task-model generation, real HumanEval "
            "execution, reward calculation, and lifecycle folding."
        )
    )
    parser.add_argument(
        "--evaluation-python",
        required=True,
        type=Path,
        help="copied Python executable for candidate-code evaluation",
    )
    parser.add_argument(
        "--snapshot-path",
        required=True,
        type=Path,
        help="HumanEval+ JSON snapshot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"durable debug artifacts (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument("--task-count", type=_positive_int, default=1)
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
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=8,
        help="maximum task/repeat rows evaluated concurrently",
    )
    parser.add_argument(
        "--task-model",
        choices=("dummy", "provider"),
        default="dummy",
        help="encoder/decoder generation route",
    )
    parser.add_argument(
        "--provider-model",
        default=ED1_CANONICAL_MODEL,
        help="OpenRouter model used when --task-model=provider",
    )
    parser.add_argument(
        "--proposer",
        choices=("dummy", "codex"),
        default="dummy",
    )
    parser.add_argument(
        "--breadth",
        type=_positive_int,
        default=3,
        help="requested proposals per COPRO round",
    )
    parser.add_argument("--depth", type=_positive_int, default=1)
    parser.add_argument(
        "--budget-mode",
        choices=("both", "unbudgeted", "budgeted"),
        default="both",
    )
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--codex-timeout-seconds",
        type=_positive_float,
        default=600.0,
    )
    parser.add_argument(
        "--codex-records",
        type=Path,
        help=(
            "durable dr-exec records for Codex calls "
            "(default: OUTPUT_DIR/codex-proposer-records)"
        ),
    )
    return parser.parse_args()


def _write_transcript(
    transcript: Ed1CoproScoringTranscript, path: Path
) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        transcript.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _ensure_open_file_limit(worker_count: int) -> int:
    required = max(
        _MINIMUM_OPEN_FILE_LIMIT,
        worker_count * _OPEN_FILES_PER_WORKER,
    )
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= required:
        return soft
    if hard != resource.RLIM_INFINITY and hard < required:
        raise SystemExit(
            f"open-file soft limit {soft} is below the required {required}, "
            f"and the hard limit {hard} prevents raising it; run "
            f"`ulimit -n {required}` in the launching shell"
        )
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"could not raise the open-file soft limit from {soft} to "
            f"{required}; run `ulimit -n {required}` in the launching shell: "
            f"{exc}"
        ) from exc
    effective, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if effective < required:
        raise SystemExit(
            f"open-file soft limit remained {effective}, below the required "
            f"{required}; run `ulimit -n {required}` in the launching shell"
        )
    return effective


def _tasks_by_id(
    pool: tuple[Ed1Instance, ...], task_ids: tuple[str, ...]
) -> tuple[Ed1Instance, ...]:
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
) -> tuple[tuple[Ed1Instance, ...], TaskRoleSelection | None, Ed1Instance]:
    has_manifest = args.task_manifest is not None
    has_role = args.task_role is not None
    if has_manifest != has_role:
        raise SystemExit(
            "--task-manifest and --task-role must be supplied together"
        )
    if has_manifest and args.task_ids:
        raise SystemExit("--task-manifest cannot be combined with --task-id")
    if has_manifest:
        try:
            manifest = load_task_split_manifest(args.task_manifest)
            selection = manifest.select_role(
                env="ed1", role=TaskSplitRole(args.task_role)
            )
        except TaskSplitManifestError as exc:
            raise SystemExit(str(exc)) from None
        pool = load_ed1_tasks(snapshot_path=args.snapshot_path)
        return (
            _tasks_by_id(pool, selection.task_ids),
            selection,
            _tasks_by_id(pool, (ED1_SCORING_PREFLIGHT_TASK_ID,))[0],
        )
    if args.task_ids:
        pool = load_ed1_tasks(snapshot_path=args.snapshot_path)
        return (
            _tasks_by_id(pool, tuple(args.task_ids)),
            None,
            _tasks_by_id(pool, (ED1_SCORING_PREFLIGHT_TASK_ID,))[0],
        )
    selected = load_ed1_tasks(
        snapshot_path=args.snapshot_path,
        limit=args.task_count,
    )
    return selected, None, selected[0]


def _write_proposal_attempt(
    attempt: Ed1CoproRoundAttempt,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    ordinal = attempt.proposal_call.request.request_ordinal
    identity = attempt.proposal_call.request.identity_hash()
    name = f"{ordinal:04d}-{identity}"
    path = directory / f"{name}.json"
    temporary = directory / f".{name}.tmp.json"
    temporary.write_text(
        attempt.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _value(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


def render_runtime(
    console: Console,
    transcript: Ed1CoproScoringTranscript,
    *,
    output_dir: Path,
) -> None:
    runtime = transcript.runtime
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("field", style="bold cyan")
    table.add_column("value", style="bright_white")
    table.add_row("dr-code", runtime.dr_code_version)
    table.add_row("evaluation Python", runtime.evaluation_python)
    table.add_row("Python", runtime.probe.python_version)
    table.add_row("NumPy", runtime.probe.numpy_version)
    table.add_row("runtime identity", runtime.runtime_identity_hash)
    table.add_row("tasks", ", ".join(transcript.task_ids))
    if transcript.task_selection is not None:
        table.add_row(
            "task manifest role", transcript.task_selection.role.value
        )
        table.add_row(
            "task manifest hash",
            transcript.task_selection.manifest_content_hash,
        )
    table.add_row("task model mode", transcript.task_model.kind.value)
    table.add_row("task model", transcript.task_model.model)
    table.add_row("row concurrency", str(transcript.concurrency))
    table.add_row(
        "preflight",
        f"{transcript.preflight.task_id}: {transcript.preflight.outcome}",
    )
    table.add_row("artifact directory", str(output_dir))
    console.print(Panel(table, title="Runtime and ground-truth preflight"))


def render_proposal_call(console: Console, call: Ed1CoproProposalCall) -> None:
    metadata = Table(show_header=False, box=None, pad_edge=False)
    metadata.add_column("field", style="bold cyan")
    metadata.add_column("value", style="bright_white")
    metadata.add_row("mode", call.request.proposal_mode)
    metadata.add_row("request ordinal", str(call.request.request_ordinal))
    metadata.add_row("requested drafts", str(call.requested_count))
    history = call.request.context[COPRO_INSTRUCTION_HISTORY_KEY]
    if not isinstance(history, tuple):
        raise ValueError("proposal history is not an ordered tuple")
    metadata.add_row("instruction history", str(len(history)))
    console.print(metadata)
    proposal_prompt = call.request.context["proposal_prompt"]
    assert isinstance(proposal_prompt, str)
    console.print(
        Panel(
            Text(proposal_prompt),
            title="Exact proposer prompt",
            border_style="yellow",
        )
    )
    drafts = Table("slot", "transport status", "instruction body", box=None)
    for index, draft in enumerate(call.drafts):
        drafts.add_row(
            str(index),
            "failed" if draft.failed else "returned",
            draft.template or "[dim](no schema-valid body)[/dim]",
        )
    console.print(Panel(drafts, title="Returned proposal slots"))

    evidence = call.drafts[0].response_evidence
    failure = Table(show_header=False, box=None, pad_edge=False)
    failure.add_column("field", style="bold red")
    failure.add_column("value", style="bright_white")
    for key in ("failure_stage", "failure_type", "failure_message"):
        value = evidence.get(key)
        if value is not None:
            failure.add_row(key.replace("_", " "), str(value))
    if failure.row_count:
        console.print(Panel(failure, title="Proposal failure"))

    for key, title, language in (
        ("artifact", "Raw final artifact", "json"),
        ("stdout", "Raw Codex JSONL / stdout", "json"),
        ("stderr", "Raw Codex stderr", "text"),
    ):
        value = evidence.get(key)
        if isinstance(value, str):
            rendered = (
                Syntax(value, language, word_wrap=True)
                if value
                else Text("(empty)", style="dim")
            )
            console.print(Panel(rendered, title=title, border_style="cyan"))


def render_proposal_attempt(
    console: Console,
    attempt: Ed1CoproRoundAttempt,
) -> None:
    render_proposal_call(console, attempt.proposal_call)
    accepted = {
        mutation.proposal_ordinal: mutation
        for mutation in attempt.candidate_mutations
    }
    rejected = {
        rejection.proposal_ordinal: rejection
        for rejection in attempt.rejections
    }
    dispositions = Table("slot", "disposition", "instruction body", "reason")
    for ordinal in range(attempt.proposal_call.requested_count):
        mutation = accepted.get(ordinal)
        rejection = rejected.get(ordinal)
        if mutation is not None:
            dispositions.add_row(
                str(ordinal),
                "accepted",
                mutation.proposed_body,
                "",
            )
        elif rejection is not None:
            dispositions.add_row(
                str(ordinal),
                rejection.kind.value,
                rejection.proposed_body or "(none)",
                rejection.reason,
            )
    console.print(Panel(dispositions, title="ED1 candidate dispositions"))
    if attempt.terminal_failure is not None:
        console.print(
            Panel(
                Text(attempt.terminal_failure),
                title="Round failed after evidence capture",
                border_style="red",
            )
        )


def _step_text(step, field: str) -> str:
    value = step.inputs.get(field)
    if value is None:
        value = step.outputs.get(field)
    if not isinstance(value, str):
        raise ValueError(f"component step {field!r} is not text")
    return value


def render_candidate(
    console: Console,
    candidate: Ed1ScoredCandidate,
    transcript: Ed1CoproScoringTranscript,
) -> None:
    body = candidate.candidate.record.payload[MUTATION_FIELD]
    assert isinstance(body, str)
    console.rule(
        f"[bold magenta]Occurrence {candidate.occurrence_ordinal}: "
        f"{candidate.candidate.record.candidate_id}"
    )
    console.print(Panel(Text(body), title="Instruction"))
    for row in candidate.component_traces.rows:
        console.print(
            f"[bold]Task {row.task_identity}, repeat {row.repeat}[/bold]"
        )
        for step in row.executed_component_trace.executed_component_steps:
            prompt = _step_text(step, "prompt")
            generation = _step_text(step, GENERATION_OUTPUT_FIELD)
            console.print(
                Panel(
                    Text(prompt),
                    title=f"{step.component_id}: model input",
                    border_style="yellow",
                )
            )
            rendered_generation = (
                Syntax(generation, "python", word_wrap=True)
                if step.component_id == "decode"
                else Text(generation)
            )
            console.print(
                Panel(
                    rendered_generation,
                    title=(
                        f"{step.component_id}: "
                        f"{transcript.task_model.kind.value} model output"
                    ),
                    border_style="cyan",
                )
            )
    primary = candidate.primary_value
    compression = candidate.compression_value
    reward = candidate.attempt.reward
    score = (
        None
        if compression is None
        else compression_score(compression, transcript.blend_config)
    )
    calculation = Table(show_header=False, box=None, pad_edge=False)
    calculation.add_column("field", style="bold cyan")
    calculation.add_column("value", style="bright_white")
    calculation.add_row("HumanEval correctness", _value(primary))
    calculation.add_row("compression ratio", _value(compression))
    calculation.add_row("bounded compression score", _value(score))
    calculation.add_row(
        "compression weight", str(transcript.blend_config.weight)
    )
    calculation.add_row("blended reward", _value(reward))
    calculation.add_row(
        "row state",
        candidate.outputs.outputs[0].failure_code or "scored",
    )
    calculation.add_row(
        "evaluation evidence",
        candidate.resolution.evaluation_result_ref.content_hash
        if candidate.resolution.evaluation_result_ref is not None
        else "None",
    )
    console.print(Panel(calculation, title="Real scoring and reward"))


def render_round(
    console: Console,
    round_record: Ed1CoproScoredRound,
    transcript: Ed1CoproScoringTranscript,
) -> None:
    plan = round_record.preview.round_plan
    console.rule(
        f"[bold blue]Round {plan.iteration + 1}: {plan.proposal_mode}"
    )
    render_proposal_attempt(
        console,
        Ed1CoproRoundAttempt(
            starting_state=round_record.preview.starting_state,
            round_plan=round_record.preview.round_plan,
            proposal_call=round_record.preview.proposal_call,
            candidate_mutations=round_record.preview.candidate_mutations,
            rejections=(),
        ),
    )
    for candidate in round_record.evaluations:
        render_candidate(console, candidate, transcript)
    ranking = Table("rank", "instruction", "reward", box=None)
    ranked = sorted(
        (item.attempt for item in round_record.evaluations),
        key=lambda item: -item.reward,
    )
    for rank, attempt in enumerate(ranked, start=1):
        ranking.add_row(str(rank), attempt.instruction, _value(attempt.reward))
    console.print(Panel(ranking, title="Round ranking"))


def render_point(
    console: Console,
    point: Ed1CoproScoringPoint,
    transcript: Ed1CoproScoringTranscript,
) -> None:
    ratio = (
        "none"
        if point.settings.budget_ratio is None
        else str(point.settings.budget_ratio)
    )
    console.rule(
        f"[bold green]Sweep {point.settings.sweep_ordinal + 1}: "
        f"budget ratio {ratio}, breadth {point.settings.copro.breadth}, "
        f"depth {point.settings.copro.depth}"
    )
    console.print(
        f"[cyan]Evaluation Binding:[/cyan] "
        f"[bright_white]{point.evaluation_binding.identity_hash()}[/bright_white]"
    )
    for round_record in point.rounds:
        render_round(console, round_record, transcript)
    final = Table("rank", "candidate", "instruction", "reward", box=None)
    for rank, attempt in enumerate(
        point.finalization.ranked_attempts, start=1
    ):
        final.add_row(
            str(rank),
            attempt.candidate_id,
            attempt.instruction,
            _value(attempt.reward),
        )
    console.print(Panel(final, title="Final COPRO ranking"))


def render_transcript(
    console: Console,
    transcript: Ed1CoproScoringTranscript,
    *,
    output_dir: Path,
    codex_records: Path | None,
) -> None:
    proposer_kind = (
        transcript.points[0].rounds[0].preview.proposal_call.proposer_kind
    )
    console.print(
        Panel.fit(
            f"{proposer_kind} proposals → "
            f"{transcript.task_model.kind.value} task-model generation → "
            "real HumanEval execution → "
            "real 90/10 bounded-compression reward → COPRO ranking",
            title="ED1 COPRO scoring preview",
            border_style="bright_blue",
        )
    )
    interpretation = (
        "Encoder and decoder outputs are deterministic fixtures. This run "
        "validates proposer transport, candidate intake, execution, and "
        "lifecycle wiring; its ranking is not model-quality evidence."
        if transcript.task_model.kind is Ed1TaskModelKind.DUMMY
        else (
            "Encoder and decoder outputs came from the configured provider "
            "model. This is a tiny wiring preview, not a powered experiment."
        )
    )
    console.print(
        Panel(
            interpretation,
            title="Interpretation",
            border_style="yellow",
        )
    )
    render_runtime(console, transcript, output_dir=output_dir)
    for point in transcript.points:
        render_point(console, point, transcript)
    console.print(
        Panel(
            "\n".join(
                (
                    f"Transcript: {output_dir / 'transcript.json'}",
                    f"Object store: {output_dir / 'objects.sqlite3'}",
                    "Execution cache: "
                    f"{output_dir / 'execution-cache.sqlite3'}",
                    f"Execution records: {output_dir / 'execution-records'}",
                    f"Proposal calls: {output_dir / 'proposal-calls'}",
                    "Codex process records: "
                    + (str(codex_records) if codex_records else "not used"),
                )
            ),
            title="Durable outputs",
            border_style="green",
        )
    )


def _budget_ratios(mode: str) -> tuple[float | None, ...]:
    if mode == "unbudgeted":
        return (None,)
    if mode == "budgeted":
        return (0.5,)
    return (None, 0.5)


def _print_progress(console: Console, message: str) -> None:
    console.print(message)
    console.file.flush()


def _render_launch_plan(
    console: Console,
    *,
    proposer_kind: str,
    task_model: Ed1TaskModelConfig,
    tasks: tuple[Ed1Instance, ...],
    sweep: Ed1CoproSweepRanges,
    repeats: int,
    concurrency: int,
    open_file_limit: int,
    output_dir: Path,
) -> None:
    points = sweep.expand()
    candidate_evaluations = sum(
        point.copro.breadth * point.copro.depth for point in points
    )
    task_model_calls = candidate_evaluations * len(tasks) * repeats * 2
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("field", style="bold cyan")
    table.add_column("value")
    table.add_row("proposer", proposer_kind)
    table.add_row("task-model route", task_model.kind.value)
    table.add_row("task model", task_model.model)
    table.add_row(
        "task IDs", ", ".join(task.humaneval_task.task_id for task in tasks)
    )
    table.add_row("repeats", str(repeats))
    table.add_row("row concurrency", str(concurrency))
    table.add_row("open-file soft limit", str(open_file_limit))
    table.add_row("COPRO points", str(len(points)))
    table.add_row("candidate evaluations", str(candidate_evaluations))
    table.add_row("planned task-model calls", str(task_model_calls))
    table.add_row("output directory", str(output_dir))
    console.print(
        Panel(table, title="COPRO demo launch plan", border_style="yellow")
    )
    console.file.flush()


def _observe_proposal(
    attempt: Ed1CoproRoundAttempt,
    *,
    console: Console,
    output_dir: Path,
) -> None:
    _write_proposal_attempt(attempt, output_dir / "proposal-calls")
    accepted = len(attempt.candidate_mutations)
    requested = attempt.proposal_call.requested_count
    status = "completed" if attempt.succeeded else "failed"
    _print_progress(
        console,
        f"[bold cyan]Proposal round {status}:[/bold cyan] "
        f"{accepted}/{requested} instruction bodies accepted; evidence saved",
    )


def _observe_candidate(
    progress: Ed1CoproCandidateProgress,
    *,
    console: Console,
) -> None:
    position = f"{progress.candidate_index + 1}/{progress.candidate_count}"
    round_number = progress.round_index + 1
    if progress.result is None:
        _print_progress(
            console,
            f"[cyan]Round {round_number} candidate {position} started[/cyan]",
        )
        return
    _print_progress(
        console,
        f"[green]Round {round_number} candidate {position} completed:[/green] "
        f"reward={progress.result.attempt.reward:.6f}",
    )


def _build_proposer(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> tuple[str, ProposerRouteConfig, ProposerTransport, Path | None]:
    if args.proposer == "dummy":
        if args.breadth > 3:
            raise SystemExit("dummy proposer supports breadth at most 3")
        return (
            "dummy",
            DummyCoproProposerConfig(
                bodies=(
                    DUMMY_PASSING_BODY,
                    DUMMY_FAILING_BODY,
                    DUMMY_ALTERNATE_PASSING_BODY,
                )
            ),
            DummyCoproProposerTransport(),
            None,
        )

    records = (
        (
            args.codex_records
            if args.codex_records is not None
            else output_dir / "codex-proposer-records"
        )
        .expanduser()
        .absolute()
    )
    records.mkdir(parents=True, exist_ok=True)
    executor = ProcessExecutor(
        runtime=IsolatedHostPythonRuntime(Path(sys.executable)),
        run_store=DirectoryRunStore(root=records),
    )
    return (
        "codex_cli",
        CodexCliProposerConfig(
            codex_binary=args.codex_binary,
            model=args.model,
            timeout_seconds=args.codex_timeout_seconds,
        ),
        CodexCliProposerTransport(executor=executor),
        records,
    )


def _build_task_model(
    args: argparse.Namespace,
) -> Ed1TaskModelConfig:
    kind = Ed1TaskModelKind(args.task_model)
    route = canonical_task_route(
        model=args.provider_model,
        temperature=None,
        max_attempts=1,
    )
    if kind is Ed1TaskModelKind.PROVIDER:
        return Ed1TaskModelConfig(
            kind=kind,
            provider_call_config=route.call_config,
            execution_policy=route.execution_policy,
        )
    return Ed1TaskModelConfig(
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


def main() -> None:
    args = _parse_args()
    if os.environ.get("DR_CODE_DISPOSABLE_WORKER") != "1":
        raise SystemExit(
            "refusing to execute generated code: set "
            "DR_CODE_DISPOSABLE_WORKER=1 in a disposable worker environment"
        )
    output_dir = args.output_dir.expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    console = Console()
    open_file_limit = _ensure_open_file_limit(args.concurrency)
    (
        proposer_kind,
        proposer_config,
        proposer_transport,
        codex_records,
    ) = _build_proposer(args, output_dir=output_dir)
    task_model = _build_task_model(args)
    runtime = build_ed1_scoring_runtime(
        runtime_executable=args.evaluation_python,
        record_root=output_dir / "execution-records",
    )
    runtime_summary = Ed1ScoringRuntimeSummary(
        evaluation_python=runtime.probe.python_executable,
        dr_code_version=version("dr-code"),
        runtime_identity_hash=runtime.runtime_identity_hash,
        probe=runtime.probe,
    )
    tasks, task_selection, preflight_task = _select_tasks(args)
    sweep = Ed1CoproSweepRanges(
        budget_ratios=_budget_ratios(args.budget_mode),
        breadths=(args.breadth,),
        depths=(args.depth,),
    )
    store = ObjectStore(SqliteBackend(output_dir / "objects.sqlite3"))
    _render_launch_plan(
        console,
        proposer_kind=proposer_kind,
        task_model=task_model,
        tasks=tasks,
        sweep=sweep,
        repeats=args.repeats,
        concurrency=args.concurrency,
        open_file_limit=open_file_limit,
        output_dir=output_dir,
    )
    try:
        with CheckpointedCodeBatchScorer(
            output_dir / "execution-cache.sqlite3",
            runtime_identity=runtime.runtime_identity,
            executor=runtime.executor,
        ) as scorer:
            transcript = run_ed1_copro_scoring_preview(
                store=store,
                tasks=tasks,
                sweep=sweep,
                proposer_kind=proposer_kind,
                proposer_config=proposer_config,
                proposer_transport=proposer_transport,
                task_model=task_model,
                task_selection=task_selection,
                preflight_task=preflight_task,
                batch_scorer=scorer,
                runtime=runtime_summary,
                concurrency=args.concurrency,
                repeats=args.repeats,
                proposal_observer=lambda attempt: _observe_proposal(
                    attempt, console=console, output_dir=output_dir
                ),
                candidate_observer=lambda progress: _observe_candidate(
                    progress, console=console
                ),
            )
    except Ed1CoproRoundFailure as exc:
        console.print(
            Panel.fit(
                "The proposal round failed only after its complete call "
                "evidence and every slot disposition were persisted.",
                title="ED1 COPRO proposal failure",
                border_style="red",
            )
        )
        render_proposal_attempt(console, exc.attempt)
        console.print(
            f"[bold]Durable proposal evidence:[/bold] "
            f"{output_dir / 'proposal-calls'}"
        )
        if codex_records is not None:
            console.print(
                f"[bold]Codex process records:[/bold] {codex_records}"
            )
        raise SystemExit(1) from None
    _write_transcript(transcript, output_dir / "transcript.json")
    render_transcript(
        console,
        transcript,
        output_dir=output_dir,
        codex_records=codex_records,
    )


if __name__ == "__main__":
    main()
