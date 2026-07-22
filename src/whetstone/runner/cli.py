"""``whetstone-validate`` -- the resumable validation runner CLI.

Two subcommands per ``reports/validation-plan.md``:

* ``pilot --env X [--lane L]`` -- the checklist-B pilot (~10 instances x both
  probes x repeats {0,1,2}), writing ``<root>/pilots/<env>.json``.
* ``cell --optimizer O --env E [--lane L] [--attempt N]`` -- one full
  validation cell (baseline + ceiling official evals, optimizer run on the
  internal split with brief-documented hyperparameters scaled to pool sizes,
  best-candidate official eval, delta + bootstrap CI), appending the
  ``cells.jsonl`` + ``spend.jsonl`` ledger lines.

Resumability, budget guards (reserve + per-cell stop-loss), and the
ceiling-once-per-env cache are enforced by :mod:`whetstone.runner.cell`.

**Live calls are off by default.** Without ``--live`` the CLI refuses to run
(this workflow makes no live paid LLM calls). ``--live`` wires the real
dr-providers transport; the surrounding workflow gates it. The Codex proposer
is the local ``codex exec`` bridge; ``--lane`` applies only to the inner
rollout calls.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from whetstone.envs.sampling import SamplingOverrides
from whetstone.execution.fanout import DEFAULT_CONCURRENCY
from whetstone.execution.partials import PartialLog
from whetstone.optimization.proposer import (
    ProposalDraft,
    ProposalRequest,
    ProposerConfig,
)
from whetstone.provider.driver import TransportCall
from whetstone.runner.budget import (
    BudgetGuard,
    ReserveError,
    openrouter_credits_fetcher,
)
from whetstone.runner.cell import (
    DEFAULT_CELL_MAX_WALL_SECONDS,
    CellBaselineFailure,
    CellConfig,
    run_cell,
)
from whetstone.runner.execution_mode import (
    ExecutionMode,
    detect_execution_mode,
)
from whetstone.runner.ledger import Ledger
from whetstone.runner.optimizers import OPTIMIZERS, scaling_help
from whetstone.runner.pilot import (
    DEFAULT_PILOT_MAX_WALL_SECONDS,
    PilotReport,
    run_pilot,
)
from whetstone.runner.routes import (
    CANONICAL_PROPOSER_MODEL,
    LANE_NAMES,
    ProviderRoute,
    route_for,
    task_model_for_env,
)

__all__ = ["build_parser", "main"]

_LANE_CHOICES = ("openrouter", *LANE_NAMES)

#: Progress heartbeat interval (seconds): how often the CLI prints a progress
#: line during a long cell/pilot run so nohup logs stream something regularly.
HEARTBEAT_SECONDS = 60.0


def _force_unbuffered_stdout() -> None:
    """Make stdout/stderr line-buffered so nohup logs stream promptly.

    A long cell run under ``nohup ... &`` buffers stdout by default, so a log
    tailer sees nothing until the process exits. The CLI sets
    ``PYTHONUNBUFFERED=1`` (for any child processes it spawns, e.g. the Codex
    bridge) and reconfigures its own streams to line buffering so each progress
    line and result flushes immediately.
    """
    os.environ["PYTHONUNBUFFERED"] = "1"
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:  # pragma: no branch - real TextIO
            with contextlib.suppress(Exception):
                reconfigure(line_buffering=True)


class _Heartbeat:
    """A background thread printing a progress line every ~60s.

    So a long-running cell (many minutes of provider calls) streams SOMETHING
    to a nohup log while it works: elapsed, the wall budget, and a spend-so-far
    estimate. It is best-effort and never affects the result; it stops when the
    surrounding ``with`` block exits.
    """

    def __init__(
        self,
        *,
        label: str,
        spend_estimate: Callable[[], float | None],
        interval: float = HEARTBEAT_SECONDS,
        sink: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._label = label
        self._spend_estimate = spend_estimate
        self._interval = interval
        self._sink = sink or (lambda m: sys.stdout.write(m))
        self._clock = clock
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start = clock()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            elapsed = self._clock() - self._start
            spend = self._spend_estimate()
            spend_str = "n/a" if spend is None else f"${spend:.4f}"
            self._sink(
                f"[heartbeat] {self._label} elapsed={elapsed:.0f}s "
                f"spend~={spend_str}\n"
            )

    def __enter__(self) -> _Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whetstone-validate",
        description=(
            "Resumable validation runner for the optimizer x env matrix. "
            "NO LIVE PAID LLM CALLS unless --live is passed (off by default). "
            + "\n\n"
            + scaling_help()
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("validation"),
        help="ledger/pilot output root (default: ./validation)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "wire the real dr-providers transport (a live paid call). OFF by "
            "default; this workflow never makes live paid calls."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=[m.value for m in ExecutionMode],
        default=None,
        help="force an execution mode instead of detecting it",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=(
            "max concurrent provider calls per fan-out phase "
            f"(default {DEFAULT_CONCURRENCY}); halved once on a rate-limit "
            "failure for the rest of the run"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pilot = sub.add_parser(
        "pilot", help="run the checklist-B pilot for one env"
    )
    pilot.add_argument("--env", required=True)
    pilot.add_argument("--lane", choices=_LANE_CHOICES, default="openrouter")
    pilot.add_argument("--instances", type=int, default=10)
    pilot.add_argument("--spec-estimate-tokens", type=int, default=None)
    pilot.add_argument(
        "--max-wall-seconds",
        type=float,
        default=DEFAULT_PILOT_MAX_WALL_SECONDS,
        help=(
            "whole-run wall deadline for the pilot (default "
            f"{DEFAULT_PILOT_MAX_WALL_SECONDS:.0f}s); on breach finish "
            "in-flight, persist partials, exit non-zero with a status note"
        ),
    )

    cell = sub.add_parser(
        "cell", help="run one full validation cell (optimizer x env)"
    )
    cell.add_argument(
        "--optimizer", required=True, choices=list(OPTIMIZERS)
    )
    cell.add_argument("--env", required=True)
    cell.add_argument("--lane", choices=_LANE_CHOICES, default="openrouter")
    cell.add_argument("--attempt", type=int, default=0)
    cell.add_argument(
        "--task-model",
        default=None,
        help=(
            "override the per-env default task model (the matrix config). "
            "c18 + c22 default to 'deepseek/deepseek-v4-flash'; others to "
            "'openai/gpt-5-nano'. The chosen model folds into the Provider "
            "Call Config (graph_hash) and is recorded in cells.jsonl "
            "models.task. Openrouter lane only."
        ),
    )
    cell.add_argument(
        "--max-wall-seconds",
        type=float,
        default=DEFAULT_CELL_MAX_WALL_SECONDS,
        help=(
            "whole-cell wall deadline (default "
            f"{DEFAULT_CELL_MAX_WALL_SECONDS:.0f}s); on breach finish "
            "in-flight, persist partials, record status=halted"
        ),
    )
    cell.add_argument(
        "--official-n",
        type=int,
        default=None,
        help=(
            "reduced-sampling override: evaluate only the FIRST-N official "
            "tasks (deterministic ordered subset of the official Task Set). "
            "Default None = the env spec-default official split size. Folds "
            "into the composite Eval Config Identity Hash, so a reduced cell "
            "is a DISTINCT Eval Config identity (cache MISS vs the "
            "full-config entry). Recorded in cells.jsonl sampling_overrides."
        ),
    )
    cell.add_argument(
        "--official-repeats",
        type=int,
        default=None,
        help=(
            "reduced-sampling override: official-split repeats (baseline/"
            "ceiling/best arms). Default None = the env spec default. Folds "
            "into the composite Eval Config Identity Hash (a different value "
            "-> a cache MISS vs the full-config entry) and is the count the "
            "official arms are driven at. Recorded in sampling_overrides."
        ),
    )
    cell.add_argument(
        "--non-canonical",
        action="store_true",
        help="mark this cell non-canonical (a debug/iteration cell)",
    )
    cell.add_argument(
        "--dry-run-fake",
        action="store_true",
        help=(
            "run the cell end-to-end against SCRIPTED FAKE transports (no "
            "live paid call, no --live needed). Proves the CLI plumbing works "
            "as a program: build_env_experiment -> baseline/ceiling/best "
            "official evals -> optimizer internal search -> delta + CI -> "
            "ledger append."
        ),
    )

    refinalize = sub.add_parser(
        "refinalize",
        help=(
            "recompute a cell's status from its PERSISTED evidence and append "
            "a corrected line (original preserved, note 'refinalized'). No "
            "provider calls; corrects a cell wrongly stamped 'halted' that "
            "actually completed every phase."
        ),
    )
    refinalize.add_argument(
        "--optimizer", required=True, choices=list(OPTIMIZERS)
    )
    refinalize.add_argument("--env", required=True)
    refinalize.add_argument("--attempt", type=int, default=0)
    return parser


def _require_live(args: argparse.Namespace) -> None:
    if not args.live:
        raise SystemExit(
            "refusing to run: --live not set. This workflow makes NO live "
            "paid LLM calls. Pass --live only when the surrounding workflow "
            "authorizes real provider calls."
        )


def _live_transport(
    route: ProviderRoute,
) -> TransportCall:  # pragma: no cover - live only
    """Build the real dr-providers transport for a route (live path only)."""
    from dr_providers.transport import HttpProvider

    provider = HttpProvider(policy=route.transport_policy)
    return provider.invoke


def _proposer_config(lane: str, optimizer: str) -> ProposerConfig:
    """The proposer route Config identity for a cell.

    Canonical proposer is the OpenRouter ``gpt-5.4-nano`` route; Codex uses the
    local ``codex exec`` bridge (its inner rollout calls follow ``--lane``).
    """
    if optimizer == "codex":
        return ProposerConfig(
            provider_call_config_ref="codex://codex_cli/gpt-5.6",
            provider_call_config_hash="c" * 64,
            temperature=1.0,
        )
    proposer_route = route_for("openrouter", role="proposer", temperature=1.0)
    return ProposerConfig(
        provider_call_config_ref=(
            f"pcc://{CANONICAL_PROPOSER_MODEL}"
        ),
        provider_call_config_hash=proposer_route.call_config.identity_hash,
        temperature=1.0,
    )


#: Exit code for a pilot whose calls ALL failed (a hard plumbing failure).
PILOT_ALL_FAILED_EXIT = 2


def _failure_summary_line(report: PilotReport) -> str:
    """A ``code=count code=count`` summary of the pilot's failed calls."""
    summary = report.failure_summary()
    if not summary:
        return "(no failures)"
    return " ".join(
        f"{code}={count}"
        for code, count in sorted(summary.items())
    )


def _run_pilot(args: argparse.Namespace) -> int:  # pragma: no cover - live
    _require_live(args)
    route = route_for(args.lane, role="task", temperature=0.0)
    transport = _live_transport(route)
    # A per-env resumable partial log (round-2 crashes lost every call). A
    # crashed run leaves this on disk; the report/resume path reads it.
    partial_log = PartialLog(
        path=args.root / "pilots" / f"{args.env}.partial.jsonl"
    )
    credits_fetcher = (
        openrouter_credits_fetcher(route.key_env)
        if args.lane == "openrouter"
        else None
    )
    report = run_pilot(
        env=args.env,
        lane=args.lane,
        model=route.model,
        transport=transport,
        execution_policy=route.execution_policy,
        instance_count=args.instances,
        spec_estimate_tokens=args.spec_estimate_tokens,
        credits_fetcher=credits_fetcher,
        concurrency=args.concurrency,
        max_wall_seconds=args.max_wall_seconds,
        partial_log=partial_log,
    )
    path = report.write(args.root)
    # --- Whole-run deadline: a partial run exits non-zero with a summary. ---
    if report.deadline_reached:
        sys.stderr.write(
            f"PILOT HALTED: env={args.env} lane={args.lane} "
            f"{report.status_note} -> {path}\n"
            f"  {report.success_count}/{report.call_count} calls done; "
            "partials persisted; resume to complete.\n"
        )
        return PILOT_ALL_FAILED_EXIT
    # A clean, complete pilot no longer needs its partial log.
    partial_log.delete()
    # --- Loud failure handling: a 0%-success pilot is a hard failure. ---
    # Without this a fully-failed run (every call rejected pre-flight) would
    # print `naive=None ceiling=None direction_ok=False` and exit 0, silently
    # masking a plumbing blocker. Zero-success -> exit non-zero with a summary
    # counting failures by code; partial failures -> warn but still exit 0.
    if report.success_rate == 0.0:
        sys.stderr.write(
            f"PILOT FAILED: env={args.env} lane={args.lane} "
            f"0/{report.call_count} calls succeeded -> {path}\n"
            f"  failures by code: {_failure_summary_line(report)}\n"
        )
        return PILOT_ALL_FAILED_EXIT
    if report.failed_count:
        sys.stderr.write(
            f"pilot warning: env={args.env} lane={args.lane} "
            f"{report.failed_count}/{report.call_count} calls failed "
            f"(success_rate={report.success_rate:.2f})\n"
            f"  failures by code: {_failure_summary_line(report)}\n"
        )
    sys.stdout.write(
        f"pilot {args.env} lane={args.lane} "
        f"success={report.success_count}/{report.call_count} "
        f"naive={report.naive.mean_score} ceiling={report.ceiling.mean_score} "
        f"direction_ok={report.direction_ok} -> {path}\n"
    )
    return 0


def _run_dry_cell(args: argparse.Namespace) -> int:
    """Run one cell against scripted fake transports (no live paid call)."""
    from whetstone.runner.dryrun import run_dry_cell

    decision = (
        detect_execution_mode(force=ExecutionMode(args.execution_mode))
        if args.execution_mode
        else detect_execution_mode()
    )
    outcome = run_dry_cell(
        env=args.env,
        optimizer=args.optimizer,
        root=args.root,
        attempt=args.attempt,
        lane=args.lane,
        execution_mode=decision.mode,
        overrides=SamplingOverrides(
            official_n=getattr(args, "official_n", None),
            official_repeats=getattr(args, "official_repeats", None),
        ),
    )
    r = outcome.record
    note = "skipped" if outcome.skipped else r.status
    sys.stdout.write(
        f"dry-run-fake cell {r.cell_id} mode={r.window_notes} {note} "
        f"baseline={r.baseline_official} best={r.best_official} "
        f"delta={r.delta} delta_ci95={r.delta_ci95} "
        f"naive_ci95={r.naive_ci95} ceiling_ci95={r.ceiling_ci95} "
        f"headroom_delta={r.headroom_delta} headroom_ci95={r.headroom_ci95} "
        f"no_headroom={r.no_demonstrable_headroom} "
        f"official_repeats_used={r.official_repeats_used} "
        f"escalated={r.escalated} "
        f"pooled={r.pooled_observation_counts} spend=${r.spend_usd:.4f}\n"
    )
    return 0


def _run_cell(args: argparse.Namespace) -> int:  # pragma: no cover - live
    if getattr(args, "dry_run_fake", False):
        return _run_dry_cell(args)
    _require_live(args)
    # Resolve the task model: explicit --task-model override, else the per-env
    # matrix default (c18/c22 -> deepseek, others -> nano). It folds into the
    # task route's Config identity (graph_hash) and is recorded as models.task.
    resolved_task_model = task_model_for_env(
        args.env, override=getattr(args, "task_model", None)
    )
    task_route = route_for(
        args.lane, role="task", temperature=0.0,
        task_model=resolved_task_model,
    )
    decision = (
        detect_execution_mode(force=ExecutionMode(args.execution_mode))
        if args.execution_mode
        else detect_execution_mode()
    )
    ledger = Ledger(root=args.root)
    ledger.load()
    config = CellConfig(
        optimizer=args.optimizer,
        env=args.env,
        lane=args.lane,
        attempt=args.attempt,
        task_model=(
            resolved_task_model if args.lane == "openrouter"
            else task_route.model
        ),
        proposer_model=(
            "codex_cli/gpt-5.6" if args.optimizer == "codex"
            else CANONICAL_PROPOSER_MODEL
        ),
        canonical=not args.non_canonical,
        proposer_config=_proposer_config(args.lane, args.optimizer),
        proposer_transport=_LiveProposerUnavailable(),
        rollout_transport=_live_transport(task_route),
        execution_policy=task_route.execution_policy,
        execution_mode=decision.mode,
        window_notes=decision.reason,
        concurrency=args.concurrency,
        max_wall_seconds=args.max_wall_seconds,
        sampling_overrides=SamplingOverrides(
            official_n=getattr(args, "official_n", None),
            official_repeats=getattr(args, "official_repeats", None),
        ),
    )
    credits_fetcher = (
        openrouter_credits_fetcher(task_route.key_env)
        if args.lane == "openrouter"
        else None
    )
    # A best-effort spend-so-far estimate for the heartbeat: the drop in
    # remaining credits since the run started (a fresh credits read each beat).
    baseline_remaining: list[float | None] = [None]

    def _spend_estimate() -> float | None:
        if credits_fetcher is None:
            return None
        try:
            snap = credits_fetcher()
        except Exception:  # pragma: no cover - heartbeat is best-effort
            return None
        if snap is None or snap.remaining_usd is None:
            return None
        if baseline_remaining[0] is None:
            baseline_remaining[0] = snap.remaining_usd
            return 0.0
        return max(0.0, baseline_remaining[0] - snap.remaining_usd)

    cell_id = f"{args.optimizer}:{args.env}:a{args.attempt}"
    try:
        with _Heartbeat(
            label=f"cell {cell_id}", spend_estimate=_spend_estimate
        ):
            outcome = run_cell(
                config,
                ledger=ledger,
                budget=BudgetGuard(),
                credits_fetcher=credits_fetcher,
            )
    except ReserveError as exc:
        sys.stderr.write(f"budget reserve guard: {exc}\n")
        return 2
    except CellBaselineFailure as exc:
        # A zero-successful-rollout baseline is a hard plumbing failure: no
        # cell line is recorded (see run_cell), and we exit non-zero loudly
        # rather than reporting a null-scores cell as if it were a result.
        sys.stderr.write(f"CELL FAILED: {exc}\n")
        return PILOT_ALL_FAILED_EXIT
    r = outcome.record
    note = "skipped" if outcome.skipped else r.status
    sys.stdout.write(
        f"cell {r.cell_id} mode={config.execution_mode.value} {note} "
        f"delta={r.delta} delta_ci95={r.delta_ci95} "
        f"headroom_delta={r.headroom_delta} headroom_ci95={r.headroom_ci95} "
        f"official_repeats_used={r.official_repeats_used} "
        f"escalated={r.escalated} spend=${r.spend_usd:.4f}\n"
    )
    return 0


class _LiveProposerUnavailable:  # pragma: no cover - live-path placeholder
    """A placeholder proposer transport for the live CLI path.

    The live proposer wiring (real dr-providers draft calls / Codex bridge) is
    intentionally not built here: this workflow is fake/fixture-only. The live
    path raises so a real run must supply a concrete proposer transport.
    """

    def draft(
        self,
        config: ProposerConfig,
        request: ProposalRequest,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        raise RuntimeError(
            "live proposer transport is not wired in this fixture-only "
            "workflow; supply a concrete ProposerTransport for a live run"
        )


def _run_refinalize(args: argparse.Namespace) -> int:
    """Recompute + append a corrected cell line from persisted evidence."""
    from whetstone.runner.refinalize import refinalize_cell

    ledger = Ledger(root=args.root)
    outcome = refinalize_cell(
        ledger,
        optimizer=args.optimizer,
        env=args.env,
        attempt=args.attempt,
    )
    cell_id = outcome.original.cell_id
    if not outcome.changed:
        sys.stdout.write(
            f"refinalize {cell_id}: no change ({outcome.reason})\n"
        )
        return 0
    assert outcome.corrected is not None
    sys.stdout.write(
        f"refinalize {cell_id}: {outcome.original.status} -> "
        f"{outcome.corrected.status} ({outcome.reason})\n"
        f"  appended corrected line to {ledger.cells_path}\n"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _force_unbuffered_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "pilot":
        return _run_pilot(args)
    if args.command == "cell":
        return _run_cell(args)
    if args.command == "refinalize":
        return _run_refinalize(args)
    parser.error("unknown command")  # pragma: no cover
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
