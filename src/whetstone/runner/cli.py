"""``whetstone-validate`` -- the resumable validation runner CLI.

Two subcommands per ``reports/validation-plan.md``:

* ``pilot --env X [--lane L]`` -- the checklist-B pilot (~10 instances x both
  probes x repeats {0,1,2}), writing ``validation/pilots/<env>.json``.
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
import sys
from collections.abc import Sequence
from pathlib import Path

from whetstone.optimization.proposer import (
    ProposalDraft,
    ProposalRequest,
    ProposerConfig,
)
from whetstone.provider.driver import TransportCall
from whetstone.runner.budget import BudgetGuard, ReserveError
from whetstone.runner.cell import CellConfig, run_cell
from whetstone.runner.execution_mode import (
    ExecutionMode,
    detect_execution_mode,
)
from whetstone.runner.ledger import Ledger
from whetstone.runner.optimizers import OPTIMIZERS, scaling_help
from whetstone.runner.pilot import run_pilot
from whetstone.runner.routes import (
    CANONICAL_PROPOSER_MODEL,
    CANONICAL_TASK_MODEL,
    LANE_NAMES,
    ProviderRoute,
    route_for,
)

__all__ = ["build_parser", "main"]

_LANE_CHOICES = ("openrouter", *LANE_NAMES)


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
    sub = parser.add_subparsers(dest="command", required=True)

    pilot = sub.add_parser(
        "pilot", help="run the checklist-B pilot for one env"
    )
    pilot.add_argument("--env", required=True)
    pilot.add_argument("--lane", choices=_LANE_CHOICES, default="openrouter")
    pilot.add_argument("--instances", type=int, default=10)
    pilot.add_argument("--spec-estimate-tokens", type=int, default=None)

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
        "--non-canonical",
        action="store_true",
        help="mark this cell non-canonical (a debug/iteration cell)",
    )
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


def _run_pilot(args: argparse.Namespace) -> int:  # pragma: no cover - live
    _require_live(args)
    route = route_for(args.lane, role="task", temperature=0.0)
    transport = _live_transport(route)
    report = run_pilot(
        env=args.env,
        lane=args.lane,
        model=route.model,
        transport=transport,
        execution_policy=route.execution_policy,
        instance_count=args.instances,
        spec_estimate_tokens=args.spec_estimate_tokens,
    )
    path = report.write(args.root)
    sys.stdout.write(
        f"pilot {args.env} lane={args.lane} "
        f"naive={report.naive.mean_score} ceiling={report.ceiling.mean_score} "
        f"direction_ok={report.direction_ok} -> {path}\n"
    )
    return 0


def _run_cell(args: argparse.Namespace) -> int:  # pragma: no cover - live
    _require_live(args)
    task_route = route_for(args.lane, role="task", temperature=0.0)
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
            CANONICAL_TASK_MODEL if args.lane == "openrouter"
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
    )
    try:
        outcome = run_cell(
            config, ledger=ledger, budget=BudgetGuard()
        )
    except ReserveError as exc:
        sys.stderr.write(f"budget reserve guard: {exc}\n")
        return 2
    r = outcome.record
    note = "skipped" if outcome.skipped else r.status
    sys.stdout.write(
        f"cell {r.cell_id} mode={config.execution_mode.value} {note} "
        f"delta={r.delta} ci95={r.ci95} spend=${r.spend_usd:.4f}\n"
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "pilot":
        return _run_pilot(args)
    if args.command == "cell":
        return _run_cell(args)
    parser.error("unknown command")  # pragma: no cover
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
