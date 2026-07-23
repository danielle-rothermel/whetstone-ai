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

from whetstone.envs.ed1 import ED1_DEFAULT_BUDGET_RATIO, ED1_ENV_NAME
from whetstone.envs.sampling import Completeness, SamplingOverrides
from whetstone.execution.fanout import DEFAULT_CONCURRENCY
from whetstone.execution.partials import PartialLog
from whetstone.optimization.codex_proposer import (
    CodexProposerTransport,
    codex_proposer_ref,
)
from whetstone.optimization.proposer import (
    ProposalDraft,
    ProposalRequest,
    ProposerConfig,
    ProposerTransport,
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
from whetstone.runner.power import (
    DEFAULT_ALPHA,
    DEFAULT_REPEAT_CAP,
    PowerConfig,
)
from whetstone.runner.routes import (
    CANONICAL_PROPOSER_MODEL,
    LANE_NAMES,
    ProviderRoute,
    completeness_for_env,
    route_for,
    task_model_for_env,
)

__all__ = ["build_parser", "main"]

_LANE_CHOICES = ("openrouter", "openai", *LANE_NAMES)

#: The default model for the local codex-CLI proposer (--proposer-cli codex).
#: Stronger than the canonical gpt-5.4-nano, and ChatGPT-plan billed ($0).
CODEX_PROPOSER_DEFAULT_MODEL = "gpt-5.4-mini"

#: The default agent model for the codex OPTIMIZER (``--optimizer codex``),
#: whose proposer IS the local ``codex exec`` CLI. The brief
#: (``reports/optimizer-briefs.md`` §5) pins ``agent_model = gpt-5.6``, but
#: that model returns HTTP 400 ("not supported when using Codex with a ChatGPT
#: account") on this machine's plan -- so the default DEVIATES to
#: ``gpt-5.6-sol`` (a working sol-tier model on this account). Overridable by
#: ``--proposer-model``; recorded as ``codex-cli/<model>`` on
#: ``models.proposer``.
CODEX_OPTIMIZER_AGENT_MODEL = "gpt-5.6-sol"

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
        progress: Callable[[], str] | None = None,
        interval: float = HEARTBEAT_SECONDS,
        sink: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._label = label
        self._spend_estimate = spend_estimate
        self._progress = progress
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
            extra = f" {self._progress()}" if self._progress else ""
            self._sink(
                f"[heartbeat] {self._label} elapsed={elapsed:.0f}s "
                f"spend~={spend_str}{extra}\n"
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
        "--task-model",
        default=None,
        help=(
            "override the task model for this pilot (openrouter lane only). "
            "Absent, the per-env matrix default applies (c18/c22/c22h -> "
            "deepseek, others -> nano). The chosen model folds into the task "
            "route's Provider Call Config identity and is recorded on the "
            "pilot report."
        ),
    )
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
    pilot.add_argument(
        "--budget-ratio",
        type=float,
        default=ED1_DEFAULT_BUDGET_RATIO,
        metavar="R",
        help=(
            "ed1 (enc-dec) pilot ONLY: the per-task Character Budget ratio "
            f"(MAX_BUDGET = round(R * chars(input_code)), default "
            f"{ED1_DEFAULT_BUDGET_RATIO}). Lets a cheap ratio scan probe a "
            "distinct budget without the cell machinery. Ignored by QA envs."
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
        "--proposer-model",
        default=None,
        help=(
            "override the proposer model. For copro/miprov2/gepa this is the "
            "canonical OpenRouter proposer (default 'openai/gpt-5.4-nano'); "
            "with --proposer-cli codex it selects the codex-CLI model "
            "(default "
            "gpt-5.4-mini); for --optimizer codex it selects the codex agent "
            "model (default gpt-5.6-sol -- the brief's gpt-5.6 is rejected by "
            "ChatGPT-account Codex). Folds into the proposer route Config "
            "identity (never a graph identity) and is recorded in cells.jsonl "
            "models.proposer. Ignored by eval (the identity optimizer)."
        ),
    )
    cell.add_argument(
        "--proposer-cli",
        default=None,
        choices=("codex",),
        help=(
            "draft proposals through a LOCAL CLI instead of an OpenRouter "
            "HTTP call, for a proposal-using optimizer (copro/miprov2/gepa). "
            "'codex' uses the local `codex exec` (ChatGPT-plan billed, so "
            "proposer spend is $0) with model --proposer-model (default "
            "'gpt-5.4-mini'). The TASK model stays on --lane. Folds "
            "'codex-cli/<model>' into the proposer Config identity and "
            "records it in cells.jsonl models.proposer. Default (unset): the "
            "canonical OpenRouter gpt-5.4-nano proposer."
        ),
    )
    cell.add_argument(
        "--power-stage",
        nargs="?",
        type=float,
        const=DEFAULT_ALPHA,
        default=None,
        metavar="ALPHA",
        help=(
            "OPT-IN pre-run statistical-power stage (default OFF). Runs after "
            "the anchor arms: estimates the internal-eval sample size "
            "(n_tasks x repeats) needed to reliably rank candidates whose "
            f"score gap >= ALPHA x headroom (default ALPHA={DEFAULT_ALPHA}) "
            "at the target ranking probability, persists a per-cell "
            "power_analysis artifact (the full n x r MDD surface + variance "
            "decomposition), and SETS the optimizer's internal sizes from the "
            "recommendation (clamped to pool; recommended-vs-used recorded). "
            "ABSENT (the default), behavior + internal-eval sizes are "
            "byte-identical to a run without it."
        ),
    )
    cell.add_argument(
        "--power-repeat-cap",
        type=int,
        default=DEFAULT_REPEAT_CAP,
        metavar="R",
        help=(
            "the maximum internal repeats the power stage's (n x r) grid "
            f"searches + may recommend (default {DEFAULT_REPEAT_CAP}). Raise "
            "it for a repeat-noise-dominated env whose recommendation needs "
            "more repeats than the default (e.g. c22 rescue ~r=11 fits within "
            "20; a tighter target may need more). Only takes effect with "
            "--power-stage; the recommended repeats are APPLIED to the "
            "optimizer's internal evals, not just recorded."
        ),
    )
    cell.add_argument(
        "--budget-ratio",
        type=float,
        default=ED1_DEFAULT_BUDGET_RATIO,
        metavar="R",
        help=(
            "ed1 (enc-dec) ONLY: the per-task Character Budget ratio "
            f"(MAX_BUDGET = round(R * chars(input_code)), default "
            f"{ED1_DEFAULT_BUDGET_RATIO}). Folds into the enc-dec graph_hash "
            "via the Character Budget rule (a distinct ratio is a distinct "
            "Rollout Variant). Ignored by the QA envs."
        ),
    )
    cell.add_argument(
        "--task-filter",
        default=None,
        metavar="SCREEN_JSON",
        help=(
            "ed1 (enc-dec) ONLY: a task_screen/ed1_<model>.json path whose "
            "always-pass task ids are EXCLUDED from the pool (train/eval/"
            "test). "
            "The filtered Task Set folds into each split's eval hash, so "
            "a filtered cell is a distinct variant. Use the screen file for "
            "SAME model the cell runs (the exclusion list is per-model)."
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
        "--missing-data",
        choices=("propagate", "skip"),
        default=None,
        help=(
            "override the per-env matrix completeness policy. 'propagate' "
            "(strict): any missing/failed official row makes the arm "
            "incomplete. 'skip': tolerate up to --max-skip-fraction skipped "
            "rows as an explicit-count SKIP, else force incomplete. Default "
            "None = the env matrix default (c18 -> skip@2%%, others -> "
            "propagate). Folds into the official Eval Config identity."
        ),
    )
    cell.add_argument(
        "--max-skip-fraction",
        type=float,
        default=None,
        help=(
            "the declared completeness tolerance (fraction in [0,1]) for a "
            "SKIP policy: the max fraction of skipped official rows still "
            "certified. Default None = the env matrix default. Identity-"
            "bearing (a distinct value -> a distinct eval_config_hash)."
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

    screen = sub.add_parser(
        "screen",
        help=(
            "run the ed1 task-informativeness screen for one task model over "
            "the HumanEval+ pool (5 direct + 2 encdec arms x repeats)"
            "; writes task_screen/ed1_<model>.json with per-task counts + "
            "the always-pass pool-exclusion list"
        ),
    )
    screen.add_argument("--env", default=ED1_ENV_NAME)
    screen.add_argument(
        "--task-model", required=True,
        help=(
            "the task model to screen (e.g. qwen/qwen3-coder-flash or "
            "deepseek/deepseek-v4-flash). Adding a new model = this one flag."
        ),
    )
    screen.add_argument("--lane", choices=_LANE_CHOICES, default="openrouter")
    screen.add_argument(
        "--budget-ratio", type=float, default=0.25, metavar="R",
        help="the encdec-arm Character Budget ratio (default 0.25)",
    )
    screen.add_argument(
        "--repeats", type=int, default=5,
        help="repeats per arm per task (default 5)",
    )
    screen.add_argument(
        "--variants", default=None, metavar="A,B,...",
        help=(
            "comma-separated subset of screen arms to run (default: all 7 -- "
            "direct_original,direct_docstring,direct_signature,direct_name,"
            "direct_renamed,encdec_naive,encdec_renamed)"
        ),
    )
    screen.add_argument(
        "--rename-token", default=None, metavar="NAME",
        help=(
            "the neutral name the *_renamed arms rename the canonical "
            "entry point to (default: target_fxn). Recorded in the artifact."
        ),
    )
    screen.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of tasks screened (default: all 164)",
    )
    screen.add_argument(
        "--dry-run-fake", action="store_true",
        help=(
            "wiring smoke test: NO live calls -- a scripted fake transport "
            "returns the canonical solution so every arm passes"
        ),
    )
    screen.add_argument(
        "--max-wall-seconds", type=float,
        default=DEFAULT_PILOT_MAX_WALL_SECONDS,
        help="whole-run wall deadline for the screen",
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


def _proposal_prompt(request: ProposalRequest) -> str:  # pragma: no cover
    """The instruction prompt a live proposer LM drafts one template from.

    COPRO's proposer rewrites the base ``user_prompt_template`` into an
    improved variant. The prompt hands the LM the current template and asks for
    a single rewritten template that (a) preserves every ``{placeholder}`` and
    (b) differs from the base (the Mutation-Surface diff check rejects a draft
    identical to the base). Only the template text is requested back, so the
    completion is used verbatim as the drafted template.
    """
    base = request.base_template
    return (
        "You are optimizing the instruction template of a prompt-based "
        "task solver. Rewrite the template below into a SINGLE improved "
        "variant that is clearer and more likely to elicit a correct answer. "
        "Rules: keep every {placeholder} token exactly as written; change the "
        "wording so the result is NOT identical to the original; output ONLY "
        "the rewritten template text with no preamble, quotes, or "
        "commentary.\n"
        f"\nORIGINAL TEMPLATE:\n{base}\n\nREWRITTEN TEMPLATE:"
    )


class _HttpProposerTransport:
    """A live OpenRouter-backed proposer transport for a real optimizer run.

    Drafts ``count`` template variants by driving the proposer route through
    the SAME bounded dr-providers attempt loop the rollout uses. Each draft is
    one provider call whose completion text becomes the drafted template; a
    failed draft yields the base template unchanged (which the optimizer's
    Mutation-Surface diff check then rejects, so a failed draft never silently
    becomes a fabricated candidate). Performs NO evaluation and derives NO
    Reward -- it only produces text.

    ``transport`` is injectable (a scripted fake in tests); the default builds
    the real dr-providers ``HttpProvider.invoke`` for the route.
    """

    def __init__(
        self, route: ProviderRoute, transport: TransportCall | None = None
    ) -> None:
        self._route = route
        if transport is None:  # pragma: no cover - live only
            from dr_providers.transport import HttpProvider

            transport = HttpProvider(policy=route.transport_policy).invoke
        self._transport: TransportCall = transport
        self._served = 0
        #: Cumulative proposer-call accounting the cell heartbeat reads so a
        #: long-running live cell streams its proposer token spend, not just
        #: the credits-delta. Total tokens is best-effort (0 when the wire
        #: usage is absent); the call count is exact.
        self.proposer_calls = 0
        self.proposer_tokens = 0

    def draft(
        self,
        config: ProposerConfig,
        request: ProposalRequest,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        from dr_providers import (
            MessageRole,
            PromptMessage,
            ProviderCallRequest,
            Transcript,
        )

        from whetstone.provider.driver import run_provider_call

        prompt = _proposal_prompt(request)
        drafts: list[ProposalDraft] = []
        for index in range(count):
            self._served += 1
            call_request = ProviderCallRequest(
                config=self._route.call_config,
                transcript=Transcript(
                    messages=(
                        PromptMessage(role=MessageRole.USER, content=prompt),
                    )
                ),
            )
            result = run_provider_call(
                request=call_request,
                policy=self._route.execution_policy,
                transport=self._transport,
                logical_call_id=(
                    f"proposer:{request.proposal_mode}:"
                    f"{request.request_ordinal}:{index}"
                ),
            )
            self.proposer_calls += 1
            base_evidence = {
                "proposal_mode": request.proposal_mode,
                "request_ordinal": request.request_ordinal,
                "draft_index": index,
            }
            # A failed OR empty-completion draft is a TYPED FAILURE -- the base
            # template is NEVER echoed back (no fabricated candidate).
            if not (result.succeeded and result.generation is not None):
                drafts.append(
                    ProposalDraft.failure(
                        detail="proposer call failed",
                        request_evidence={**base_evidence, "failed": True},
                        usage={"proposer_calls": 1},
                    )
                )
                continue
            generation = result.generation
            template = generation.text.strip()
            if not template:
                drafts.append(
                    ProposalDraft.failure(
                        detail="proposer returned an empty completion",
                        request_evidence={**base_evidence, "failed": True},
                        usage={"proposer_calls": 1},
                    )
                )
                continue
            usage = generation.response.usage
            if usage is not None and usage.total_tokens is not None:
                self.proposer_tokens += usage.total_tokens
            drafts.append(
                ProposalDraft(
                    template=template,
                    request_evidence=base_evidence,
                    response_evidence={"finish": "stop"},
                    usage={
                        "proposer_calls": 1,
                        "total_tokens": (
                            usage.total_tokens if usage is not None else 0
                        ),
                    },
                    # Per-call USD cost is not on the wire usage; cell spend is
                    # attributed via credits-delta in the ledger.
                    cost=None,
                )
            )
        return tuple(drafts)


def _proposer_config(
    lane: str,
    optimizer: str,
    *,
    proposer_model: str | None = None,
    codex_cli_model: str | None = None,
) -> ProposerConfig:
    """The proposer route Config identity for a cell.

    Canonical proposer is the OpenRouter ``gpt-5.4-nano`` route. When
    ``codex_cli_model`` is set -- the codex OPTIMIZER (its proposer IS the
    codex CLI) OR ``--proposer-cli codex`` on copro/miprov2/gepa -- the
    proposer route is the LOCAL codex CLI: ``codex-cli/<model>`` folds into the
    proposer Config identity (never a graph identity), and the model choice
    changes that identity distinctly. ``proposer_model`` overrides the
    canonical OpenRouter proposer model when the codex CLI is not selected.
    """
    if codex_cli_model is not None:
        ref = f"pcc://{codex_proposer_ref(codex_cli_model)}"
        return ProposerConfig(
            provider_call_config_ref=ref,
            provider_call_config_hash=_deterministic_hash(ref),
            temperature=1.0,
        )
    model = proposer_model or CANONICAL_PROPOSER_MODEL
    proposer_route = route_for(
        "openrouter", role="proposer", temperature=1.0, proposer_model=model
    )
    return ProposerConfig(
        provider_call_config_ref=f"pcc://{model}",
        provider_call_config_hash=proposer_route.call_config.identity_hash,
        temperature=1.0,
    )


def _deterministic_hash(value: str) -> str:
    """A stable 64-hex Identity Hash for a proposer route reference string.

    Used for the local codex-CLI proposer route, which has no dr-providers
    Provider Call Config to hash: the ``codex-cli/<model>`` ref is hashed so
    the proposer Config identity folds the lane + model distinctly and changes
    with the model choice (never a graph identity).
    """
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    if args.env == ED1_ENV_NAME:
        return _run_ed1_pilot(args)
    # Resolve the task model the same way a cell does: explicit --task-model
    # override, else the per-env matrix default (c18/c22/c22h -> deepseek,
    # others -> nano). Applied to the openrouter task route only; plan lanes
    # ignore it (their model is the endpoint's advertised model).
    resolved_task_model = (
        task_model_for_env(
            args.env, override=getattr(args, "task_model", None)
        )
        if args.lane == "openrouter"
        else None
    )
    route = route_for(
        args.lane,
        role="task",
        temperature=0.0,
        task_model=resolved_task_model,
    )
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


def _run_ed1_pilot(args: argparse.Namespace) -> int:  # pragma: no cover - live
    """Run the ed1 enc-dec pilot: both encoder probes on a small task slice.

    Drives the 3-node encoder->decoder->code-eval rollout for the naive (A) +
    ceiling (B) encoder templates over a small HumanEval+ slice and reports
    each
    probe's pass rate + Mean Compression Ratio (dual scores). The task-side
    rollouts run on the live openrouter route; the code eval is the LOCAL
    subprocess sandbox (no container).
    """
    from whetstone.runner.ed1_pilot import run_ed1_pilot

    resolved_task_model = task_model_for_env(
        args.env, override=getattr(args, "task_model", None)
    )
    route = route_for(
        args.lane, role="task", temperature=0.0,
        task_model=resolved_task_model,
    )
    report = run_ed1_pilot(
        transport=_live_transport(route),
        execution_policy=route.execution_policy,
        model=route.model,
        budget_ratio=getattr(args, "budget_ratio", ED1_DEFAULT_BUDGET_RATIO),
        tasks=args.instances,
        concurrency=args.concurrency,
    )
    path = report.write(args.root)

    def _arm_line(name: str, arm) -> str:
        loud = (
            f" [{arm.none_reason}]" if arm.none_reason is not None else ""
        )
        return (
            f"  {name}: pass={arm.pass_rate} "
            f"compression={arm.mean_compression} "
            f"present={arm.present_rows} failed={arm.failed_rows}{loud}\n"
        )

    sys.stdout.write(
        f"ed1 pilot env={args.env} model={report.model} "
        f"budget_ratio={report.budget_ratio} tasks={report.tasks}\n"
        + _arm_line("naive", report.naive)
        + _arm_line("ceiling", report.ceiling)
        + f"  pass_direction_ok={report.pass_direction_ok} "
        f"compression_direction_ok={report.compression_direction_ok} "
        f"dataset={report.dataset_revision} -> {path}\n"
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
        budget_ratio=getattr(args, "budget_ratio", ED1_DEFAULT_BUDGET_RATIO),
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


def _uses_codex_cli_proposer(args: argparse.Namespace) -> bool:
    """Whether this cell drafts through the local codex CLI proposer.

    True for the codex OPTIMIZER (``--optimizer codex``) -- its proposer IS the
    local ``codex exec`` CLI (``reports/optimizer-briefs.md`` §5: the opaque
    codex agent proposes; Whetstone measures each draft on the internal split;
    ``run_optimize`` reduces that tool-using inner loop to the shared
    breadth x depth seam) -- OR when ``--proposer-cli codex`` is passed for
    another proposal-using optimizer (copro/miprov2/gepa). The eval identity
    optimizer never drafts, so it never uses the codex proposer.
    """
    if args.optimizer == "eval":
        return False
    if args.optimizer == "codex":
        return True
    return getattr(args, "proposer_cli", None) == "codex"


def _codex_proposer_model(args: argparse.Namespace) -> str:
    """The codex-CLI proposer model.

    ``--proposer-model`` overrides; else the codex OPTIMIZER uses its default
    agent model (``gpt-5.6-sol`` -- the brief pins ``gpt-5.6`` but ChatGPT-
    account Codex rejects it) and a ``--proposer-cli codex`` cell uses the
    codex proposer default (``gpt-5.4-mini``).
    """
    override = getattr(args, "proposer_model", None)
    if override:
        return override
    if args.optimizer == "codex":
        return CODEX_OPTIMIZER_AGENT_MODEL
    return CODEX_PROPOSER_DEFAULT_MODEL


def _proposer_transport_for(
    args: argparse.Namespace, *, proposer_model: str | None
) -> ProposerTransport:
    """Build the proposer transport for a live cell.

    A proposal-using optimizer drafts template mutations through either:

    * the LOCAL codex CLI (:class:`CodexProposerTransport`) -- the codex
      OPTIMIZER always (its proposer IS the codex CLI), and copro/miprov2/gepa
      when ``--proposer-cli codex`` is passed. Available on ANY task lane,
      since the proposer route is a separate identity from the task route; or
    * the live OpenRouter ``_HttpProposerTransport`` (the default for
      copro/miprov2/gepa on the openrouter task lane).

    Only the eval identity optimizer never drafts, so ONLY it keeps the raising
    :class:`_LiveProposerUnavailable` placeholder -- and ``run_optimize`` never
    calls ``draft()`` for the identity optimizer (breadth x depth = 0), so that
    placeholder is never reached from a live run. (Pre-fix the codex optimizer
    ALSO got the placeholder, yet ``run_optimize`` DOES draft for codex
    (breadth 4 x depth 1) -> an unhandled ``RuntimeError`` that killed the live
    codex cell.)
    """
    if _uses_codex_cli_proposer(args):
        return CodexProposerTransport(model=_codex_proposer_model(args))
    if args.optimizer == "eval" or args.lane != "openrouter":
        return _LiveProposerUnavailable()
    proposer_route = route_for(
        args.lane, role="proposer", temperature=1.0,
        proposer_model=proposer_model,
    )
    return _HttpProposerTransport(proposer_route)


def _build_cell_config(
    args: argparse.Namespace,
) -> tuple[CellConfig, ProviderRoute]:
    """Build the FULL live cell Config (routes, policies, transports).

    Pure construction with NO network: it resolves the task/proposer routes and
    execution policies from the routes registry and wires the live task
    (``rollout_transport``) and proposer (``proposer_transport``) transports a
    real cell runs. Factored out of :func:`_run_cell` so the live cell path is
    constructible and assertable without a paid call -- the wiring test drives
    it for every optimizer kind to prove the fixture-only proposer seam is
    unreachable from the live entrypoint. Returns the Config plus the task
    route (the caller needs its ``key_env`` for the credits fetcher).
    """
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
    # Resolve the completeness tolerance: explicit flags override the per-env
    # matrix default. c18 defaults to SKIP-with-2%-tolerance; others PROPAGATE.
    default_missing, default_skip = completeness_for_env(args.env)
    missing_data = getattr(args, "missing_data", None) or default_missing
    max_skip_fraction = (
        args.max_skip_fraction
        if getattr(args, "max_skip_fraction", None) is not None
        else default_skip
    )
    completeness = Completeness(missing_data)
    # Resolve the proposer model: explicit --proposer-model override, else the
    # routes-registry canonical (gpt-5.4-nano). The codex OPTIMIZER and any
    # --proposer-cli codex cell draft through the local codex CLI, recorded
    # distinctly as codex-cli/<model>.
    proposer_model_override = getattr(args, "proposer_model", None)
    codex_cli_model = (
        _codex_proposer_model(args)
        if _uses_codex_cli_proposer(args) else None
    )
    proposer_transport = _proposer_transport_for(
        args, proposer_model=proposer_model_override
    )
    if codex_cli_model is not None:
        recorded_proposer_model = codex_proposer_ref(codex_cli_model)
    else:
        recorded_proposer_model = (
            proposer_model_override or CANONICAL_PROPOSER_MODEL
        )
    # ed1 pool filter: --task-filter names a screen artifact whose always-pass
    # task ids are excluded from the pool (folds into each split's identity).
    ed1_exclude = None
    task_filter = getattr(args, "task_filter", None)
    if task_filter:
        from whetstone.runner.task_screen import load_exclusion_ids
        ed1_exclude = load_exclusion_ids(Path(task_filter))
    config = CellConfig(
        optimizer=args.optimizer,
        env=args.env,
        lane=args.lane,
        attempt=args.attempt,
        task_model=(
            resolved_task_model if args.lane == "openrouter"
            else task_route.model
        ),
        proposer_model=recorded_proposer_model,
        canonical=not args.non_canonical,
        proposer_config=_proposer_config(
            args.lane, args.optimizer,
            proposer_model=proposer_model_override,
            codex_cli_model=codex_cli_model,
        ),
        proposer_transport=proposer_transport,
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
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        power_config=_power_config_for(args),
        budget_ratio=getattr(args, "budget_ratio", ED1_DEFAULT_BUDGET_RATIO),
        ed1_exclude_task_ids=ed1_exclude,
    )
    return config, task_route


def _power_config_for(args: argparse.Namespace) -> PowerConfig | None:
    """The opt-in power-stage config, or ``None`` (the strict default = OFF).

    ``--power-stage`` with no value uses the default alpha; ``--power-stage
    0.3`` sets alpha=0.3. ``--power-repeat-cap`` sets the (n x r) grid's repeat
    ceiling (default unchanged). Absent ``--power-stage``, returns ``None`` --
    the power stage does not run and the cell is byte-identical to a run
    without it.
    """
    alpha = getattr(args, "power_stage", None)
    if alpha is None:
        return None
    repeat_cap = getattr(args, "power_repeat_cap", DEFAULT_REPEAT_CAP)
    return PowerConfig(alpha=float(alpha), repeat_cap=int(repeat_cap))


def _run_cell(args: argparse.Namespace) -> int:  # pragma: no cover - live
    if getattr(args, "dry_run_fake", False):
        return _run_dry_cell(args)
    _require_live(args)
    ledger = Ledger(root=args.root)
    ledger.load()
    config, task_route = _build_cell_config(args)
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

    # Proposer-call token accounting for the heartbeat: a live proposer
    # transport tallies its calls + wire tokens as it drafts, so a long cell
    # streams its proposer usage (not just the credits-delta spend). The eval/
    # codex placeholder has no counters, so the progress line is task-only.
    proposer_transport = config.proposer_transport

    def _proposer_progress() -> str:
        calls = getattr(proposer_transport, "proposer_calls", None)
        if calls is None:
            return ""
        tokens = getattr(proposer_transport, "proposer_tokens", 0)
        return f"proposer_calls={calls} proposer_tokens={tokens}"

    cell_id = f"{args.optimizer}:{args.env}:a{args.attempt}"
    try:
        with _Heartbeat(
            label=f"cell {cell_id}",
            spend_estimate=_spend_estimate,
            progress=_proposer_progress,
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


def _run_screen(args: argparse.Namespace) -> int:  # pragma: no cover - live
    """Run the ed1 task-informativeness screen for one task model.

    Drives the selected arms (5 direct + 2 encdec by default) x repeats over
    HumanEval+ pool on the live route, with incremental partials + an output
    sidecar (resumable), and writes ``task_screen/ed1_<model>.json`` (per-task
    pass counts + verdicts + the pool-exclusion list + rename deltas). The code
    eval is the LOCAL subprocess sandbox (no container). ``--dry-run-fake``
    swaps a scripted fake transport (no live calls) for wiring validation.
    """
    from whetstone.execution.partials import PartialLog
    from whetstone.runner.task_screen import (
        DEFAULT_RENAME_TOKEN,
        model_tag,
        run_task_screen,
    )

    route = route_for(
        args.lane, role="task", temperature=0.0, task_model=args.task_model,
    )
    tag = model_tag(args.task_model)
    root = Path(args.root)
    partials = PartialLog(
        path=root / "partials" / f"screen_{tag}.partial.jsonl"
    )
    sidecar = root / "task_screen" / f"ed1_{tag}.outputs.jsonl"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    variants = (
        tuple(v.strip() for v in args.variants.split(",") if v.strip())
        if getattr(args, "variants", None) else None
    )
    rename_token = getattr(args, "rename_token", None) or DEFAULT_RENAME_TOKEN
    transport = (
        _screen_fake_transport()
        if getattr(args, "dry_run_fake", False)
        else _live_transport(route)
    )
    report = run_task_screen(
        model=route.model,
        transport=transport,
        execution_policy=route.execution_policy,
        budget_ratio=args.budget_ratio,
        repeats=args.repeats,
        variants=variants,
        rename_token=rename_token,
        limit=args.limit,
        concurrency=args.concurrency,
        partial_log=partials,
        sidecar_path=sidecar,
    )
    path = report.write(root)
    summary = report.arm_summary()
    deltas = report.rename_deltas()
    mode = " (dry-run-fake, no live calls)" if getattr(
        args, "dry_run_fake", False) else ""
    sys.stdout.write(
        f"ed1 screen model={report.model} tasks={len(report.rows)} "
        f"repeats={report.repeats} budget_ratio={report.budget_ratio}"
        f"{mode}\n"
        f"  excluded (always-pass, all screened arms): "
        f"{len(report.excluded_task_ids)} tasks\n"
    )
    def _fnum(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    for arm in report.arms:
        s = summary[arm]
        lat = _fnum(s["mean_latency_s"])
        reason = s["total_reasoning_tokens"]
        sys.stdout.write(
            f"  {arm}: mean_pass={_fnum(s['mean_pass_rate'])} "
            f"tasks_full_pass={int(s['tasks_full_pass'] or 0)} "
            f"mean_latency_s={lat} reasoning_tokens={reason}\n"
        )
    for pair, d in deltas.items():
        sys.stdout.write(
            f"  DELTA {pair}: canonical={_fnum(d['canonical_mean_pass'])} - "
            f"renamed={_fnum(d['renamed_mean_pass'])} = {_fnum(d['delta'])}\n"
        )
    sys.stdout.write(f"  -> {path}\n")
    return 0


def _screen_fake_transport():  # pragma: no cover - dry-run wiring only
    """A scripted fake transport for ``screen --dry-run-fake`` (no live call).

    Returns the canonical solution (renamed as each arm needs) so every arm
    passes -- it validates the screen wiring end-to-end without paid calls.
    """
    from whetstone.envs.ed1 import load_ed1_tasks
    from whetstone.runner.dryrun import ScriptedRolloutTransport
    from whetstone.runner.task_screen import rename_identifier, split_prompt

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=None)
    fp: dict[str, tuple[str, str]] = {}
    for t in tasks:
        ht = t.humaneval_task
        parts = split_prompt(ht.prompt, ht.entry_point)
        fp[parts.docstring[:40]] = (ht.entry_point, ht.ground_truth_code)
        for code in (
            t.input_code,
            rename_identifier(t.input_code, ht.entry_point, "target_fxn"),
        ):
            body = code.split("):", 1)[-1].strip()[:40]
            if body:
                fp[body] = (ht.entry_point, ht.ground_truth_code)

    def _reply(prompt: str) -> str:
        if prompt.startswith("Provide") or prompt.startswith("Compress"):
            for key, (ep, _gt) in fp.items():
                if key and key in prompt:
                    renamed = "1" if "def target_fxn(" in prompt else "0"
                    return f"REBUILD:{ep}:{renamed}"
            return "REBUILD::x"
        for t in tasks:
            ht = t.humaneval_task
            if f"REBUILD:{ht.entry_point}:1" in prompt:
                return rename_identifier(
                    ht.ground_truth_code, ht.entry_point, "target_fxn"
                )
            if f"REBUILD:{ht.entry_point}:0" in prompt:
                return ht.ground_truth_code
        for key, (ep, gt) in fp.items():
            if key and key in prompt:
                if "target_fxn" in prompt and ep not in prompt:
                    return rename_identifier(gt, ep, "target_fxn")
                return gt
        for t in tasks:
            if t.humaneval_task.entry_point in prompt:
                return t.humaneval_task.ground_truth_code
        return tasks[0].humaneval_task.ground_truth_code

    return ScriptedRolloutTransport(reply=_reply)


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
    if args.command == "screen":
        return _run_screen(args)
    parser.error("unknown command")  # pragma: no cover
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
