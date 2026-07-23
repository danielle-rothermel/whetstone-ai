"""A local-Codex-CLI-backed proposer transport (free, plan-billed drafting).

The proposal-using optimizers (COPRO / MIPROv2 / GEPA) draft new encoder
``user_prompt_template`` text through a **proposer route** whose identity lives
in the optimizer Config, never a graph identity (see
:mod:`whetstone.optimization.proposer`). This module reaches that proposer
route through the LOCAL ``codex exec`` CLI instead of an OpenRouter HTTP call:
the ChatGPT-plan account bills the drafting, so every proposer call is $0 and
can use a stronger model (e.g. ``gpt-5.4-mini``) than the canonical
``gpt-5.4-nano``.

Each draft is ONE ``codex exec`` invocation, non-interactively, in a read-only
sandbox, with ``--skip-git-repo-check`` and a scratch cwd, capturing ONLY the
agent's final message via ``--output-last-message FILE`` (codex stdout is
polluted by hook/counter lines, so it is never scraped). A per-call timeout,
non-zero exit, or empty output is a TYPED :class:`CodexProposerError`; the
transport does NOT retry -- the optimizer's own failed-draft handling (the base
template is returned so the Mutation-Surface diff check rejects it, and the
candidate never becomes a fabricated proposal) deals with it, exactly as the
HTTP proposer's failed-draft path does.

The subprocess call is injected (:class:`CodexCliInvoker` is the protocol; the
default :class:`SubprocessCodexInvoker` runs the real binary), so the transport
is unit-testable with a fake invoker or a stub ``codex`` on ``PATH`` and never
needs a real Codex login in tests.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from whetstone.optimization.proposer import (
    ProposalDraft,
    ProposalRequest,
    ProposerConfig,
)

__all__ = [
    "CODEX_CLI_LANE",
    "CodexCliInvoker",
    "CodexInvocation",
    "CodexProposerError",
    "CodexProposerTransport",
    "SubprocessCodexInvoker",
    "codex_proposer_ref",
]

#: The lane token that identifies the local codex CLI proposer route (folded
#: into the proposer Config identity ref, never a graph identity).
CODEX_CLI_LANE = "codex-cli"

#: Default per-call wall timeout for one ``codex exec`` draft.
_DEFAULT_TIMEOUT_S = 180.0


def codex_proposer_ref(model: str) -> str:
    """The proposer Config identity ref for a codex-CLI model (e.g.
    ``codex-cli/gpt-5.4-mini``)."""
    return f"{CODEX_CLI_LANE}/{model}"


class CodexProposerError(RuntimeError):
    """A codex-CLI proposer draft failed (timeout / nonzero exit / empty).

    A TYPED failure the transport raises internally and converts into a
    failed :class:`~whetstone.optimization.proposer.ProposalDraft` (base
    template returned), so the optimizer's diff check rejects it without a
    fabricated candidate -- never a retry storm and never a cell crash.
    """


@dataclass(frozen=True, slots=True)
class CodexInvocation:
    """The terminal result of one ``codex exec`` draft invocation.

    ``text`` is the agent's final message (the drafted template) captured from
    the ``--output-last-message`` file. ``returncode`` is the process exit
    code; ``timed_out`` records a wall-timeout; ``total_tokens`` is best-effort
    (0 when the CLI exposes no structured token count on this path).
    """

    text: str
    returncode: int
    timed_out: bool = False
    total_tokens: int = 0


class CodexCliInvoker(Protocol):
    """Runs one ``codex exec`` draft and returns its terminal invocation.

    A real invoker launches the binary; the test double is scripted. Either
    way it takes the drafting prompt + model and returns a
    :class:`CodexInvocation`;
    it MUST NOT raise on a nonzero exit or timeout -- it reports them on the
    returned invocation so the transport maps them to a typed failure.
    """

    def __call__(self, *, prompt: str, model: str) -> CodexInvocation: ...


class SubprocessCodexInvoker:
    """The default invoker: launches the real ``codex exec`` binary.

    Captures ONLY the agent's final message via ``--output-last-message`` (a
    temp file), so polluted stdout is never scraped. Runs read-only, with
    ``--skip-git-repo-check`` and a scratch cwd, under a per-call wall timeout.
    A nonzero exit / timeout is reported on the returned invocation (never
    raised) so the transport owns the typed-failure mapping.
    """

    def __init__(
        self,
        *,
        codex_binary: str = "codex",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        scratch_dir: str | None = None,
    ) -> None:
        self._codex_binary = codex_binary
        self._timeout_s = timeout_s
        self._scratch_dir = scratch_dir

    def __call__(self, *, prompt: str, model: str) -> CodexInvocation:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "last_message.txt"
            cwd = self._scratch_dir or tmp
            argv = [
                self._codex_binary,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "read-only",
                "--output-last-message",
                str(out_path),
            ]
            if model:
                argv.extend(["-m", model])
            argv.append(prompt)
            try:
                completed = subprocess.run(
                    argv,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_s,
                    cwd=cwd,
                    env={**os.environ},
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return CodexInvocation(
                    text="", returncode=-1, timed_out=True
                )
            text = out_path.read_text() if out_path.exists() else ""
            return CodexInvocation(
                text=text, returncode=completed.returncode
            )


class CodexProposerTransport:
    """A :class:`ProposerTransport` that drafts through the local codex CLI.

    ``draft`` issues ``count`` independent ``codex exec`` invocations (one per
    requested draft) through the injected :class:`CodexCliInvoker`. A clean
    invocation yields a :class:`ProposalDraft` carrying the returned template
    text and ``cost=0.0`` (plan-billed). A timeout / nonzero exit / empty
    output yields a FAILED draft (base template returned, ``failed=True``
    evidence), so the optimizer's diff check rejects it without a fabricated
    candidate -- never a retry storm. Best-effort proposer token counts are
    tallied for the cell heartbeat (0 when the CLI exposes none on this path).
    """

    def __init__(
        self,
        *,
        model: str,
        invoker: CodexCliInvoker | None = None,
        prompt_builder: Callable[[ProposalRequest], str] | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._model = model
        self._invoker: CodexCliInvoker = invoker or SubprocessCodexInvoker(
            timeout_s=timeout_s
        )
        self._prompt_builder = prompt_builder or _default_proposal_prompt
        #: Cumulative accounting the cell heartbeat reads (calls exact, tokens
        #: best-effort). Proposer spend is $0 (plan-billed).
        self.proposer_calls = 0
        self.proposer_tokens = 0

    def draft(
        self,
        config: ProposerConfig,
        request: ProposalRequest,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        prompt = self._prompt_builder(request)
        drafts: list[ProposalDraft] = []
        for index in range(count):
            self.proposer_calls += 1
            evidence = {
                "proposal_mode": request.proposal_mode,
                "request_ordinal": request.request_ordinal,
                "draft_index": index,
                "proposer_lane": CODEX_CLI_LANE,
                "proposer_model": self._model,
            }
            try:
                template = self._draft_one(prompt)
            except CodexProposerError as exc:
                drafts.append(
                    ProposalDraft(
                        template=request.base_template,
                        request_evidence={**evidence, "failed": True},
                        response_evidence={
                            "finish": "failed", "error": str(exc)
                        },
                        usage={"proposer_calls": 1, "total_tokens": 0},
                        cost=0.0,
                    )
                )
                continue
            drafts.append(
                ProposalDraft(
                    template=template,
                    request_evidence=evidence,
                    response_evidence={"finish": "stop"},
                    usage={"proposer_calls": 1, "total_tokens": 0},
                    # Plan-billed: the proposer call costs $0.
                    cost=0.0,
                )
            )
        return tuple(drafts)

    def _draft_one(self, prompt: str) -> str:
        invocation = self._invoker(prompt=prompt, model=self._model)
        self.proposer_tokens += invocation.total_tokens
        if invocation.timed_out:
            raise CodexProposerError(
                f"codex exec timed out drafting a proposal "
                f"(model={self._model!r})"
            )
        if invocation.returncode != 0:
            raise CodexProposerError(
                f"codex exec exited non-zero ({invocation.returncode}) "
                f"drafting a proposal (model={self._model!r})"
            )
        text = invocation.text.strip()
        if not text:
            raise CodexProposerError(
                "codex exec produced empty output drafting a proposal "
                f"(model={self._model!r})"
            )
        return text


def _default_proposal_prompt(request: ProposalRequest) -> str:
    """The drafting instruction handed to ``codex exec`` for one template.

    Mirrors the HTTP proposer's instruction: rewrite the base
    ``user_prompt_template`` into a single improved variant that keeps every
    ``{placeholder}`` token, differs from the base, and is returned verbatim
    (no preamble/quotes) so the CLI's final message IS the drafted template.
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
