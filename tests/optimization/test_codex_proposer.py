"""Unit tests for the local-Codex-CLI-backed proposer transport.

No network and no real ``codex``: the subprocess invoker is injected (a
scripted
fake for the transport-behavior tests) or exercised against a STUB ``codex``
executable written onto a temp ``PATH`` (the real-subprocess path). Covers the
four terminal outcomes -- success, nonzero exit, timeout, empty output -- and
proves a failed draft returns the base template (so the diff check rejects it)
without a retry storm or a crash. Proposer spend is asserted $0 (plan-billed).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from whetstone.optimization.codex_proposer import (
    CODEX_CLI_LANE,
    CodexInvocation,
    CodexProposerTransport,
    SubprocessCodexInvoker,
    codex_proposer_ref,
)
from whetstone.optimization.proposer import ProposalRequest, ProposerConfig

MODEL = "gpt-5.4-mini"


def _config() -> ProposerConfig:
    ref = f"pcc://{codex_proposer_ref(MODEL)}"
    return ProposerConfig(
        provider_call_config_ref=ref,
        provider_call_config_hash="a" * 64,
        temperature=1.0,
    )


def _request() -> ProposalRequest:
    return ProposalRequest(
        proposal_mode="seed_proposal",
        request_ordinal=0,
        base_ref="base",
        base_template="Answer: {input}",
    )


class _FakeInvoker:
    """A scripted codex invoker: cycles a fixed list of invocations."""

    def __init__(self, invocations: list[CodexInvocation]) -> None:
        self._invocations = invocations
        self.calls: list[tuple[str, str]] = []

    def __call__(self, *, prompt: str, model: str) -> CodexInvocation:
        self.calls.append((prompt, model))
        idx = min(len(self.calls) - 1, len(self._invocations) - 1)
        return self._invocations[idx]


# --- Success ---------------------------------------------------------------


def test_success_returns_template_text_and_zero_cost() -> None:
    invoker = _FakeInvoker(
        [CodexInvocation(text="Improved: {input}\n", returncode=0)]
    )
    transport = CodexProposerTransport(model=MODEL, invoker=invoker)
    drafts = transport.draft(_config(), _request(), count=2)
    assert len(drafts) == 2
    for draft in drafts:
        assert draft.template == "Improved: {input}"  # stripped
        assert draft.cost == 0.0  # plan-billed
        assert draft.request_evidence["proposer_lane"] == CODEX_CLI_LANE
        assert draft.request_evidence["proposer_model"] == MODEL
        assert "failed" not in draft.request_evidence
    # One codex invocation per requested draft; the pinned model is used.
    assert len(invoker.calls) == 2
    assert all(model == MODEL for _prompt, model in invoker.calls)
    assert transport.proposer_calls == 2


def test_success_prompt_carries_base_template() -> None:
    invoker = _FakeInvoker([CodexInvocation(text="X {input}", returncode=0)])
    transport = CodexProposerTransport(model=MODEL, invoker=invoker)
    transport.draft(_config(), _request(), count=1)
    prompt, _model = invoker.calls[0]
    assert "Answer: {input}" in prompt  # the base template is in the prompt


# --- Nonzero exit ----------------------------------------------------------


def test_nonzero_exit_yields_failed_draft_base_template() -> None:
    invoker = _FakeInvoker([CodexInvocation(text="", returncode=3)])
    transport = CodexProposerTransport(model=MODEL, invoker=invoker)
    drafts = transport.draft(_config(), _request(), count=1)
    assert len(drafts) == 1
    draft = drafts[0]
    # A failed draft returns the base unchanged (the diff check rejects it),
    # never a fabricated candidate.
    assert draft.template == "Answer: {input}"
    assert draft.request_evidence["failed"] is True
    assert draft.response_evidence["finish"] == "failed"
    assert draft.cost == 0.0
    # No retry storm: exactly one invocation for the one requested draft.
    assert len(invoker.calls) == 1


# --- Timeout ---------------------------------------------------------------


def test_timeout_yields_failed_draft() -> None:
    invoker = _FakeInvoker(
        [CodexInvocation(text="", returncode=-1, timed_out=True)]
    )
    transport = CodexProposerTransport(model=MODEL, invoker=invoker)
    drafts = transport.draft(_config(), _request(), count=1)
    assert drafts[0].template == "Answer: {input}"
    assert drafts[0].request_evidence["failed"] is True
    assert "timed out" in drafts[0].response_evidence["error"]


# --- Empty output ----------------------------------------------------------


def test_empty_output_yields_failed_draft() -> None:
    invoker = _FakeInvoker([CodexInvocation(text="   \n  ", returncode=0)])
    transport = CodexProposerTransport(model=MODEL, invoker=invoker)
    drafts = transport.draft(_config(), _request(), count=1)
    assert drafts[0].template == "Answer: {input}"
    assert drafts[0].request_evidence["failed"] is True
    assert "empty" in drafts[0].response_evidence["error"]


def test_mixed_success_and_failure_across_drafts() -> None:
    invoker = _FakeInvoker(
        [
            CodexInvocation(text="Good {input}", returncode=0),
            CodexInvocation(text="", returncode=1),
        ]
    )
    transport = CodexProposerTransport(model=MODEL, invoker=invoker)
    drafts = transport.draft(_config(), _request(), count=2)
    assert drafts[0].template == "Good {input}"
    assert "failed" not in drafts[0].request_evidence
    assert drafts[1].template == "Answer: {input}"
    assert drafts[1].request_evidence["failed"] is True


def test_token_accounting_tallied() -> None:
    invoker = _FakeInvoker(
        [CodexInvocation(text="Y {input}", returncode=0, total_tokens=42)]
    )
    transport = CodexProposerTransport(model=MODEL, invoker=invoker)
    transport.draft(_config(), _request(), count=2)
    assert transport.proposer_tokens == 84  # 42 x 2


# --- Real subprocess path via a STUB codex executable on PATH --------------


def _write_stub_codex(dir_path: Path, *, script: str) -> None:
    """Write an executable ``codex`` stub into ``dir_path`` (POSIX shell)."""
    stub = dir_path / "codex"
    stub.write_text(script)
    mode = stub.stat().st_mode
    stub.chmod(mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def test_subprocess_invoker_success_via_stub(
    tmp_path: Path, monkeypatch
) -> None:
    # A stub `codex` that writes the drafted template to the
    # --output-last-message file (the clean-capture path) and exits 0.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_stub_codex(
        bindir,
        script=(
            "#!/bin/sh\n"
            "# find the --output-last-message FILE argument and write to it\n"
            'while [ "$#" -gt 0 ]; do\n'
            '  if [ "$1" = "--output-last-message" ]; then\n'
            "    shift\n"
            '    printf "STUB TEMPLATE {input}" > "$1"\n'
            "    break\n"
            "  fi\n"
            "  shift\n"
            "done\n"
            "exit 0\n"
        ),
    )
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    invoker = SubprocessCodexInvoker(timeout_s=30.0)
    result = invoker(prompt="rewrite this", model=MODEL)
    assert result.returncode == 0
    assert result.timed_out is False
    assert result.text.strip() == "STUB TEMPLATE {input}"


def test_subprocess_invoker_nonzero_exit_via_stub(
    tmp_path: Path, monkeypatch
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_stub_codex(bindir, script="#!/bin/sh\nexit 7\n")
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    invoker = SubprocessCodexInvoker(timeout_s=30.0)
    result = invoker(prompt="p", model=MODEL)
    assert result.returncode == 7
    assert result.text == ""  # no last-message file written


def test_subprocess_invoker_timeout_via_stub(
    tmp_path: Path, monkeypatch
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_stub_codex(bindir, script="#!/bin/sh\nsleep 5\nexit 0\n")
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    invoker = SubprocessCodexInvoker(timeout_s=0.5)
    result = invoker(prompt="p", model=MODEL)
    assert result.timed_out is True


def test_transport_over_stub_codex_end_to_end(
    tmp_path: Path, monkeypatch
) -> None:
    # The full transport driving the REAL subprocess invoker over a stub codex:
    # a clean draft round-trips the template text with $0 cost.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_stub_codex(
        bindir,
        script=(
            "#!/bin/sh\n"
            'while [ "$#" -gt 0 ]; do\n'
            '  if [ "$1" = "--output-last-message" ]; then\n'
            "    shift\n"
            '    printf "E2E {input}" > "$1"\n'
            "    break\n"
            "  fi\n"
            "  shift\n"
            "done\n"
            "exit 0\n"
        ),
    )
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    transport = CodexProposerTransport(model=MODEL, timeout_s=30.0)
    drafts = transport.draft(_config(), _request(), count=1)
    assert drafts[0].template == "E2E {input}"
    assert drafts[0].cost == 0.0


def test_codex_proposer_ref_format() -> None:
    assert codex_proposer_ref("gpt-5.4-mini") == "codex-cli/gpt-5.4-mini"


def test_stub_uses_read_only_sandbox_flag(
    tmp_path: Path, monkeypatch
) -> None:
    # The invoker must pass a read-only sandbox + skip-git-repo-check; the stub
    # records its argv so we can assert the safety flags.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_dump = tmp_path / "argv.txt"
    _write_stub_codex(
        bindir,
        script=(
            "#!/bin/sh\n"
            f'printf "%s\\n" "$@" > "{argv_dump}"\n'
            'while [ "$#" -gt 0 ]; do\n'
            '  if [ "$1" = "--output-last-message" ]; then\n'
            "    shift\n"
            '    printf "ok {input}" > "$1"\n'
            "    break\n"
            "  fi\n"
            "  shift\n"
            "done\n"
            "exit 0\n"
        ),
    )
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    SubprocessCodexInvoker(timeout_s=30.0)(prompt="p", model=MODEL)
    argv = argv_dump.read_text()
    assert "--skip-git-repo-check" in argv
    assert "read-only" in argv
    assert MODEL in argv
