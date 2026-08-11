from __future__ import annotations

import json
from base64 import b64encode
from pathlib import Path

import pytest
from dr_exec import ExitedOutcome, FakeExecutor

from tests.optimization.codex.test_runner import (
    _codex_argv,
    _completed,
    _option,
)
from whetstone.core.identity import compute_identity_hash, typed_ref_for_record
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optimization.codex.proposer import (
    CodexCliProposerConfig,
    CodexCliProposerTransport,
)
from whetstone.optimization.proposal.proposer import ProposalRequest


def test_codex_cli_proposer_returns_exact_bodies_without_mcp() -> None:
    def respond(job, _cancellation):
        argv = _codex_argv(job)
        args = argv[1:]
        assert not any("mcp_servers" in arg for arg in args)
        assert not any("temperature" in arg for arg in args)
        schema_path = Path(_option(args, "--output-schema"))
        artifact_path = Path(_option(args, "--output-last-message"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["properties"]["bodies"]["minItems"] == 2
        assert schema["properties"]["bodies"]["maxItems"] == 2
        artifact_path.write_text(
            json.dumps({"bodies": ["First body", "Second body"]}),
            encoding="utf-8",
        )
        return _completed(
            job,
            outcome=ExitedOutcome(exit_code=0),
            stdout=b'{"type":"turn.completed"}\n',
            stderr=b"",
        )

    transport = CodexCliProposerTransport(
        executor=FakeExecutor(responder=respond),
        environment={},
    )
    candidate = Candidate(
        candidate_id="baseline",
        base_ref=typed_ref_for_record("test.parent", {"id": "parent"}),
        payload={"user_prompt_template": "Describe the code"},
    )
    request = ProposalRequest(
        proposal_mode="seed_proposal",
        request_ordinal=0,
        proposal_authority_identity_hash=compute_identity_hash(
            schema="test.authority", schema_version=1, payload={}
        ),
        base_candidate=candidate_reference(candidate),
        context={"proposal_prompt": "Improve this instruction"},
    )

    drafts = transport.draft(
        CodexCliProposerConfig(codex_binary="/usr/bin/true"),
        request,
        2,
    )

    assert [draft.template for draft in drafts] == [
        "First body",
        "Second body",
    ]
    assert all(
        draft.request_evidence["proposer"] == "codex_cli" for draft in drafts
    )
    assert all(
        draft.response_evidence["stdout"] == '{"type":"turn.completed"}\n'
        for draft in drafts
    )
    for draft in drafts:
        artifact = draft.response_evidence["artifact"]
        assert isinstance(artifact, str)
        assert json.loads(artifact) == {
            "bodies": ["First body", "Second body"]
        }


def test_codex_cli_proposer_preserves_raw_invalid_artifact_evidence() -> None:
    artifact = b'{"wrong":["visible body"]}'
    stdout = b'{"type":"item.completed"}\n\xff'
    stderr = b"exact diagnostic"

    def respond(job, _cancellation):
        args = _codex_argv(job)[1:]
        Path(_option(args, "--output-last-message")).write_bytes(artifact)
        return _completed(
            job,
            outcome=ExitedOutcome(exit_code=0),
            stdout=stdout,
            stderr=stderr,
        )

    transport = CodexCliProposerTransport(
        executor=FakeExecutor(responder=respond),
        environment={},
    )
    candidate = Candidate(
        candidate_id="baseline",
        base_ref=typed_ref_for_record("test.parent", {"id": "parent"}),
        payload={"user_prompt_template": "Describe the code"},
    )
    request = ProposalRequest(
        proposal_mode="seed_proposal",
        request_ordinal=0,
        proposal_authority_identity_hash=compute_identity_hash(
            schema="test.authority", schema_version=1, payload={}
        ),
        base_candidate=candidate_reference(candidate),
        context={"proposal_prompt": "Improve this instruction"},
    )

    drafts = transport.draft(
        CodexCliProposerConfig(codex_binary="/usr/bin/true"),
        request,
        2,
    )

    assert len(drafts) == 2
    assert all(draft.failed for draft in drafts)
    for index, draft in enumerate(drafts):
        evidence = draft.response_evidence
        assert evidence["draft_index"] == index
        assert evidence["failure_stage"] == "artifact_validation"
        assert evidence["artifact"] == artifact.decode()
        assert evidence["artifact_base64"] == b64encode(artifact).decode()
        assert evidence["stdout_base64"] == b64encode(stdout).decode()
        assert evidence["stderr"] == stderr.decode()


def test_codex_cli_proposer_stages_default_paid_plan_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_home = tmp_path / ".codex"
    source_home.mkdir()
    (source_home / "auth.json").write_text(
        '{"fake":"credential"}', encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CODEX_HOME", raising=False)

    def respond(job, _cancellation):
        environment = {item.name: item.value for item in job.env.variables}
        staged_home = Path(environment["CODEX_HOME"])
        assert staged_home != source_home
        assert (staged_home / "auth.json").read_text(encoding="utf-8") == (
            '{"fake":"credential"}'
        )
        args = _codex_argv(job)[1:]
        Path(_option(args, "--output-last-message")).write_text(
            '{"bodies":["Visible body"]}', encoding="utf-8"
        )
        return _completed(
            job,
            outcome=ExitedOutcome(exit_code=0),
            stdout=b"",
            stderr=b"",
        )

    transport = CodexCliProposerTransport(
        executor=FakeExecutor(responder=respond)
    )
    candidate = Candidate(
        candidate_id="baseline",
        base_ref=typed_ref_for_record("test.parent", {"id": "parent"}),
        payload={"user_prompt_template": "Describe the code"},
    )
    request = ProposalRequest(
        proposal_mode="seed_proposal",
        request_ordinal=0,
        proposal_authority_identity_hash=compute_identity_hash(
            schema="test.authority", schema_version=1, payload={}
        ),
        base_candidate=candidate_reference(candidate),
        context={"proposal_prompt": "Improve this instruction"},
    )

    drafts = transport.draft(CodexCliProposerConfig(), request, 1)

    assert drafts[0].template == "Visible body"


def test_codex_cli_proposer_config_has_no_temperature_control() -> None:
    config = CodexCliProposerConfig()

    assert "temperature" not in CodexCliProposerConfig.model_fields
    assert "temperature" not in config.identity_payload()
