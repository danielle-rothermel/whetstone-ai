from __future__ import annotations

import math
from base64 import b64encode
from collections.abc import Mapping
from dataclasses import dataclass
from subprocess import TimeoutExpired
from typing import Any

from dr_exec import Executor
from dr_serialize import StrictJsonDecodeError, decode_strict_json_bytes
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    ValidationError,
    model_validator,
)

from whetstone.core.identity import compute_identity_hash
from whetstone.optimization.codex.adapter import OpaqueStepError
from whetstone.optimization.codex.runner import (
    CodexStructuredExecutionFailure,
    SubprocessCodexRunner,
)
from whetstone.optimization.proposal.proposer import (
    ProposalDraft,
    ProposalRequest,
    ProposerRouteConfig,
    prompt_adapter_identity_hash,
)
from whetstone.provider.language_model import PlainPromptAdapter

CODEX_CLI_PROPOSER_CONFIG_SCHEMA = "whetstone.codex_cli_proposer_config"
CODEX_CLI_PROPOSER_CONFIG_SCHEMA_VERSION = 1
CODEX_CLI_PROPOSER_EXECUTION_SCHEMA = (
    "whetstone.codex_cli_proposer_execution_policy"
)
CODEX_CLI_PROPOSER_EXECUTION_SCHEMA_VERSION = 1
CODEX_CLI_PROPOSER_DURABILITY_SCHEMA = (
    "whetstone.codex_cli_proposer_transport_durability"
)
CODEX_CLI_PROPOSER_DURABILITY_SCHEMA_VERSION = 1


class CodexCliProposerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    codex_binary: StrictStr = "codex"
    model: StrictStr = ""
    timeout_seconds: float = 600.0

    @model_validator(mode="after")
    def _validate(self) -> CodexCliProposerConfig:
        if not self.codex_binary:
            raise ValueError("Codex proposer binary must be non-empty")
        if (
            not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError(
                "Codex proposer timeout_seconds must be finite and positive"
            )
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "kind": "codex_cli",
            "codex_binary": self.codex_binary,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=CODEX_CLI_PROPOSER_CONFIG_SCHEMA,
            schema_version=CODEX_CLI_PROPOSER_CONFIG_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class CodexCliProposalArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bodies: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def _validate(self) -> CodexCliProposalArtifact:
        if any(not body for body in self.bodies):
            raise ValueError("Codex proposal bodies must be non-empty")
        return self


def _proposal_prompt(request: ProposalRequest, *, count: int) -> str:
    prompt = request.context.get("proposal_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "Codex CLI proposer requires one nonblank proposal_prompt"
        )
    return (
        f"{prompt}\n\n"
        f"Produce exactly {count} candidate outputs satisfying the proposal "
        "request. Return the schema-conforming object and nothing else."
    )


def _artifact_schema(*, count: int) -> dict[str, Any]:
    schema = CodexCliProposalArtifact.model_json_schema()
    bodies = schema["properties"]["bodies"]
    assert isinstance(bodies, dict)
    bodies["minItems"] = count
    bodies["maxItems"] = count
    return schema


def _bytes_evidence(value: bytes) -> tuple[str, str]:

    return (
        value.decode("utf-8", errors="replace"),
        b64encode(value).decode("ascii"),
    )


def _failed_drafts(
    *,
    count: int,
    detail: str,
    request_evidence: dict[str, Any],
    response_evidence: dict[str, Any],
) -> tuple[ProposalDraft, ...]:

    return tuple(
        ProposalDraft.failure(
            detail=detail,
            request_evidence={**request_evidence, "draft_index": index},
            response_evidence={
                **response_evidence,
                "draft_index": index,
            },
            usage={},
            cost=None,
        )
        for index in range(count)
    )


@dataclass(frozen=True, slots=True)
class CodexCliProposerTransport:
    _executor: Executor
    _environment: Mapping[str, str] | None
    _prompt_adapter: PlainPromptAdapter

    def __init__(
        self,
        *,
        executor: Executor,
        environment: Mapping[str, str] | None = None,
        prompt_adapter: PlainPromptAdapter | None = None,
    ) -> None:
        object.__setattr__(self, "_executor", executor)
        object.__setattr__(
            self,
            "_environment",
            None if environment is None else dict(environment),
        )
        object.__setattr__(
            self,
            "_prompt_adapter",
            prompt_adapter or PlainPromptAdapter(),
        )

    @property
    def execution_policy_hash(self) -> str:
        return compute_identity_hash(
            schema=CODEX_CLI_PROPOSER_EXECUTION_SCHEMA,
            schema_version=CODEX_CLI_PROPOSER_EXECUTION_SCHEMA_VERSION,
            payload={
                "effect": "one_codex_exec",
                "retry": "none",
                "mcp": "disabled",
            },
        )

    @property
    def prompt_adapter_identity_hash(self) -> str:
        return prompt_adapter_identity_hash(self._prompt_adapter)

    @property
    def durability_identity_hash(self) -> str:
        return compute_identity_hash(
            schema=CODEX_CLI_PROPOSER_DURABILITY_SCHEMA,
            schema_version=CODEX_CLI_PROPOSER_DURABILITY_SCHEMA_VERSION,
            payload={
                "execution_policy_hash": self.execution_policy_hash,
                "prompt_adapter_identity_hash": (
                    self.prompt_adapter_identity_hash
                ),
            },
        )

    def draft(
        self,
        config: ProposerRouteConfig,
        request: ProposalRequest,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        if not isinstance(config, CodexCliProposerConfig):
            raise TypeError(
                "Codex CLI proposer requires CodexCliProposerConfig"
            )
        if type(count) is not int or count < 1:
            raise ValueError("proposer draft count must be a positive integer")
        request_evidence = {
            "proposal_request_hash": request.identity_hash(),
            "proposer_config_hash": config.identity_hash(),
            "proposer": "codex_cli",
        }
        runner = SubprocessCodexRunner(
            executor=self._executor,
            codex_binary=config.codex_binary,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            environment=self._environment,
        )
        try:
            execution = runner.run_structured_prompt(
                prompt=_proposal_prompt(request, count=count),
                output_schema=_artifact_schema(count=count),
            )
        except TimeoutExpired as exc:
            stdout_bytes = exc.output if isinstance(exc.output, bytes) else b""
            stderr_bytes = exc.stderr if isinstance(exc.stderr, bytes) else b""
            stdout, stdout_base64 = _bytes_evidence(stdout_bytes)
            stderr, stderr_base64 = _bytes_evidence(stderr_bytes)
            return _failed_drafts(
                count=count,
                detail="Codex proposal execution timed out",
                request_evidence=request_evidence,
                response_evidence={
                    "failure_stage": "execution",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "stdout": stdout,
                    "stdout_base64": stdout_base64,
                    "stderr": stderr,
                    "stderr_base64": stderr_base64,
                },
            )
        except CodexStructuredExecutionFailure as exc:
            stdout, stdout_base64 = _bytes_evidence(exc.stdout)
            stderr, stderr_base64 = _bytes_evidence(exc.stderr)
            artifact, artifact_base64 = _bytes_evidence(exc.artifact_bytes)
            return _failed_drafts(
                count=count,
                detail="Codex proposal execution failed",
                request_evidence=request_evidence,
                response_evidence={
                    "failure_stage": "execution",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "artifact": artifact,
                    "artifact_base64": artifact_base64,
                    "stdout": stdout,
                    "stdout_base64": stdout_base64,
                    "stderr": stderr,
                    "stderr_base64": stderr_base64,
                    "isolation": exc.isolation,
                },
            )
        except OpaqueStepError as exc:
            return _failed_drafts(
                count=count,
                detail="Codex proposal execution failed",
                request_evidence=request_evidence,
                response_evidence={
                    "failure_stage": "execution",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                },
            )
        stdout, stdout_base64 = _bytes_evidence(execution.stdout)
        artifact_text, artifact_base64 = _bytes_evidence(
            execution.artifact_bytes
        )
        process_evidence = {
            "artifact": artifact_text,
            "artifact_base64": artifact_base64,
            "stdout": stdout,
            "stdout_base64": stdout_base64,
            "stderr": execution.stderr,
            "stderr_base64": b64encode(
                execution.stderr.encode("utf-8")
            ).decode("ascii"),
            "isolation": execution.isolation,
        }
        try:
            decode_strict_json_bytes(
                execution.artifact_bytes,
                max_bytes=len(execution.artifact_bytes),
                max_depth=len(execution.artifact_bytes),
            )
            artifact = CodexCliProposalArtifact.model_validate_json(
                execution.artifact_bytes
            )
        except (StrictJsonDecodeError, ValidationError) as exc:
            return _failed_drafts(
                count=count,
                detail="Codex proposal artifact failed schema validation",
                request_evidence=request_evidence,
                response_evidence={
                    **process_evidence,
                    "failure_stage": "artifact_validation",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                },
            )
        if len(artifact.bodies) != count:
            return _failed_drafts(
                count=count,
                detail="Codex proposal artifact changed requested cardinality",
                request_evidence=request_evidence,
                response_evidence={
                    **process_evidence,
                    "failure_stage": "artifact_validation",
                    "failure_type": "proposal_cardinality",
                    "failure_message": (
                        f"expected {count} bodies, got {len(artifact.bodies)}"
                    ),
                },
            )
        return tuple(
            ProposalDraft(
                template=body,
                request_evidence={
                    **request_evidence,
                    "draft_index": index,
                },
                response_evidence={
                    **process_evidence,
                    "draft_index": index,
                },
                usage={},
                cost=None,
            )
            for index, body in enumerate(artifact.bodies)
        )


__all__ = [
    "CODEX_CLI_PROPOSER_CONFIG_SCHEMA",
    "CODEX_CLI_PROPOSER_CONFIG_SCHEMA_VERSION",
    "CodexCliProposalArtifact",
    "CodexCliProposerConfig",
    "CodexCliProposerTransport",
]
