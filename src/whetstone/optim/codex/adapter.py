from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from whetstone.core.effects.authority import ReplayPolicy
from whetstone.core.identity import TerminalFailure, TypedRef
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optim.adapters import AdapterOutput
from whetstone.optim.contracts import (
    OptimStepRequest,
    StepMode,
    StepStatus,
)
from whetstone.optim.proposal.mutation import DiffCheckError, diff_check
from whetstone.optim.tools.contracts import RuntimeToolHandle
from whetstone.optim.tools.facade import ToolCallStore

CODEX_ADAPTER_KEY = "codex"
CODEX_OUTPUT_ARTIFACT_SCHEMA = "whetstone.codex_output_artifact"


class OpaqueStepError(RuntimeError):
    pass


class CodexOutputArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: StrictStr
    proposals: tuple[Candidate, ...]
    conversation_evidence: dict[str, Any] = Field(default_factory=dict)
    control_cost: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CodexRunResult:
    artifact: CodexOutputArtifact


class CodexRunner(Protocol):
    def run(
        self,
        request: OptimStepRequest,
        handle: RuntimeToolHandle,
    ) -> CodexRunResult: ...


class CodexAdapter:
    def __init__(
        self,
        runner: CodexRunner,
        *,
        store: ObjectStore,
        tool_store: ToolCallStore,
    ) -> None:
        self._runner = runner
        self._store = store
        self._tool_store = tool_store

    @property
    def key(self) -> str:
        return CODEX_ADAPTER_KEY

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.NO_REDRIVE

    def invoke(
        self,
        request: OptimStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        if request.step_index != 0:
            raise OpaqueStepError("Codex runs exactly one opaque step")
        if len(handles) != 1:
            raise OpaqueStepError(
                "Codex requires exactly one Runtime Tool Handle for its "
                "external MCP evaluation Tool"
            )
        handle = handles[0]
        config = handle.config
        if not str(config.endpoint_key).startswith("mcp"):
            raise OpaqueStepError("Codex evaluation must be external MCP")
        run = self._runner.run(request, handle)
        if run.artifact.run_id != request.run_id:
            raise OpaqueStepError(
                "Codex output artifact belongs to another run"
            )
        artifact_content = run.artifact.model_dump(mode="json")
        artifact_ref, _ = self._store.put(
            CODEX_OUTPUT_ARTIFACT_SCHEMA, artifact_content
        )
        typed_artifact_ref = TypedRef(
            schema_name=artifact_ref.schema,
            content_hash=artifact_ref.content_hash,
        )
        proposals = self._validate_proposals(request, run.artifact.proposals)
        rejected = proposals is None
        return AdapterOutput(
            proposed_candidates=proposals or (),
            accepted_candidates=proposals or (),
            proposed_status=(
                StepStatus.FAILED if rejected else StepStatus.COMPLETE
            ),
            terminal_failure=(
                TerminalFailure(
                    code="codex_proposal_contract",
                    message=(
                        "Codex output artifact proposals violate the exact "
                        "Step output contract"
                    ),
                    details={
                        "returned_proposal_count": (
                            request.step_output_contract.returned_proposal_count
                        ),
                        "artifact_proposal_count": len(run.artifact.proposals),
                    },
                )
                if rejected
                else None
            ),
            state_delta={
                "codex_output_artifact_ref": typed_artifact_ref.model_dump(
                    mode="json"
                ),
                "tool_namespace": str(config.store_namespace_key),
                "harness_store_accepted_call_count": (
                    self._tool_store.accepted_count(config, handle.binding)
                ),
            },
        )

    @staticmethod
    def _validate_proposals(
        request: OptimStepRequest,
        proposals: tuple[Candidate, ...],
    ) -> tuple[Candidate, ...] | None:
        contract = request.step_output_contract
        if len(proposals) != contract.returned_proposal_count:
            return None
        base_by_ref = {
            candidate_reference(candidate).record_ref: candidate
            for candidate in request.candidates
        }
        seen: set[TypedRef] = set()
        for proposal in proposals:
            if proposal.base_ref not in base_by_ref:
                return None
            if contract.require_distinct_bases and proposal.base_ref in seen:
                return None
            seen.add(proposal.base_ref)
            try:
                diff_check(
                    base=base_by_ref[proposal.base_ref],
                    proposed=proposal,
                    run=request.run,
                )
            except DiffCheckError:
                return None
        return proposals


__all__ = [
    "CODEX_ADAPTER_KEY",
    "CODEX_OUTPUT_ARTIFACT_SCHEMA",
    "CodexAdapter",
    "CodexOutputArtifact",
    "CodexRunResult",
    "CodexRunner",
    "OpaqueStepError",
]
