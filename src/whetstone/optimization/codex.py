"""One opaque external Codex step over the canonical MCP evaluation tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from whetstone.optimization.adapters import AdapterOutput, ToolCallRecord
from whetstone.optimization.identity import TypedRef
from whetstone.optimization.mutation import DiffCheckError, diff_check
from whetstone.optimization.schema import (
    Candidate,
    OptimizationStepRequest,
    StepMode,
    StepStatus,
)
from whetstone.optimization.tool_store import (
    ToolCallState,
    ToolCallStore,
)
from whetstone.optimization.tools import ToolConfig, ToolResult

CODEX_ADAPTER_KEY = "codex"
CODEX_OUTPUT_ARTIFACT_SCHEMA = "whetstone.codex_output_artifact"


class OpaqueStepError(RuntimeError):
    """The external agent failed to produce its typed opaque result."""


class CodexOutputArtifact(BaseModel):
    """Required serialized output of the external Codex process."""

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
        self, request: OptimizationStepRequest, tool_config: ToolConfig
    ) -> CodexRunResult: ...


class CodexAdapter:
    """Validate the typed artifact and reconstruct MCP calls durably."""

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

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[Any, ...],
    ) -> AdapterOutput:
        del handles
        if request.step_index != 0:
            raise OpaqueStepError("Codex runs exactly one opaque step")
        if len(request.tool_configs) != 1:
            raise OpaqueStepError(
                "Codex requires one serialized external MCP Tool Config"
            )
        config = request.tool_configs[0]
        if not config.endpoint.startswith("mcp"):
            raise OpaqueStepError("Codex evaluation must be external MCP")
        run = self._runner.run(request, config)
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
        records = self._reconstruct_records(config)
        return AdapterOutput(
            proposed_candidates=proposals or (),
            accepted_candidates=proposals or (),
            tool_call_records=records,
            proposed_status=(
                StepStatus.COMPLETE
                if proposals is not None
                else StepStatus.FAILED
            ),
            state_delta={
                "codex_output_artifact_ref": typed_artifact_ref.model_dump(
                    mode="json"
                ),
                "tool_namespace": config.store_namespace,
                "tool_call_count": len(records),
                "accepted_call_count": self._tool_store.accepted_count(
                    config.identity_hash()
                ),
            },
        )

    def _reconstruct_records(
        self, config: ToolConfig
    ) -> tuple[ToolCallRecord, ...]:
        config_hash = config.identity_hash()
        records: list[ToolCallRecord] = []
        for call in self._tool_store.namespace_calls(
            config.store_namespace, config_hash
        ):
            entry = self._tool_store.get(config_hash, call.call_id)
            if entry is None:
                raise OpaqueStepError(
                    f"namespace call {call.call_id!r} has no store entry"
                )
            if entry.state is ToolCallState.COMPLETED:
                result = self._tool_store.load_completed_result(entry)
            elif entry.state is ToolCallState.REFUSED:
                assert entry.refusal is not None
                result = ToolResult(
                    call_id=call.call_id,
                    tool_config_ref=config.tool_definition_ref,
                    tool_config_hash=config_hash,
                    store_namespace=config.store_namespace,
                    refusal=entry.refusal,
                )
            else:
                raise OpaqueStepError(
                    f"namespace call {call.call_id!r} did not terminate"
                )
            records.append(ToolCallRecord(call=call, result=result))
        return tuple(records)

    @staticmethod
    def _validate_proposals(
        request: OptimizationStepRequest,
        proposals: tuple[Candidate, ...],
    ) -> tuple[Candidate, ...] | None:
        contract = request.output_contract
        if len(proposals) != contract.returned_proposal_count:
            return None
        bases = {candidate.base_ref for candidate in request.candidates}
        base_by_ref = {
            candidate.base_ref: candidate for candidate in request.candidates
        }
        seen: set[str] = set()
        for proposal in proposals:
            if proposal.base_ref not in bases:
                return None
            if contract.require_distinct_bases and proposal.base_ref in seen:
                return None
            seen.add(proposal.base_ref)
            try:
                diff_check(
                    base=base_by_ref[proposal.base_ref],
                    proposed=proposal,
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
