"""Canonical GEPA tool-using adapter."""

from __future__ import annotations

import hashlib
from typing import Any

from whetstone.optimization.adapters import AdapterOutput, ToolCallRecord
from whetstone.optimization.mutation import DiffCheckError, diff_check
from whetstone.optimization.proposal_prompts import gepa_reflection_prompt
from whetstone.optimization.proposer import (
    ProposalPromptBuilder,
    ProposalRequest,
    ProposerConfig,
    ProposerTransport,
)
from whetstone.optimization.schema import (
    Candidate,
    OptimizationStepRequest,
    StepMode,
    StepStatus,
)
from whetstone.optimization.tools import (
    RuntimeToolHandle,
    ToolCall,
    ToolResult,
)

GEPA_ADAPTER_KEY = "gepa"
GEPA_VARIANT = "whetstone_multi_objective/v1"
ACCEPTANCE_POLICY = "same_minibatch_strict_pareto/v1"
TOOL_EVALUATE_MINIBATCH = "evaluate_minibatch"
TOOL_EVALUATE_SUBSET = "evaluate_subset"
_CORRECTNESS = "correctness"
_COMPRESSION = "compression"


def strict_pareto_accepts(
    *, parent: dict[str, float], child: dict[str, float]
) -> bool:
    """Accept only same-minibatch no-worse-and-one-better children."""
    no_worse = (
        child[_CORRECTNESS] >= parent[_CORRECTNESS]
        and child[_COMPRESSION] <= parent[_COMPRESSION]
    )
    better = (
        child[_CORRECTNESS] > parent[_CORRECTNESS]
        or child[_COMPRESSION] < parent[_COMPRESSION]
    )
    return no_worse and better


class GepaAdapter:
    def __init__(
        self,
        *,
        reflection_config: ProposerConfig,
        reflection_transport: ProposerTransport,
        prompt_builder: ProposalPromptBuilder = gepa_reflection_prompt,
    ) -> None:
        self._reflection_config = reflection_config
        self._reflection_transport = reflection_transport
        self._prompt_builder = prompt_builder

    @property
    def key(self) -> str:
        return GEPA_ADAPTER_KEY

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    @property
    def reflection_config(self) -> ProposerConfig:
        return self._reflection_config

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        minibatch, subset = self._handles(handles)
        state = _GepaState(request)
        records: list[ToolCallRecord] = []
        if not state.seed_done:
            for candidate in request.candidates:
                result = self._call_subset(subset, candidate, records)
                state.add(candidate, _objectives(result), accepted=True)
            state.seed_done = True
        parent = state.parent(request)
        task_ids = state.minibatch(request)
        parent_result = self._call_minibatch(
            minibatch, parent, task_ids, records
        )
        parent_objectives = _objectives(parent_result)
        diagnostic = {
            "task_ids": list(task_ids),
            "parent_objectives": parent_objectives,
            "tool_output": parent_result.output,
            "evidence_refs": [
                ref.model_dump(mode="json")
                for ref in parent_result.evaluation_evidence_refs
            ],
        }
        child, reflection = self._reflect(request, state, parent, diagnostic)
        accepted = False
        child_objectives: dict[str, float] | None = None
        if child is not None:
            child_result = self._call_minibatch(
                minibatch, child, task_ids, records
            )
            child_objectives = _objectives(child_result)
            accepted = strict_pareto_accepts(
                parent=parent_objectives,
                child=child_objectives,
            )
            state.add(child, child_objectives, accepted=accepted)
            if accepted:
                subset_result = self._call_subset(subset, child, records)
                state.replace_objectives(child, _objectives(subset_result))
        terminal_target = int(
            request.hyperparameters.get("returned_proposal_count", 1)
        )
        if state.accepted_count >= terminal_target:
            status = StepStatus.COMPLETE
        elif state.reflection_calls >= int(
            request.hyperparameters.get("max_reflection_lm_calls", 8)
        ):
            status = StepStatus.FAILED
        else:
            status = StepStatus.CONTINUE
        output_count = request.output_contract.returned_proposal_count
        returned = state.best(output_count)
        if output_count and len(returned) != output_count:
            status = StepStatus.FAILED
            returned = ()
        return AdapterOutput(
            proposed_candidates=returned,
            accepted_candidates=returned,
            tool_call_records=tuple(records),
            proposed_status=status,
            state_delta={
                "gepa_state": state.record(),
                "diagnostic_evidence": diagnostic,
                "reflection_evidence": reflection,
                "same_minibatch": list(task_ids),
                "acceptance": {
                    "policy": ACCEPTANCE_POLICY,
                    "decision": accepted,
                    "parent_objectives": parent_objectives,
                    "child_objectives": child_objectives,
                },
            },
        )

    @staticmethod
    def _handles(
        handles: tuple[RuntimeToolHandle, ...],
    ) -> tuple[RuntimeToolHandle, RuntimeToolHandle]:
        by_name = {handle.config.tool_name: handle for handle in handles}
        try:
            return (
                by_name[TOOL_EVALUATE_MINIBATCH],
                by_name[TOOL_EVALUATE_SUBSET],
            )
        except KeyError as exc:
            raise ValueError(
                f"GEPA is missing required tool {exc.args[0]!r}"
            ) from None

    @staticmethod
    def _call_minibatch(
        handle: RuntimeToolHandle,
        candidate: Candidate,
        task_ids: tuple[str, ...],
        records: list[ToolCallRecord],
    ) -> ToolResult:
        call = _tool_call(handle, candidate, task_ids)
        result = handle(call)
        records.append(ToolCallRecord(call=call, result=result))
        return result

    @staticmethod
    def _call_subset(
        handle: RuntimeToolHandle,
        candidate: Candidate,
        records: list[ToolCallRecord],
    ) -> ToolResult:
        call = _tool_call(handle, candidate, ())
        result = handle(call)
        records.append(ToolCallRecord(call=call, result=result))
        return result

    def _reflect(
        self,
        request: OptimizationStepRequest,
        state: _GepaState,
        parent: Candidate,
        diagnostic: dict[str, Any],
    ) -> tuple[Candidate | None, dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        per_step = int(
            request.hyperparameters.get("max_reflection_attempts_per_step", 3)
        )
        total = int(request.hyperparameters.get("max_reflection_lm_calls", 8))
        allowed = min(per_step, max(0, total - state.reflection_calls))
        for attempt in range(allowed):
            proposal = ProposalRequest(
                proposal_mode="gepa_reflection",
                request_ordinal=state.reflection_calls,
                base_ref=parent.base_ref,
                base_template=str(
                    parent.payload.get("user_prompt_template", "")
                ),
                context={
                    "parent_candidate_id": parent.candidate_id,
                    "diagnostic_evidence": diagnostic,
                },
            )
            prompt = self._prompt_builder(proposal)
            proposal = proposal.model_copy(
                update={
                    "context": {
                        **proposal.context,
                        "proposal_prompt": prompt,
                    }
                }
            )
            draft = self._reflection_transport.draft(
                self._reflection_config, proposal, 1
            )[0]
            state.reflection_calls += 1
            child = Candidate(
                candidate_id=f"gepa-{request.step_index}-{attempt}",
                base_ref=parent.base_ref,
                payload={"user_prompt_template": draft.template},
            )
            record = {
                "attempt": attempt,
                "request": draft.request_evidence,
                "response": draft.response_evidence,
            }
            try:
                diff_check(base=parent, proposed=child)
            except DiffCheckError as exc:
                attempts.append({**record, "rejected": str(exc)})
                continue
            if state.duplicate(child):
                attempts.append({**record, "rejected": "duplicate"})
                continue
            attempts.append({**record, "accepted": True})
            return child, {
                "attempts": attempts,
                "reflection_calls": state.reflection_calls,
                "diagnostic_evidence": diagnostic,
            }
        return None, {
            "attempts": attempts,
            "reflection_calls": state.reflection_calls,
            "diagnostic_evidence": diagnostic,
        }


def _tool_call(
    handle: RuntimeToolHandle,
    candidate: Candidate,
    task_ids: tuple[str, ...],
) -> ToolCall:
    content = (
        handle.config.tool_name,
        candidate.identity_hash(),
        task_ids,
    )
    digest = hashlib.sha256(repr(content).encode()).hexdigest()
    return ToolCall(
        call_id=f"{handle.config.tool_name}-{digest[:20]}",
        tool_config_hash=handle.tool_config_hash,
        store_namespace=handle.config.store_namespace,
        args={
            "base_ref": candidate.base_ref,
            "model_route": candidate.base_ref,
            "template": candidate.payload.get("user_prompt_template", ""),
            "task_ids": list(task_ids),
        },
    )


def _objectives(result: ToolResult) -> dict[str, float]:
    if result.refusal is not None or result.output is None:
        raise ValueError("a refused tool call has no objective evidence")
    raw = result.output.get("objective_values")
    if not isinstance(raw, dict):
        raise ValueError("tool result has no objective_values")
    return {
        _CORRECTNESS: float(raw[_CORRECTNESS]),
        _COMPRESSION: float(raw[_COMPRESSION]),
    }


class _GepaState:
    def __init__(self, request: OptimizationStepRequest) -> None:
        raw = request.pools.get("gepa_state", {})
        if not isinstance(raw, dict):
            raise ValueError("gepa_state must be an object")
        self.catalog: list[dict[str, Any]] = list(raw.get("catalog", []))
        self.seed_done = bool(raw.get("seed_done", False))
        self.reflection_calls = int(raw.get("reflection_calls", 0))
        self.rng_cursor = int(raw.get("rng_cursor", 0))
        self.task_pool = tuple(
            str(item) for item in request.pools.get("task_pool", [])
        )

    @property
    def accepted_count(self) -> int:
        return sum(bool(entry["accepted"]) for entry in self.catalog)

    def add(
        self,
        candidate: Candidate,
        objectives: dict[str, float],
        *,
        accepted: bool,
    ) -> None:
        self.catalog.append(
            {
                "candidate": candidate.model_dump(mode="json"),
                "objectives": objectives,
                "accepted": accepted,
            }
        )

    def replace_objectives(
        self, candidate: Candidate, objectives: dict[str, float]
    ) -> None:
        for entry in reversed(self.catalog):
            raw = entry["candidate"]
            if raw["candidate_id"] == candidate.candidate_id:
                entry["objectives"] = objectives
                return

    def duplicate(self, candidate: Candidate) -> bool:
        template = candidate.payload.get("user_prompt_template")
        return any(
            entry["candidate"]["base_ref"] == candidate.base_ref
            and entry["candidate"]["payload"].get("user_prompt_template")
            == template
            for entry in self.catalog
        )

    def parent(self, request: OptimizationStepRequest) -> Candidate:
        accepted = [entry for entry in self.catalog if entry["accepted"]]
        if not accepted:
            raise ValueError("GEPA has no measured parent")
        digest = hashlib.sha256(
            f"{request.run_id}:{request.step_index}:{self.rng_cursor}".encode()
        ).digest()
        entry = accepted[int.from_bytes(digest[:8], "big") % len(accepted)]
        return Candidate.model_validate(entry["candidate"])

    def minibatch(self, request: OptimizationStepRequest) -> tuple[str, ...]:
        size = int(request.hyperparameters.get("minibatch_size", 3))
        pool = self.task_pool or tuple(
            f"task-{index}" for index in range(max(size, 24))
        )
        digest = hashlib.sha256(
            f"{request.run_id}:{request.step_index}:tasks".encode()
        ).digest()
        start = int.from_bytes(digest[:8], "big") % len(pool)
        return tuple(
            pool[(start + index) % len(pool)] for index in range(size)
        )

    def best(self, count: int) -> tuple[Candidate, ...]:
        accepted = [entry for entry in self.catalog if entry["accepted"]]
        accepted.sort(
            key=lambda entry: (
                -float(entry["objectives"][_CORRECTNESS]),
                float(entry["objectives"][_COMPRESSION]),
                entry["candidate"]["candidate_id"],
            )
        )
        return tuple(
            Candidate.model_validate(entry["candidate"])
            for entry in accepted[:count]
        )

    def record(self) -> dict[str, Any]:
        self.rng_cursor += 1
        return {
            "catalog": self.catalog,
            "seed_done": self.seed_done,
            "reflection_calls": self.reflection_calls,
            "rng_cursor": self.rng_cursor,
        }


__all__ = [
    "ACCEPTANCE_POLICY",
    "GEPA_ADAPTER_KEY",
    "GEPA_VARIANT",
    "TOOL_EVALUATE_MINIBATCH",
    "TOOL_EVALUATE_SUBSET",
    "GepaAdapter",
    "strict_pareto_accepts",
]
