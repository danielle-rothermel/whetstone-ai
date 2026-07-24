"""The GEPA adapter: many structured, bounded, tool-using Steps.

GEPA (``gepa-run.html``, ``optimizer-briefs.md`` section 4) is a multi-step
tool-using Optimization Run under the variant ``whetstone_multi_objective/v1``.
Each outer Step Request drives **exactly one** GEPA step -- the full
``restore -> select parent -> fresh diagnostic -> reflect & validate ->
same-minibatch child -> accept -> accepted-child full eval -> update``
transition -- and returns one immutable Step Result. Internal subphases never
escape as outer request kinds.

Two tools reach measurement (never Evaluation Intents):

* ``evaluate_minibatch(candidate, task_ids)`` -- one candidate on the caller's
  ``minibatch_size = 3`` stable task IDs at ``optimization_budget_ratio =
  0.5``;
  one Tool Result with 3 Rollout refs + objective values + a bounded Rollout
  Trace projection. **Parent and child calls in one step name the SAME task
  IDs.**
* ``evaluate_subset(candidate)`` -- a full ``objective_task_count = 24``
  internal evaluation of an accepted child (and, on the first step, of each
  seed A/B); one Tool Result with 24 Rollout refs + per-instance/objective
  values that update the frontiers.

The adapter issues these Tool Calls through the :class:`RuntimeToolHandle`s the
harness constructs at the execution boundary. Every Step Result references
every Tool Result Object Reference + Content Hash and its terminal Tool Call
Store Entry (the harness records that evidence); the adapter carries the
restart minimum in its typed ``state_delta``/candidate output. Acceptance is
``same_minibatch_strict_pareto/v1``: accept iff the child, on the parent's
exact
minibatch, is no worse on either objective and strictly better on at least one
(correctness up, compression down).

Durable optimizer state (proposal catalog, per-instance/objective frontiers,
RNG state, budgets) crosses Step boundaries only by reference: the adapter
reads it from the immutable request ``pools`` the harness threaded from the
prior Step's state snapshot, and returns the post-step state as a typed
``state_delta`` the harness content-addresses. No mutable in-memory object is
the restart authority.
"""

from __future__ import annotations

import hashlib
from typing import Any

from whetstone.optimization.adapters import AdapterOutput, ToolCallRecord
from whetstone.optimization.mutation import DiffCheckError, diff_check
from whetstone.optimization.proposer import (
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

__all__ = [
    "ACCEPTANCE_POLICY",
    "GEPA_VARIANT",
    "TOOL_EVALUATE_MINIBATCH",
    "TOOL_EVALUATE_SUBSET",
    "GepaAdapter",
    "GepaHyperparameters",
    "strict_pareto_accepts",
]

GEPA_VARIANT = "whetstone_multi_objective/v1"
ACCEPTANCE_POLICY = "same_minibatch_strict_pareto/v1"
TOOL_EVALUATE_MINIBATCH = "evaluate_minibatch"
TOOL_EVALUATE_SUBSET = "evaluate_subset"

# Objective directions for the two-objective Pareto test.
#   correctness (pass rate) -> maximize; compression length -> minimize.
_CORRECTNESS = "correctness"
_COMPRESSION = "compression"


class GepaHyperparameters:
    """The GEPA algorithm hyperparameters that control search.

    Only these four control GEPA's search behavior (``gepa-run.html``
    hyperparameter registry); the rest are Eval Config / cardinality settings
    the harness owns.
    """

    __slots__ = (
        "max_optimization_rollouts",
        "max_reflection_attempts_per_step",
        "max_reflection_lm_calls",
        "minibatch_size",
    )

    def __init__(
        self,
        *,
        minibatch_size: int = 3,
        max_reflection_lm_calls: int = 8,
        max_reflection_attempts_per_step: int = 3,
        max_optimization_rollouts: int = 400,
    ) -> None:
        self.minibatch_size = minibatch_size
        self.max_reflection_lm_calls = max_reflection_lm_calls
        self.max_reflection_attempts_per_step = (
            max_reflection_attempts_per_step
        )
        self.max_optimization_rollouts = max_optimization_rollouts


def strict_pareto_accepts(
    *,
    parent: dict[str, float],
    child: dict[str, float],
) -> bool:
    """The ``same_minibatch_strict_pareto/v1`` acceptance test.

    Accept iff the child is **no worse** than the parent on either objective
    and **strictly better** on at least one, with correctness maximized and
    compression minimized. Equality, a trade-off, or a regression rejects.
    """
    # Normalize to "higher is better" per objective: correctness as-is,
    # compression negated (shorter is better).
    p_corr, c_corr = parent[_CORRECTNESS], child[_CORRECTNESS]
    p_comp, c_comp = -parent[_COMPRESSION], -child[_COMPRESSION]
    no_worse = c_corr >= p_corr and c_comp >= p_comp
    strictly_better = c_corr > p_corr or c_comp > p_comp
    return no_worse and strictly_better


class GepaAdapter:
    """The tool-using GEPA adapter (one GEPA step per invocation).

    The reflection LM is reached through an optimizer-owned proposer route
    (:class:`ProposerConfig` + :class:`ProposerTransport`) injected at
    construction -- process-side compute, never serialized into the request
    identity, distinct from the encoder/decoder routes inside a Rollout.
    """

    def __init__(
        self,
        *,
        reflection_config: ProposerConfig,
        reflection_transport: ProposerTransport,
        hyperparameters: GepaHyperparameters | None = None,
    ) -> None:
        self._reflection_config = reflection_config
        self._reflection_transport = reflection_transport
        self._hp = hyperparameters or GepaHyperparameters()

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        minibatch_handle, subset_handle = self._resolve_handles(handles)
        state = _GepaState.restore(request)

        records: list[ToolCallRecord] = []
        # First step only: seed A and B with evaluate_subset.
        if state.is_seed_step:
            self._seed(request, subset_handle, state, records)

        # Select a parent from the frontier union (deterministic RNG).
        parent = state.sample_parent(request)

        # Fresh parent diagnostic on a fresh minibatch.
        task_ids = state.sample_minibatch(request, self._hp.minibatch_size)
        _parent_result, parent_objs = self._evaluate_minibatch(
            minibatch_handle, parent, task_ids, records
        )

        # Reflect & validate (bounded retry loop).
        child, reflection_evidence = self._reflect(request, parent, state)

        if child is None:
            # Exhausted reflection without a valid unique child is a
            # *successful* transition (not a failure): record the evidence
            # and continue.
            return state.finalize(
                request=request,
                records=records,
                accepted_child=None,
                parent=parent,
                task_ids=task_ids,
                parent_objectives=parent_objs,
                child_objectives=None,
                acceptance=None,
                reflection_evidence=reflection_evidence,
            )

        # Same-minibatch child call (exact parent diagnostic task IDs).
        _child_result, child_objs = self._evaluate_minibatch(
            minibatch_handle, child, task_ids, records
        )

        # Acceptance test (same_minibatch_strict_pareto/v1).
        accepted = strict_pareto_accepts(parent=parent_objs, child=child_objs)

        accepted_child: Candidate | None = None
        if accepted:
            # Accepted-child full evaluation (24 tasks).
            self._evaluate_subset(subset_handle, child, records)
            accepted_child = child

        return state.finalize(
            request=request,
            records=records,
            accepted_child=accepted_child,
            parent=parent,
            task_ids=task_ids,
            parent_objectives=parent_objs,
            child_objectives=child_objs,
            acceptance={
                "policy": ACCEPTANCE_POLICY,
                "decision": accepted,
                "parent_objectives": parent_objs,
                "child_objectives": child_objs,
            },
            reflection_evidence=reflection_evidence,
            child=child,
        )

    # -- handle resolution ---------------------------------------------------

    def _resolve_handles(
        self, handles: tuple[RuntimeToolHandle, ...]
    ) -> tuple[RuntimeToolHandle, RuntimeToolHandle]:
        by_name: dict[str, RuntimeToolHandle] = {}
        for handle in handles:
            by_name[handle.config.tool_name] = handle
        try:
            return (
                by_name[TOOL_EVALUATE_MINIBATCH],
                by_name[TOOL_EVALUATE_SUBSET],
            )
        except KeyError as exc:
            raise ValueError(
                "GEPA requires both evaluate_minibatch and evaluate_subset "
                f"Runtime Tool Handles; missing {exc}"
            ) from None

    # -- tool calls ----------------------------------------------------------

    def _evaluate_minibatch(
        self,
        handle: RuntimeToolHandle,
        candidate: Candidate,
        task_ids: tuple[str, ...],
        records: list[ToolCallRecord],
    ) -> tuple[ToolResult, dict[str, float]]:
        call = ToolCall(
            call_id=_call_id(TOOL_EVALUATE_MINIBATCH, candidate, task_ids),
            tool_config_hash=handle.tool_config_hash,
            store_namespace=handle.config.store_namespace,
            args={
                "model_route": candidate.base_ref,
                "template": candidate.payload.get("user_prompt_template", ""),
                "task_ids": list(task_ids),
            },
        )
        result = handle(call)
        records.append(ToolCallRecord(call=call, result=result))
        return result, _objectives(result)

    def _evaluate_subset(
        self,
        handle: RuntimeToolHandle,
        candidate: Candidate,
        records: list[ToolCallRecord],
    ) -> ToolResult:
        call = ToolCall(
            call_id=_call_id(TOOL_EVALUATE_SUBSET, candidate, ()),
            tool_config_hash=handle.tool_config_hash,
            store_namespace=handle.config.store_namespace,
            args={
                "model_route": candidate.base_ref,
                "template": candidate.payload.get("user_prompt_template", ""),
            },
        )
        result = handle(call)
        records.append(ToolCallRecord(call=call, result=result))
        return result

    def _seed(
        self,
        request: OptimizationStepRequest,
        subset_handle: RuntimeToolHandle,
        state: _GepaState,
        records: list[ToolCallRecord],
    ) -> None:
        for candidate in request.candidates:
            result = self._evaluate_subset(subset_handle, candidate, records)
            state.seed_candidate(candidate, _objectives(result))

    # -- reflection ----------------------------------------------------------

    def _reflect(
        self,
        request: OptimizationStepRequest,
        parent: Candidate,
        state: _GepaState,
    ) -> tuple[Candidate | None, dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        base_template = parent.payload.get("user_prompt_template", "")
        max_attempts = min(
            self._hp.max_reflection_attempts_per_step,
            self._hp.max_reflection_lm_calls - state.reflection_calls_used,
        )
        for attempt_index in range(max(max_attempts, 0)):
            nonce = f"{request.run_id}:{request.step_index}:{attempt_index}"
            proposal_request = ProposalRequest(
                proposal_mode="gepa_reflection",
                request_ordinal=state.reflection_calls_used,
                base_ref=parent.base_ref,
                base_template=base_template,
                context={
                    "nonce": nonce,
                    "parent_candidate_id": parent.candidate_id,
                },
            )
            drafts = self._reflection_transport.draft(
                self._reflection_config, proposal_request, 1
            )
            state.reflection_calls_used += 1
            draft = drafts[0]
            child = Candidate(
                candidate_id=f"G{request.step_index}-{attempt_index}",
                base_ref=parent.base_ref,
                payload={"user_prompt_template": draft.template},
            )
            attempt_record: dict[str, Any] = {
                "attempt_index": attempt_index,
                "nonce": nonce,
                "request_evidence": draft.request_evidence,
                "response_evidence": draft.response_evidence,
                "usage": draft.usage,
            }
            try:
                diff_check(base=parent, proposed=child)
            except DiffCheckError as exc:
                attempt_record["validation"] = {
                    "valid": False,
                    "reason": str(exc),
                }
                attempts.append(attempt_record)
                continue
            if state.is_duplicate(child):
                attempt_record["validation"] = {
                    "valid": False,
                    "reason": "duplicate proposal (already in catalog)",
                }
                attempts.append(attempt_record)
                continue
            attempt_record["validation"] = {"valid": True}
            attempt_record["child_candidate_id"] = child.candidate_id
            attempts.append(attempt_record)
            return child, {
                "attempts": attempts,
                "reflection_calls_used": state.reflection_calls_used,
                "accepted_attempt_index": attempt_index,
            }
        return None, {
            "attempts": attempts,
            "reflection_calls_used": state.reflection_calls_used,
            "accepted_attempt_index": None,
        }


def _objectives(result: ToolResult) -> dict[str, float]:
    """Extract the two objective values from an evaluation Tool Result.

    A refused Tool Result carries no measurement, so a caller must never reach
    here with one; this raises rather than fabricating objectives.
    """
    if result.output is None:
        raise ValueError(
            "a refused Tool Result carries no objective values: a refusal "
            "never masquerades as a measurement"
        )
    objs = result.output.get("objective_values")
    if not isinstance(objs, dict):
        raise ValueError("evaluation Tool Result carries no objective_values")
    return {
        _CORRECTNESS: float(objs[_CORRECTNESS]),
        _COMPRESSION: float(objs[_COMPRESSION]),
    }


def _call_id(
    tool_name: str, candidate: Candidate, task_ids: tuple[str, ...]
) -> str:
    """A stable, deterministic call ID for a GEPA tool invocation.

    Keyed by the tool, the candidate identity payload, and the exact minibatch
    task IDs, so a restart replays the identical call ID (idempotent in the
    Tool Call Store) and the parent/child same-minibatch calls stay distinct.
    """
    digest = hashlib.sha256()
    digest.update(tool_name.encode())
    digest.update(repr(candidate.identity_payload()).encode())
    digest.update(repr(list(task_ids)).encode())
    return f"{tool_name}-{digest.hexdigest()[:16]}"


class _GepaState:
    """The durable GEPA state restored from the request and re-serialized.

    Holds the append-only proposal catalog, the per-instance/objective
    frontiers, the RNG cursor, and the reflection-call counter. It is
    reconstructed from the immutable request ``pools`` (which the harness
    threaded from the prior Step's state snapshot) and re-emitted as a typed
    ``state_delta`` -- no mutable in-memory object is the restart authority.
    """

    def __init__(self, raw: dict[str, Any]) -> None:
        self._catalog: list[dict[str, Any]] = list(raw.get("catalog", []))
        self._frontier: dict[str, Any] = dict(raw.get("frontier", {}))
        self._seed_done: bool = bool(raw.get("seed_done", False))
        self.reflection_calls_used: int = int(
            raw.get("reflection_calls_used", 0)
        )
        self._rng_cursor: int = int(raw.get("rng_cursor", 0))

    @classmethod
    def restore(cls, request: OptimizationStepRequest) -> _GepaState:
        return cls(dict(request.pools.get("gepa_state", {})))

    @property
    def is_seed_step(self) -> bool:
        return not self._seed_done

    def seed_candidate(
        self, candidate: Candidate, objectives: dict[str, float]
    ) -> None:
        self._append_catalog(candidate, objectives, accepted=True)
        self._update_frontier(candidate, objectives)
        self._seed_done = True

    def is_duplicate(self, candidate: Candidate) -> bool:
        template = candidate.payload.get("user_prompt_template")
        return any(
            entry["base_ref"] == candidate.base_ref
            and entry["template"] == template
            for entry in self._catalog
        )

    def sample_parent(self, request: OptimizationStepRequest) -> Candidate:
        """Sample a parent deterministically from the frontier union."""
        members = self._frontier.get("members", [])
        if not members:
            raise ValueError(
                "GEPA cannot sample a parent: the frontier is empty (seed "
                "step must run evaluate_subset for the initial candidates)"
            )
        index = self._deterministic_index(request, len(members), "parent")
        chosen = members[index]
        return Candidate(
            candidate_id=chosen["candidate_id"],
            base_ref=chosen["base_ref"],
            payload={"user_prompt_template": chosen["template"]},
        )

    def sample_minibatch(
        self, request: OptimizationStepRequest, size: int
    ) -> tuple[str, ...]:
        """Sample stable minibatch task IDs deterministically.

        The parent diagnostic and the child comparison in this step both use
        these exact task IDs (the same-minibatch invariant).
        """
        pool = request.pools.get("task_pool")
        if isinstance(pool, list) and pool:
            span = len(pool)
            start = self._deterministic_index(request, span, "minibatch")
            return tuple(
                str(pool[(start + offset) % span]) for offset in range(size)
            )
        # No explicit pool: derive stable synthetic task IDs from the step.
        seed = self._deterministic_index(request, 10_000, "minibatch")
        return tuple(f"task-{seed + offset}" for offset in range(size))

    def _deterministic_index(
        self, request: OptimizationStepRequest, span: int, salt: str
    ) -> int:
        digest = hashlib.sha256()
        digest.update(request.run_id.encode())
        digest.update(str(request.step_index).encode())
        digest.update(str(self._rng_cursor).encode())
        digest.update(salt.encode())
        return int.from_bytes(digest.digest()[:8], "big") % span

    def _append_catalog(
        self,
        candidate: Candidate,
        objectives: dict[str, float],
        *,
        accepted: bool,
    ) -> None:
        self._catalog.append(
            {
                "candidate_id": candidate.candidate_id,
                "base_ref": candidate.base_ref,
                "template": candidate.payload.get("user_prompt_template"),
                "objectives": objectives,
                "accepted": accepted,
            }
        )

    def _update_frontier(
        self, candidate: Candidate, objectives: dict[str, float]
    ) -> None:
        members = self._frontier.setdefault("members", [])
        members.append(
            {
                "candidate_id": candidate.candidate_id,
                "base_ref": candidate.base_ref,
                "template": candidate.payload.get("user_prompt_template"),
                "objectives": objectives,
            }
        )

    def finalize(
        self,
        *,
        request: OptimizationStepRequest,
        records: list[ToolCallRecord],
        accepted_child: Candidate | None,
        parent: Candidate,
        task_ids: tuple[str, ...],
        parent_objectives: dict[str, float],
        child_objectives: dict[str, float] | None,
        acceptance: dict[str, Any] | None,
        reflection_evidence: dict[str, Any],
        child: Candidate | None = None,
    ) -> AdapterOutput:
        """Build the AdapterOutput carrying the Step Result restart minimum.

        Every applicable restart-minimum field is present and typed; an
        inapplicable field (e.g. no accepted child) is an explicit ``None``,
        never inferred from a missing event.
        """
        if child is not None:
            objectives = child_objectives or {}
            self._append_catalog(
                child, objectives, accepted=accepted_child is not None
            )
        if accepted_child is not None and child_objectives is not None:
            self._update_frontier(accepted_child, child_objectives)

        self._rng_cursor += 1

        accepted_count = sum(
            1 for entry in self._catalog if entry["accepted"]
        )
        status = self._status(request, accepted_count)

        state_delta = {
            "gepa_state": {
                "catalog": self._catalog,
                "frontier": self._frontier,
                "seed_done": self._seed_done,
                "reflection_calls_used": self.reflection_calls_used,
                "rng_cursor": self._rng_cursor,
            },
            # The Step Result restart minimum.
            "restart_minimum": {
                "run_id": request.run_id,
                "step_id": request.step_id,
                "step_index": request.step_index,
                "optimizer_config_hash": request.optimizer_config_hash,
                "gepa_variant": GEPA_VARIANT,
                "acceptance_policy": ACCEPTANCE_POLICY,
                "rng_cursor": self._rng_cursor,
                "parent": parent.identity_payload(),
                "sampled_task_ids": list(task_ids),
                "parent_objectives": parent_objectives,
                "child": child.identity_payload() if child else None,
                "child_objectives": child_objectives,
                "reflection_evidence": reflection_evidence,
                "acceptance": acceptance,
                "accepted_child": (
                    accepted_child.identity_payload()
                    if accepted_child
                    else None
                ),
                "catalog_size": len(self._catalog),
                "frontier_size": len(self._frontier.get("members", [])),
                "reflection_calls_used": self.reflection_calls_used,
                "tool_call_count": len(records),
            },
        }

        accepted_candidates = self._return_candidates(request)
        return AdapterOutput(
            proposed_candidates=accepted_candidates,
            accepted_candidates=accepted_candidates,
            tool_call_records=tuple(records),
            proposed_status=status,
            state_delta=state_delta,
        )

    def _status(
        self, request: OptimizationStepRequest, accepted_count: int
    ) -> StepStatus:
        if self._enough_returnable(request):
            return StepStatus.COMPLETE
        if self.reflection_calls_used >= self._reflection_cap(request):
            # Reflection budget exhausted without enough proposals: the run
            # cannot satisfy its contract -> failed (blocks materialization).
            return StepStatus.FAILED
        return StepStatus.CONTINUE

    @staticmethod
    def _reflection_cap(request: OptimizationStepRequest) -> int:
        return int(request.hyperparameters.get("max_reflection_lm_calls", 8))

    def _enough_returnable(self, request: OptimizationStepRequest) -> bool:
        target = request.output_contract.returned_proposal_count
        return len(self._accepted_children()) >= target

    def _accepted_children(self) -> list[dict[str, Any]]:
        # Accepted, fully evaluated catalog entries (the seed A/B included:
        # they are accepted and fully evaluated, and eligible for return).
        return [entry for entry in self._catalog if entry["accepted"]]

    def _return_candidates(
        self, request: OptimizationStepRequest
    ) -> tuple[Candidate, ...]:
        target = request.output_contract.returned_proposal_count
        # Order accepted catalog entries deterministically: correctness desc,
        # compression asc, candidate_id asc (a stable return policy).
        accepted = self._accepted_children()
        ordered = sorted(
            accepted,
            key=lambda e: (
                -float(e["objectives"][_CORRECTNESS]),
                float(e["objectives"][_COMPRESSION]),
                e["candidate_id"],
            ),
        )
        return tuple(
            Candidate(
                candidate_id=entry["candidate_id"],
                base_ref=entry["base_ref"],
                payload={"user_prompt_template": entry["template"]},
            )
            for entry in ordered[:target]
        )
