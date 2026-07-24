"""The durable Whetstone rollout-execution stage body.

This is the Whetstone-executor plane of Workstream 3's Durable Retry
Implementation. dr-platform owns the generic Platform Stage Attempt (one DBOS
workflow identity per attempt) and the ``retry_stage`` recovery; Whetstone owns
the bounded semantic retry *inside* one Platform Stage Attempt; DBOS owns step
checkpoints, deterministic replay, and durable sleep.

The stage body (:meth:`ExecutorContext.run_stage`) is registered as a linear
pipeline stage callable through dr-platform's ``wrap_pipeline_workflows``, so
each Platform Stage Attempt runs it under one ``@DBOS.workflow`` identity. Its
shape, per the design:

* **One retry-disabled step per Provider Call Attempt.** Each logical provider
  attempt runs in one ``@DBOS.step(retries_allowed=False)`` that calls
  dr-providers (native retries zero) and returns a serializable checkpointed
  Provider Call Attempt observation. Automatic DBOS step retries are disabled;
  Whetstone alone bounds, branches, and backs off.

* **Durable sleep for backoff.** Pre-attempt backoff uses ``DBOS.sleep`` so a
  crash mid-backoff resumes deterministically rather than re-sleeping from
  wall-clock.

* **Deterministic replay.** On DBOS recovery within the same Platform Stage
  Attempt, completed steps replay from their checkpoints; the driver
  reconstructs the identical attempt sequence and the Node Outcomes / Graph Run
  Result are rebuilt from those checkpointed observations. **No completed Node
  Output is injected across workflow identities.**

* **Terminal assembly and binding.** After semantic success *or* exhaustion the
  body deterministically builds Node Outcomes and the Graph Run Result, nests
  them in one immutable Rollout Result, immutably ``put``s it into dr-store,
  and atomically binds its Rollout Execution Key through the authoritative
  Result Store.

* **Result-based terminality.** The stage is operationally SUCCEEDED iff a
  terminal Result (semantic success OR exhausted failure) is persisted and
  bound/confirmed. It is FAILED only when the body cannot produce, persist,
  bind, or confirm that exact terminal Result (those cases raise, and the
  dr-platform handoff wrapper records the FAILED outcome).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dbos import DBOS
from dr_graph import (
    GraphRunResult,
    GraphRunStatus,
    NodeOutcome,
    NodeOutput,
)

from whetstone.orchestration.work_request import (
    RolloutWorkRequest,
    encode_object_reference,
)
from whetstone.provider.attempt import ProviderCallResult
from whetstone.provider.classification import Generation
from whetstone.provider.driver import run_provider_call
from whetstone.result import (
    ExhaustedCausalFailure,
    PlatformStageAttemptEvidence,
    ProviderCallAttemptObservation,
    ResultBindStatus,
    ResultStore,
    RolloutResult,
    persist_rollout_result,
    rollout_result_reference,
)
from whetstone.result.result_store import ResultStoreConflictError

if TYPE_CHECKING:
    from dr_providers import ProviderCallRequest

    from whetstone.provider.driver import TransportCall
    from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = [
    "ExecutorContext",
    "RolloutExecutionOutcome",
    "RolloutWorkResolver",
    "TerminalBindError",
]

#: Resolves the opaque ``WorkInput.input_ref`` string to the immutable Rollout
#: Work Request it references. Whetstone owns this resolution; the platform
#: never parses the reference.
RolloutWorkResolver = Callable[[str], RolloutWorkRequest]

#: Builds the concrete Provider Call Request for one Work Request. Injected so
#: the executor stays free of dr-code / LM-boundary request-construction
#: details; tests supply a direct builder.
ProviderRequestBuilder = Callable[
    [RolloutWorkRequest], "ProviderCallRequest"
]


class TerminalBindError(RuntimeError):
    """The terminal Result could not be produced, persisted, or confirmed.

    Raised so the dr-platform stage wrapper records the Stage as operationally
    FAILED. Semantic exhaustion is NOT this error — an exhausted-failure Result
    that is persisted and bound is a SUCCEEDED Stage.
    """


@dataclass(frozen=True, slots=True)
class RolloutExecutionOutcome:
    """The in-process return of one rollout-execution stage body run.

    ``output_reference`` is the opaque typed Rollout Result Object Reference
    string the stage returns to dr-platform. ``bind_status`` records whether
    this run acquired the binding or replayed an existing identical one.
    """

    output_reference: str
    reference_schema: str
    bind_status: ResultBindStatus
    #: Whether the terminal Result was a semantic success. ``None`` on the
    #: idempotent retried-executor stop (this attempt did not re-derive it —
    #: the durable winner already holds the answer).
    semantic_success: bool | None


@dataclass(frozen=True)
class ExecutorContext:
    """Bound dependencies for the durable rollout-execution stage body.

    A DBOS-registered stage callable receives only positional submission args,
    so the executor's transport/result-store/request-builder dependencies are
    bound here once (before launch) and closed over by :meth:`stage_callable`.

    Attributes:
        result_store: the authoritative Whetstone Result Store (over dr-store).
        transport: the injectable dr-providers transport callable, invoked
            inside one retry-disabled DBOS step per Provider Call Attempt.
        policy: the composing Provider Execution Policy (bounds + backoff +
            per-class retry eligibility).
        resolve_work_request: resolves the opaque input-ref to its Work
            Request.
        build_request: builds the Provider Call Request from the Work Request.
        terminal_node_id: the graph's unique terminal/sink Node id.
        llm_node_id: the LLM Call Node id the provider call drives.
        use_durable_sleep: inject ``DBOS.sleep`` as the driver's backoff sleep
            (the default). Tests off the DBOS runtime set this False.
    """

    result_store: ResultStore
    transport: TransportCall
    policy: ProviderExecutionPolicy
    resolve_work_request: RolloutWorkResolver
    build_request: ProviderRequestBuilder
    terminal_node_id: str = "terminal"
    llm_node_id: str = "llm_call"
    use_durable_sleep: bool = True

    def stage_callable(self) -> Callable[[str], str]:
        """Return the plain stage callable for ``wrap_pipeline_workflows``.

        The returned callable takes the opaque ``input_ref`` and returns the
        opaque output-reference string. dr-platform's handoff wrapper adds the
        ``@DBOS.workflow`` identity and records SUCCEEDED/FAILED from a normal
        return / raise.
        """

        def run_stage(input_ref: str) -> str:
            return self.run_stage(input_ref).output_reference

        return run_stage

    def run_stage(self, input_ref: str) -> RolloutExecutionOutcome:
        """Execute one rollout-execution Stage Attempt.

        Resolves the Work Request, **rechecks the authoritative binding before
        any new semantic effect** (a concurrent actor may have won after an
        operator-retry gateway precheck), runs the durable bounded provider
        loop, assembles the terminal Rollout Result, and persists+binds it.
        Returns the opaque output reference; raises :class:`TerminalBindError`
        only when it cannot produce/persist/bind/confirm the exact terminal
        Result.

        If the recheck finds the key already bound, the executor stops via the
        idempotent-or-conflict rule: it issues no new provider call and returns
        the existing authoritative binding without overwriting it.
        """
        request = self.resolve_work_request(input_ref)
        idempotent = self._stop_if_already_bound(request)
        if idempotent is not None:
            return idempotent
        provider_result = self._run_durable_provider_loop(request)
        rollout_result = self._assemble_rollout_result(
            request=request,
            provider_result=provider_result,
        )
        return self._persist_and_bind(
            request=request,
            rollout_result=rollout_result,
            semantic_success=provider_result.succeeded,
        )

    def _stop_if_already_bound(
        self, request: RolloutWorkRequest
    ) -> RolloutExecutionOutcome | None:
        """Stop idempotently when the key already won a binding.

        The retried-executor recheck. If the Rollout Execution Key is already
        bound (a concurrent actor won after any operator-retry gateway
        precheck), the executor issues no new provider call and returns the
        existing authoritative binding as an idempotent success — it never
        overwrites the durable winner. Returns ``None`` when the key is unbound
        so the caller proceeds to the provider loop.
        """
        existing = self.result_store.resolve(request.rollout_execution_key)
        if existing is None:
            return None
        return RolloutExecutionOutcome(
            output_reference=encode_object_reference(existing),
            reference_schema=existing.schema,
            bind_status=ResultBindStatus.IDEMPOTENT,
            semantic_success=None,
        )

    # -- durable provider loop ------------------------------------------------

    def _run_durable_provider_loop(
        self, request: RolloutWorkRequest
    ) -> ProviderCallResult:
        """Run the bounded attempt loop with one DBOS step per attempt.

        Reuses the pure stage-03 driver (``run_provider_call``) with the real
        transport wrapped in one ``@DBOS.step(retries_allowed=False)`` so every
        logical Provider Call Attempt's typed observation is checkpointed and
        replayed. Backoff sleeps durably via ``DBOS.sleep`` (deterministic
        across recovery). Whetstone alone bounds/branches/backs off; DBOS
        automatic step retries stay disabled.
        """
        call_request = self.build_request(request)
        logical_call_id = self._logical_call_id(request)
        return run_provider_call(
            request=call_request,
            policy=self.policy,
            transport=self._durable_transport(),
            logical_call_id=logical_call_id,
            sleep=self._durable_sleep,
        )

    def _durable_transport(self) -> TransportCall:
        """Wrap the injected transport in one retry-disabled DBOS step.

        Each physical provider invocation checkpoints exactly one Provider
        Invocation Evidence; on replay the checkpoint is reused rather than
        re-issuing the wire call. Automatic DBOS step retries are disabled
        (``retries_allowed=False``), so Whetstone's loop is the only retry
        authority. Programming/infrastructure errors inside the step raise
        (they are not classified transport outcomes).
        """
        transport = self.transport

        @DBOS.step(retries_allowed=False)
        def provider_call_step(
            request: ProviderCallRequest,
        ) -> Any:
            return transport(request)

        return provider_call_step

    def _durable_sleep(self, seconds: float) -> None:
        if seconds > 0 and self.use_durable_sleep:
            DBOS.sleep(seconds)

    # -- terminal assembly ----------------------------------------------------

    def _assemble_rollout_result(
        self,
        *,
        request: RolloutWorkRequest,
        provider_result: ProviderCallResult,
    ) -> RolloutResult:
        """Deterministically build the terminal Rollout Result.

        Reconstructs the completed Provider Call Attempt observations, the
        per-Node Outcomes, and the nested Graph Run Result from the
        checkpointed provider result, then nests everything in one immutable
        Rollout Result. The Graph Run Result *references* the provider bodies
        held here; it never duplicates them and holds no Platform Stage state.
        No completed Node Output is injected from another workflow identity —
        every observation comes from this attempt's own checkpoints.
        """
        observations = self._observations(provider_result)
        evidence_refs = tuple(obs.evidence_ref for obs in observations)
        key = request.rollout_execution_key
        rollout_key = key.rollout_key

        terminal_generation = (
            provider_result.generation
            if provider_result.succeeded
            else None
        )
        node_outcomes = self._node_outcomes(
            terminal_generation=terminal_generation,
        )
        graph_run_result = GraphRunResult(
            graph_hash=rollout_key.graph_hash,
            external_inputs=dict(request.task_inputs),
            status=(
                GraphRunStatus.SUCCESS
                if provider_result.succeeded
                else GraphRunStatus.ERROR
            ),
            outcomes={
                outcome.node_id: outcome for outcome in node_outcomes
            },
            execution_order=tuple(
                outcome.node_id for outcome in node_outcomes
            ),
            terminal_node_id=self.terminal_node_id,
            terminal_output=(
                None
                if terminal_generation is None
                else {"text": terminal_generation.text}
            ),
            terminal_error=self._terminal_error(provider_result),
            attempt_evidence_refs=evidence_refs,
            provenance={"logical_call_id": provider_result.logical_call_id},
        )

        exhausted_failure = self._exhausted_failure(provider_result)
        scores = self._success_scores(terminal_generation)

        return RolloutResult(
            rollout_execution_key=key,
            graph_config_ref=request.graph_config_ref,
            graph_hash=rollout_key.graph_hash,
            eval_config_ref=request.evaluation_context_ref,
            eval_config_hash=rollout_key.eval_config_hash,
            evaluation_context_id=key.evaluation_context_id,
            input_identities=dict(request.task_inputs),
            graph_run_result=graph_run_result,
            scores=scores,
            exhausted_failure=exhausted_failure,
            provider_call_attempts=observations,
            stage_attempt_evidence=self._stage_attempt_evidence(),
        )

    def _observations(
        self, provider_result: ProviderCallResult
    ) -> tuple[ProviderCallAttemptObservation, ...]:
        """Project the checkpointed attempts into Rollout Result slots.

        Each observation carries a stable evidence ref (the Content Hash of the
        attempt's Provider Invocation Evidence), the logical identity, attempt
        number, policy identity, latency, classification, and the full evidence
        JSON — the provider bodies live here on the enclosing Rollout Result.
        """
        observations: list[ProviderCallAttemptObservation] = []
        for attempt in provider_result.attempts:
            evidence_json = attempt.evidence.to_stable_dict()
            observations.append(
                ProviderCallAttemptObservation(
                    evidence_ref=self._evidence_ref(attempt.evidence),
                    logical_call_id=attempt.logical_call_id,
                    attempt_number=attempt.attempt_number,
                    provider_execution_policy_ref=(
                        attempt.execution_policy_hash
                    ),
                    semantic_classification=self._classification_label(
                        attempt
                    ),
                    latency_ms=attempt.latency_ms,
                    provider_invocation_evidence=evidence_json,
                )
            )
        return tuple(observations)

    @staticmethod
    def _evidence_ref(evidence: Any) -> str:
        """A stable within-record reference for one attempt's evidence.

        The Graph Run Result's ``attempt_evidence_refs`` point at this string;
        it must equal one observation's ``evidence_ref``. A Content Hash of the
        stable evidence dict is deterministic and collision-free.
        """
        from dr_serialize import sha256_json_digest

        return sha256_json_digest(evidence.to_stable_dict())

    @staticmethod
    def _classification_label(attempt: Any) -> str:
        if attempt.succeeded:
            return "generation"
        return attempt.failure_class.value

    def _node_outcomes(
        self,
        *,
        terminal_generation: Generation | None,
    ) -> tuple[NodeOutcome, ...]:
        """Reconstruct per-Node Outcomes for the (single-LLM) graph.

        The concrete graph is one LLM Call Node feeding the terminal Node. On
        semantic success both succeed and the terminal Output is the accepted
        Generation text; on exhaustion the LLM Node errored, so the terminal
        Node is not injected a success Output (no cross-identity Node Output).
        """
        if terminal_generation is not None:
            llm_outcome = NodeOutcome.success(
                node_id=self.llm_node_id,
                output=NodeOutput(
                    values={"text": terminal_generation.text}
                ),
            )
            terminal_outcome = NodeOutcome.success(
                node_id=self.terminal_node_id,
                output=NodeOutput(
                    values={"text": terminal_generation.text}
                ),
            )
            return (llm_outcome, terminal_outcome)
        return ()

    def _terminal_error(
        self, provider_result: ProviderCallResult
    ) -> Any | None:
        if provider_result.succeeded:
            return None
        from dr_graph.results import NodeError, TerminalError
        from dr_graph.results import NodeOutcomeStatus as Status

        failure = provider_result.semantic_failure
        assert failure is not None
        return TerminalError(
            node_id=self.terminal_node_id,
            status=Status.ERROR,
            error=NodeError(
                error_type="whetstone.provider.exhausted",
                message=failure.message,
                failure_class=failure.failure_class.value,
            ),
        )

    def _exhausted_failure(
        self, provider_result: ProviderCallResult
    ) -> ExhaustedCausalFailure | None:
        if provider_result.succeeded:
            return None
        failure = provider_result.semantic_failure
        assert failure is not None
        return ExhaustedCausalFailure(
            failure_class=failure.failure_class.value,
            failure_exception_type="whetstone.provider.ProviderSemanticFailure",
            underlying_exception_type=(
                "dr_providers.ProviderTransportFailure"
                if failure.transport_failure is not None
                else "dr_providers.ProviderTransportResponse"
            ),
            message=failure.message,
            failure_metadata={
                "attempt_count": provider_result.attempt_count,
            },
        )

    @staticmethod
    def _success_scores(
        terminal_generation: Generation | None,
    ) -> tuple[Any, ...]:
        """On semantic success carry a minimal placeholder Score.

        Full dr-code Metric Facts / Scores extraction is a downstream concern;
        the terminal Result requires *some* measurement on success so the
        record is a complete success Result rather than an empty one.
        """
        if terminal_generation is None:
            return ()
        from whetstone.result import ScoreFact

        return (
            ScoreFact(
                name="generation_length",
                value=len(terminal_generation.text),
            ),
        )

    def _stage_attempt_evidence(self) -> PlatformStageAttemptEvidence:
        """Record the surrounding Platform Stage Attempt / DBOS identity.

        Evidence only — no Platform Stage *state* is owned here. The DBOS
        workflow id is the per-attempt identity; a Durability Replay never
        becomes a new Whetstone semantic retry.
        """
        workflow_id = DBOS.workflow_id if _in_dbos_workflow() else None
        return PlatformStageAttemptEvidence(
            dbos_workflow_id=workflow_id,
        )

    # -- persistence + binding ------------------------------------------------

    def _persist_and_bind(
        self,
        *,
        request: RolloutWorkRequest,
        rollout_result: RolloutResult,
        semantic_success: bool,
    ) -> RolloutExecutionOutcome:
        """Persist the immutable Result and bind its Rollout Execution Key.

        Result-based terminality: a persisted-and-bound success OR
        exhausted-failure Result makes the Stage SUCCEEDED. A different-
        reference conflict on an already-bound key is NOT overwritten and NOT
        an operational failure of *producing* the result — but this body did
        not win the binding, so it surfaces the conflict as a terminal-bind
        error (a concurrent winner exists; this attempt must not proceed as
        though it produced the terminal Result).
        """
        expected_schema = request.expected_schema_identities
        reference = rollout_result_reference(rollout_result)
        if reference.schema != expected_schema.rollout_result_schema:
            raise TerminalBindError(
                "assembled Rollout Result schema "
                f"{reference.schema!r} does not match the expected schema "
                f"{expected_schema.rollout_result_schema!r}"
            )
        try:
            bound_reference, status = persist_rollout_result(
                self.result_store, rollout_result
            )
        except ResultStoreConflictError as conflict:
            raise TerminalBindError(
                "Rollout Execution Key already bound to a different Result; "
                "this attempt did not produce the terminal Result"
            ) from conflict
        except Exception as error:
            raise TerminalBindError(
                "could not persist or bind the terminal Rollout Result"
            ) from error

        # Confirm the exact reference is the authoritative binding.
        confirmed = self.result_store.resolve(request.rollout_execution_key)
        if confirmed != bound_reference:
            raise TerminalBindError(
                "the terminal Rollout Result reference is not the confirmed "
                "authoritative binding"
            )
        return RolloutExecutionOutcome(
            output_reference=encode_object_reference(bound_reference),
            reference_schema=bound_reference.schema,
            bind_status=status,
            semantic_success=semantic_success,
        )

    # -- identity -------------------------------------------------------------

    @staticmethod
    def _logical_call_id(request: RolloutWorkRequest) -> str:
        """A stable logical-call identity for the Work Request's provider call.

        Deterministic in the Rollout Execution Key so replays and re-derivation
        produce the same logical call id.
        """
        from whetstone.result import encode_rollout_execution_key

        return "llm:" + encode_rollout_execution_key(
            request.rollout_execution_key
        )


def _in_dbos_workflow() -> bool:
    try:
        return DBOS.workflow_id is not None
    except Exception:
        return False
