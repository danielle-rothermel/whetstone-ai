"""The immutable terminal Rollout Result schema.

A :class:`RolloutResult` is the immutable terminal record that maps directly
to one Rollout and its Rollout Execution Key. Per the Settled Graph and
Evaluation Contract (``design/concrete-changes.html``) and the vocabulary
(``design/vocab_and_defs.html`` — *Rollout Result*, *Graph Run Result*), it
contains:

* the Rollout Execution Key;
* the Graph Config reference and full Graph Hash;
* the Eval Config reference and full Identity Hash;
* the Evaluation Context and authority identities;
* the input identities (Graph External Input / task identities);
* a nested native dr-graph :class:`~dr_graph.GraphRunResult`;
* Metric Facts and named Scores **or** an exhausted causal failure
  (exactly one of the two, never both);
* Provider Call Attempt observation slots;
* Platform Stage Attempt and Durability Replay evidence slots;
* record-local typed provenance fields.

Two invariants are load-bearing and tested:

* **No Materialization Record reference.** Materialization lineage is
  excluded from the Rollout Result; the schema has no field that could carry
  a Materialization Record Object Reference.
* **Nested Graph Run Result references — never duplicates — provider
  bodies.** The completed Provider Call Attempt observations (with their
  provider bodies) are held once, here on the enclosing Rollout Result. The
  nested :class:`~dr_graph.GraphRunResult` carries only
  ``attempt_evidence_refs`` pointing back at those observations; it holds no
  Platform Stage state and has no separate authoritative persistence path
  (it is persisted only as part of this enclosing record).
"""

from __future__ import annotations

from typing import Any

from dr_graph import GraphRunResult
from dr_store import ObjectReference
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    model_validator,
)

from whetstone.graph.rollout import RolloutExecutionKey
from whetstone.result.schema import (
    ROLLOUT_RESULT_SCHEMA,
    require_full_hash,
)

__all__ = [
    "ROLLOUT_RESULT_SCHEMA",
    "ExhaustedCausalFailure",
    "PlatformStageAttemptEvidence",
    "ProviderCallAttemptObservation",
    "RolloutResult",
    "ScoreFact",
    "rollout_result_reference",
]


class ProviderCallAttemptObservation(BaseModel):
    """One completed Provider Call Attempt observation slot.

    This is the DBOS-checkpointed Whetstone logical-attempt wrapper carrying
    the logical call identity, attempt number, Provider Execution Policy
    identity, timing, its stable Provider Invocation Evidence reference, and
    its Whetstone semantic classification. The **provider bodies live here**
    on the enclosing Rollout Result; the nested Graph Run Result references
    this observation by ``evidence_ref`` rather than duplicating the body.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Stable reference the nested Graph Run Result points back at.
    evidence_ref: StrictStr
    logical_call_id: StrictStr
    attempt_number: int
    provider_execution_policy_ref: StrictStr | None = None
    semantic_classification: StrictStr | None = None
    latency_ms: int | None = None
    # Stable Provider Invocation Evidence artifact (typed transport outcome
    # plus complete least-processed raw bodies), carried as JSON.
    provider_invocation_evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> ProviderCallAttemptObservation:
        if not self.evidence_ref:
            raise ValueError("evidence_ref must be non-empty")
        if not self.logical_call_id:
            raise ValueError("logical_call_id must be non-empty")
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be a positive integer")
        return self


class PlatformStageAttemptEvidence(BaseModel):
    """Platform Stage Attempt and Durability Replay evidence slot.

    This records *evidence about* the generic Platform Stage Attempt and
    DBOS Durability Replay that surrounded the Rollout — the append-row
    stage-attempt identity and the DBOS workflow/replay identity. It is
    evidence only: no Platform Stage *state* is owned or reconstructed here,
    and none of it enters the nested Graph Run Result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    platform_stage_attempt_id: StrictStr | None = None
    dbos_workflow_id: StrictStr | None = None
    # Durability Replay evidence: how many DBOS recoveries were observed
    # under the same Platform Stage Attempt / workflow identity. A replay is
    # never a new Whetstone semantic retry.
    durability_replay_count: int = 0

    @model_validator(mode="after")
    def _validate(self) -> PlatformStageAttemptEvidence:
        if self.durability_replay_count < 0:
            raise ValueError("durability_replay_count cannot be negative")
        return self


class ScoreFact(BaseModel):
    """A named Score derived from Metric Facts.

    The measured Metric Facts and their derived named Scores are dr-code's
    types; Whetstone does not redefine them. They are carried here as their
    JSON projections so the terminal Rollout Result is a complete,
    self-describing, canonicalizable record without importing a second copy
    of the dr-code schema into the persisted identity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: StrictStr
    value: float | int | str | bool | None = None
    # Derivation lineage projected from the dr-code Score.
    lineage: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> ScoreFact:
        if not self.name:
            raise ValueError("Score name must be non-empty")
        return self


class ExhaustedCausalFailure(BaseModel):
    """The terminal exhausted causal failure alternative to facts/scores.

    A semantic failure that Whetstone exhausted (bounded retries spent) is an
    expected domain output, not an error: a Rollout Result that carries this
    instead of Metric Facts/Scores is still a complete terminal Result, and
    a persisted-and-bound exhausted-failure Result makes its Platform Stage
    operationally SUCCEEDED.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_class: StrictStr
    failure_exception_type: StrictStr
    underlying_exception_type: StrictStr
    message: StrictStr
    failure_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> ExhaustedCausalFailure:
        if not self.failure_class:
            raise ValueError("failure_class must be non-empty")
        return self


class RolloutResult(BaseModel):
    """Immutable terminal result for one Rollout.

    Maps directly to one Rollout and its Rollout Execution Key. Exactly one
    of (``metric_facts`` + ``scores``) or ``exhausted_failure`` is populated:
    a semantic success carries the facts and named Scores; an exhausted
    causal failure carries the failure instead. Materialization lineage is
    excluded — there is deliberately no Materialization Record reference
    field, and no official-specific result role or type.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Rollout Execution Key (the terminal execution identity + Result Store
    # uniqueness key). Carried as the nested typed key so the persisted
    # record is self-describing.
    rollout_execution_key: RolloutExecutionKey

    # Rollout Variant config: Graph Config reference + full Graph Hash.
    graph_config_ref: StrictStr
    graph_hash: StrictStr

    # Eval Config reference + full Identity Hash.
    eval_config_ref: StrictStr
    eval_config_hash: StrictStr

    # Evaluation Context and authority identities.
    evaluation_context_id: StrictStr
    authority: StrictStr | None = None

    # Input identities (Graph External Input / task identities that fed the
    # run, as immutable identities/references — not bodies).
    input_identities: dict[str, Any] = Field(default_factory=dict)

    # Nested native dr-graph Graph Run Result. It references — never
    # duplicates — provider bodies held by ``provider_call_attempts`` below.
    graph_run_result: GraphRunResult

    # Semantic outcome: EITHER facts + scores OR an exhausted causal failure.
    metric_facts: tuple[dict[str, Any], ...] = ()
    scores: tuple[ScoreFact, ...] = ()
    exhausted_failure: ExhaustedCausalFailure | None = None

    # Provider Call Attempt observation slots (provider bodies live here).
    provider_call_attempts: tuple[ProviderCallAttemptObservation, ...] = ()

    # Platform Stage Attempt / Durability Replay evidence slot.
    stage_attempt_evidence: PlatformStageAttemptEvidence = Field(
        default_factory=PlatformStageAttemptEvidence
    )

    # Record-local typed provenance fields (no universal Provenance class).
    provenance_note: StrictStr | None = None
    provenance_ordinal: int | None = None
    producing_versions: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> RolloutResult:
        require_full_hash(self.graph_hash, field="graph_hash")
        require_full_hash(self.eval_config_hash, field="eval_config_hash")
        # The nested key's graph/eval hashes must name the same cell.
        rollout_key = self.rollout_execution_key.rollout_key
        if rollout_key.graph_hash != self.graph_hash:
            raise ValueError(
                "rollout_execution_key.rollout_key.graph_hash must match "
                "graph_hash"
            )
        if rollout_key.eval_config_hash != self.eval_config_hash:
            raise ValueError(
                "rollout_execution_key.rollout_key.eval_config_hash must "
                "match eval_config_hash"
            )
        if (
            self.rollout_execution_key.evaluation_context_id
            != self.evaluation_context_id
        ):
            raise ValueError(
                "rollout_execution_key.evaluation_context_id must match "
                "evaluation_context_id"
            )
        # The nested Graph Run Result names the same Graph Run identity.
        if self.graph_run_result.graph_hash != self.graph_hash:
            raise ValueError(
                "nested graph_run_result.graph_hash must match graph_hash"
            )

        # Exactly one of facts/scores or an exhausted causal failure.
        has_measurement = bool(self.metric_facts) or bool(self.scores)
        has_failure = self.exhausted_failure is not None
        if has_measurement and has_failure:
            raise ValueError(
                "a Rollout Result carries EITHER Metric Facts/Scores OR an "
                "exhausted causal failure, never both"
            )
        if not has_measurement and not has_failure:
            raise ValueError(
                "a Rollout Result must carry Metric Facts/Scores or an "
                "exhausted causal failure"
            )

        # The nested Graph Run Result references — never duplicates —
        # provider bodies: every attempt-evidence ref it carries MUST resolve
        # to a Provider Call Attempt observation held here.
        held_evidence_refs = {
            attempt.evidence_ref for attempt in self.provider_call_attempts
        }
        for ref in self.graph_run_result.attempt_evidence_refs:
            if ref not in held_evidence_refs:
                raise ValueError(
                    "nested graph_run_result references attempt evidence "
                    f"{ref!r} that is not held by an enclosing Provider Call "
                    "Attempt observation; the Graph Run Result must reference "
                    "provider bodies held by the Rollout Result, never "
                    "duplicate or orphan them"
                )
        return self

    def record_content(self) -> dict[str, Any]:
        """The complete canonical persisted content (for the Content Hash).

        This is a *record* projection for content-addressed storage, not an
        Identity Document: a Rollout Result has a Content Hash, not an
        Identity Hash.
        """
        return self.model_dump(mode="json")


def rollout_result_reference(result: RolloutResult) -> ObjectReference:
    """The typed Object Reference a Rollout Result resolves under.

    Addressed by Content Hash under the Rollout Result record schema; no
    Identity Hash is ever computed for a Rollout Result.
    """
    return ObjectReference.for_record(
        ROLLOUT_RESULT_SCHEMA, result.record_content()
    )
