from __future__ import annotations

import random
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

from gepa import EvaluationBatch

from whetstone.core.identity import compute_identity_hash
from whetstone.optim.gepa.contracts import (
    GepaCandidateComponent,
    GepaDataInstance,
    GepaEffectBroker,
    GepaEffectContext,
    GepaEffectSlot,
    GepaEvalAuthorityBinding,
    GepaEvaluationEffectRequest,
    GepaProposalAuthorityBinding,
    GepaProposalEffectRequest,
    GepaScoreMismatchEvidence,
    GepaSkippedMutation,
    GepaTrajectoryProjection,
)
from whetstone.optim.gepa.prompts import (
    GepaPromptServices,
    GepaReflectionRequest,
    GepaRejectedAttempt,
)

GEPA_UPSTREAM_ADAPTER_SCHEMA = "whetstone.gepa.upstream_adapter"
GEPA_UPSTREAM_ADAPTER_SCHEMA_VERSION = 2
#: One reflection attempt plus one bounded retry with the rejection fed back.
GEPA_REFLECTION_MAX_ATTEMPTS = 2
GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH = compute_identity_hash(
    schema=GEPA_UPSTREAM_ADAPTER_SCHEMA,
    schema_version=GEPA_UPSTREAM_ADAPTER_SCHEMA_VERSION,
    payload={
        "upstream_package": "gepa==0.1.1",
        "candidate_mapping": "ordered_named_components/v1",
        "data_mapping": "integer_position_plus_typed_task_ref/v1",
        "effect_replay": "semantic_plus_ordinal/v1",
        "reflection_projection": "canonical_trajectory/v1",
        "reflection_retry_policy": "bounded_single_retry_then_skip/v1",
        "merge_proposals": False,
    },
)


def _candidate_components(
    candidate: Mapping[str, str],
) -> tuple[GepaCandidateComponent, ...]:
    return tuple(
        GepaCandidateComponent(name=name, text=text)
        for name, text in candidate.items()
    )


class WhetstoneGepaAdapter:
    def __init__(
        self,
        *,
        context: GepaEffectContext,
        broker: GepaEffectBroker,
        evaluation_authority: GepaEvalAuthorityBinding,
        proposal_authority: GepaProposalAuthorityBinding,
        prompt_services: GepaPromptServices,
    ) -> None:
        if (
            context.adapter_identity_hash
            != GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH
        ):
            raise ValueError(
                "GEPA effect context does not bind the installed upstream "
                "adapter identity"
            )
        if (
            proposal_authority.prompt_binding_identity_hash
            != prompt_services.binding.identity_hash()
        ):
            raise ValueError(
                "GEPA proposal authority does not bind the exact prompt "
                "services"
            )
        self._context = context
        self._broker = broker
        self._evaluation_authority = evaluation_authority
        self._proposal_authority = proposal_authority
        self._prompt_services = prompt_services
        self._next_invocation_ordinal = 0
        self._rng = random.Random(evaluation_authority.selection_seed)
        self._score_mismatch_warned = False
        self._score_mismatch_evidence: list[GepaScoreMismatchEvidence] = []
        self._skipped_mutations: list[GepaSkippedMutation] = []

    @property
    def runtime_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_UPSTREAM_ADAPTER_SCHEMA,
            schema_version=GEPA_UPSTREAM_ADAPTER_SCHEMA_VERSION,
            payload={
                "context": self._context.model_dump(mode="json"),
                "evaluation_authority": (
                    self._evaluation_authority.model_dump(mode="json")
                ),
                "proposal_authority": (
                    self._proposal_authority.model_dump(mode="json")
                ),
                "prompt_binding_identity_hash": (
                    self._prompt_services.binding.identity_hash()
                ),
                "failure_score": self.failure_score,
                "add_format_failure_as_feedback": (
                    self.add_format_failure_as_feedback
                ),
                "warn_on_score_mismatch": self.warn_on_score_mismatch,
            },
        )

    @property
    def effect_context(self) -> GepaEffectContext:

        return self._context

    @property
    def evaluation_authority(self) -> GepaEvalAuthorityBinding:
        return self._evaluation_authority

    @property
    def evaluation_authority_binding(
        self,
    ) -> GepaEvalAuthorityBinding:
        return self._evaluation_authority

    @property
    def proposal_authority(self) -> GepaProposalAuthorityBinding:
        return self._proposal_authority

    @property
    def proposal_authority_binding(self) -> GepaProposalAuthorityBinding:
        return self._proposal_authority

    @property
    def prompt_format_identity_hash(self) -> str:
        return self._prompt_services.descriptor.identity_hash()

    @property
    def failure_score(self) -> float:
        return self._evaluation_authority.failure_score

    @property
    def add_format_failure_as_feedback(self) -> bool:
        return self._evaluation_authority.add_format_failure_as_feedback

    @property
    def warn_on_score_mismatch(self) -> bool:
        return self._evaluation_authority.warn_on_score_mismatch

    @property
    def score_mismatch_evidence(
        self,
    ) -> tuple[GepaScoreMismatchEvidence, ...]:
        return tuple(self._score_mismatch_evidence)

    @property
    def effect_count(self) -> int:

        return self._next_invocation_ordinal

    def reset_effect_ordinal(self) -> None:

        self._next_invocation_ordinal = 0
        self._rng = random.Random(self._evaluation_authority.selection_seed)
        self._score_mismatch_warned = False
        self._score_mismatch_evidence.clear()
        self._skipped_mutations.clear()

    def _slot(self) -> GepaEffectSlot:
        slot = GepaEffectSlot(
            context=self._context,
            invocation_ordinal=self._next_invocation_ordinal,
        )
        self._next_invocation_ordinal += 1
        return slot

    def _validate_candidate(self, candidate: Mapping[str, str]) -> None:
        expected = tuple(
            component.component_name
            for component in self._prompt_services.descriptor.components
        )
        actual = tuple(candidate)
        if actual != expected:
            raise ValueError(
                "GEPA candidate component order does not match its bound "
                f"prompt format: expected {expected!r}, got {actual!r}"
            )
        for component in self._prompt_services.descriptor.components:
            component.validate_replacement(candidate[component.component_name])

    def evaluate(
        self,
        batch: list[GepaDataInstance],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[GepaTrajectoryProjection, Any]:
        self._validate_candidate(candidate)
        request = GepaEvaluationEffectRequest(
            slot=self._slot(),
            candidate=_candidate_components(candidate),
            upstream_candidate_index=None,
            data=tuple(batch),
            capture_traces=capture_traces,
            authority=self._evaluation_authority,
        )
        result = self._broker.evaluate(request)
        if result.request_hash != request.identity_hash():
            raise ValueError(
                "GEPA evaluation result belongs to another effect request"
            )
        if len(result.rows) != len(batch):
            raise ValueError(
                "GEPA evaluation result row count does not match the batch"
            )
        for expected, row in zip(batch, result.rows, strict=True):
            if row.data != expected:
                raise ValueError(
                    "GEPA evaluation rows are not in requested data order"
                )
        trajectories: list[GepaTrajectoryProjection] | None
        if capture_traces:
            if any(row.trajectory is None for row in result.rows):
                raise ValueError(
                    "GEPA trace-capturing evaluation omitted a trajectory"
                )
            trajectories = [
                row.trajectory
                for row in result.rows
                if row.trajectory is not None
            ]
        else:
            if any(row.trajectory is not None for row in result.rows):
                raise ValueError(
                    "GEPA non-tracing evaluation returned trajectories"
                )
            trajectories = None
        objective_presence = [
            row.objective_scores is not None for row in result.rows
        ]
        if any(objective_presence) and not all(objective_presence):
            raise ValueError(
                "GEPA objective scores must be present for every row or none"
            )
        objective_scores = (
            [
                dict(row.objective_scores)
                for row in result.rows
                if row.objective_scores is not None
            ]
            if all(objective_presence)
            else None
        )
        return EvaluationBatch(
            outputs=[row.output for row in result.rows],
            scores=[
                (
                    self.failure_score
                    if row.failure_ref is not None
                    else float(row.score)
                )
                for row in result.rows
            ],
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[GepaTrajectoryProjection, Any],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        self._validate_candidate(candidate)
        if len(components_to_update) != len(set(components_to_update)):
            raise ValueError(
                "GEPA reflection component selection contains duplicates"
            )
        if eval_batch.trajectories is None:
            raise ValueError(
                "GEPA reflective dataset requires captured trajectories"
            )
        reflective: dict[str, tuple[dict[str, Any], ...]] = {}
        for component_name in components_to_update:
            self._prompt_services.descriptor.component(component_name)
            items: list[dict[str, Any]] = []
            for trajectory in eval_batch.trajectories:
                trace_candidates = list(
                    trajectory.component_records.get(component_name, ())
                )
                if not self.add_format_failure_as_feedback:
                    trace_candidates = [
                        trace
                        for trace in trace_candidates
                        if not trace.format_failure
                    ]
                if not trace_candidates:
                    if trajectory.component_records.get(component_name):
                        continue
                    items.append(trajectory.reflective_record(component_name))
                    continue
                selected = next(
                    (
                        trace
                        for trace in trace_candidates
                        if trace.format_failure
                    ),
                    None,
                )
                if selected is None:
                    if trajectory.prediction_failed:
                        continue
                    selected = self._rng.choice(trace_candidates)
                if (
                    selected.feedback_score is not None
                    and trajectory.module_score is not None
                    and selected.feedback_score != trajectory.module_score
                    and self.warn_on_score_mismatch
                    and not self._score_mismatch_warned
                ):
                    evidence = GepaScoreMismatchEvidence(
                        data_id=trajectory.data_id,
                        component_name=component_name,
                        feedback_score=selected.feedback_score,
                        module_score=trajectory.module_score,
                        source_refs=selected.source_refs,
                    )
                    self._score_mismatch_evidence.append(evidence)
                    self._score_mismatch_warned = True
                    warnings.warn(
                        "GEPA component feedback score differs from the "
                        "module-level score; GEPA uses the module score.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                items.append(selected.reflective_record())
            if items:
                reflective[component_name] = tuple(items)
        if not reflective:
            raise ValueError(
                "No valid predictions found for any GEPA component."
            )
        return reflective

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[
            str,
            Sequence[Mapping[str, Any]],
        ],
        components_to_update: list[str],
    ) -> dict[str, str]:

        self._validate_candidate(candidate)
        concrete_dataset = {
            name: tuple(dict(item) for item in items)
            for name, items in reflective_dataset.items()
        }
        selected = tuple(components_to_update)
        replacements: dict[str, str] = {}
        for component_name in selected:
            if not concrete_dataset.get(component_name):
                continue
            replacement = self._propose_one_component(
                candidate=candidate,
                concrete_dataset=concrete_dataset,
                selected=selected,
                component_name=component_name,
            )
            if replacement is not None:
                replacements[component_name] = replacement
        return replacements

    def _propose_one_component(
        self,
        *,
        candidate: Mapping[str, str],
        concrete_dataset: Mapping[str, tuple[dict[str, Any], ...]],
        selected: tuple[str, ...],
        component_name: str,
    ) -> str | None:
        """Reflect once, retry once on rejection, then skip the mutation.

        A reflection response that will not parse or fails the component's
        format contract is a normal model failure, not an infrastructure
        error. Retrying once with the rejection fed back recovers most of
        them; a second rejection records a skipped mutation in the step
        evidence and leaves this component unchanged, so the search
        continues instead of ending the whole optimization.
        """
        prior_attempt: GepaRejectedAttempt | None = None
        for attempt_index in range(GEPA_REFLECTION_MAX_ATTEMPTS):
            reflection_request = GepaReflectionRequest(
                candidate=dict(candidate),
                reflective_dataset=dict(concrete_dataset),
                components_to_update=selected,
                component_name=component_name,
                prior_attempt=prior_attempt,
            )
            rendered = self._prompt_services.reflection_builder.render(
                self._prompt_services.descriptor,
                reflection_request,
            )
            request = GepaProposalEffectRequest(
                slot=self._slot(),
                candidate=_candidate_components(candidate),
                upstream_candidate_index=None,
                components_to_update=selected,
                component_name=component_name,
                rendered_prompt=rendered,
                authority=self._proposal_authority,
            )
            result = self._broker.propose(request)
            if result.request_hash != request.identity_hash():
                raise ValueError(
                    "GEPA proposal result belongs to another effect request"
                )
            if result.failed:
                # A rejected response is retryable; a transport or provider
                # failure is not, and must still surface.
                if not result.rejected_by_parser:
                    raise RuntimeError(
                        result.failure_detail or "GEPA proposal effect failed"
                    )
                prior_attempt = GepaRejectedAttempt(
                    raw_response=result.raw_response or "",
                    rejection_detail=(
                        result.failure_detail
                        or "GEPA reflection response was rejected"
                    ),
                )
                self._skipped_mutations.append(
                    GepaSkippedMutation(
                        component_name=component_name,
                        attempt_ordinal=attempt_index,
                        rejection_detail=prior_attempt.rejection_detail,
                        raw_response=prior_attempt.raw_response,
                        provider_attempt_refs=result.provider_attempt_refs,
                        exhausted=(
                            attempt_index
                            == GEPA_REFLECTION_MAX_ATTEMPTS - 1
                        ),
                    )
                )
                continue
            expected = (component_name,)
            actual = tuple(item.name for item in result.parsed_components)
            if actual != expected:
                raise ValueError(
                    "GEPA proposal result must contain exactly the requested "
                    f"component {expected!r}, got {actual!r}"
                )
            replacement = result.parsed_components[0].text
            validated = self._prompt_services.validate_replacement(
                component_name,
                replacement,
            )
            if validated != replacement:
                raise ValueError(
                    "GEPA proposal authority and adapter parser disagree"
                )
            return replacement
        return None

    @property
    def skipped_mutations(self) -> tuple[GepaSkippedMutation, ...]:
        """Rejected reflection attempts recorded during this step.

        One entry per rejected attempt, including a rejection the bounded
        retry then recovered from; ``exhausted=True`` marks the attempts
        that actually dropped the component's mutation.
        """
        return tuple(self._skipped_mutations)


__all__ = [
    "GEPA_REFLECTION_MAX_ATTEMPTS",
    "GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH",
    "GEPA_UPSTREAM_ADAPTER_SCHEMA",
    "GEPA_UPSTREAM_ADAPTER_SCHEMA_VERSION",
    "WhetstoneGepaAdapter",
]
