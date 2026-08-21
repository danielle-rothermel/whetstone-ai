from __future__ import annotations

from dr_store import ObjectStore

from whetstone.core.identity import (
    TypedRef,
    compute_identity_hash,
)
from whetstone.optim.gepa.authorities import (
    CanonicalGepaEvalAuthority,
    CanonicalGepaProposalAuthority,
)
from whetstone.optim.gepa.contracts import (
    GepaEffectContext,
    GepaEffectRecorder,
    GepaSkippedMutation,
)
from whetstone.optim.contracts import SearchEvidence
from whetstone.optim.gepa.control import GepaControl
from whetstone.optim.gepa.engine import GepaDetailedResult
from whetstone.optim.gepa.prompts import GepaPromptServices
from whetstone.optim.gepa.result_artifact import (
    GepaResultArtifactStore,
)
from whetstone.optim.gepa.upstream_adapter import (
    GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
    WhetstoneGepaAdapter,
)

GEPA_ADAPTER_FACTORY_SCHEMA = "whetstone.gepa.adapter_factory"
GEPA_ADAPTER_FACTORY_SCHEMA_VERSION = 1


class CanonicalGepaAdapterFactory:
    def __init__(
        self,
        *,
        store: ObjectStore,
        run_id: str,
        control: GepaControl,
        evaluation_authority: CanonicalGepaEvalAuthority,
        proposal_authority: CanonicalGepaProposalAuthority,
        prompt_services: GepaPromptServices,
    ) -> None:
        if not run_id:
            raise ValueError("GEPA factory run_id must be non-empty")
        control_hash = control.identity_hash()
        if (
            evaluation_authority.control_identity_hash != control_hash
            or proposal_authority.control_identity_hash != control_hash
        ):
            raise ValueError("GEPA factory authorities bind another control")
        if (
            prompt_services.binding.identity_hash()
            != control.prompt_binding_identity_hash
            or prompt_services.descriptor.identity_hash()
            != control.prompt_format_identity_hash
        ):
            raise ValueError(
                "GEPA factory prompt services bind another control"
            )
        descriptor_component_names = tuple(
            component.component_name
            for component in prompt_services.descriptor.components
        )
        if (
            descriptor_component_names != control.component_names
            or evaluation_authority.component_names != control.component_names
        ):
            raise ValueError(
                "GEPA factory component names conflict with control"
            )
        self._store = store
        self._run_id = run_id
        self._control = control
        self._evaluation_authority = evaluation_authority
        self._proposal_authority = proposal_authority
        self._prompt_services = prompt_services
        self._step_index: int | None = None
        self._adapters: list[WhetstoneGepaAdapter] = []

    @property
    def runtime_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_ADAPTER_FACTORY_SCHEMA,
            schema_version=GEPA_ADAPTER_FACTORY_SCHEMA_VERSION,
            payload={
                "run_id": self._run_id,
                "control_identity_hash": self._control.identity_hash(),
                "source_manifest_identity_hash": (
                    self._control.gepa_source_manifest_hash
                ),
                "adapter_identity_hash": (GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH),
                "evaluation_authority_identity_hash": (
                    self._evaluation_authority.runtime_hash
                ),
                "proposal_authority_identity_hash": (
                    self._proposal_authority.runtime_hash
                ),
                "prompt_binding_identity_hash": (
                    self._prompt_services.binding.identity_hash()
                ),
            },
        )

    def begin_step(self, *, step_index: int) -> None:
        """Bind this Step and drop evidence from earlier Steps.

        ``step_index`` is the harness step index. It deliberately stays out of
        the effect context, whose identity must remain step-agnostic so this
        Step can replay the prefix its predecessors already paid for. The
        authority stamps it onto the ``OptimEvalRequest`` of each evaluation
        it actually executes, so a fresh evaluation carries the Step that
        caused it while a replayed one never reaches the intent layer.
        """
        if step_index < 0:
            raise ValueError("GEPA factory step_index cannot be negative")
        self._step_index = step_index
        self._evaluation_authority.begin_step(step_index=step_index)
        self._adapters.clear()

    def _require_step(self) -> int:
        if self._step_index is None:
            raise ValueError(
                "GEPA factory requires begin_step before it serves a Step"
            )
        return self._step_index

    def search_evidence(
        self,
        *,
        run_id: str,
        step_index: int,
    ) -> tuple[SearchEvidence, ...]:
        """Evidence for every evaluation this Step's search drove."""
        authority = self._evaluation_authority
        resolutions = authority.resolved_intents
        replayed = authority.replayed_flags
        return tuple(
            (
                SearchEvidence.from_replayed_resolution
                if was_replayed
                else SearchEvidence.from_resolution
            )(
                resolution,
                optim_run_id=run_id,
                optim_step_index=step_index,
            )
            for resolution, was_replayed in zip(
                resolutions, replayed, strict=True
            )
        )

    def skipped_mutations(self) -> tuple[GepaSkippedMutation, ...]:
        """Reflection responses this Step's search rejected, in order.

        Read off the live adapters rather than the terminal transcript, so a
        rejection on a continuing Step is durable on that Step's own result.
        """
        return tuple(
            skipped
            for adapter in self._adapters
            for skipped in adapter.skipped_mutations
        )

    def create(
        self,
        *,
        control: GepaControl,
    ) -> WhetstoneGepaAdapter:
        self._require_control(control)
        from whetstone.optim.gepa.harness_broker import HarnessGepaEffectBroker

        broker = HarnessGepaEffectBroker(
            self._store,
            evaluation_authority=self._evaluation_authority,
            proposal_authority=self._proposal_authority,
        )
        adapter = WhetstoneGepaAdapter(
            context=GepaEffectContext(
                run_id=self._run_id,
                control_identity_hash=control.identity_hash(),
                source_manifest_identity_hash=(
                    control.gepa_source_manifest_hash
                ),
                adapter_identity_hash=GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
            ),
            broker=broker,
            evaluation_authority=self._evaluation_authority.binding,
            proposal_authority=self._proposal_authority.binding,
            prompt_services=self._prompt_services,
        )
        self._adapters.append(adapter)
        return adapter

    def persist_result(
        self,
        *,
        control: GepaControl,
        adapter,
        detailed_result: GepaDetailedResult,
    ) -> TypedRef:
        self._require_control(control)
        if not isinstance(adapter, WhetstoneGepaAdapter):
            raise TypeError(
                "canonical GEPA factory can persist only its native adapter"
            )
        expected_context = GepaEffectContext(
            run_id=self._run_id,
            control_identity_hash=control.identity_hash(),
            source_manifest_identity_hash=control.gepa_source_manifest_hash,
            adapter_identity_hash=GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
        )
        if (
            adapter.effect_context != expected_context
            or adapter.evaluation_authority
            != self._evaluation_authority.binding
            or adapter.proposal_authority != self._proposal_authority.binding
            or adapter.prompt_format_identity_hash
            != control.prompt_format_identity_hash
        ):
            raise ValueError(
                "GEPA terminal adapter conflicts with concrete factory"
            )
        recorder = GepaEffectRecorder(self._store)
        transcript = recorder.build_transcript(
            context=expected_context,
            effect_count=adapter.effect_count,
            score_mismatch_evidence=adapter.score_mismatch_evidence,
            skipped_mutations=adapter.skipped_mutations,
        )
        transcript_ref = recorder.persist_transcript(transcript)
        return GepaResultArtifactStore(self._store).persist(
            context=expected_context,
            detailed_result=detailed_result,
            transcript_ref=transcript_ref,
        )

    def _require_control(self, control: GepaControl) -> None:
        if control != self._control:
            raise ValueError(
                "GEPA concrete factory refuses control identity drift"
            )


__all__ = [
    "GEPA_ADAPTER_FACTORY_SCHEMA",
    "GEPA_ADAPTER_FACTORY_SCHEMA_VERSION",
    "CanonicalGepaAdapterFactory",
]
