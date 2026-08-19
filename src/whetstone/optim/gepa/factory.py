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
)
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

    def create(
        self,
        *,
        control: GepaControl,
        effect_broker: str = "dbos",
    ) -> WhetstoneGepaAdapter:
        from whetstone.optim.gepa.effect_runtime import (
            DbosGepaEffectBroker,
            register_gepa_evaluation_authority,
            register_gepa_proposal_authority,
        )

        self._require_control(control)
        register_gepa_evaluation_authority(
            self._evaluation_authority.runtime_hash,
            self._evaluation_authority,
        )
        register_gepa_proposal_authority(
            self._proposal_authority.runtime_hash,
            self._proposal_authority,
        )
        if effect_broker == "harness":
            from whetstone.optim.gepa.harness_broker import HarnessGepaEffectBroker

            broker = HarnessGepaEffectBroker(
                self._store,
                evaluation_authority=self._evaluation_authority,
                proposal_authority=self._proposal_authority,
            )
        elif effect_broker == "dbos":
            broker = DbosGepaEffectBroker(self._store)
        else:
            raise ValueError(
                f"unsupported GEPA effect broker {effect_broker!r}; "
                "expected 'dbos' or 'harness'"
            )
        return WhetstoneGepaAdapter(
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
