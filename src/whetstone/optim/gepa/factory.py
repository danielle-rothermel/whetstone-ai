from __future__ import annotations

from dr_store import ObjectStore

from whetstone.core.identity import (
    TypedRef,
    compute_identity_hash,
)
from whetstone.coordination.eval_service import (
    EvalEngineService,
    EvalExecutionContext,
)
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optim.gepa.authorities import (
    CanonicalGepaCandidateAssembler,
    CanonicalGepaEvalAuthority,
    CanonicalGepaProposalAuthority,
    GepaCandidateFieldBinding,
    GepaDataRegistry,
)
from whetstone.optim.gepa.harness_adapter import (
    GepaHarnessAdapter,
    GepaHarnessAdapterFactory,
    gepa_candidate_field_name,
    seed_components_from_candidate,
)
from whetstone.optim.gepa.prompts import (
    GepaComponentFormat,
    GepaPromptFormatDescriptor,
    GepaPromptServices,
    NativeGepaReflectionPromptBuilder,
    NativeGepaReflectionResponseParser,
)
from whetstone.optim.gepa.contracts import (
    GepaDataInstance,
    GepaEffectContext,
    GepaEffectRecorder,
    GepaSkippedMutation,
)
from whetstone.optim.contracts import SearchEvidence
from whetstone.optim.gepa.control import GepaControl
from whetstone.optim.gepa.engine import GepaDetailedResult
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

    def bind_eval_context(self, context: EvalExecutionContext) -> None:
        binder = getattr(self._evaluation_authority, "bind_eval_context", None)
        if callable(binder):
            binder(context)

    def bind_evaluation_service(self, service: EvalEngineService) -> None:
        binder = getattr(
            self._evaluation_authority, "bind_evaluation_service", None
        )
        if callable(binder):
            binder(service)

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


def default_gepa_prompt_services(
    *,
    component_names: tuple[str, ...],
    mutation_field: str,
) -> GepaPromptServices:
    """Native reflection prompts bound to one field per component."""
    components = tuple(
        GepaComponentFormat(
            component_name=name,
            component_schema_identity_hash=compute_identity_hash(
                schema="whetstone.gepa.component_schema",
                schema_version=1,
                payload={"field": mutation_field, "component": name},
            ),
            allowed_placeholders=("prompt",),
            required_placeholders=("prompt",),
        )
        for name in component_names
    )
    return GepaPromptServices(
        descriptor=GepaPromptFormatDescriptor(
            format_name="gepa_prompt_template",
            components=components,
        ),
        reflection_builder=NativeGepaReflectionPromptBuilder(),
        reflection_parser=NativeGepaReflectionResponseParser(),
    )


def _split_registry_entries(
    *,
    control: GepaControl,
    registry: GepaDataRegistry,
) -> tuple[
    tuple[GepaDataInstance, ...],
    tuple[GepaDataInstance, ...] | None,
]:
    """Partition the registry into the control's train and validation splits.

    The registry holds the ordered union of both splits, because one eval
    engine serves them. Upstream GEPA reflects on the trainset and selects on
    the valset, so handing it the union as a trainset would both train on
    validation instances and let selection see training instances. Return the
    two ordered splits instead, and keep upstream's ``valset=None`` default
    for the controls that bind validation back to the trainset.
    """
    by_task_hash = {entry.task_hash: entry for entry in registry.entries}

    def ordered(task_hashes: tuple[str, ...], *, field: str):
        try:
            return tuple(by_task_hash[identity] for identity in task_hashes)
        except KeyError as exc:
            raise ValueError(
                f"GEPA data registry has no task with hash {exc.args[0]!r} "
                f"for {field}"
            ) from None

    trainset = ordered(control.trainset_task_hashes, field="trainset")
    if control.source_valset_task_hashes is None:
        if tuple(entry.task_hash for entry in trainset) != registry.task_hashes:
            raise ValueError(
                "GEPA registry conflicts with a control that binds validation "
                "to the trainset"
            )
        return trainset, None

    valset = ordered(control.valset_task_hashes, field="valset")
    train_hashes = {entry.task_hash for entry in trainset}
    val_hashes = {entry.task_hash for entry in valset}
    overlap = train_hashes & val_hashes
    if overlap:
        raise ValueError(
            "GEPA trainset and valset share tasks: "
            f"{sorted(overlap)}"
        )
    if train_hashes | val_hashes != set(registry.task_hashes):
        raise ValueError(
            "GEPA trainset and valset do not cover the data registry"
        )
    return trainset, valset


def build_gepa_harness_adapter(
    *,
    store: ObjectStore,
    engine,
    control: GepaControl,
    run_id: str,
    initial_candidate: Candidate,
    mutation_field: str,
    prompt_services: GepaPromptServices,
    transport,
    proposal_executor,
    evaluation_service=None,
) -> GepaHarnessAdapter:
    """Assemble the production GEPA harness adapter for one run."""
    seed = seed_components_from_candidate(
        initial_candidate,
        component_names=control.component_names,
        mutation_field=mutation_field,
    )
    registry = GepaDataRegistry.from_engine(store=store, engine=engine)
    assembler = CanonicalGepaCandidateAssembler(
        base_candidate=candidate_reference(initial_candidate),
        fields=tuple(
            GepaCandidateFieldBinding(
                component_name=name,
                candidate_field=gepa_candidate_field_name(
                    component_name=name,
                    component_names=control.component_names,
                    mutation_field=mutation_field,
                ),
            )
            for name in control.component_names
        ),
    )
    eval_authority = CanonicalGepaEvalAuthority(
        store=store,
        engine=engine,
        control=control,
        candidate_assembler=assembler,
        data_registry=registry,
        evaluation_service=evaluation_service,
    )
    proposal_authority = CanonicalGepaProposalAuthority(
        store=store,
        control=control,
        prompt_services=prompt_services,
        transport=transport,
        proposal_executor=proposal_executor,
    )
    factory = CanonicalGepaAdapterFactory(
        store=store,
        run_id=run_id,
        control=control,
        evaluation_authority=eval_authority,
        proposal_authority=proposal_authority,
        prompt_services=prompt_services,
    )
    trainset, valset = _split_registry_entries(
        control=control,
        registry=registry,
    )
    return GepaHarnessAdapter(
        control=control,
        seed_candidate=seed,
        trainset=trainset,
        valset=valset,
        adapter_factory=GepaHarnessAdapterFactory(factory=factory),
    )


__all__ = [
    "GEPA_ADAPTER_FACTORY_SCHEMA",
    "GEPA_ADAPTER_FACTORY_SCHEMA_VERSION",
    "CanonicalGepaAdapterFactory",
    "build_gepa_harness_adapter",
    "default_gepa_prompt_services",
]
