from __future__ import annotations

from dr_store import BindingConflictError, ObjectStore
from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from whetstone.core.identity import (
    TypedRef,
    require_full_hash,
)
from whetstone.optimization.gepa.contracts import (
    GEPA_EFFECT_TRANSCRIPT_SCHEMA,
    GepaEffectContext,
    GepaEffectTranscript,
)
from whetstone.optimization.gepa.engine import GepaDetailedResult

GEPA_DETAILED_RESULT_RECORD_SCHEMA = "whetstone.gepa.detailed_result"
GEPA_RUN_RESULT_ARTIFACT_SCHEMA = "whetstone.gepa.run_result_artifact"


class GepaRunResultArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    context: GepaEffectContext
    detailed_result_ref: TypedRef
    effect_transcript_ref: TypedRef
    control_identity_hash: StrictStr
    source_manifest_identity_hash: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> GepaRunResultArtifact:
        for field_name in (
            "control_identity_hash",
            "source_manifest_identity_hash",
        ):
            require_full_hash(getattr(self, field_name), field=field_name)
        if (
            self.context.control_identity_hash != self.control_identity_hash
            or self.context.source_manifest_identity_hash
            != self.source_manifest_identity_hash
        ):
            raise ValueError(
                "GEPA result artifact conflicts with its effect context"
            )
        if (
            self.detailed_result_ref.schema_name
            != GEPA_DETAILED_RESULT_RECORD_SCHEMA
            or self.effect_transcript_ref.schema_name
            != GEPA_EFFECT_TRANSCRIPT_SCHEMA
        ):
            raise ValueError("GEPA result artifact has an invalid record type")
        return self


class GepaResultArtifactStore:
    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    @staticmethod
    def _typed_ref(reference) -> TypedRef:
        return TypedRef(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )

    @staticmethod
    def _binding_key(context: GepaEffectContext) -> str:
        return f"whetstone.gepa.run_result_artifact:{context.identity_hash()}"

    def persist(
        self,
        *,
        context: GepaEffectContext,
        detailed_result: GepaDetailedResult,
        transcript_ref: TypedRef,
    ) -> TypedRef:
        if (
            detailed_result.control_identity_hash
            != context.control_identity_hash
            or detailed_result.source_manifest_hash
            != context.source_manifest_identity_hash
        ):
            raise ValueError(
                "GEPA detailed result conflicts with terminal effect context"
            )
        transcript = GepaEffectTranscript.model_validate(
            self._store.get(transcript_ref.reference)
        )
        if transcript.context != context:
            raise ValueError(
                "GEPA effect transcript conflicts with terminal context"
            )
        detail_raw_ref, _ = self._store.put(
            GEPA_DETAILED_RESULT_RECORD_SCHEMA,
            detailed_result.model_dump(mode="json"),
        )
        artifact = GepaRunResultArtifact(
            context=context,
            detailed_result_ref=self._typed_ref(detail_raw_ref),
            effect_transcript_ref=transcript_ref,
            control_identity_hash=context.control_identity_hash,
            source_manifest_identity_hash=(
                context.source_manifest_identity_hash
            ),
        )
        artifact_raw_ref, _ = self._store.put(
            GEPA_RUN_RESULT_ARTIFACT_SCHEMA,
            artifact.model_dump(mode="json"),
        )
        key = self._binding_key(context)
        try:
            self._store.bind(key, artifact_raw_ref)
        except BindingConflictError:
            pass
        bound = self._store.resolve(key)
        if bound != artifact_raw_ref:
            raise ValueError(
                "GEPA run already has a different terminal result artifact"
            )
        return self._typed_ref(artifact_raw_ref)


__all__ = [
    "GEPA_DETAILED_RESULT_RECORD_SCHEMA",
    "GEPA_RUN_RESULT_ARTIFACT_SCHEMA",
    "GepaResultArtifactStore",
    "GepaRunResultArtifact",
]
