from __future__ import annotations

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.optimization.gepa.support import data_instance
from whetstone.optimization.gepa.contracts import (
    GepaEffectContext,
    GepaEffectRecorder,
    GepaEffectTranscript,
)
from whetstone.optimization.gepa.engine import GepaDetailedResult
from whetstone.optimization.gepa.result_artifact import (
    GepaResultArtifactStore,
    GepaRunResultArtifact,
)
from whetstone.optimization.gepa.source import GEPA_SOURCE_MANIFEST_HASH
from whetstone.optimization.gepa.upstream_adapter import (
    GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
)

_A = "a" * 64


def test_result_artifact_pairs_detail_and_effect_transcript_idempotently(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "artifact.sqlite"))
    context = GepaEffectContext(
        run_id="gepa:artifact",
        control_identity_hash=_A,
        source_manifest_identity_hash=GEPA_SOURCE_MANIFEST_HASH,
        adapter_identity_hash=GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
    )
    recorder = GepaEffectRecorder(store)
    transcript_ref = recorder.persist_transcript(
        GepaEffectTranscript(context=context, entries=())
    )
    detail = GepaDetailedResult(
        candidates=({"alpha": "alpha-0"},),
        parents=((),),
        val_aggregate_scores=(0.5,),
        val_subscores=({data_instance(0).data_id: 0.5},),
        per_val_instance_best_candidates={data_instance(0).data_id: (0,)},
        discovery_eval_counts=(1,),
        seed=0,
        best_idx=0,
        control_identity_hash=_A,
    )
    artifact_store = GepaResultArtifactStore(store)

    first = artifact_store.persist(
        context=context,
        detailed_result=detail,
        transcript_ref=transcript_ref,
    )
    replay = artifact_store.persist(
        context=context,
        detailed_result=detail,
        transcript_ref=transcript_ref,
    )

    assert replay == first
    artifact = GepaRunResultArtifact.model_validate(store.get(first.reference))
    assert artifact.effect_transcript_ref == transcript_ref
    conflicting = detail.model_copy(update={"val_aggregate_scores": (0.75,)})
    with pytest.raises(ValueError, match="different terminal"):
        artifact_store.persist(
            context=context,
            detailed_result=conflicting,
            transcript_ref=transcript_ref,
        )
