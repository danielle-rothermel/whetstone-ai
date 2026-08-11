from __future__ import annotations

from dr_store import MemoryBackend, ObjectStore

from tests.evaluation.support import _engine
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.preview.binding import preview_evaluation_binding
from whetstone.experiment.binding import ExecutionEnvironmentFingerprint


def test_preview_evaluation_binding_uses_engine_refs(tmp_path) -> None:
    store = ObjectStore(MemoryBackend())
    engine = _engine(tmp_path, store=store)
    fingerprint = ExecutionEnvironmentFingerprint(
        dependency_versions=(("dr-code", "0.1.5"), ("numpy", "2.0.0")),
        runtime_identity="c" * 64,
    )

    binding = preview_evaluation_binding(
        engine,
        campaign="preview-test",
        provenance_note="unit-test",
        environment_fingerprint=fingerprint,
        role=EvaluationRole.INTERNAL,
    )

    assert binding.eval_config == engine.eval_config_ref
    assert binding.provider_execution_policy_ref == (
        engine.provider_execution_policy_ref
    )
    assert binding.environment_fingerprint == fingerprint
    assert binding.provenance_note == "unit-test"
