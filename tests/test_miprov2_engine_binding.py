"""EngineEvalBindingResolver attests the engine's task-model route."""

from __future__ import annotations

import pytest
from dr_store.sync import open_sqlite

from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.miprov2.engine_binding import EngineEvalBindingResolver
from whetstone.optim.miprov2.eval_config import (
    EvalBindingRequest,
    Miprov2EvaluationExecutionPolicy,
)
from whetstone.testing.fakes.engine import FakeEvalEngine


class _MismatchedTaskModelEngine(FakeEvalEngine):
    def task_model_identity_hash(self) -> str:
        return "b" * 64


def test_resolve_rejects_a_mismatched_task_model(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "bind.sqlite")) as store:
        real = ReferenceEvalRuntimeConfig().build_engine(store)
        engine = _MismatchedTaskModelEngine(
            eval_config_ref=real.eval_config_ref,
            provider_execution_policy_ref=real.provider_execution_policy_ref,
            provider_execution_policy_record=(
                real.provider_execution_policy_record
            ),
            plan_snapshot=real.plan_snapshot,
            sampling=real.sampling,
            sampling_split=real.sampling_split,
        )
        request = EvalBindingRequest(
            control_identity_hash="a" * 64,
            source_eval_config=real.eval_config_ref,
            purpose="baseline",
            effect_identity_hash="c" * 64,
            execution_policy=Miprov2EvaluationExecutionPolicy(
                max_errors=4,
                provide_traceback=None,
                task_model_identity_hash="d" * 64,
                provider_execution_policy_hash=(
                    real.execution_policy_identity_hash()
                ),
            ),
            task_batch_hashes=real.sampling.task_hashes[:1],
        )

        with pytest.raises(ValueError, match="task-model route"):
            EngineEvalBindingResolver(engine=engine).resolve(request)
