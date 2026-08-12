#!/usr/bin/env -S uv run python
"""Minimal smoke checks for the generic evaluation library boundary."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass
from typing import Any

from whetstone.core.identity import IdentityRef, TypedRef
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.drivers.row_jobs import row_job_from_entrypoint
from whetstone.evaluation.protocol import (
    EngineEvaluation,
    EvaluationEngine,
    EvaluationPlanSnapshot,
    EvaluationRequest,
    EvaluationSamplingView,
    EvaluationTaskView,
)
from whetstone.experiment.binding import EvalConfigRef, EvaluationBinding
from whetstone.experiment.candidate import Candidate
from whetstone.optimization.tools.contracts import ToolCall, ToolConfig
from whetstone.optimization.tools.evaluator import EngineToolEvaluator

OPTIONAL_EXTRA_MODULES = frozenset(
    {
        "whetstone.coordination.proposal_provider",
        "whetstone.coordination.run_workflow",
        "whetstone.optimization.gepa.effect_runtime",
        "whetstone.optimization.gepa.runner",
    }
)


@dataclass(frozen=True, slots=True)
class _FakeTask:
    task_id: str
    task_hash: str
    prompt_inputs: dict[str, str]


@dataclass(frozen=True, slots=True)
class _FakeSampling:
    task_hashes: tuple[str, ...]
    num_samples: int
    split_role: str
    tasks: tuple[_FakeTask, ...]


@dataclass
class FakeEvaluationEngine:
    eval_config_ref: EvalConfigRef
    provider_execution_policy_ref: IdentityRef
    provider_execution_policy_record: dict[str, Any]
    plan_snapshot: EvaluationPlanSnapshot
    sampling: _FakeSampling
    model_route: str

    def task_model_identity_hash(self) -> str:
        return "fake-task-model"

    def execution_policy_identity_hash(self) -> str:
        return "fake-execution-policy"

    def reward_policy_identity_hash(self) -> str:
        return "fake-reward-policy"

    def expected_model_route(self) -> str:
        return self.model_route

    def validate_request(self, request: EvaluationRequest) -> None:
        return None

    def evaluate(self, request: EvaluationRequest) -> EngineEvaluation:
        raise NotImplementedError("smoke test uses validate() only")

    def for_task_ids(self, task_ids: tuple[str, ...]) -> EvaluationEngine:
        if not task_ids:
            raise ValueError("task_ids must be non-empty")
        return self


def _import_sweep() -> list[tuple[str, Exception]]:
    sys.path.insert(0, "src")
    errors: list[tuple[str, Exception]] = []
    for module in pkgutil.walk_packages(["src/whetstone"], "whetstone."):
        if module.name in OPTIONAL_EXTRA_MODULES:
            continue
        try:
            importlib.import_module(module.name)
        except Exception as exc:
            errors.append((module.name, exc))
    return errors


def _protocol_smoke() -> None:
    from pydantic import BaseModel

    factory = row_job_from_entrypoint("example.module:worker")

    class RowRequest(BaseModel):
        task_id: str

    process_job = factory(RowRequest(task_id="t1"))
    assert process_job.entrypoint == "example.module:worker"
    assert process_job.payload == {"task_id": "t1"}


def _p0_graph_smoke() -> None:
    from whetstone.evaluation.drivers.graph_row import run_rollout_row
    from whetstone.evaluation.eval_procedure import EvalProcedureRunner
    from whetstone.experiment.graph.run_node_registry import build_run_node
    from whetstone.experiment.graph.rollout_template import (
        build_single_llm_eval_graph,
    )
    from whetstone.provider.llm_call import LlmCallContext, execute_llm_call

    _ = (
        run_rollout_row,
        EvalProcedureRunner,
        build_run_node,
        build_single_llm_eval_graph,
        LlmCallContext,
        execute_llm_call,
    )


def _p0_engine_smoke() -> None:
    from whetstone.evaluation.driver import EvaluationDriver
    from whetstone.evaluation.runtime_engine import RuntimeEvaluationEngine

    _ = (EvaluationDriver, RuntimeEvaluationEngine)


def _p1_preview_smoke() -> None:
    from whetstone.evaluation.analysis.calibration import run_anchor_calibration
    from whetstone.evaluation.preview import evaluate_and_resolve

    _ = (evaluate_and_resolve, run_anchor_calibration)


def _reference_runtime_smoke() -> None:
    import tempfile

    from whetstone.core.blocking_store import open_blocking_sqlite_store

    from whetstone.evaluation.preview.binding import preview_evaluation_binding
    from whetstone.evaluation.protocol import EvaluationRequest
    from whetstone.evaluation.reference_runtime import (
        ReferenceEvaluationRuntimeConfig,
    )
    from whetstone.testing.fakes import (
        DummyProposerTransport,
        FakeEvalProcedureRunner,
        FakeEvaluationDriver,
    )
    from whetstone.testing.toy import ToyTask, build_toy_experiment

    _ = (
        DummyProposerTransport,
        FakeEvalProcedureRunner,
        FakeEvaluationDriver,
        ToyTask,
        build_toy_experiment,
    )
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as handle, open_blocking_sqlite_store(
        handle.name
    ) as store:
        runtime = ReferenceEvaluationRuntimeConfig.model_validate({})
        engine = runtime.build_engine(store)
        request = EvaluationRequest(
            candidate=engine.experiment.initial_candidate,
            evaluation_binding=preview_evaluation_binding(
                engine,
                campaign="smoke",
                provenance_note="reference-runtime",
            ),
            purpose="smoke",
        )
        engine.validate_request(request)
        evaluated = engine.evaluate(request)
        assert evaluated.evidence.aggregate_value is not None


def main() -> None:
    errors = _import_sweep()
    if errors:
        for name, exc in errors:
            print(f"IMPORT FAIL {name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
    print(f"import sweep ok ({len(list(pkgutil.walk_packages(['src/whetstone'], 'whetstone.')))} modules)")

    _protocol_smoke()
    print("row job factory ok")

    from whetstone.experiment.env import Experiment, GenerationGraphLike
    from whetstone.experiment.sampling import EvalConfigs
    from whetstone.optimization.validation.matrix import MatrixTreatmentState

    _ = (Experiment, GenerationGraphLike, EvalConfigs, MatrixTreatmentState)
    print("salvaged modules import ok")

    _p0_graph_smoke()
    print("p0 graph modules import ok")

    _p0_engine_smoke()
    print("p0 engine modules import ok")

    _p1_preview_smoke()
    print("p1 preview modules import ok")

    _reference_runtime_smoke()
    print("reference runtime smoke ok")


if __name__ == "__main__":
    main()
