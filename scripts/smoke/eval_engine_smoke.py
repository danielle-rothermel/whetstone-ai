#!/usr/bin/env -S uv run python
"""Minimal smoke checks for the generic evaluation library boundary."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass
from typing import Any

from whetstone.core.identity import IdentityRef, TypedRef
from whetstone.core.roles import EvalRole
from whetstone.eval.drivers.row_jobs import row_job_from_entrypoint
from whetstone.eval.metadata import metadata_with_purpose
from whetstone.eval.protocol import (
    EvalRequest,
    EvalResult,
    EvalEngine,
    EvalPlanSnapshot,
    EvalSplitView,
    EvalTaskView,
    eval_is_success,
)
from whetstone.experiment.binding import EvalConfigRef
from whetstone.experiment.candidate import Candidate
from whetstone.optim.tools.contracts import ToolCall, ToolConfig
from whetstone.optim.tools.evaluator import EngineToolEvaluator

OPTIONAL_EXTRA_MODULES = frozenset(
    {
        "whetstone.coordination.proposal_provider",
        "whetstone.coordination.run_workflow",
        "whetstone.optim.gepa.effect_runtime",
        "whetstone.optim.gepa.runner",
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
    num_seeds: int
    split_role: str
    tasks: tuple[_FakeTask, ...]


@dataclass
class FakeEvalEngine:
    eval_config_ref: EvalConfigRef
    provider_execution_policy_ref: IdentityRef
    provider_execution_policy_record: dict[str, Any]
    plan_snapshot: EvalPlanSnapshot
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

    def evaluate(self, request: EvalRequest) -> EvalResult:
        raise NotImplementedError("smoke test uses validate() only")

    def for_task_ids(self, task_ids: tuple[str, ...]) -> EvalEngine:
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
    from whetstone.eval.drivers.graph_row import execute_rollout_graph
    from whetstone.eval.eval_procedure import EvalProcedureRunner
    from whetstone.experiment.graph.run_node_registry import build_run_node
    from whetstone.experiment.graph.rollout_template import (
        build_single_llm_eval_graph,
    )
    from whetstone.provider.llm_call import LlmCallContext, execute_llm_call

    _ = (
        execute_rollout_graph,
        EvalProcedureRunner,
        build_run_node,
        build_single_llm_eval_graph,
        LlmCallContext,
        execute_llm_call,
    )


def _p0_engine_smoke() -> None:
    from whetstone.eval.driver import EvalDriver
    from whetstone.eval.runtime_engine import RuntimeEvalEngine

    _ = (EvalDriver, RuntimeEvalEngine)


def _p1_preview_smoke() -> None:
    from whetstone.eval.analysis.calibration import run_anchor_calibration
    from whetstone.eval.preview import evaluate_and_resolve

    _ = (evaluate_and_resolve, run_anchor_calibration)


def _graph_rollout_driver_smoke() -> None:
    import tempfile

    from whetstone.core.blocking_store import open_blocking_sqlite_store
    from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
    from whetstone.eval.preview.persisted import load_component_traces
    from whetstone.eval.protocol import EvalRequest
    from whetstone.eval.runtime_engine import RuntimeEvalEngine
    from whetstone.provider.policy import ProviderExecutionPolicy, default_transport_policy
    from whetstone.testing.fakes.eval_procedure import FakeEvalProcedureRunner
    from whetstone.testing.fakes.transport import FakeLlmTransport
    from whetstone.testing.toy.experiment import (
        TOY_MUTATION_FIELD,
        build_toy_experiment,
        toy_template_render_contract,
    )

    experiment = build_toy_experiment()
    sampling = experiment.eval_configs.internal
    transport_policy = default_transport_policy(api_key_env="WHETSTONE_TOY_API_KEY")
    execution_policy = ProviderExecutionPolicy(transport_policy=transport_policy)
    driver = GraphRolloutEvalDriver(
        eval_runner=FakeEvalProcedureRunner(),
        mutation_field=TOY_MUTATION_FIELD,
        render_contract=toy_template_render_contract(),
        transport_factory=lambda policy: FakeLlmTransport(
            transport_policy=policy.transport_policy
        ),
    )
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as handle, open_blocking_sqlite_store(
        handle.name
    ) as store:
        engine = RuntimeEvalEngine(
            store=store,
            experiment=experiment,
            sampling=sampling,
            execution_policy=execution_policy,
            driver=driver,
            concurrency=2,
        )
        request = EvalRequest(
            request_id="smoke:graph-rollout-driver",
            candidate=experiment.initial_candidate,
            metadata=metadata_with_purpose("smoke"),
        )
        evaluated = engine.evaluate(request)
        assert eval_is_success(evaluated)
        evidence = evaluated.evidence
        assert evidence.aggregate_value is not None
        expected_rows = len(sampling.tasks) * sampling.seed_plan.num_seeds
        assert evidence.row_accounting.present == expected_rows
        component_traces = load_component_traces(store, evidence)
        assert any(
            row.trace.trace_steps
            for row in component_traces.rows
        )


def _reference_runtime_smoke() -> None:
    import tempfile

    from whetstone.core.blocking_store import open_blocking_sqlite_store

    from whetstone.eval.protocol import EvalRequest
    from whetstone.eval.reference_runtime import (
        ReferenceEvalRuntimeConfig,
    )
    from whetstone.testing.fakes import (
        DummyProposerTransport,
        FakeEvalProcedureRunner,
        FakeEvalDriver,
    )
    from whetstone.testing.toy import ToyTask, build_toy_experiment

    _ = (
        DummyProposerTransport,
        FakeEvalProcedureRunner,
        FakeEvalDriver,
        ToyTask,
        build_toy_experiment,
    )
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as handle, open_blocking_sqlite_store(
        handle.name
    ) as store:
        runtime = ReferenceEvalRuntimeConfig.model_validate({})
        engine = runtime.build_engine(store)
        request = EvalRequest(
            request_id="smoke:reference-runtime",
            candidate=engine.experiment.initial_candidate,
            metadata=metadata_with_purpose("smoke"),
        )
        evaluated = engine.evaluate(request)
        assert eval_is_success(evaluated)
        assert evaluated.evidence.aggregate_value is not None


def _reference_runtime_with_cache_smoke() -> None:
    import tempfile
    from pathlib import Path

    from whetstone.core.blocking_store import open_blocking_sqlite_store
    from whetstone.eval.protocol import EvalRequest
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.execution._file_lock import ensure_private_directory

    with tempfile.TemporaryDirectory() as tmp, tempfile.NamedTemporaryFile(
        suffix=".sqlite"
    ) as handle, open_blocking_sqlite_store(handle.name) as store:
        cache_root = Path(tmp).resolve() / "cache"
        partial_path = Path(tmp).resolve() / "partials.jsonl"
        ensure_private_directory(cache_root)
        ensure_private_directory(partial_path.parent)
        runtime = ReferenceEvalRuntimeConfig.model_validate(
            {
                "partial_log_path": str(partial_path),
                "prompt_cache_path": str(cache_root),
            }
        )
        engine = runtime.build_engine(store)
        request = EvalRequest(
            request_id="smoke:reference-runtime-cache",
            candidate=engine.experiment.initial_candidate,
            metadata=metadata_with_purpose("smoke"),
        )
        first = engine.evaluate(request)
        assert eval_is_success(first)
        second = engine.evaluate(
            EvalRequest(
                request_id="smoke:reference-runtime-cache-2",
                candidate=engine.experiment.initial_candidate,
                metadata=metadata_with_purpose("smoke"),
            )
        )
        assert eval_is_success(second)
        assert second.evidence.cache.cache_hit_count >= 1


def _subprocess_row_smoke() -> None:
    from whetstone.eval.drivers.graph_row_request import (
        GraphRowRequest,
        decode_graph_row_output,
    )
    from whetstone.eval.drivers.graph_worker import run_row
    from whetstone.provider.llm_call import derive_rng_seed
    from whetstone.provider.policy import ProviderExecutionPolicy, default_transport_policy
    from whetstone.testing.toy.experiment import (
        TOY_MUTATION_FIELD,
        build_toy_experiment,
        toy_template_render_contract,
    )

    experiment = build_toy_experiment(num_seeds=1)
    sampling = experiment.eval_configs.internal
    task = sampling.tasks[0]
    rollout_graph = experiment.rollout_graph
    candidate = experiment.initial_candidate
    rendered = toy_template_render_contract().render(
        candidate.payload[TOY_MUTATION_FIELD],
        task.prompt_inputs,
    )
    execution_policy = ProviderExecutionPolicy(
        transport_policy=default_transport_policy(
            api_key_env="WHETSTONE_TOY_API_KEY"
        )
    )
    request = GraphRowRequest(
        candidate_id=candidate.candidate_id,
        task_id=task.task_id,
        task_index=0,
        seed_index=0,
        split_role=sampling.split_role,
        rendered_prompt=rendered,
        graph_config=rollout_graph.graph_config.model_dump(mode="json"),
        rollout_graph_hash=rollout_graph.graph_hash,
        provider_call_config=rollout_graph.provider_call_config.model_dump(
            mode="json"
        ),
        rng_seed=derive_rng_seed(candidate.candidate_id, task.task_id, 0),
        mutation_field=TOY_MUTATION_FIELD,
        eval_procedure_config_hash=rollout_graph.procedure_config_hash,
        execution_policy_hash=execution_policy.identity_hash,
        prompt_inputs=dict(task.prompt_inputs),
        gold=task.gold,
    )
    payload = run_row(request.model_dump(mode="json"))
    output = decode_graph_row_output(payload, request=request)
    assert output.score is not None


def _subprocess_driver_smoke() -> None:
    import tempfile

    from whetstone.core.blocking_store import open_blocking_sqlite_store
    from whetstone.eval.protocol import EvalRequest
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    with tempfile.NamedTemporaryFile(suffix=".sqlite") as handle, open_blocking_sqlite_store(
        handle.name
    ) as store:
        runtime = ReferenceEvalRuntimeConfig.model_validate(
            {"driver_mode": "subprocess"}
        )
        engine = runtime.build_engine(store)
        request = EvalRequest(
            request_id="smoke:subprocess-driver",
            candidate=engine.experiment.initial_candidate,
            metadata=metadata_with_purpose("smoke"),
        )
        evaluated = engine.evaluate(request)
        assert eval_is_success(evaluated)
        sampling = engine.experiment.eval_configs.internal
        expected_rows = len(sampling.tasks) * sampling.seed_plan.num_seeds
        assert evaluated.evidence.row_accounting.present == expected_rows


def main() -> None:
    errors = _import_sweep()
    if errors:
        for name, exc in errors:
            print(f"IMPORT FAIL {name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
    print(f"import sweep ok ({len(list(pkgutil.walk_packages(['src/whetstone'], 'whetstone.')))} modules)")

    _protocol_smoke()
    print("row job factory ok")

    from whetstone.experiment.env import Experiment, RolloutGraphLike
    from whetstone.experiment.sampling import EvalConfigs
    from whetstone.optim.validation.matrix import MatrixTreatmentState

    _ = (Experiment, RolloutGraphLike, EvalConfigs, MatrixTreatmentState)
    print("salvaged modules import ok")

    _p0_graph_smoke()
    print("p0 graph modules import ok")

    _p0_engine_smoke()
    print("p0 engine modules import ok")

    _p1_preview_smoke()
    print("p1 preview modules import ok")

    _reference_runtime_smoke()
    print("reference runtime smoke ok")

    _graph_rollout_driver_smoke()
    print("graph rollout driver smoke ok")

    _reference_runtime_with_cache_smoke()
    print("reference runtime cache smoke ok")

    _subprocess_row_smoke()
    print("subprocess row smoke ok")

    _subprocess_driver_smoke()
    print("subprocess driver smoke ok")


if __name__ == "__main__":
    main()
