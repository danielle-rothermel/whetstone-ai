from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
from whetstone.eval.drivers.rollout_aggregate import aggregate_rollout_outputs
from whetstone.eval.drivers.row_common import RolloutRowOutput
from whetstone.eval.metadata import metadata_with_purpose
from whetstone.eval.protocol import EvalRequest
from whetstone.eval.traces import ExecutedRowState
from whetstone.execution._file_lock import ensure_private_directory
from whetstone.execution.partials import PartialLog
from whetstone.provider.policy import ProviderExecutionPolicy, default_transport_policy
from whetstone.testing.fakes.eval_procedure import FakeEvalProcedureRunner
from whetstone.testing.fakes.transport import FakeLlmTransport
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)


def _build_driver() -> GraphRolloutEvalDriver:
    return GraphRolloutEvalDriver(
        eval_runner=FakeEvalProcedureRunner(),
        mutation_field=TOY_MUTATION_FIELD,
        render_contract=toy_template_render_contract(),
        transport_factory=lambda policy: FakeLlmTransport(
            transport_policy=policy.transport_policy
        ),
    )


class GraphRolloutDriverTests(unittest.TestCase):
    def test_aggregate_rollout_outputs_mean(self) -> None:
        experiment = build_toy_experiment(num_seeds=1)
        sampling = experiment.eval_configs.internal
        task_hashes = sampling.task_set.task_hashes
        outputs = tuple(
            RolloutRowOutput(
                candidate_id="cand",
                task_id=f"task-{idx}",
                task_index=idx,
                seed_index=0,
                row_state=ExecutedRowState.SUCCESS,
                trace_steps=(),
                output_text="ok",
                score=1.0,
            )
            for idx in range(len(task_hashes))
        )
        result = aggregate_rollout_outputs(
            outputs=outputs,
            task_hashes=task_hashes,
            num_seeds=1,
            graph_hash=experiment.rollout_graph.graph_hash,
            matrix_plan=sampling.evaluation_matrix_plan,
            aggregate_name="score",
        )
        self.assertIsNotNone(result.aggregate.aggregation_output.value)
        self.assertEqual(len(result.per_task_scores), len(task_hashes))

    def test_driver_populates_request_identities(self) -> None:
        experiment = build_toy_experiment(num_seeds=1)
        sampling = experiment.eval_configs.internal
        driver = _build_driver()
        transport = default_transport_policy(api_key_env="WHETSTONE_TOY_API_KEY")
        execution_policy = ProviderExecutionPolicy(transport_policy=transport)
        with tempfile.TemporaryDirectory() as tmp:
            partial_path = Path(tmp).resolve() / "partials.jsonl"
            ensure_private_directory(partial_path.parent)
            partial_log = PartialLog(partial_path)
            request = EvalRequest(
                request_id="test:request-identities",
                candidate=experiment.initial_candidate,
                metadata=metadata_with_purpose("test"),
            )
            result = driver.run(
                experiment=experiment,
                sampling=sampling,
                request=request,
                eval_config_hash="test-hash",
                execution_policy=execution_policy,
                concurrency=2,
                max_wall_seconds=None,
                partial_log=partial_log,
                prompt_cache=None,
            )
        self.assertGreater(len(result.request_identities), 0)
        for identity in result.request_identities:
            self.assertEqual(len(identity), 64)
            int(identity, 16)

    def test_deadline_produces_missing_rows(self) -> None:
        experiment = build_toy_experiment(num_seeds=1)
        sampling = experiment.eval_configs.internal
        driver = _build_driver()
        transport = default_transport_policy(api_key_env="WHETSTONE_TOY_API_KEY")
        execution_policy = ProviderExecutionPolicy(transport_policy=transport)
        request = EvalRequest(
            request_id="test:deadline",
            candidate=experiment.initial_candidate,
            metadata=metadata_with_purpose("test"),
        )
        result = driver.run(
            experiment=experiment,
            sampling=sampling,
            request=request,
            eval_config_hash="test-hash",
            execution_policy=execution_policy,
            concurrency=1,
            max_wall_seconds=0.0,
            partial_log=None,
            prompt_cache=None,
        )
        self.assertTrue(result.deadline_reached)
        missing = [output for output in result.outputs if output.missing]
        self.assertEqual(len(missing), len(result.outputs))
        for output in missing:
            self.assertEqual(output.failure_code, "deadline")


if __name__ == "__main__":
    unittest.main()
