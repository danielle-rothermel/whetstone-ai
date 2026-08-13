from __future__ import annotations

import unittest

from whetstone.eval.drivers.graph_row_request import decode_graph_row_output
from whetstone.eval.drivers.graph_worker import run_row
from whetstone.eval.traces import ExecutedRowState
from whetstone.execution.fanout import CallSpec, ProcessJob, run_call_pool
from tests.eval.drivers._helpers import sample_graph_row_request


class SubprocessGraphRolloutTests(unittest.TestCase):
    def test_run_call_pool_one_graph_row(self) -> None:
        request = sample_graph_row_request()
        specs = [
            CallSpec(
                key=(0, 0),
                job=ProcessJob(
                    entrypoint="whetstone.eval.drivers.graph_worker:run_row",
                    payload=request.model_dump(mode="json"),
                ),
                decode=lambda payload, req=request: decode_graph_row_output(
                    payload, request=req
                ),
                deadline_seconds=60.0,
            )
        ]
        outcome = run_call_pool(
            specs,
            concurrency=1,
            is_rate_limited=lambda _output: False,
            max_wall_seconds=None,
        )
        self.assertFalse(outcome.deadline_reached)
        self.assertEqual(len(outcome.results), 1)
        result = outcome.results[0]
        self.assertTrue(result.completed)
        self.assertIsNotNone(result.value)
        assert result.value is not None
        self.assertEqual(result.value.row_state, ExecutedRowState.SUCCESS)
        self.assertIsNotNone(result.value.score)

    def test_graph_worker_entrypoint_direct(self) -> None:
        request = sample_graph_row_request()
        payload = run_row(request.model_dump(mode="json"))
        output = decode_graph_row_output(payload, request=request)
        self.assertEqual(output.row_state, ExecutedRowState.SUCCESS)


if __name__ == "__main__":
    unittest.main()
