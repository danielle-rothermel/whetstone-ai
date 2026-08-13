from __future__ import annotations

import unittest

from whetstone.eval.drivers.graph_row_request import (
    GraphRowRequest,
    decode_graph_row_output,
    rollout_row_output_from_worker_payload,
)
from whetstone.eval.drivers.graph_worker import run_row
from whetstone.eval.traces import ExecutedRowState
from tests.eval.drivers._helpers import sample_graph_row_request


class GraphWorkerTests(unittest.TestCase):
    def test_run_row_round_trip(self) -> None:
        request = sample_graph_row_request()
        payload = run_row(request.model_dump(mode="json"))
        self.assertIsInstance(payload, dict)
        output = rollout_row_output_from_worker_payload(payload)
        self.assertEqual(output.candidate_id, request.candidate_id)
        self.assertEqual(output.task_id, request.task_id)
        self.assertEqual(output.row_state, ExecutedRowState.SUCCESS)
        self.assertIsNotNone(output.score)

    def test_decode_graph_row_output_not_dispatched(self) -> None:
        request = sample_graph_row_request()
        from whetstone.execution.fanout import FanoutStatus

        output = decode_graph_row_output(
            {},
            request=request,
            fanout_status=FanoutStatus.NOT_DISPATCHED,
        )
        self.assertTrue(output.missing)
        self.assertEqual(output.failure_code, "not-dispatched")


if __name__ == "__main__":
    unittest.main()
