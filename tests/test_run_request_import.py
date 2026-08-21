from __future__ import annotations

import sys
from unittest.mock import patch


def test_run_request_imports_without_dbos() -> None:
    with patch.dict(sys.modules, {"dbos": None}):
        from whetstone.coordination.harness_run_controller import RunRequest

        request = RunRequest(
            controller_identity_hash="a" * 64,
            run_id="run-1",
            control_identity_hash="b" * 64,
        )
        assert request.run_id == "run-1"
        assert request.identity_hash()
