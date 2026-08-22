"""FakeEvalEngine must bind a real EvalSplit up front."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from whetstone.testing.fakes.engine import FakeEvalEngine


def test_constructing_without_a_sampling_split_fails() -> None:
    with pytest.raises(TypeError, match="sampling_split"):
        FakeEvalEngine(
            eval_config_ref=MagicMock(),
            provider_execution_policy_ref=MagicMock(),
            provider_execution_policy_record={},
            plan_snapshot=MagicMock(),
            sampling=MagicMock(),
        )
