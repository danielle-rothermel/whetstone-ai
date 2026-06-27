from __future__ import annotations

from types import ModuleType

import pytest

import dr_dspy.humaneval_direct_dbos as direct_dbos


@pytest.fixture(scope="session")
def eval_dbos_harness() -> ModuleType:
    return direct_dbos
