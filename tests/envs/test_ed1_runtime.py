from __future__ import annotations

import sys
from pathlib import Path

import pytest

from whetstone.envs.ed1_runtime import build_ed1_scoring_runtime


def test_scoring_runtime_requires_an_existing_python(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not a file"):
        build_ed1_scoring_runtime(
            runtime_executable=tmp_path / "missing-python",
            record_root=tmp_path / "records",
        )


def test_scoring_runtime_rejects_a_symlinked_python(tmp_path: Path) -> None:
    executable = tmp_path / "python"
    executable.symlink_to(sys.executable)

    with pytest.raises(ValueError, match="copied executable"):
        build_ed1_scoring_runtime(
            runtime_executable=executable,
            record_root=tmp_path / "records",
        )
