from __future__ import annotations

import sys
from pathlib import Path

import pytest

from whetstone.envs.ed1_runtime import (
    Ed1RuntimeProbe,
    Ed1ScoringRuntimeSummary,
    build_ed1_scoring_runtime,
    ed1_environment_fingerprint,
)


def test_ed1_environment_fingerprint_copies_runtime_fields() -> None:
    runtime = Ed1ScoringRuntimeSummary(
        evaluation_python="/copied/python",
        dr_code_version="0.1.5",
        runtime_identity_hash="a" * 64,
        probe=Ed1RuntimeProbe(
            implementation="CPython",
            numpy_version="2.0.0",
            python_executable="/copied/python",
            python_version="3.13.0",
        ),
    )
    fingerprint = ed1_environment_fingerprint(runtime)

    assert fingerprint.dependency_versions == (
        ("dr-code", "0.1.5"),
        ("numpy", "2.0.0"),
    )
    assert fingerprint.runtime_identity == "a" * 64


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
