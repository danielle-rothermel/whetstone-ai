from __future__ import annotations

import sys
from pathlib import Path

import pytest

from whetstone.envs.code_comp.runtime import (
    CodeCompRuntimeProbe,
    EncDecScoringRuntimeSummary,
    build_code_comp_scoring_runtime,
    code_comp_environment_fingerprint,
)


def test_code_comp_environment_fingerprint_copies_runtime_fields() -> None:
    runtime = EncDecScoringRuntimeSummary(
        evaluation_python="/copied/python",
        dr_code_version="0.1.5",
        runtime_hash="a" * 64,
        probe=CodeCompRuntimeProbe(
            implementation="CPython",
            numpy_version="2.0.0",
            python_executable="/copied/python",
            python_version="3.13.0",
        ),
    )
    fingerprint = code_comp_environment_fingerprint(runtime)

    assert fingerprint.dependency_versions == (
        ("dr-code", "0.1.5"),
        ("numpy", "2.0.0"),
    )
    assert fingerprint.runtime_hash == "a" * 64


def test_scoring_runtime_requires_an_existing_python(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not a file"):
        build_code_comp_scoring_runtime(
            runtime_executable=tmp_path / "missing-python",
            record_root=tmp_path / "records",
        )


def test_scoring_runtime_rejects_a_symlinked_python(tmp_path: Path) -> None:
    executable = tmp_path / "python"
    executable.symlink_to(sys.executable)

    with pytest.raises(ValueError, match="copied executable"):
        build_code_comp_scoring_runtime(
            runtime_executable=executable,
            record_root=tmp_path / "records",
        )
