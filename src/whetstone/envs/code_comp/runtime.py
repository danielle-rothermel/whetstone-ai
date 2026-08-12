from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dr_code.core.execution.executor import (
    host_process_executor,
    run_python_source,
)
from dr_exec import ProcessExecutor
from dr_serialize import IdentityDocument
from pydantic import BaseModel, ConfigDict, StrictStr

from whetstone.core.identity import compute_identity_hash
from whetstone.experiment.binding import ExecutionEnvironmentFingerprint

CODE_COMP_SCORING_RUNTIME_SCHEMA = "whetstone.code_comp_scoring_runtime"
CODE_COMP_SCORING_RUNTIME_SCHEMA_VERSION = 1

_RUNTIME_PROBE_SOURCE = """\
import json
import platform
import sys

def dr_exec_main(request, emit):
    import numpy
    print(json.dumps({
        "implementation": platform.python_implementation(),
        "numpy_version": numpy.__version__,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }, sort_keys=True, separators=(",", ":")))
"""


class CodeCompRuntimeProbe(BaseModel):
    """Validated dependency probe returned by the code-evaluation runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    implementation: StrictStr
    numpy_version: StrictStr
    python_executable: StrictStr
    python_version: StrictStr


class EncDecScoringRuntimeSummary(BaseModel):
    """Runtime identity displayed and persisted with a scoring preview."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_python: StrictStr
    dr_code_version: StrictStr
    runtime_hash: StrictStr
    probe: CodeCompRuntimeProbe


def code_comp_environment_fingerprint(
    runtime: EncDecScoringRuntimeSummary,
) -> ExecutionEnvironmentFingerprint:
    return ExecutionEnvironmentFingerprint(
        dependency_versions=(
            ("dr-code", runtime.dr_code_version),
            ("numpy", runtime.probe.numpy_version),
        ),
        runtime_hash=runtime.runtime_hash,
    )


@dataclass(frozen=True, slots=True)
class CodeCompScoringRuntime:
    """One explicit code executor and its cache-scope identity."""

    executor: ProcessExecutor
    runtime_document: IdentityDocument
    runtime_hash: str
    probe: CodeCompRuntimeProbe


def _require_copied_python(path: Path) -> Path:
    executable = path.expanduser().absolute()
    if not executable.is_file():
        raise ValueError(f"evaluation Python is not a file: {executable}")
    if executable.is_symlink():
        raise ValueError(
            "evaluation Python must be a copied executable, not a symlink: "
            f"{executable}"
        )
    return executable


def build_code_comp_scoring_runtime(
    *,
    runtime_executable: Path,
    record_root: Path,
) -> CodeCompScoringRuntime:
    """Construct and validate the explicit HumanEval execution runtime."""

    executable = _require_copied_python(runtime_executable)
    record_root.mkdir(parents=True, exist_ok=True)
    # Whetstone runs this module with dr-code's evaluation-runtime branch
    # layered editably until the same API is released as dr-code 0.1.5.
    executor_factory = cast(
        Callable[..., ProcessExecutor], host_process_executor
    )
    executor = executor_factory(
        record_root,
        runtime_executable=executable,
    )
    completed = run_python_source(
        executor,
        source=_RUNTIME_PROBE_SOURCE,
        input_json="{}",
        timeout_seconds=10.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "evaluation runtime dependency probe failed: "
            + completed.stderr.strip()
        )
    try:
        raw_probe = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "evaluation runtime dependency probe returned invalid JSON"
        ) from exc
    probe = CodeCompRuntimeProbe.model_validate(raw_probe)
    if Path(probe.python_executable).resolve() != executable.resolve():
        raise RuntimeError(
            "evaluation runtime probe used a different Python executable"
        )
    runtime_description = executor.runtime.describe().id_doc.to_json_dict()
    payload = {
        "runtime": runtime_description,
        "packages": probe.model_dump(mode="json"),
    }
    identity = IdentityDocument(
        schema=CODE_COMP_SCORING_RUNTIME_SCHEMA,
        schema_version=CODE_COMP_SCORING_RUNTIME_SCHEMA_VERSION,
        payload=payload,
    )
    return CodeCompScoringRuntime(
        executor=executor,
        runtime_document=identity,
        runtime_hash=compute_identity_hash(
            schema=CODE_COMP_SCORING_RUNTIME_SCHEMA,
            schema_version=CODE_COMP_SCORING_RUNTIME_SCHEMA_VERSION,
            payload=payload,
        ),
        probe=probe,
    )


__all__ = [
    "CODE_COMP_SCORING_RUNTIME_SCHEMA",
    "CODE_COMP_SCORING_RUNTIME_SCHEMA_VERSION",
    "CodeCompRuntimeProbe",
    "CodeCompScoringRuntime",
    "EncDecScoringRuntimeSummary",
    "build_code_comp_scoring_runtime",
    "code_comp_environment_fingerprint",
]
