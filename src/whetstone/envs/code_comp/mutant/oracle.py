"""ED1M dual scoring through the released dr-code/dr-exec boundary."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

from dr_code.core.execution.executor import (
    ExecutionKilledError,
    ExecutionOutputLimitError,
    ExecutionTimeoutError,
    run_python_source,
)
from dr_code.humaneval import DEFAULT_HUMANEVAL_TIMEOUT_SECONDS
from dr_exec import DeclarationError, Executor, ExecutorFailure
from pydantic import BaseModel, ConfigDict, ValidationError

from whetstone.envs.code_comp.mutant.dataset import (
    ExpectedOutcome,
    MutantRecord,
)

_RESULT_BEGIN: Final = "<<<WHETSTONE_ED1M_V1_BEGIN>>>"
_RESULT_END: Final = "<<<WHETSTONE_ED1M_V1_END>>>"
_PROTOCOL_VERSION: Final = 1


class _OutcomeKind(StrEnum):
    VALUE = "value"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class _InputOutcome:
    kind: _OutcomeKind
    output_repr: str


class _WireOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["value", "error"]
    output_repr: str


class _WireEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1]
    invocation_id: str
    outcomes: list[_WireOutcome]


class _OracleError(RuntimeError):
    """The execution oracle did not produce a trustworthy complete result."""


_CANDIDATE_SOURCE: Final = """
import ast
import json
import os
import sys

request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
trusted_fd = os.dup(1)
try:
    with open(os.devnull, "w") as discarded:
        os.dup2(discarded.fileno(), 1)
        namespace = {}
        exec(request["program"], namespace)
        function = namespace[request["entry_point"]]
        outcomes = []
        for input_repr in request["input_reprs"]:
            args = ast.literal_eval(input_repr)
            try:
                value = function(*args)
                outcomes.append({"kind": "value", "output_repr": repr(value)})
            except Exception as exc:
                outcomes.append(
                    {"kind": "error", "output_repr": type(exc).__name__}
                )
    os.write(trusted_fd, json.dumps(outcomes, sort_keys=True).encode())
finally:
    os.close(trusted_fd)
"""


_RUNNER_SOURCE: Final = f"""
import json
import os
import subprocess
import sys

def _validate_outcomes(raw_outcomes, expected_count):
    if type(raw_outcomes) is not list or len(raw_outcomes) != expected_count:
        raise ValueError("candidate returned the wrong number of outcomes")
    for outcome in raw_outcomes:
        if (
            type(outcome) is not dict
            or set(outcome) != {{"kind", "output_repr"}}
            or outcome["kind"] not in ("value", "error")
            or type(outcome["output_repr"]) is not str
        ):
            raise ValueError("candidate returned an invalid outcome")
    return raw_outcomes

def dr_exec_main(request, emit):
    del emit
    payload = request["payload"]
    trusted_fd = os.dup(1)
    try:
        with open(os.devnull, "w") as discarded:
            os.dup2(discarded.fileno(), 1)
            candidate_request = json.dumps(
                {{
                    "entry_point": payload["entry_point"],
                    "input_reprs": payload["input_reprs"],
                    "program": payload["program"],
                }},
                sort_keys=True,
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-c", {_CANDIDATE_SOURCE!r}],
                input=candidate_request,
                capture_output=True,
                check=False,
                close_fds=True,
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"candidate child exited {{completed.returncode}}"
                )
            outcomes = _validate_outcomes(
                json.loads(completed.stdout),
                len(payload["input_reprs"]),
            )
        envelope = {{
            "invocation_id": payload["invocation_id"],
            "protocol_version": {_PROTOCOL_VERSION},
            "outcomes": outcomes,
        }}
        encoded = (
            {_RESULT_BEGIN!r}
            + json.dumps(envelope, sort_keys=True)
            + {_RESULT_END!r}
        ).encode()
        os.write(trusted_fd, encoded)
    finally:
        os.close(trusted_fd)
"""


@dataclass(frozen=True, slots=True)
class MutantScore:
    fidelity_to_mutant: float | None
    attractor_pull: float | None
    matched_mutant: int
    matched_canonical_on_distinct: int
    total_inputs: int
    distinct_inputs: int
    infrastructure_unknown: bool


def score_ed1m_reconstruction(
    *,
    reconstruction: str,
    mutant: MutantRecord,
    executor: Executor,
) -> MutantScore:
    """Dual-score one reconstruction against an authenticated mutant."""

    distinct = frozenset(mutant.distinct_input_indices)
    total = len(mutant.input_reprs)
    try:
        outcomes = _run_program_on_inputs(
            program=reconstruction,
            entry_point=mutant.entry_point,
            input_reprs=mutant.input_reprs,
            timeout_seconds=DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
            executor=executor,
        )
    except _OracleError:
        return MutantScore(
            fidelity_to_mutant=None,
            attractor_pull=None,
            matched_mutant=0,
            matched_canonical_on_distinct=0,
            total_inputs=total,
            distinct_inputs=len(distinct),
            infrastructure_unknown=True,
        )

    observed = tuple(
        ExpectedOutcome(
            kind=outcome.kind.value, output_repr=outcome.output_repr
        )
        for outcome in outcomes
    )
    matched_mutant = 0
    matched_canonical = 0
    for index, outcome in enumerate(observed):
        if outcome == mutant.mutant_expected[index]:
            matched_mutant += 1
        if index in distinct and outcome == mutant.canonical_expected[index]:
            matched_canonical += 1
    return MutantScore(
        fidelity_to_mutant=(matched_mutant / total if total else None),
        attractor_pull=(
            matched_canonical / len(distinct) if distinct else None
        ),
        matched_mutant=matched_mutant,
        matched_canonical_on_distinct=matched_canonical,
        total_inputs=total,
        distinct_inputs=len(distinct),
        infrastructure_unknown=False,
    )


def _run_program_on_inputs(
    *,
    program: str,
    entry_point: str,
    input_reprs: tuple[str, ...],
    timeout_seconds: float,
    executor: Executor,
) -> tuple[_InputOutcome, ...]:
    invocation_id = secrets.token_hex(32)
    try:
        input_json = json.dumps(
            {
                "entry_point": entry_point,
                "input_reprs": input_reprs,
                "invocation_id": invocation_id,
                "program": program,
            },
            sort_keys=True,
        )
        completed = run_python_source(
            executor,
            source=_RUNNER_SOURCE,
            input_json=input_json,
            timeout_seconds=timeout_seconds,
        )
    except (
        DeclarationError,
        ExecutorFailure,
        ExecutionKilledError,
        ExecutionOutputLimitError,
        ExecutionTimeoutError,
        TypeError,
        ValueError,
    ) as exc:
        raise _OracleError(str(exc)) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().replace("\n", " ")[:200]
        raise _OracleError(
            f"oracle child exited {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
    return _parse_outcomes(
        completed.stdout,
        expected_count=len(input_reprs),
        expected_invocation_id=invocation_id,
    )


def _parse_outcomes(
    stdout: str,
    *,
    expected_count: int,
    expected_invocation_id: str,
) -> tuple[_InputOutcome, ...]:
    if (
        not stdout.startswith(_RESULT_BEGIN)
        or not stdout.endswith(_RESULT_END)
        or stdout.count(_RESULT_BEGIN) != 1
        or stdout.count(_RESULT_END) != 1
    ):
        raise _OracleError("oracle did not emit one complete final envelope")
    try:
        envelope = _WireEnvelope.model_validate_json(
            stdout[len(_RESULT_BEGIN) : -len(_RESULT_END)]
        )
    except ValidationError as exc:
        raise _OracleError(
            "oracle result did not match protocol version 1"
        ) from exc
    if not secrets.compare_digest(
        envelope.invocation_id, expected_invocation_id
    ):
        raise _OracleError("oracle invocation binding mismatch")
    if len(envelope.outcomes) != expected_count:
        raise _OracleError(
            f"oracle returned {len(envelope.outcomes)} outcomes; "
            f"expected {expected_count}"
        )
    return tuple(
        _InputOutcome(
            kind=_OutcomeKind(outcome.kind),
            output_repr=outcome.output_repr,
        )
        for outcome in envelope.outcomes
    )


__all__ = ["MutantScore", "score_ed1m_reconstruction"]
