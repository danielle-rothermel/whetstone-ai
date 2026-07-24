"""The ed1m per-input execution oracle: dual scoring vs mutant + canonical.

A reconstructed program is scored by RUNNING its ``entry_point`` on each of the
mutant's recorded test inputs (in the injectable, no-Docker Python subprocess
runner ed1 already uses) and comparing the per-input outcome to BOTH recorded
oracles:

  * ``fidelity_to_mutant`` -- the fraction of ALL inputs whose reconstruction
    outcome matches the mutant's expected outcome. This is the TASK metric (the
    reward-bearing one, blended with compression per task 22): the enc-dec
    channel should faithfully reconstruct the buggy program's behavior.
  * ``attractor_pull`` -- the fraction of the DISCRIMINATING inputs (mutant !=
    canonical) whose reconstruction outcome matches the CANONICAL expected
    outcome (the reconstruction "fixed" the seeded bug toward the training-
    data attractor). This is the REPORTED contamination measurement -- NEVER a
    reward objective. ``None`` when a mutant has no discriminating inputs.

The reconstruction outcome per input is captured by a tiny driver program run
under the subprocess runner: it reads a JSON request from stdin,
stdin, runs ``entry_point(*args)`` and prints ``{"kind": "value"|"error",
"output_repr": ...}`` (repr of the value, or the exception type) to stdout.
Comparison is by exact (kind, output_repr) equality -- the SAME semantics the
artifact's expected outcomes were captured with.

Infrastructure-unknown (subprocess crash / timeout / malformed driver output on
EVERY input) fails the row (never scores 0), matching the ed1 invariant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from dr_code.humaneval import DEFAULT_HUMANEVAL_TIMEOUT_SECONDS
from dr_code.humaneval.subprocess_runner import (
    SubprocessRunner,
    run_python_subprocess,
)

from whetstone.envs.ed1m_dataset import Ed1mMutant, ExpectedOutcome

#: The driver program run under the subprocess runner. Reads a request from
#: stdin, executes ``entry_point(*args)`` inside the reconstructed ``source``,
#: and writes a single-line JSON outcome to stdout. Isolated from the parent by
#: the runner's ``-I`` interpreter mode; no container (matches ed1's scorer).
_DRIVER_SOURCE = r"""
import sys, json, ast
req = json.loads(sys.stdin.read())
ns = {}
try:
    exec(req["source"], ns)
    fn = ns.get(req["entry_point"])
    if fn is None:
        print(json.dumps({"kind": "error", "output_repr": "NameError"}))
        sys.exit(0)
    args = ast.literal_eval(req["args_repr"])
    if not isinstance(args, (list, tuple)):
        args = (args,)
    out = fn(*args)
    print(json.dumps({"kind": "value", "output_repr": repr(out)}))
except Exception as exc:
    print(json.dumps({"kind": "error", "output_repr": type(exc).__name__}))
"""


@dataclass(frozen=True, slots=True)
class MutantScore:
    """One reconstruction's dual score against a mutant's oracle.

    ``fidelity_to_mutant`` is the reward-bearing task metric (fraction of all
    inputs matching the mutant); ``attractor_pull`` is the reported measurement
    (fraction of the DISCRIMINATING inputs matching canonical; ``None`` when
    mutant has none). ``infrastructure_unknown`` marks a row that could not be
    scored (subprocess crash/timeout on every input) -- the rollout fails,
    scores 0.
    """

    fidelity_to_mutant: float | None
    attractor_pull: float | None
    matched_mutant: int
    matched_canonical_on_distinct: int
    total_inputs: int
    distinct_inputs: int
    infrastructure_unknown: bool


def _run_one_input(
    *,
    source: str,
    entry_point: str,
    args_repr: str,
    run_in_subprocess: SubprocessRunner,
    timeout_seconds: float,
) -> ExpectedOutcome | None:
    """Run the reconstruction on one input -> its outcome, else ``None``.

    ``None`` is an INFRASTRUCTURE failure for THIS input (subprocess crash /
    timeout / unparseable driver output) -- distinct from a definitive
    value/error outcome, which is returned as an :class:`ExpectedOutcome`.
    """
    request = json.dumps(
        {
            "source": source,
            "entry_point": entry_point,
            "args_repr": args_repr,
        }
    )
    try:
        completed = run_in_subprocess(
            source=_DRIVER_SOURCE,
            input_json=request,
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        return ExpectedOutcome(
            kind=str(payload["kind"]),
            output_repr=str(payload["output_repr"]),
        )
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def score_ed1m_reconstruction(
    *,
    reconstruction: str,
    mutant: Ed1mMutant,
    run_in_subprocess: SubprocessRunner = run_python_subprocess,
    timeout_seconds: float = DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
) -> MutantScore:
    """Dual-score a reconstruction against a mutant's per-input oracle.

    Runs the reconstruction's ``entry_point`` on every recorded input and
    compares to the mutant's + canonical's expected outcomes. Returns fidelity-
    to-mutant (rewarded) + attractor-pull (reported). If EVERY input is an
    infrastructure failure (never a definitive outcome), the row is
    infrastructure-unknown (fails, never 0); otherwise infrastructure-failed
    inputs count as a fidelity mismatch (an un-runnable reconstruction is
    not faithful) but are EXCLUDED from the attractor-pull denominator (an
    un-runnable reconstruction did not "snap to canonical").
    """
    distinct = frozenset(mutant.distinct_input_indices)
    total = len(mutant.input_reprs)
    matched_mutant = 0
    matched_canonical = 0
    distinct_scored = 0
    any_definitive = False
    for i, args_repr in enumerate(mutant.input_reprs):
        outcome = _run_one_input(
            source=reconstruction,
            entry_point=mutant.entry_point,
            args_repr=args_repr,
            run_in_subprocess=run_in_subprocess,
            timeout_seconds=timeout_seconds,
        )
        if outcome is None:
            # Infrastructure failure for this input: a fidelity MISMATCH (an
            # un-runnable reconstruction is not faithful), excluded from the
            # attractor denominator.
            continue
        any_definitive = True
        if outcome == mutant.mutant_expected[i]:
            matched_mutant += 1
        if i in distinct:
            distinct_scored += 1
            if outcome == mutant.canonical_expected[i]:
                matched_canonical += 1
    if not any_definitive:
        return MutantScore(
            fidelity_to_mutant=None,
            attractor_pull=None,
            matched_mutant=0,
            matched_canonical_on_distinct=0,
            total_inputs=total,
            distinct_inputs=len(distinct),
            infrastructure_unknown=True,
        )
    fidelity = matched_mutant / total if total else None
    attractor = (
        matched_canonical / distinct_scored if distinct_scored else None
    )
    return MutantScore(
        fidelity_to_mutant=fidelity,
        attractor_pull=attractor,
        matched_mutant=matched_mutant,
        matched_canonical_on_distinct=matched_canonical,
        total_inputs=total,
        distinct_inputs=len(distinct),
        infrastructure_unknown=False,
    )


__all__ = [
    "MutantScore",
    "score_ed1m_reconstruction",
]
