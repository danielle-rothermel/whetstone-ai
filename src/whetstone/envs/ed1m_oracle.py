"""The ed1m execution oracle: dual scoring vs mutant and canonical behavior.

A reconstructed program is scored by the canonical dr-code mutant oracle on
the authenticated record's complete input sequence. Its typed outcomes are
compared with both persisted expected-outcome vectors:

  * ``fidelity_to_mutant`` -- the fraction of ALL inputs whose reconstruction
    outcome matches the mutant's expected outcome. This is the TASK metric (the
    reward-bearing one, blended with compression per task 22): the enc-dec
    channel should faithfully reconstruct the buggy program's behavior.
  * ``attractor_pull`` -- the fraction of the DISCRIMINATING inputs (mutant !=
    canonical) whose reconstruction outcome matches the CANONICAL expected
    outcome (the reconstruction "fixed" the seeded bug toward the training-
    data attractor). This is the REPORTED contamination measurement -- NEVER a
    reward objective. ``None`` when a mutant has no discriminating inputs.

Comparison is exact ``ExpectedOutcome`` equality. A canonical oracle failure
marks the row infrastructure-unknown and never converts it into score zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from dr_code.execution import PythonSubprocessRunner, run_python_subprocess
from dr_code.humaneval import DEFAULT_HUMANEVAL_TIMEOUT_SECONDS
from dr_code.mutants.dataset import ExpectedOutcome, MutantRecord
from dr_code.mutants.oracle import (
    OracleError,
    run_program_on_inputs,
)


@dataclass(frozen=True, slots=True)
class MutantScore:
    """One reconstruction's dual score against a mutant's oracle.

    ``fidelity_to_mutant`` is the reward-bearing task metric (fraction of all
    inputs matching the mutant); ``attractor_pull`` is the reported measurement
    (fraction of the DISCRIMINATING inputs matching canonical; ``None`` when
    mutant has none). ``infrastructure_unknown`` marks a row that could not be
    scored because the canonical oracle failed -- the rollout fails, never
    scores 0.
    """

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
    run_in_subprocess: PythonSubprocessRunner = run_python_subprocess,
    timeout_seconds: float = DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
) -> MutantScore:
    """Dual-score a reconstruction against a mutant's per-input oracle.

    Runs the reconstruction's ``entry_point`` once over the complete recorded
    input sequence and compares the canonical typed outcomes to the mutant's
    and canonical's expected outcomes. Any oracle execution or protocol error
    makes the whole row infrastructure-unknown.
    """
    distinct = frozenset(mutant.distinct_input_indices)
    total = len(mutant.input_reprs)
    try:
        program_outcomes = run_program_on_inputs(
            program=reconstruction,
            entry_point=mutant.entry_point,
            input_reprs=mutant.input_reprs,
            timeout_seconds=timeout_seconds,
            runner=run_in_subprocess,
        )
    except OracleError:
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
            kind=outcome.kind.value,
            output_repr=outcome.output_repr,
        )
        for outcome in program_outcomes.outcomes
    )
    matched_mutant = 0
    matched_canonical = 0
    for i, outcome in enumerate(observed):
        if outcome == mutant.mutant_expected[i]:
            matched_mutant += 1
        if i in distinct:
            if outcome == mutant.canonical_expected[i]:
                matched_canonical += 1
    fidelity = matched_mutant / total if total else None
    attractor = matched_canonical / len(distinct) if distinct else None
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
