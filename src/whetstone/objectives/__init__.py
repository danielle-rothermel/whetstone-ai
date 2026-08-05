"""Objectives, Objective Vectors, Pareto Fronts, and official selection.

Workstream 8 of ``design/concrete-changes.html``: direction-bearing
:class:`Objective` criteria deterministically derived from Scores / Rollout
Aggregates, ordered :class:`ObjectiveVector` tuples, deterministic
:class:`ParetoFront` construction (stable ordering, explicit tie behavior,
direction per objective), and the official-selection procedure over complete
certified aggregate evidence with persisted derivation / order / tie /
selection evidence.

The one non-negotiable invariant — **Reward is never accepted as an
Objective** — is enforced at the type level:
:class:`ObjectiveDerivationSource` has no Reward member, and :class:`Objective`
rejects the reserved ``reward`` name at construction
(:class:`RewardIsNotAnObjectiveError`).
"""

from whetstone.objectives.objective import (
    RESERVED_OBJECTIVE_NAMES,
    Direction,
    Objective,
    ObjectiveDerivation,
    ObjectiveDerivationSource,
    ObjectiveVector,
    ParetoFront,
    ParetoMember,
    RewardIsNotAnObjectiveError,
    TieBehavior,
    dominates,
    objective_from_aggregate_value,
    objective_from_score_value,
    pareto_front,
    reject_reward_name,
)
from whetstone.objectives.selection import (
    IncompleteEvidenceError,
    ObjectiveSpec,
    SelectionCandidate,
    SelectionEvidence,
    select_official,
)

__all__ = [
    "RESERVED_OBJECTIVE_NAMES",
    "Direction",
    "IncompleteEvidenceError",
    "Objective",
    "ObjectiveDerivation",
    "ObjectiveDerivationSource",
    "ObjectiveSpec",
    "ObjectiveVector",
    "ParetoFront",
    "ParetoMember",
    "RewardIsNotAnObjectiveError",
    "SelectionCandidate",
    "SelectionEvidence",
    "TieBehavior",
    "dominates",
    "objective_from_aggregate_value",
    "objective_from_score_value",
    "pareto_front",
    "reject_reward_name",
    "select_official",
]
