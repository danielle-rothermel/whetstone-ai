from __future__ import annotations

from whetstone.envs.registry import EnvSpec
from whetstone.evaluation import (
    Applicability,
    MetricFact,
    OperatorLineage,
    Score,
)

#: The Metric / Score name and unit the terminal Eval Node emits.
ENV_EXACT_MATCH_NAME = "env_exact_match"
ENV_EXACT_MATCH_UNIT = "correct/1"

#: The whetstone-owned operator identity folded into the Metric Extraction
#: Config. A behaviour change to the operator is a version bump here, which
#: changes the Procedure/Eval identities (and hence ``graph_hash``).
ENV_ORACLE_OPERATOR_NAME = "whetstone.env_exact_match"
ENV_ORACLE_OPERATOR_VERSION = "1"


def env_exact_match_fact(
    *,
    env: EnvSpec,
    generation: str,
    gold: str,
    evaluation_procedure_config_hash: str,
) -> MetricFact:
    """Score ``generation`` via the env oracle into an ``env_exact_match``
    Metric Fact (unit ``correct/1``).

    Applies the env's shared normalization inside ``score_gold``; the fact is
    always applicable (the oracle grades any generation to 0/1, never
    raising) and carries the Evaluation Procedure Config identity plus the
    whetstone operator lineage.
    """
    score = env.score_gold(generation, gold)
    lineage = OperatorLineage(
        evaluation_procedure_config_hash=evaluation_procedure_config_hash,
        operator=ENV_ORACLE_OPERATOR_NAME,
        operator_version=ENV_ORACLE_OPERATOR_VERSION,
    )
    return MetricFact(
        name=ENV_EXACT_MATCH_NAME,
        value=score,
        unit=ENV_EXACT_MATCH_UNIT,
        applicability=Applicability.APPLICABLE,
        lineage=lineage,
    )


def env_exact_match_score(
    *,
    env: EnvSpec,
    generation: str,
    gold: str,
    evaluation_procedure_config_hash: str,
) -> Score:
    """Derive the ``env_exact_match`` Score from the Metric Fact.

    A 0/1 Score (unit ``correct/1``) named ``env_exact_match``,
    deterministically derived from the ``env_exact_match`` Metric Fact and
    retaining the Evaluation Procedure Config identity in its lineage.
    """
    fact = env_exact_match_fact(
        env=env,
        generation=generation,
        gold=gold,
        evaluation_procedure_config_hash=evaluation_procedure_config_hash,
    )
    return Score(
        name=ENV_EXACT_MATCH_NAME,
        value=fact.value,
        unit=ENV_EXACT_MATCH_UNIT,
        evaluation_procedure_config_hash=evaluation_procedure_config_hash,
        derived_from=(ENV_EXACT_MATCH_NAME,),
    )


__all__ = [
    "ENV_EXACT_MATCH_NAME",
    "ENV_EXACT_MATCH_UNIT",
    "ENV_ORACLE_OPERATOR_NAME",
    "ENV_ORACLE_OPERATOR_VERSION",
    "env_exact_match_fact",
    "env_exact_match_score",
]
