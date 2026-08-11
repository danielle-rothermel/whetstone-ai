from __future__ import annotations

import pytest

from whetstone.envs.oracle_operator import (
    ENV_EXACT_MATCH_NAME,
    ENV_EXACT_MATCH_UNIT,
    ENV_ORACLE_OPERATOR_NAME,
    ENV_ORACLE_OPERATOR_VERSION,
    env_exact_match_fact,
    env_exact_match_score,
)
from whetstone.envs.procedure import (
    env_metric_extraction_config,
    env_procedure_config,
)
from whetstone.envs.registry import ENV_NAMES, env_spec
from whetstone.evaluation import Applicability, MetricFact, Score

_PROC_HASH = "a" * 64


def _fixtures(env_name: str) -> tuple[str, str, str]:
    if env_name == "c22":
        import json

        gold = json.dumps(
            {
                "base_task": "Name a color.",
                "constraint_descriptions": [
                    "no comma",
                    "wrap in quotes",
                    ">=2 placeholders",
                ],
                "instruction_id_list": [
                    "punctuation:no_comma",
                    "startend:quotation",
                    "detectable_content:number_placeholders",
                ],
                "kwargs_list": [{}, {}, {"num_placeholders": 2}],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        correct = '"blue [1] and green [2] together"'
        wrong = "blue, green"
        return gold, correct, wrong
    if env_name == "c22h":
        import json

        gold = json.dumps(
            {
                "base_task": "Name a fruit.",
                "constraint_descriptions": [
                    ">=5 words",
                    "no z",
                    "forbid quarnex",
                ],
                "instruction_id_list": [
                    "length_constraints:number_words",
                    "keywords:letter_frequency",
                    "keywords:forbidden_words",
                ],
                "kwargs_list": [
                    {"num_words": 5, "relation": "at least"},
                    {
                        "letter": "z",
                        "let_frequency": 1,
                        "let_relation": "less than",
                    },
                    {"forbidden_words": ["quarnex"]},
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        correct = "apple banana cherry mango melon"
        wrong = "apple banana cherry mango quarnex"
        return gold, correct, wrong
    if env_name == "c11":
        gold = '{"a":1,"b":2}'
        return gold, '{"a":1,"b":2}', '{"b":2,"a":1}'
    if env_name == "c19":
        return "2,3", "2,3", "9,9"
    if env_name in {"c18", "c18h"}:
        return "True", "True", "False"
    return "abcd", "abcd", "zzzz"


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_correct_response_scores_one(env_name: str) -> None:
    env = env_spec(env_name)
    gold, correct, _ = _fixtures(env_name)
    fact = env_exact_match_fact(
        env=env,
        generation=correct,
        gold=gold,
        evaluation_procedure_config_hash=_PROC_HASH,
    )
    assert isinstance(fact, MetricFact)
    assert fact.name == ENV_EXACT_MATCH_NAME
    assert fact.unit == ENV_EXACT_MATCH_UNIT
    assert fact.applicability is Applicability.APPLICABLE
    assert fact.value == 1
    assert fact.lineage.evaluation_procedure_config_hash == _PROC_HASH
    assert fact.lineage.operator == ENV_ORACLE_OPERATOR_NAME


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_wrong_response_scores_zero(env_name: str) -> None:
    env = env_spec(env_name)
    gold, _, wrong = _fixtures(env_name)
    fact = env_exact_match_fact(
        env=env,
        generation=wrong,
        gold=gold,
        evaluation_procedure_config_hash=_PROC_HASH,
    )
    assert fact.value == 0


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_score_mirrors_the_fact(env_name: str) -> None:
    env = env_spec(env_name)
    gold, correct, _ = _fixtures(env_name)
    score = env_exact_match_score(
        env=env,
        generation=correct,
        gold=gold,
        evaluation_procedure_config_hash=_PROC_HASH,
    )
    assert isinstance(score, Score)
    assert score.name == ENV_EXACT_MATCH_NAME
    assert score.unit == ENV_EXACT_MATCH_UNIT
    assert score.value == 1
    assert score.evaluation_procedure_config_hash == _PROC_HASH
    assert score.derived_from == (ENV_EXACT_MATCH_NAME,)


def test_shared_normalization_is_applied_by_the_oracle() -> None:
    env = env_spec("c18")
    fact = env_exact_match_fact(
        env=env,
        generation="```\nTrue\n```",
        gold="True",
        evaluation_procedure_config_hash=_PROC_HASH,
    )
    assert fact.value == 1


def test_c18_verdict_extraction_flows_through_adapter() -> None:
    env = env_spec("c18")
    cot = "...the query property is not entailed.\n\nFalse"
    fact = env_exact_match_fact(
        env=env,
        generation=cot,
        gold="False",
        evaluation_procedure_config_hash=_PROC_HASH,
    )
    assert fact.value == 1


def test_c23_output_extraction_flows_through_adapter() -> None:
    env = env_spec("c23")
    fact = env_exact_match_fact(
        env=env,
        generation="Output: abcd",
        gold="abcd",
        evaluation_procedure_config_hash=_PROC_HASH,
    )
    assert fact.value == 1


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_procedure_identity_is_env_distinct(env_name: str) -> None:
    proc = env_procedure_config(env_spec(env_name))
    others = {
        env_procedure_config(env_spec(other)).config_hash
        for other in ENV_NAMES
        if other != env_name
    }
    assert proc.config_hash not in others


def test_operator_version_is_resolved_explicitly() -> None:
    config = env_metric_extraction_config(env_spec(ENV_NAMES[0]))
    assert config.resolved_operator_versions == (
        (ENV_ORACLE_OPERATOR_NAME, ENV_ORACLE_OPERATOR_VERSION),
    )
