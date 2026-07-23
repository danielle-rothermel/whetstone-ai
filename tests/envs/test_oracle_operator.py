"""The whetstone-owned metric-extraction operator over the env oracle.

Proves per env, with hand-built fixtures, that the operator invokes the
env oracle and emits an ``env_exact_match`` Metric Fact (unit ``correct/1``)
and Score; and that the operator is not a dr-code registry operator.
"""

from __future__ import annotations

import pytest
from dr_code.eval import Applicability, MetricFact, Score

from whetstone.envs.oracle_operator import (
    ENV_EXACT_MATCH_NAME,
    ENV_EXACT_MATCH_UNIT,
    ENV_ORACLE_OPERATOR_NAME,
    env_exact_match_fact,
    env_exact_match_score,
)
from whetstone.envs.procedure import env_procedure_config
from whetstone.envs.registry import ENV_NAMES, env_spec

_PROC_HASH = "a" * 64


def _fixtures(env_name: str) -> tuple[str, str, str]:
    """Return ``(gold, correct_response, wrong_response)`` per env.

    Hand-built so the oracle wiring is proven against a known-good and a
    known-bad answer, not a coincidence of a live generation.
    """
    if env_name == "c22":
        # A constraint stack the response must satisfy (no comma; wrapped in
        # quotes; >= 2 bracket placeholders). Gold is the serialized stack.
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
        # c22h reuses the c22 oracle (same serialized-constraint-stack gold),
        # but a pure-hard stack: >= 5 words, no 'z', forbid 'quarnex'.
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
        # Canonical JSON gold; a matching response scores 1.
        gold = '{"a":1,"b":2}'
        return gold, '{"a":1,"b":2}', '{"b":2,"a":1}'
    if env_name == "c19":
        return "2,3", "2,3", "9,9"
    if env_name == "c18":
        return "True", "True", "False"
    # c23: a transformed string gold.
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
    # A fenced/whitespaced correct answer still scores 1 because the env
    # oracle applies the shared normalize() before comparing.
    env = env_spec("c18")
    fact = env_exact_match_fact(
        env=env,
        generation="```\nTrue\n```",
        gold="True",
        evaluation_procedure_config_hash=_PROC_HASH,
    )
    assert fact.value == 1


def test_c18_verdict_extraction_flows_through_adapter() -> None:
    # The env-side verdict extraction (final True/False token) reaches the
    # adapter path: a chain-of-thought ceiling reply ending on the verdict
    # scores 1, not 0. Verbatim shape of the live D5 CoT reply.
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
    # The env-side Output:-line extraction reaches the adapter path: an
    # Output:-prefixed ceiling reply scores 1 against the bare-string gold.
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
        env_procedure_config(env_spec(other)).config_identity_hash
        for other in ENV_NAMES
        if other != env_name
    }
    assert proc.config_identity_hash not in others


def test_operator_is_not_a_dr_code_registry_operator() -> None:
    from dr_code.metrics.registry import REGISTRY

    # The whetstone env-oracle operator is deliberately NOT a dr-code
    # metrics-engine operator: it never registers in dr-code's metric
    # registry.
    assert ENV_ORACLE_OPERATOR_NAME not in REGISTRY
