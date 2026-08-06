import pytest

from whetstone.core.roles import EvaluationRole


def test_evaluation_role_values_are_stable() -> None:
    assert EvaluationRole.INTERNAL.value == "internal"
    assert EvaluationRole.OFFICIAL.value == "official"


def test_evaluation_role_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="not a valid EvaluationRole"):
        EvaluationRole("candidate")
