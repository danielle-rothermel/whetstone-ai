"""MIPROv2's three versioned algorithm-local identity domains."""

from __future__ import annotations

import pytest

from whetstone.optimization import (
    DEMO_SET_SCHEMA,
    INSTRUCTION_SCHEMA,
    TRIAL_COMBINATION_SCHEMA,
    DemoPair,
    DemoSetIdentity,
    InstructionIdentity,
    TrialCombinationIdentity,
)


def test_demo_set_identity_over_ordered_pairs() -> None:
    a = DemoSetIdentity(
        pairs=(DemoPair(ground_truth_code="g1", encoded_representation="e1"),)
    )
    b = DemoSetIdentity(
        pairs=(DemoPair(ground_truth_code="g1", encoded_representation="e1"),)
    )
    empty = DemoSetIdentity(pairs=())
    assert a.identity_hash() == b.identity_hash()
    assert a.identity_hash() != empty.identity_hash()
    # Order is identity-bearing.
    reordered = DemoSetIdentity(
        pairs=(
            DemoPair(ground_truth_code="g2", encoded_representation="e2"),
            DemoPair(ground_truth_code="g1", encoded_representation="e1"),
        )
    )
    forward = DemoSetIdentity(
        pairs=(
            DemoPair(ground_truth_code="g1", encoded_representation="e1"),
            DemoPair(ground_truth_code="g2", encoded_representation="e2"),
        )
    )
    assert reordered.identity_hash() != forward.identity_hash()
    assert len(a.identity_hash()) == 64


def test_instruction_identity_is_text_only() -> None:
    # Two attempts producing identical text share one instruction identity;
    # attempt nonce / evidence / cost never enter the identity payload.
    i1 = InstructionIdentity(instruction_text="do the task well")
    i2 = InstructionIdentity(instruction_text="do the task well")
    assert i1.identity_hash() == i2.identity_hash()
    assert i1.identity_payload() == {"instruction_text": "do the task well"}
    with pytest.raises(ValueError, match="instruction_text"):
        InstructionIdentity(instruction_text="")


def test_trial_combination_groups_repeated_trials() -> None:
    ih = InstructionIdentity(instruction_text="x").identity_hash()
    dh = DemoSetIdentity(pairs=()).identity_hash()
    c1 = TrialCombinationIdentity(instruction_hash=ih, demo_set_hash=dh)
    c2 = TrialCombinationIdentity(instruction_hash=ih, demo_set_hash=dh)
    # Excludes trial ID + scores -> repeated trials share the combination hash.
    assert c1.identity_hash() == c2.identity_hash()
    other = TrialCombinationIdentity(
        instruction_hash=InstructionIdentity(
            instruction_text="y"
        ).identity_hash(),
        demo_set_hash=dh,
    )
    assert c1.identity_hash() != other.identity_hash()
    with pytest.raises(ValueError, match="instruction_hash"):
        TrialCombinationIdentity(instruction_hash="short", demo_set_hash=dh)


def test_identity_domains_have_distinct_schemas() -> None:
    assert DEMO_SET_SCHEMA == "miprov2.demo-set"
    assert INSTRUCTION_SCHEMA == "miprov2.instruction"
    assert TRIAL_COMBINATION_SCHEMA == "miprov2.trial-combination"
    # Same content under different domains hashes differently (schema is part
    # of the identity document).
    ih = InstructionIdentity(instruction_text="x").identity_hash()
    dh = DemoSetIdentity(pairs=()).identity_hash()
    assert ih != dh
