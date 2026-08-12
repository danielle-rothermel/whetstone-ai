"""Golden pins for the generation work-key derivation."""

from __future__ import annotations

from whetstone.core.identity import (
    IdentityHash,
    compute_identity_hash,
    compute_prefixed_identity_key,
)
from whetstone.evaluation.generation import GenerationIndex
from whetstone.evaluation.work_key import (
    GENERATION_WORK_KEY_PREFIX,
    GENERATION_WORK_KEY_SCHEMA,
    GENERATION_WORK_KEY_SCHEMA_VERSION,
    generation_work_key,
)

_CONFIG_HASH = IdentityHash(
    "00c37b9099b27f60f908231333ca3a33cd4476339fa0282f8eea1a22e8a07e0e"
)


def test_generation_work_key_literals_are_golden() -> None:
    assert GENERATION_WORK_KEY_SCHEMA == "whetstone.generation_work_key"
    assert GENERATION_WORK_KEY_SCHEMA_VERSION == 1
    assert GENERATION_WORK_KEY_PREFIX == "whetstone.generation_work:"
    key = generation_work_key(
        experiment_config_hash=_CONFIG_HASH,
        index=GenerationIndex(task_index=0, sample_index=0),
    )
    assert key == (
        f"{GENERATION_WORK_KEY_PREFIX}"
        "250e14b09ef3704a3819899a8421335be287e80f16b5b58444cc64f977f256fb"
    )
    other = generation_work_key(
        experiment_config_hash=_CONFIG_HASH,
        index=GenerationIndex(task_index=3, sample_index=1),
    )
    assert other == (
        f"{GENERATION_WORK_KEY_PREFIX}"
        "35a10fdd403eb6275271ccf192fb1d9e90849ec9ae0f934e974d4fd4104eb2ba"
    )


def test_work_key_derives_through_canonical_prefixed_key() -> None:
    index = GenerationIndex(task_index=2, sample_index=5)
    payload = {
        "experiment_config_hash": str(_CONFIG_HASH),
        "task_index": 2,
        "sample_index": 5,
    }
    assert generation_work_key(
        experiment_config_hash=_CONFIG_HASH, index=index
    ) == compute_prefixed_identity_key(
        schema=GENERATION_WORK_KEY_SCHEMA,
        schema_version=GENERATION_WORK_KEY_SCHEMA_VERSION,
        prefix=GENERATION_WORK_KEY_PREFIX,
        payload=payload,
    )
    assert generation_work_key(
        experiment_config_hash=_CONFIG_HASH, index=index
    ) == GENERATION_WORK_KEY_PREFIX + compute_identity_hash(
        schema=GENERATION_WORK_KEY_SCHEMA,
        schema_version=GENERATION_WORK_KEY_SCHEMA_VERSION,
        payload=payload,
    )
