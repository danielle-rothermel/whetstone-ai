"""Per-layer golden pins for the layered canonical experiment config.

Every layer payload, layer hash, and config hash below is persisted format
under ``whetstone.code_comp.experiment_config`` v2. A change to any literal
is a schema-version event, never a silent edit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whetstone.envs.code_comp.config import (
    CODE_COMP_EXPERIMENT_CONFIG_SCHEMA,
    CODE_COMP_EXPERIMENT_CONFIG_SCHEMA_VERSION,
    default_code_comp_config,
)
from whetstone.envs.code_comp.mode import CodeCompMode
from whetstone.experiment.config_layers import (
    EXPERIMENT_CONFIG_LAYER_SCHEMA,
    EXPERIMENT_CONFIG_LAYER_SCHEMA_VERSION,
    ExperimentConfigLayer,
    experiment_config_layer_hash,
    layered_experiment_config_payload,
)

_SHARED_LAYER_PAYLOADS = {
    ExperimentConfigLayer.COMP: {"algorithm": "zstd", "level": 19},
    ExperimentConfigLayer.PROFILES: {
        "num_samples": 3,
        "completeness": "propagate",
        "max_skip_fraction": 0.0,
    },
    ExperimentConfigLayer.PROVIDER_POLICY: {
        "encoder": {"kind": "model", "model": "deepseek/deepseek-v4-flash"},
        "decoder": None,
    },
    ExperimentConfigLayer.SPLITS: {
        "pool": {"snapshot_path": None, "limit": None, "task_ids": None},
        "split": {
            "internal_n": None,
            "official_n": None,
            "split_manifest": None,
            "exclude_task_ids": [],
        },
    },
}

_SHARED_LAYER_HASHES = {
    ExperimentConfigLayer.COMP: (
        "d4c60be31f96ffa103ab277415f149988d295f16c8a909ab071b590ddbf0e652"
    ),
    ExperimentConfigLayer.PROFILES: (
        "aa60f59edce2c7ea5bce6f1c01d86734042acaec23bd5bf7cb0d12166dd4be94"
    ),
    ExperimentConfigLayer.PROVIDER_POLICY: (
        "2a20fa9ec98124cd571a5996f90c59f2d293387a782cc030ce9be5ff2852d8a8"
    ),
    ExperimentConfigLayer.SPLITS: (
        "ac9656a332dc55d7c855b31656377cd7d6b840f07537cb666f01003a8eb9dfd9"
    ),
}

_ENCDEC_CANDIDATE_BINDING = {
    "initial_candidate_hash": (
        "010a8c79749615daff4a3ee2c3164512c3f6626df20347edd847155432c979ef"
    ),
    "ceiling_candidate_hash": (
        "de09f35b0fe5bf046f48fe0abd73edb97291f5f1005b52f6adcda9d2adcb6f02"
    ),
}


def test_schema_literals_are_pinned() -> None:
    assert (
        CODE_COMP_EXPERIMENT_CONFIG_SCHEMA
        == "whetstone.code_comp.experiment_config"
    )
    assert CODE_COMP_EXPERIMENT_CONFIG_SCHEMA_VERSION == 2
    assert (
        EXPERIMENT_CONFIG_LAYER_SCHEMA == "whetstone.experiment_config_layer"
    )
    assert EXPERIMENT_CONFIG_LAYER_SCHEMA_VERSION == 1
    assert [layer.value for layer in ExperimentConfigLayer] == [
        "graph",
        "candidate_binding",
        "comp",
        "profiles",
        "provider_policy",
        "splits",
    ]


def test_direct_config_layer_payloads_and_hashes_are_golden() -> None:
    config = default_code_comp_config(CodeCompMode.DIRECT)
    payloads = config.layer_payloads()
    assert payloads == {
        **_SHARED_LAYER_PAYLOADS,
        ExperimentConfigLayer.GRAPH: {
            "mode": "direct",
            "direct": {
                "input_arm": "original",
                "rename_token": "target_fxn",
                "model": "deepseek/deepseek-v4-flash",
            },
        },
        ExperimentConfigLayer.CANDIDATE_BINDING: {
            "initial_candidate_hash": (
                "e641cc8cc2c9d08969e48a52bd4215381052e0bc0f84dfbaf5679de4"
                "76d9272c"
            ),
            "ceiling_candidate_hash": (
                "ecec457927998897abf81f4a999af64c05f2d1e374dde49efbf2d34f"
                "764b9899"
            ),
        },
    }
    assert {
        layer: str(digest) for layer, digest in config.layer_hashes().items()
    } == {
        **_SHARED_LAYER_HASHES,
        ExperimentConfigLayer.GRAPH: (
            "dbdbc429acc516dfd3fb8c5f6060ed7ead1731edc0ca595bcb0ee50de1e1bfd0"
        ),
        ExperimentConfigLayer.CANDIDATE_BINDING: (
            "01399d4eb69c7f553f104c8e8707ae42593ca2931de0b5e247e6b281b90f362c"
        ),
    }
    assert str(config.identity_hash()) == (
        "fae239758d3b27beb10f8425c6fb1e5c76ca76b4e870c3a4d3a36aa765ce0479"
    )


def test_encdec_config_layer_payloads_and_hashes_are_golden() -> None:
    config = default_code_comp_config(CodeCompMode.ENCDEC)
    payloads = config.layer_payloads()
    assert payloads == {
        **_SHARED_LAYER_PAYLOADS,
        ExperimentConfigLayer.GRAPH: {
            "mode": "encdec",
            "encdec": {
                "budget_ratio": 0.5,
                "blend_config": {
                    "weight": 0.1,
                    "min_compression_ratio": 0.01,
                    "max_compression_ratio": 4.0,
                    "metric_id": (
                        "primary_score_with_bounded_compression_penalty"
                    ),
                },
            },
        },
        ExperimentConfigLayer.CANDIDATE_BINDING: _ENCDEC_CANDIDATE_BINDING,
    }
    assert {
        layer: str(digest) for layer, digest in config.layer_hashes().items()
    } == {
        **_SHARED_LAYER_HASHES,
        ExperimentConfigLayer.GRAPH: (
            "238f4d5738f53b361c1b682c6ae0219c9cc330ad33aac9113e901bf1812097b1"
        ),
        ExperimentConfigLayer.CANDIDATE_BINDING: (
            "0ae1d3814645ef4df02f627cb950019b919cc396d4f5325a6ef45d49e37d3478"
        ),
    }
    assert str(config.identity_hash()) == (
        "00c37b9099b27f60f908231333ca3a33cd4476339fa0282f8eea1a22e8a07e0e"
    )


def test_mutant_config_layer_payloads_and_hashes_are_golden() -> None:
    config = default_code_comp_config(
        CodeCompMode.ENCDEC_MUTANT,
        artifact_dir=Path("artifacts/ed1m"),
    )
    payloads = config.layer_payloads()
    assert payloads[ExperimentConfigLayer.GRAPH] == {
        "mode": "encdec_mutant",
        "mutant": {
            "artifact_dir": "artifacts/ed1m",
            "exclude_mutant_ids": [],
            "budget_ratio": None,
            "blend_config": None,
        },
    }
    assert (
        payloads[ExperimentConfigLayer.CANDIDATE_BINDING]
        == _ENCDEC_CANDIDATE_BINDING
    )
    assert (
        str(config.layer_hashes()[ExperimentConfigLayer.GRAPH])
        == "2bb43dbc0f3667f55d83aae80498f49d7fcb563bdd3692f74af895b63c03c42b"
    )
    assert str(config.identity_hash()) == (
        "5c5861a33719a179fd4245a0cd6798659cf631f3ea475b9f744c0473f65bb03a"
    )


def test_config_hash_composes_layer_hashes_exactly() -> None:
    config = default_code_comp_config(CodeCompMode.ENCDEC)
    payload = layered_experiment_config_payload(config.layer_payloads())
    assert payload == {
        "layers": {
            layer.value: str(
                experiment_config_layer_hash(
                    layer, config.layer_payloads()[layer]
                )
            )
            for layer in ExperimentConfigLayer
        }
    }


def test_layered_payload_requires_every_layer() -> None:
    config = default_code_comp_config(CodeCompMode.ENCDEC)
    payloads = config.layer_payloads()
    del payloads[ExperimentConfigLayer.SPLITS]
    with pytest.raises(ValueError, match="missing \\['splits'\\]"):
        layered_experiment_config_payload(payloads)


def test_candidate_content_binds_config_identity() -> None:
    direct = default_code_comp_config(CodeCompMode.DIRECT)
    encdec = default_code_comp_config(CodeCompMode.ENCDEC)
    direct_binding = direct.layer_payloads()[
        ExperimentConfigLayer.CANDIDATE_BINDING
    ]
    encdec_binding = encdec.layer_payloads()[
        ExperimentConfigLayer.CANDIDATE_BINDING
    ]
    assert direct_binding != encdec_binding
    assert direct.identity_hash() != encdec.identity_hash()
