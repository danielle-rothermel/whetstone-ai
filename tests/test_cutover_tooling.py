from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from whetstone.platform.cutover_tooling import (
    APP,
    EXPECTED_CELLS,
    _initialize_dbos_store,
    generate_estimates,
    validate_estimates,
)


def _campaign(tmp_path: Path) -> Path:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    rows = [
        {"cell_id": f"cell-{index}", "model": "model-a"}
        for index in range(EXPECTED_CELLS)
    ]
    (campaign / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return campaign


def _prices(tmp_path: Path, *, output_price: str = "0.2") -> Path:
    path = tmp_path / "prices.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "effective_at": "2026-07-13T00:00:00Z",
                "currency": "USD",
                "assumptions_version": "legacy-token-envelope-v1",
                "source": "operator-reviewed-price-snapshot",
                "models": {
                    "model-a": {
                        "input_usd_per_million": "0.1",
                        "output_usd_per_million": output_price,
                        "assumed_input_tokens": 10,
                        "assumed_output_tokens": 20,
                    }
                },
            }
        )
    )
    return path


def test_estimate_artifact_is_complete_hashed_and_valid(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    payload = generate_estimates(campaign, _prices(tmp_path))
    artifact = tmp_path / "estimates.json"
    artifact.write_text(json.dumps(payload))

    validate_estimates(campaign, artifact)
    cells = cast(dict[str, str], payload["cells"])
    provenance = cast(dict[str, Any], payload["provenance"])
    assert len(cells) == EXPECTED_CELLS
    assert provenance["implicit_price_fetch"] is False


def test_estimate_generation_rejects_ceiling(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exceeds ceiling"):
        generate_estimates(
            _campaign(tmp_path), _prices(tmp_path, output_price="100000")
        )


def test_estimate_validation_rejects_tampering(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    payload = generate_estimates(campaign, _prices(tmp_path))
    cast(dict[str, str], payload["cells"])["cell-0"] = "NaN"
    artifact = tmp_path / "estimates.json"
    artifact.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="checksum"):
        validate_estimates(campaign, artifact)


def test_store_prepare_defaults_to_zero_mutation(tmp_path: Path) -> None:
    descriptor = tmp_path / "stores.json"
    result = CliRunner().invoke(
        APP,
        [
            "stores",
            "prepare",
            "--run-id",
            "acceptance_171",
            "--descriptor",
            str(descriptor),
        ],
    )

    assert result.exit_code == 0
    assert not descriptor.exists()
    assert "whetstone_run_acceptance_171" in result.stdout
    assert "postgresql" not in result.stdout


def test_store_prepare_execute_requires_exact_confirmation(
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(
        APP,
        [
            "stores",
            "prepare",
            "--run-id",
            "acceptance_171",
            "--descriptor",
            str(tmp_path / "stores.json"),
            "--execute",
            "--confirm",
            "wrong",
        ],
    )

    assert result.exit_code != 0
    assert "equal to run ID" in result.output


def test_dbos_store_is_initialized_not_just_touched(tmp_path: Path) -> None:
    path = tmp_path / "dbos.sqlite3"

    _initialize_dbos_store(path, "acceptance_171")

    assert path.stat().st_size > 0
