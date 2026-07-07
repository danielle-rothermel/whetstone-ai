"""Golden identity tests for the composable migration.

Assert that current code reproduces the committed golden fixtures
byte-for-byte. These fixtures freeze the identity contract (canonical
JSON, digests, record ID axes, graph digests, parser/scoring outputs
under the v1 profiles) captured before any extraction stage; every
migration stage must keep them green. If one of these tests fails and
identity cannot be restored, stop the stage and write up the
discrepancy — do not regenerate the fixtures.

Regenerate (only before Stage 0 lands, never to paper over a
migration-caused mismatch) with:
    uv run python scripts/golden/generate_golden_fixtures.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    REPO_ROOT / "scripts" / "golden" / "generate_golden_fixtures.py"
)
GOLDEN_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "golden"


def load_generator_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_golden_fixtures", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return load_generator_module()


def stored_fixture(filename: str) -> dict[str, Any]:
    path = GOLDEN_FIXTURES_DIR / filename
    assert path.exists(), (
        f"missing golden fixture {path}; generate it with "
        "`uv run python scripts/golden/generate_golden_fixtures.py`"
    )
    return json.loads(path.read_text())


def test_golden_hashing_fixture_reproduces(generator: ModuleType) -> None:
    stored = stored_fixture(generator.HASHING_FIXTURE)
    assert generator.hashing_payload() == stored


def test_golden_graph_digests_fixture_reproduces(
    generator: ModuleType,
) -> None:
    stored = stored_fixture(generator.GRAPH_DIGESTS_FIXTURE)
    assert generator.graph_digests_payload() == stored


def test_golden_record_ids_fixture_reproduces(
    generator: ModuleType,
) -> None:
    stored = stored_fixture(generator.RECORD_IDS_FIXTURE)
    assert generator.record_ids_payload() == stored


def test_golden_parser_scoring_fixture_reproduces(
    generator: ModuleType,
) -> None:
    stored = stored_fixture(generator.PARSER_SCORING_FIXTURE)
    assert generator.parser_scoring_payload() == stored
