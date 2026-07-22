"""Stale-name scan: proves no retired rollout-config / graph-wiring spelling,
alias, or dual-read branch survives in whetstone source or tests
(deliverable 1).

whetstone-ai never shipped a rollout-config model, so this is an
absence-by-construction guard: it fails loudly if any migration-source
spelling is ever (re)introduced. The migration-source names below come from
the dr-graph break (build-dr-graph.md) plus the retired Whetstone
rollout-config vocabulary named in the migration authority.
"""

from __future__ import annotations

import re
from pathlib import Path

# Retired migration-source spellings. If any of these reappears in whetstone
# source or tests, the migration has regressed to an alias / dual-read path.
RETIRED_NAMES = (
    # dr-graph break: retired graph-config / wiring / hash spellings.
    "GraphSpec",
    "NodeSpec",
    "FieldSpec",
    "BindingRef",
    "BindingSource",
    "graph_digest",
    "GRAPH_DIGEST_LENGTH",
    "canonical_graph_payload",
    "input_bindings",
    "as_binding_ref",
    "validate_external_bindings",
    "external_binding_fields",
    "validate_binding_ref",
    "validate_graph_spec",
    "graph_digests_golden",
    # Retired Whetstone rollout-config identity (never adopt a parallel one).
    "RolloutConfig",
    "rollout_config",
    "rollout_config_hash",
    "RolloutConfigHash",
    "rollout_config_schema",
    # No standalone Character Budget policy artifact.
    "CharacterBudgetPolicy",
)

_SRC = Path(__file__).resolve().parents[2] / "src"
_TESTS = Path(__file__).resolve().parents[1]
_THIS_FILE = Path(__file__).resolve()


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for root in (_SRC, _TESTS):
        files.extend(root.rglob("*.py"))
    # The scan file itself names the retired spellings and is exempt.
    return [path for path in files if path.resolve() != _THIS_FILE]


_PATTERNS = {
    name: re.compile(rf"\b{re.escape(name)}\b") for name in RETIRED_NAMES
}


def test_no_retired_names_survive() -> None:
    offenders: dict[str, list[str]] = {}
    for path in _scanned_files():
        text = path.read_text(encoding="utf-8")
        hits = [
            name
            for name, pattern in _PATTERNS.items()
            if pattern.search(text)
        ]
        if hits:
            offenders[str(path)] = hits
    assert not offenders, (
        f"retired migration-source spellings found: {offenders}"
    )


def test_scan_covers_source_and_tests() -> None:
    files = {str(path) for path in _scanned_files()}
    assert any("/src/whetstone/graph/" in f for f in files)
    assert any("/tests/graph/" in f for f in files)
