from __future__ import annotations

import tomllib
from datetime import date
from importlib import import_module
from pathlib import Path
from shlex import split
from subprocess import run
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFS_ROOT = REPO_ROOT / ".defs"


def _load_document(path: Path) -> dict[str, object]:
    return cast(dict[str, object], tomllib.loads(path.read_text()))


def _entries(
    document: dict[str, object], key: str
) -> tuple[dict[str, object], ...]:
    value = document.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array of tables")

    entries: list[dict[str, object]] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(f"{key}[{index}] must be a table")
        entries.append(cast(dict[str, object], entry))
    return tuple(entries)


def _required_string(
    entry: dict[str, object], key: str, *, location: str
) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be a non-empty string")
    return value


def _string_list(
    entry: dict[str, object], key: str, *, location: str
) -> tuple[str, ...]:
    value = entry.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(
            f"{location}.{key} must be an array of non-empty strings"
        )
    return tuple(cast(list[str], value))


def _validate_exported_symbol(symbol: str, *, term_name: str) -> None:
    module_name, separator, attribute = symbol.rpartition(".")
    if not separator or not module_name or not attribute:
        raise ValueError(
            f"term {term_name!r} has invalid exported symbol {symbol!r}"
        )

    module = import_module(module_name)
    if not hasattr(module, attribute):
        raise ValueError(
            f"term {term_name!r} exports missing symbol {symbol!r}"
        )

    public_names = getattr(module, "__all__", ())
    if (
        not isinstance(public_names, (list, tuple))
        or attribute not in public_names
    ):
        raise ValueError(
            f"term {term_name!r} maps to non-public symbol {symbol!r}"
        )


def _validate_acyclic(edges: dict[str, tuple[str, ...]]) -> None:
    active: list[str] = []
    complete: set[str] = set()

    def visit(name: str) -> None:
        if name in complete:
            return
        if name in active:
            cycle_start = active.index(name)
            cycle = [*active[cycle_start:], name]
            raise ValueError(f"term relationship cycle: {' -> '.join(cycle)}")

        active.append(name)
        for target in edges[name]:
            visit(target)
        active.pop()
        complete.add(name)

    for name in edges:
        visit(name)


def _validate_terms() -> int:
    terms = _entries(_load_document(DEFS_ROOT / "terms.toml"), "terms")
    names: list[str] = []
    for index, term in enumerate(terms):
        name = _required_string(term, "name", location=f"terms[{index}]")
        if name != name.lower():
            raise ValueError(f"term name must be lowercase: {name!r}")
        names.append(name)

    if len(names) != len(set(names)):
        raise ValueError("term names must be unique")

    known_names = set(names)
    edges: dict[str, tuple[str, ...]] = {}
    for index, (name, term) in enumerate(zip(names, terms, strict=True)):
        location = f"terms[{index}]"
        relationships = _string_list(
            term, "is_a", location=location
        ) + _string_list(term, "part_of", location=location)
        if name in relationships:
            raise ValueError(f"term {name!r} cannot relate to itself")
        unknown = set(relationships) - known_names
        if unknown:
            raise ValueError(
                f"term {name!r} has unknown relationship targets: "
                f"{sorted(unknown)!r}"
            )
        edges[name] = relationships

        for symbol in _string_list(
            term, "exported_symbols", location=location
        ):
            _validate_exported_symbol(symbol, term_name=name)

    _validate_acyclic(edges)
    return len(terms)


def _validate_contracts() -> int:
    contracts = _entries(
        _load_document(DEFS_ROOT / "contracts.toml"), "contracts"
    )
    required = {"title", "statement", "rationale", "date"}
    allowed = required | {"check"}
    titles: list[str] = []

    for index, contract in enumerate(contracts):
        location = f"contracts[{index}]"
        missing = required - contract.keys()
        unexpected = contract.keys() - allowed
        if missing:
            raise ValueError(
                f"{location} is missing fields: {sorted(missing)!r}"
            )
        if unexpected:
            raise ValueError(
                f"{location} has unexpected fields: {sorted(unexpected)!r}"
            )

        title = _required_string(contract, "title", location=location)
        titles.append(title)
        _required_string(contract, "statement", location=location)
        _required_string(contract, "rationale", location=location)
        date_text = _required_string(contract, "date", location=location)
        try:
            parsed_date = date.fromisoformat(date_text)
        except ValueError as error:
            raise ValueError(f"{location}.date must be an ISO date") from error
        if parsed_date.isoformat() != date_text:
            raise ValueError(f"{location}.date must be a canonical ISO date")
        if "check" in contract:
            check = _required_string(contract, "check", location=location)
            command = split(check)
            if not command:
                raise ValueError(f"{location}.check must name a command")
            print(f"verifying contract check: {check}", flush=True)
            run(command, check=True, cwd=REPO_ROOT)

    if len(titles) != len(set(titles)):
        raise ValueError("contract titles must be unique")
    return len(contracts)


def main() -> None:
    term_count = _validate_terms()
    contract_count = _validate_contracts()
    print(
        f"validated {term_count} terms and {contract_count} contracts "
        "including relationship and export semantics"
    )


if __name__ == "__main__":
    main()
