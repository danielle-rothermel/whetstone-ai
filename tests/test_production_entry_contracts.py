"""Hard-cutover contracts for the production runtime/deploy split."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()


def _python_sources(root: Path) -> tuple[Path, ...]:
    return tuple(path for path in root.rglob("*.py") if path.resolve() != SELF)


def test_coordination_and_platform_do_not_import_testing() -> None:
    forbidden = "whetstone.testing"
    offenders: list[str] = []
    for path in (
        *_python_sources(REPO_ROOT / "src" / "whetstone" / "coordination"),
        *_python_sources(REPO_ROOT / "src" / "whetstone" / "platform"),
    ):
        text = path.read_text()
        if forbidden in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_platform_helpers_has_no_dbos_queue_assembly() -> None:
    helper = (
        REPO_ROOT / "tests" / "integration" / "platform_helpers.py"
    ).read_text()
    for token in (
        "DBOS.launch",
        "DBOS.destroy",
        "Queue(",
        "set_stage_capacity",
        "initialize_dbos_runtime",
        "register_scheduled_dispatcher",
        "register_runtime(",
    ):
        assert token not in helper, token


def test_cli_persists_effect_leases_on_the_store_path() -> None:
    cli = (REPO_ROOT / "src" / "whetstone" / "platform" / "cli.py").read_text()
    assert "EffectLeaseAuthority.memory()" not in cli
    assert "EffectLeaseAuthority.sqlite(store_path)" in cli


def test_register_runtime_is_gone() -> None:
    bootstrap = (
        REPO_ROOT
        / "src"
        / "whetstone"
        / "coordination"
        / "runtime_bootstrap.py"
    ).read_text()
    assert "def register_runtime" not in bootstrap
    assert "def build_runtime" in bootstrap
