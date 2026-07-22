"""Execution-mode detection tests (no real DB / Docker required)."""

from __future__ import annotations

from whetstone.runner import execution_mode as em
from whetstone.runner.execution_mode import (
    ExecutionMode,
    detect_execution_mode,
)


def test_force_mode_skips_probes() -> None:
    for mode in ExecutionMode:
        decision = detect_execution_mode(force=mode)
        assert decision.mode is mode
        assert "forced" in decision.reason


def test_postgres_available_when_probe_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(em, "postgres_available", lambda *_a, **_k: True)
    decision = detect_execution_mode(database_url="postgresql:///scratch")
    assert decision.mode is ExecutionMode.POSTGRES
    assert decision.database_url == "postgresql:///scratch"


def test_docker_postgres_when_no_pg_but_docker_running(monkeypatch) -> None:
    monkeypatch.setattr(em, "postgres_available", lambda *_a, **_k: False)
    monkeypatch.setattr(em, "docker_daemon_running", lambda: True)
    decision = detect_execution_mode()
    assert decision.mode is ExecutionMode.DOCKER_POSTGRES
    assert "Docker" in decision.reason


def test_in_process_when_no_pg_and_no_docker(monkeypatch) -> None:
    monkeypatch.setattr(em, "postgres_available", lambda *_a, **_k: False)
    monkeypatch.setattr(em, "docker_daemon_running", lambda: False)
    decision = detect_execution_mode()
    assert decision.mode is ExecutionMode.IN_PROCESS
    assert "in-process" in decision.reason


def test_docker_disabled_falls_through_to_in_process(monkeypatch) -> None:
    monkeypatch.setattr(em, "postgres_available", lambda *_a, **_k: False)
    monkeypatch.setattr(em, "docker_daemon_running", lambda: True)
    decision = detect_execution_mode(allow_docker=False)
    assert decision.mode is ExecutionMode.IN_PROCESS


def test_postgres_probe_never_raises_on_bad_url() -> None:
    # A clearly-unreachable DSN must return False, not raise.
    assert em.postgres_available("postgresql+psycopg://:0/nope") is False


def test_decision_as_dict_shape() -> None:
    decision = detect_execution_mode(force=ExecutionMode.IN_PROCESS)
    d = decision.as_dict()
    assert d["execution_mode"] == "in-process"
    assert d["database_url_present"] is False
