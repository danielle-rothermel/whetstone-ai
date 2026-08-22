"""A Codex run proves its session before it commits any eval budget.

The preflight is the only thing standing between a broken Codex login and
a run that burns its wall budget producing nothing. Each guard is checked
here, and ``prepare_codex_run`` is checked to propagate a failure without
binding the launch -- so no capacity is committed.
"""

from __future__ import annotations

import inspect

import sys
from pathlib import Path

import pytest
from dr_store.sync import open_sqlite

from tests.codex_support import toy_codex_control
from whetstone.coordination.harness_run_controller import load_launch
from whetstone.coordination.runtime_bootstrap import (
    build_runtime,
    prepare_codex_run,
)
from whetstone.core.leasing import EffectLeaseAuthority
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.adapters import MappingAdapterRegistry
from whetstone.optim.codex.adapter import CODEX_ADAPTER_KEY, CodexAdapter
from whetstone.optim.codex.executor import build_codex_executor
from whetstone.optim.codex.preflight import (
    CODEX_AUTH_ENV_KEY,
    CODEX_AUTH_FILENAMES,
    CodexPreflightError,
    codex_auth_preflight,
)
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)

_PREFLIGHT_RUN_ID = "codex-preflight-run"


class _NeverRunner:
    """A runner that fails loudly if a run gets past the preflight."""

    def run(self, request, handle, *, lease_token):  # pragma: no cover
        raise AssertionError("the preflight should have stopped this run")


def _write_binary(directory: Path, *, script: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "codex"
    binary.write_text(script, encoding="utf-8")
    binary.chmod(0o755)
    return binary


def _auth_home(tmp_path: Path) -> Path:
    home = tmp_path / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    (home / CODEX_AUTH_FILENAMES[0]).write_text("{}", encoding="utf-8")
    return home


def test_a_missing_binary_fails_the_preflight(tmp_path) -> None:
    executor = build_codex_executor(run_root=tmp_path / "runs")

    with pytest.raises(CodexPreflightError, match="was not found"):
        codex_auth_preflight(
            executor=executor,
            codex_binary="definitely-not-a-real-codex-binary",
            environment={"PATH": str(tmp_path)},
        )


def test_a_missing_auth_source_fails_the_preflight(tmp_path) -> None:
    binary_dir = tmp_path / "bin"
    _write_binary(binary_dir, script="#!/bin/sh\nexit 0\n")
    executor = build_codex_executor(run_root=tmp_path / "runs")
    empty_home = tmp_path / "empty-codex-home"
    empty_home.mkdir()

    with pytest.raises(CodexPreflightError, match="no usable auth source"):
        codex_auth_preflight(
            executor=executor,
            codex_binary="codex",
            environment={
                "PATH": str(binary_dir),
                "CODEX_HOME": str(empty_home),
            },
        )


def test_an_api_key_satisfies_the_auth_source_check(tmp_path) -> None:
    binary_dir = tmp_path / "bin"
    _write_binary(binary_dir, script="#!/bin/sh\nexit 7\n")
    executor = build_codex_executor(run_root=tmp_path / "runs")
    empty_home = tmp_path / "empty-codex-home"
    empty_home.mkdir()

    # The auth check passes on the key alone, so the failure that follows
    # is the probe's, not the auth source's.
    with pytest.raises(CodexPreflightError) as caught:
        codex_auth_preflight(
            executor=executor,
            codex_binary="codex",
            environment={
                "PATH": str(binary_dir),
                "CODEX_HOME": str(empty_home),
                CODEX_AUTH_ENV_KEY: "sk-not-a-real-key",
            },
        )
    assert "no usable auth source" not in str(caught.value)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the Codex sandbox is macOS sandbox-exec only",
)
def test_a_nonzero_probe_exit_fails_the_preflight(tmp_path) -> None:
    binary_dir = tmp_path / "bin"
    _write_binary(
        binary_dir,
        script="#!/bin/sh\necho 'codex is not logged in' >&2\nexit 3\n",
    )
    executor = build_codex_executor(run_root=tmp_path / "runs")

    with pytest.raises(CodexPreflightError, match="preflight failed"):
        codex_auth_preflight(
            executor=executor,
            codex_binary="codex",
            environment={
                "PATH": str(binary_dir),
                "CODEX_HOME": str(_auth_home(tmp_path)),
            },
        )


def test_prepare_codex_run_propagates_a_failed_preflight(tmp_path) -> None:
    store_path = str(tmp_path / "preflight.sqlite")
    with open_sqlite(store_path) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        control = toy_codex_control(engine=engine)
        adapter = CodexAdapter(_NeverRunner(), store=store)
        runtime = build_runtime(
            store=store,
            engine=engine,
            adapter_registry=MappingAdapterRegistry(
                {CODEX_ADAPTER_KEY: adapter}
            ),
            effect_authority=EffectLeaseAuthority.memory(),
        )

        def _failing_preflight() -> None:
            raise CodexPreflightError("Codex preflight failed: exit 3")

        with pytest.raises(CodexPreflightError):
            prepare_codex_run(
                runtime,
                run_id=_PREFLIGHT_RUN_ID,
                control=control,
                experiment=build_toy_experiment(num_seeds=1),
                render_contract=toy_template_render_contract(),
                mutation_field=TOY_MUTATION_FIELD,
                preflight=_failing_preflight,
            )

        # No launch was bound, so no capacity or eval budget was
        # committed for this run.
        with pytest.raises(Exception):
            load_launch(store, _PREFLIGHT_RUN_ID)


def test_a_successful_preflight_lets_the_launch_bind(tmp_path) -> None:
    store_path = str(tmp_path / "preflight-ok.sqlite")
    with open_sqlite(store_path) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        control = toy_codex_control(engine=engine)
        adapter = CodexAdapter(_NeverRunner(), store=store)
        runtime = build_runtime(
            store=store,
            engine=engine,
            adapter_registry=MappingAdapterRegistry(
                {CODEX_ADAPTER_KEY: adapter}
            ),
            effect_authority=EffectLeaseAuthority.memory(),
        )
        calls: list[int] = []

        launch = prepare_codex_run(
            runtime,
            run_id=_PREFLIGHT_RUN_ID,
            control=control,
            experiment=build_toy_experiment(num_seeds=1),
            render_contract=toy_template_render_contract(),
            mutation_field=TOY_MUTATION_FIELD,
            preflight=lambda: calls.append(1),
        )

        assert calls == [1]
        assert launch.run.adapter_key == CODEX_ADAPTER_KEY
        assert len(launch.run.tool_configs) == 1
        assert load_launch(store, _PREFLIGHT_RUN_ID).run == launch.run


def test_prepare_codex_run_has_no_preflight_default(tmp_path) -> None:
    """A budgeted run cannot start without naming a session proof.

    An optional preflight is only as good as each caller remembering it,
    and the guarantee is that no capacity or eval budget is committed
    against an unproven session. So the parameter is required: omitting
    it is a TypeError, not a silently unchecked launch.
    """
    del tmp_path
    signature = inspect.signature(prepare_codex_run)
    parameter = signature.parameters["preflight"]

    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_probe_stages_the_default_codex_credentials(
    tmp_path, monkeypatch
) -> None:
    """An ordinary logged-in user must pass the preflight.

    ``_require_auth_source`` accepts the default ``~/.codex/auth.json``
    when ``CODEX_HOME`` is unset, but the probe runner is constructed with
    an explicit environment -- which used to resolve its auth source to
    ``None`` and stage nothing into the scratch ``CODEX_HOME``. The
    preflight then rejected a perfectly valid Codex login. The evidence is
    what reaches the probe's scratch home, so the fake CLI reports it
    instead of a real Codex ever being invoked.
    """
    from whetstone.optim.codex.runner import SubprocessCodexRunner

    fake_home = tmp_path / "home"
    codex_dir = fake_home / ".codex"
    codex_dir.mkdir(parents=True)
    (codex_dir / CODEX_AUTH_FILENAMES[0]).write_text(
        '{"token": "fake"}', encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("CODEX_HOME", raising=False)

    binary_dir = tmp_path / "bin"
    _write_binary(binary_dir, script="#!/bin/sh\nexit 0\n")

    runner = SubprocessCodexRunner(
        executor=object(),
        codex_binary="codex",
        environment={"PATH": str(binary_dir)},
    )

    # The default location is what the run's own auth staging will copy
    # from, so the probe sees exactly the credentials the real run would.
    assert runner.auth_source == codex_dir

    staged = tmp_path / "scratch-codex-home"
    staged.mkdir()
    runner.stage_auth(staged)
    assert (staged / CODEX_AUTH_FILENAMES[0]).is_file()


def test_an_explicit_codex_home_still_wins_over_the_default(
    tmp_path, monkeypatch
) -> None:
    """An explicitly configured CODEX_HOME is the auth source."""
    from whetstone.optim.codex.runner import SubprocessCodexRunner

    fake_home = tmp_path / "home"
    (fake_home / ".codex").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))

    configured = _auth_home(tmp_path)
    runner = SubprocessCodexRunner(
        executor=object(),
        codex_binary="codex",
        environment={"PATH": str(tmp_path), "CODEX_HOME": str(configured)},
    )

    assert runner.auth_source == configured


def test_the_probe_environment_carries_no_secrets(tmp_path, monkeypatch) -> None:
    """Staging credentials must not put them in the agent's environment.

    The Codex process is untrusted and network-capable. Its credentials
    reach it as files in its own scratch ``CODEX_HOME``; nothing about
    resolving the default location may widen the allowlisted environment.
    """
    from whetstone.optim.codex.runner import SubprocessCodexRunner

    fake_home = tmp_path / "home"
    (fake_home / ".codex").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("CODEX_HOME", raising=False)

    runner = SubprocessCodexRunner(
        executor=object(),
        codex_binary="codex",
        environment={"PATH": str(tmp_path), "OPENAI_API_KEY": "sk-fake"},
    )

    environment = runner.codex_process_environment()
    assert "CODEX_HOME" not in environment
    assert set(environment) <= {"PATH", "OPENAI_API_KEY"}
