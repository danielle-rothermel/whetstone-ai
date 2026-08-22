"""How the evaluation endpoint proves it is up, and whose it is.

The host is whetstone's only boundary against the agent talking to a
listener whetstone does not own: the run's bearer token is presented to
whatever answers the reserved port. So "ready" must mean *this* server
accepted the port, not that something did, and the startup budget must
be real time rather than a spin count.

These run on every platform -- no sandbox and no Codex binary is
involved, only uvicorn and a trivial ASGI app.
"""

from __future__ import annotations

import socket
import threading

import pytest

from whetstone.optim.codex.mcp_host import (
    CodexMcpHost,
    CodexMcpHostError,
    reserve_loopback_port,
)


class _TrivialServer:
    """The smallest thing ``CodexMcpHost`` will host.

    ``CodexMcpHost`` needs only ``streamable_http_app``; the evaluation
    bridge's behavior is covered by its own suites, and pulling a real
    one in here would make a lifecycle test depend on a store.
    """

    def __init__(self, *, on_startup=None) -> None:
        self._on_startup = on_startup

    def streamable_http_app(self, *, streamable_http_path: str):
        import contextlib

        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def _ok(_request):
            return PlainTextResponse("ok")

        on_startup = self._on_startup

        @contextlib.asynccontextmanager
        async def _lifespan(_app):
            if on_startup is not None:
                on_startup()
            yield

        return Starlette(
            routes=[Route(streamable_http_path, _ok)],
            lifespan=_lifespan,
        )


@pytest.fixture
def squatted_port():
    """A port already owned by a foreign listener."""
    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    squatter.bind(("127.0.0.1", 0))
    squatter.listen(8)
    port = int(squatter.getsockname()[1])
    try:
        yield port
    finally:
        squatter.close()


def test_a_foreign_listener_on_the_port_fails_the_host_loudly(
    squatted_port,
) -> None:
    """A connect that succeeds is not proof the server is whetstone's.

    If the host treats "something accepted" as readiness, it hands the
    agent a URL pointing at a process whetstone does not control, and
    the run's bearer token is presented to it. The bind failure uvicorn
    already recorded must surface instead.
    """
    host = CodexMcpHost(
        _TrivialServer(),
        auth_token="run-token",
        port=squatted_port,
        startup_seconds=5.0,
    )

    with pytest.raises(CodexMcpHostError) as excinfo:
        with host:
            pytest.fail("the host must not hand out a squatted endpoint")

    # The failure names the bind, not a generic timeout: the operator
    # needs to know the port was taken, not that the server was slow.
    rendered = f"{excinfo.value}{excinfo.value.__cause__}"
    assert "address already in use" in rendered.lower()


def test_the_startup_budget_is_real_time_not_a_spin_count() -> None:
    """A slow-but-healthy server must still be waited for.

    A loop that increments a counter without sleeping burns its whole
    nominal budget in a fraction of a second, so a server that binds
    slowly -- a cold import, a loaded CI box -- is failed spuriously.
    The gate here is a state signal, not a delay: the host must not
    return until the test releases startup.
    """
    released = threading.Event()
    host = CodexMcpHost(
        _TrivialServer(on_startup=lambda: released.wait(timeout=30.0)),
        auth_token="run-token",
        startup_seconds=30.0,
    )

    def _release_once_the_host_is_waiting() -> None:
        # The host must still be blocked in __enter__ when this fires;
        # a spin-count budget would already have given up.
        released.set()

    releaser = threading.Timer(0.75, _release_once_the_host_is_waiting)
    releaser.start()
    try:
        with host as endpoint:
            assert released.is_set()
            assert endpoint.url.endswith("/mcp")
    finally:
        releaser.cancel()


def test_a_server_that_never_starts_fails_at_the_deadline() -> None:
    """The deadline still fires, and it says so.

    Evidence is the raised error, not elapsed time: a gate that never
    opens must produce a host error rather than hanging forever.
    """
    never = threading.Event()
    host = CodexMcpHost(
        _TrivialServer(on_startup=lambda: never.wait(timeout=10.0)),
        auth_token="run-token",
        startup_seconds=0.2,
    )

    with pytest.raises(CodexMcpHostError) as excinfo:
        with host:
            pytest.fail("the host must not report a server that never started")

    assert "did not start" in str(excinfo.value)


def test_the_per_step_store_session_is_closed_when_the_host_exits(
    tmp_path,
) -> None:
    """whetstone is now the long-lived process, so it must release the store.

    The deleted stdio server died with its subprocess, which closed the
    persistent session for free. The in-process host outlives every Step
    it runs, so a session left open keeps an event-loop thread and an
    open SQLite backend for the life of the CLI. The evidence is the
    session's own state -- a closed session refuses further use -- not a
    thread count or a delay.
    """
    from dr_store.sync import SyncSessionClosedError, persistent_sqlite

    sqlite_path = str(tmp_path / "host-session.sqlite")
    store = persistent_sqlite(sqlite_path)
    server = _TrivialServer()
    server.sqlite_path = sqlite_path

    with CodexMcpHost(
        server, auth_token="run-token", startup_seconds=5.0
    ) as endpoint:
        assert endpoint.auth_token == "run-token"

    with pytest.raises(SyncSessionClosedError):
        store.put("whetstone.host_session_probe", {"probe": True})


def test_the_reserved_port_is_never_unowned() -> None:
    """The reservation is handed to uvicorn, not closed and re-bound.

    Reserving by binding and closing leaves a real window in which any
    local process can take the port; the host then authenticates the
    agent to the winner. Holding the listening socket and serving on it
    removes the window instead of arguing it is small.
    """
    reserved = reserve_loopback_port()

    # The reservation is a live listening socket, so its port cannot be
    # bound by anyone else while the host owns it.
    assert reserved.socket.fileno() >= 0
    other = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError):
            other.bind(("127.0.0.1", reserved.port))
            other.listen(1)
    finally:
        other.close()
        reserved.socket.close()
