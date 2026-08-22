"""Run the MCP evaluation server outside the Codex sandbox.

The server is the only process that may write the whetstone store: it
persists the Tool Results the adapter later resolves, and the admission
capacity rows it debits are the per-run cap on paid evaluations. SQLite
needs real write access to that file, so anything sharing the server's
sandbox profile can rewrite the ledger underneath the adapter.

Running the server as a child of the untrusted Codex process gave it
exactly that profile. So whetstone starts the server itself, before the
sandbox exists, and hands the agent only a loopback URL plus a bearer
token. The agent's profile then grants no write access to the store at
all, and the boundary between the agent and the ledger is a process it
cannot reach into.

The endpoint is bound to loopback and authenticated, because the Codex
containment profile permits network: reachability alone is not a
boundary, so every request must carry the run's own lease token.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from dataclasses import dataclass

from hmac import compare_digest

from whetstone.optim.codex.mcp_bridge import EvaluateCandidateServer

#: The path the streamable-HTTP transport is mounted at, and the header
#: the agent must present. Both cross into the Codex CLI's configuration,
#: so they are pinned by a golden test.
CODEX_MCP_HTTP_PATH = "/mcp"
CODEX_MCP_AUTH_HEADER = "authorization"
CODEX_MCP_AUTH_SCHEME = "Bearer"
#: How long the host waits for uvicorn to report the endpoint started
#: before it gives up. This is real elapsed time against a monotonic
#: deadline: a server that has not started by then is not going to.
CODEX_MCP_STARTUP_SECONDS = 30.0


class CodexMcpHostError(RuntimeError):
    """The evaluation endpoint could not be brought up or shut down."""

    def __init__(self, message: str, *, log: str = "") -> None:
        super().__init__(f"{message}: {log}" if log else message)
        self.log = log


@dataclass(frozen=True, slots=True)
class CodexMcpEndpoint:
    """Where the agent reaches the evaluation server, and with what."""

    url: str
    auth_token: str


@dataclass(frozen=True, slots=True)
class ReservedLoopbackPort:
    """A loopback port held open, to be served on directly.

    The socket is already bound and listening. Handing it to uvicorn
    rather than closing it and letting uvicorn re-bind means the port is
    never unowned, so no other local process can win it and become the
    endpoint the agent authenticates to.
    """

    socket: socket.socket
    port: int


def reserve_loopback_port() -> ReservedLoopbackPort:
    """Take a loopback port and keep holding it.

    The returned socket is listening. Its owner is responsible for
    closing it -- :class:`CodexMcpHost` does so on exit, after uvicorn
    has released it.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
    except BaseException:
        listener.close()
        raise
    return ReservedLoopbackPort(
        socket=listener, port=int(listener.getsockname()[1])
    )


def bearer_auth_middleware(expected_token: str):
    """Reject any request that does not carry this run's lease token.

    The Codex sandbox profile allows network, so any process in the
    agent's tree can reach a loopback port. The token is what makes the
    endpoint this run's rather than the machine's.
    """
    from starlette.responses import PlainTextResponse

    expected_header = f"{CODEX_MCP_AUTH_SCHEME} {expected_token}"

    async def middleware(request, call_next):
        presented = request.headers.get(CODEX_MCP_AUTH_HEADER, "")
        if not compare_digest(presented, expected_header):
            return PlainTextResponse("unauthorized", status_code=401)
        return await call_next(request)

    return middleware


class CodexMcpHost:
    """A running evaluation endpoint, owned by whetstone.

    Use it as a context manager: the endpoint exists for exactly the
    lifetime of the block, and the server thread is joined on exit so no
    listener outlives the Step that owns it.

    Readiness is uvicorn's own started signal, never a connect probe. A
    probe proves only that *something* accepts on the port; taking that
    as readiness would hand the agent -- and this run's bearer token --
    to whatever process happened to own it.
    """

    def __init__(
        self,
        server: EvaluateCandidateServer,
        *,
        auth_token: str,
        port: int | None = None,
        startup_seconds: float = CODEX_MCP_STARTUP_SECONDS,
    ) -> None:
        self._server = server
        self._auth_token = auth_token
        self._reserved = reserve_loopback_port() if port is None else None
        self._port = self._reserved.port if self._reserved else int(port or 0)
        self._startup_seconds = startup_seconds
        self._thread: threading.Thread | None = None
        self._uvicorn = None
        self._settled = threading.Event()
        self._failure: BaseException | None = None

    @property
    def endpoint(self) -> CodexMcpEndpoint:
        return CodexMcpEndpoint(
            url=f"http://127.0.0.1:{self._port}{CODEX_MCP_HTTP_PATH}",
            auth_token=self._auth_token,
        )

    def __enter__(self) -> CodexMcpEndpoint:
        import uvicorn
        from starlette.middleware.base import BaseHTTPMiddleware

        app = self._server.streamable_http_app(
            streamable_http_path=CODEX_MCP_HTTP_PATH,
        )
        app.add_middleware(
            BaseHTTPMiddleware,
            dispatch=bearer_auth_middleware(self._auth_token),
        )
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self._port,
            log_level="warning",
            lifespan="on",
        )
        server = uvicorn.Server(config)
        self._uvicorn = server
        # uvicorn logs bind and lifespan failures rather than raising
        # them, and a failed startup exits the thread through SystemExit
        # carrying no detail. Capturing its log is what lets the host
        # say *why* the endpoint is unavailable.
        log = _CapturedServerLog()
        # Serve on the socket already reserved for this host. uvicorn
        # binds nothing of its own, so there is no window between the
        # reservation and the listener in which another process could
        # take the port.
        sockets = (
            [self._reserved.socket] if self._reserved is not None else None
        )

        def _serve() -> None:
            try:
                with log.capturing():
                    asyncio.run(server.serve(sockets=sockets))
            except BaseException as exc:  # noqa: BLE001 - reported to caller
                self._failure = exc
            finally:
                self._settled.set()

        self._thread = threading.Thread(
            target=_serve, name="whetstone-codex-mcp", daemon=True
        )
        self._thread.start()
        self._await_started(server, log)
        return self.endpoint

    def _await_started(self, server, log: _CapturedServerLog) -> None:
        """Block until uvicorn reports started, or fail loudly.

        Readiness is ``Server.started``, which uvicorn sets only after
        its own listeners are up -- so it is proof that *this* server
        owns the endpoint. The wait is against a monotonic deadline, so
        the budget is the real time it names.
        """
        deadline = time.monotonic() + self._startup_seconds
        while True:
            if self._failure is not None or self._settled.is_set():
                self.__exit__(None, None, None)
                raise CodexMcpHostError(
                    "the Codex MCP evaluation endpoint failed to start",
                    log=log.text(),
                ) from self._failure
            if server.started:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            # The serve thread sets ``_settled`` on any terminal exit, so
            # waiting on it means a failed startup wakes the host at once
            # rather than at the deadline. ``started`` has no event of its
            # own, hence the bounded poll interval.
            self._settled.wait(timeout=min(remaining, 0.01))
        self.__exit__(None, None, None)
        raise CodexMcpHostError(
            "the Codex MCP evaluation endpoint did not start on "
            f"127.0.0.1:{self._port} within {self._startup_seconds} seconds",
            log=log.text(),
        )

    def __exit__(self, *_exc_info: object) -> None:
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._startup_seconds)
        self._thread = None
        self._uvicorn = None
        reserved = self._reserved
        if reserved is not None:
            self._reserved = None
            reserved.socket.close()
        self._close_store_session()

    def _close_store_session(self) -> None:
        """Release the store session this Step's server opened.

        ``persistent_sqlite`` sessions are process-lifetime and keyed by
        path: each one owns an event loop, a thread, and an open SQLite
        backend. The stdio server that preceded this host released them
        by dying; the in-process host outlives its Steps, so it closes
        the session it owns instead.
        """
        sqlite_path = getattr(self._server, "sqlite_path", None)
        if not sqlite_path:
            return
        from dr_store.sync import close_persistent

        close_persistent(sqlite_path)


class _CapturedServerLog:
    """uvicorn's own error log for one server run.

    uvicorn reports a failed bind or a failed lifespan by logging and
    exiting, so the raised ``SystemExit`` says nothing useful. This keeps
    the log line the operator actually needs on the host error.
    """

    def __init__(self) -> None:
        self._records: list[str] = []
        self._lock = threading.Lock()

    def capturing(self):
        import contextlib
        import logging

        captured = self

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                with captured._lock:
                    captured._records.append(record.getMessage())

        @contextlib.contextmanager
        def _scope():
            handler = _Handler(level=logging.ERROR)
            logger = logging.getLogger("uvicorn.error")
            logger.addHandler(handler)
            try:
                yield
            finally:
                logger.removeHandler(handler)

        return _scope()

    def text(self) -> str:
        with self._lock:
            return " ".join(self._records)


__all__ = [
    "CODEX_MCP_AUTH_HEADER",
    "CODEX_MCP_AUTH_SCHEME",
    "CODEX_MCP_HTTP_PATH",
    "CODEX_MCP_STARTUP_SECONDS",
    "CodexMcpEndpoint",
    "CodexMcpHost",
    "CodexMcpHostError",
    "ReservedLoopbackPort",
    "bearer_auth_middleware",
    "reserve_loopback_port",
]
