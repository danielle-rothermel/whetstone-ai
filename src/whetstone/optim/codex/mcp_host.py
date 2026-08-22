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
from contextlib import closing
from dataclasses import dataclass
from hmac import compare_digest

from whetstone.optim.codex.mcp_bridge import EvaluateCandidateServer

#: The path the streamable-HTTP transport is mounted at, and the header
#: the agent must present. Both cross into the Codex CLI's configuration,
#: so they are pinned by a golden test.
CODEX_MCP_HTTP_PATH = "/mcp"
CODEX_MCP_AUTH_HEADER = "authorization"
CODEX_MCP_AUTH_SCHEME = "Bearer"
#: How long the host waits for uvicorn to report a bound port before it
#: gives up. A server that has not bound by then is not going to.
CODEX_MCP_STARTUP_SECONDS = 30.0


class CodexMcpHostError(RuntimeError):
    """The evaluation endpoint could not be brought up or shut down."""


@dataclass(frozen=True, slots=True)
class CodexMcpEndpoint:
    """Where the agent reaches the evaluation server, and with what."""

    url: str
    auth_token: str


def reserve_loopback_port() -> int:
    """Take a loopback port the server will bind.

    Binding with ``SO_REUSEADDR`` and closing immediately leaves the port
    free for uvicorn while making a collision with another local listener
    vanishingly unlikely.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


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
        self._port = reserve_loopback_port() if port is None else port
        self._startup_seconds = startup_seconds
        self._thread: threading.Thread | None = None
        self._uvicorn = None
        self._ready = threading.Event()
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
        self._uvicorn = uvicorn.Server(config)

        def _serve() -> None:
            try:
                asyncio.run(self._uvicorn.serve())
            except BaseException as exc:  # noqa: BLE001 - reported to caller
                self._failure = exc
            finally:
                self._ready.set()

        self._thread = threading.Thread(
            target=_serve, name="whetstone-codex-mcp", daemon=True
        )
        self._thread.start()
        self._await_started()
        return self.endpoint

    def _await_started(self) -> None:
        """Block until the socket is accepting, or fail loudly.

        Readiness is the listener answering, not a delay: a connect that
        succeeds is the state that makes the endpoint usable.
        """
        deadline = self._startup_seconds
        step = 0.01
        waited = 0.0
        while waited < deadline:
            if self._failure is not None:
                self.__exit__(None, None, None)
                raise CodexMcpHostError(
                    "the Codex MCP evaluation endpoint failed to start"
                ) from self._failure
            with closing(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ) as probe:
                probe.settimeout(step)
                if probe.connect_ex(("127.0.0.1", self._port)) == 0:
                    return
            waited += step
        self.__exit__(None, None, None)
        raise CodexMcpHostError(
            "the Codex MCP evaluation endpoint did not bind "
            f"127.0.0.1:{self._port} within {deadline} seconds"
        )

    def __exit__(self, *_exc_info: object) -> None:
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._startup_seconds)
        self._thread = None
        self._uvicorn = None


__all__ = [
    "CODEX_MCP_AUTH_HEADER",
    "CODEX_MCP_AUTH_SCHEME",
    "CODEX_MCP_HTTP_PATH",
    "CODEX_MCP_STARTUP_SECONDS",
    "CodexMcpEndpoint",
    "CodexMcpHost",
    "CodexMcpHostError",
    "bearer_auth_middleware",
    "reserve_loopback_port",
]
