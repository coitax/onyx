"""Host-header poisoning of request.url.path (Starlette 0.49.3 / CVE-2026-48710).

These tests send a raw HTTP/1.1 request through the Uvicorn h11 parser the
backend ships (httptools is not locked) into the real license-enforcement
middleware and the real MCP health shortcut. They fail while a malformed
Host header can make request.url.path differ from scope["path"].

A fix that reads request.url.path again, including once more between the
allowlist check and call_next, still fails: the request below is a real
Starlette request, so every read of request.url.path observes the poisoned
path while routing keeps using scope["path"].
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from datetime import timezone
from typing import Any
from unittest.mock import patch

import pytest
import uvicorn
from fastapi import FastAPI

from ee.onyx.server.license.models import LicenseMetadata
from ee.onyx.server.license.models import LicenseSource
from ee.onyx.server.license.models import PlanType
from ee.onyx.server.middleware.license_enforcement import (
    add_license_enforcement_middleware,
)
from onyx.server.settings.models import ApplicationStatus
from onyx.utils.logger import setup_logger

_HEALTH_BODY = {"status": "healthy", "service": "mcp_server"}
_HANDLED: list[str] = []


def _metadata(
    status: ApplicationStatus, seats: int, used_seats: int
) -> LicenseMetadata:
    now = datetime.now(timezone.utc)
    return LicenseMetadata(
        tenant_id="tenant",
        seats=seats,
        used_seats=used_seats,
        plan_type=PlanType.MONTHLY,
        issued_at=now,
        expires_at=now,
        status=status,
        source=LicenseSource.MANUAL_UPLOAD,
    )


def _license_app() -> FastAPI:
    app = FastAPI()

    @app.get("/chat")
    def chat() -> dict[str, str]:
        _HANDLED.append("chat")
        return {"resource": "chat"}

    @app.get("/analytics")
    def analytics() -> dict[str, str]:
        _HANDLED.append("analytics")
        return {"resource": "analytics"}

    @app.get("/auth/login")
    def auth_login() -> dict[str, str]:
        _HANDLED.append("auth")
        return {"resource": "auth"}

    @app.get("/api/chat")
    def api_chat() -> dict[str, str]:
        _HANDLED.append("api-chat")
        return {"resource": "api-chat"}

    @app.get("/api/analytics")
    def api_analytics() -> dict[str, str]:
        _HANDLED.append("api-analytics")
        return {"resource": "api-analytics"}

    add_license_enforcement_middleware(app, setup_logger())
    return app


def _start_server(app: FastAPI) -> tuple[uvicorn.Server, int]:
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="error",
        access_log=False,
        # The lockfile depends on h11 and does not install httptools, so
        # uvicorn's auto selector uses H11Protocol. Pin that parser here.
        http="h11",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    sockets = server.servers[0].sockets
    assert sockets is not None
    port = int(sockets[0].getsockname()[1])
    return server, port


def _stop_server(server: uvicorn.Server) -> None:
    server.should_exit = True
    deadline = time.monotonic() + 5
    while not server.started or server.servers:
        # main_loop exits after should_exit; give the thread a moment
        if time.monotonic() > deadline:
            break
        time.sleep(0.05)


def _raw_request(port: int, method: str, path: str, host: str) -> tuple[int, bytes]:
    message = (
        f"{method} {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        "Connection: close\r\n"
        "Accept: application/json, text/event-stream\r\n"
        "\r\n"
    ).encode("ascii")
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(message)
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    raw = b"".join(chunks)
    head, _, body = raw.partition(b"\r\n\r\n")
    status = int(head.split(b"\r\n", 1)[0].split(b" ")[1])
    return status, body


def _json_body(body: bytes) -> Any:
    return json.loads(body.decode("utf-8"))


@contextmanager
def _license(
    metadata: LicenseMetadata | None,
) -> Iterator[None]:
    @contextmanager
    def _empty_session() -> Iterator[None]:
        yield None

    with (
        patch(
            "ee.onyx.server.middleware.license_enforcement.LICENSE_ENFORCEMENT_ENABLED",
            True,
        ),
        patch(
            "ee.onyx.server.middleware.license_enforcement.get_cached_license_metadata",
            return_value=metadata,
        ),
        patch(
            "ee.onyx.server.middleware.license_enforcement.get_session_with_current_tenant",
            _empty_session,
        ),
        patch(
            "ee.onyx.server.middleware.license_enforcement.refresh_license_cache",
            return_value=metadata,
        ),
    ):
        yield


@pytest.fixture(scope="module")
def license_port() -> Iterator[int]:
    server, port = _start_server(_license_app())
    try:
        yield port
    finally:
        _stop_server(server)


@pytest.fixture(scope="module")
def mcp_port() -> Iterator[int]:
    from onyx.mcp_server.api import mcp_app

    server, port = _start_server(mcp_app)
    try:
        yield port
    finally:
        _stop_server(server)


def _gated() -> LicenseMetadata:
    return _metadata(ApplicationStatus.GATED_ACCESS, seats=10, used_seats=1)


def _over_seat() -> LicenseMetadata:
    return _metadata(ApplicationStatus.ACTIVE, seats=1, used_seats=5)


def _assert_blocked(status: int, body: bytes, error: str) -> None:
    assert _HANDLED == []
    assert status == 402
    payload = _json_body(body)
    assert payload["detail"]["error"] == error


def test_normal_host_gated_path_is_blocked(license_port: int) -> None:
    """A gated tenant is blocked on the real routed path."""
    _HANDLED.clear()
    with _license(_gated()):
        status, body = _raw_request(license_port, "GET", "/chat", "127.0.0.1")
    _assert_blocked(status, body, "license_expired")


def test_normal_host_allowlisted_path_reaches_that_route(license_port: int) -> None:
    """The allowlist still reaches only the route that was requested."""
    _HANDLED.clear()
    with _license(_gated()):
        status, body = _raw_request(license_port, "GET", "/auth/login", "127.0.0.1")
    assert status == 200
    assert _HANDLED == ["auth"]
    assert _json_body(body)["resource"] == "auth"


def test_malformed_host_does_not_skip_gated_license(
    license_port: int,
) -> None:
    """Host `127.0.0.1/auth?` must not turn /chat into an allowlisted path.

    Starlette 0.49.3 builds request.url as scheme://{Host}{scope path}, so
    this header makes request.url.path `/auth` while Uvicorn routes `/chat`.
    """
    _HANDLED.clear()
    with _license(_gated()):
        status, body = _raw_request(license_port, "GET", "/chat", "127.0.0.1/auth?")
    _assert_blocked(status, body, "license_expired")


def test_malformed_host_does_not_skip_seat_limit(license_port: int) -> None:
    """An over-seat tenant stays blocked when Host disguises the path."""
    _HANDLED.clear()
    with _license(_over_seat()):
        status, body = _raw_request(license_port, "GET", "/chat", "127.0.0.1/health?")
    _assert_blocked(status, body, "seat_limit_exceeded")


def test_api_prefix_strip_uses_the_routed_path(license_port: int) -> None:
    """Stripping a poisoned `/api` prefix must not allow the real route.

    Host `127.0.0.1/api/auth?` makes request.url.path `/api/auth`. The
    middleware then strips `/api` and would allow `/auth`, while the routed
    path remains `/api/analytics`.
    """
    _HANDLED.clear()
    with _license(_gated()):
        status, body = _raw_request(
            license_port, "GET", "/api/analytics", "127.0.0.1/api/auth?"
        )
    _assert_blocked(status, body, "license_expired")


def test_no_license_ee_path_ignores_community_disguise(license_port: int) -> None:
    """Without a license, /analytics stays EE-only if Host makes the path /chat.

    This is the second read of the path (the EE-only check). A fix that
    allowlists from scope['path'] but classifies EE routes from
    request.url.path again still fails.
    """
    _HANDLED.clear()
    with _license(None):
        status, body = _raw_request(
            license_port, "GET", "/analytics", "127.0.0.1/chat?"
        )
    _assert_blocked(status, body, "enterprise_license_required")


def test_no_license_community_path_stays_open(license_port: int) -> None:
    """No license still allows a community route when Host matches that path."""
    _HANDLED.clear()
    with _license(None):
        status, body = _raw_request(license_port, "GET", "/chat", "127.0.0.1")
    assert status == 200
    assert _HANDLED == ["chat"]
    assert _json_body(body)["resource"] == "chat"


def test_no_license_ee_path_blocked_with_normal_host(license_port: int) -> None:
    _HANDLED.clear()
    with _license(None):
        status, body = _raw_request(license_port, "GET", "/analytics", "127.0.0.1")
    _assert_blocked(status, body, "enterprise_license_required")


def test_mcp_health_shortcut_matches_only_the_routed_path(mcp_port: int) -> None:
    """GET /health is the public health response."""
    status, body = _raw_request(mcp_port, "GET", "/health", "127.0.0.1")
    assert status == 200
    assert _json_body(body) == _HEALTH_BODY


def test_malformed_host_does_not_skip_mcp_auth(mcp_port: int) -> None:
    """Host `127.0.0.1/health?` must not answer GET / with the health body.

    The health shortcut returns before FastMCP auth. The routed path is `/`,
    which requires a bearer token, so the response stays 401.
    """
    status, body = _raw_request(mcp_port, "GET", "/", "127.0.0.1/health?")
    assert status == 401
    assert _json_body(body) != _HEALTH_BODY
    assert b"invalid_token" in body


def test_malformed_host_does_not_healthcheck_other_mcp_paths(mcp_port: int) -> None:
    """A non-health route keeps its own status when Host looks like /health."""
    status, body = _raw_request(mcp_port, "GET", "/mcp", "127.0.0.1/health?")
    assert status == 404
    assert b'"service":"mcp_server"' not in body
