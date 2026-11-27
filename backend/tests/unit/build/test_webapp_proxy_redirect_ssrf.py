"""Craft webapp proxy must not fetch redirect targets chosen by the sandbox.

The session owner controls the sandbox HTTP server (the Craft agent can write
the Next.js app and run other commands). ``get_webapp`` proxies that server
with an HTTP client that follows redirects and copies the final body back to
the caller, including client-supplied headers on the outbound request.

These tests call ``get_webapp`` against local listeners. They fail while the
proxy follows a sandbox ``Location`` to another origin or forwards headers
outside the allowlist below. A fix should:

- not follow redirects server-side
- return a relative ``Location`` under the webapp proxy prefix only when the
  sandbox ``Location`` stays on that app (relative path, or an absolute URL
  whose origin is the sandbox origin)
- drop every other ``Location`` instead of passing an absolute off-origin URL
  through
- forward only the allowlisted request headers to the sandbox
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast
from urllib.parse import urlparse
from uuid import UUID

import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse
from starlette.responses import Response
from sqlalchemy.orm import Session

from onyx.db.enums import SharingScope
from onyx.db.models import User
from onyx.server.features.build.api import api
from onyx.server.features.build.sandbox.kubernetes.kubernetes_sandbox_manager import (
    KubernetesSandboxManager,
)
from onyx.server.features.build.sandbox.local.local_sandbox_manager import (
    LocalSandboxManager,
)

SESSION_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
USER_ID = UUID("11111111-2222-3333-4444-555555555555")
SANDBOX_ID = UUID("99999999-8888-7777-6666-555555555555")
WEBAPP_BASE = f"/api/build/sessions/{SESSION_ID}/webapp"

# Distinct from any sandbox page body so a followed redirect is observable.
INTERNAL_SECRET = b'{"indexed":"craft-proxy-internal-sentinel"}'
SANDBOX_PAGE = b"sandbox-page-body"

# Headers a browser needs for ordinary asset fetches. Everything else the
# caller supplies, including session cookies and metadata probes, stays here.
_ALLOWED_REQUEST_HEADERS = frozenset(
    {
        "accept",
        "accept-language",
        "accept-encoding",
        "user-agent",
        "range",
        "if-match",
        "if-none-match",
        "if-modified-since",
        "if-unmodified-since",
        "if-range",
        "cache-control",
    }
)
_FORBIDDEN_CLIENT_HEADERS = (
    "cookie",
    "authorization",
    "metadata-flavor",
    "x-attacker-header",
)


@dataclass
class Route:
    status: int
    headers: dict[str, str]
    body: bytes


@dataclass
class RecordedRequest:
    path: str
    headers: dict[str, str]


class _RecordingServer(ThreadingHTTPServer):
    recorded: list[RecordedRequest]
    routes: dict[str, Route]


class _QuietHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        server = cast(_RecordingServer, self.server)
        server.recorded.append(
            RecordedRequest(
                self.path,
                {key.lower(): value for key, value in self.headers.items()},
            )
        )
        path_only = self.path.split("?", 1)[0]
        route = server.routes.get(path_only)
        if route is None:
            body = b"missing"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(route.status)
        for key, value in route.headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(route.body)))
        self.end_headers()
        if route.body:
            self.wfile.write(route.body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def _redirect(status: int, location: str) -> Route:
    return Route(status, {"Location": location, "Content-Type": "text/plain"}, b"")


def _page(body: bytes, content_type: str) -> Route:
    return Route(200, {"Content-Type": content_type}, body)


@contextmanager
def _serve(routes: dict[str, Route]) -> Iterator[_RecordingServer]:
    server = _RecordingServer(("127.0.0.1", 0), _QuietHandler)
    server.recorded = []
    server.routes = routes
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _port(server: _RecordingServer) -> int:
    address = server.server_address
    assert isinstance(address, tuple)
    return int(address[1])


def _client_headers() -> dict[str, str]:
    return {
        "accept": "application/json",
        "cookie": "onyx_session=placeholder",
        "authorization": "Bearer placeholder",
        "metadata-flavor": "Google",
        "x-attacker-header": "probe",
    }


def _response_body(response: Response) -> bytes:
    if isinstance(response, StreamingResponse):

        async def _collect() -> bytes:
            parts: list[bytes] = []
            async for chunk in response.body_iterator:
                if isinstance(chunk, bytes):
                    parts.append(chunk)
                elif isinstance(chunk, memoryview):
                    parts.append(chunk.tobytes())
                else:
                    parts.append(str(chunk).encode())
            return b"".join(parts)

        return asyncio.run(_collect())
    body = response.body
    if isinstance(body, memoryview):
        return body.tobytes()
    return bytes(body)


def _assert_not_exposed(response: Response, internal: _RecordingServer) -> None:
    """The API server must not fetch the redirect target or return its body."""
    assert internal.recorded == []
    assert INTERNAL_SECRET not in _response_body(response)
    location = response.headers.get("location")
    if location is not None:
        lowered = location.lower()
        assert not lowered.startswith("http://")
        assert not lowered.startswith("https://")
        assert not location.startswith("//")
        assert "127.0.0.1" not in location
        assert location.startswith(f"{WEBAPP_BASE}/") or location == WEBAPP_BASE


def _assert_request_headers_allowlisted(recorded: list[RecordedRequest]) -> None:
    assert recorded, "expected the proxy to contact the sandbox"
    for request in recorded:
        for name in _FORBIDDEN_CLIENT_HEADERS:
            assert name not in request.headers
        for name in request.headers:
            if name in {"host", "connection", "content-length"}:
                continue
            assert name in _ALLOWED_REQUEST_HEADERS
        assert request.headers.get("accept") == "application/json"


class _FakeDb:
    def __init__(self, session: SimpleNamespace) -> None:
        self._session = session

    def get(self, _model: object, _identity: object) -> SimpleNamespace:
        return self._session


def _call_webapp(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sandbox_port: int,
    path: str,
    query: str,
    sharing: SharingScope,
    user_id: UUID | None,
    manager: object,
) -> Response:
    session = SimpleNamespace(
        id=SESSION_ID,
        user_id=USER_ID,
        nextjs_port=sandbox_port,
        sharing_scope=sharing,
    )
    monkeypatch.setattr(
        api,
        "get_sandbox_by_user_id",
        lambda *_args, **_kwargs: SimpleNamespace(id=SANDBOX_ID),
    )
    monkeypatch.setattr(api, "get_sandbox_manager", lambda: manager)
    request = cast(
        Request,
        SimpleNamespace(headers=_client_headers(), query_params=query),
    )
    user = None if user_id is None else cast(User, SimpleNamespace(id=user_id))
    return api.get_webapp(
        SESSION_ID,
        request,
        path,
        user,
        cast(Session, _FakeDb(session)),
    )


def _local_manager() -> LocalSandboxManager:
    return object.__new__(LocalSandboxManager)


def _kubernetes_manager() -> KubernetesSandboxManager:
    manager = object.__new__(KubernetesSandboxManager)
    manager._namespace = "onyx-sandboxes"
    return manager


def _write_hosts(contents: str) -> None:
    hosts = Path("/etc/hosts")
    if os.access(hosts, os.W_OK):
        hosts.write_text(contents)
        return
    subprocess.run(
        ["sudo", "-n", "tee", str(hosts)],
        input=contents,
        text=True,
        check=True,
        capture_output=True,
    )


@contextmanager
def _map_hostname(hostname: str) -> Iterator[None]:
    """Point a cluster-DNS sandbox hostname at this machine's loopback.

    ``KubernetesSandboxManager.get_webapp_url`` returns that hostname. The
    hosts file is the resolver the HTTP client uses; the entry is removed
    when the test ends.
    """
    hosts = Path("/etc/hosts")
    original = hosts.read_text()
    if hostname not in original:
        _write_hosts(original + f"127.0.0.1 {hostname}\n")
    try:
        yield
    finally:
        if hostname not in original:
            _write_hosts(original)


@pytest.fixture
def internal() -> Iterator[_RecordingServer]:
    with _serve({"/secret": _page(INTERNAL_SECRET, "application/json")}) as server:
        yield server


def _internal_url(internal: _RecordingServer, path: str = "/secret") -> str:
    return f"http://127.0.0.1:{_port(internal)}{path}"


def _port_host(port: int) -> str:
    return f"127.0.0.1:{port}"


class TestWebappProxyDoesNotFollowSandboxRedirects:
    def test_owner_subpath_does_not_fetch_absolute_redirect_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
        internal: _RecordingServer,
    ) -> None:
        target = _internal_url(internal)
        with _serve({"/search": _redirect(302, target)}) as sandbox:
            response = _call_webapp(
                monkeypatch,
                sandbox_port=_port(sandbox),
                path="search",
                query="yql=select%20%2A",
                sharing=SharingScope.PRIVATE,
                user_id=USER_ID,
                manager=_local_manager(),
            )

        assert sandbox.recorded[0].path.startswith("/search?yql=")
        _assert_not_exposed(response, internal)
        _assert_request_headers_allowlisted(sandbox.recorded)

    @pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
    def test_redirect_statuses_are_not_followed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        internal: _RecordingServer,
        status: int,
    ) -> None:
        with _serve({"/go": _redirect(status, _internal_url(internal))}) as sandbox:
            response = _call_webapp(
                monkeypatch,
                sandbox_port=_port(sandbox),
                path="go",
                query="",
                sharing=SharingScope.PRIVATE,
                user_id=USER_ID,
                manager=_local_manager(),
            )

        _assert_not_exposed(response, internal)

    def test_protocol_relative_location_is_not_followed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        internal: _RecordingServer,
    ) -> None:
        port = _port(internal)
        with _serve({"/go": _redirect(302, f"//127.0.0.1:{port}/secret")}) as sandbox:
            response = _call_webapp(
                monkeypatch,
                sandbox_port=_port(sandbox),
                path="go",
                query="",
                sharing=SharingScope.PRIVATE,
                user_id=USER_ID,
                manager=_local_manager(),
            )

        _assert_not_exposed(response, internal)

    def test_userinfo_location_is_not_followed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        internal: _RecordingServer,
    ) -> None:
        port = _port(internal)
        location = f"http://sandbox.invalid@{_port_host(port)}/secret"
        with _serve({"/go": _redirect(302, location)}) as sandbox:
            response = _call_webapp(
                monkeypatch,
                sandbox_port=_port(sandbox),
                path="go",
                query="",
                sharing=SharingScope.PRIVATE,
                user_id=USER_ID,
                manager=_local_manager(),
            )

        _assert_not_exposed(response, internal)

    def test_same_hostname_different_port_is_not_followed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        internal: _RecordingServer,
    ) -> None:
        """A check that only compares hostnames still has to fail.

        The sandbox URL is ``localhost`` on its own port. A Location that
        keeps that hostname and switches ports is a different origin.
        """
        location = f"http://localhost:{_port(internal)}/secret"
        with _serve({"/go": _redirect(302, location)}) as sandbox:
            response = _call_webapp(
                monkeypatch,
                sandbox_port=_port(sandbox),
                path="go",
                query="",
                sharing=SharingScope.PRIVATE,
                user_id=USER_ID,
                manager=_local_manager(),
            )

        _assert_not_exposed(response, internal)

    def test_relative_hop_then_absolute_redirect_is_not_followed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        internal: _RecordingServer,
    ) -> None:
        """Checking only the first Location is not enough.

        ``/start`` redirects to a same-app path, and that path redirects to
        the internal listener. Following after a same-app check still fetches
        the internal URL.
        """
        with _serve(
            {
                "/start": _redirect(302, "/hop"),
                "/hop": _redirect(302, _internal_url(internal)),
            }
        ) as sandbox:
            response = _call_webapp(
                monkeypatch,
                sandbox_port=_port(sandbox),
                path="start",
                query="",
                sharing=SharingScope.PRIVATE,
                user_id=USER_ID,
                manager=_local_manager(),
            )

        _assert_not_exposed(response, internal)
        assert [hit.path.split("?", 1)[0] for hit in sandbox.recorded] == ["/start"]

    def test_relative_redirect_is_passed_through_under_the_webapp_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        internal: _RecordingServer,
    ) -> None:
        with _serve(
            {
                "/page": _redirect(302, "/ok"),
                "/ok": _page(SANDBOX_PAGE, "text/plain"),
            }
        ) as sandbox:
            response = _call_webapp(
                monkeypatch,
                sandbox_port=_port(sandbox),
                path="page",
                query="",
                sharing=SharingScope.PRIVATE,
                user_id=USER_ID,
                manager=_local_manager(),
            )

        assert response.status_code == 302
        assert response.headers["location"] == f"{WEBAPP_BASE}/ok"
        assert SANDBOX_PAGE not in _response_body(response)
        assert [hit.path for hit in sandbox.recorded] == ["/page"]
        assert internal.recorded == []

    def test_same_origin_absolute_location_is_rewritten_not_fetched(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _serve(
            {
                "/page": Route(302, {}, b""),
                "/ok": _page(SANDBOX_PAGE, "text/html"),
            }
        ) as sandbox:
            port = _port(sandbox)
            sandbox.routes["/page"] = _redirect(302, f"http://localhost:{port}/ok")
            response = _call_webapp(
                monkeypatch,
                sandbox_port=port,
                path="page",
                query="",
                sharing=SharingScope.PRIVATE,
                user_id=USER_ID,
                manager=_local_manager(),
            )

        assert response.status_code == 302
        assert response.headers["location"] == f"{WEBAPP_BASE}/ok"
        assert SANDBOX_PAGE not in _response_body(response)
        assert [hit.path for hit in sandbox.recorded] == ["/page"]

    def test_html_body_of_redirect_target_is_not_returned(
        self,
        monkeypatch: pytest.MonkeyPatch,
        internal: _RecordingServer,
    ) -> None:
        internal.routes["/secret"] = _page(
            b"<html><body>" + INTERNAL_SECRET + b"</body></html>",
            "text/html",
        )
        with _serve({"/go": _redirect(302, _internal_url(internal))}) as sandbox:
            response = _call_webapp(
                monkeypatch,
                sandbox_port=_port(sandbox),
                path="go",
                query="",
                sharing=SharingScope.PRIVATE,
                user_id=USER_ID,
                manager=_local_manager(),
            )

        _assert_not_exposed(response, internal)

    def test_public_global_request_does_not_follow_redirect(
        self,
        monkeypatch: pytest.MonkeyPatch,
        internal: _RecordingServer,
    ) -> None:
        with _serve({"/": _redirect(302, _internal_url(internal))}) as sandbox:
            response = _call_webapp(
                monkeypatch,
                sandbox_port=_port(sandbox),
                path="",
                query="",
                sharing=SharingScope.PUBLIC_GLOBAL,
                user_id=None,
                manager=_local_manager(),
            )

        _assert_not_exposed(response, internal)

    def test_direct_sandbox_response_still_proxies_and_drops_extra_headers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _serve({"/app": _page(SANDBOX_PAGE, "text/plain")}) as sandbox:
            response = _call_webapp(
                monkeypatch,
                sandbox_port=_port(sandbox),
                path="app",
                query="",
                sharing=SharingScope.PRIVATE,
                user_id=USER_ID,
                manager=_local_manager(),
            )

        assert response.status_code == 200
        assert _response_body(response) == SANDBOX_PAGE
        _assert_request_headers_allowlisted(sandbox.recorded)

    def test_kubernetes_sandbox_url_redirect_is_not_followed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        internal: _RecordingServer,
    ) -> None:
        manager = _kubernetes_manager()
        with _serve({"/go": _redirect(302, _internal_url(internal))}) as sandbox:
            port = _port(sandbox)
            sandbox_url = manager.get_webapp_url(SANDBOX_ID, port)
            hostname = urlparse(sandbox_url).hostname
            assert hostname is not None
            with _map_hostname(hostname):
                response = _call_webapp(
                    monkeypatch,
                    sandbox_port=port,
                    path="go",
                    query="",
                    sharing=SharingScope.PRIVATE,
                    user_id=USER_ID,
                    manager=manager,
                )

        assert sandbox.recorded, "proxy did not request the kubernetes sandbox URL"
        assert urlparse(sandbox_url).port == port
        _assert_not_exposed(response, internal)
        _assert_request_headers_allowlisted(sandbox.recorded)
