"""Security spec for the Craft webapp proxy.

These tests fail while the proxy copies a visitor's request headers onto the
sandbox connection. They pass once that connection receives only an allowlist
of safe headers.

Both GET routes (`/build/sessions/{id}/webapp` and
`/build/sessions/{id}/webapp/{path}`) share `_proxy_request`. POST and other
methods are not proxied, so they are not covered.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from typing import cast
from uuid import UUID
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from onyx.auth.users import optional_user
from onyx.configs.constants import FASTAPI_USERS_AUTH_COOKIE_NAME
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import SharingScope
from onyx.db.models import BuildSession
from onyx.server.features.build.api import api
from onyx.server.features.build.api.api import public_build_router

# Obvious placeholders. These are not credentials.
_SESSION_PLACEHOLDER = "visitor-session-placeholder"
_AUTHORIZATION_PLACEHOLDER = "visitor-authorization-placeholder"
_MARKER_PLACEHOLDER = "visitor-marker-placeholder"
_FORWARDED_FOR_PLACEHOLDER = "203.0.113.50"
_VISITOR_HOST = "onyx.example"

_SAFE_HEADERS = {
    "accept": "text/html",
    "accept-language": "en",
    "if-none-match": "visitor-etag",
    "range": "bytes=0-1",
    "user-agent": "VisitorBrowser/1.0",
}

# Headers the sandbox must never see, plus one header outside the allowlist.
_FORBIDDEN_HEADER_NAMES = {
    "cookie",
    "authorization",
    "proxy-authorization",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-forwarded-port",
    "x-forwarded-user",
    "x-real-ip",
    "x-visitor-marker",
}

_NEXTJS_PORT = 4317
_HTML = b"<html><head></head><body>ok</body></html>"


class _RecordingServer(ThreadingHTTPServer):
    requests: list[tuple[str, dict[str, str]]]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        server = cast(_RecordingServer, self.server)
        headers = {key.lower(): value for key, value in self.headers.items()}
        server.requests.append((self.path, headers))
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(_HTML)))
        self.end_headers()
        self.wfile.write(_HTML)

    def log_message(self, format: str, *args: object) -> None:  # noqa: ARG002
        return


@contextmanager
def _sandbox_server() -> Iterator[_RecordingServer]:
    server = _RecordingServer(("127.0.0.1", 0), _Handler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _visitor_headers() -> dict[str, str]:
    return {
        "Cookie": f"{FASTAPI_USERS_AUTH_COOKIE_NAME}={_SESSION_PLACEHOLDER}",
        "Authorization": f"Bearer {_AUTHORIZATION_PLACEHOLDER}",
        "Proxy-Authorization": f"Basic {_AUTHORIZATION_PLACEHOLDER}",
        "X-Forwarded-For": _FORWARDED_FOR_PLACEHOLDER,
        "X-Forwarded-Host": _VISITOR_HOST,
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Port": "443",
        "X-Forwarded-User": "visitor",
        "X-Real-IP": _FORWARDED_FOR_PLACEHOLDER,
        "X-Visitor-Marker": _MARKER_PLACEHOLDER,
        "Host": _VISITOR_HOST,
        "Accept": _SAFE_HEADERS["accept"],
        "Accept-Language": _SAFE_HEADERS["accept-language"],
        "If-None-Match": _SAFE_HEADERS["if-none-match"],
        "Range": _SAFE_HEADERS["range"],
        "User-Agent": _SAFE_HEADERS["user-agent"],
    }


class _Db:
    def __init__(self, build_session: SimpleNamespace) -> None:
        self._build_session = build_session

    def get(self, model: type, key: UUID) -> SimpleNamespace | None:
        if model is BuildSession and key == self._build_session.id:
            return self._build_session
        return None


def _open_webapp(
    monkeypatch: pytest.MonkeyPatch,
    server: _RecordingServer,
    *,
    sharing_scope: SharingScope,
    visitor_id: UUID | None,
    owner_id: UUID,
    session_id: UUID,
    sandbox_id: UUID,
    webapp_path: str,
) -> tuple[int, list[UUID], list[tuple[UUID, int]]]:
    build_session = SimpleNamespace(
        id=session_id,
        user_id=owner_id,
        sharing_scope=sharing_scope,
        nextjs_port=_NEXTJS_PORT,
    )
    db = _Db(build_session)
    visitor = None if visitor_id is None else SimpleNamespace(id=visitor_id)
    sandbox_user_ids: list[UUID] = []
    webapp_lookups: list[tuple[UUID, int]] = []
    port = server.server_address[1]
    sandbox_url = f"http://127.0.0.1:{port}"

    def get_sandbox_by_user_id(_db: Session, user_id: UUID) -> SimpleNamespace | None:
        sandbox_user_ids.append(user_id)
        if user_id != owner_id:
            return None
        return SimpleNamespace(id=sandbox_id)

    class _Manager:
        def get_webapp_url(self, looked_up_sandbox_id: UUID, port: int) -> str:
            webapp_lookups.append((looked_up_sandbox_id, port))
            if looked_up_sandbox_id != sandbox_id or port != _NEXTJS_PORT:
                return "http://127.0.0.1:9"
            return sandbox_url

    monkeypatch.setattr(api, "get_sandbox_by_user_id", get_sandbox_by_user_id)
    monkeypatch.setattr(api, "get_sandbox_manager", lambda: _Manager())

    app = FastAPI()
    app.include_router(public_build_router)
    app.dependency_overrides[optional_user] = lambda: visitor
    app.dependency_overrides[get_session] = lambda: db

    route = f"/build/sessions/{session_id}/webapp"
    if webapp_path:
        route = f"{route}/{webapp_path}"

    with TestClient(app) as client:
        response = client.get(
            route,
            params={"view": "1"},
            headers=_visitor_headers(),
            follow_redirects=False,
        )
    return response.status_code, sandbox_user_ids, webapp_lookups


def _assert_sandbox_saw_only_safe_headers(
    server: _RecordingServer,
    *,
    webapp_path: str,
    sandbox_user_ids: list[UUID],
    webapp_lookups: list[tuple[UUID, int]],
    owner_id: UUID,
    sandbox_id: UUID,
) -> None:
    assert sandbox_user_ids == [owner_id]
    assert webapp_lookups == [(sandbox_id, _NEXTJS_PORT)]
    assert len(server.requests) == 1
    path, headers = server.requests[0]
    expected_path = f"/{webapp_path}" if webapp_path else "/"
    assert path == f"{expected_path}?view=1"

    for name in _FORBIDDEN_HEADER_NAMES:
        assert name not in headers
    blob = "\n".join(f"{name}: {value}" for name, value in headers.items()).lower()
    for secret in (
        _SESSION_PLACEHOLDER,
        _AUTHORIZATION_PLACEHOLDER,
        _MARKER_PLACEHOLDER,
        _FORWARDED_FOR_PLACEHOLDER,
        FASTAPI_USERS_AUTH_COOKIE_NAME,
        _VISITOR_HOST,
    ):
        assert secret.lower() not in blob

    for name, value in _SAFE_HEADERS.items():
        assert headers.get(name) == value


@pytest.mark.parametrize("webapp_path", ["", "assets/app.js"])
@pytest.mark.parametrize(
    "sharing_scope",
    [SharingScope.PUBLIC_ORG, SharingScope.PUBLIC_GLOBAL],
)
def test_shared_webapp_does_not_forward_visitor_credentials(
    monkeypatch: pytest.MonkeyPatch,
    webapp_path: str,
    sharing_scope: SharingScope,
) -> None:
    owner_id = uuid4()
    visitor_id = uuid4()
    session_id = uuid4()
    sandbox_id = uuid4()

    with _sandbox_server() as server:
        status, sandbox_user_ids, webapp_lookups = _open_webapp(
            monkeypatch,
            server,
            sharing_scope=sharing_scope,
            visitor_id=visitor_id,
            owner_id=owner_id,
            session_id=session_id,
            sandbox_id=sandbox_id,
            webapp_path=webapp_path,
        )

    assert status == 200
    _assert_sandbox_saw_only_safe_headers(
        server,
        webapp_path=webapp_path,
        sandbox_user_ids=sandbox_user_ids,
        webapp_lookups=webapp_lookups,
        owner_id=owner_id,
        sandbox_id=sandbox_id,
    )


def test_public_global_forwards_no_credentials_when_auth_does_not_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cookie that does not authenticate is still on the request for public apps."""
    owner_id = uuid4()
    session_id = uuid4()
    sandbox_id = uuid4()

    with _sandbox_server() as server:
        status, sandbox_user_ids, webapp_lookups = _open_webapp(
            monkeypatch,
            server,
            sharing_scope=SharingScope.PUBLIC_GLOBAL,
            visitor_id=None,
            owner_id=owner_id,
            session_id=session_id,
            sandbox_id=sandbox_id,
            webapp_path="",
        )

    assert status == 200
    _assert_sandbox_saw_only_safe_headers(
        server,
        webapp_path="",
        sandbox_user_ids=sandbox_user_ids,
        webapp_lookups=webapp_lookups,
        owner_id=owner_id,
        sandbox_id=sandbox_id,
    )


def test_owner_webapp_does_not_forward_session_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_id = uuid4()
    session_id = uuid4()
    sandbox_id = uuid4()

    with _sandbox_server() as server:
        status, sandbox_user_ids, webapp_lookups = _open_webapp(
            monkeypatch,
            server,
            sharing_scope=SharingScope.PRIVATE,
            visitor_id=owner_id,
            owner_id=owner_id,
            session_id=session_id,
            sandbox_id=sandbox_id,
            webapp_path="assets/app.js",
        )

    assert status == 200
    _assert_sandbox_saw_only_safe_headers(
        server,
        webapp_path="assets/app.js",
        sandbox_user_ids=sandbox_user_ids,
        webapp_lookups=webapp_lookups,
        owner_id=owner_id,
        sandbox_id=sandbox_id,
    )


def test_private_webapp_does_not_contact_another_users_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_id = uuid4()
    visitor_id = uuid4()
    session_id = uuid4()
    sandbox_id = uuid4()

    with _sandbox_server() as server:
        status, sandbox_user_ids, webapp_lookups = _open_webapp(
            monkeypatch,
            server,
            sharing_scope=SharingScope.PRIVATE,
            visitor_id=visitor_id,
            owner_id=owner_id,
            session_id=session_id,
            sandbox_id=sandbox_id,
            webapp_path="",
        )
        received = list(server.requests)

    assert status == 404
    assert received == []
    assert sandbox_user_ids == []
    assert webapp_lookups == []


def test_org_webapp_refuses_anonymous_before_contacting_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_id = uuid4()
    session_id = uuid4()
    sandbox_id = uuid4()

    with _sandbox_server() as server:
        status, sandbox_user_ids, webapp_lookups = _open_webapp(
            monkeypatch,
            server,
            sharing_scope=SharingScope.PUBLIC_ORG,
            visitor_id=None,
            owner_id=owner_id,
            session_id=session_id,
            sandbox_id=sandbox_id,
            webapp_path="assets/app.js",
        )
        received = list(server.requests)

    assert status == 302
    assert received == []
    assert sandbox_user_ids == []
    assert webapp_lookups == []
