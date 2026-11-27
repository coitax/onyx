"""Craft webapp responses must not run as the victim's Onyx origin.

Sandbox HTML is attacker-controlled (the session owner instructs the Craft
agent, which may write ``outputs/web``). These tests drive the public webapp
proxy and the preview iframe. They fail while that document is still served
on the Onyx origin with the victim's session cookie.
"""

import asyncio
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse
from uuid import UUID
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import StreamingResponse
from starlette.responses import Response

from onyx.auth.users import cookie_transport
from onyx.configs.app_configs import WEB_DOMAIN
from onyx.db.enums import SharingScope
from onyx.db.models import BuildSession
from onyx.server.features.build.api import api
from onyx.server.features.build.db.build_session import set_build_session_sharing_scope
from onyx.server.features.build.sandbox.local.local_sandbox_manager import (
    LocalSandboxManager,
)

_MARKER = "ONYX_CRAFT_WEBAPP_PAYLOAD"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_PREVIEW_TAB = _REPO_ROOT / "web/src/app/craft/components/output-panel/PreviewTab.tsx"
_EXECUTABLE_DOCUMENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "image/svg+xml",
    "text/xml",
    "application/xml",
)
_UNSAFE_SCRIPT_SOURCES = {
    "*",
    "'self'",
    "'unsafe-inline'",
    "'unsafe-eval'",
    "'unsafe-hashes'",
    "'strict-dynamic'",
    "data:",
    "blob:",
    "filesystem:",
    "http:",
    "https:",
    "ws:",
    "wss:",
}


class _SandboxHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        server = self.server
        assert isinstance(server, _SandboxServer)
        parsed = urlparse(self.path)
        server.hits.append(parsed.path)
        status, headers, body = server.routes.get(parsed.path, (404, {}, b"missing"))
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: ARG002
        return


class _SandboxServer(ThreadingHTTPServer):
    def __init__(self, routes: dict[str, tuple[int, dict[str, str], bytes]]) -> None:
        self.routes = routes
        self.hits: list[str] = []
        super().__init__(("127.0.0.1", 0), _SandboxHandler)


class _UrlManager:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, int]] = []

    def get_webapp_url(self, sandbox_id: UUID, port: int) -> str:
        self.calls.append((sandbox_id, port))
        # Same URL the local sandbox manager gives the proxy.
        return LocalSandboxManager.get_webapp_url(
            self,  # type: ignore[arg-type]
            sandbox_id,
            port,
        )


def _html(marker: str) -> bytes:
    return (
        "<!DOCTYPE html><html><head><title>craft</title></head><body>"
        "<script>fetch('/api/me',{credentials:'include'})</script>"
        f"<!-- {_MARKER}:{marker} --></body></html>"
    ).encode()


def _svg(marker: str) -> bytes:
    return (f'<svg xmlns="http://www.w3.org/2000/svg"><script>/* {_MARKER}:{marker} */</script></svg>').encode()


def _xhtml(marker: str) -> bytes:
    return (
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>craft</title>'
        f"</head><body><script>/* {_MARKER}:{marker} */</script></body></html>"
    ).encode()


def _request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
        }
    )


def _response_bytes(response: Response) -> bytes:
    if isinstance(response, StreamingResponse):

        async def _collect() -> bytes:
            chunks: list[bytes] = []
            async for chunk in response.body_iterator:
                if isinstance(chunk, str):
                    chunks.append(chunk.encode())
                else:
                    chunks.append(bytes(chunk))
            return b"".join(chunks)

        return asyncio.run(_collect())
    return bytes(response.body)


def _cookie_path_covers(cookie_path: str, request_path: str) -> bool:
    if cookie_path == "":
        cookie_path = "/"
    if request_path == cookie_path:
        return True
    prefix = cookie_path if cookie_path.endswith("/") else cookie_path + "/"
    return request_path.startswith(prefix)


def _session_cookie_on_same_origin(request_path: str) -> bool:
    """True when the auth cookie is stored and sent on a same-origin request."""
    if cookie_transport.cookie_samesite == "none" and not cookie_transport.cookie_secure:
        return False
    return _cookie_path_covers(cookie_transport.cookie_path, request_path)


def _opaque_origin_withholds_cookie() -> bool:
    return cookie_transport.cookie_samesite in ("lax", "strict")


def _parse_directives(policy: str) -> dict[str, list[str]]:
    directives: dict[str, list[str]] = {}
    for part in policy.split(";"):
        tokens = part.strip().split()
        if not tokens:
            continue
        name = tokens[0].lower()
        values = [token.lower() for token in tokens[1:]]
        if name == "sandbox" and name in directives:
            directives[name] = sorted(set(directives[name]) | set(values))
        elif name not in directives:
            directives[name] = values
    return directives


def _scripts_blocked(directives: dict[str, list[str]]) -> bool:
    sandbox = directives.get("sandbox")
    if sandbox is not None and "allow-scripts" not in sandbox:
        return True
    sources = directives.get("script-src")
    if sources is None:
        sources = directives.get("default-src")
    if sources is None:
        return False
    if "'none'" in sources:
        return True
    return not any(source in _UNSAFE_SCRIPT_SOURCES or source.endswith(":") for source in sources) and all(
        source.startswith("'nonce-")
        or source.startswith("'sha256-")
        or source.startswith("'sha384-")
        or source.startswith("'sha512-")
        for source in sources
    )


def _policy_isolates(policy: str) -> bool:
    directives = _parse_directives(policy)
    if _scripts_blocked(directives):
        return True
    sandbox = directives.get("sandbox")
    if sandbox is None:
        return False
    if "allow-same-origin" in sandbox:
        return False
    return _opaque_origin_withholds_cookie()


def _content_type(response: Response) -> str:
    return response.headers.get("content-type", "").split(";", 1)[0].strip().lower()


def _is_executable_document(response: Response) -> bool:
    disposition = response.headers.get("content-disposition", "").lower()
    if disposition.startswith("attachment"):
        return False
    return _content_type(response) in _EXECUTABLE_DOCUMENT_TYPES


def _document_isolated(response: Response) -> bool:
    if not _is_executable_document(response):
        return True
    policies = response.headers.getlist("content-security-policy")
    return any(_policy_isolates(policy) for policy in policies)


def _cross_origin_redirect(response: Response) -> bool:
    if response.status_code not in {301, 302, 303, 307, 308}:
        return False
    location = response.headers.get("location", "")
    parsed = urlparse(location)
    # Relative redirects stay on the Onyx host that served the webapp.
    if not parsed.scheme or not parsed.netloc:
        return False
    return parsed.netloc != urlparse(WEB_DOMAIN).netloc


def _assert_not_credentialed(response: Response, marker: str) -> None:
    """The sandbox document must not run with the viewer's session cookie."""
    webapp_path = f"/api/build/sessions/{uuid4()}/webapp"
    assert _session_cookie_on_same_origin(webapp_path), (
        "session cookie would not be sent on the webapp URL; "
        f"name={cookie_transport.cookie_name} "
        f"path={cookie_transport.cookie_path} "
        f"samesite={cookie_transport.cookie_samesite} "
        f"secure={cookie_transport.cookie_secure}"
    )
    if _cross_origin_redirect(response):
        return
    body = _response_bytes(response)
    assert marker.encode() in body
    assert _document_isolated(response), (
        "sandbox document is executable on the Onyx origin "
        f"status={response.status_code} "
        f"content-type={response.headers.get('content-type', '')!r} "
        f"csp={response.headers.getlist('content-security-policy')!r}"
    )


@contextmanager
def _serve(
    routes: dict[str, tuple[int, dict[str, str], bytes]],
) -> Iterator[_SandboxServer]:
    server = _SandboxServer(routes)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _user(user_id: UUID) -> Any:
    return SimpleNamespace(id=user_id)


def _session(session_id: UUID, owner_id: UUID, scope: SharingScope, port: int) -> Any:
    return SimpleNamespace(
        id=session_id,
        user_id=owner_id,
        sharing_scope=scope,
        nextjs_port=port,
    )


def _db_from_sessions(sessions: dict[UUID, Any]) -> Any:
    class _Db:
        def get(self, model: object, ident: UUID) -> Any:
            if model is BuildSession:
                return sessions.get(ident)
            return None

    return _Db()


def _proxy_as(
    monkeypatch: pytest.MonkeyPatch,
    sandboxes_by_user: dict[UUID, Any],
) -> _UrlManager:
    manager = _UrlManager()
    monkeypatch.setattr(
        api,
        "get_sandbox_by_user_id",
        lambda _db, user_id: sandboxes_by_user.get(user_id),
    )
    monkeypatch.setattr(api, "get_sandbox_manager", lambda: manager)
    return manager


def _fetch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scope: SharingScope,
    path: str,
    content_type: str,
    body: bytes,
    viewer: Any,
    extra_headers: dict[str, str] | None = None,
) -> tuple[Response, _SandboxServer, UUID]:
    owner_id = uuid4()
    session_id = uuid4()
    sandbox = SimpleNamespace(id=uuid4())
    route = "/" if path == "" else f"/{path}"
    headers = {"Content-Type": content_type}
    if extra_headers:
        headers.update(extra_headers)
    routes = {route: (200, headers, body)}
    server_cm = _serve(routes)
    server = server_cm.__enter__()
    port = server.server_address[1]
    assert isinstance(port, int)
    sessions = {
        session_id: _session(session_id, owner_id, scope, port),
    }
    manager = _proxy_as(monkeypatch, {owner_id: sandbox})
    try:
        response = api.get_webapp(
            session_id,
            _request(f"/build/sessions/{session_id}/webapp"),
            path,
            viewer,
            _db_from_sessions(sessions),
        )
    except Exception:
        server_cm.__exit__(None, None, None)
        raise
    assert manager.calls, "proxy did not resolve the sandbox URL"
    assert manager.calls[-1] == (sandbox.id, port)
    # Keep the server alive until the caller finishes reading the body.
    response.background = None  # type: ignore[attr-defined]
    setattr(response, "_sandbox_cm", server_cm)
    return response, server, session_id


def _close(response: Response) -> None:
    server_cm = getattr(response, "_sandbox_cm", None)
    if server_cm is not None:
        server_cm.__exit__(None, None, None)


def test_owner_can_publish_webapp_to_org_and_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publishing is the owner-controlled step that exposes the document."""
    from onyx.server.features.build.db import build_session as sharing_db

    stored = SimpleNamespace(sharing_scope=SharingScope.PRIVATE)

    def _get_build_session(_session_id: UUID, _user_id: UUID, _db_session: object) -> Any:
        return stored

    class _Db:
        def commit(self) -> None:
            return None

    monkeypatch.setattr(sharing_db, "get_build_session", _get_build_session)
    for scope in (SharingScope.PUBLIC_ORG, SharingScope.PUBLIC_GLOBAL):
        updated = set_build_session_sharing_scope(uuid4(), uuid4(), scope, _Db())
        assert updated is stored
        assert stored.sharing_scope == scope


def test_public_global_html_runs_as_onyx_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "global-root"
    response, _server, _session_id = _fetch(
        monkeypatch,
        scope=SharingScope.PUBLIC_GLOBAL,
        path="",
        content_type="text/html; charset=UTF-8",
        body=_html(marker),
        viewer=None,
    )
    try:
        _assert_not_credentialed(response, f"{_MARKER}:{marker}")
    finally:
        _close(response)


def test_public_org_html_for_other_user_runs_as_onyx_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "org-subpath"
    response, _server, _session_id = _fetch(
        monkeypatch,
        scope=SharingScope.PUBLIC_ORG,
        path="payload.html",
        content_type="text/html; charset=UTF-8",
        body=_html(marker),
        viewer=_user(uuid4()),
    )
    try:
        _assert_not_credentialed(response, f"{_MARKER}:{marker}")
    finally:
        _close(response)


def test_upstream_csp_cannot_keep_same_origin_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sandbox chooses its own CSP. The proxy must still isolate it."""
    marker = "attacker-csp"
    response, _server, _session_id = _fetch(
        monkeypatch,
        scope=SharingScope.PUBLIC_GLOBAL,
        path="",
        content_type="text/html; charset=UTF-8",
        body=_html(marker),
        viewer=_user(uuid4()),
        extra_headers={
            "Content-Security-Policy": "sandbox allow-scripts allow-same-origin",
        },
    )
    try:
        _assert_not_credentialed(response, f"{_MARKER}:{marker}")
    finally:
        _close(response)


def test_svg_webapp_path_runs_as_onyx_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "svg"
    response, _server, _session_id = _fetch(
        monkeypatch,
        scope=SharingScope.PUBLIC_GLOBAL,
        path="payload.svg",
        content_type="image/svg+xml",
        body=_svg(marker),
        viewer=None,
    )
    try:
        _assert_not_credentialed(response, f"{_MARKER}:{marker}")
    finally:
        _close(response)


def test_xhtml_webapp_path_runs_as_onyx_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "xhtml"
    response, _server, _session_id = _fetch(
        monkeypatch,
        scope=SharingScope.PUBLIC_ORG,
        path="payload.xhtml",
        content_type="application/xhtml+xml",
        body=_xhtml(marker),
        viewer=_user(uuid4()),
    )
    try:
        _assert_not_credentialed(response, f"{_MARKER}:{marker}")
    finally:
        _close(response)


def test_private_webapp_is_refused_before_the_sandbox_is_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_id = uuid4()
    session_id = uuid4()
    sandbox = SimpleNamespace(id=uuid4())
    body = _html("private")
    with _serve({"/": (200, {"Content-Type": "text/html"}, body)}) as server:
        port = server.server_address[1]
        assert isinstance(port, int)
        sessions = {
            session_id: _session(session_id, owner_id, SharingScope.PRIVATE, port),
        }
        _proxy_as(monkeypatch, {owner_id: sandbox})
        with pytest.raises(HTTPException) as exc_info:
            api.get_webapp(
                session_id,
                _request(f"/build/sessions/{session_id}/webapp"),
                "",
                _user(uuid4()),
                _db_from_sessions(sessions),
            )
        assert exc_info.value.status_code == 404
        assert server.hits == []


def test_public_org_anonymous_redirects_before_the_sandbox_is_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_id = uuid4()
    session_id = uuid4()
    sandbox = SimpleNamespace(id=uuid4())
    with _serve({"/": (200, {"Content-Type": "text/html"}, _html("anon"))}) as server:
        port = server.server_address[1]
        assert isinstance(port, int)
        sessions = {
            session_id: _session(session_id, owner_id, SharingScope.PUBLIC_ORG, port),
        }
        _proxy_as(monkeypatch, {owner_id: sandbox})
        response = api.get_webapp(
            session_id,
            _request(f"/build/sessions/{session_id}/webapp"),
            "",
            None,
            _db_from_sessions(sessions),
        )
        assert response.status_code == 302
        assert response.headers["location"] == "/auth/login"
        assert server.hits == []


def test_proxy_fetches_the_authorized_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The document returned is the session that passed the sharing check."""
    owner_a = uuid4()
    owner_b = uuid4()
    session_a = uuid4()
    session_b = uuid4()
    sandbox_a = SimpleNamespace(id=uuid4())
    sandbox_b = SimpleNamespace(id=uuid4())
    with (
        _serve(
            {
                "/": (
                    200,
                    {"Content-Type": "text/html; charset=UTF-8"},
                    _html("session-a"),
                )
            }
        ) as server_a,
        _serve(
            {
                "/": (
                    200,
                    {"Content-Type": "text/html; charset=UTF-8"},
                    _html("session-b"),
                )
            }
        ) as server_b,
    ):
        port_a = server_a.server_address[1]
        port_b = server_b.server_address[1]
        assert isinstance(port_a, int)
        assert isinstance(port_b, int)
        sessions = {
            session_a: _session(session_a, owner_a, SharingScope.PUBLIC_GLOBAL, port_a),
            session_b: _session(session_b, owner_b, SharingScope.PUBLIC_GLOBAL, port_b),
        }
        manager = _proxy_as(
            monkeypatch,
            {owner_a: sandbox_a, owner_b: sandbox_b},
        )
        response = api.get_webapp(
            session_a,
            _request(f"/build/sessions/{session_a}/webapp"),
            "",
            None,
            _db_from_sessions(sessions),
        )
        body = _response_bytes(response)
        assert f"{_MARKER}:session-a".encode() in body
        assert f"{_MARKER}:session-b".encode() not in body
        assert manager.calls == [(sandbox_a.id, port_a)]
        assert server_b.hits == []


def _iframe_open_tags(source: str) -> list[str]:
    blocks: list[str] = []
    start = 0
    lowered = source.lower()
    while True:
        idx = lowered.find("<iframe", start)
        if idx < 0:
            break
        end = source.find("/>", idx)
        assert end != -1, "preview iframe is not self-closing"
        blocks.append(source[idx:end])
        start = end + 2
    return blocks


def test_preview_iframe_cannot_run_same_origin_scripts() -> None:
    source = _PREVIEW_TAB.read_text()
    iframes = _iframe_open_tags(source)
    assert iframes, "preview iframe is missing"
    for attrs in iframes:
        match = re.search(
            r"""sandbox\s*=\s*(?:"([^"]*)"|'([^']*)')""",
            attrs,
        )
        assert match is not None, "iframe has no sandbox attribute"
        raw = match.group(1) if match.group(1) is not None else match.group(2)
        assert raw is not None
        tokens = set(raw.split())
        assert "allow-scripts" not in tokens or "allow-same-origin" not in tokens, (
            f"preview iframe grants scripts the Onyx origin: sandbox={raw!r}"
        )
