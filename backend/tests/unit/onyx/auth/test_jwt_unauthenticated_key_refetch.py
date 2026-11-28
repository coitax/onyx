"""Regression spec for unauthenticated JWT key refetch.

These tests describe the behavior required to close the denial-of-service in
JWT bearer login. They fail while an invalid ``Authorization: Bearer`` token
still clears the cached key set and fetches ``JWT_PUBLIC_KEY_URL`` again with
a synchronous ``requests.get`` that has no timeout.

A passing run means:

* A bearer value that is not a three-part JWT, including API-key and personal
  access token prefixes, never contacts the key URL. API-key and personal
  access token bearers still resolve through their own authenticators.
* A well-formed token that fails signature verification does not refetch.
* A well-formed token whose ``kid`` is absent may trigger at most one refetch
  per burst, including when many such tokens are verified concurrently.
* A PEM key document follows the same rule: invalid tokens do not refetch.
* When a fetch does happen, it carries a finite timeout of at most 30 seconds
  and does not block the asyncio event loop.
* A correctly signed token still verifies, with a single fetch.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import threading
import time
import urllib.request
from collections.abc import AsyncIterator
from collections.abc import Callable
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import jwt as pyjwt
import pytest
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from fastapi import Depends
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm
from starlette.requests import Request

from onyx.auth import jwt as jwt_module
from onyx.auth import users as users_module
from onyx.auth.constants import API_KEY_PREFIX
from onyx.auth.constants import DEPRECATED_API_KEY_PREFIX
from onyx.auth.constants import PAT_PREFIX
from onyx.auth.permissions import require_permission
from onyx.auth.users import current_chat_accessible_user
from onyx.auth.users import current_curator_or_admin_user
from onyx.auth.users import current_limited_user
from onyx.auth.users import current_user
from onyx.auth.users import current_user_with_expired_token
from onyx.auth.users import optional_user
from onyx.db.engine.async_sql_engine import get_async_session
from onyx.db.enums import Permission
from onyx.db.models import User


_MAX_TIMEOUT_SECONDS = 30.0
_GARBAGE_BEARER = "Bearer x"
_REPEAT_COUNT = 8


class _KeyHTTPServer(ThreadingHTTPServer):
    def __init__(self, body: bytes, content_type: str, delay_seconds: float) -> None:
        super().__init__(("127.0.0.1", 0), _KeyHandler)
        self.body = body
        self.content_type = content_type
        self.delay_seconds = delay_seconds
        self.hits = 0


class _KeyHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        server = cast(_KeyHTTPServer, self.server)
        server.hits += 1
        if server.delay_seconds:
            time.sleep(server.delay_seconds)
        payload = server.body
        self.send_response(200)
        self.send_header("Content-Type", server.content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        return


class _KeyEndpoint:
    def __init__(self, body: bytes, content_type: str, delay_seconds: float) -> None:
        self._server = _KeyHTTPServer(body, content_type, delay_seconds)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/keys"

    @property
    def hits(self) -> int:
        return self._server.hits

    @hits.setter
    def hits(self, value: int) -> None:
        self._server.hits = value

    def close(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=2)
        self._server.server_close()


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


# Valid base64url so PyJWT accepts the segment, but not a real RSA signature.
_BAD_SIGNATURE_SEGMENT = _b64url(b"\x00" * 64)


def _unsigned_jwt(header: dict[str, Any], payload: dict[str, Any]) -> str:
    header_segment = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_segment = _b64url(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )
    return f"{header_segment}.{payload_segment}.{_BAD_SIGNATURE_SEGMENT}"


def _generate_rsa_key() -> RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwks_document(private_key: RSAPrivateKey, kid: str) -> bytes:
    jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk["kid"] = kid
    jwk["use"] = "sig"
    jwk["alg"] = "RS256"
    return json.dumps({"keys": [jwk]}).encode("utf-8")


def _pem_document(private_key: RSAPrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _request_with_bearer(value: str) -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/me",
        "raw_path": b"/api/me",
        "query_string": b"",
        "headers": [(b"authorization", value.encode("ascii"))],
        "client": ("203.0.113.5", 12345),
        "server": ("onyx.example", 80),
    }
    return Request(scope)


def _bounded_timeout(value: object) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return 0 < float(value) <= _MAX_TIMEOUT_SECONDS
    if isinstance(value, tuple):
        return len(value) > 0 and all(_bounded_timeout(part) for part in value)
    if isinstance(value, dict):
        return _bounded_timeout(value.get("connect")) and _bounded_timeout(
            value.get("read")
        )
    total = getattr(value, "total", None)
    if total is not None or hasattr(value, "total"):
        return _bounded_timeout(total)
    connect = getattr(value, "connect_timeout", None)
    read = getattr(value, "read_timeout", None)
    if connect is not None or read is not None:
        return _bounded_timeout(connect) and _bounded_timeout(read)
    return False


class _FetchObserver:
    """Record the timeout used by the HTTP client that contacts the key URL.

    The wrappers call the real client afterwards, so the key endpoint still
    sees the request the product code makes.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, key_url: str) -> None:
        self.timeouts: list[object] = []
        self._key_url = key_url
        self._install(monkeypatch)

    def _matches(self, url: object) -> bool:
        return self._key_url in str(url)

    def _install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original_request = requests.sessions.Session.request

        def wrapped_request(
            session: requests.sessions.Session,
            method: str,
            url: str,
            *args: Any,
            **kwargs: Any,
        ) -> requests.Response:
            if self._matches(url):
                self.timeouts.append(kwargs.get("timeout"))
            return original_request(session, method, url, *args, **kwargs)

        monkeypatch.setattr(requests.sessions.Session, "request", wrapped_request)

        import httpx

        original_send = httpx.Client.send

        def wrapped_send(
            client: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any
        ) -> httpx.Response:
            try:
                return original_send(client, request, *args, **kwargs)
            finally:
                if self._matches(request.url):
                    self.timeouts.append(request.extensions.get("timeout"))

        monkeypatch.setattr(httpx.Client, "send", wrapped_send)

        original_async_send = httpx.AsyncClient.send

        async def wrapped_async_send(
            client: httpx.AsyncClient,
            request: httpx.Request,
            *args: Any,
            **kwargs: Any,
        ) -> httpx.Response:
            try:
                return await original_async_send(client, request, *args, **kwargs)
            finally:
                if self._matches(request.url):
                    self.timeouts.append(request.extensions.get("timeout"))

        monkeypatch.setattr(httpx.AsyncClient, "send", wrapped_async_send)

        original_urlopen = urllib.request.urlopen

        def wrapped_urlopen(
            url: Any,
            data: Any = None,
            timeout: Any = socket.getdefaulttimeout(),
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            target = getattr(url, "full_url", url)
            if self._matches(target):
                observed: object = timeout
                if timeout is socket._GLOBAL_DEFAULT_TIMEOUT:  # type: ignore[attr-defined]
                    observed = socket.getdefaulttimeout()
                self.timeouts.append(observed)
            return original_urlopen(url, data, timeout, *args, **kwargs)

        monkeypatch.setattr(urllib.request, "urlopen", wrapped_urlopen)

        try:
            import aiohttp
        except ImportError:
            return

        original_aiohttp = aiohttp.ClientSession._request

        async def wrapped_aiohttp(
            session: aiohttp.ClientSession, method: str, url: Any, **kwargs: Any
        ) -> Any:
            if self._matches(url):
                timeout = kwargs.get("timeout", getattr(aiohttp, "sentinel", None))
                if timeout is getattr(aiohttp, "sentinel", None):
                    self.timeouts.append(session.timeout.total)
                elif hasattr(timeout, "total"):
                    self.timeouts.append(timeout.total)
                else:
                    self.timeouts.append(timeout)
            return await original_aiohttp(session, method, url, **kwargs)

        monkeypatch.setattr(aiohttp.ClientSession, "_request", wrapped_aiohttp)


@pytest.fixture
def key_endpoints() -> Iterator[Callable[..., _KeyEndpoint]]:
    opened: list[_KeyEndpoint] = []

    def start(
        body: bytes, content_type: str, delay_seconds: float = 0.0
    ) -> _KeyEndpoint:
        endpoint = _KeyEndpoint(body, content_type, delay_seconds)
        opened.append(endpoint)
        return endpoint

    yield start
    for endpoint in opened:
        endpoint.close()


@pytest.fixture(autouse=True)
def _clear_key_cache() -> Iterator[None]:
    jwt_module._fetch_public_key_payload.cache_clear()
    yield
    jwt_module._fetch_public_key_payload.cache_clear()


def _enable_jwt_login(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setattr(jwt_module, "JWT_PUBLIC_KEY_URL", url)
    monkeypatch.setattr(users_module, "JWT_PUBLIC_KEY_URL", url)
    jwt_module._fetch_public_key_payload.cache_clear()


async def _optional_user_for_bearer(bearer_value: str) -> User | None:
    return await optional_user(
        _request_with_bearer(bearer_value), MagicMock(), user=None
    )


def _signed_token(private_key: RSAPrivateKey, kid: str) -> str:
    token = pyjwt.encode(
        {"email": "person@example.com"},
        private_key,
        algorithm="RS256",
        headers={"kid": kid},
    )
    assert isinstance(token, str)
    return token


@pytest.mark.asyncio
async def test_garbage_bearer_does_not_fetch_key_url(
    monkeypatch: pytest.MonkeyPatch,
    key_endpoints: Callable[..., _KeyEndpoint],
) -> None:
    """``Authorization: Bearer x`` is not a JWT and must not contact the key URL."""
    private_key = _generate_rsa_key()
    endpoint = key_endpoints(_jwks_document(private_key, "cached-kid"), "application/json")
    _enable_jwt_login(monkeypatch, endpoint.url)

    outcomes = await asyncio.gather(
        *[
            _optional_user_for_bearer(_GARBAGE_BEARER)
            for _ in range(_REPEAT_COUNT)
        ]
    )

    assert outcomes == [None] * _REPEAT_COUNT
    assert endpoint.hits == 0


@pytest.mark.asyncio
async def test_api_key_and_pat_bearers_skip_jwt_fetch_and_still_authenticate(
    monkeypatch: pytest.MonkeyPatch,
    key_endpoints: Callable[..., _KeyEndpoint],
) -> None:
    """Prefixed bearers stay on the API-key and personal-access-token paths."""
    private_key = _generate_rsa_key()
    endpoint = key_endpoints(_jwks_document(private_key, "cached-kid"), "application/json")
    _enable_jwt_login(monkeypatch, endpoint.url)

    api_key_user = MagicMock()
    pat_user = MagicMock()
    fetch_api_key = AsyncMock(return_value=api_key_user)
    fetch_pat = AsyncMock(return_value=pat_user)
    monkeypatch.setattr(users_module, "fetch_user_for_api_key", fetch_api_key)
    monkeypatch.setattr(users_module, "fetch_user_for_pat", fetch_pat)

    placeholders = {
        API_KEY_PREFIX: f"Bearer {API_KEY_PREFIX}placeholder",
        DEPRECATED_API_KEY_PREFIX: f"Bearer {DEPRECATED_API_KEY_PREFIX}placeholder",
        PAT_PREFIX: f"Bearer {PAT_PREFIX}placeholder",
    }

    api_key_result = await _optional_user_for_bearer(placeholders[API_KEY_PREFIX])
    deprecated_result = await _optional_user_for_bearer(
        placeholders[DEPRECATED_API_KEY_PREFIX]
    )
    pat_result = await _optional_user_for_bearer(placeholders[PAT_PREFIX])

    assert api_key_result is api_key_user
    assert deprecated_result is api_key_user
    assert pat_result is pat_user
    assert fetch_api_key.await_count == 2
    assert fetch_pat.await_count == 1
    assert endpoint.hits == 0

    await asyncio.gather(
        *[
            _optional_user_for_bearer(placeholders[API_KEY_PREFIX])
            for _ in range(_REPEAT_COUNT)
        ]
    )
    assert endpoint.hits == 0


@pytest.mark.asyncio
async def test_invalid_signature_does_not_refetch_jwks(
    monkeypatch: pytest.MonkeyPatch,
    key_endpoints: Callable[..., _KeyEndpoint],
) -> None:
    private_key = _generate_rsa_key()
    kid = "signing-kid"
    endpoint = key_endpoints(_jwks_document(private_key, kid), "application/json")
    _enable_jwt_login(monkeypatch, endpoint.url)
    forged = _unsigned_jwt(
        {"alg": "RS256", "kid": kid}, {"email": "person@example.com"}
    )

    assert await jwt_module.verify_jwt_token(forged) is None
    endpoint.hits = 0

    outcomes = await asyncio.gather(
        *[jwt_module.verify_jwt_token(forged) for _ in range(_REPEAT_COUNT)]
    )

    assert outcomes == [None] * _REPEAT_COUNT
    assert endpoint.hits == 0


@pytest.mark.asyncio
async def test_unknown_kid_refetch_is_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
    key_endpoints: Callable[..., _KeyEndpoint],
) -> None:
    """A missing ``kid`` may refresh the cache at most once per burst."""
    private_key = _generate_rsa_key()
    endpoint = key_endpoints(_jwks_document(private_key, "known-kid"), "application/json")
    _enable_jwt_login(monkeypatch, endpoint.url)

    first = _unsigned_jwt(
        {"alg": "RS256", "kid": "missing-0"}, {"email": "person@example.com"}
    )
    assert await jwt_module.verify_jwt_token(first) is None
    endpoint.hits = 0

    tokens = [
        _unsigned_jwt(
            {"alg": "RS256", "kid": f"missing-{index}"},
            {"email": "person@example.com"},
        )
        for index in range(1, _REPEAT_COUNT + 1)
    ]
    outcomes = await asyncio.gather(
        *[jwt_module.verify_jwt_token(token) for token in tokens]
    )

    assert outcomes == [None] * len(tokens)
    assert endpoint.hits <= 1


@pytest.mark.asyncio
async def test_pem_key_invalid_token_does_not_refetch(
    monkeypatch: pytest.MonkeyPatch,
    key_endpoints: Callable[..., _KeyEndpoint],
) -> None:
    private_key = _generate_rsa_key()
    endpoint = key_endpoints(_pem_document(private_key), "text/plain")
    _enable_jwt_login(monkeypatch, endpoint.url)

    forged = _unsigned_jwt({"alg": "RS256"}, {"email": "person@example.com"})
    assert await jwt_module.verify_jwt_token(forged) is None
    endpoint.hits = 0

    bearers = ["x", forged, "Bearer-not-used", "a.b.c"]
    outcomes = await asyncio.gather(
        *[jwt_module.verify_jwt_token(token) for token in bearers * 2]
    )

    assert outcomes == [None] * len(bearers) * 2
    assert endpoint.hits == 0


@pytest.mark.asyncio
async def test_key_fetch_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    key_endpoints: Callable[..., _KeyEndpoint],
) -> None:
    private_key = _generate_rsa_key()
    endpoint = key_endpoints(
        _jwks_document(private_key, "known-kid"),
        "application/json",
        delay_seconds=0.3,
    )
    _enable_jwt_login(monkeypatch, endpoint.url)
    token = _unsigned_jwt(
        {"alg": "RS256", "kid": "known-kid"}, {"email": "person@example.com"}
    )
    progressed_at: list[float] = []

    async def watcher() -> None:
        await asyncio.sleep(0.05)
        progressed_at.append(time.monotonic())

    watcher_task = asyncio.create_task(watcher())
    await asyncio.sleep(0)
    started = time.monotonic()
    assert await jwt_module.verify_jwt_token(token) is None
    elapsed = time.monotonic() - started

    assert endpoint.hits >= 1
    assert elapsed >= 0.2
    assert progressed_at, "the event loop did not run while the key URL was fetched"
    assert progressed_at[0] < started + 0.15
    watcher_task.cancel()


@pytest.mark.asyncio
async def test_key_fetch_uses_a_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch,
    key_endpoints: Callable[..., _KeyEndpoint],
) -> None:
    private_key = _generate_rsa_key()
    endpoint = key_endpoints(_jwks_document(private_key, "known-kid"), "application/json")
    _enable_jwt_login(monkeypatch, endpoint.url)
    observer = _FetchObserver(monkeypatch, endpoint.url)
    token = _unsigned_jwt(
        {"alg": "RS256", "kid": "known-kid"}, {"email": "person@example.com"}
    )

    assert await jwt_module.verify_jwt_token(token) is None

    assert endpoint.hits >= 1
    assert observer.timeouts, (
        "the key URL was fetched without going through requests, httpx, urllib, "
        "or aiohttp, so the timeout could not be bounded"
    )
    assert len(observer.timeouts) >= endpoint.hits
    assert all(_bounded_timeout(timeout) for timeout in observer.timeouts)


@pytest.mark.asyncio
async def test_valid_jwt_verifies_with_a_single_key_fetch(
    monkeypatch: pytest.MonkeyPatch,
    key_endpoints: Callable[..., _KeyEndpoint],
) -> None:
    private_key = _generate_rsa_key()
    kid = "signing-kid"
    endpoint = key_endpoints(_jwks_document(private_key, kid), "application/json")
    _enable_jwt_login(monkeypatch, endpoint.url)
    token = _signed_token(private_key, kid)

    payload = await jwt_module.verify_jwt_token(token)

    assert payload is not None
    assert payload["email"] == "person@example.com"
    assert endpoint.hits == 1

    assert await jwt_module.verify_jwt_token(token) is not None
    assert endpoint.hits == 1


@pytest.mark.asyncio
async def test_unauthenticated_routes_do_not_fetch_keys_for_garbage_bearer(
    monkeypatch: pytest.MonkeyPatch,
    key_endpoints: Callable[..., _KeyEndpoint],
) -> None:
    """Every dependency that funnels through ``optional_user`` shares the bug."""
    private_key = _generate_rsa_key()
    endpoint = key_endpoints(_jwks_document(private_key, "cached-kid"), "application/json")
    _enable_jwt_login(monkeypatch, endpoint.url)
    monkeypatch.setattr(
        users_module, "anonymous_user_enabled", lambda **kwargs: False
    )

    async def override_session() -> AsyncIterator[MagicMock]:
        yield MagicMock()

    permission_dependency = require_permission(Permission.BASIC_ACCESS)
    application = FastAPI()
    application.dependency_overrides[get_async_session] = override_session

    @application.get("/optional")
    async def optional_route(user: User | None = Depends(optional_user)) -> dict[str, bool]:
        return {"present": user is not None}

    @application.get("/current")
    async def current_route(user: User = Depends(current_user)) -> dict[str, str]:
        return {"email": user.email}

    @application.get("/limited")
    async def limited_route(user: User = Depends(current_limited_user)) -> dict[str, str]:
        return {"email": user.email}

    @application.get("/expired")
    async def expired_route(
        user: User = Depends(current_user_with_expired_token),
    ) -> dict[str, str]:
        return {"email": user.email}

    @application.get("/chat")
    async def chat_route(
        user: User = Depends(current_chat_accessible_user),
    ) -> dict[str, str]:
        return {"email": user.email}

    @application.get("/curator")
    async def curator_route(
        user: User = Depends(current_curator_or_admin_user),
    ) -> dict[str, str]:
        return {"email": user.email}

    @application.get("/permission")
    async def permission_route(
        user: User = Depends(permission_dependency),
    ) -> dict[str, str]:
        return {"email": user.email}

    paths = [
        "/optional",
        "/current",
        "/limited",
        "/expired",
        "/chat",
        "/curator",
        "/permission",
    ]
    with TestClient(application) as client:
        for path in paths:
            response = client.get(path, headers={"Authorization": _GARBAGE_BEARER})
            assert response.status_code < 500, response.text

    assert endpoint.hits == 0
