"""Open redirect regression tests for Google OAuth and OIDC login.

These tests fail while `next` is copied into the OAuth state and used as the
post-login Location. They pass once authorize and the callback only allow a
relative path (single leading slash, no backslash, no scheme, no host) and
fall back to `/` otherwise.
"""

from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from urllib.parse import parse_qs
from urllib.parse import urlparse

import pytest
from fastapi import FastAPI
from fastapi import Response
from fastapi.testclient import TestClient
from fastapi_users.authentication import AuthenticationBackend
from fastapi_users.authentication import CookieTransport
from fastapi_users.jwt import decode_jwt
from fastapi_users.jwt import generate_jwt
from httpx_oauth.oauth2 import BaseOAuth2

from onyx.auth.users import CSRF_TOKEN_COOKIE_NAME
from onyx.auth.users import CSRF_TOKEN_KEY
from onyx.auth.users import get_oauth_router
from onyx.auth.users import get_pkce_cookie_name
from onyx.auth.users import PKCE_COOKIE_NAME_PREFIX
from onyx.auth.users import STATE_TOKEN_AUDIENCE
from onyx.error_handling.exceptions import register_onyx_exception_handlers

_STATE_SECRET = "test-secret"

# Attacker-controlled `next` values. Each one must fall back to `/`.
_MALICIOUS_NEXT_VALUES = [
    "https://evil.example/login",
    "http://evil.example/login",
    "//evil.example/login",
    "/\\evil.example",
    "javascript:alert(1)",
    "/javascript:alert(1)",
    " https://evil.example/login",
    "\\\\evil.example",
    "https://evil.example/login?stolen=1",
]

# (oauth client name, mount prefix, pkce). Mirrors create_onyx_oauth_router
# for Google OAuth (pkce off) and OIDC (pkce off by default, on when configured).
_PROVIDERS = [
    ("google", "/auth/oauth", False),
    ("openid", "/auth/oidc", False),
    ("openid", "/auth/oidc", True),
]


class _StubOAuthClient:
    def __init__(self, name: str) -> None:
        self.name = name

    async def get_authorization_url(
        self,
        redirect_uri: str,
        state: str | None = None,
        scope: list[str] | None = None,
        code_challenge: str | None = None,
        code_challenge_method: str | None = None,
    ) -> str:
        del redirect_uri, scope, code_challenge, code_challenge_method
        return f"https://idp.example.com/authorize?state={state}"

    async def get_access_token(self, code: str, redirect_uri: str, code_verifier: str | None = None) -> dict[str, str | int]:
        del code, redirect_uri, code_verifier
        return {
            "access_token": "oauth-access-token",
            "refresh_token": "oauth-refresh-token",
            "expires_at": 1730000000,
        }

    async def get_id_email(self, _access_token: str) -> tuple[str, str | None]:
        return ("oauth-account-id", "oauth-user@example.com")


def _build_client(
    client_name: str,
    prefix: str,
    enable_pkce: bool,
) -> TestClient:
    oauth_client = _StubOAuthClient(client_name)
    transport = CookieTransport(cookie_name="testsession")

    async def get_strategy() -> MagicMock:
        return MagicMock()

    backend = AuthenticationBackend(
        name="test_backend",
        transport=transport,
        get_strategy=get_strategy,
    )
    login_response = Response(status_code=204)
    login_response.set_cookie("testsession", "session-token")
    backend.login = AsyncMock(return_value=login_response)  # type: ignore[method-assign]

    user = MagicMock()
    user.is_active = True
    user_manager = MagicMock()
    user_manager.oauth_callback = AsyncMock(return_value=user)
    user_manager.on_after_login = AsyncMock()

    async def get_user_manager() -> MagicMock:
        return user_manager

    router = get_oauth_router(
        oauth_client=cast(BaseOAuth2[Any], oauth_client),
        backend=backend,
        get_user_manager=get_user_manager,
        state_secret=_STATE_SECRET,
        redirect_url=f"http://localhost:3000{prefix}/callback",
        associate_by_email=True,
        is_verified_by_default=True,
        enable_pkce=enable_pkce,
    )
    app = FastAPI()
    app.include_router(router, prefix=prefix)
    register_onyx_exception_handlers(app)
    return TestClient(app, raise_server_exceptions=False)


def _is_safe_internal_redirect(location: str) -> bool:
    """Same allowlist as SAML `_sanitize_relay_state`."""
    if not location.startswith("/") or location.startswith("//"):
        return False
    if "\\" in location:
        return False
    path_portion = location.split("?", 1)[0].split("#", 1)[0]
    if ":" in path_portion:
        return False
    parsed = urlparse(location)
    return not parsed.scheme and not parsed.netloc


def _assert_fallback(location: str) -> None:
    assert _is_safe_internal_redirect(location), location
    assert urlparse(location).path == "/", location
    assert "evil.example" not in location
    assert "javascript" not in location.lower()


def _state_from_authorize_redirect(response: Any) -> str:
    assert response.status_code == 302, response.text
    location = response.headers["location"]
    assert location.startswith("https://idp.example.com/authorize?"), location
    return parse_qs(urlparse(location).query)["state"][0]


def _decode_state(state: str) -> dict[str, Any]:
    return decode_jwt(state, _STATE_SECRET, [STATE_TOKEN_AUDIENCE])


def _finish_login(
    client: TestClient,
    prefix: str,
    state: str,
    tenant_id: str | None,
) -> Any:
    """Call the callback without letting the HTTP client parse Location.

    httpx builds the next request for every redirect before honoring
    follow_redirects=False, and rejects non-http Location values. Sending a
    single request keeps those Location headers observable.
    """
    with patch(
        "onyx.auth.users.fetch_ee_implementation_or_noop",
        return_value=lambda _email: tenant_id,
    ):
        request = client.build_request(
            "GET",
            f"{prefix}/callback",
            params={"code": "abc123", "state": state},
        )
        response = client._send_single_request(request)  # type: ignore[attr-defined]
        response.read()
        return response


def _assert_login_redirect(response: Any) -> str:
    assert response.status_code == 302, response.text
    assert "testsession" in response.headers.get("set-cookie", "")
    location = response.headers["location"]
    return location


@pytest.mark.parametrize("client_name,prefix,enable_pkce", _PROVIDERS)
@pytest.mark.parametrize("tenant_id", [None, "tenant_1"])
@pytest.mark.parametrize("next_url", _MALICIOUS_NEXT_VALUES)
def test_authorize_and_callback_do_not_redirect_off_site(
    client_name: str,
    prefix: str,
    enable_pkce: bool,
    tenant_id: str | None,
    next_url: str,
) -> None:
    """Attacker-supplied `next` on the public authorize URL must not survive.

    The callback Location is the value the browser follows (directly, or via
    the Next.js callback route, which resolves an absolute Location as-is).
    """
    client = _build_client(client_name, prefix, enable_pkce)
    authorize = client.get(
        f"{prefix}/authorize",
        params={"next": next_url, "redirect": "true"},
        follow_redirects=False,
    )
    state = _state_from_authorize_redirect(authorize)
    stored_next = _decode_state(state).get("next_url")
    if stored_next is not None:
        _assert_fallback(stored_next)

    response = _finish_login(client, prefix, state, tenant_id)
    location = _assert_login_redirect(response)
    _assert_fallback(location)


@pytest.mark.parametrize("client_name,prefix,enable_pkce", _PROVIDERS)
@pytest.mark.parametrize("tenant_id", [None, "tenant_1"])
@pytest.mark.parametrize("next_url", _MALICIOUS_NEXT_VALUES)
def test_callback_rejects_malicious_next_already_in_signed_state(
    client_name: str,
    prefix: str,
    enable_pkce: bool,
    tenant_id: str | None,
    next_url: str,
) -> None:
    """Callback must sanitize `next_url` from the state it redirects to.

    Signing the state here uses the test placeholder secret so the callback
    sees a structurally valid state whose `next_url` was not rewritten by
    authorize. A fix that only checks `next` while building the state, then
    redirects to `state_data["next_url"]`, still fails this test.
    """
    client = _build_client(client_name, prefix, enable_pkce)
    authorize = client.get(
        f"{prefix}/authorize",
        params={"redirect": "true"},
        follow_redirects=False,
    )
    _state_from_authorize_redirect(authorize)
    csrf_token = client.cookies.get(CSRF_TOKEN_COOKIE_NAME)
    assert csrf_token is not None

    verifier: str | None = None
    for key in client.cookies.keys():
        if key.startswith(PKCE_COOKIE_NAME_PREFIX):
            verifier = client.cookies.get(key)

    malicious_state = generate_jwt(
        {
            "aud": STATE_TOKEN_AUDIENCE,
            "next_url": next_url,
            "referral_source": "default_referral",
            CSRF_TOKEN_KEY: csrf_token,
        },
        _STATE_SECRET,
        3600,
    )
    if enable_pkce:
        assert verifier is not None
        client.cookies.set(get_pkce_cookie_name(malicious_state), verifier)

    response = _finish_login(client, prefix, malicious_state, tenant_id)
    location = _assert_login_redirect(response)
    _assert_fallback(location)


@pytest.mark.parametrize("client_name,prefix,enable_pkce", _PROVIDERS)
@pytest.mark.parametrize("tenant_id", [None, "tenant_1"])
@pytest.mark.parametrize("next_url", ["/chat", "/chat?tab=1", "/app?from=login"])
def test_internal_next_is_preserved(
    client_name: str,
    prefix: str,
    enable_pkce: bool,
    tenant_id: str | None,
    next_url: str,
) -> None:
    client = _build_client(client_name, prefix, enable_pkce)
    authorize = client.get(
        f"{prefix}/authorize",
        params={"next": next_url, "redirect": "true"},
        follow_redirects=False,
    )
    state = _state_from_authorize_redirect(authorize)
    stored_next = _decode_state(state).get("next_url")
    assert stored_next == next_url

    response = _finish_login(client, prefix, state, tenant_id)
    location = _assert_login_redirect(response)
    assert _is_safe_internal_redirect(location), location
    expected_path = urlparse(next_url).path
    assert urlparse(location).path == expected_path, location
    if tenant_id is None:
        assert parse_qs(urlparse(location).query).get("new_team") == ["true"]
    else:
        assert "new_team" not in parse_qs(urlparse(location).query)
        assert location == next_url
