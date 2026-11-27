"""Bearer JWTs signed by the configured key set must still be for this Onyx app.

``JWT_PUBLIC_KEY_URL`` is operator configuration. The attacker supplies
``Authorization: Bearer``. Verification has to enforce the configured
audience and issuer, require ``exp`` and ``iat``, and accept an email only
when the provider sets ``email_verified`` to true. These tests fail until
that behavior is in place.

Settings honored by the check (read at request time, not only at import):

- ``JWT_AUDIENCE`` on ``onyx.auth.jwt`` / ``onyx.configs.app_configs`` / env
- ``JWT_ISSUER`` on ``onyx.auth.jwt`` / ``onyx.configs.app_configs`` / env

When either setting is missing, bearer JWTs are not authenticated.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization import PublicFormat
from jwt.algorithms import RSAAlgorithm
from starlette.requests import Request

from onyx.auth import jwt as jwt_module
from onyx.auth import users as users_module
from onyx.auth.users import current_user
from onyx.auth.users import optional_user
from onyx.configs import app_configs
from onyx.configs.constants import AuthType
from onyx.db.enums import AccountType
from onyx.server.utils import BasicAuthenticationError


_OMIT = object()

EXPECTED_AUDIENCE = "onyx-api"
EXPECTED_ISSUER = "https://login.example.test/tenant/onyx"
OTHER_AUDIENCE = "another-app"
OTHER_ISSUER = "https://login.example.test/other-tenant"
KEY_ID = "onyx-test-key"
ADMIN_EMAIL = "admin@example.com"
OTHER_EMAIL = "other.user@example.com"
OUTSIDER_EMAIL = "outsider@example.com"


class _KeyHandler(BaseHTTPRequestHandler):
    jwks_body = b""
    pem_body = b""

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/pem"):
            body = self.pem_body
            content_type = "application/x-pem-file"
        else:
            body = self.jwks_body
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@dataclass
class PublishedKeys:
    private_key: rsa.RSAPrivateKey
    other_private_key: rsa.RSAPrivateKey
    jwks_url: str
    pem_url: str


@dataclass
class UserDirectory:
    users: dict[str, MagicMock] = field(default_factory=dict)
    created_emails: list[str] = field(default_factory=list)
    created_verified: list[bool] = field(default_factory=list)


@pytest.fixture(scope="module")
def published_keys() -> Iterator[PublishedKeys]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    jwk = json.loads(RSAAlgorithm.to_jwk(public_key))
    jwk["kid"] = KEY_ID
    jwk["use"] = "sig"
    jwk["alg"] = "RS256"
    _KeyHandler.jwks_body = json.dumps({"keys": [jwk]}).encode()
    _KeyHandler.pem_body = public_key.public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), _KeyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        yield PublishedKeys(
            private_key=private_key,
            other_private_key=other_private_key,
            jwks_url=f"{base}/jwks",
            pem_url=f"{base}/pem",
        )
    finally:
        server.shutdown()
        server.server_close()


def _sign(
    private_key: rsa.RSAPrivateKey,
    claims: dict[str, Any],
    *,
    algorithm: str = "RS256",
) -> str:
    token = pyjwt.encode(
        claims, private_key, algorithm=algorithm, headers={"kid": KEY_ID}
    )
    assert isinstance(token, str)
    return token


def _claims(**overrides: Any) -> dict[str, Any]:
    now = int(time.time())
    payload: dict[str, Any] = {
        "email": ADMIN_EMAIL,
        "aud": EXPECTED_AUDIENCE,
        "iss": EXPECTED_ISSUER,
        "exp": now + 600,
        "iat": now,
        "email_verified": True,
    }
    for key, value in overrides.items():
        if value is _OMIT:
            payload.pop(key, None)
        else:
            payload[key] = value
    return payload


def _configure_expectations(
    monkeypatch: pytest.MonkeyPatch,
    *,
    audience: str | None,
    issuer: str | None,
    key_url: str,
) -> None:
    monkeypatch.setattr(jwt_module, "JWT_PUBLIC_KEY_URL", key_url, raising=False)
    monkeypatch.setattr(users_module, "JWT_PUBLIC_KEY_URL", key_url, raising=False)
    for module in (jwt_module, app_configs, users_module):
        monkeypatch.setattr(module, "JWT_AUDIENCE", audience, raising=False)
        monkeypatch.setattr(module, "JWT_ISSUER", issuer, raising=False)
    if audience is None:
        monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    else:
        monkeypatch.setenv("JWT_AUDIENCE", audience)
    if issuer is None:
        monkeypatch.delenv("JWT_ISSUER", raising=False)
    else:
        monkeypatch.setenv("JWT_ISSUER", issuer)
    jwt_module._fetch_public_key_payload.cache_clear()


def _request_with_bearer(token: str) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/search",
            "raw_path": b"/api/search",
            "query_string": b"",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
            "client": ("203.0.113.10", 44321),
            "server": ("testserver", 80),
        }
    )


def _user(email: str) -> MagicMock:
    user = MagicMock()
    user.email = email
    user.is_active = True
    user.is_verified = True
    user.oidc_expiry = None
    user.account_type = AccountType.STANDARD
    return user


def _install_directory(
    monkeypatch: pytest.MonkeyPatch, directory: UserDirectory
) -> None:
    class StubUserManager:
        def __init__(self, _user_db: object) -> None:
            self.user_db = MagicMock()
            self.user_db.update = AsyncMock()

        async def get_by_email(self, email: str) -> MagicMock:
            try:
                return directory.users[email]
            except KeyError:
                raise users_module.exceptions.UserNotExists()

        async def create(
            self,
            user_create: Any,
            safe: bool = False,
            request: Any = None,
        ) -> MagicMock:
            del safe, request
            directory.created_emails.append(user_create.email)
            directory.created_verified.append(bool(user_create.is_verified))
            created = _user(user_create.email)
            created.is_verified = user_create.is_verified
            directory.users[user_create.email] = created
            return created

    monkeypatch.setattr(users_module, "UserManager", StubUserManager)
    monkeypatch.setattr(
        users_module,
        "SQLAlchemyUserAdminDB",
        lambda *args, **kwargs: MagicMock(),  # noqa: ARG005
    )


def _arm_authenticated_user_policy(monkeypatch: pytest.MonkeyPatch, auth_type: AuthType) -> None:
    """Use the real invite check. OIDC and SAML return before the list."""
    monkeypatch.setattr(users_module, "AUTH_TYPE", auth_type)
    monkeypatch.setattr(users_module, "workspace_invite_only_enabled", lambda: True)
    monkeypatch.setattr(
        users_module,
        "get_invited_users",
        lambda: ["someone-else@example.com"],
    )
    import onyx.auth.disposable_email_validator as disposable

    monkeypatch.setattr(disposable, "DISPOSABLE_EMAIL_DOMAINS_URL", "")
    disposable._validator._domains = set()
    disposable._validator._last_fetch_time = 0


async def _authenticated_user(token: str) -> Any:
    """Run the bearer token through the same dependencies as ``current_user``."""
    authenticated = await optional_user(_request_with_bearer(token), MagicMock(), None)
    if authenticated is not None:
        return await current_user(authenticated)
    with pytest.raises(BasicAuthenticationError):
        await current_user(authenticated)
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("key_format", ["jwks", "pem"])
async def test_token_for_another_audience_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    published_keys: PublishedKeys,
    key_format: str,
) -> None:
    key_url = published_keys.jwks_url if key_format == "jwks" else published_keys.pem_url
    _configure_expectations(
        monkeypatch,
        audience=EXPECTED_AUDIENCE,
        issuer=EXPECTED_ISSUER,
        key_url=key_url,
    )
    token = _sign(
        published_keys.private_key,
        _claims(aud=OTHER_AUDIENCE, email="Admin@Example.com"),
    )

    assert await jwt_module.verify_jwt_token(token) is None


@pytest.mark.asyncio
async def test_token_for_another_issuer_is_rejected(
    monkeypatch: pytest.MonkeyPatch, published_keys: PublishedKeys
) -> None:
    _configure_expectations(
        monkeypatch,
        audience=EXPECTED_AUDIENCE,
        issuer=EXPECTED_ISSUER,
        key_url=published_keys.jwks_url,
    )
    token = _sign(published_keys.private_key, _claims(iss=OTHER_ISSUER))

    assert await jwt_module.verify_jwt_token(token) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_claim", ["aud", "iss", "exp", "iat"])
async def test_token_missing_required_claim_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    published_keys: PublishedKeys,
    missing_claim: str,
) -> None:
    _configure_expectations(
        monkeypatch,
        audience=EXPECTED_AUDIENCE,
        issuer=EXPECTED_ISSUER,
        key_url=published_keys.jwks_url,
    )
    token = _sign(
        published_keys.private_key, _claims(**{missing_claim: _OMIT})
    )

    assert await jwt_module.verify_jwt_token(token) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("audience", "issuer"),
    [(None, EXPECTED_ISSUER), (EXPECTED_AUDIENCE, None), (None, None)],
)
async def test_bearer_auth_disabled_without_audience_and_issuer(
    monkeypatch: pytest.MonkeyPatch,
    published_keys: PublishedKeys,
    audience: str | None,
    issuer: str | None,
) -> None:
    _configure_expectations(
        monkeypatch,
        audience=audience,
        issuer=issuer,
        key_url=published_keys.jwks_url,
    )
    token = _sign(published_keys.private_key, _claims())

    assert await jwt_module.verify_jwt_token(token) is None


@pytest.mark.asyncio
async def test_matching_audience_and_issuer_is_accepted(
    monkeypatch: pytest.MonkeyPatch, published_keys: PublishedKeys
) -> None:
    _configure_expectations(
        monkeypatch,
        audience=EXPECTED_AUDIENCE,
        issuer=EXPECTED_ISSUER,
        key_url=published_keys.jwks_url,
    )
    token = _sign(published_keys.private_key, _claims())

    payload = await jwt_module.verify_jwt_token(token)

    assert payload is not None
    assert payload["email"] == ADMIN_EMAIL
    assert payload["aud"] == EXPECTED_AUDIENCE
    assert payload["iss"] == EXPECTED_ISSUER


@pytest.mark.asyncio
async def test_token_signed_by_a_different_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch, published_keys: PublishedKeys
) -> None:
    _configure_expectations(
        monkeypatch,
        audience=EXPECTED_AUDIENCE,
        issuer=EXPECTED_ISSUER,
        key_url=published_keys.jwks_url,
    )
    token = _sign(published_keys.other_private_key, _claims())

    assert await jwt_module.verify_jwt_token(token) is None


@pytest.mark.asyncio
async def test_expired_token_is_rejected(
    monkeypatch: pytest.MonkeyPatch, published_keys: PublishedKeys
) -> None:
    _configure_expectations(
        monkeypatch,
        audience=EXPECTED_AUDIENCE,
        issuer=EXPECTED_ISSUER,
        key_url=published_keys.jwks_url,
    )
    now = int(time.time())
    token = _sign(
        published_keys.private_key,
        _claims(exp=now - 60, iat=now - 120),
    )

    assert await jwt_module.verify_jwt_token(token) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_type", [AuthType.OIDC, AuthType.SAML])
async def test_foreign_token_does_not_authenticate_existing_user(
    monkeypatch: pytest.MonkeyPatch,
    published_keys: PublishedKeys,
    auth_type: AuthType,
) -> None:
    _configure_expectations(
        monkeypatch,
        audience=EXPECTED_AUDIENCE,
        issuer=EXPECTED_ISSUER,
        key_url=published_keys.jwks_url,
    )
    _arm_authenticated_user_policy(monkeypatch, auth_type)
    directory = UserDirectory(users={ADMIN_EMAIL: _user(ADMIN_EMAIL)})
    _install_directory(monkeypatch, directory)
    token = _sign(
        published_keys.private_key,
        _claims(aud=OTHER_AUDIENCE, iss=OTHER_ISSUER, email="Admin@Example.com"),
    )

    session_user = await _authenticated_user(token)

    assert directory.created_emails == []
    assert session_user is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim_name",
    ["email", "preferred_username", "upn"],
)
async def test_foreign_token_does_not_provision_user(
    monkeypatch: pytest.MonkeyPatch,
    published_keys: PublishedKeys,
    claim_name: str,
) -> None:
    _configure_expectations(
        monkeypatch,
        audience=EXPECTED_AUDIENCE,
        issuer=EXPECTED_ISSUER,
        key_url=published_keys.jwks_url,
    )
    _arm_authenticated_user_policy(monkeypatch, AuthType.OIDC)
    directory = UserDirectory()
    _install_directory(monkeypatch, directory)
    token = _sign(
        published_keys.private_key,
        _claims(
            **{
                "email": _OMIT,
                "preferred_username": _OMIT,
                "upn": _OMIT,
                claim_name: OUTSIDER_EMAIL,
                "aud": OTHER_AUDIENCE,
            }
        ),
    )

    session_user = await _authenticated_user(token)

    assert directory.created_emails == []
    assert OUTSIDER_EMAIL not in directory.users
    assert session_user is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim_name",
    ["email", "preferred_username", "upn"],
)
async def test_unverified_email_claim_does_not_authenticate(
    monkeypatch: pytest.MonkeyPatch,
    published_keys: PublishedKeys,
    claim_name: str,
) -> None:
    _configure_expectations(
        monkeypatch,
        audience=EXPECTED_AUDIENCE,
        issuer=EXPECTED_ISSUER,
        key_url=published_keys.jwks_url,
    )
    _arm_authenticated_user_policy(monkeypatch, AuthType.OIDC)
    directory = UserDirectory(
        users={
            ADMIN_EMAIL: _user(ADMIN_EMAIL),
            OTHER_EMAIL: _user(OTHER_EMAIL),
        }
    )
    _install_directory(monkeypatch, directory)
    token = _sign(
        published_keys.private_key,
        _claims(
            **{
                "email": _OMIT,
                "preferred_username": _OMIT,
                "upn": _OMIT,
                claim_name: ADMIN_EMAIL,
                "email_verified": False,
            }
        ),
    )

    session_user = await _authenticated_user(token)

    assert directory.created_emails == []
    assert session_user is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim_name",
    ["email", "preferred_username", "upn"],
)
async def test_missing_email_verified_does_not_authenticate(
    monkeypatch: pytest.MonkeyPatch,
    published_keys: PublishedKeys,
    claim_name: str,
) -> None:
    _configure_expectations(
        monkeypatch,
        audience=EXPECTED_AUDIENCE,
        issuer=EXPECTED_ISSUER,
        key_url=published_keys.jwks_url,
    )
    _arm_authenticated_user_policy(monkeypatch, AuthType.SAML)
    directory = UserDirectory(users={ADMIN_EMAIL: _user(ADMIN_EMAIL)})
    _install_directory(monkeypatch, directory)
    token = _sign(
        published_keys.private_key,
        _claims(
            **{
                "email": _OMIT,
                "preferred_username": _OMIT,
                "upn": _OMIT,
                claim_name: ADMIN_EMAIL,
                "email_verified": _OMIT,
            }
        ),
    )

    session_user = await _authenticated_user(token)

    assert directory.created_emails == []
    assert session_user is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim_name",
    ["email", "preferred_username", "upn"],
)
async def test_verified_token_logs_in_as_the_named_existing_user(
    monkeypatch: pytest.MonkeyPatch,
    published_keys: PublishedKeys,
    claim_name: str,
) -> None:
    _configure_expectations(
        monkeypatch,
        audience=EXPECTED_AUDIENCE,
        issuer=EXPECTED_ISSUER,
        key_url=published_keys.jwks_url,
    )
    _arm_authenticated_user_policy(monkeypatch, AuthType.OIDC)
    admin = _user(ADMIN_EMAIL)
    other = _user(OTHER_EMAIL)
    directory = UserDirectory(users={ADMIN_EMAIL: admin, OTHER_EMAIL: other})
    _install_directory(monkeypatch, directory)
    token = _sign(
        published_keys.private_key,
        _claims(
            **{
                "email": _OMIT,
                "preferred_username": _OMIT,
                "upn": _OMIT,
                claim_name: ADMIN_EMAIL,
                "email_verified": True,
            }
        ),
    )

    session_user = await _authenticated_user(token)

    assert session_user is admin
    assert session_user.email == ADMIN_EMAIL
    assert directory.created_emails == []


@pytest.mark.asyncio
async def test_verified_token_for_unknown_email_can_provision(
    monkeypatch: pytest.MonkeyPatch, published_keys: PublishedKeys
) -> None:
    """A token that really is for this app may still open a new account.

    The cross-app token must not. This locks the rejection tests to the
    audience and issuer, rather than to disabling bearer login entirely.
    """
    _configure_expectations(
        monkeypatch,
        audience=EXPECTED_AUDIENCE,
        issuer=EXPECTED_ISSUER,
        key_url=published_keys.jwks_url,
    )
    _arm_authenticated_user_policy(monkeypatch, AuthType.OIDC)
    directory = UserDirectory()
    _install_directory(monkeypatch, directory)
    token = _sign(
        published_keys.private_key,
        _claims(email=OUTSIDER_EMAIL, email_verified=True),
    )

    session_user = await _authenticated_user(token)

    assert session_user.email == OUTSIDER_EMAIL
    assert directory.created_emails == [OUTSIDER_EMAIL]
    assert directory.created_verified == [True]
