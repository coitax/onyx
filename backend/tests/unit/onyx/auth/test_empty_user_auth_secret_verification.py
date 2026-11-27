"""Email verification and password reset must not accept the default signing key.

USER_AUTH_SECRET defaults to an empty string. These tests post tokens signed
with that default through the same verify and reset routes the API mounts.
They fail while the server still treats that default as a valid HMAC key.
"""

import asyncio
import secrets
import uuid
from collections.abc import AsyncIterator
from collections.abc import Iterator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from typing import Any

import jwt
import pytest
from fastapi import FastAPI
from fastapi_users.password import PasswordHelper
from fastapi.testclient import TestClient

from onyx.auth.schemas import UserCreate
from onyx.auth.schemas import UserRead
from onyx.auth.schemas import UserRole
from onyx.auth.users import double_check_user
from onyx.auth.users import fastapi_users
from onyx.auth.users import get_user_manager
from onyx.auth.users import UserManager
from onyx.auth.users import verify_email_domain
from onyx.configs.app_configs import USER_AUTH_SECRET
from onyx.configs.constants import AuthType
from onyx.error_handling.exceptions import OnyxError
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.server.utils import BasicAuthenticationError
from shared_configs.configs import MULTI_TENANT

import onyx.auth.users as users_module

_COMPANY_DOMAIN = "example.com"
_COMPANY_EMAIL = "someone@example.com"
_REPLACEMENT_PASSWORD = "replacement-password"


class _Account:
    """In-memory user row. Email lookup is case-insensitive, matching the
    SQLAlchemy adapter the verify path calls.
    """

    def __init__(self, email: str) -> None:
        self.id: uuid.UUID = uuid.uuid4()
        self.email: str = email
        # Runtime hash of a placeholder password. The value is not a credential.
        self.hashed_password: str = PasswordHelper().hash("placeholder-password")
        self.is_active: bool = True
        self.is_verified: bool = False
        self.is_superuser: bool = False
        self.oidc_expiry: None = None
        self.role: UserRole = UserRole.BASIC


class _MemoryUserDB:
    def __init__(self, user: _Account) -> None:
        self.user = user

    async def get_by_email(self, email: str) -> _Account | None:
        if self.user.email.casefold() == email.casefold():
            return self.user
        return None

    async def get(self, user_id: uuid.UUID) -> _Account | None:
        if self.user.id == user_id:
            return self.user
        return None

    async def update(self, user: _Account, update_dict: dict[str, Any]) -> _Account:
        for field, value in update_dict.items():
            setattr(user, field, value)
        return user


def _idle_session(_tenant_id: str | None = None) -> Any:
    @asynccontextmanager
    async def _ctx() -> AsyncIterator[None]:
        yield None

    return _ctx()


def _require_default_basic_auth() -> str:
    """The claimed deployment: basic auth and the unset signing secret."""
    assert not MULTI_TENANT
    assert users_module.AUTH_TYPE == AuthType.BASIC
    assert USER_AUTH_SECRET == ""
    assert UserManager.verification_token_secret == USER_AUTH_SECRET
    assert UserManager.reset_password_token_secret == USER_AUTH_SECRET
    return USER_AUTH_SECRET


def _forge(payload: dict[str, Any], signing_key: str) -> str:
    token = jwt.encode(payload, signing_key, algorithm="HS256")
    assert isinstance(token, str)
    return token


@pytest.fixture
def company_user(monkeypatch: pytest.MonkeyPatch) -> _Account:
    monkeypatch.setattr(users_module, "REQUIRE_EMAIL_VERIFICATION", True)
    monkeypatch.setattr(users_module, "VALID_EMAIL_DOMAINS", [_COMPANY_DOMAIN])
    monkeypatch.setattr(users_module, "is_disposable_email", lambda _email: False)
    monkeypatch.setattr(users_module, "get_async_session_context_manager", _idle_session)
    return _Account(_COMPANY_EMAIL)


@contextmanager
def _client(
    user: _Account,
    *,
    verification_secret: str | None = None,
    reset_secret: str | None = None,
    create_returns_user: bool = False,
) -> Iterator[TestClient]:
    database = _MemoryUserDB(user)
    application = FastAPI()
    register_onyx_exception_handlers(application)
    application.include_router(
        fastapi_users.get_register_router(UserRead, UserCreate),
        prefix="/auth",
    )
    application.include_router(
        fastapi_users.get_verify_router(UserRead),
        prefix="/auth",
    )
    application.include_router(
        fastapi_users.get_reset_password_router(),
        prefix="/auth",
    )

    async def _override_user_manager() -> AsyncIterator[UserManager]:
        manager = UserManager(database)  # type: ignore[arg-type]
        if verification_secret is not None:
            manager.verification_token_secret = verification_secret
        if reset_secret is not None:
            manager.reset_password_token_secret = reset_secret
        if create_returns_user:

            async def _create(
                user_create: UserCreate,  # noqa: ARG001
                safe: bool = False,  # noqa: ARG001
                request: Any = None,  # noqa: ARG001
            ) -> _Account:
                return user

            manager.create = _create  # type: ignore[method-assign]
        yield manager

    application.dependency_overrides[get_user_manager] = _override_user_manager
    with TestClient(application) as client:
        yield client


def test_register_response_returns_user_id_for_allowed_domain(
    company_user: _Account,
) -> None:
    """Registration of an address in the configured domain is accepted, and
    the public register route returns that user's id.
    """
    verify_email_domain(_COMPANY_EMAIL, is_registration=True)
    with pytest.raises(OnyxError):
        verify_email_domain("outsider@other.example", is_registration=True)

    with _client(company_user, create_returns_user=True) as client:
        response = client.post(
            "/auth/register",
            json={"email": _COMPANY_EMAIL, "password": _REPLACEMENT_PASSWORD},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == str(company_user.id)
    assert body["email"] == _COMPANY_EMAIL
    assert body["is_verified"] is False


def test_empty_default_secret_cannot_verify_email(company_user: _Account) -> None:
    """An outsider who only knows the id from registration cannot mark the
    account verified by signing a token with the default empty secret.
    """
    signing_key = _require_default_basic_auth()
    assert company_user.is_verified is False
    with pytest.raises(BasicAuthenticationError):
        asyncio.run(double_check_user(company_user))  # type: ignore[arg-type]

    token = _forge(
        {
            "sub": str(company_user.id),
            "email": company_user.email,
            "aud": UserManager.verification_token_audience,
        },
        signing_key,
    )

    with _client(company_user) as client:
        response = client.post("/auth/verify", json={"token": token})

    assert response.status_code >= 400, response.text
    assert company_user.is_verified is False
    with pytest.raises(BasicAuthenticationError):
        asyncio.run(double_check_user(company_user))  # type: ignore[arg-type]


def test_short_secret_cannot_verify_email(company_user: _Account) -> None:
    """A one-character signing key is not enough to verify an address."""
    _require_default_basic_auth()
    short_key = "x"
    token = _forge(
        {
            "sub": str(company_user.id),
            "email": company_user.email,
            "aud": UserManager.verification_token_audience,
        },
        short_key,
    )

    with _client(company_user, verification_secret=short_key) as client:
        response = client.post("/auth/verify", json={"token": token})

    assert response.status_code >= 400, response.text
    assert company_user.is_verified is False


def test_real_secret_still_verifies_email(company_user: _Account) -> None:
    """A long signing key issued by the server still verifies the account."""
    _require_default_basic_auth()
    signing_key = secrets.token_urlsafe(32)
    token = _forge(
        {
            "sub": str(company_user.id),
            "email": company_user.email,
            "aud": UserManager.verification_token_audience,
        },
        signing_key,
    )

    with _client(company_user, verification_secret=signing_key) as client:
        response = client.post("/auth/verify", json={"token": token})

    assert response.status_code == 200
    assert response.json()["is_verified"] is True
    assert company_user.is_verified is True
    verified = asyncio.run(double_check_user(company_user))  # type: ignore[arg-type]
    assert verified.email == company_user.email


def test_public_registration_data_cannot_reset_password(
    company_user: _Account,
) -> None:
    """The id and email from registration are not enough to reset the password."""
    signing_key = _require_default_basic_auth()
    original_hash = company_user.hashed_password
    # A well-formed fingerprint of a different value. The attacker does not
    # have the stored password hash, so they cannot produce the real one.
    decoy_fingerprint = PasswordHelper().hash("not-the-stored-hash")
    token = _forge(
        {
            "sub": str(company_user.id),
            "password_fgpt": decoy_fingerprint,
            "aud": UserManager.reset_password_token_audience,
        },
        signing_key,
    )

    with _client(company_user) as client:
        response = client.post(
            "/auth/reset-password",
            json={"token": token, "password": _REPLACEMENT_PASSWORD},
        )

    assert response.status_code >= 400, response.text
    assert company_user.hashed_password == original_hash


def test_empty_default_secret_cannot_reset_password_with_matching_fingerprint(
    company_user: _Account,
) -> None:
    """Knowing the stored password hash must not turn the empty default key
    into a password-reset capability.
    """
    signing_key = _require_default_basic_auth()
    original_hash = company_user.hashed_password
    manager = UserManager(_MemoryUserDB(company_user))  # type: ignore[arg-type]
    fingerprint = manager.password_helper.hash(original_hash)
    token = _forge(
        {
            "sub": str(company_user.id),
            "password_fgpt": fingerprint,
            "aud": UserManager.reset_password_token_audience,
        },
        signing_key,
    )

    with _client(company_user) as client:
        response = client.post(
            "/auth/reset-password",
            json={"token": token, "password": _REPLACEMENT_PASSWORD},
        )

    assert response.status_code >= 400, response.text
    assert company_user.hashed_password == original_hash
