"""Account pre-hijack: OAuth email linking must not keep an attacker's password.

These tests are the spec for the fix. They fail while the current
``UserManager.oauth_callback`` links a Google/OIDC identity onto an unverified
password account and leaves that password in place.

A fix passes when, after the OAuth callback for an unverified password account,
either:

* the callback does not log the victim into that account and does not attach
  the OAuth identity to it, or
* the callback logs the victim into that account only after the attacker's
  password no longer authenticates and the account is marked verified.

Verified local accounts may still be linked by email, and the password the
owner set must keep working.

Not asserted here: revocation of sessions that were already issued before the
callback. The password re-login in the attack is what these tests cover.
``SingleTenantJWTStrategy.destroy_token`` is a no-op, so a takeover fix must
not rely on JWT revocation alone.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_users import exceptions
from fastapi_users.authentication import AuthenticationBackend
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseOAuthAccountTableUUID
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID
from httpx import ASGITransport
from httpx_oauth.clients.google import GoogleOAuth2
from sqlalchemy import create_engine
from sqlalchemy import Enum
from sqlalchemy import func
from sqlalchemy import Text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session
from sqlalchemy.orm import validates
from starlette.requests import Request

from onyx.auth.disposable_email_validator import _validator as disposable_validator
from onyx.auth.schemas import UserCreate
from onyx.auth.schemas import UserRole
from onyx.auth.users import cookie_transport
from onyx.auth.users import create_onyx_oauth_router
from onyx.auth.users import CSRF_TOKEN_COOKIE_NAME
from onyx.auth.users import CSRF_TOKEN_KEY
from onyx.auth.users import generate_csrf_token
from onyx.auth.users import generate_state_token
from onyx.auth.users import get_user_manager
from onyx.auth.users import UserManager
from onyx.configs.app_configs import WEB_DOMAIN
from onyx.db.auth import SQLAlchemyUserAdminDB
from onyx.db.enums import AccountType
from onyx.key_value_store.interface import KvKeyNotFoundError
from onyx.key_value_store.store import PgRedisKVStore
from onyx.utils.variable_functionality import fetch_versioned_implementation
from onyx.utils.variable_functionality import global_version

# Obvious placeholders. These are not credentials from any deployment.
ATTACKER_PASSWORD = "placeholder-attacker-password"
OAUTH_CLIENT_ID = "placeholder-oauth-client-id"
OAUTH_CLIENT_SECRET = "placeholder-oauth-client-secret"
STATE_SECRET = "placeholder-oauth-state-secret-value"
GOOGLE_ACCESS_TOKEN = "placeholder-google-access-token"
GOOGLE_REFRESH_TOKEN = "placeholder-google-refresh-token"
GOOGLE_SUBJECT = "people/placeholder-google-subject"
REGISTERED_EMAIL = "Victim@Example.com"
# Different case and a non-primary address, so the real Google profile parser
# (httpx-oauth 0.15.1 GoogleOAuth2.get_id_email) has to pick the primary value
# and the product lookup has to match it case-insensitively.
GOOGLE_PRIMARY_EMAIL = "VICTIM@example.com"
STORED_EMAIL = "victim@example.com"


class _Base(DeclarativeBase):
    pass


class _OAuthAccount(SQLAlchemyBaseOAuthAccountTableUUID, _Base):
    """Mirrors onyx.db.models.OAuthAccount token columns used by the callback."""

    access_token: Mapped[str] = mapped_column(Text, nullable=False)  # type: ignore[assignment]
    refresh_token: Mapped[str] = mapped_column(Text, nullable=False)  # type: ignore[assignment]


class _User(SQLAlchemyBaseUserTableUUID, _Base):
    """Columns UserManager.oauth_callback and authenticate read on a web user.

    Email lowercasing matches ``User.validate_email`` in ``onyx.db.models``.
    """

    oauth_accounts: Mapped[list[_OAuthAccount]] = relationship("_OAuthAccount", lazy="joined", cascade="all, delete-orphan")
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, native_enum=False),
        nullable=False,
        default=UserRole.BASIC,
    )
    account_type: Mapped[AccountType] = mapped_column(
        Enum(AccountType, native_enum=False),
        nullable=False,
        default=AccountType.STANDARD,
    )
    oidc_expiry: Mapped[datetime | None] = mapped_column(nullable=True, default=None)

    @validates("email")
    def validate_email(self, key: str, value: str) -> str:  # noqa: ARG002
        return value.lower() if value else value


@dataclass
class _Harness:
    session: AsyncSession
    make_manager: Callable[[], UserManager]


def _empty_cache() -> Any:
    class _Cache:
        def get(self, _key: str) -> None:
            return None

        def set(self, _key: str, _value: str, _ex: int | None = None) -> None:
            return None

    return _Cache()


def _install_io_stubs(monkeypatch: pytest.MonkeyPatch, session: AsyncSession, db_path: Path) -> None:
    """Keep the auth code on the sqlite session instead of Postgres or Redis.

    Invite lookup, seat checks, and disposable-domain refresh are left on their
    real functions. The KV store answers as a fresh install (key missing), which
    is the default invite-only-off settings path.
    """
    import onyx.auth.users as users_module
    import onyx.db.auth as auth_db_module
    import onyx.server.settings.store as settings_store

    global_version.unset_ee()
    fetch_versioned_implementation.cache_clear()

    disposable_validator._domains = set(disposable_validator._fallback_domains)
    disposable_validator._last_fetch_time = 10**12

    sync_engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    def get_user_by_email(email: str, db_session: Session) -> _User | None:  # noqa: ARG001
        # Same comparison as onyx.db.users.get_user_by_email.
        with Session(sync_engine) as sync_session:
            return sync_session.query(_User).filter(func.lower(_User.email) == func.lower(email)).one_or_none()

    @contextmanager
    def sync_session_cm(*_args: Any, **_kwargs: Any) -> Iterator[Any]:
        class _Sync:
            def commit(self) -> None:
                return None

        yield _Sync()

    @asynccontextmanager
    async def async_session_cm(_tenant_id: str | None = None) -> AsyncIterator[AsyncSession]:
        yield session

    async def user_count(only_admin_users: bool = False) -> int:  # noqa: ARG001
        # Deployment already has users, so this registration is not the first admin.
        return 5

    def load_missing(self: PgRedisKVStore, key: str, refresh_cache: bool = False) -> Any:  # noqa: ARG001
        raise KvKeyNotFoundError

    monkeypatch.setattr(users_module, "User", _User)
    monkeypatch.setattr(users_module, "get_user_by_email", get_user_by_email)
    monkeypatch.setattr(users_module, "get_session_with_tenant", sync_session_cm)
    monkeypatch.setattr(users_module, "get_session_with_current_tenant", sync_session_cm)
    monkeypatch.setattr(users_module, "get_async_session_context_manager", async_session_cm)
    monkeypatch.setattr(users_module, "get_user_count", user_count)

    def skip_default_groups(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        users_module,
        "assign_user_to_default_groups__no_commit",
        skip_default_groups,
    )
    monkeypatch.setattr(auth_db_module, "get_user_count", user_count)
    monkeypatch.setattr(settings_store, "get_cache_backend", _empty_cache)
    monkeypatch.setattr(PgRedisKVStore, "load", load_missing)


@pytest_asyncio.fixture
async def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_Harness]:
    db_path = tmp_path / "auth.sqlite"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as connection:
        await connection.run_sync(_Base.metadata.create_all)
    session = AsyncSession(engine, expire_on_commit=False)
    _install_io_stubs(monkeypatch, session, db_path)

    def make_manager() -> UserManager:
        user_db = SQLAlchemyUserAdminDB(session, _User, _OAuthAccount)
        manager = UserManager(user_db)  # type: ignore[arg-type]

        async def no_featured_personas(user: _User, db_session: AsyncSession) -> None:  # noqa: ARG001
            # Production returns immediately when no featured persona rows exist.
            return None

        manager._assign_default_pinned_assistants = no_featured_personas  # type: ignore[method-assign]
        return manager

    try:
        yield _Harness(session=session, make_manager=make_manager)
    finally:
        await session.close()
        await engine.dispose()


def _registration_request() -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/auth/register",
            "raw_path": b"/auth/register",
            "query_string": b"",
            "headers": [],
            "client": ("192.0.2.10", 12345),
            "server": ("test", 80),
        }
    )


async def _register_unverified(harness: _Harness, *, claim_verified: bool) -> _User:
    """POST /auth/register calls UserManager.create(..., safe=True)."""
    manager = harness.make_manager()
    created = await manager.create(
        UserCreate(
            email=REGISTERED_EMAIL,
            password=ATTACKER_PASSWORD,
            is_verified=claim_verified,
        ),
        safe=True,
        request=_registration_request(),
    )
    return created  # type: ignore[return-value]


def _password_form(email: str) -> OAuth2PasswordRequestForm:
    return OAuth2PasswordRequestForm(
        grant_type="password",
        username=email,
        password=ATTACKER_PASSWORD,
        scope="",
    )


async def _attacker_login(harness: _Harness) -> _User | None:
    harness.session.expire_all()
    manager = harness.make_manager()
    user = await manager.authenticate(_password_form(GOOGLE_PRIMARY_EMAIL))
    return user  # type: ignore[return-value]


async def _linked_user(harness: _Harness, oauth_name: str, account_id: str) -> _User | None:
    harness.session.expire_all()
    manager = harness.make_manager()
    try:
        user = await manager.get_by_oauth_account(oauth_name, account_id)
    except exceptions.UserNotExists:
        return None
    return user  # type: ignore[return-value]


async def _assert_unverified_link_is_closed(
    harness: _Harness,
    preregistered_id: uuid.UUID,
    oauth_name: str,
    account_id: str,
    logged_in_user: _User | None,
) -> None:
    """The victim's login must not be an account the attacker's password still opens."""
    authenticated = await _attacker_login(harness)
    linked = await _linked_user(harness, oauth_name, account_id)

    if logged_in_user is not None:
        assert authenticated is None or authenticated.id != logged_in_user.id, (
            "OAuth callback authenticated the victim into an account that still accepts the password from open registration"
        )

    if linked is not None and linked.id == preregistered_id:
        assert authenticated is None, (
            "OAuth identity was attached to the pre-registered account and the registration password still authenticates"
        )
        harness.session.expire_all()
        fresh = await harness.session.get(_User, preregistered_id)
        assert fresh is not None
        assert fresh.is_verified is True


class _ScriptedGoogleOAuth2(GoogleOAuth2):
    """Real Google client methods, with the provider HTTP responses scripted.

    ``get_access_token`` and ``get_id_email`` are the httpx-oauth 0.15.1
    implementations. Only the network responses are substituted.
    """

    def __init__(self, profile: dict[str, Any]) -> None:
        super().__init__(OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET)
        self._profile = profile

    def get_httpx_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._respond))

    def _respond(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "oauth2.googleapis.com":
            return httpx.Response(
                200,
                json={
                    "access_token": GOOGLE_ACCESS_TOKEN,
                    "refresh_token": GOOGLE_REFRESH_TOKEN,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )
        if request.url.host == "people.googleapis.com":
            return httpx.Response(200, json=self._profile)
        raise AssertionError(f"unexpected OAuth request {request.method} {request.url}")


class _RecordingStrategy:
    def __init__(self) -> None:
        self.logged_in_user: _User | None = None

    async def read_token(self, token: str | None, user_manager: UserManager) -> _User | None:  # noqa: ARG002
        return None

    async def write_token(self, user: _User) -> str:
        self.logged_in_user = user
        return "placeholder-session-token"

    async def destroy_token(self, token: str, user: _User) -> None:  # noqa: ARG002
        return None


def _google_profile() -> dict[str, Any]:
    return {
        "resourceName": GOOGLE_SUBJECT,
        "emailAddresses": [
            {"value": "other@example.com", "metadata": {"primary": False}},
            {"value": GOOGLE_PRIMARY_EMAIL, "metadata": {"primary": True}},
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("oauth_name", ["google", "openid", "custom"])
async def test_oauth_callback_does_not_keep_attacker_password_on_unverified_account(
    harness: _Harness,
    oauth_name: str,
) -> None:
    """Registration then oauth_callback(associate_by_email=True).

    ``google`` is the basic+Google and cloud routers. ``openid`` is the OIDC
    router. ``custom`` is the enterprise refresh-token handler. All three call
    ``UserManager.oauth_callback`` with ``associate_by_email=True`` and
    ``is_verified_by_default=True``.
    """
    created = await _register_unverified(harness, claim_verified=True)
    assert created.is_verified is False
    assert created.email == STORED_EMAIL
    assert created.account_type == AccountType.STANDARD
    assert (await _attacker_login(harness)) is not None
    preregistered_id = created.id

    manager = harness.make_manager()
    try:
        logged_in = await manager.oauth_callback(
            oauth_name,
            GOOGLE_ACCESS_TOKEN,
            GOOGLE_SUBJECT,
            GOOGLE_PRIMARY_EMAIL,
            None,
            GOOGLE_REFRESH_TOKEN,
            _registration_request(),
            associate_by_email=True,
            is_verified_by_default=True,
        )
    except Exception:
        logged_in = None

    await _assert_unverified_link_is_closed(
        harness,
        preregistered_id,
        oauth_name,
        GOOGLE_SUBJECT,
        logged_in,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_verified_password_account_can_link_google_and_keep_its_password(
    harness: _Harness,
) -> None:
    """Linking by email stays allowed once the local account is actually verified."""
    created = await _register_unverified(harness, claim_verified=False)
    manager = harness.make_manager()
    await manager.user_db.update(created, {"is_verified": True})  # type: ignore[arg-type]

    linked = await manager.oauth_callback(
        "google",
        GOOGLE_ACCESS_TOKEN,
        GOOGLE_SUBJECT,
        GOOGLE_PRIMARY_EMAIL,
        None,
        GOOGLE_REFRESH_TOKEN,
        _registration_request(),
        associate_by_email=True,
        is_verified_by_default=True,
    )
    assert linked.id == created.id
    authenticated = await _attacker_login(harness)
    assert authenticated is not None
    assert authenticated.id == created.id
    attached = await _linked_user(harness, "google", GOOGLE_SUBJECT)
    assert attached is not None
    assert attached.id == created.id


@pytest.mark.asyncio
async def test_google_oauth_callback_route_does_not_preserve_attacker_password(
    harness: _Harness,
) -> None:
    """Drive /auth/oauth/callback the way basic auth mounts it in onyx.main.

    Token exchange and the People API profile parse run through GoogleOAuth2.
    The router is built with associate_by_email=True and is_verified_by_default=True,
    matching the basic, Google-only, cloud, and OIDC call sites.
    """
    created = await _register_unverified(harness, claim_verified=True)
    assert created.is_verified is False
    assert (await _attacker_login(harness)) is not None

    strategy = _RecordingStrategy()

    async def get_strategy() -> _RecordingStrategy:
        return strategy

    backend = AuthenticationBackend(
        name="redis",
        transport=cookie_transport,
        get_strategy=get_strategy,  # type: ignore[arg-type]
    )
    oauth_client = _ScriptedGoogleOAuth2(_google_profile())
    router = create_onyx_oauth_router(
        oauth_client,
        backend,
        STATE_SECRET,
        associate_by_email=True,
        is_verified_by_default=True,
        redirect_url=f"{WEB_DOMAIN}/auth/oauth/callback",
    )
    callback_manager = harness.make_manager()

    async def override_user_manager() -> AsyncIterator[UserManager]:
        yield callback_manager

    app = FastAPI()
    app.include_router(router, prefix="/auth/oauth")
    app.dependency_overrides[get_user_manager] = override_user_manager

    csrf_token = generate_csrf_token()
    state = generate_state_token(
        {
            "next_url": "/",
            "referral_source": "default_referral",
            CSRF_TOKEN_KEY: csrf_token,
        },
        STATE_SECRET,
    )
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/auth/oauth/callback",
            params={"code": "placeholder-auth-code", "state": state},
            cookies={CSRF_TOKEN_COOKIE_NAME: csrf_token},
        )

    logged_in = strategy.logged_in_user if response.status_code == 302 else None
    await _assert_unverified_link_is_closed(
        harness,
        created.id,
        "google",
        GOOGLE_SUBJECT,
        logged_in,
    )
