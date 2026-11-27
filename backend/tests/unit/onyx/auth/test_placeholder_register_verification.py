"""Public signup must not verify a placeholder account from the request body.

Placeholder BOT (Slack) and EXT_PERM_USER accounts are upgraded to standard
web accounts when POST /auth/register is called for their email. The upgrade
must not treat a client-supplied is_verified flag as proof of inbox ownership.
With REQUIRE_EMAIL_VERIFICATION enabled, the upgraded account has to stay
unverifiable through current_user until a real verification step says so.

These tests fail while the upgrade copies is_verified from the register body.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_users.password import PasswordHelper
from sqlalchemy.exc import IntegrityError

from onyx.auth.disposable_email_validator import _validator
from onyx.auth.schemas import UserCreate
from onyx.auth.schemas import UserRead
from onyx.auth.schemas import UserRole
from onyx.auth.users import auth_backend
from onyx.auth.users import current_user
from onyx.auth.users import fastapi_users
from onyx.auth.users import get_user_manager
from onyx.auth.users import TenantAwareRedisStrategy
from onyx.auth.users import UserManager
from onyx.configs.constants import AuthType
from onyx.db.enums import AccountType
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.db.models import UserGroup
from onyx.server.utils import BasicAuthenticationError

# Obvious placeholders. Not credentials for any real account.
_ORIGINAL_PASSWORD = "placeholder-original"
_REGISTER_PASSWORD = "placeholder-password"
_SESSION_TOKEN = "placeholder-session"
_EMAIL_DOMAIN = "company.example"


class _QueryResult:
    def __init__(self, user: User | None) -> None:
        self._user = user

    def all(self) -> list[object]:
        return []

    def unique(self) -> _QueryResult:
        return self

    def scalar_one_or_none(self) -> User | None:
        return self._user


class _AccountState:
    def __init__(self, user: User) -> None:
        self.user = user
        self.default_group = SimpleNamespace(id=1, name="Basic")


class _Savepoint:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _SyncSession:
    """Sync session stand-in for the writes the upgrade path performs.

    Query results follow the calls in get_user_by_email,
    _upgrade_user_to_standard__sync, assign_user_to_default_groups__no_commit,
    and the KV-store load used by invite cleanup. Commit keeps attribute
    changes; leaving the block without commit restores them, matching a
    transaction rollback.
    """

    def __init__(self, state: _AccountState) -> None:
        self.state = state
        self.committed = False
        self._snapshot: tuple[bool, str, UserRole, AccountType] | None = None

    def capture(self) -> None:
        user = self.state.user
        self._snapshot = (
            user.is_verified,
            user.hashed_password,
            user.role,
            user.account_type,
        )

    def restore(self) -> None:
        if self._snapshot is None:
            return
        user = self.state.user
        (
            user.is_verified,
            user.hashed_password,
            user.role,
            user.account_type,
        ) = self._snapshot

    def query(self, model: type[object]) -> _SyncQuery:
        return _SyncQuery(self, model)

    def begin_nested(self) -> _Savepoint:
        return _Savepoint()

    def add(self, _obj: object) -> None:
        return None

    def flush(self) -> None:
        return None

    def execute(self, _statement: object) -> _QueryResult:
        return _QueryResult(None)

    def commit(self) -> None:
        self.committed = True


class _SyncQuery:
    def __init__(self, session: _SyncSession, model: type[object]) -> None:
        self._session = session
        self._model = model

    def filter(self, *_args: object, **_kwargs: object) -> _SyncQuery:
        return self

    def filter_by(self, **_kwargs: object) -> _SyncQuery:
        return self

    def first(self) -> object | None:
        if self._model is User:
            return self._session.state.user
        if self._model is UserGroup:
            return self._session.state.default_group
        if self._model is User__UserGroup:
            return None
        return None


class _AsyncSession:
    def __init__(self, state: _AccountState) -> None:
        self.state = state

    async def execute(self, _statement: object) -> _QueryResult:
        return _QueryResult(self.state.user)

    def add(self, _obj: object) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def refresh(self, _obj: object) -> None:
        return None

    async def rollback(self) -> None:
        return None


class _AsyncSessionContext:
    def __init__(self, state: _AccountState) -> None:
        self._state = state

    async def __aenter__(self) -> _AsyncSession:
        return _AsyncSession(self._state)

    async def __aexit__(
        self,
        _exc_type: object,
        _exc: object,
        _tb: object,
    ) -> bool:
        return False


class _PlaceholderUserDB:
    """User DB that either finds the placeholder or hits the duplicate-insert race."""

    def __init__(self, state: _AccountState, *, race: bool) -> None:
        self.state = state
        self.race = race
        self.lookups = 0
        self.session = SimpleNamespace(
            expire=lambda _user: None,
            rollback=AsyncMock(),
        )

    async def get_by_email(self, email: str) -> User | None:
        self.lookups += 1
        if self.race and self.lookups == 1:
            return None
        if self.state.user.email.lower() == email.lower():
            return self.state.user
        return None

    async def get(self, user_id: uuid.UUID) -> User | None:
        if self.state.user.id == user_id:
            return self.state.user
        return None

    async def create(self, _create_dict: dict[str, Any]) -> User:
        if self.race:
            raise IntegrityError("INSERT", {}, Exception("duplicate email"))
        raise AssertionError("register tried to insert a second row for this email")


@contextmanager
def _sync_sessions(state: _AccountState) -> Generator[_SyncSession, None, None]:
    session = _SyncSession(state)
    session.capture()
    try:
        yield session
    except Exception:
        if not session.committed:
            session.restore()
        raise
    else:
        if not session.committed:
            session.restore()


def _async_sessions(state: _AccountState) -> Any:
    def _open(_tenant_id: str | None = None) -> _AsyncSessionContext:
        return _AsyncSessionContext(state)

    return _open


async def _write_token_without_redis(_self: object, _user: User) -> str:
    return _SESSION_TOKEN


def _placeholder_user(email: str, account_type: AccountType, role: UserRole) -> User:
    helper = PasswordHelper()
    return User(
        id=uuid.uuid4(),
        email=email,
        hashed_password=helper.hash(_ORIGINAL_PASSWORD),
        is_active=True,
        is_superuser=False,
        is_verified=False,
        role=role,
        account_type=account_type,
    )


def _register_app(manager: UserManager) -> FastAPI:
    """Mount the same register and login routers main.py includes for basic auth."""
    application = FastAPI()
    application.include_router(
        fastapi_users.get_register_router(UserRead, UserCreate),
        prefix="/auth",
    )
    application.include_router(
        fastapi_users.get_auth_router(auth_backend),
        prefix="/auth",
    )

    async def _override_user_manager() -> Any:
        yield manager

    application.dependency_overrides[get_user_manager] = _override_user_manager
    return application


@pytest.fixture(autouse=True)
def _disposable_domain_list_without_network() -> None:
    """Use the in-repo fallback list so domain checks do not fetch the network."""
    _validator._domains = set(_validator._fallback_domains)
    _validator._last_fetch_time = time.time()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("account_type", "role", "race"),
    [
        (AccountType.BOT, UserRole.SLACK_USER, False),
        (AccountType.EXT_PERM_USER, UserRole.EXT_PERM_USER, False),
        (AccountType.BOT, UserRole.SLACK_USER, True),
        (AccountType.EXT_PERM_USER, UserRole.EXT_PERM_USER, True),
    ],
    ids=[
        "slack_placeholder",
        "ext_perm_placeholder",
        "slack_placeholder_integrity_race",
        "ext_perm_placeholder_integrity_race",
    ],
)
async def test_public_register_cannot_verify_placeholder(
    account_type: AccountType,
    role: UserRole,
    race: bool,
) -> None:
    email = f"victim-{uuid.uuid4().hex[:8]}@{_EMAIL_DOMAIN}"
    user = _placeholder_user(email, account_type, role)
    state = _AccountState(user)
    manager = UserManager(_PlaceholderUserDB(state, race=race))
    application = _register_app(manager)
    body = {
        "email": email,
        "password": _REGISTER_PASSWORD,
        "is_verified": True,
    }

    with (
        patch("onyx.auth.users.MULTI_TENANT", False),
        patch("onyx.auth.users.AUTH_TYPE", AuthType.BASIC),
        patch("onyx.auth.users.REQUIRE_EMAIL_VERIFICATION", True),
        patch("onyx.auth.users.VALID_EMAIL_DOMAINS", [_EMAIL_DOMAIN]),
        patch("onyx.auth.users.get_user_count", new_callable=AsyncMock) as user_count,
        patch("onyx.auth.users.get_session_with_current_tenant", _sync_sessions_for(state)),
        patch(
            "onyx.key_value_store.store.get_session_with_current_tenant",
            _sync_sessions_for(state),
        ),
        patch(
            "onyx.auth.users.get_async_session_context_manager",
            _async_sessions(state),
        ),
        patch.object(TenantAwareRedisStrategy, "write_token", _write_token_without_redis),
    ):
        user_count.return_value = 2
        client = TestClient(application)
        response = client.post("/auth/register", json=body)

        assert response.status_code in (201, 400), response.status_code
        if response.status_code == 201:
            login = client.post(
                "/auth/login",
                data={"username": email, "password": _REGISTER_PASSWORD},
            )
            assert login.status_code == 204, login.status_code
        else:
            helper = PasswordHelper()
            attacker_matches, _updated = helper.verify_and_update(_REGISTER_PASSWORD, user.hashed_password)
            original_matches, _original_updated = helper.verify_and_update(_ORIGINAL_PASSWORD, user.hashed_password)
            assert attacker_matches is False
            assert original_matches is True
            assert user.account_type == account_type

        granted = False
        try:
            await current_user(user)
            granted = True
        except BasicAuthenticationError:
            granted = False

        assert granted is False
        assert user.is_verified is False
        if response.status_code == 201:
            assert response.json()["is_verified"] is False


def _sync_sessions_for(state: _AccountState) -> Any:
    @contextmanager
    def _open() -> Generator[_SyncSession, None, None]:
        with _sync_sessions(state) as session:
            yield session

    return _open
