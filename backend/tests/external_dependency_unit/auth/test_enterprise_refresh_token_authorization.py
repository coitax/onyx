"""Authorization tests for POST /enterprise-settings/refresh-token.

The endpoint is mounted on the enterprise app and accepts any signed-in user,
including one whose OIDC expiry has already passed. These tests call that route
with a real session cookie and the real user manager. They fail while the
handler trusts access tokens, refresh tokens, expiry, user id, and email from
the body. They pass once a refresh is limited to an OAuth account already
linked to the caller, or the route is removed.

Run from backend/ with Postgres available and the JWT auth backend selected
before pytest imports application config:

    AUTH_BACKEND=jwt \\
    USER_AUTH_SECRET=placeholder-user-auth-secret-value \\
    DISPOSABLE_EMAIL_DOMAINS_URL= \\
    pytest -xv tests/external_dependency_unit/auth/test_enterprise_refresh_token_authorization.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import Any
from uuid import UUID
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_users.password import PasswordHelper
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from ee.onyx.server.enterprise_settings.api import basic_router
from ee.onyx.server.tenants import provisioning
from ee.onyx.server.tenants import user_mapping
from onyx.auth.schemas import AuthBackend
from onyx.auth.schemas import UserRole
from onyx.auth import users as users_module
from onyx.auth.users import double_check_user
from onyx.auth.users import get_jwt_strategy
from onyx.configs.app_configs import AUTH_BACKEND
from onyx.configs.constants import FASTAPI_USERS_AUTH_COOKIE_NAME
from onyx.db.engine import async_sql_engine
from onyx.db.engine import sql_engine as sync_sql_engine
from onyx.db.engine.sql_engine import SqlEngine
from onyx.db.enums import AccountType
from onyx.db.models import KVStore
from onyx.db.models import License
from onyx.db.models import OAuthAccount
from onyx.db.models import PermissionGrant
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.db.models import UserGroup
from onyx.db.models import UserTenantMapping
from onyx.server.utils import BasicAuthenticationError
from shared_configs import contextvars as tenant_contextvars

# Session expiry is milliseconds. Dividing by 1000 lands in 1970.
_PAST_EXPIRY_MS = 1000
_BODY_ACCESS_TOKEN = "body-access-token"
_BODY_REFRESH_TOKEN = "body-refresh-token"
_EXISTING_ACCESS_TOKEN = "existing-access-token"
_EXISTING_REFRESH_TOKEN = "existing-refresh-token"
_OTHER_TENANT_SCHEMA = "tenant_refresh_authz"

_REQUIRED_TABLES = (
    User.__table__,
    OAuthAccount.__table__,
    UserGroup.__table__,
    User__UserGroup.__table__,
    PermissionGrant.__table__,
    KVStore.__table__,
    License.__table__,
    UserTenantMapping.__table__,
)


@dataclass(frozen=True)
class StoredIdentity:
    email: str
    role: UserRole
    account_type: AccountType
    is_active: bool
    is_verified: bool
    oidc_expiry: datetime | None
    oauth_accounts: tuple[tuple[str, str, str, str, int | None], ...]
    group_ids: tuple[int, ...]


def _require_jwt_backend() -> None:
    if AUTH_BACKEND != AuthBackend.JWT:
        pytest.fail(
            "These tests sign the caller with the JWT auth backend. "
            "Start pytest with AUTH_BACKEND=jwt and "
            "USER_AUTH_SECRET=placeholder-user-auth-secret-value."
        )


def _engine() -> Engine:
    SqlEngine.init_engine(pool_size=5, max_overflow=2)
    return SqlEngine.get_engine()


def _ensure_tables(engine: Engine, schema: str | None = None) -> None:
    from onyx.db.models import Base

    # The mapping table is pinned to the public schema. Tenant schemas only
    # need the per-tenant user tables.
    tables = list(_REQUIRED_TABLES)
    if schema is not None:
        tables = [
            table
            for table in tables
            if table is not UserTenantMapping.__table__
        ]
    if schema is None:
        Base.metadata.create_all(engine, tables=tables)
        return
    with engine.begin() as connection:
        connection.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    with engine.connect().execution_options(
        schema_translate_map={None: schema}
    ) as connection:
        Base.metadata.create_all(connection, tables=tables)
        connection.commit()


def _session(engine: Engine, schema: str | None = None) -> Session:
    if schema is None:
        return Session(engine)
    connection = engine.connect().execution_options(schema_translate_map={None: schema})
    return Session(bind=connection)


def _insert_user(
    session: Session,
    email: str,
    role: UserRole,
    account_type: AccountType,
    *,
    oidc_expiry: datetime | None = None,
) -> User:
    password_helper = PasswordHelper()
    user = User(
        id=uuid4(),
        email=email,
        hashed_password=password_helper.hash(password_helper.generate()),
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role=role,
        account_type=account_type,
        oidc_expiry=oidc_expiry,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def _ensure_basic_group(session: Session) -> None:
    existing = (
        session.query(UserGroup)
        .filter(UserGroup.name == "Basic", UserGroup.is_default.is_(True))
        .one_or_none()
    )
    if existing is None:
        session.add(
            UserGroup(
                name="Basic",
                is_up_to_date=True,
                is_up_for_deletion=False,
                is_default=True,
            )
        )
        session.commit()


def _read_identity(
    engine: Engine, email: str, schema: str | None = None
) -> StoredIdentity | None:
    session = _session(engine, schema)
    try:
        user = session.query(User).filter(User.email == email).one_or_none()
        if user is None:
            return None
        accounts = (
            session.query(OAuthAccount)
            .filter(OAuthAccount.user_id == user.id)
            .order_by(OAuthAccount.account_id)
            .all()
        )
        groups = (
            session.query(User__UserGroup.user_group_id)
            .filter(User__UserGroup.user_id == user.id)
            .order_by(User__UserGroup.user_group_id)
            .all()
        )
        return StoredIdentity(
            email=user.email,
            role=user.role,
            account_type=user.account_type,
            is_active=user.is_active,
            is_verified=user.is_verified,
            oidc_expiry=user.oidc_expiry,
            oauth_accounts=tuple(
                (
                    account.oauth_name,
                    account.account_id,
                    account.access_token,
                    account.account_email,
                    account.expires_at,
                )
                for account in accounts
            ),
            group_ids=tuple(row[0] for row in groups),
        )
    finally:
        connection = session.connection()
        session.close()
        if schema is not None:
            connection.close()


def _refresh_body(email: str, user_id: str, exp_ms: int = _PAST_EXPIRY_MS) -> dict[str, Any]:
    return {
        "access_token": _BODY_ACCESS_TOKEN,
        "refresh_token": _BODY_REFRESH_TOKEN,
        "session": {"exp": exp_ms},
        "userinfo": {"userId": user_id, "email": email},
    }


def _assert_no_cross_account_effect(
    status_code: int,
    before: StoredIdentity | None,
    after: StoredIdentity | None,
) -> None:
    # State is checked first. A 500 from the handler's catch-all still fails
    # when the OAuth row, role, or expiry was committed before the exception.
    assert after == before
    assert status_code >= 400, status_code


def _session_cookie(user_id: UUID) -> str:
    return asyncio.run(get_jwt_strategy().write_token(User(id=user_id, email="")))


@asynccontextmanager
async def _dispose_async_engine_on_shutdown(
    _app: FastAPI,
) -> AsyncIterator[None]:
    yield
    engine = async_sql_engine._ASYNC_ENGINE
    if engine is not None:
        await engine.dispose()
        async_sql_engine._ASYNC_ENGINE = None


@pytest.fixture(scope="module")
def database_engine() -> Engine:
    _require_jwt_backend()
    engine = _engine()
    _ensure_tables(engine)
    _ensure_tables(engine, _OTHER_TENANT_SCHEMA)
    return engine


@pytest.fixture()
def enterprise_client(enable_ee: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # The enterprise server enables EE before serving this route, and the
    # lockout behavior is the TRACK_EXTERNAL_IDP_EXPIRY configuration.
    monkeypatch.setattr(
        "onyx.auth.users.TRACK_EXTERNAL_IDP_EXPIRY",
        True,
    )
    app = FastAPI(lifespan=_dispose_async_engine_on_shutdown)
    app.include_router(basic_router)
    with TestClient(app) as client:
        yield client


def _signed_in(client: TestClient, user_id: UUID) -> None:
    client.cookies.set(FASTAPI_USERS_AUTH_COOKIE_NAME, _session_cookie(user_id))


def test_anonymous_caller_cannot_name_another_account(
    database_engine: Engine,
    enterprise_client: TestClient,
) -> None:
    email = f"admin-{uuid4().hex[:8]}@example.com"
    session = _session(database_engine)
    try:
        _insert_user(session, email, UserRole.ADMIN, AccountType.STANDARD)
    finally:
        session.close()
    before = _read_identity(database_engine, email)

    response = enterprise_client.post(
        "/enterprise-settings/refresh-token",
        json=_refresh_body(email, f"anonymous-chosen-{uuid4().hex}"),
    )

    _assert_no_cross_account_effect(
        response.status_code,
        before,
        _read_identity(database_engine, email),
    )


def test_expired_oidc_caller_cannot_attach_identity_or_lock_out_another_user(
    database_engine: Engine,
    enterprise_client: TestClient,
) -> None:
    """A caller whose own oidc_expiry is in the past is still authenticated.

    Naming another account, including an admin, must not add an OAuth account
    or move that account's oidc_expiry.
    """
    caller_email = f"expired-caller-{uuid4().hex[:8]}@example.com"
    victim_email = f"admin-{uuid4().hex[:8]}@example.com"
    session = _session(database_engine)
    try:
        caller_user = _insert_user(
            session,
            caller_email,
            UserRole.BASIC,
            AccountType.STANDARD,
            oidc_expiry=datetime(2000, 1, 1, tzinfo=timezone.utc),
        )
        _insert_user(session, victim_email, UserRole.ADMIN, AccountType.STANDARD)
        caller_id = caller_user.id
    finally:
        session.close()
    before = _read_identity(database_engine, victim_email)
    assert before is not None

    _signed_in(enterprise_client, caller_id)
    first = enterprise_client.post(
        "/enterprise-settings/refresh-token",
        json=_refresh_body(victim_email, f"chosen-admin-{uuid4().hex}"),
    )
    second = enterprise_client.post(
        "/enterprise-settings/refresh-token",
        json=_refresh_body(victim_email, f"chosen-admin-{uuid4().hex}"),
    )

    after = _read_identity(database_engine, victim_email)
    _assert_no_cross_account_effect(first.status_code, before, after)
    _assert_no_cross_account_effect(second.status_code, before, after)
    # The victim's stored expiry must still admit them. double_check_user is
    # the gate on later requests and only looks at the user row.
    assert after is not None
    asyncio.run(
        double_check_user(User(oidc_expiry=after.oidc_expiry, is_verified=True))
    )


def test_active_caller_cannot_refresh_a_different_email(
    database_engine: Engine,
    enterprise_client: TestClient,
) -> None:
    caller_email = f"active-caller-{uuid4().hex[:8]}@example.com"
    victim_email = f"admin-{uuid4().hex[:8]}@example.com"
    session = _session(database_engine)
    try:
        caller_user = _insert_user(
            session, caller_email, UserRole.BASIC, AccountType.STANDARD
        )
        _insert_user(session, victim_email, UserRole.ADMIN, AccountType.STANDARD)
        caller_id = caller_user.id
    finally:
        session.close()
    before = _read_identity(database_engine, victim_email)

    _signed_in(enterprise_client, caller_id)
    response = enterprise_client.post(
        "/enterprise-settings/refresh-token",
        json=_refresh_body(victim_email, f"chosen-for-admin-{uuid4().hex}"),
    )

    _assert_no_cross_account_effect(
        response.status_code,
        before,
        _read_identity(database_engine, victim_email),
    )
    caller_after = _read_identity(database_engine, caller_email)
    assert caller_after is not None
    assert caller_after.oauth_accounts == ()


def test_matching_email_does_not_authorize_another_users_oauth_account(
    database_engine: Engine,
    enterprise_client: TestClient,
) -> None:
    """Email equality with the caller must not select a different OAuth row.

    The body email is the caller's. The body user id is an account id already
    linked to someone else. Reloading that account id after the email check
    still has to leave the other user untouched.
    """
    caller_email = f"caller-{uuid4().hex[:8]}@example.com"
    owner_email = f"owner-{uuid4().hex[:8]}@example.com"
    # One row must match. A duplicate account id makes the lookup raise before
    # the write, which would hide the update.
    victim_account_id = f"victim-idp-{uuid4().hex}"
    session = _session(database_engine)
    try:
        caller_user = _insert_user(
            session, caller_email, UserRole.BASIC, AccountType.STANDARD
        )
        owner = _insert_user(
            session, owner_email, UserRole.ADMIN, AccountType.STANDARD
        )
        session.add(
            OAuthAccount(
                id=uuid4(),
                oauth_name="custom",
                access_token=_EXISTING_ACCESS_TOKEN,
                refresh_token=_EXISTING_REFRESH_TOKEN,
                account_id=victim_account_id,
                account_email=owner_email,
                expires_at=2_000_000_000,
                user_id=owner.id,
            )
        )
        session.commit()
        caller_id = caller_user.id
    finally:
        session.close()
    before = _read_identity(database_engine, owner_email)
    assert before is not None
    assert before.oauth_accounts[0][2] == _EXISTING_ACCESS_TOKEN

    _signed_in(enterprise_client, caller_id)
    response = enterprise_client.post(
        "/enterprise-settings/refresh-token",
        json=_refresh_body(caller_email, victim_account_id),
    )

    _assert_no_cross_account_effect(
        response.status_code,
        before,
        _read_identity(database_engine, owner_email),
    )


@pytest.mark.parametrize(
    ("role", "account_type"),
    [
        (UserRole.SLACK_USER, AccountType.BOT),
        (UserRole.EXT_PERM_USER, AccountType.EXT_PERM_USER),
    ],
)
def test_placeholder_account_is_not_upgraded_or_linked(
    database_engine: Engine,
    enterprise_client: TestClient,
    role: UserRole,
    account_type: AccountType,
) -> None:
    caller_email = f"caller-{uuid4().hex[:8]}@example.com"
    placeholder_email = f"placeholder-{uuid4().hex[:8]}@example.com"
    session = _session(database_engine)
    try:
        _ensure_basic_group(session)
        caller_user = _insert_user(
            session, caller_email, UserRole.BASIC, AccountType.STANDARD
        )
        _insert_user(session, placeholder_email, role, account_type)
        caller_id = caller_user.id
    finally:
        session.close()
    before = _read_identity(database_engine, placeholder_email)
    assert before is not None
    assert before.role == role
    assert before.account_type == account_type
    assert before.group_ids == ()

    _signed_in(enterprise_client, caller_id)
    response = enterprise_client.post(
        "/enterprise-settings/refresh-token",
        json=_refresh_body(placeholder_email, f"chosen-placeholder-{uuid4().hex}"),
    )

    _assert_no_cross_account_effect(
        response.status_code,
        before,
        _read_identity(database_engine, placeholder_email),
    )


def test_unknown_email_does_not_create_an_account(
    database_engine: Engine,
    enterprise_client: TestClient,
) -> None:
    caller_email = f"caller-{uuid4().hex[:8]}@example.com"
    new_email = f"newcomer-{uuid4().hex[:8]}@example.com"
    session = _session(database_engine)
    try:
        caller_user = _insert_user(
            session, caller_email, UserRole.BASIC, AccountType.STANDARD
        )
        caller_id = caller_user.id
    finally:
        session.close()

    _signed_in(enterprise_client, caller_id)
    response = enterprise_client.post(
        "/enterprise-settings/refresh-token",
        json=_refresh_body(new_email, f"chosen-new-{uuid4().hex}"),
    )

    _assert_no_cross_account_effect(
        response.status_code,
        None,
        _read_identity(database_engine, new_email),
    )


def test_refresh_does_not_write_into_another_tenant(
    database_engine: Engine,
    enterprise_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_email = f"caller-{uuid4().hex[:8]}@example.com"
    victim_email = f"other-tenant-{uuid4().hex[:8]}@example.com"
    session = _session(database_engine)
    tenant_session = _session(database_engine, _OTHER_TENANT_SCHEMA)
    try:
        caller_user = _insert_user(
            session, caller_email, UserRole.BASIC, AccountType.STANDARD
        )
        session.add(
            UserTenantMapping(
                email=victim_email,
                tenant_id=_OTHER_TENANT_SCHEMA,
                active=True,
            )
        )
        session.commit()
        _ensure_basic_group(tenant_session)
        _insert_user(
            tenant_session, victim_email, UserRole.ADMIN, AccountType.STANDARD
        )
        caller_id = caller_user.id
    finally:
        session.close()
        tenant_session.close()
    before = _read_identity(database_engine, victim_email, _OTHER_TENANT_SCHEMA)
    assert before is not None

    # MULTI_TENANT is copied into each module at import. Setting the same flag
    # the process would have read from the environment makes get_or_provision_tenant
    # select the tenant stored for the body email.
    for module in (
        users_module,
        provisioning,
        user_mapping,
        sync_sql_engine,
        async_sql_engine,
        tenant_contextvars,
    ):
        monkeypatch.setattr(module, "MULTI_TENANT", True)

    _signed_in(enterprise_client, caller_id)
    response = enterprise_client.post(
        "/enterprise-settings/refresh-token",
        json=_refresh_body(victim_email, f"chosen-cross-tenant-{uuid4().hex}"),
    )

    _assert_no_cross_account_effect(
        response.status_code,
        before,
        _read_identity(database_engine, victim_email, _OTHER_TENANT_SCHEMA),
    )
    assert _read_identity(database_engine, victim_email) is None


def test_double_check_user_rejects_expired_oidc_expiry() -> None:
    """Documents the lock that a past oidc_expiry applies on later requests."""
    with pytest.raises(BasicAuthenticationError):
        asyncio.run(
            double_check_user(
                User(
                    oidc_expiry=datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                    is_verified=True,
                )
            )
        )
