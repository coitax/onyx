"""Cross-tenant invite approval must not detach or capture another workspace.

These tests exercise the multi-tenant cloud path:

* first user of a tenant is an admin (``UserManager.create`` / ``on_after_register``)
* ``POST /tenants/users/invite/approve`` as that admin
* password ``authenticate`` and Google/OIDC ``oauth_callback``, which both resolve
  the tenant from ``user_tenant_mapping``

They fail while approval of an arbitrary email deletes every mapping for that
email and inserts an active mapping to the caller's tenant.

Run from ``backend/`` with Enterprise code importable and multi-tenant mode on:

    DISPOSABLE_EMAIL_DOMAINS_URL= CACHE_BACKEND=postgres MULTI_TENANT=true \\
    DISABLE_TELEMETRY=true DEV_MODE=true \\
    pytest tests/unit/ee/onyx/server/tenants/test_invite_approval_authorization.py -q
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Generator
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.testclient import TestClient
from sqlalchemy import text
from ee.onyx.server.middleware.tenant_tracking import (
    add_api_server_tenant_id_middleware,
)
from ee.onyx.server.tenants.user_invitations_api import router as invitations_router
from ee.onyx.server.tenants.user_mapping import add_users_to_tenant
from ee.onyx.server.tenants.user_mapping import get_tenant_id_for_email
from onyx.auth.invited_users import get_invited_users
from onyx.auth.invited_users import write_pending_users
from onyx.auth.schemas import UserCreate
from onyx.auth.users import TenantAwareRedisStrategy
from onyx.auth.users import UserManager
from onyx.configs.constants import FASTAPI_USERS_AUTH_COOKIE_NAME
from onyx.configs.constants import KV_PENDING_USERS_KEY
from onyx.configs.constants import KV_SETTINGS_KEY
from onyx.db.engine.sql_engine import get_session_with_shared_schema
from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.engine.sql_engine import SqlEngine
from onyx.db.enums import GrantSource
from onyx.db.enums import Permission
from onyx.db.models import CacheStore
from onyx.db.models import KVStore
from onyx.db.models import OAuthAccount
from onyx.db.models import PermissionGrant
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.db.models import UserGroup
from onyx.db.models import UserTenantMapping
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.key_value_store.factory import get_kv_store
from onyx.key_value_store.store import PgRedisKVStore
from onyx.utils.logger import setup_logger
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

# Obvious placeholder. Not a deployed credential.
_PLACEHOLDER_PASSWORD = "placeholder-password"

_TENANT_TABLES = (
    User.__table__,
    OAuthAccount.__table__,
    UserGroup.__table__,
    User__UserGroup.__table__,
    PermissionGrant.__table__,
    KVStore.__table__,
    CacheStore.__table__,
)


def _require_runtime() -> None:
    """The product path under test is the MULTI_TENANT cloud configuration."""
    if os.environ.get("MULTI_TENANT", "").lower() != "true":
        raise RuntimeError("MULTI_TENANT=true is required for this test module")
    if os.environ.get("CACHE_BACKEND") != "postgres":
        raise RuntimeError("CACHE_BACKEND=postgres is required for this test module")


_require_runtime()


@pytest.fixture(scope="module", autouse=True)
def _database() -> Iterator[None]:
    SqlEngine.init_engine(pool_size=5, max_overflow=2)
    engine = SqlEngine.get_engine()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS public.user_tenant_mapping (
                    email VARCHAR NOT NULL,
                    tenant_id VARCHAR NOT NULL,
                    active BOOLEAN NOT NULL DEFAULT true,
                    CONSTRAINT uq_user_tenant PRIMARY KEY (email, tenant_id)
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_user_active_email_idx
                ON public.user_tenant_mapping (email)
                WHERE active = true
                """
            )
        )
    yield


@pytest.fixture
def app(enable_ee: None) -> FastAPI:  # noqa: ARG001
    application = FastAPI()
    register_onyx_exception_handlers(application)
    add_api_server_tenant_id_middleware(application, setup_logger())
    application.include_router(invitations_router)
    return application


def _suffix() -> str:
    return uuid.uuid4().hex[:12]


def _create_schema(schema: str) -> None:
    engine = SqlEngine.get_engine()
    with engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
    with engine.connect().execution_options(
        schema_translate_map={None: schema}
    ) as connection:
        for table in _TENANT_TABLES:
            table.create(connection, checkfirst=True)
        connection.commit()


def _drop_schema(schema: str) -> None:
    engine = SqlEngine.get_engine()
    with engine.begin() as connection:
        connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


def _seed_default_groups(tenant_id: str) -> None:
    """Same grants as alembic revision 977e834c1427 (Admin/admin, Basic/basic)."""
    with get_session_with_tenant(tenant_id=tenant_id) as db_session:
        for name, permission in (
            ("Admin", Permission.FULL_ADMIN_PANEL_ACCESS),
            ("Basic", Permission.BASIC_ACCESS),
        ):
            group = UserGroup(
                name=name,
                is_up_to_date=True,
                is_up_for_deletion=False,
                is_default=True,
            )
            db_session.add(group)
            db_session.flush()
            db_session.add(
                PermissionGrant(
                    group_id=group.id,
                    permission=permission,
                    grant_source=GrantSource.SYSTEM,
                )
            )
        db_session.commit()


def _mapping_rows(email: str) -> set[tuple[str, bool]]:
    with get_session_with_shared_schema() as db_session:
        rows = (
            db_session.query(UserTenantMapping)
            .filter(UserTenantMapping.email == email)
            .all()
        )
        return {(row.tenant_id, row.active) for row in rows}


def _delete_mappings(email: str) -> None:
    with get_session_with_shared_schema() as db_session:
        db_session.query(UserTenantMapping).filter(
            UserTenantMapping.email == email
        ).delete()
        db_session.commit()


def _emails_in_tenant(tenant_id: str) -> set[str]:
    with get_session_with_tenant(tenant_id=tenant_id) as db_session:
        return {email for (email,) in db_session.query(User.email).all()}


def _invited_users(tenant_id: str) -> list[str]:
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(tenant_id)
    try:
        return list(get_invited_users())
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)


def _write_pending(tenant_id: str, emails: list[str]) -> None:
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(tenant_id)
    try:
        write_pending_users(emails)
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)


def _write_settings(tenant_id: str, settings: dict[str, Any]) -> None:
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(tenant_id)
    try:
        get_kv_store().store(KV_SETTINGS_KEY, settings)
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)


def _drop_loop_bound_clients() -> None:
    """Async SQLAlchemy and Redis clients are bound to the loop that created them."""
    from onyx.db.engine import async_sql_engine
    from onyx.redis import redis_pool

    async_sql_engine._ASYNC_ENGINE = None
    redis_pool._async_redis_connection = None


def _run(coro: Any) -> Any:
    _drop_loop_bound_clients()
    return asyncio.run(coro)


async def _register_first_user(email: str, tenant_id: str) -> User:
    """Register the first user of ``tenant_id`` through ``UserManager.create``.

    The tenant schema stands in for ``provision_tenant`` (schema + migrations).
    Membership is recorded with ``add_users_to_tenant``, which
    ``assign_tenant_to_user`` calls, so ``get_or_provision_tenant`` resolves this
    tenant instead of provisioning another one. ``safe=True`` matches the
    public register router.
    """
    add_users_to_tenant([email], tenant_id)
    manager = UserManager(object())  # type: ignore[arg-type]
    try:
        user = await manager.create(
            UserCreate(email=email, password=_PLACEHOLDER_PASSWORD),
            safe=True,
        )
    except Exception:
        # Persona lookup runs after the user row and admin group assignment
        # commit. A missing persona table must not hide that assignment.
        user = None
    with get_session_with_tenant(tenant_id=tenant_id) as db_session:
        stored = db_session.query(User).filter(User.email == email).one()
        # Load the column before the session closes.
        stored.effective_permissions = list(stored.effective_permissions)
        db_session.expunge(stored)
    if user is None:
        return stored
    return stored


async def _session_token(user: User) -> str:
    return await TenantAwareRedisStrategy().write_token(user)


async def _password_login(email: str) -> User | None:
    manager = UserManager(object())  # type: ignore[arg-type]
    return await manager.authenticate(
        OAuth2PasswordRequestForm(username=email, password=_PLACEHOLDER_PASSWORD)
    )


async def _google_oauth(email: str) -> None:
    manager = UserManager(object())  # type: ignore[arg-type]
    try:
        await manager.oauth_callback(
            "google",
            "placeholder-access-token",
            f"acct-{email}",
            email,
            None,
            "placeholder-refresh-token",
            None,
            associate_by_email=True,
            is_verified_by_default=True,
        )
    except Exception:
        # Creation commits before later workspace setup. The assertions below
        # read the committed rows either way.
        return


def _approve(client: TestClient, token: str, email: str) -> Any:
    return client.post(
        "/tenants/users/invite/approve",
        json={"email": email},
        cookies={FASTAPI_USERS_AUTH_COOKIE_NAME: token},
    )


class _Workspaces:
    def __init__(self) -> None:
        self.schemas: list[str] = []
        self.emails: list[str] = []

    def tenant(self) -> str:
        schema = f"tenant_{_suffix()}"
        _create_schema(schema)
        _seed_default_groups(schema)
        self.schemas.append(schema)
        return schema

    def register(self, tenant_id: str, local_part: str) -> User:
        email = f"{local_part}-{_suffix()}@example.com"
        self.emails.append(email)
        return _run(_register_first_user(email, tenant_id))

    def close(self) -> None:
        for email in self.emails:
            _delete_mappings(email)
        for schema in self.schemas:
            _drop_schema(schema)


@pytest.fixture
def workspaces() -> Generator[_Workspaces, None, None]:
    bundle = _Workspaces()
    try:
        yield bundle
    finally:
        bundle.close()


def test_first_user_is_admin_and_unsolicited_approval_cannot_hijack_victim(
    app: FastAPI, workspaces: _Workspaces
) -> None:
    """Attack path: own-tenant admin approves a customer who never asked to join."""
    attacker_tenant = workspaces.tenant()
    victim_tenant = workspaces.tenant()
    other_tenant = workspaces.tenant()
    attacker = workspaces.register(attacker_tenant, "attacker")
    victim = workspaces.register(victim_tenant, "victim")

    assert Permission.FULL_ADMIN_PANEL_ACCESS.value in list(
        attacker.effective_permissions
    )
    # A pre-existing invitation to a third tenant must survive as well.
    add_users_to_tenant([victim.email], other_tenant)

    before = _mapping_rows(victim.email)
    assert (victim_tenant, True) in before
    assert (other_tenant, False) in before
    assert (attacker_tenant, True) not in before

    logged_in = _run(_password_login(victim.email))
    assert logged_in is not None
    assert logged_in.id == victim.id

    invited_before = _invited_users(attacker_tenant)
    token = _run(_session_token(attacker))
    _drop_loop_bound_clients()
    with TestClient(app) as client:
        response = _approve(client, token, victim.email)

    after = _mapping_rows(victim.email)
    logged_in_after = _run(_password_login(victim.email))
    _run(_google_oauth(victim.email))

    assert after == before
    assert victim.email not in _invited_users(attacker_tenant)
    assert _invited_users(attacker_tenant) == invited_before
    assert logged_in_after is not None
    assert logged_in_after.id == victim.id
    assert get_tenant_id_for_email(victim.email) == victim_tenant
    assert victim.email not in _emails_in_tenant(attacker_tenant)
    assert response.status_code >= 400


def test_invite_only_workspace_still_rejects_unsolicited_approval(
    app: FastAPI, workspaces: _Workspaces
) -> None:
    """Invite-only does not make an arbitrary email a member of this tenant."""
    attacker_tenant = workspaces.tenant()
    victim_tenant = workspaces.tenant()
    attacker = workspaces.register(attacker_tenant, "attacker")
    victim = workspaces.register(victim_tenant, "victim")
    _write_settings(attacker_tenant, {"invite_only_enabled": True})
    before = _mapping_rows(victim.email)

    token = _run(_session_token(attacker))
    _drop_loop_bound_clients()
    with TestClient(app) as client:
        response = _approve(client, token, victim.email)

    _run(_google_oauth(victim.email))

    assert response.status_code >= 400
    assert _mapping_rows(victim.email) == before
    assert victim.email not in _invited_users(attacker_tenant)
    assert get_tenant_id_for_email(victim.email) == victim_tenant
    assert victim.email not in _emails_in_tenant(attacker_tenant)
    logged_in = _run(_password_login(victim.email))
    assert logged_in is not None and logged_in.id == victim.id


def test_pending_request_does_not_authorize_a_different_email(
    app: FastAPI, workspaces: _Workspaces
) -> None:
    attacker_tenant = workspaces.tenant()
    victim_tenant = workspaces.tenant()
    attacker = workspaces.register(attacker_tenant, "attacker")
    victim = workspaces.register(victim_tenant, "victim")
    other = f"other-{_suffix()}@example.com"
    workspaces.emails.append(other)
    _write_pending(attacker_tenant, [other])
    before = _mapping_rows(victim.email)

    token = _run(_session_token(attacker))
    _drop_loop_bound_clients()
    with TestClient(app) as client:
        response = _approve(client, token, victim.email)

    assert response.status_code >= 400
    assert _mapping_rows(victim.email) == before
    assert get_tenant_id_for_email(victim.email) == victim_tenant
    logged_in = _run(_password_login(victim.email))
    assert logged_in is not None and logged_in.id == victim.id


def test_pending_list_of_another_tenant_does_not_authorize_approval(
    app: FastAPI, workspaces: _Workspaces
) -> None:
    attacker_tenant = workspaces.tenant()
    victim_tenant = workspaces.tenant()
    foreign_tenant = workspaces.tenant()
    attacker = workspaces.register(attacker_tenant, "attacker")
    victim = workspaces.register(victim_tenant, "victim")
    _write_pending(foreign_tenant, [victim.email])
    before = _mapping_rows(victim.email)

    token = _run(_session_token(attacker))
    _drop_loop_bound_clients()
    with TestClient(app) as client:
        response = _approve(client, token, victim.email)

    assert response.status_code >= 400
    assert _mapping_rows(victim.email) == before
    assert (attacker_tenant, True) not in _mapping_rows(victim.email)
    assert get_tenant_id_for_email(victim.email) == victim_tenant


def test_approving_pending_user_keeps_their_existing_active_tenant(
    app: FastAPI, workspaces: _Workspaces
) -> None:
    """A real join request may invite, and must not detach the user's workspace."""
    attacker_tenant = workspaces.tenant()
    victim_tenant = workspaces.tenant()
    attacker = workspaces.register(attacker_tenant, "attacker")
    victim = workspaces.register(victim_tenant, "victim")
    _write_pending(attacker_tenant, [victim.email])

    token = _run(_session_token(attacker))
    _drop_loop_bound_clients()
    with TestClient(app) as client:
        response = _approve(client, token, victim.email)

    rows = _mapping_rows(victim.email)
    assert response.status_code < 400
    assert (victim_tenant, True) in rows
    assert (attacker_tenant, True) not in rows
    assert (attacker_tenant, False) in rows
    assert get_tenant_id_for_email(victim.email) == victim_tenant
    logged_in = _run(_password_login(victim.email))
    assert logged_in is not None and logged_in.id == victim.id
    _run(_google_oauth(victim.email))
    assert victim.email not in _emails_in_tenant(attacker_tenant)


def test_revoked_pending_request_is_not_applied(
    app: FastAPI, workspaces: _Workspaces, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pending row observed by the check must still be pending at the write.

    The first read of this tenant's pending list withdraws the victim. A later
    read sees an empty list. Approval that writes mappings from the first read,
    or that loads the victim's mappings again and updates those rows, fails.
    """
    attacker_tenant = workspaces.tenant()
    victim_tenant = workspaces.tenant()
    attacker = workspaces.register(attacker_tenant, "attacker")
    victim = workspaces.register(victim_tenant, "victim")
    _write_pending(attacker_tenant, [victim.email])
    before = _mapping_rows(victim.email)
    revoked = {"done": False}
    original_load = PgRedisKVStore.load

    def _load(
        self: PgRedisKVStore, key: str, refresh_cache: bool = False
    ) -> Any:
        value = original_load(self, key, refresh_cache=refresh_cache)
        if (
            key == KV_PENDING_USERS_KEY
            and not revoked["done"]
            and isinstance(value, list)
            and victim.email in value
        ):
            revoked["done"] = True
            write_pending_users([item for item in value if item != victim.email])
        return value

    monkeypatch.setattr(PgRedisKVStore, "load", _load)
    token = _run(_session_token(attacker))
    _drop_loop_bound_clients()
    with TestClient(app) as client:
        response = _approve(client, token, victim.email)

    assert revoked["done"] is True
    assert response.status_code >= 400
    assert _mapping_rows(victim.email) == before
    assert get_tenant_id_for_email(victim.email) == victim_tenant


def test_basic_user_cannot_approve_an_email(
    app: FastAPI, workspaces: _Workspaces
) -> None:
    attacker_tenant = workspaces.tenant()
    victim_tenant = workspaces.tenant()
    # First user becomes the admin; the second user is a basic member.
    workspaces.register(attacker_tenant, "admin")
    basic = workspaces.register(attacker_tenant, "basic")
    victim = workspaces.register(victim_tenant, "victim")
    assert Permission.FULL_ADMIN_PANEL_ACCESS.value not in list(
        basic.effective_permissions
    )
    before = _mapping_rows(victim.email)

    token = _run(_session_token(basic))
    _drop_loop_bound_clients()
    with TestClient(app) as client:
        response = _approve(client, token, victim.email)

    assert response.status_code == 403
    assert _mapping_rows(victim.email) == before
