"""Curators must not be able to change groups they do not curate.

These tests exercise the user-group HTTP handlers against PostgreSQL. They
fail while a curator or global curator can change membership or connector
links on a group they do not curate, including the seeded default Admin
group (is_default, permission grant "admin", is_up_to_date).

`set-curator` is not covered: it already calls
`_validate_curator_relationship_update_requester`. Rename, delete, permission
toggles, and agent updates are not covered: those routes require
`Permission.FULL_ADMIN_PANEL_ACCESS` before they run.

Run from `backend/` against the configured Postgres instance::

    pytest tests/external_dependency_unit/user_group/test_curator_group_membership_scope.py
"""

import asyncio
from collections.abc import Generator
from collections.abc import Iterator
from dataclasses import dataclass
from uuid import UUID
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_users.password import PasswordHelper
from sqlalchemy import create_engine
from sqlalchemy import delete
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from ee.onyx.server.user_group.api import router as user_group_router
from onyx.auth.permissions import get_effective_permissions
from onyx.auth.permissions import require_permission
from onyx.auth.schemas import UserRole
from onyx.auth.users import current_curator_or_admin_user
from onyx.auth.users import current_user
from onyx.configs.constants import DocumentSource
from onyx.connectors.models import InputType
from onyx.db.engine.sql_engine import build_connection_string
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import AccessType
from onyx.db.enums import AccountType
from onyx.db.enums import ConnectorCredentialPairStatus
from onyx.db.enums import GrantSource
from onyx.db.enums import Permission
from onyx.db.enums import ProcessingMode
from onyx.db.models import Base
from onyx.db.models import Connector
from onyx.db.models import ConnectorCredentialPair
from onyx.db.models import Credential
from onyx.db.models import DocumentSet
from onyx.db.models import DocumentSet__UserGroup
from onyx.db.models import LLMProvider
from onyx.db.models import ModelConfiguration
from onyx.db.models import OAuthAccount
from onyx.db.models import PermissionGrant
from onyx.db.models import Persona
from onyx.db.models import Persona__UserGroup
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.db.models import UserGroup
from onyx.db.models import UserGroup__ConnectorCredentialPair
from onyx.error_handling.exceptions import OnyxError

_SCHEMA_TABLES = [
    OAuthAccount.__table__,
    User.__table__,
    LLMProvider.__table__,
    ModelConfiguration.__table__,
    Persona.__table__,
    DocumentSet.__table__,
    Connector.__table__,
    Credential.__table__,
    ConnectorCredentialPair.__table__,
    UserGroup.__table__,
    User__UserGroup.__table__,
    PermissionGrant.__table__,
    UserGroup__ConnectorCredentialPair.__table__,
    Persona__UserGroup.__table__,
    DocumentSet__UserGroup.__table__,
]


@dataclass
class _World:
    curator_id: UUID
    bystander_id: UUID
    global_curator_id: UUID
    victim_id: UUID
    own_group_id: int
    admin_group_id: int
    elevated_group_id: int
    cc_pair_id: int
    connector_id: int
    credential_id: int


def _engine() -> Engine:
    return create_engine(build_connection_string(db_api="psycopg2"), pool_pre_ping=True)


def _ensure_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine, tables=_SCHEMA_TABLES, checkfirst=True)


@pytest.fixture()
def session_factory() -> Iterator[sessionmaker[Session]]:
    engine = _engine()
    _ensure_schema(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    engine.dispose()


def _new_user(db_session: Session, role: UserRole) -> User:
    password_helper = PasswordHelper()
    user = User(
        email=f"scope-{uuid4().hex[:12]}@example.com",
        hashed_password=password_helper.hash(password_helper.generate()),
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role=role,
        account_type=AccountType.STANDARD,
        effective_permissions=[Permission.BASIC_ACCESS.value],
    )
    db_session.add(user)
    db_session.flush()
    return user


def _build_world(factory: sessionmaker[Session]) -> _World:
    """Curator of one ordinary group, plus a default admin group and a
    private connector the curator does not already control.
    """
    db_session = factory()
    try:
        curator = _new_user(db_session, UserRole.CURATOR)
        bystander = _new_user(db_session, UserRole.BASIC)
        global_curator = _new_user(db_session, UserRole.GLOBAL_CURATOR)
        victim = _new_user(db_session, UserRole.BASIC)

        suffix = uuid4().hex[:8]
        own_group = UserGroup(
            name=f"curated-{suffix}",
            is_up_to_date=True,
            is_up_for_deletion=False,
            is_default=False,
        )
        db_session.add(own_group)
        db_session.flush()
        # Same security-relevant shape as the Admin group seeded by
        # alembic revision 977e834c1427: default, up to date, grant "admin".
        admin_group = UserGroup(
            name=f"Admin-{suffix}",
            is_up_to_date=True,
            is_up_for_deletion=False,
            is_default=True,
        )
        db_session.add(admin_group)
        db_session.flush()
        elevated_group = UserGroup(
            name=f"restricted-{suffix}",
            is_up_to_date=True,
            is_up_for_deletion=False,
            is_default=False,
        )
        db_session.add(elevated_group)
        db_session.flush()

        db_session.add(
            User__UserGroup(
                user_group_id=own_group.id,
                user_id=curator.id,
                is_curator=True,
            )
        )
        db_session.add(
            User__UserGroup(
                user_group_id=admin_group.id,
                user_id=victim.id,
                is_curator=False,
            )
        )
        db_session.add(
            PermissionGrant(
                group_id=own_group.id,
                permission=Permission.BASIC_ACCESS,
                grant_source=GrantSource.SYSTEM,
            )
        )
        db_session.add(
            PermissionGrant(
                group_id=admin_group.id,
                permission=Permission.FULL_ADMIN_PANEL_ACCESS,
                grant_source=GrantSource.SYSTEM,
            )
        )
        db_session.add(
            PermissionGrant(
                group_id=elevated_group.id,
                permission=Permission.MANAGE_CONNECTORS,
                grant_source=GrantSource.SYSTEM,
            )
        )

        connector = Connector(
            name=f"file-{suffix}",
            source=DocumentSource.FILE,
            input_type=InputType.LOAD_STATE,
            connector_specific_config={},
            refresh_freq=3600,
            prune_freq=3600,
        )
        db_session.add(connector)
        db_session.flush()
        credential = Credential(
            name=f"cred-{suffix}",
            source=DocumentSource.FILE,
            credential_json=None,
            admin_public=False,
            curator_public=False,
            user_id=victim.id,
        )
        db_session.add(credential)
        db_session.flush()
        cc_pair = ConnectorCredentialPair(
            name=f"pair-{suffix}",
            status=ConnectorCredentialPairStatus.ACTIVE,
            connector_id=connector.id,
            credential_id=credential.id,
            access_type=AccessType.PRIVATE,
            processing_mode=ProcessingMode.REGULAR,
        )
        db_session.add(cc_pair)
        db_session.commit()

        accepted = asyncio.run(current_curator_or_admin_user(user=curator))
        assert accepted.id == curator.id
        accepted_global = asyncio.run(
            current_curator_or_admin_user(user=global_curator)
        )
        assert accepted_global.id == global_curator.id

        return _World(
            curator_id=curator.id,
            bystander_id=bystander.id,
            global_curator_id=global_curator.id,
            victim_id=victim.id,
            own_group_id=own_group.id,
            admin_group_id=admin_group.id,
            elevated_group_id=elevated_group.id,
            cc_pair_id=cc_pair.id,
            connector_id=connector.id,
            credential_id=credential.id,
        )
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def _cleanup(factory: sessionmaker[Session], world: _World) -> None:
    db_session = factory()
    try:
        group_ids = [
            world.own_group_id,
            world.admin_group_id,
            world.elevated_group_id,
        ]
        user_ids = [
            world.curator_id,
            world.bystander_id,
            world.global_curator_id,
            world.victim_id,
        ]
        db_session.execute(
            delete(UserGroup__ConnectorCredentialPair).where(
                UserGroup__ConnectorCredentialPair.user_group_id.in_(group_ids)
            )
        )
        db_session.execute(
            delete(PermissionGrant).where(PermissionGrant.group_id.in_(group_ids))
        )
        db_session.execute(
            delete(User__UserGroup).where(
                User__UserGroup.user_group_id.in_(group_ids)
                | User__UserGroup.user_id.in_(user_ids)
            )
        )
        db_session.execute(
            delete(ConnectorCredentialPair).where(
                ConnectorCredentialPair.id == world.cc_pair_id
            )
        )
        db_session.execute(delete(Credential).where(Credential.id == world.credential_id))
        db_session.execute(delete(Connector).where(Connector.id == world.connector_id))
        db_session.execute(delete(UserGroup).where(UserGroup.id.in_(group_ids)))
        db_session.execute(delete(User).where(User.id.in_(user_ids)))
        db_session.commit()
    finally:
        db_session.close()


@pytest.fixture()
def world(session_factory: sessionmaker[Session]) -> Generator[_World, None, None]:
    built = _build_world(session_factory)
    try:
        yield built
    finally:
        _cleanup(session_factory, built)


def _client(factory: sessionmaker[Session], actor_id: UUID) -> TestClient:
    db_session = factory()
    actor = db_session.get(User, actor_id)
    assert actor is not None
    db_session.expunge(actor)
    db_session.close()

    app = FastAPI()
    app.include_router(user_group_router)

    def override_user() -> User:
        return actor

    def override_session() -> Generator[Session, None, None]:
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[current_user] = override_user
    app.dependency_overrides[get_session] = override_session
    return TestClient(app, raise_server_exceptions=False)


def _post_add_users(client: TestClient, group_id: int, user_ids: list[UUID]) -> None:
    client.post(
        f"/manage/admin/user-group/{group_id}/add-users",
        json={"user_ids": [str(user_id) for user_id in user_ids]},
    )


def _patch_group(
    client: TestClient, group_id: int, user_ids: list[UUID], cc_pair_ids: list[int]
) -> None:
    client.patch(
        f"/manage/admin/user-group/{group_id}",
        json={
            "user_ids": [str(user_id) for user_id in user_ids],
            "cc_pair_ids": cc_pair_ids,
        },
    )


def _member_ids(db_session: Session, group_id: int) -> set[UUID]:
    rows = db_session.query(User__UserGroup).filter_by(user_group_id=group_id).all()
    return {row.user_id for row in rows if row.user_id is not None}


def _current_cc_pair_ids(db_session: Session, group_id: int) -> set[int]:
    rows = (
        db_session.query(UserGroup__ConnectorCredentialPair)
        .filter_by(user_group_id=group_id, is_current=True)
        .all()
    )
    return {row.cc_pair_id for row in rows}


def _permission_denied(user: User, permission: Permission) -> bool:
    dependency = require_permission(permission)
    try:
        asyncio.run(dependency(user=user))
    except OnyxError:
        return True
    return False


def _escalation_problems(
    factory: sessionmaker[Session],
    user_id: UUID,
    forbidden: set[Permission],
) -> list[str]:
    db_session = factory()
    try:
        user = db_session.get(User, user_id)
        assert user is not None
        stored = set(user.effective_permissions)
        effective = get_effective_permissions(user)
        problems: list[str] = []
        for permission in forbidden:
            if permission.value in stored or permission in effective:
                problems.append(
                    f"{user.role.value} gained {permission.value}; "
                    f"stored={sorted(stored)}"
                )
            elif not _permission_denied(user, permission):
                problems.append(
                    f"require_permission({permission.value}) allowed "
                    f"{user.role.value}"
                )
        return problems
    finally:
        db_session.close()


def test_curator_cannot_add_users_to_default_admin_group(
    session_factory: sessionmaker[Session], world: _World
) -> None:
    """POST add-users targets the default admin group, not the curated group."""
    client = _client(session_factory, world.curator_id)
    _post_add_users(
        client,
        world.admin_group_id,
        [world.curator_id, world.bystander_id],
    )

    db_session = session_factory()
    try:
        admin_members = _member_ids(db_session, world.admin_group_id)
        own_members = _member_ids(db_session, world.own_group_id)
    finally:
        db_session.close()

    problems: list[str] = []
    if world.curator_id in admin_members:
        problems.append("curator was added to the default admin group")
    if world.bystander_id in admin_members:
        problems.append("bystander was added to the default admin group")
    if admin_members != {world.victim_id}:
        problems.append(f"admin group members changed: {admin_members}")
    if own_members != {world.curator_id}:
        problems.append(f"curated group membership changed: {own_members}")
    problems.extend(
        _escalation_problems(
            session_factory,
            world.curator_id,
            {Permission.FULL_ADMIN_PANEL_ACCESS},
        )
    )
    problems.extend(
        _escalation_problems(
            session_factory,
            world.bystander_id,
            {Permission.FULL_ADMIN_PANEL_ACCESS},
        )
    )
    assert not problems, "; ".join(problems)


def test_curator_cannot_patch_unrelated_group_membership_or_connectors(
    session_factory: sessionmaker[Session], world: _World
) -> None:
    """PATCH uses the URL group id, which is not the group the curator curates."""
    client = _client(session_factory, world.curator_id)
    _patch_group(
        client,
        world.elevated_group_id,
        [world.curator_id],
        [world.cc_pair_id],
    )
    _patch_group(
        client,
        world.admin_group_id,
        [],
        [world.cc_pair_id],
    )

    db_session = session_factory()
    try:
        elevated_members = _member_ids(db_session, world.elevated_group_id)
        elevated_cc_pairs = _current_cc_pair_ids(db_session, world.elevated_group_id)
        admin_members = _member_ids(db_session, world.admin_group_id)
        admin_cc_pairs = _current_cc_pair_ids(db_session, world.admin_group_id)
        own_members = _member_ids(db_session, world.own_group_id)
        own_cc_pairs = _current_cc_pair_ids(db_session, world.own_group_id)
    finally:
        db_session.close()

    problems: list[str] = []
    if elevated_members:
        problems.append(
            f"curator changed membership of a group they do not curate: {elevated_members}"
        )
    if elevated_cc_pairs:
        problems.append(
            f"curator attached a connector to a group they do not curate: {elevated_cc_pairs}"
        )
    if world.victim_id not in admin_members:
        problems.append("curator removed an existing member of the default admin group")
    if world.curator_id in admin_members:
        problems.append("curator joined the default admin group via PATCH")
    if admin_cc_pairs:
        problems.append(
            f"curator attached a connector to the default admin group: {admin_cc_pairs}"
        )
    if own_members != {world.curator_id} or own_cc_pairs:
        problems.append("the curated group was modified while updating other groups")
    problems.extend(
        _escalation_problems(
            session_factory,
            world.curator_id,
            {
                Permission.FULL_ADMIN_PANEL_ACCESS,
                Permission.MANAGE_CONNECTORS,
            },
        )
    )
    assert not problems, "; ".join(problems)


def test_global_curator_cannot_add_self_to_default_admin_group(
    session_factory: sessionmaker[Session], world: _World
) -> None:
    client = _client(session_factory, world.global_curator_id)
    _post_add_users(client, world.admin_group_id, [world.global_curator_id])

    db_session = session_factory()
    try:
        admin_members = _member_ids(db_session, world.admin_group_id)
    finally:
        db_session.close()

    problems: list[str] = []
    if world.global_curator_id in admin_members:
        problems.append("global curator was added to the default admin group")
    if admin_members != {world.victim_id}:
        problems.append(f"admin group members changed: {admin_members}")
    problems.extend(
        _escalation_problems(
            session_factory,
            world.global_curator_id,
            {Permission.FULL_ADMIN_PANEL_ACCESS},
        )
    )
    assert not problems, "; ".join(problems)
