"""Basic group members must not edit agents an admin shared with their group.

The agent update endpoint (PATCH /persona/{persona_id}) only requires
BASIC_ACCESS, then create_update_persona -> upsert_persona authorizes the
caller with fetch_persona_by_id_for_user(..., get_editable=True). That filter
requires a curator relationship only when the caller's role is CURATOR. A
BASIC member of the single group an agent is shared with therefore passes the
edit check and upsert_persona overwrites the agent.

These tests call the endpoint functions that a basic user can reach, against
PostgreSQL. The enable_ee fixture turns on Enterprise the same way the API
process does at startup (global_version.set_ee), so group sharing uses
ee.onyx.db.persona.update_persona_access. They must fail while a basic member
can rewrite a group-shared agent, and pass once that edit is refused and no
earlier write is committed.

Operations that use the same edit filter but are not reachable by a basic user,
so they are not exercised here:

- PATCH /admin/persona/{id}/listed and PATCH /admin/persona/{id}/featured go
  through current_curator_or_admin_user before update_persona_visibility /
  update_persona_featured.
- PATCH /admin/agents/display-priorities requires FULL_ADMIN_PANEL_ACCESS.
  update_personas_display_priority also queries with get_editable=False.

DELETE /persona/{id}, PATCH /persona/{id}/share, and PATCH /persona/{id}/public
are reachable with BASIC_ACCESS but do not use this filter as their only
check (delete uses get_persona_by_id; share and public status reject non-owners
after the filter). Tests below keep those refusals, including the committed
state, so a fix cannot open them or write before refusing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import Session

from ee.onyx.db.user_group import insert_user_group
from ee.onyx.db.user_group import update_user_curator_relationship
from ee.onyx.server.user_group.models import SetCuratorRequest
from ee.onyx.server.user_group.models import UserGroupCreate
from onyx.auth.permissions import require_permission
from onyx.auth.schemas import UserRole
from onyx.db.enums import GrantSource
from onyx.db.enums import Permission
from onyx.db.enums import UserFileStatus
from onyx.db.models import DocumentSet
from onyx.db.models import Notification
from onyx.db.models import PermissionGrant
from onyx.db.models import Persona
from onyx.db.models import Tool
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.db.models import UserFile
from onyx.db.models import UserGroup
from onyx.db.persona import create_update_persona
from onyx.db.persona import get_minimal_persona_snapshots_for_user
from onyx.db.users import assign_user_to_default_groups__no_commit
from onyx.server.features.persona.api import delete_persona
from onyx.server.features.persona.api import patch_user_persona_public_status
from onyx.server.features.persona.api import PersonaShareRequest
from onyx.server.features.persona.api import share_persona
from onyx.server.features.persona.api import update_persona
from onyx.server.features.persona.api import IsPublicRequest
from onyx.server.features.persona.models import PersonaUpsertRequest
from tests.external_dependency_unit.conftest import create_test_user

_INJECTED_SYSTEM_PROMPT = "INJECTED system prompt from a basic group member"
_INJECTED_TASK_PROMPT = "INJECTED task prompt from a basic group member"
_ORIGINAL_SYSTEM_PROMPT = "admin-written system prompt"
_ORIGINAL_TASK_PROMPT = "admin-written task prompt"


@dataclass
class AgentState:
    name: str
    description: str
    system_prompt: str | None
    task_prompt: str | None
    is_public: bool
    replace_base_system_prompt: bool
    datetime_aware: bool
    deleted: bool
    owner_id: UUID | None
    tool_ids: frozenset[int]
    document_set_ids: frozenset[int]
    user_file_ids: frozenset[UUID]
    group_ids: frozenset[int]
    shared_user_ids: frozenset[UUID]
    file_sync_flags: frozenset[tuple[UUID, bool]]


@dataclass
class GroupAgentWorld:
    admin: User
    member: User
    outsider: User
    curator: User
    group_g: UserGroup
    group_h: UserGroup
    original_tool: Tool
    replacement_tool: Tool
    original_document_set: DocumentSet
    replacement_document_set: DocumentSet
    admin_file: UserFile
    member_file: UserFile
    group_agent_id: int
    both_groups_agent_id: int
    member_agent_id: int
    original_group_agent: AgentState
    original_member_agent: AgentState


def _ensure_default_group(
    db_session: Session, name: str, permission: Permission
) -> UserGroup:
    """The product seeds default Basic and Admin groups in a migration.

    Reuse them when this database already has them. Otherwise create the same
    rows that migration assigns new users into.
    """
    existing = (
        db_session.query(UserGroup)
        .filter(UserGroup.name == name, UserGroup.is_default.is_(True))
        .one_or_none()
    )
    if existing is not None:
        return existing
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
    db_session.flush()
    return group


def _create_tool(db_session: Session, name: str) -> Tool:
    tool = Tool(
        name=name,
        description=name,
        in_code_tool_id=None,
        display_name=name,
        enabled=True,
    )
    db_session.add(tool)
    db_session.flush()
    return tool


def _create_document_set(db_session: Session, owner: User, name: str) -> DocumentSet:
    document_set = DocumentSet(
        name=name,
        description=name,
        user_id=owner.id,
        is_up_to_date=True,
        is_public=False,
    )
    db_session.add(document_set)
    db_session.flush()
    return document_set


def _create_user_file(db_session: Session, owner: User, name: str) -> UserFile:
    user_file = UserFile(
        id=uuid4(),
        user_id=owner.id,
        file_id=f"file-{uuid4().hex}",
        name=name,
        file_type="text/plain",
        status=UserFileStatus.COMPLETED,
        needs_persona_sync=False,
        chunk_count=1,
    )
    db_session.add(user_file)
    db_session.flush()
    return user_file


def _upsert_request(
    *,
    name: str,
    description: str,
    system_prompt: str,
    task_prompt: str,
    is_public: bool,
    tool_ids: list[int],
    document_set_ids: list[int],
    user_ids: list[UUID],
    group_ids: list[int],
    user_file_ids: list[str] | None,
    replace_base_system_prompt: bool = False,
    datetime_aware: bool = True,
) -> PersonaUpsertRequest:
    return PersonaUpsertRequest(
        name=name,
        description=description,
        document_set_ids=document_set_ids,
        is_public=is_public,
        users=user_ids,
        groups=group_ids,
        tool_ids=tool_ids,
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        datetime_aware=datetime_aware,
        replace_base_system_prompt=replace_base_system_prompt,
        user_file_ids=user_file_ids,
        is_featured=False,
    )


def _load_agent(db_session: Session, persona_id: int) -> Persona:
    db_session.expire_all()
    return db_session.scalars(
        select(Persona)
        .where(Persona.id == persona_id)
        .options(
            selectinload(Persona.tools),
            selectinload(Persona.document_sets),
            selectinload(Persona.user_files),
            selectinload(Persona.groups),
            selectinload(Persona.users),
        )
    ).one()


def _snapshot(db_session: Session, persona_id: int) -> AgentState:
    persona = _load_agent(db_session, persona_id)
    file_flags = frozenset(
        (user_file.id, user_file.needs_persona_sync) for user_file in persona.user_files
    )
    return AgentState(
        name=persona.name,
        description=persona.description,
        system_prompt=persona.system_prompt,
        task_prompt=persona.task_prompt,
        is_public=persona.is_public,
        replace_base_system_prompt=persona.replace_base_system_prompt,
        datetime_aware=persona.datetime_aware,
        deleted=persona.deleted,
        owner_id=persona.user_id,
        tool_ids=frozenset(tool.id for tool in persona.tools),
        document_set_ids=frozenset(document_set.id for document_set in persona.document_sets),
        user_file_ids=frozenset(user_file.id for user_file in persona.user_files),
        group_ids=frozenset(group.id for group in persona.groups),
        shared_user_ids=frozenset(user.id for user in persona.users if user.id is not None),
        file_sync_flags=file_flags,
    )


def _notification_count(db_session: Session, user_id: UUID) -> int:
    db_session.expire_all()
    return (
        db_session.query(Notification)
        .filter(Notification.user_id == user_id)
        .count()
    )


def _call_and_close_request(db_session: Session, action: object) -> Exception | None:
    """Run an endpoint body the way get_session does.

    A successful create_update_persona commits. An exception leaves the
    request session to roll back uncommitted work when it closes. Commits
    that already happened, including create_notification's autocommit, stay
    visible after that rollback.
    """
    try:
        action()  # type: ignore[operator]
    except Exception as exc:
        db_session.rollback()
        return exc
    return None


@pytest.fixture
def group_agent_world(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
    enable_ee: None,  # noqa: ARG001
) -> GroupAgentWorld:
    suffix = uuid4().hex[:8]
    _ensure_default_group(db_session, "Basic", Permission.BASIC_ACCESS)
    _ensure_default_group(db_session, "Admin", Permission.FULL_ADMIN_PANEL_ACCESS)

    admin = create_test_user(db_session, f"admin-{suffix}", role=UserRole.ADMIN)
    member = create_test_user(db_session, f"member-{suffix}", role=UserRole.BASIC)
    outsider = create_test_user(db_session, f"outsider-{suffix}", role=UserRole.BASIC)
    curator = create_test_user(db_session, f"curator-{suffix}", role=UserRole.BASIC)

    assign_user_to_default_groups__no_commit(db_session, admin, is_admin=True)
    assign_user_to_default_groups__no_commit(db_session, member, is_admin=False)
    assign_user_to_default_groups__no_commit(db_session, outsider, is_admin=False)
    assign_user_to_default_groups__no_commit(db_session, curator, is_admin=False)
    db_session.commit()

    group_g = insert_user_group(
        db_session,
        UserGroupCreate(
            name=f"group-g-{suffix}",
            user_ids=[member.id, curator.id],
            cc_pair_ids=[],
        ),
    )
    group_h = insert_user_group(
        db_session,
        UserGroupCreate(
            name=f"group-h-{suffix}",
            user_ids=[],
            cc_pair_ids=[],
        ),
    )
    update_user_curator_relationship(
        db_session,
        group_g.id,
        SetCuratorRequest(user_id=curator.id, is_curator=True),
        admin,
    )

    db_session.refresh(admin)
    db_session.refresh(member)
    db_session.refresh(curator)

    original_tool = _create_tool(db_session, f"original-tool-{suffix}")
    replacement_tool = _create_tool(db_session, f"replacement-tool-{suffix}")
    original_document_set = _create_document_set(
        db_session, admin, f"original-docs-{suffix}"
    )
    replacement_document_set = _create_document_set(
        db_session, admin, f"replacement-docs-{suffix}"
    )
    admin_file = _create_user_file(db_session, admin, f"admin-secret-{suffix}.txt")
    member_file = _create_user_file(db_session, member, f"member-file-{suffix}.txt")
    db_session.commit()

    group_agent = create_update_persona(
        persona_id=None,
        create_persona_request=_upsert_request(
            name=f"group-agent-{suffix}",
            description="admin managed group agent",
            system_prompt=_ORIGINAL_SYSTEM_PROMPT,
            task_prompt=_ORIGINAL_TASK_PROMPT,
            is_public=False,
            tool_ids=[original_tool.id],
            document_set_ids=[original_document_set.id],
            user_ids=[],
            group_ids=[group_g.id],
            user_file_ids=[str(admin_file.id)],
        ),
        user=admin,
        db_session=db_session,
    )
    both_groups_agent = create_update_persona(
        persona_id=None,
        create_persona_request=_upsert_request(
            name=f"both-groups-agent-{suffix}",
            description="shared with two groups",
            system_prompt=_ORIGINAL_SYSTEM_PROMPT,
            task_prompt=_ORIGINAL_TASK_PROMPT,
            is_public=False,
            tool_ids=[original_tool.id],
            document_set_ids=[original_document_set.id],
            user_ids=[],
            group_ids=[group_g.id, group_h.id],
            user_file_ids=None,
        ),
        user=admin,
        db_session=db_session,
    )
    member_agent = create_update_persona(
        persona_id=None,
        create_persona_request=_upsert_request(
            name=f"member-agent-{suffix}",
            description="owned by the basic member",
            system_prompt="member original prompt",
            task_prompt="member original task",
            is_public=False,
            tool_ids=[original_tool.id],
            document_set_ids=[],
            user_ids=[],
            group_ids=[],
            user_file_ids=None,
        ),
        user=member,
        db_session=db_session,
    )

    for user_file in (admin_file, member_file):
        db_session.refresh(user_file)
        user_file.needs_persona_sync = False
    db_session.commit()

    world = GroupAgentWorld(
        admin=admin,
        member=member,
        outsider=outsider,
        curator=curator,
        group_g=group_g,
        group_h=group_h,
        original_tool=original_tool,
        replacement_tool=replacement_tool,
        original_document_set=original_document_set,
        replacement_document_set=replacement_document_set,
        admin_file=admin_file,
        member_file=member_file,
        group_agent_id=group_agent.id,
        both_groups_agent_id=both_groups_agent.id,
        member_agent_id=member_agent.id,
        original_group_agent=_snapshot(db_session, group_agent.id),
        original_member_agent=_snapshot(db_session, member_agent.id),
    )
    _assert_setup(db_session, world)
    return world


def _assert_setup(db_session: Session, world: GroupAgentWorld) -> None:
    db_session.refresh(world.member)
    db_session.refresh(world.admin)
    db_session.refresh(world.curator)

    assert world.member.role == UserRole.BASIC
    membership = (
        db_session.query(User__UserGroup)
        .filter(
            User__UserGroup.user_id == world.member.id,
            User__UserGroup.user_group_id == world.group_g.id,
        )
        .one()
    )
    assert membership.is_curator is False
    assert world.curator.role == UserRole.CURATOR

    # The endpoint's only gate is BASIC_ACCESS, which group membership grants.
    allowed = asyncio.run(
        require_permission(Permission.BASIC_ACCESS)(user=world.member)
    )
    assert allowed.id == world.member.id

    visible = get_minimal_persona_snapshots_for_user(
        user=world.member,
        db_session=db_session,
        get_editable=False,
        include_default=False,
    )
    assert world.group_agent_id in {persona.id for persona in visible}

    agent = world.original_group_agent
    assert agent.owner_id == world.admin.id
    assert agent.is_public is False
    assert agent.group_ids == frozenset({world.group_g.id})
    assert agent.system_prompt == _ORIGINAL_SYSTEM_PROMPT
    assert agent.tool_ids == frozenset({world.original_tool.id})
    assert agent.document_set_ids == frozenset({world.original_document_set.id})
    assert agent.user_file_ids == frozenset({world.admin_file.id})
    assert agent.shared_user_ids == frozenset()
    assert agent.file_sync_flags == frozenset({(world.admin_file.id, False)})


def _assert_unchanged(
    db_session: Session,
    persona_id: int,
    expected: AgentState,
    *,
    outsider_notifications_before: int,
    outsider_id: UUID,
) -> None:
    assert _snapshot(db_session, persona_id) == expected
    assert _notification_count(db_session, outsider_id) == outsider_notifications_before


def _attack(
    db_session: Session, world: GroupAgentWorld, persona_id: int
) -> Exception | None:
    request = _upsert_request(
        name=_load_agent(db_session, persona_id).name,
        description="rewritten by a basic group member",
        system_prompt=_INJECTED_SYSTEM_PROMPT,
        task_prompt=_INJECTED_TASK_PROMPT,
        is_public=True,
        tool_ids=[world.replacement_tool.id],
        document_set_ids=[world.replacement_document_set.id],
        user_ids=[world.outsider.id],
        group_ids=[],
        user_file_ids=[str(world.member_file.id)],
        replace_base_system_prompt=True,
        datetime_aware=False,
    )
    return _call_and_close_request(
        db_session,
        lambda: update_persona(
            persona_id=persona_id,
            persona_upsert_request=request,
            user=world.member,
            db_session=db_session,
        ),
    )


def test_basic_group_member_cannot_rewrite_group_shared_agent(
    db_session: Session,
    group_agent_world: GroupAgentWorld,
) -> None:
    """A basic member of the agent's only group cannot change it.

    The requested id is the group agent. The member also owns a different
    agent. Refusal must leave the group agent as the admin wrote it and must
    not write the injected prompts onto the member's own agent.
    """
    world = group_agent_world
    notifications_before = _notification_count(db_session, world.outsider.id)

    error = _attack(db_session, world, world.group_agent_id)

    if (
        isinstance(error, HTTPException)
        and error.status_code == 400
        and "User Knowledge is disabled" in str(error.detail)
    ):
        raise AssertionError(
            "The edit was rejected by the user-knowledge pre-check before "
            "persona authorization ran, so it does not show whether a basic "
            f"member is allowed to edit the agent ({error.detail})."
        )
    assert error is not None, (
        "PATCH /persona/{id} let a basic group member edit an admin-owned "
        "agent that is shared only with that member's group"
    )
    _assert_unchanged(
        db_session,
        world.group_agent_id,
        world.original_group_agent,
        outsider_notifications_before=notifications_before,
        outsider_id=world.outsider.id,
    )
    assert _snapshot(db_session, world.member_agent_id) == world.original_member_agent
    group_agent = _load_agent(db_session, world.group_agent_id)
    member_agent = _load_agent(db_session, world.member_agent_id)
    assert group_agent.system_prompt != _INJECTED_SYSTEM_PROMPT
    assert member_agent.system_prompt != _INJECTED_SYSTEM_PROMPT
    assert group_agent.id == world.group_agent_id


def test_basic_group_member_cannot_rewrite_agent_shared_with_an_overlapping_group(
    db_session: Session,
    group_agent_world: GroupAgentWorld,
) -> None:
    """Membership in one of an agent's groups is not edit access either."""
    world = group_agent_world
    before = _snapshot(db_session, world.both_groups_agent_id)
    notifications_before = _notification_count(db_session, world.outsider.id)

    error = _attack(db_session, world, world.both_groups_agent_id)

    assert error is not None
    _assert_unchanged(
        db_session,
        world.both_groups_agent_id,
        before,
        outsider_notifications_before=notifications_before,
        outsider_id=world.outsider.id,
    )


def test_basic_user_outside_the_group_cannot_rewrite_the_agent(
    db_session: Session,
    group_agent_world: GroupAgentWorld,
) -> None:
    world = group_agent_world
    notifications_before = _notification_count(db_session, world.outsider.id)
    request = _upsert_request(
        name=world.original_group_agent.name,
        description="rewritten by an outsider",
        system_prompt=_INJECTED_SYSTEM_PROMPT,
        task_prompt=_INJECTED_TASK_PROMPT,
        is_public=True,
        tool_ids=[world.replacement_tool.id],
        document_set_ids=[world.replacement_document_set.id],
        user_ids=[world.member.id],
        group_ids=[],
        user_file_ids=[str(world.member_file.id)],
        replace_base_system_prompt=True,
        datetime_aware=False,
    )

    error = _call_and_close_request(
        db_session,
        lambda: update_persona(
            persona_id=world.group_agent_id,
            persona_upsert_request=request,
            user=world.outsider,
            db_session=db_session,
        ),
    )

    assert error is not None
    _assert_unchanged(
        db_session,
        world.group_agent_id,
        world.original_group_agent,
        outsider_notifications_before=notifications_before,
        outsider_id=world.outsider.id,
    )


def test_basic_user_can_still_edit_an_agent_they_own(
    db_session: Session,
    group_agent_world: GroupAgentWorld,
) -> None:
    world = group_agent_world
    request = _upsert_request(
        name=world.original_member_agent.name,
        description="updated by the owner",
        system_prompt="member updated prompt",
        task_prompt=world.original_member_agent.task_prompt or "",
        is_public=False,
        tool_ids=[world.original_tool.id],
        document_set_ids=[],
        user_ids=[],
        group_ids=[],
        user_file_ids=None,
    )

    error = _call_and_close_request(
        db_session,
        lambda: update_persona(
            persona_id=world.member_agent_id,
            persona_upsert_request=request,
            user=world.member,
            db_session=db_session,
        ),
    )

    assert error is None
    updated = _snapshot(db_session, world.member_agent_id)
    assert updated.description == "updated by the owner"
    assert updated.system_prompt == "member updated prompt"
    assert updated.owner_id == world.member.id
    assert _snapshot(db_session, world.group_agent_id) == world.original_group_agent


def test_curator_of_the_group_can_still_edit_the_group_agent(
    db_session: Session,
    group_agent_world: GroupAgentWorld,
) -> None:
    world = group_agent_world
    request = _upsert_request(
        name=world.original_group_agent.name,
        description="updated by the curator",
        system_prompt=world.original_group_agent.system_prompt or "",
        task_prompt=world.original_group_agent.task_prompt or "",
        is_public=False,
        tool_ids=[world.original_tool.id],
        document_set_ids=[world.original_document_set.id],
        user_ids=[],
        group_ids=[world.group_g.id],
        user_file_ids=[str(world.admin_file.id)],
    )

    error = _call_and_close_request(
        db_session,
        lambda: update_persona(
            persona_id=world.group_agent_id,
            persona_upsert_request=request,
            user=world.curator,
            db_session=db_session,
        ),
    )

    assert error is None
    updated = _snapshot(db_session, world.group_agent_id)
    assert updated.description == "updated by the curator"
    assert updated.system_prompt == _ORIGINAL_SYSTEM_PROMPT
    assert updated.is_public is False
    assert updated.group_ids == frozenset({world.group_g.id})
    assert updated.owner_id == world.admin.id


def test_basic_group_member_cannot_share_or_publish_or_delete_group_agent(
    db_session: Session,
    group_agent_world: GroupAgentWorld,
) -> None:
    """Share, public-status, and delete stay refused and commit nothing."""
    world = group_agent_world
    notifications_before = _notification_count(db_session, world.outsider.id)

    share_error = _call_and_close_request(
        db_session,
        lambda: share_persona(
            persona_id=world.group_agent_id,
            persona_share_request=PersonaShareRequest(
                user_ids=[world.outsider.id],
                group_ids=[],
                is_public=True,
            ),
            user=world.member,
            db_session=db_session,
        ),
    )
    public_error = _call_and_close_request(
        db_session,
        lambda: patch_user_persona_public_status(
            persona_id=world.group_agent_id,
            is_public_request=IsPublicRequest(is_public=True),
            user=world.member,
            db_session=db_session,
        ),
    )
    delete_error = _call_and_close_request(
        db_session,
        lambda: delete_persona(
            persona_id=world.group_agent_id,
            user=world.member,
            db_session=db_session,
        ),
    )

    assert share_error is not None
    assert public_error is not None
    assert delete_error is not None
    _assert_unchanged(
        db_session,
        world.group_agent_id,
        world.original_group_agent,
        outsider_notifications_before=notifications_before,
        outsider_id=world.outsider.id,
    )
