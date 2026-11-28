"""Authorization tests for deleting ownerless and builtin agents.

These tests exercise DELETE /api/persona/{id} (``delete_persona`` ->
``mark_persona_as_deleted`` -> ``get_persona_by_id``) as a signed-in basic
user. They fail while a non-admin can mark an ownerless or builtin agent
deleted, and they pass once that is refused.

The seeded default assistant (id 0) and Slack channel agents are created
with no owner. ``get_persona_by_id`` treats ``Persona.user_id IS NULL`` as
editable for every non-admin when ``is_for_edit`` is true, which is the
default used by deletion.
"""

import asyncio
from collections.abc import Callable
from uuid import uuid4

from fastapi import HTTPException
from fastapi_users.password import PasswordHelper
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.auth.schemas import UserRole
from onyx.configs.constants import DEFAULT_PERSONA_ID
from onyx.configs.constants import DocumentSource
from onyx.connectors.models import InputType
from onyx.db.connector import check_connectors_exist
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.enums import Permission
from onyx.db.models import Connector
from onyx.db.models import Persona
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.db.models import UserGroup
from onyx.db.persona import get_minimal_persona_snapshots_for_user
from onyx.db.persona import get_persona_by_id
from onyx.db.persona import mark_persona_as_deleted
from onyx.db.persona import mark_persona_as_not_deleted
from onyx.db.persona import upsert_persona
from onyx.db.slack_channel_config import create_slack_channel_persona
from onyx.db.users import assign_user_to_default_groups__no_commit
from onyx.db.users import delete_user_from_db
from onyx.error_handling.exceptions import OnyxError
from onyx.server.features.persona.api import delete_persona

_REFUSAL_TYPES = (ValueError, HTTPException, OnyxError, PermissionError)


def _create_signed_in_user(
    db_session: Session,
    email_prefix: str,
    *,
    is_admin: bool = False,
) -> User:
    """Create a user the way registration does: default group plus permissions."""
    password_helper = PasswordHelper()
    user = User(
        id=uuid4(),
        email=f"{email_prefix}_{uuid4().hex[:8]}@example.com",
        hashed_password=password_helper.hash(password_helper.generate()),
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role=UserRole.ADMIN if is_admin else UserRole.BASIC,
    )
    db_session.add(user)
    db_session.flush()
    assign_user_to_default_groups__no_commit(db_session, user, is_admin=is_admin)
    db_session.commit()
    db_session.refresh(user)
    return user


def _pass_basic_access_gate(user: User) -> User:
    """Run the same permission dependency the delete route declares."""
    allowed = asyncio.run(require_permission(Permission.BASIC_ACCESS)(user=user))
    assert allowed.id == user.id
    assert Permission.BASIC_ACCESS.value in user.effective_permissions or (
        Permission.FULL_ADMIN_PANEL_ACCESS.value in user.effective_permissions
    )
    return allowed


def _deleted_flags(db_session: Session) -> dict[int, bool]:
    flags: dict[int, bool] = {}
    for persona_id, deleted in db_session.execute(select(Persona.id, Persona.deleted)):
        flags[int(persona_id)] = bool(deleted)
    return flags


def _committed_deleted(persona_id: int) -> bool:
    """Read the deleted flag from a new session so a prior commit is visible."""
    with get_session_with_current_tenant() as fresh:
        persona = fresh.get(Persona, persona_id)
        assert persona is not None
        return bool(persona.deleted)


def _restore_default_assistant(db_session: Session) -> None:
    persona = db_session.get(Persona, DEFAULT_PERSONA_ID)
    if persona is None:
        return
    if persona.deleted:
        persona.deleted = False
        db_session.commit()


def _add_unrelated_group_memberships(db_session: Session, user: User) -> None:
    """Extra group rows make the delete query's outer joins non-trivial."""
    for _ in range(2):
        group = UserGroup(
            name=f"unrelated-{uuid4().hex[:8]}",
            is_up_to_date=False,
            is_up_for_deletion=False,
            is_default=False,
        )
        db_session.add(group)
        db_session.flush()
        db_session.add(
            User__UserGroup(
                user_id=user.id,
                user_group_id=group.id,
                is_curator=False,
            )
        )
    db_session.commit()


def _require_refused(
    db_session: Session,
    action: Callable[[], None],
    persona_id: int,
) -> None:
    """The action must raise, and it must not delete this persona or any other.

    The committed flag is checked even when the action returns normally, so a
    delete that lands and then raises still fails.
    """
    before = _deleted_flags(db_session)
    assert persona_id in before
    assert before[persona_id] is False

    error: Exception | None = None
    try:
        action()
    except Exception as exc:
        error = exc

    db_session.expire_all()
    after = _deleted_flags(db_session)
    assert _committed_deleted(persona_id) is False
    for existing_id, was_deleted in before.items():
        assert after[existing_id] is was_deleted
    assert isinstance(error, _REFUSAL_TYPES)


def _new_persona(
    db_session: Session,
    user: User | None,
    *,
    name_prefix: str,
    is_public: bool,
    builtin_persona: bool = False,
) -> Persona:
    return upsert_persona(
        user=user,
        name=f"{name_prefix}-{uuid4().hex[:8]}",
        description="",
        llm_model_provider_override=None,
        llm_model_version_override=None,
        starter_messages=None,
        system_prompt="",
        task_prompt="",
        datetime_aware=True,
        is_public=is_public,
        db_session=db_session,
        builtin_persona=builtin_persona,
        commit=True,
    )


def _ensure_search_tool_can_be_attached(db_session: Session) -> None:
    """Slack channel agents are created with the built-in search tool.

    ``SearchTool.is_available`` is false until some connector other than the
    ingestion-api default exists. A file connector is enough for that check.
    """
    if check_connectors_exist(db_session):
        return
    db_session.add(
        Connector(
            name=f"file-source-{uuid4().hex[:8]}",
            source=DocumentSource.FILE,
            input_type=InputType.LOAD_STATE,
            connector_specific_config={},
            refresh_freq=None,
            prune_freq=None,
        )
    )
    db_session.commit()


def test_basic_user_cannot_delete_seeded_default_assistant(
    db_session: Session,
) -> None:
    """DELETE of persona 0 by a basic user must leave the default assistant intact.

    Listings and a non-edit lookup must still return it. The Slack handler in
    ``handle_regular_answer`` loads this id and filters out deleted rows, so a
    committed delete makes that lookup raise.
    """
    _restore_default_assistant(db_session)
    attacker = _pass_basic_access_gate(_create_signed_in_user(db_session, "default_delete_attacker"))
    _add_unrelated_group_memberships(db_session, attacker)

    seeded = db_session.get(Persona, DEFAULT_PERSONA_ID)
    assert seeded is not None
    assert seeded.user_id is None
    assert seeded.builtin_persona is True
    assert seeded.deleted is False
    assert seeded.is_public is True
    assert seeded.is_listed is True

    try:
        _require_refused(
            db_session,
            lambda: delete_persona(
                persona_id=DEFAULT_PERSONA_ID,
                user=attacker,
                db_session=db_session,
            ),
            DEFAULT_PERSONA_ID,
        )

        listed = get_minimal_persona_snapshots_for_user(
            user=attacker,
            include_deleted=False,
            db_session=db_session,
            get_editable=False,
        )
        assert DEFAULT_PERSONA_ID in {persona.id for persona in listed}

        readable = get_persona_by_id(
            persona_id=DEFAULT_PERSONA_ID,
            user=attacker,
            db_session=db_session,
            is_for_edit=False,
        )
        assert readable.id == DEFAULT_PERSONA_ID
        assert readable.deleted is False
    finally:
        _restore_default_assistant(db_session)


def test_basic_user_cannot_delete_slack_channel_agent(db_session: Session) -> None:
    """Slack channel agents are created with ``user=None`` and must not be deletable."""
    attacker = _pass_basic_access_gate(_create_signed_in_user(db_session, "slack_delete_attacker"))
    _add_unrelated_group_memberships(db_session, attacker)
    _ensure_search_tool_can_be_attached(db_session)

    slack_persona = create_slack_channel_persona(
        db_session=db_session,
        channel_name=f"chan-{uuid4().hex[:8]}",
        document_set_ids=[],
    )
    db_session.commit()
    assert slack_persona.user_id is None
    assert slack_persona.builtin_persona is False
    assert slack_persona.deleted is False

    _require_refused(
        db_session,
        lambda: delete_persona(
            persona_id=slack_persona.id,
            user=attacker,
            db_session=db_session,
        ),
        slack_persona.id,
    )


def test_basic_user_cannot_delete_persona_orphaned_by_user_deletion(
    db_session: Session,
) -> None:
    """Deleting a user nulls persona ownership. That row must not become editable."""
    attacker = _pass_basic_access_gate(_create_signed_in_user(db_session, "orphan_delete_attacker"))
    owner = _create_signed_in_user(db_session, "orphan_owner")
    private_persona = _new_persona(
        db_session,
        owner,
        name_prefix="owned-private",
        is_public=False,
    )
    assert private_persona.user_id == owner.id

    delete_user_from_db(owner, db_session)
    db_session.expire_all()
    orphaned = db_session.get(Persona, private_persona.id)
    assert orphaned is not None
    assert orphaned.user_id is None
    assert orphaned.is_public is False
    assert orphaned.deleted is False

    _require_refused(
        db_session,
        lambda: delete_persona(
            persona_id=orphaned.id,
            user=attacker,
            db_session=db_session,
        ),
        orphaned.id,
    )


def test_basic_user_cannot_delete_builtin_persona_they_own(
    db_session: Session,
) -> None:
    """Builtin agents stay, including when the caller is the stored owner."""
    attacker = _pass_basic_access_gate(_create_signed_in_user(db_session, "builtin_owner"))
    builtin = _new_persona(
        db_session,
        attacker,
        name_prefix="builtin-owned",
        is_public=False,
        builtin_persona=True,
    )
    assert builtin.user_id == attacker.id
    assert builtin.builtin_persona is True

    _require_refused(
        db_session,
        lambda: delete_persona(
            persona_id=builtin.id,
            user=attacker,
            db_session=db_session,
        ),
        builtin.id,
    )


def test_admin_cannot_delete_default_assistant(db_session: Session) -> None:
    """The default assistant id is not deletable through this endpoint by a user."""
    _restore_default_assistant(db_session)
    admin = _pass_basic_access_gate(_create_signed_in_user(db_session, "admin_default_delete", is_admin=True))
    assert admin.role == UserRole.ADMIN

    try:
        _require_refused(
            db_session,
            lambda: delete_persona(
                persona_id=DEFAULT_PERSONA_ID,
                user=admin,
                db_session=db_session,
            ),
            DEFAULT_PERSONA_ID,
        )
    finally:
        _restore_default_assistant(db_session)


def test_basic_user_cannot_undelete_ownerless_persona(db_session: Session) -> None:
    """``mark_persona_as_not_deleted`` uses the same edit lookup as deletion."""
    attacker = _pass_basic_access_gate(_create_signed_in_user(db_session, "undelete_attacker"))
    ownerless = _new_persona(
        db_session,
        None,
        name_prefix="ownerless-undelete",
        is_public=False,
    )
    assert ownerless.user_id is None

    mark_persona_as_deleted(
        persona_id=ownerless.id,
        user=None,  # type: ignore[arg-type]
        db_session=db_session,
    )
    assert _committed_deleted(ownerless.id) is True

    error: Exception | None = None
    try:
        mark_persona_as_not_deleted(
            persona_id=ownerless.id,
            user=attacker,
            db_session=db_session,
        )
    except Exception as exc:
        error = exc

    assert _committed_deleted(ownerless.id) is True
    assert isinstance(error, _REFUSAL_TYPES)


def test_basic_user_can_delete_own_non_builtin_persona(db_session: Session) -> None:
    attacker = _pass_basic_access_gate(_create_signed_in_user(db_session, "own_delete"))
    owned = _new_persona(
        db_session,
        attacker,
        name_prefix="own-agent",
        is_public=False,
    )
    other = _new_persona(
        db_session,
        _create_signed_in_user(db_session, "other_owner"),
        name_prefix="other-agent",
        is_public=False,
    )

    delete_persona(persona_id=owned.id, user=attacker, db_session=db_session)

    assert _committed_deleted(owned.id) is True
    assert _committed_deleted(other.id) is False


def test_basic_user_cannot_delete_another_users_persona(db_session: Session) -> None:
    attacker = _pass_basic_access_gate(_create_signed_in_user(db_session, "cross_delete_attacker"))
    victim = _new_persona(
        db_session,
        _create_signed_in_user(db_session, "cross_delete_owner"),
        name_prefix="victim-agent",
        is_public=True,
    )
    assert victim.user_id is not None
    assert victim.user_id != attacker.id

    _require_refused(
        db_session,
        lambda: delete_persona(
            persona_id=victim.id,
            user=attacker,
            db_session=db_session,
        ),
        victim.id,
    )


def test_admin_can_delete_non_builtin_ownerless_persona(db_session: Session) -> None:
    """Admins keep the ability to remove a Slack-style agent that has no owner."""
    admin = _pass_basic_access_gate(_create_signed_in_user(db_session, "admin_slack_delete", is_admin=True))
    _ensure_search_tool_can_be_attached(db_session)
    slack_persona = create_slack_channel_persona(
        db_session=db_session,
        channel_name=f"admin-chan-{uuid4().hex[:8]}",
        document_set_ids=[],
    )
    db_session.commit()
    assert slack_persona.user_id is None
    assert slack_persona.builtin_persona is False

    delete_persona(persona_id=slack_persona.id, user=admin, db_session=db_session)

    assert _committed_deleted(slack_persona.id) is True


def test_system_path_without_user_may_delete_default_assistant(
    db_session: Session,
) -> None:
    """A caller with no user is the system path and may delete the default agent."""
    _restore_default_assistant(db_session)
    try:
        mark_persona_as_deleted(
            persona_id=DEFAULT_PERSONA_ID,
            user=None,  # type: ignore[arg-type]
            db_session=db_session,
        )
        assert _committed_deleted(DEFAULT_PERSONA_ID) is True
    finally:
        _restore_default_assistant(db_session)
