"""Authorization tests for POST /chat/seed-chat-session-from-slack.

A signed-in basic user who knows another user's chat session id must not be
able to copy that conversation into their own account. Seeding is only
allowed for an OnyxBot Slack session the caller is entitled to: the session
has onyxbot_flow set, and its user_id is either unset or the caller.

The copy must use the messages loaded with the session that was authorized.
A later lookup of that session id must not be able to substitute a different
conversation. A refusal must not leave a new chat session or copied messages.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import event
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.configs.constants import MessageType
from onyx.db.chat import create_chat_session
from onyx.db.chat import create_new_chat_message
from onyx.db.chat import get_chat_messages_by_session
from onyx.db.chat import get_chat_session_by_id
from onyx.db.chat import get_or_create_root_message
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.enums import ChatSessionSharedStatus
from onyx.db.enums import Permission
from onyx.db.models import ChatMessage
from onyx.db.models import ChatSession
from onyx.db.models import User
from onyx.server.query_and_chat.chat_backend import router as chat_router
from onyx.server.query_and_chat.chat_backend import SeedChatFromSlackRequest
from onyx.server.query_and_chat.chat_backend import seed_chat_from_slack
from tests.external_dependency_unit.conftest import create_test_user


def _basic_user(db_session: Session, prefix: str) -> User:
    user = create_test_user(db_session, prefix)
    user.effective_permissions = [Permission.BASIC_ACCESS.value]
    db_session.commit()
    db_session.refresh(user)
    return user


def _assert_passes_basic_access_gate(user: User) -> None:
    """The route's only gate is require_permission(BASIC_ACCESS)."""
    dependency = require_permission(Permission.BASIC_ACCESS)
    allowed = asyncio.run(dependency(user=user))
    assert allowed.id == user.id


def _session_with_messages(
    db_session: Session,
    *,
    owner_id: UUID | None,
    texts: list[str],
    onyxbot_flow: bool,
    shared_status: ChatSessionSharedStatus = ChatSessionSharedStatus.PRIVATE,
) -> ChatSession:
    chat_session = create_chat_session(
        db_session=db_session,
        description="seed-auth-fixture",
        user_id=owner_id,
        persona_id=None,
        onyxbot_flow=onyxbot_flow,
    )
    chat_session.shared_status = shared_status
    db_session.commit()

    parent = get_or_create_root_message(chat_session_id=chat_session.id, db_session=db_session)
    for index, body in enumerate(texts):
        message_type = MessageType.USER if index % 2 == 0 else MessageType.ASSISTANT
        reasoning = f"reasoning:{body}" if message_type == MessageType.ASSISTANT else None
        parent = create_new_chat_message(
            chat_session_id=chat_session.id,
            parent_message=parent,
            message=body,
            token_count=len(body),
            message_type=message_type,
            db_session=db_session,
            reasoning_tokens=reasoning,
            error=None,
            commit=True,
        )
    db_session.refresh(chat_session)
    return chat_session


def _owned_session_ids(db_session: Session, user_id: UUID) -> set[UUID]:
    rows = db_session.execute(select(ChatSession.id).where(ChatSession.user_id == user_id)).scalars()
    return set(rows.all())


def _message_count(db_session: Session) -> int:
    return int(db_session.execute(select(func.count()).select_from(ChatMessage)).scalar_one())


def _texts_for_user(db_session: Session, user_id: UUID) -> list[str]:
    rows = db_session.execute(
        select(ChatMessage.message)
        .join(ChatSession, ChatSession.id == ChatMessage.chat_session_id)
        .where(ChatSession.user_id == user_id)
    ).scalars()
    return list(rows.all())


def _seed(db_session: Session, user: User, chat_session_id: UUID) -> str:
    response = seed_chat_from_slack(
        SeedChatFromSlackRequest(chat_session_id=chat_session_id),
        user=user,
        db_session=db_session,
    )
    return response.redirect_url


def _assert_seed_refused(
    db_session: Session,
    attacker: User,
    source_id: UUID,
    secrets: list[str],
) -> None:
    sessions_before = _owned_session_ids(db_session, attacker.id)
    messages_before = _message_count(db_session)
    try:
        redirect_url = _seed(db_session, attacker, source_id)
    except Exception:
        db_session.rollback()
    else:
        new_id = _redirect_session_id(redirect_url)
        leaked = get_chat_messages_by_session(
            chat_session_id=new_id,
            user_id=attacker.id,
            db_session=db_session,
            skip_permission_check=False,
        )
        bodies = [message.message for message in leaked]
        pytest.fail(f"seed-chat-session-from-slack copied {source_id} into {new_id} for {attacker.id}: {bodies}")

    assert _owned_session_ids(db_session, attacker.id) == sessions_before
    assert _message_count(db_session) == messages_before
    copied = "\n".join(_texts_for_user(db_session, attacker.id))
    for secret in secrets:
        assert secret not in copied


def _redirect_session_id(redirect_url: str) -> UUID:
    marker = "chatId="
    assert marker in redirect_url
    return UUID(redirect_url.split(marker, 1)[1].split("&", 1)[0])


def _assert_seed_copies_only(
    db_session: Session,
    attacker: User,
    source: ChatSession,
    expected_texts: list[str],
    forbidden_texts: list[str],
) -> None:
    redirect_url = _seed(db_session, attacker, source.id)
    new_id = _redirect_session_id(redirect_url)
    assert new_id != source.id

    copied_session = get_chat_session_by_id(
        chat_session_id=new_id,
        user_id=attacker.id,
        db_session=db_session,
    )
    assert copied_session.user_id == attacker.id

    # Read through the permissioned message loader the chat UI uses.
    messages = get_chat_messages_by_session(
        chat_session_id=new_id,
        user_id=attacker.id,
        db_session=db_session,
        skip_permission_check=False,
    )
    bodies = [message.message for message in messages]
    for expected in expected_texts:
        assert expected in bodies
    for forbidden in forbidden_texts:
        assert forbidden not in bodies
        assert forbidden not in "\n".join(message.reasoning_tokens or "" for message in messages)


def test_seed_route_is_exposed_to_basic_users() -> None:
    """The handler is mounted and its user dependency is the basic-access gate."""
    matches = [route for route in chat_router.routes if getattr(route, "path", None) == "/chat/seed-chat-session-from-slack"]
    assert len(matches) == 1
    route = matches[0]
    assert "POST" in getattr(route, "methods", set())
    assert route.endpoint is seed_chat_from_slack  # type: ignore[attr-defined]

    dependency_calls = [
        dependency.call
        for dependency in route.dependant.dependencies  # type: ignore[attr-defined]
    ]
    assert any(getattr(call, "_is_require_permission", False) for call in dependency_calls)


def test_basic_user_cannot_seed_another_users_private_web_chat(
    db_session: Session,
) -> None:
    victim = _basic_user(db_session, "seed-victim-private")
    attacker = _basic_user(db_session, "seed-attacker-private")
    _assert_passes_basic_access_gate(attacker)

    secret = "private-web-secret-do-not-copy"
    later = "private-web-secret-added-later"
    source = _session_with_messages(
        db_session,
        owner_id=victim.id,
        texts=[secret, later],
        onyxbot_flow=False,
    )
    with pytest.raises(ValueError):
        get_chat_session_by_id(
            chat_session_id=source.id,
            user_id=attacker.id,
            db_session=db_session,
        )

    _assert_seed_refused(db_session, attacker, source.id, [secret, later])


def test_basic_user_cannot_seed_chat_after_public_sharing_is_turned_off(
    db_session: Session,
) -> None:
    victim = _basic_user(db_session, "seed-victim-unshared")
    attacker = _basic_user(db_session, "seed-attacker-unshared")
    _assert_passes_basic_access_gate(attacker)

    while_public = "secret-while-link-was-public"
    after_private = "secret-added-after-sharing-turned-off"
    source = _session_with_messages(
        db_session,
        owner_id=victim.id,
        texts=[while_public],
        onyxbot_flow=False,
        shared_status=ChatSessionSharedStatus.PUBLIC,
    )
    source.shared_status = ChatSessionSharedStatus.PRIVATE
    db_session.commit()
    parent = get_or_create_root_message(chat_session_id=source.id, db_session=db_session)
    # The chain walker is not required; the seed copies every non-system row.
    create_new_chat_message(
        chat_session_id=source.id,
        parent_message=parent,
        message=after_private,
        token_count=len(after_private),
        message_type=MessageType.USER,
        db_session=db_session,
        commit=True,
    )
    with pytest.raises(ValueError):
        get_chat_session_by_id(
            chat_session_id=source.id,
            user_id=attacker.id,
            db_session=db_session,
        )

    _assert_seed_refused(db_session, attacker, source.id, [while_public, after_private])


def test_basic_user_cannot_seed_another_users_slack_bot_session(
    db_session: Session,
) -> None:
    victim = _basic_user(db_session, "seed-victim-slack")
    attacker = _basic_user(db_session, "seed-attacker-slack")
    _assert_passes_basic_access_gate(attacker)

    secret = "other-users-slack-thread-secret"
    source = _session_with_messages(
        db_session,
        owner_id=victim.id,
        texts=[secret],
        onyxbot_flow=True,
    )
    with pytest.raises(ValueError):
        get_chat_session_by_id(
            chat_session_id=source.id,
            user_id=attacker.id,
            db_session=db_session,
        )

    _assert_seed_refused(db_session, attacker, source.id, [secret])


def test_basic_user_cannot_seed_unassigned_web_session(db_session: Session) -> None:
    attacker = _basic_user(db_session, "seed-attacker-unassigned-web")
    _assert_passes_basic_access_gate(attacker)

    secret = "unassigned-web-session-secret"
    source = _session_with_messages(
        db_session,
        owner_id=None,
        texts=[secret],
        onyxbot_flow=False,
    )

    _assert_seed_refused(db_session, attacker, source.id, [secret])


def test_basic_user_cannot_seed_their_own_non_slack_session(
    db_session: Session,
) -> None:
    attacker = _basic_user(db_session, "seed-own-web")
    _assert_passes_basic_access_gate(attacker)

    secret = "own-web-session-is-not-a-slack-seed"
    source = _session_with_messages(
        db_session,
        owner_id=attacker.id,
        texts=[secret],
        onyxbot_flow=False,
    )

    sessions_before = _owned_session_ids(db_session, attacker.id)
    _assert_seed_refused(db_session, attacker, source.id, [secret])
    # The caller's own web session is still there, and it was not duplicated.
    assert source.id in _owned_session_ids(db_session, attacker.id)
    assert _owned_session_ids(db_session, attacker.id) == sessions_before


def test_owner_can_seed_their_slack_bot_session(db_session: Session) -> None:
    owner = _basic_user(db_session, "seed-slack-owner")
    _assert_passes_basic_access_gate(owner)

    question = "slack-owner-question"
    answer = "slack-owner-answer"
    source = _session_with_messages(
        db_session,
        owner_id=owner.id,
        texts=[question, answer],
        onyxbot_flow=True,
    )

    _assert_seed_copies_only(
        db_session,
        owner,
        source,
        expected_texts=[question, answer],
        forbidden_texts=[],
    )
    new_ids = _owned_session_ids(db_session, owner.id) - {source.id}
    assert len(new_ids) == 1
    new_messages = get_chat_messages_by_session(
        chat_session_id=next(iter(new_ids)),
        user_id=owner.id,
        db_session=db_session,
        skip_permission_check=False,
    )
    reasoning = [message.reasoning_tokens for message in new_messages]
    assert f"reasoning:{answer}" in reasoning


def test_basic_user_can_seed_unassigned_slack_bot_session(db_session: Session) -> None:
    """Slack standard-answer sessions are created with no Onyx user id.

    The continue-in-web button addresses that session id. A basic user may
    seed that unassigned OnyxBot session into their own account.
    """
    attacker = _basic_user(db_session, "seed-unassigned-slack")
    _assert_passes_basic_access_gate(attacker)

    question = "unassigned-slack-question"
    source = _session_with_messages(
        db_session,
        owner_id=None,
        texts=[question],
        onyxbot_flow=True,
    )

    _assert_seed_copies_only(
        db_session,
        attacker,
        source,
        expected_texts=[question],
        forbidden_texts=[],
    )


class _MessageSwap:
    """Move another session's messages onto the source id between authorization and use.

    The swap runs only when the seed path has already read the source session
    and written a new session before reading the source messages, or after the
    source messages have been read once and a later statement reads them again.
    A copy that keeps the messages loaded with the authorized session does not
    observe the substituted rows.
    """

    def __init__(self, source_id: UUID, victim_id: UUID) -> None:
        self.source_id = source_id
        self.victim_id = victim_id
        self.saw_source_session = False
        self.saw_source_messages = False
        self.wrote_new_session = False
        self.swapped = False
        self.in_swap = False
        self.statements: list[str] = []

    def _blob(self, parameters: Any) -> str:
        return repr(parameters)

    def _is_source(self, parameters: Any) -> bool:
        return str(self.source_id) in self._blob(parameters)

    def _swap(self) -> None:
        if self.swapped:
            return
        self.swapped = True
        self.in_swap = True
        try:
            with get_session_with_current_tenant() as other:
                other.execute(
                    update(ChatMessage)
                    .where(ChatMessage.chat_session_id == self.victim_id)
                    .values(chat_session_id=self.source_id)
                )
                other.commit()
        finally:
            self.in_swap = False

    def before(
        self,
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        del conn, cursor, context, executemany
        if self.in_swap:
            return
        sql = " ".join(statement.lower().split())
        self.statements.append(sql[:180])
        if sql.startswith("insert") and "chat_session" in sql and "chat_message" not in sql:
            self.wrote_new_session = True
        if (
            self.saw_source_session
            and self.wrote_new_session
            and not self.saw_source_messages
            and sql.startswith("select")
            and "chat_message" in sql
            and self._is_source(parameters)
        ):
            self._swap()

    def after(
        self,
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        del conn, cursor, context, executemany
        if self.in_swap:
            return
        sql = " ".join(statement.lower().split())
        if (
            sql.startswith("select")
            and " from chat_session" in f" {sql} "
            and "chat_message" not in sql
            and self._is_source(parameters)
        ):
            self.saw_source_session = True
        if sql.startswith("select") and "chat_message" in sql and self._is_source(parameters) and not self.wrote_new_session:
            self.saw_source_messages = True
            self._swap()


def test_seed_does_not_copy_messages_from_a_later_lookup(
    db_session: Session,
) -> None:
    """The messages copied are the ones on the session when it was authorized.

    After that read, another transaction moves a different user's messages onto
    the same session id. A second lookup would copy those messages. Acting on
    the authorized snapshot does not.
    """
    owner = _basic_user(db_session, "seed-toctou-owner")
    victim = _basic_user(db_session, "seed-toctou-victim")
    _assert_passes_basic_access_gate(owner)

    original = "authorized-slack-message"
    substituted = "substituted-victim-message-after-authorization"
    source = _session_with_messages(
        db_session,
        owner_id=owner.id,
        texts=[original],
        onyxbot_flow=True,
    )
    victim_session = _session_with_messages(
        db_session,
        owner_id=victim.id,
        texts=[substituted],
        onyxbot_flow=False,
    )

    # Commits inside the seed path return the connection to the pool, so the
    # listener has to sit on the engine rather than one connection.
    engine = db_session.get_bind()
    swap = _MessageSwap(source.id, victim_session.id)
    event.listen(engine, "before_cursor_execute", swap.before)
    event.listen(engine, "after_cursor_execute", swap.after)
    try:
        redirect_url = _seed(db_session, owner, source.id)
    finally:
        event.remove(engine, "before_cursor_execute", swap.before)
        event.remove(engine, "after_cursor_execute", swap.after)

    new_id = _redirect_session_id(redirect_url)
    messages = get_chat_messages_by_session(
        chat_session_id=new_id,
        user_id=owner.id,
        db_session=db_session,
        skip_permission_check=False,
    )
    bodies = [message.message for message in messages]
    assert substituted not in bodies
    assert original in bodies


def test_refused_seed_does_not_create_a_session_before_raising(
    db_session: Session,
) -> None:
    """Creating the destination session and then refusing is still a failure."""
    victim = _basic_user(db_session, "seed-side-effect-victim")
    attacker = _basic_user(db_session, "seed-side-effect-attacker")
    secret = "side-effect-secret"
    source = _session_with_messages(
        db_session,
        owner_id=victim.id,
        texts=[secret],
        onyxbot_flow=False,
    )
    _assert_seed_refused(db_session, attacker, source.id, [secret])
