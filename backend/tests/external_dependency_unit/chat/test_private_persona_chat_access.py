"""Authorization tests for persona ids supplied when a chat session is created.

These tests exercise the product path a signed-in basic user hits:

1. ``POST /api/chat/create-chat-session`` → ``create_new_chat_session``
2. ``POST /api/chat/send-chat-message`` with ``chat_session_info`` →
   ``handle_stream_message_objects`` → ``build_chat_turn``
3. ``POST /api/chat/send-chat-message`` with an existing ``chat_session_id``
   whose stored persona the caller cannot use

The model call itself is replaced after the prompt is built. File bytes come
from the real file store, and the prompt text is the list passed to
``LLM.stream``. The tests fail while a private, group-restricted, or deleted
persona can be selected and its instructions or attached file text reach that
prompt.
"""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from typing import Any
from uuid import UUID
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import event
from sqlalchemy.orm import Session

from onyx.chat.models import StreamingError
from onyx.chat.process_message import handle_stream_message_objects
from onyx.configs.constants import FileOrigin
from onyx.db.chat import create_chat_session
from onyx.db.enums import UserFileStatus
from onyx.db.llm import fetch_default_llm_model
from onyx.db.llm import update_default_provider
from onyx.db.llm import upsert_llm_provider
from onyx.db.models import ChatMessage
from onyx.db.models import ChatSession
from onyx.db.models import Persona
from onyx.db.models import Persona__User
from onyx.db.models import Persona__UserGroup
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.db.models import UserFile
from onyx.db.models import UserGroup
from onyx.db.models import UserRole
from onyx.db.persona import upsert_persona
from onyx.db.persona import user_can_access_persona
from onyx.error_handling.exceptions import OnyxError
from onyx.file_store.file_store import get_default_file_store
from onyx.file_store.utils import load_user_file
from onyx.file_store.utils import store_user_file_plaintext
from onyx.llm.constants import LlmProviderNames
from onyx.llm.model_response import Delta
from onyx.llm.model_response import ModelResponseStream
from onyx.llm.model_response import StreamingChoice
from onyx.llm.models import LanguageModelInput
from onyx.server.manage.llm.models import LLMProviderUpsertRequest
from onyx.server.manage.llm.models import ModelConfigurationUpsertRequest
from onyx.server.query_and_chat.chat_backend import create_new_chat_session
from onyx.server.query_and_chat.models import ChatSessionCreationRequest
from onyx.server.query_and_chat.models import SendMessageRequest
from onyx.server.query_and_chat.streaming_models import OverallStop
from onyx.server.query_and_chat.streaming_models import Packet
from tests.external_dependency_unit.conftest import create_test_user

_AUTH_ERROR_CODES = {
    "UNAUTHORIZED",
    "INSUFFICIENT_PERMISSIONS",
    "PERSONA_NOT_FOUND",
    "NOT_FOUND",
}
_AUTH_TEXT_MARKERS = (
    "does not have access",
    "not authorized",
    "unauthorized",
    "insufficient_permissions",
    "persona_not_found",
    "cannot access persona",
    "no access to persona",
    "access denied",
    "not permitted",
)


class _PrivatePersona(object):
    def __init__(
        self,
        persona: Persona,
        user_file: UserFile,
        file_secret: str,
        instruction_secret: str,
    ) -> None:
        self.persona = persona
        self.user_file = user_file
        self.file_secret = file_secret
        self.instruction_secret = instruction_secret


def _ensure_placeholder_llm_provider(db_session: Session) -> None:
    """Install a default provider without reading a real API key.

    ``LLM.stream`` is stubbed after the prompt is assembled, so this value is
    never sent anywhere.
    """
    if fetch_default_llm_model(db_session) is not None:
        return
    provider = upsert_llm_provider(
        llm_provider_upsert_request=LLMProviderUpsertRequest(
            name=f"persona-auth-test-{uuid4().hex[:8]}",
            provider=LlmProviderNames.OPENAI,
            api_key="placeholder",
            is_public=True,
            model_configurations=[
                ModelConfigurationUpsertRequest(name="gpt-4o-mini", is_visible=True)
            ],
            groups=[],
        ),
        db_session=db_session,
    )
    update_default_provider(provider.id, "gpt-4o-mini", db_session)


def _create_private_persona(
    db_session: Session,
    owner: User,
    *,
    is_public: bool = False,
    is_listed: bool = False,
) -> _PrivatePersona:
    file_secret = f"file-body-{uuid4().hex}"
    instruction_secret = f"instruction-body-{uuid4().hex}"
    file_store = get_default_file_store()
    file_store.initialize()
    stored_id = file_store.save_file(
        content=BytesIO(file_secret.encode("utf-8")),
        display_name="private.txt",
        file_origin=FileOrigin.USER_FILE,
        file_type="text/plain",
    )
    user_file = UserFile(
        id=uuid4(),
        user_id=owner.id,
        file_id=stored_id,
        name="private.txt",
        file_type="text/plain",
        status=UserFileStatus.COMPLETED,
        token_count=max(1, len(file_secret) // 4),
        chunk_count=1,
        needs_persona_sync=False,
        needs_project_sync=False,
    )
    db_session.add(user_file)
    db_session.commit()
    db_session.refresh(user_file)
    stored = store_user_file_plaintext(user_file.id, file_secret)
    assert stored, "plaintext cache for the persona file was not stored"

    persona = upsert_persona(
        user=owner,
        name=f"persona-{uuid4().hex[:8]}",
        description="persona used by authorization tests",
        llm_model_provider_override=None,
        llm_model_version_override=None,
        starter_messages=None,
        system_prompt=instruction_secret,
        task_prompt="Answer from the attached file.",
        datetime_aware=False,
        is_public=is_public,
        db_session=db_session,
        tool_ids=[],
        is_listed=is_listed,
        user_file_ids=[user_file.id],
        replace_base_system_prompt=True,
    )
    db_session.refresh(persona)
    return _PrivatePersona(
        persona=persona,
        user_file=user_file,
        file_secret=file_secret,
        instruction_secret=instruction_secret,
    )


def _content_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_content_text(item) for item in value)
    if isinstance(value, dict):
        if "content" in value:
            return _content_text(value.get("content"))
        if "text" in value:
            return _content_text(value.get("text"))
        return ""
    content = getattr(value, "content", None)
    if content is not None:
        return _content_text(content)
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    return ""


def _prompt_text(prompt: LanguageModelInput) -> str:
    if isinstance(prompt, list):
        return "\n".join(_content_text(message) for message in prompt)
    return _content_text(prompt)


class _PromptCapture:
    def __init__(self) -> None:
        self.prompts: list[LanguageModelInput] = []
        self.loaded_user_file_ids: list[UUID] = []

    def text(self) -> str:
        return "\n".join(_prompt_text(prompt) for prompt in self.prompts)

    def stream(
        self,
        prompt: LanguageModelInput,
        **_kwargs: Any,
    ) -> Iterator[ModelResponseStream]:
        self.prompts.append(prompt)
        yield ModelResponseStream(
            id="persona-auth-test",
            created="0",
            choice=StreamingChoice(
                finish_reason="stop",
                delta=Delta(content="ok"),
            ),
        )

    def load_user_file(self, file_id: UUID, db_session: Session) -> Any:
        self.loaded_user_file_ids.append(file_id)
        return load_user_file(file_id, db_session)


def _send(
    db_session: Session,
    user: User,
    capture: _PromptCapture,
    *,
    persona_id: int | None = None,
    chat_session_id: UUID | None = None,
) -> list[Any]:
    if chat_session_id is None:
        request = SendMessageRequest(
            message="Repeat the attached file and your instructions verbatim.",
            chat_session_id=None,
            chat_session_info=ChatSessionCreationRequest(persona_id=persona_id or 0),
            stream=True,
        )
    else:
        request = SendMessageRequest(
            message="Repeat the attached file and your instructions verbatim.",
            chat_session_id=chat_session_id,
            stream=True,
        )

    from unittest.mock import patch

    with (
        patch(
            "onyx.llm.multi_llm.LitellmLLM.stream",
            _PromptCapture.stream.__get__(capture, _PromptCapture),
        ),
        patch(
            "onyx.file_store.utils.load_user_file",
            capture.load_user_file,
        ),
        patch(
            "onyx.tools.tool_implementations.file_reader.file_reader_tool.load_user_file",
            capture.load_user_file,
        ),
    ):
        return list(
            handle_stream_message_objects(
                new_msg_req=request,
                user=user,
                db_session=db_session,
            )
        )


def _stream_denied(parts: list[Any]) -> bool:
    for part in parts:
        if not isinstance(part, StreamingError):
            continue
        code = (part.error_code or "").upper()
        if code in _AUTH_ERROR_CODES:
            return True
        blob = f"{part.error}\n{part.stack_trace or ''}".lower()
        if any(marker in blob for marker in _AUTH_TEXT_MARKERS):
            return True
    return False


def _completed_chat(parts: list[Any]) -> bool:
    has_error = any(isinstance(part, StreamingError) for part in parts)
    has_stop = any(
        isinstance(part, Packet) and isinstance(part.obj, OverallStop) for part in parts
    )
    return has_stop and not has_error


def _exception_is_denial(exc: Exception) -> bool:
    status: int | None
    if isinstance(exc, HTTPException):
        status = exc.status_code
    elif isinstance(exc, OnyxError):
        status = exc.status_code
    elif isinstance(exc, ValueError):
        message = str(exc).lower()
        return any(marker in message for marker in _AUTH_TEXT_MARKERS) or (
            "persona" in message and "delet" in message
        )
    else:
        return False
    return status in (400, 403, 404)


def _assert_attack_blocked(
    parts: list[Any],
    capture: _PromptCapture,
    private: _PrivatePersona,
    inserted_sessions: list[UUID],
    inserted_messages: list[int],
) -> None:
    prompt = capture.text()
    assert private.file_secret not in prompt
    assert private.instruction_secret not in prompt
    assert private.user_file.id not in capture.loaded_user_file_ids
    assert inserted_sessions == []
    assert inserted_messages == []
    assert not _completed_chat(parts)
    assert _stream_denied(parts), [
        (part.error_code, part.error)
        for part in parts
        if isinstance(part, StreamingError)
    ]


def _watch_inserts(
    persona_id: int, user_id: UUID
) -> tuple[list[UUID], list[int], Any, Any]:
    sessions: list[UUID] = []
    messages: list[int] = []

    def after_session_insert(
        _mapper: Any, _connection: Any, target: ChatSession
    ) -> None:
        if target.persona_id == persona_id and target.user_id == user_id:
            sessions.append(target.id)

    def after_message_insert(
        _mapper: Any, _connection: Any, target: ChatMessage
    ) -> None:
        messages.append(target.id)

    event.listen(ChatSession, "after_insert", after_session_insert)
    event.listen(ChatMessage, "after_insert", after_message_insert)
    return sessions, messages, after_session_insert, after_message_insert


def _unwatch(after_session_insert: Any, after_message_insert: Any) -> None:
    event.remove(ChatSession, "after_insert", after_session_insert)
    event.remove(ChatMessage, "after_insert", after_message_insert)


def _assert_allowed_prompt(
    parts: list[Any], capture: _PromptCapture, private: _PrivatePersona
) -> None:
    prompt = capture.text()
    assert private.file_secret in prompt
    assert private.instruction_secret in prompt
    assert private.user_file.id in capture.loaded_user_file_ids
    assert not any(isinstance(part, StreamingError) for part in parts)
    assert _completed_chat(parts)


@pytest.fixture()
def ready_db(tenant_context: None, db_session: Session) -> Session:  # noqa: ARG001
    _ensure_placeholder_llm_provider(db_session)
    return db_session


def _assert_access(
    db_session: Session, persona_id: int, user: User, allowed: bool
) -> None:
    assert (
        user_can_access_persona(db_session, persona_id, user, get_editable=False)
        is allowed
    )


def test_create_chat_session_rejects_other_users_private_persona(
    ready_db: Session,
) -> None:
    owner = create_test_user(ready_db, "persona_owner")
    attacker = create_test_user(ready_db, "persona_attacker")
    private = _create_private_persona(ready_db, owner)
    _assert_access(ready_db, private.persona.id, owner, True)
    _assert_access(ready_db, private.persona.id, attacker, False)
    inserted, messages, on_session, on_message = _watch_inserts(
        private.persona.id, attacker.id
    )
    try:
        try:
            created = create_new_chat_session(
                chat_session_creation_request=ChatSessionCreationRequest(
                    persona_id=private.persona.id,
                    description="should not be stored",
                ),
                user=attacker,
                db_session=ready_db,
            )
        except (HTTPException, OnyxError, ValueError) as exc:
            assert _exception_is_denial(exc)
            assert inserted == []
            assert messages == []
            return
        pytest.fail(
            f"create-chat-session stored a session for a private persona: {created}"
        )
    finally:
        _unwatch(on_session, on_message)


def test_send_with_chat_session_info_does_not_expose_private_persona(
    ready_db: Session,
) -> None:
    owner = create_test_user(ready_db, "send_owner")
    attacker = create_test_user(ready_db, "send_attacker")
    private = _create_private_persona(ready_db, owner)
    # The attacker can use their own persona. A check against that persona,
    # or against the default persona, must not then run the requested one.
    _create_private_persona(ready_db, attacker)
    capture = _PromptCapture()
    inserted, messages, on_session, on_message = _watch_inserts(
        private.persona.id, attacker.id
    )
    try:
        parts = _send(
            ready_db,
            attacker,
            capture,
            persona_id=private.persona.id,
        )
    finally:
        _unwatch(on_session, on_message)
    _assert_attack_blocked(parts, capture, private, inserted, messages)


def test_existing_session_does_not_reuse_unauthorized_persona(
    ready_db: Session,
) -> None:
    """A session row already pointing at a private persona must not be runnable.

    The row is inserted with the database helper, which is what a session
    created before an authorization check would look like. The product path
    under test is sending a message with that ``chat_session_id``.
    """
    owner = create_test_user(ready_db, "existing_owner")
    attacker = create_test_user(ready_db, "existing_attacker")
    private = _create_private_persona(ready_db, owner)
    chat_session = create_chat_session(
        db_session=ready_db,
        description="pre-existing unauthorized session",
        user_id=attacker.id,
        persona_id=private.persona.id,
    )
    capture = _PromptCapture()
    _inserted, messages, on_session, on_message = _watch_inserts(
        private.persona.id, attacker.id
    )
    try:
        parts = _send(
            ready_db,
            attacker,
            capture,
            chat_session_id=chat_session.id,
        )
    finally:
        _unwatch(on_session, on_message)
    # The session row already existed; creating another one is not required
    # for this path. New messages, file reads, and prompt contents are.
    _assert_attack_blocked(parts, capture, private, [], messages)


def test_deleted_persona_is_rejected_for_new_and_existing_sessions(
    ready_db: Session,
) -> None:
    owner = create_test_user(ready_db, "deleted_owner")
    attacker = create_test_user(ready_db, "deleted_attacker")
    private = _create_private_persona(ready_db, owner)
    private.persona.deleted = True
    ready_db.commit()

    inserted, messages, on_session, on_message = _watch_inserts(
        private.persona.id, attacker.id
    )
    try:
        try:
            created = create_new_chat_session(
                chat_session_creation_request=ChatSessionCreationRequest(
                    persona_id=private.persona.id
                ),
                user=attacker,
                db_session=ready_db,
            )
        except (HTTPException, OnyxError, ValueError) as exc:
            assert _exception_is_denial(exc)
            assert inserted == []
        else:
            pytest.fail(f"deleted persona was stored on a new session: {created}")
    finally:
        _unwatch(on_session, on_message)

    chat_session = create_chat_session(
        db_session=ready_db,
        description="session on a deleted persona",
        user_id=attacker.id,
        persona_id=private.persona.id,
    )
    capture = _PromptCapture()
    _inserted, messages, on_session, on_message = _watch_inserts(
        private.persona.id, attacker.id
    )
    try:
        parts = _send(
            ready_db,
            attacker,
            capture,
            chat_session_id=chat_session.id,
        )
    finally:
        _unwatch(on_session, on_message)
    _assert_attack_blocked(parts, capture, private, [], messages)


def test_group_restricted_persona_is_rejected_for_non_member(
    ready_db: Session,
) -> None:
    owner = create_test_user(ready_db, "group_owner")
    outsider = create_test_user(ready_db, "group_outsider")
    member = create_test_user(ready_db, "group_member")
    private = _create_private_persona(ready_db, owner)
    group = UserGroup(
        name=f"persona-group-{uuid4().hex[:8]}",
        is_up_to_date=True,
        is_up_for_deletion=False,
        is_default=False,
    )
    ready_db.add(group)
    ready_db.flush()
    ready_db.add(
        User__UserGroup(
            user_group_id=group.id,
            user_id=member.id,
            is_curator=False,
        )
    )
    ready_db.add(
        Persona__UserGroup(persona_id=private.persona.id, user_group_id=group.id)
    )
    ready_db.commit()
    _assert_access(ready_db, private.persona.id, member, True)
    _assert_access(ready_db, private.persona.id, outsider, False)

    member_capture = _PromptCapture()
    member_parts = _send(
        ready_db,
        member,
        member_capture,
        persona_id=private.persona.id,
    )
    _assert_allowed_prompt(member_parts, member_capture, private)

    capture = _PromptCapture()
    inserted, messages, on_session, on_message = _watch_inserts(
        private.persona.id, outsider.id
    )
    try:
        parts = _send(
            ready_db,
            outsider,
            capture,
            persona_id=private.persona.id,
        )
    finally:
        _unwatch(on_session, on_message)
    _assert_attack_blocked(parts, capture, private, inserted, messages)


def test_directly_shared_user_can_use_persona_and_outsider_cannot(
    ready_db: Session,
) -> None:
    owner = create_test_user(ready_db, "share_owner")
    shared = create_test_user(ready_db, "share_recipient")
    outsider = create_test_user(ready_db, "share_outsider")
    private = _create_private_persona(ready_db, owner)
    ready_db.add(Persona__User(persona_id=private.persona.id, user_id=shared.id))
    ready_db.commit()
    _assert_access(ready_db, private.persona.id, shared, True)
    _assert_access(ready_db, private.persona.id, outsider, False)

    shared_capture = _PromptCapture()
    shared_parts = _send(
        ready_db,
        shared,
        shared_capture,
        persona_id=private.persona.id,
    )
    _assert_allowed_prompt(shared_parts, shared_capture, private)

    capture = _PromptCapture()
    inserted, messages, on_session, on_message = _watch_inserts(
        private.persona.id, outsider.id
    )
    try:
        parts = _send(
            ready_db,
            outsider,
            capture,
            persona_id=private.persona.id,
        )
    finally:
        _unwatch(on_session, on_message)
    _assert_attack_blocked(parts, capture, private, inserted, messages)


def test_owner_admin_and_public_persona_still_receive_attached_file(
    ready_db: Session,
) -> None:
    owner = create_test_user(ready_db, "allowed_owner")
    admin = create_test_user(ready_db, "allowed_admin", role=UserRole.ADMIN)
    other = create_test_user(ready_db, "allowed_other")
    private = _create_private_persona(ready_db, owner)
    public = _create_private_persona(ready_db, owner, is_public=True, is_listed=True)
    _assert_access(ready_db, private.persona.id, owner, True)
    _assert_access(ready_db, private.persona.id, admin, True)
    _assert_access(ready_db, private.persona.id, other, False)
    _assert_access(ready_db, public.persona.id, other, True)

    owner_capture = _PromptCapture()
    owner_parts = _send(ready_db, owner, owner_capture, persona_id=private.persona.id)
    _assert_allowed_prompt(owner_parts, owner_capture, private)

    admin_capture = _PromptCapture()
    admin_parts = _send(ready_db, admin, admin_capture, persona_id=private.persona.id)
    _assert_allowed_prompt(admin_parts, admin_capture, private)

    public_capture = _PromptCapture()
    public_parts = _send(ready_db, other, public_capture, persona_id=public.persona.id)
    _assert_allowed_prompt(public_parts, public_capture, public)
