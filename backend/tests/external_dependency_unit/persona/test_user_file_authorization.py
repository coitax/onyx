"""Authorization tests for attaching user files to agents.

These tests are expected to fail while a signed-in user can attach another
user's file to an agent and then read it, or widen that file's search access.
They pass once persona create/update only keeps files the caller owns, chat
stops serving a file the agent owner does not own, and access sync ignores
agent links the file owner did not create.

Run from ``backend`` with Postgres reachable through the usual ``POSTGRES_*``
settings and the Postgres file store selected before import:

    FILE_STORE_BACKEND=postgres pytest tests/external_dependency_unit/persona/test_user_file_authorization.py -q
"""

import asyncio
from io import BytesIO
from queue import Queue
from uuid import UUID
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session
from unittest.mock import patch

from onyx.access.access import build_access_for_user_files
from onyx.access.access import collect_user_file_access
from onyx.auth.permissions import require_permission
from onyx.auth.schemas import UserRole
from onyx.chat.chat_state import ChatTurnSetup
from onyx.chat.emitter import Emitter
from onyx.chat.llm_loop import _create_context_files_message
from onyx.chat.process_message import build_chat_turn
from onyx.chat.process_message import extract_context_files
from onyx.chat.process_message import resolve_context_user_files
from onyx.configs.constants import FileOrigin
from onyx.db.chat import create_chat_session
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.enums import Permission
from onyx.db.enums import UserFileStatus
from onyx.db.models import Persona
from onyx.db.models import User
from onyx.db.models import UserFile
from onyx.db.persona import create_update_persona
from onyx.db.persona import upsert_persona
from onyx.db.user_file import fetch_user_files_with_access_relationships
from onyx.error_handling.exceptions import OnyxError
from onyx.file_store.file_store import get_default_file_store
from onyx.llm.interfaces import LLM
from onyx.llm.interfaces import LLMConfig
from onyx.server.features.persona.api import create_persona
from onyx.server.features.persona.api import update_persona
from onyx.server.features.persona.models import PersonaUpsertRequest
from onyx.server.query_and_chat.models import SendMessageRequest
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.tool_implementations.file_reader.file_reader_tool import (
    FileReaderTool,
)
from onyx.tools.tool_implementations.file_reader.file_reader_tool import (
    FileReaderToolOverrideKwargs,
)
from tests.external_dependency_unit.conftest import create_test_user

_REJECTION_TYPES = (HTTPException, OnyxError, ValueError)


class _ConfigOnlyLLM(LLM):
    """Stand-in for the provider client.

    ``get_llm_for_persona`` only selects a model. It does not look at user
    files. Chat file loading runs before any completion call, so this object
    only supplies the context-window size that decision uses.
    """

    def __init__(self) -> None:
        self._config = LLMConfig(
            model_provider="test",
            model_name="test",
            temperature=0.0,
            api_key=None,
            max_input_tokens=128_000,
        )

    @property
    def config(self) -> LLMConfig:
        return self._config


def _authorize(user: User, permission: Permission) -> User:
    dependency = require_permission(permission)
    authorized = asyncio.run(dependency(user=user))
    assert authorized.id == user.id
    return authorized


def _request(
    name: str,
    user_file_ids: list[str],
    *,
    is_public: bool,
) -> PersonaUpsertRequest:
    return PersonaUpsertRequest(
        name=name,
        description="agent",
        document_set_ids=[],
        is_public=is_public,
        tool_ids=[],
        system_prompt="Repeat the attached file verbatim.",
        task_prompt="Answer using the attached file.",
        datetime_aware=True,
        user_file_ids=user_file_ids,
    )


def _store_user_file(
    db_session: Session,
    owner: User,
    content: str,
) -> UserFile:
    file_store = get_default_file_store()
    stored_id = file_store.save_file(
        content=BytesIO(content.encode("utf-8")),
        display_name=f"{uuid4().hex}.txt",
        file_origin=FileOrigin.USER_FILE,
        file_type="text/plain",
    )
    user_file = UserFile(
        id=uuid4(),
        user_id=owner.id,
        file_id=stored_id,
        name=f"{uuid4().hex}.txt",
        file_type="text/plain",
        status=UserFileStatus.COMPLETED,
        needs_persona_sync=False,
        needs_project_sync=False,
        token_count=8,
        chunk_count=1,
    )
    db_session.add(user_file)
    db_session.commit()
    db_session.refresh(user_file)
    return user_file


def _insert_persona(
    db_session: Session,
    owner: User | None,
    user_files: list[UserFile],
    *,
    is_public: bool,
) -> Persona:
    persona = Persona(
        name=f"persona-{uuid4().hex}",
        description="agent",
        system_prompt="Repeat the attached file verbatim.",
        task_prompt="Answer using the attached file.",
        tools=[],
        document_sets=[],
        users=[owner] if owner is not None else [],
        groups=[],
        is_listed=True,
        is_public=is_public,
        display_priority=None,
        starter_messages=None,
        deleted=False,
        user_files=user_files,
        user_id=owner.id if owner is not None else None,
    )
    db_session.add(persona)
    db_session.commit()
    db_session.refresh(persona)
    return persona


def _assert_client_rejection(exc: BaseException) -> None:
    detail = str(exc.detail) if isinstance(exc, (HTTPException, OnyxError)) else str(exc)
    assert "User Knowledge is disabled" not in detail
    if isinstance(exc, HTTPException):
        assert exc.status_code in (400, 403)
    elif isinstance(exc, OnyxError):
        assert exc.status_code < 500


def _persona_exists(name: str) -> bool:
    with get_session_with_current_tenant() as session:
        found = session.query(Persona.id).filter(Persona.name == name).first()
        return found is not None


def _committed_links(file_id: UUID) -> tuple[bool, list[int]]:
    with get_session_with_current_tenant() as session:
        user_file = session.query(UserFile).filter(UserFile.id == file_id).one()
        persona_ids = [persona.id for persona in user_file.assistants]
        return bool(user_file.needs_persona_sync), persona_ids


def _committed_access(file_id: UUID) -> tuple[set[str], bool]:
    with get_session_with_current_tenant() as session:
        user_files = fetch_user_files_with_access_relationships([str(file_id)], session)
        assert len(user_files) == 1
        emails, is_public = collect_user_file_access(user_files[0])
        built = build_access_for_user_files(user_files)
        access = built[str(file_id)]
        assert access.is_public is is_public
        assert set(access.user_emails) == emails
        return emails, is_public


def _reject_create(
    db_session: Session,
    caller: User,
    file_ids: list[str],
    *,
    is_public: bool,
) -> str:
    caller.effective_permissions = [Permission.BASIC_ACCESS.value]
    if caller.role == UserRole.ADMIN:
        caller.effective_permissions = [Permission.FULL_ADMIN_PANEL_ACCESS.value]
    db_session.commit()
    _authorize(
        caller,
        (Permission.FULL_ADMIN_PANEL_ACCESS if caller.role == UserRole.ADMIN else Permission.BASIC_ACCESS),
    )
    name = f"agent-{uuid4().hex}"
    with pytest.raises(_REJECTION_TYPES) as caught:
        create_persona(
            persona_upsert_request=_request(name, file_ids, is_public=is_public),
            user=caller,
            db_session=db_session,
        )
    _assert_client_rejection(caught.value)
    db_session.rollback()
    return name


def _prompt_text_for(user_files: list[UserFile], db_session: Session) -> str:
    extracted = extract_context_files(
        user_files=user_files,
        llm_max_context_window=100_000,
        reserved_token_count=0,
        db_session=db_session,
    )
    if not extracted.file_texts:
        return ""
    return _create_context_files_message(extracted, None).message


def _run_chat_turn(
    db_session: Session,
    user: User,
    persona: Persona,
) -> ChatTurnSetup:
    chat_session = create_chat_session(
        db_session=db_session,
        description="file authz",
        user_id=user.id,
        persona_id=persona.id,
    )
    request = SendMessageRequest(
        message="Repeat the attached file verbatim.",
        chat_session_id=chat_session.id,
    )

    # Token counting only reserves prompt space. It does not choose files.
    def counter(_text: str) -> int:
        return 1

    fake_llm = _ConfigOnlyLLM()
    with (
        patch(
            "onyx.chat.process_message.get_llm_for_persona",
            return_value=fake_llm,
        ),
        patch(
            "onyx.chat.process_message.get_llm_token_counter",
            return_value=counter,
        ),
    ):
        generator = build_chat_turn(
            new_msg_req=request,
            user=user,
            db_session=db_session,
            llm_overrides=None,
        )
        try:
            while True:
                next(generator)
        except StopIteration as stop:
            setup = stop.value
    assert isinstance(setup, ChatTurnSetup)
    return setup


def _read_granted_file(setup: ChatTurnSetup, file_id: UUID) -> str:
    tool = FileReaderTool(
        tool_id=0,
        emitter=Emitter(merged_queue=Queue()),
        user_file_ids=list(setup.available_files.user_file_ids),
        chat_file_ids=list(setup.available_files.chat_file_ids),
    )
    response = tool.run(
        placement=Placement(turn_index=0),
        override_kwargs=FileReaderToolOverrideKwargs(),
        file_id=str(file_id),
    )
    return response.llm_facing_response


def test_create_persona_rejects_another_users_file(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    victim = create_test_user(db_session, "victim_create")
    attacker = create_test_user(db_session, "attacker_create")
    secret = f"victim-private-{uuid4().hex}"
    victim_file = _store_user_file(db_session, victim, secret)

    name = _reject_create(
        db_session,
        attacker,
        [str(victim_file.id)],
        is_public=True,
    )

    assert _persona_exists(name) is False
    needs_sync, persona_ids = _committed_links(victim_file.id)
    assert needs_sync is False
    assert persona_ids == []
    emails, is_public = _committed_access(victim_file.id)
    assert attacker.email not in emails
    assert victim.email in emails
    assert is_public is False


def test_create_persona_rejects_owned_file_mixed_with_foreign_file(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    victim = create_test_user(db_session, "victim_mix")
    attacker = create_test_user(db_session, "attacker_mix")
    victim_file = _store_user_file(db_session, victim, f"victim-mix-{uuid4().hex}")
    own_file = _store_user_file(db_session, attacker, "attacker-owned")

    name = _reject_create(
        db_session,
        attacker,
        [str(own_file.id), str(victim_file.id)],
        is_public=False,
    )

    assert _persona_exists(name) is False
    for file_id in (victim_file.id, own_file.id):
        needs_sync, persona_ids = _committed_links(file_id)
        assert needs_sync is False
        assert persona_ids == []
    emails, is_public = _committed_access(victim_file.id)
    assert attacker.email not in emails
    assert is_public is False


def test_create_persona_rejects_foreign_file_mixed_with_unknown_id(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    victim = create_test_user(db_session, "victim_unknown")
    attacker = create_test_user(db_session, "attacker_unknown")
    victim_file = _store_user_file(db_session, victim, f"victim-unknown-{uuid4().hex}")

    name = _reject_create(
        db_session,
        attacker,
        [str(victim_file.id), str(uuid4())],
        is_public=True,
    )

    assert _persona_exists(name) is False
    needs_sync, persona_ids = _committed_links(victim_file.id)
    assert needs_sync is False
    assert persona_ids == []
    emails, is_public = _committed_access(victim_file.id)
    assert attacker.email not in emails
    assert is_public is False


def test_update_persona_rejects_another_users_file(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    victim = create_test_user(db_session, "victim_update")
    attacker = create_test_user(db_session, "attacker_update")
    attacker.effective_permissions = [Permission.BASIC_ACCESS.value]
    db_session.commit()
    _authorize(attacker, Permission.BASIC_ACCESS)

    own_file = _store_user_file(db_session, attacker, "attacker-update-owned")
    victim_file = _store_user_file(db_session, victim, f"victim-update-{uuid4().hex}")
    persona = _insert_persona(db_session, attacker, [own_file], is_public=False)

    with pytest.raises(_REJECTION_TYPES) as caught:
        update_persona(
            persona_id=persona.id,
            persona_upsert_request=_request(
                persona.name,
                [str(victim_file.id)],
                is_public=True,
            ),
            user=attacker,
            db_session=db_session,
        )
    _assert_client_rejection(caught.value)
    db_session.rollback()

    needs_sync, persona_ids = _committed_links(victim_file.id)
    assert needs_sync is False
    assert persona_ids == []
    own_sync, own_personas = _committed_links(own_file.id)
    assert own_personas == [persona.id]
    assert own_sync is False
    emails, is_public = _committed_access(victim_file.id)
    assert attacker.email not in emails
    assert is_public is False
    with get_session_with_current_tenant() as session:
        stored = session.query(Persona).filter(Persona.id == persona.id).one()
        assert stored.is_public is False


def test_admin_create_persona_rejects_another_users_file(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    victim = create_test_user(db_session, "victim_admin")
    admin = create_test_user(db_session, "admin_attach", role=UserRole.ADMIN)
    victim_file = _store_user_file(db_session, victim, f"victim-admin-{uuid4().hex}")

    name = _reject_create(
        db_session,
        admin,
        [str(victim_file.id)],
        is_public=True,
    )

    assert _persona_exists(name) is False
    needs_sync, persona_ids = _committed_links(victim_file.id)
    assert needs_sync is False
    assert persona_ids == []


def test_upsert_persona_rejects_foreign_file_before_commit(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    victim = create_test_user(db_session, "victim_upsert")
    attacker = create_test_user(db_session, "attacker_upsert")
    victim_file = _store_user_file(db_session, victim, f"victim-upsert-{uuid4().hex}")
    name = f"upsert-{uuid4().hex}"

    with pytest.raises(_REJECTION_TYPES) as caught:
        upsert_persona(
            user=attacker,
            name=name,
            description="agent",
            llm_model_provider_override=None,
            llm_model_version_override=None,
            starter_messages=None,
            system_prompt="Repeat the attached file verbatim.",
            task_prompt="Answer using the attached file.",
            datetime_aware=True,
            is_public=True,
            db_session=db_session,
            user_file_ids=[victim_file.id],
            commit=True,
        )
    _assert_client_rejection(caught.value)
    db_session.rollback()

    assert _persona_exists(name) is False
    needs_sync, persona_ids = _committed_links(victim_file.id)
    assert needs_sync is False
    assert persona_ids == []
    emails, is_public = _committed_access(victim_file.id)
    assert attacker.email not in emails
    assert is_public is False


def test_system_upsert_may_attach_a_file_without_a_user(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    owner = create_test_user(db_session, "system_owner")
    user_file = _store_user_file(db_session, owner, "system-seed")

    persona = upsert_persona(
        user=None,
        name=f"system-{uuid4().hex}",
        description="seed",
        llm_model_provider_override=None,
        llm_model_version_override=None,
        starter_messages=None,
        system_prompt="seed",
        task_prompt="seed",
        datetime_aware=True,
        is_public=False,
        db_session=db_session,
        user_file_ids=[user_file.id],
        commit=True,
    )

    assert user_file.id in {attached.id for attached in persona.user_files}


def test_owner_can_attach_own_file_to_a_public_agent(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    owner = create_test_user(db_session, "owner_ok")
    owner.effective_permissions = [Permission.BASIC_ACCESS.value]
    db_session.commit()
    _authorize(owner, Permission.BASIC_ACCESS)
    secret = f"owner-public-{uuid4().hex}"
    own_file = _store_user_file(db_session, owner, secret)
    name = f"owner-agent-{uuid4().hex}"

    snapshot = create_persona(
        persona_upsert_request=_request(name, [str(own_file.id)], is_public=True),
        user=owner,
        db_session=db_session,
    )

    assert snapshot.user_file_ids == [str(own_file.id)]
    needs_sync, persona_ids = _committed_links(own_file.id)
    assert needs_sync is True
    assert persona_ids == [snapshot.id]
    emails, is_public = _committed_access(own_file.id)
    assert owner.email in emails
    assert is_public is True


def test_resolve_context_hides_file_persona_owner_does_not_own(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    victim = create_test_user(db_session, "victim_resolve")
    attacker = create_test_user(db_session, "attacker_resolve")
    secret = f"victim-resolve-{uuid4().hex}"
    victim_file = _store_user_file(db_session, victim, secret)
    persona = _insert_persona(db_session, attacker, [victim_file], is_public=False)

    resolved = resolve_context_user_files(
        persona=persona,
        project_id=None,
        user_id=attacker.id,
        db_session=db_session,
    )

    assert all(user_file.user_id == persona.user_id for user_file in resolved)
    assert secret not in _prompt_text_for(resolved, db_session)


def test_resolve_context_keeps_file_owned_by_the_persona_owner(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    owner = create_test_user(db_session, "owner_resolve")
    reader = create_test_user(db_session, "reader_resolve")
    secret = f"owner-resolve-{uuid4().hex}"
    own_file = _store_user_file(db_session, owner, secret)
    persona = _insert_persona(db_session, owner, [own_file], is_public=True)

    resolved = resolve_context_user_files(
        persona=persona,
        project_id=None,
        user_id=reader.id,
        db_session=db_session,
    )

    assert own_file.id in {user_file.id for user_file in resolved}
    assert secret in _prompt_text_for(resolved, db_session)


def test_chat_turn_does_not_inject_or_grant_a_foreign_file(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    victim = create_test_user(db_session, "victim_chat")
    attacker = create_test_user(db_session, "attacker_chat")
    secret = f"victim-chat-{uuid4().hex}"
    victim_file = _store_user_file(db_session, victim, secret)
    persona = _insert_persona(db_session, attacker, [victim_file], is_public=True)

    setup = _run_chat_turn(db_session, attacker, persona)
    prompt = "\n".join(setup.extracted_context_files.file_texts)
    assert secret not in prompt
    if victim_file.id in setup.available_files.user_file_ids:
        granted_text = _read_granted_file(setup, victim_file.id)
        assert secret not in granted_text
    assert victim_file.id not in setup.available_files.user_file_ids


def test_chat_turn_still_injects_file_owned_by_the_persona_owner(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    owner = create_test_user(db_session, "owner_chat")
    reader = create_test_user(db_session, "reader_chat")
    secret = f"owner-chat-{uuid4().hex}"
    own_file = _store_user_file(db_session, owner, secret)
    persona = _insert_persona(db_session, owner, [own_file], is_public=True)

    setup = _run_chat_turn(db_session, reader, persona)

    assert secret in "\n".join(setup.extracted_context_files.file_texts)
    assert own_file.id in setup.available_files.user_file_ids
    assert secret in _read_granted_file(setup, own_file.id)


def test_access_sync_ignores_persona_the_file_owner_did_not_create(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    victim = create_test_user(db_session, "victim_access")
    attacker = create_test_user(db_session, "attacker_access")
    victim_file = _store_user_file(db_session, victim, f"victim-access-{uuid4().hex}")
    _insert_persona(db_session, attacker, [victim_file], is_public=True)

    emails, is_public = _committed_access(victim_file.id)

    assert attacker.email not in emails
    assert victim.email in emails
    assert is_public is False


def test_create_update_persona_wrapper_uses_the_same_rejection(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    victim = create_test_user(db_session, "victim_wrapper")
    attacker = create_test_user(db_session, "attacker_wrapper")
    victim_file = _store_user_file(db_session, victim, f"victim-wrapper-{uuid4().hex}")
    name = f"wrapper-{uuid4().hex}"

    with pytest.raises(_REJECTION_TYPES) as caught:
        create_update_persona(
            persona_id=None,
            create_persona_request=_request(name, [str(victim_file.id)], is_public=True),
            user=attacker,
            db_session=db_session,
        )
    _assert_client_rejection(caught.value)
    db_session.rollback()

    assert _persona_exists(name) is False
    needs_sync, persona_ids = _committed_links(victim_file.id)
    assert needs_sync is False
    assert persona_ids == []
