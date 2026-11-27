"""Chat attachment descriptors must not grant access to another user's file.

A descriptor's ``user_file_id`` is the field the ownership check consults. The
bytes that are stored, injected into the prompt, and allow-listed for the file
reader are loaded from the descriptor's ``id``. These tests send the descriptor
through the chat entry points and then read back what those entry points
persisted and loaded.

They fail while a caller can supply another user's file-store id in ``id``
together with a missing, invalid, or self-owned ``user_file_id``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from io import BytesIO
from queue import Queue
from unittest.mock import patch
from uuid import UUID
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.chat.chat_utils import convert_chat_history
from onyx.chat.chat_utils import create_chat_history_chain
from onyx.chat.chat_utils import create_chat_session_from_request
from onyx.chat.chat_utils import load_all_chat_files
from onyx.chat.emitter import Emitter
from onyx.chat.models import ChatLoadedFile
from onyx.chat.models import StreamingError
from onyx.chat.process_message import _collect_available_file_ids
from onyx.chat.process_message import handle_multi_model_stream
from onyx.chat.process_message import handle_stream_message_objects
from onyx.configs.constants import FileOrigin
from onyx.db.models import ChatMessage
from onyx.db.models import ChatSession
from onyx.db.models import User
from onyx.db.models import UserFile
from onyx.db.models import UserProject
from onyx.error_handling.exceptions import OnyxError
from onyx.file_store.file_store import get_default_file_store
from onyx.file_store.models import ChatFileType
from onyx.file_store.models import FileDescriptor
from onyx.file_store.utils import plaintext_file_name_for_id
from onyx.llm.override_models import LLMOverride
from onyx.server.query_and_chat.models import ChatSessionCreationRequest
from onyx.server.query_and_chat.models import SendMessageRequest
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import ToolCallException
from onyx.tools.tool_implementations.file_reader.file_reader_tool import (
    FileReaderTool,
)
from onyx.tools.tool_implementations.file_reader.file_reader_tool import (
    FileReaderToolOverrideKwargs,
)
from tests.external_dependency_unit.conftest import create_test_user
from tests.external_dependency_unit.mock_llm import use_mock_llm

_VICTIM_MARKER = "victim-private-document-marker-74b4f998"
_ATTACKER_MARKER = "attacker-owned-document-marker-74b4f998"


def _token_counter(text: str) -> int:
    return max(1, len(text))


def _store_owned_file(
    db_session: Session,
    owner: User,
    marker: str,
    name: str,
) -> tuple[str, UUID]:
    """Save bytes in the file store and a UserFile row owned by ``owner``."""

    file_id = get_default_file_store().save_file(
        content=BytesIO(f"{marker}\n".encode()),
        display_name=name,
        file_origin=FileOrigin.USER_FILE,
        file_type="text/plain",
    )
    user_file_id = uuid4()
    db_session.add(
        UserFile(
            id=user_file_id,
            user_id=owner.id,
            file_id=file_id,
            name=name,
            file_type="text/plain",
            token_count=4,
        )
    )
    db_session.commit()
    return file_id, user_file_id


def _descriptor(
    file_store_id: str,
    *,
    user_file_id: str | None,
    file_type: ChatFileType,
) -> FileDescriptor:
    descriptor: FileDescriptor = {
        "id": file_store_id,
        "type": file_type,
        "name": "notes.txt" if file_type.is_text_file() else "notes.png",
    }
    if user_file_id is not None:
        descriptor["user_file_id"] = user_file_id
    return descriptor


def _create_session(
    db_session: Session,
    user: User,
    project_id: int | None = None,
) -> ChatSession:
    return create_chat_session_from_request(
        chat_session_request=ChatSessionCreationRequest(
            description="file-descriptor-authorization",
            project_id=project_id,
        ),
        user_id=user.id,
        db_session=db_session,
    )


def _idle_models(*_args: object, **_kwargs: object) -> Iterator[object]:
    """Stand in for the LLM loop.

    ``build_chat_turn`` has already checked the descriptor, stored it, loaded
    the bytes, and built the prompt before this is called. The model call does
    not choose which file those steps used.
    """

    yield from ()


def _send(
    db_session: Session,
    user: User,
    chat_session_id: UUID,
    descriptors: list[FileDescriptor],
    *,
    multi_model: bool,
) -> object | None:
    """Run one chat entry point through prompt construction.

    Token counting does not choose which file is read, so it is a length count.
    The descriptor is otherwise handled by the product code: the ownership
    check, the message write, the file-store read, and prompt assembly.
    """

    overrides = (
        [
            LLMOverride(model_version="mock-a", display_name="mock-a"),
            LLMOverride(model_version="mock-b", display_name="mock-b"),
        ]
        if multi_model
        else None
    )
    request = SendMessageRequest(
        message="Summarize the attached file.",
        chat_session_id=chat_session_id,
        file_descriptors=descriptors,
        llm_overrides=overrides,
        stream=True,
    )

    with (
        use_mock_llm(),
        patch(
            "onyx.chat.process_message.get_llm_token_counter",
            return_value=_token_counter,
        ),
        patch(
            "onyx.chat.process_message._run_models",
            _idle_models,
        ),
    ):
        stream: Iterator[object]
        if multi_model:
            assert overrides is not None
            stream = handle_multi_model_stream(
                new_msg_req=request,
                user=user,
                db_session=db_session,
                llm_overrides=overrides,
            )
        else:
            stream = handle_stream_message_objects(
                new_msg_req=request,
                user=user,
                db_session=db_session,
            )
        first: object | None = None
        try:
            for part in stream:
                if first is None:
                    first = part
        finally:
            stream.close()
            db_session.rollback()
        return first


def _read_if_present(file_id: str) -> bytes | None:
    try:
        return get_default_file_store().read_file(file_id, mode="b").read()
    except Exception:
        return None


def _file_reader_text(
    chat_file_ids: list[UUID],
    user_file_ids: list[UUID],
    requested_id: str,
) -> str | None:
    tool = FileReaderTool(
        tool_id=1,
        emitter=Emitter(merged_queue=Queue()),
        user_file_ids=user_file_ids,
        chat_file_ids=chat_file_ids,
    )
    try:
        response = tool.run(
            placement=Placement(turn_index=0),
            override_kwargs=FileReaderToolOverrideKwargs(),
            file_id=requested_id,
        )
    except ToolCallException:
        return None
    return response.llm_facing_response


def _exposure(
    db_session: Session,
    chat_session: ChatSession,
    *,
    victim_file_id: str,
    victim_marker: str,
    caller: User,
    cache_keys: list[str],
) -> list[str]:
    """Return every place the victim document became visible to the caller."""

    found: list[str] = []
    marker_bytes = victim_marker.encode()
    messages = (
        db_session.query(ChatMessage)
        .filter(ChatMessage.chat_session_id == chat_session.id)
        .all()
    )
    for message in messages:
        serialized = json.dumps(message.files or [])
        if victim_file_id in serialized or victim_marker in serialized:
            found.append(
                f"chat message {message.id} stored the victim file descriptor: {serialized}"
            )
        if victim_marker in (message.message or ""):
            found.append(f"chat message {message.id} text contains the victim document")

    history = create_chat_history_chain(
        chat_session_id=chat_session.id,
        db_session=db_session,
    )
    loaded: list[ChatLoadedFile] = []
    if any(message.files for message in history):
        try:
            loaded = load_all_chat_files(history, db_session)
        except Exception as exc:
            found.append(f"load_all_chat_files failed on the saved descriptors: {exc}")
    for loaded_file in loaded:
        if marker_bytes in loaded_file.content or victim_marker in (
            loaded_file.content_text or ""
        ):
            found.append(
                "load_all_chat_files returned the victim document "
                f"for file_id={loaded_file.file_id}"
            )

    history_result = convert_chat_history(
        chat_history=history,
        files=loaded,
        context_image_files=[],
        additional_context=None,
        token_counter=_token_counter,
        tool_id_to_name_map={},
    )
    for simple in history_result.simple_messages:
        if victim_marker in simple.message:
            found.append(
                "convert_chat_history placed the victim document in the prompt: "
                f"{simple.message[:200]}"
            )
        for image in simple.image_files or []:
            if victim_marker.encode() in image.content or victim_marker in (
                image.content_text or ""
            ):
                found.append(
                    "convert_chat_history attached the victim document as an image "
                    f"file_id={image.file_id}"
                )

    available = _collect_available_file_ids(
        chat_history=history,
        project_id=chat_session.project_id,
        user_id=caller.id,
        db_session=db_session,
    )
    readable_ids = [str(file_id) for file_id in available.chat_file_ids] + [
        str(file_id) for file_id in available.user_file_ids
    ]
    if victim_file_id in readable_ids:
        found.append(
            "file reader allow-list contains the victim file id "
            f"{victim_file_id}"
        )
    ids_to_read = list(dict.fromkeys([victim_file_id, *readable_ids]))
    for requested_id in ids_to_read:
        text = _file_reader_text(
            chat_file_ids=available.chat_file_ids,
            user_file_ids=available.user_file_ids,
            requested_id=requested_id,
        )
        if text is not None and victim_marker in text:
            found.append(
                f"file reader returned the victim document for file_id={requested_id}"
            )

    cache_ids = [plaintext_file_name_for_id(key) for key in cache_keys]
    for cache_id in dict.fromkeys(cache_ids):
        cached = _read_if_present(cache_id)
        if cached is not None and marker_bytes in cached:
            found.append(f"plaintext cache {cache_id} contains the victim document")

    return found


def _assert_owned_file_reaches_loader(
    db_session: Session,
    caller: User,
    file_store_id: str,
    user_file_id: UUID,
    marker: str,
) -> None:
    """Prove this process reaches the real loader before asserting a refusal.

    A setup failure would otherwise look like a successful authorization check.
    """

    chat_session = _create_session(db_session, caller)
    first = _send(
        db_session,
        caller,
        chat_session.id,
        [
            _descriptor(
                file_store_id,
                user_file_id=str(user_file_id),
                file_type=ChatFileType.PLAIN_TEXT,
            )
        ],
        multi_model=False,
    )
    assert not isinstance(first, StreamingError), (
        "Owned-file send did not reach file loading: "
        f"{first.error if isinstance(first, StreamingError) else first}"
    )
    loaded = load_all_chat_files(
        create_chat_history_chain(
            chat_session_id=chat_session.id, db_session=db_session
        ),
        db_session,
    )
    assert any(
        marker.encode() in loaded_file.content
        or marker in (loaded_file.content_text or "")
        for loaded_file in loaded
    ), "Owned file was not loaded, so an attack refusal would not exercise the loader"


def _assert_victim_is_not_exposed(
    db_session: Session,
    caller: User,
    victim_file_id: str,
    *,
    user_file_id: str | None,
    file_type: ChatFileType,
    project_id: int | None,
    multi_model: bool,
    cache_keys: list[str],
) -> None:
    # Drop plaintext cached by the owned-file setup so this send extracts
    # from the descriptor id instead of a previous file's cache entry.
    for key in cache_keys:
        get_default_file_store().delete_file(
            plaintext_file_name_for_id(key),
            error_on_missing=False,
        )

    chat_session = _create_session(db_session, caller, project_id=project_id)
    outcome: object | None
    try:
        outcome = _send(
            db_session,
            caller,
            chat_session.id,
            [
                _descriptor(
                    victim_file_id,
                    user_file_id=user_file_id,
                    file_type=file_type,
                )
            ],
            multi_model=multi_model,
        )
    except (ValueError, OnyxError):
        outcome = None
        db_session.rollback()

    found = _exposure(
        db_session,
        chat_session,
        victim_file_id=victim_file_id,
        victim_marker=_VICTIM_MARKER,
        caller=caller,
        cache_keys=cache_keys,
    )
    if isinstance(outcome, StreamingError):
        error_text = f"{outcome.error}\n{outcome.stack_trace or ''}"
        if _VICTIM_MARKER in error_text:
            found.append("the chat error response included the victim document")
    assert not found, "Victim file was exposed to the caller:\n" + "\n".join(found)


@pytest.fixture()
def attacker_and_victim(
    db_session: Session,
) -> tuple[User, User, tuple[str, UUID], tuple[str, UUID]]:
    attacker = create_test_user(db_session, email_prefix="descriptor_attacker")
    victim = create_test_user(db_session, email_prefix="descriptor_victim")
    attacker_file = _store_owned_file(
        db_session,
        attacker,
        _ATTACKER_MARKER,
        "attacker.txt",
    )
    victim_file = _store_owned_file(
        db_session,
        victim,
        _VICTIM_MARKER,
        "victim.txt",
    )
    _assert_owned_file_reaches_loader(
        db_session,
        attacker,
        attacker_file[0],
        attacker_file[1],
        _ATTACKER_MARKER,
    )
    return attacker, victim, attacker_file, victim_file


def test_owned_user_file_id_with_foreign_file_store_id_is_not_loaded(
    db_session: Session,
    attacker_and_victim: tuple[User, User, tuple[str, UUID], tuple[str, UUID]],
) -> None:
    """The checked UserFile belongs to the caller and the loaded id does not.

    Re-checking ``user_file_id`` and then reading ``id`` still fails this test.
    """

    attacker, _victim, attacker_file, victim_file = attacker_and_victim
    _assert_victim_is_not_exposed(
        db_session,
        attacker,
        victim_file[0],
        user_file_id=str(attacker_file[1]),
        file_type=ChatFileType.PLAIN_TEXT,
        project_id=None,
        multi_model=False,
        cache_keys=[str(attacker_file[1]), victim_file[0]],
    )


def test_unknown_user_file_id_with_foreign_file_store_id_is_not_loaded(
    db_session: Session,
    attacker_and_victim: tuple[User, User, tuple[str, UUID], tuple[str, UUID]],
) -> None:
    attacker, _victim, _attacker_file, victim_file = attacker_and_victim
    unknown_user_file_id = str(uuid4())
    _assert_victim_is_not_exposed(
        db_session,
        attacker,
        victim_file[0],
        user_file_id=unknown_user_file_id,
        file_type=ChatFileType.PLAIN_TEXT,
        project_id=None,
        multi_model=False,
        cache_keys=[unknown_user_file_id, victim_file[0]],
    )


def test_invalid_user_file_id_with_foreign_file_store_id_is_not_loaded(
    db_session: Session,
    attacker_and_victim: tuple[User, User, tuple[str, UUID], tuple[str, UUID]],
) -> None:
    attacker, _victim, _attacker_file, victim_file = attacker_and_victim
    _assert_victim_is_not_exposed(
        db_session,
        attacker,
        victim_file[0],
        user_file_id="not-a-uuid",
        file_type=ChatFileType.PLAIN_TEXT,
        project_id=None,
        multi_model=False,
        cache_keys=["not-a-uuid", victim_file[0]],
    )


def test_image_descriptor_with_foreign_file_store_id_is_not_loaded(
    db_session: Session,
    attacker_and_victim: tuple[User, User, tuple[str, UUID], tuple[str, UUID]],
) -> None:
    attacker, _victim, attacker_file, victim_file = attacker_and_victim
    _assert_victim_is_not_exposed(
        db_session,
        attacker,
        victim_file[0],
        user_file_id=str(attacker_file[1]),
        file_type=ChatFileType.IMAGE,
        project_id=None,
        multi_model=False,
        cache_keys=[str(attacker_file[1]), victim_file[0]],
    )


def test_project_chat_skips_foreign_file_when_user_file_id_is_set(
    db_session: Session,
    attacker_and_victim: tuple[User, User, tuple[str, UUID], tuple[str, UUID]],
) -> None:
    """A project session still has to bind ``id`` to a file the caller can use.

    Setting ``user_file_id`` takes the user-file branch, so the project-file
    check never sees the file-store id.
    """

    attacker, _victim, attacker_file, victim_file = attacker_and_victim
    project = UserProject(
        user_id=attacker.id,
        name=f"project-{uuid4().hex[:8]}",
        instructions="",
    )
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    _assert_victim_is_not_exposed(
        db_session,
        attacker,
        victim_file[0],
        user_file_id=str(attacker_file[1]),
        file_type=ChatFileType.PLAIN_TEXT,
        project_id=project.id,
        multi_model=False,
        cache_keys=[str(attacker_file[1]), victim_file[0]],
    )


def test_multi_model_send_does_not_load_foreign_file_store_id(
    db_session: Session,
    attacker_and_victim: tuple[User, User, tuple[str, UUID], tuple[str, UUID]],
) -> None:
    attacker, _victim, attacker_file, victim_file = attacker_and_victim
    _assert_victim_is_not_exposed(
        db_session,
        attacker,
        victim_file[0],
        user_file_id=str(attacker_file[1]),
        file_type=ChatFileType.PLAIN_TEXT,
        project_id=None,
        multi_model=True,
        cache_keys=[str(attacker_file[1]), victim_file[0]],
    )


def test_other_users_user_file_id_is_rejected(
    db_session: Session,
    attacker_and_victim: tuple[User, User, tuple[str, UUID], tuple[str, UUID]],
) -> None:
    """The existing owner check on ``user_file_id`` must keep refusing."""

    attacker, _victim, _attacker_file, victim_file = attacker_and_victim
    _assert_victim_is_not_exposed(
        db_session,
        attacker,
        victim_file[0],
        user_file_id=str(victim_file[1]),
        file_type=ChatFileType.PLAIN_TEXT,
        project_id=None,
        multi_model=False,
        cache_keys=[str(victim_file[1]), victim_file[0]],
    )


def test_descriptor_without_user_file_id_outside_a_project_is_not_loaded(
    db_session: Session,
    attacker_and_victim: tuple[User, User, tuple[str, UUID], tuple[str, UUID]],
) -> None:
    attacker, _victim, _attacker_file, victim_file = attacker_and_victim
    _assert_victim_is_not_exposed(
        db_session,
        attacker,
        victim_file[0],
        user_file_id=None,
        file_type=ChatFileType.PLAIN_TEXT,
        project_id=None,
        multi_model=False,
        cache_keys=[victim_file[0]],
    )
