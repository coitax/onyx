"""Authorization for files served by the chat file download endpoint.

Access is decided by the origin of the stored file that would be streamed:

- USER_FILE: the owner of a user_file row pointing at the blob, a user who can
  view a persona the user file is attached to, or any user when the owner
  referenced the blob in one of their publicly shared chat sessions.
- CHAT_UPLOAD / CHAT_IMAGE_GEN: the user recorded as the creator, users who can
  read the chat session the file was created in or that references it, and
  users who can view a persona using it as its avatar.
- Every other origin (connector files, indexing checkpoints and batches,
  snapshots, caches, reports, ...) is never served.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from onyx.configs.constants import FileOrigin
from onyx.db.enums import ChatSessionSharedStatus
from onyx.db.file_record import get_filerecord_by_file_id_optional
from onyx.db.models import ChatMessage
from onyx.db.models import ChatSession
from onyx.db.models import FileRecord
from onyx.db.models import Persona
from onyx.db.models import Persona__UserFile
from onyx.db.models import ToolCall
from onyx.db.models import User
from onyx.db.models import UserFile
from onyx.db.persona import user_can_access_persona
from onyx.file_store.models import CHAT_FILE_CHAT_SESSION_ID_KEY
from onyx.file_store.models import CHAT_FILE_OWNER_USER_ID_KEY

_CHAT_SCOPED_ORIGINS = frozenset({FileOrigin.CHAT_UPLOAD, FileOrigin.CHAT_IMAGE_GEN})


@dataclass(frozen=True)
class AuthorizedChatFile:
    """Snapshot of the file record that was authorized.

    The bytes streamed must belong to this record; callers compare the record
    again after reading and refuse if it changed in between.
    """

    file_id: str
    file_origin: FileOrigin
    file_type: str
    file_metadata: Any
    bucket_name: str
    object_key: str

    @classmethod
    def from_record(cls, record: FileRecord) -> "AuthorizedChatFile":
        return cls(
            file_id=record.file_id,
            file_origin=record.file_origin,
            file_type=record.file_type,
            file_metadata=record.file_metadata,
            bucket_name=record.bucket_name,
            object_key=record.object_key,
        )

    def matches(self, record: FileRecord) -> bool:
        return self == AuthorizedChatFile.from_record(record)


def _parse_uuid(value: Any) -> UUID | None:
    if not isinstance(value, str):
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _readable_chat_session_clause(user: User) -> ColumnElement[bool]:
    return or_(
        ChatSession.user_id == user.id,
        (ChatSession.shared_status == ChatSessionSharedStatus.PUBLIC)
        & ChatSession.deleted.is_(False),
    )


def _can_view_any_persona(
    persona_ids: list[int], user: User, db_session: Session
) -> bool:
    return any(
        user_can_access_persona(db_session, persona_id, user)
        for persona_id in persona_ids
    )


def _user_can_access_user_file(
    user_file_id: UUID,
    user_file_owner_id: UUID | None,
    blob_id: str,
    user: User,
    db_session: Session,
) -> bool:
    if user_file_owner_id is not None and user_file_owner_id == user.id:
        return True

    persona_ids = list(
        db_session.scalars(
            select(Persona__UserFile.persona_id).where(
                Persona__UserFile.user_file_id == user_file_id
            )
        )
    )
    if _can_view_any_persona(persona_ids, user, db_session):
        return True

    if user_file_owner_id is None:
        return False

    shared_by_owner = db_session.scalar(
        select(ChatMessage.id)
        .join(ChatSession, ChatMessage.chat_session_id == ChatSession.id)
        .where(
            ChatSession.user_id == user_file_owner_id,
            ChatSession.shared_status == ChatSessionSharedStatus.PUBLIC,
            ChatSession.deleted.is_(False),
            ChatMessage.files.contains([{"id": blob_id}]),
        )
        .limit(1)
    )
    return shared_by_owner is not None


def _user_can_access_chat_scoped_file(
    blob_id: str,
    file_metadata: Any,
    user: User,
    db_session: Session,
) -> bool:
    metadata = file_metadata if isinstance(file_metadata, dict) else {}

    if metadata.get(CHAT_FILE_OWNER_USER_ID_KEY) == str(user.id):
        return True

    chat_session_id = _parse_uuid(metadata.get(CHAT_FILE_CHAT_SESSION_ID_KEY))
    if chat_session_id is not None:
        readable_session = db_session.scalar(
            select(ChatSession.id).where(
                ChatSession.id == chat_session_id,
                _readable_chat_session_clause(user),
            )
        )
        if readable_session is not None:
            return True

    message_files = db_session.scalars(
        select(ChatMessage.files)
        .join(ChatSession, ChatMessage.chat_session_id == ChatSession.id)
        .where(
            _readable_chat_session_clause(user),
            ChatMessage.files.contains([{"id": blob_id}]),
        )
    )
    for files in message_files:
        for descriptor in files or []:
            # Descriptors carrying a user_file_id point at user files, never at
            # chat-scoped blobs.
            if descriptor.get("id") == blob_id and not descriptor.get("user_file_id"):
                return True

    generated_image_call = db_session.scalar(
        select(ToolCall.id)
        .join(ChatSession, ToolCall.chat_session_id == ChatSession.id)
        .where(
            _readable_chat_session_clause(user),
            ToolCall.generated_images.contains([{"file_id": blob_id}]),
        )
        .limit(1)
    )
    if generated_image_call is not None:
        return True

    avatar_persona_ids = list(
        db_session.scalars(
            select(Persona.id).where(Persona.uploaded_image_id == blob_id)
        )
    )
    return _can_view_any_persona(avatar_persona_ids, user, db_session)


def authorize_chat_file_access(
    requested_id: str,
    user: User,
    db_session: Session,
) -> AuthorizedChatFile | None:
    """Return the file record `user` may download for `requested_id`, or None.

    `requested_id` is either a user_file id or a file store id. The user_file
    row is read once and the blob it points at is the only one considered.
    """
    requested_uuid = _parse_uuid(requested_id)
    user_file = (
        db_session.scalar(select(UserFile).where(UserFile.id == requested_uuid))
        if requested_uuid is not None
        else None
    )

    if user_file is not None:
        blob_id = user_file.file_id
        if not _user_can_access_user_file(
            user_file.id, user_file.user_id, blob_id, user, db_session
        ):
            return None
        record = get_filerecord_by_file_id_optional(blob_id, db_session)
        if record is None or record.file_origin != FileOrigin.USER_FILE:
            return None
        return AuthorizedChatFile.from_record(record)

    record = get_filerecord_by_file_id_optional(requested_id, db_session)
    if record is None:
        return None
    authorized = AuthorizedChatFile.from_record(record)

    if authorized.file_origin == FileOrigin.USER_FILE:
        owners = db_session.execute(
            select(UserFile.id, UserFile.user_id).where(
                UserFile.file_id == authorized.file_id
            )
        ).all()
        allowed = any(
            _user_can_access_user_file(
                user_file_id, owner_id, authorized.file_id, user, db_session
            )
            for user_file_id, owner_id in owners
        )
    elif authorized.file_origin in _CHAT_SCOPED_ORIGINS:
        allowed = _user_can_access_chat_scoped_file(
            authorized.file_id, authorized.file_metadata, user, db_session
        )
    else:
        allowed = False

    return authorized if allowed else None
