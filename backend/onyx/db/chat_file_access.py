"""Authorization for files served by the chat file download endpoint.

A file store id is only served when it is reachable from something the caller
can already see: one of their user files (or one shared with them through a
project or persona), a persona profile image, a file attached to or generated
in a chat session they can view, or a connector image whose document passes
their document ACL.
"""

import json
import re
from typing import Any
from uuid import UUID

from sqlalchemy import and_
from sqlalchemy import exists
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from onyx.access.access import get_access_for_documents
from onyx.access.access import get_acl_for_user
from onyx.configs.constants import BlobType
from onyx.configs.constants import PUBLIC_DOC_PAT
from onyx.db.document_access import get_accessible_documents_by_ids
from onyx.db.models import ChatMessage
from onyx.db.models import ChatSession
from onyx.db.models import ChatSessionSharedStatus
from onyx.db.models import Document
from onyx.db.models import Persona
from onyx.db.models import Persona__UserFile
from onyx.db.models import Project__UserFile
from onyx.db.models import Tool
from onyx.db.models import ToolCall
from onyx.db.models import User
from onyx.db.models import User__ExternalUserGroupId
from onyx.db.models import UserFile
from onyx.db.models import UserProject
from onyx.db.persona import user_can_view_persona_matching
from onyx.file_store.models import FileDescriptor
from onyx.tools.constants import PYTHON_TOOL_ID

_FILE_CONNECTOR_DOC_ID_PREFIX = "FILE_CONNECTOR__"
_EMBEDDED_IMAGE_SUFFIX_RE = re.compile(r"^(?P<parent>.+)_(?:img|embedded)_\d+$")
_CUSTOM_TOOL_FILE_RESPONSE_TYPES = ("image", "csv")


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
    except ValueError:
        return False
    return True


def _user_file_is_accessible(
    user_file: UserFile, user: User, db_session: Session
) -> bool:
    if user_file.user_id == user.id:
        return True

    in_owned_project = db_session.scalar(
        select(Project__UserFile.project_id)
        .join(UserProject, UserProject.id == Project__UserFile.project_id)
        .where(
            Project__UserFile.user_file_id == user_file.id,
            UserProject.user_id == user.id,
        )
        .limit(1)
    )
    if in_owned_project is not None:
        return True

    return user_can_view_persona_matching(
        db_session,
        user,
        Persona.id.in_(
            select(Persona__UserFile.persona_id).where(
                Persona__UserFile.user_file_id == user_file.id
            )
        ),
    )


def _viewable_chat_session_filter(user: User) -> ColumnElement[bool]:
    # Mirrors the visibility rules of get_chat_session_by_id.
    return and_(
        ChatSession.deleted.is_(False),
        or_(
            ChatSession.user_id == user.id,
            ChatSession.user_id.is_(None),
            ChatSession.shared_status == ChatSessionSharedStatus.PUBLIC,
        ),
    )


def _session_may_reference_file(file_id: str) -> ColumnElement[bool]:
    # A user file only counts as shared through a session owned by that file's
    # owner, so a forged attachment in someone else's session grants nothing.
    is_user_file = exists().where(UserFile.file_id == file_id)
    owned_by_session_user = exists().where(
        UserFile.file_id == file_id, UserFile.user_id == ChatSession.user_id
    )
    return or_(~is_user_file, owned_by_session_user)


def _descriptors_reference_file(
    descriptors: list[FileDescriptor], file_id: str, db_session: Session
) -> bool:
    for descriptor in descriptors:
        if descriptor.get("id") != file_id:
            continue
        raw_user_file_id = descriptor.get("user_file_id")
        if not raw_user_file_id:
            return True
        # A descriptor naming a user file only references that user file's
        # own store id.
        try:
            user_file_id = UUID(str(raw_user_file_id))
        except ValueError:
            continue
        matching_user_file = db_session.scalar(
            select(UserFile.id).where(
                UserFile.id == user_file_id, UserFile.file_id == file_id
            )
        )
        if matching_user_file is not None:
            return True
    return False


def _python_tool_response_has_file(response: str, file_id: str) -> bool:
    try:
        parsed: Any = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(parsed, dict):
        return False
    generated_files = parsed.get("generated_files")
    if not isinstance(generated_files, list):
        return False
    for generated_file in generated_files:
        if not isinstance(generated_file, dict):
            continue
        file_link = generated_file.get("file_link")
        if isinstance(file_link, str) and file_link.split("/")[-1] == file_id:
            return True
    return False


def _custom_tool_response_has_file(response: str, file_id: str) -> bool:
    try:
        parsed: Any = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(parsed, dict):
        return False
    # Only image and csv responses carry file ids, and those ids are assigned
    # by Onyx when it saves the tool output.
    if parsed.get("response_type") not in _CUSTOM_TOOL_FILE_RESPONSE_TYPES:
        return False
    tool_result = parsed.get("tool_result")
    if not isinstance(tool_result, dict):
        return False
    file_ids = tool_result.get("file_ids")
    return isinstance(file_ids, list) and file_id in file_ids


def _file_is_in_viewable_chat_session(
    file_id: str, user: User, db_session: Session
) -> bool:
    session_filter = and_(
        _viewable_chat_session_filter(user), _session_may_reference_file(file_id)
    )

    message_files = db_session.scalars(
        select(ChatMessage.files)
        .join(ChatSession, ChatSession.id == ChatMessage.chat_session_id)
        .where(session_filter, ChatMessage.files.contains([{"id": file_id}]))
    ).all()
    for descriptors in message_files:
        if _descriptors_reference_file(descriptors or [], file_id, db_session):
            return True

    generated_image = db_session.scalar(
        select(ToolCall.id)
        .join(ChatSession, ChatSession.id == ToolCall.chat_session_id)
        .where(
            session_filter,
            ToolCall.generated_images.contains([{"file_id": file_id}]),
        )
        .limit(1)
    )
    if generated_image is not None:
        return True

    tool_responses = db_session.execute(
        select(ToolCall.tool_call_response, Tool.in_code_tool_id)
        .join(ChatSession, ChatSession.id == ToolCall.chat_session_id)
        .join(Tool, Tool.id == ToolCall.tool_id)
        .where(
            session_filter,
            or_(
                Tool.in_code_tool_id == PYTHON_TOOL_ID,
                # OpenAPI custom tools: neither built in nor backed by MCP.
                and_(
                    Tool.in_code_tool_id.is_(None),
                    Tool.mcp_server_id.is_(None),
                ),
            ),
            ToolCall.tool_call_response.contains(file_id, autoescape=True),
        )
    ).all()
    for response, in_code_tool_id in tool_responses:
        if in_code_tool_id == PYTHON_TOOL_ID:
            if _python_tool_response_has_file(response, file_id):
                return True
        elif _custom_tool_response_has_file(response, file_id):
            return True

    return False


def _candidate_document_ids_for_connector_image(
    file_id: str, db_session: Session
) -> list[str]:
    """Document ids that store an image under `file_id`, derived from the
    naming used by the file and blob connectors."""
    candidates: list[str] = []

    embedded_match = _EMBEDDED_IMAGE_SUFFIX_RE.match(file_id)
    for parent_file_id in (
        file_id,
        embedded_match.group("parent") if embedded_match else None,
    ):
        if parent_file_id and _is_uuid(parent_file_id):
            candidates.append(f"{_FILE_CONNECTOR_DOC_ID_PREFIX}{parent_file_id}")

    # Blob connector: document id "{type}:{bucket}:{key}" stores its image as
    # "{type}_{bucket}_{key with '/' replaced by '_'}". Neither the type nor
    # the bucket name contains ':' or '/'.
    for blob_type in BlobType:
        type_prefix = f"{blob_type}"
        if not file_id.startswith(f"{type_prefix}_"):
            continue
        derived_file_id = func.regexp_replace(
            func.replace(Document.id, "/", "_"), "^([^:]*):([^:]*):", "\\1_\\2_"
        )
        candidates.extend(
            db_session.scalars(
                select(Document.id).where(
                    Document.id.startswith(f"{type_prefix}:", autoescape=True),
                    func.length(Document.id) == len(file_id),
                    derived_file_id == file_id,
                )
            ).all()
        )

    return candidates


def _user_external_group_ids(user: User, db_session: Session) -> list[str]:
    return list(
        db_session.scalars(
            select(User__ExternalUserGroupId.external_user_group_id).where(
                User__ExternalUserGroupId.user_id == user.id
            )
        ).all()
    )


def _connector_image_is_accessible(
    file_id: str, user: User, db_session: Session
) -> bool:
    candidate_ids = _candidate_document_ids_for_connector_image(file_id, db_session)
    if not candidate_ids:
        return False
    document_ids = list(
        db_session.scalars(
            select(Document.id).where(Document.id.in_(candidate_ids))
        ).all()
    )
    if not document_ids:
        return False

    if get_accessible_documents_by_ids(
        db_session=db_session,
        document_ids=document_ids,
        user_email=None if user.is_anonymous else user.email,
        external_group_ids=(
            [] if user.is_anonymous else _user_external_group_ids(user, db_session)
        ),
    ):
        return True

    # Per-user and per-group grants (connector owner, user groups, synced
    # permissions). Public access is decided only by the filter above: the
    # document ACL can mark a document public so that results are censored
    # after search, which must not make its files downloadable.
    user_acl = get_acl_for_user(user, db_session) - {PUBLIC_DOC_PAT}
    for access in get_access_for_documents(document_ids, db_session).values():
        if (access.to_acl() - {PUBLIC_DOC_PAT}) & user_acl:
            return True
    return False


def user_can_access_chat_file(
    file_id: str,
    user_file_id: UUID | None,
    user: User,
    db_session: Session,
) -> bool:
    """Whether `user` may download the file store file `file_id`.

    When the request named a user file, `user_file_id` is that user file and
    `file_id` the store id it resolved to. Access is then granted only if that
    user file still points at `file_id`.
    """
    if user_file_id is not None:
        user_file = db_session.scalars(
            select(UserFile).where(
                UserFile.id == user_file_id, UserFile.file_id == file_id
            )
        ).first()
        return user_file is not None and _user_file_is_accessible(
            user_file, user, db_session
        )

    for user_file in db_session.scalars(
        select(UserFile).where(UserFile.file_id == file_id)
    ).all():
        if _user_file_is_accessible(user_file, user, db_session):
            return True

    if user_can_view_persona_matching(
        db_session, user, Persona.uploaded_image_id == file_id
    ):
        return True

    if _file_is_in_viewable_chat_session(file_id, user, db_session):
        return True

    return _connector_image_is_accessible(file_id, user, db_session)
