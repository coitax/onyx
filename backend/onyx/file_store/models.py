import base64
from enum import Enum
from typing import NotRequired
from uuid import UUID
from typing_extensions import TypedDict  # noreorder

from pydantic import BaseModel

# file_metadata keys on chat-scoped files (FileOrigin.CHAT_UPLOAD / CHAT_IMAGE_GEN)
# that record who may download them through the chat file endpoint.
CHAT_FILE_OWNER_USER_ID_KEY = "chat_file_owner_user_id"
CHAT_FILE_CHAT_SESSION_ID_KEY = "chat_file_chat_session_id"


def build_chat_file_scope_metadata(
    owner_user_id: UUID | None,
    chat_session_id: UUID | None = None,
) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if owner_user_id is not None:
        metadata[CHAT_FILE_OWNER_USER_ID_KEY] = str(owner_user_id)
    if chat_session_id is not None:
        metadata[CHAT_FILE_CHAT_SESSION_ID_KEY] = str(chat_session_id)
    return metadata


class ChatFileType(str, Enum):
    # Image types only contain the binary data
    IMAGE = "image"
    # Doc types are saved as both the binary, and the parsed text
    DOC = "document"
    # Plain text only contain the text
    PLAIN_TEXT = "plain_text"
    # Tabular data files (CSV, XLSX)
    TABULAR = "tabular"

    def is_text_file(self) -> bool:
        return self in (
            ChatFileType.PLAIN_TEXT,
            ChatFileType.DOC,
            ChatFileType.TABULAR,
        )

    def use_metadata_only(self) -> bool:
        """File types where we can ignore the file content
        and only use the metadata."""
        return self in (ChatFileType.TABULAR,)


class FileDescriptor(TypedDict):
    """NOTE: is a `TypedDict` so it can be used as a type hint for a JSONB column
    in Postgres"""

    id: str
    type: ChatFileType
    name: NotRequired[str | None]
    user_file_id: NotRequired[str | None]


class InMemoryChatFile(BaseModel):
    file_id: str
    content: bytes
    file_type: ChatFileType
    filename: str | None = None

    def to_base64(self) -> str:
        if self.file_type == ChatFileType.IMAGE:
            return base64.b64encode(self.content).decode()
        else:
            raise RuntimeError(
                "Should not be trying to convert a non-image file to base64"
            )

    def to_file_descriptor(self) -> FileDescriptor:
        return {
            "id": str(self.file_id),
            "type": self.file_type,
            "name": self.filename,
            "user_file_id": str(self.file_id) if self.file_id else None,
        }
