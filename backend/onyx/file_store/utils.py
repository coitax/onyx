import base64
from collections.abc import Callable
from io import BytesIO
from typing import cast
from uuid import UUID

import requests
from sqlalchemy.orm import Session

from onyx.configs.app_configs import WEB_DOMAIN
from onyx.configs.constants import FileOrigin
from onyx.db.models import UserFile
from onyx.file_store.file_store import get_default_file_store
from onyx.file_store.models import ChatFileType
from onyx.file_store.models import FileDescriptor
from onyx.file_store.models import InMemoryChatFile
from onyx.server.query_and_chat.chat_utils import mime_type_to_chat_file_type
from onyx.utils.b64 import get_image_type
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel
from onyx.utils.timing import log_function_time

logger = setup_logger()


def plaintext_file_name_for_id(file_id: str) -> str:
    """Generate a consistent file name for storing plaintext content of a file."""
    return f"plaintext_{file_id}"


def store_plaintext(file_id: str, plaintext_content: str) -> bool:
    """
    Store plaintext content for a file in the file store.

    Args:
        file_id: The ID of the file (user_file or artifact_file)
        plaintext_content: The plaintext content to store

    Returns:
        bool: True if storage was successful, False otherwise
    """
    if not plaintext_content:
        return False

    plaintext_file_name = plaintext_file_name_for_id(file_id)
    try:
        file_store = get_default_file_store()
        file_content = BytesIO(plaintext_content.encode("utf-8"))
        file_store.save_file(
            content=file_content,
            display_name=f"Plaintext for {file_id}",
            file_origin=FileOrigin.PLAINTEXT_CACHE,
            file_type="text/plain",
            file_id=plaintext_file_name,
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to store plaintext for {file_id}: {e}")
        return False


# --- Convenience wrappers for callers that use user-file UUIDs ---


def user_file_id_to_plaintext_file_name(user_file_id: UUID) -> str:
    """Generate a consistent file name for storing plaintext content of a user file."""
    return plaintext_file_name_for_id(str(user_file_id))


def store_user_file_plaintext(user_file_id: UUID, plaintext_content: str) -> bool:
    """Store plaintext content for a user file (delegates to :func:`store_plaintext`)."""
    return store_plaintext(str(user_file_id), plaintext_content)


def load_chat_file_by_id(file_id: str) -> InMemoryChatFile:
    """Load a file directly from the file store using its file_record ID.

    This is the fallback path for chat-attached files that don't have a
    corresponding row in the ``user_file`` table."""
    file_store = get_default_file_store()
    file_record = file_store.read_file_record(file_id)
    chat_file_type = mime_type_to_chat_file_type(file_record.file_type)

    file_io = file_store.read_file(file_id, mode="b")
    return InMemoryChatFile(
        file_id=file_id,
        content=file_io.read(),
        file_type=chat_file_type,
        filename=file_record.display_name,
    )


def load_user_file(file_id: UUID, db_session: Session) -> InMemoryChatFile:
    status = "not_loaded"

    user_file = db_session.query(UserFile).filter(UserFile.id == file_id).first()
    if not user_file:
        raise ValueError(f"User file with id {file_id} not found")

    # Get the file record to determine the appropriate chat file type
    file_store = get_default_file_store()
    file_record = file_store.read_file_record(user_file.file_id)

    # Determine appropriate chat file type based on the original file's MIME type
    chat_file_type = mime_type_to_chat_file_type(file_record.file_type)

    # Try to load plaintext version first
    plaintext_file_name = user_file_id_to_plaintext_file_name(file_id)

    # check for plain text normalized version first, then use original file otherwise
    try:
        file_io = file_store.read_file(plaintext_file_name, mode="b")
        # Metadata-only file types preserve their original type so
        # downstream injection paths can route them correctly.
        if chat_file_type.use_metadata_only():
            plaintext_chat_file_type = chat_file_type
        elif file_io is not None:
            # if we have plaintext for image (which happens when image
            # extraction is enabled), we use PLAIN_TEXT type
            plaintext_chat_file_type = ChatFileType.PLAIN_TEXT
        else:
            plaintext_chat_file_type = (
                ChatFileType.PLAIN_TEXT
                if chat_file_type != ChatFileType.IMAGE
                else chat_file_type
            )

        chat_file = InMemoryChatFile(
            file_id=str(user_file.file_id),
            content=file_io.read(),
            file_type=plaintext_chat_file_type,
            filename=user_file.name,
        )
        status = "plaintext"
        return chat_file
    except Exception as e:
        logger.warning(f"Failed to load plaintext for user file {user_file.id}: {e}")
        # Fall back to original file if plaintext not available
        file_io = file_store.read_file(user_file.file_id, mode="b")

        chat_file = InMemoryChatFile(
            file_id=str(user_file.file_id),
            content=file_io.read(),
            file_type=chat_file_type,
            filename=user_file.name,
        )
        status = "original"
        return chat_file
    finally:
        logger.debug(
            f"load_user_file finished: file_id={user_file.file_id} chat_file_type={chat_file_type} status={status}"
        )


def load_in_memory_chat_files(
    user_file_ids: list[UUID],
    db_session: Session,
) -> list[InMemoryChatFile]:
    """
    Loads the actual content of user files specified by individual IDs and those
    within specified project IDs into memory.

    Args:
        user_file_ids: A list of specific UserFile IDs to load.
        db_session: The SQLAlchemy database session.

    Returns:
        A list of InMemoryChatFile objects, each containing the file content (as bytes),
        file ID, file type, and filename. Prioritizes loading plaintext versions if available.
    """
    # Use parallel execution to load files concurrently
    return cast(
        list[InMemoryChatFile],
        run_functions_tuples_in_parallel(
            # 1. Load files specified by individual IDs
            [(load_user_file, (file_id, db_session)) for file_id in user_file_ids]
        ),
    )


def save_file_from_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()

    file_io = BytesIO(response.content)
    file_store = get_default_file_store()
    file_id = file_store.save_file(
        content=file_io,
        display_name="GeneratedImage",
        file_origin=FileOrigin.CHAT_IMAGE_GEN,
        file_type="image/png;base64",
    )
    return file_id


def save_file_from_base64(base64_string: str) -> str:
    file_store = get_default_file_store()
    file_id = file_store.save_file(
        content=BytesIO(base64.b64decode(base64_string)),
        display_name="GeneratedImage",
        file_origin=FileOrigin.CHAT_IMAGE_GEN,
        file_type=get_image_type(base64_string),
    )
    return file_id


def save_file(
    url: str | None = None,
    base64_data: str | None = None,
) -> str:
    """Save a file from either a URL or base64 encoded string.

    Args:
        url: URL to download file from
        base64_data: Base64 encoded file data

    Returns:
        The unique ID of the saved file

    Raises:
        ValueError: If neither url nor base64_data is provided, or if both are provided
    """
    if url is not None and base64_data is not None:
        raise ValueError("Cannot specify both url and base64_data")

    if url is not None:
        return save_file_from_url(url)
    elif base64_data is not None:
        return save_file_from_base64(base64_data)
    else:
        raise ValueError("Must specify either url or base64_data")


def save_files(urls: list[str], base64_files: list[str]) -> list[str]:
    # NOTE: be explicit about typing so that if we change things, we get notified
    funcs: list[
        tuple[
            Callable[[str | None, str | None], str],
            tuple[str | None, str | None],
        ]
    ] = [(save_file, (url, None)) for url in urls] + [
        (save_file, (None, base64_file)) for base64_file in base64_files
    ]

    return run_functions_tuples_in_parallel(funcs)


@log_function_time(print_only=True)
def verify_user_files(
    user_files: list[FileDescriptor],
    user_id: UUID | None,
    db_session: Session,
    project_id: int | None = None,
) -> list[FileDescriptor]:
    """
    Authorize client-supplied file descriptors and rebuild them from the
    UserFile rows they resolve to.

    A descriptor with a ``user_file_id`` must reference an existing UserFile
    owned by the user. A descriptor without one must reference a file in
    ``project_id``, which the user must own. In both cases the descriptor's
    ``id`` must equal that UserFile's ``file_id``.

    Callers must use the returned descriptors instead of the input: their
    ``id`` and ``user_file_id`` come from the authorized UserFile rows, and
    ``id`` is what downstream code reads from the file store.

    Raises:
        ValueError: If any descriptor is malformed, references a file that does
            not exist, or references a file the user cannot access.
    """
    from onyx.db.models import Project__UserFile
    from onyx.db.projects import check_project_ownership

    requested: list[tuple[FileDescriptor, UUID | None]] = []
    for file_descriptor in user_files:
        if not file_descriptor.get("id"):
            raise ValueError("File descriptor is missing an id")
        raw_user_file_id = file_descriptor.get("user_file_id")
        if not raw_user_file_id:
            requested.append((file_descriptor, None))
            continue
        try:
            requested.append((file_descriptor, UUID(str(raw_user_file_id))))
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid user_file_id in file descriptor: {raw_user_file_id}"
            )

    user_file_ids = [uf_id for _, uf_id in requested if uf_id is not None]
    project_file_ids = [fd["id"] for fd, uf_id in requested if uf_id is None]

    user_files_by_id: dict[UUID, UserFile] = {}
    if user_file_ids:
        user_files_by_id = {
            uf.id: uf
            for uf in db_session.query(UserFile)
            .filter(UserFile.id.in_(user_file_ids))
            .all()
        }

    project_files_by_file_id: dict[str, UserFile] = {}
    if project_file_ids:
        if project_id is None:
            raise ValueError(
                "Project files provided but no project_id specified for verification"
            )

        if not check_project_ownership(project_id, user_id, db_session):
            raise ValueError(
                f"User {user_id} does not have access to project {project_id}"
            )

        project_files_by_file_id = {
            uf.file_id: uf
            for uf in db_session.query(UserFile)
            .join(Project__UserFile)
            .filter(
                Project__UserFile.project_id == project_id,
                UserFile.file_id.in_(project_file_ids),
            )
            .all()
        }

    authorized: list[FileDescriptor] = []
    for file_descriptor, user_file_id in requested:
        user_file: UserFile | None
        if user_file_id is not None:
            user_file = user_files_by_id.get(user_file_id)
            if user_file is None:
                raise ValueError(f"User file {user_file_id} not found")
            # Note: if user_id is None, then all files should be None as well
            # (since auth must be disabled in this case)
            if user_file.user_id != user_id:
                raise ValueError(
                    f"User {user_id} does not have access to file {user_file.id}"
                )
        else:
            user_file = project_files_by_file_id.get(file_descriptor["id"])
            if user_file is None:
                raise ValueError(
                    f"File {file_descriptor['id']} is not associated with project {project_id}"
                )

        if file_descriptor["id"] != user_file.file_id:
            raise ValueError(
                f"File descriptor id does not match user file {user_file.id}"
            )

        try:
            file_type = ChatFileType(file_descriptor["type"])
        except (KeyError, ValueError):
            raise ValueError(
                f"Invalid file type in file descriptor for user file {user_file.id}"
            )

        authorized.append(
            FileDescriptor(
                id=user_file.file_id,
                type=file_type,
                name=file_descriptor.get("name") or user_file.name,
                user_file_id=str(user_file.id),
            )
        )

    return authorized


def build_frontend_file_url(file_id: str) -> str:
    return f"/api/chat/file/{file_id}"


def build_full_frontend_file_url(file_id: str) -> str:
    return f"{WEB_DOMAIN}/api/chat/file/{file_id}"
