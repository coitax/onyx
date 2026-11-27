"""Authorization spec for GET /chat/file/{file_id}.

The browser calls /api/chat/file/{file_id}. In development, Next.js forwards
that path to this backend route without checking file ownership
(web/src/app/api/[...path]/route.ts). The backend route is
GET /chat/file/{file_id:path} (chat_backend.fetch_chat_file).

These tests call that route through the same FastAPI router, auth dependency,
and Postgres file store the API server uses. They fail while any signed-in
basic user can stream a file they are not allowed to see.

A download may return the file bytes only when the caller is allowed to see
that exact file:

- user file the caller owns
- user file linked to a persona the caller can access, or to a project the
  caller owns
- file attached to a chat message (ChatMessage.files) or recorded on a tool
  call (ToolCall.generated_images / tool_call_response) in a chat session the
  caller owns or that is shared publicly
- persona profile image (Persona.uploaded_image_id) for a persona the caller
  can access
- connector image whose file id is a file-connector upload UUID and whose
  document id is "FILE_CONNECTOR__{file_id}", an embedded-image id
  "{parent_uuid}_img_{n}" for document id "FILE_CONNECTOR__{parent_uuid}",
  or the blob-connector id
  "{bucket_type}_{bucket}_{key with slashes replaced by underscores}" for
  document id "{bucket_type}:{bucket}:{key}", and that document passes the
  caller's document ACL (public connector, document.is_public, external user
  email, or external group membership)

Every other file store id returns 404 and does not create or delete a file.
The bytes returned must be the bytes of the file that was authorized. If the
user-file row's file_id changes after it is first read for this request, the
response must still be the original bytes or 404.

Not covered here: allow-paths for connector images whose file id is only
stored on a search-index chunk (Vespa field image_file_name). This environment
has no search index, so those associations are not inserted. An unlinked
connector file is still required to 404.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from io import BytesIO
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_users.password import PasswordHelper
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserCreate
from onyx.auth.schemas import UserRead
from onyx.auth.schemas import UserRole
from onyx.auth.users import auth_backend
from onyx.auth.users import fastapi_users
from onyx.configs.app_configs import AUTH_BACKEND
from onyx.configs.app_configs import FILE_STORE_BACKEND
from onyx.configs.constants import BlobType
from onyx.configs.constants import DocumentSource
from onyx.configs.constants import FileOrigin
from onyx.configs.constants import MessageType
from onyx.connectors.models import InputType
from onyx.db.chat import create_chat_session
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.engine.sql_engine import SqlEngine
from onyx.db.enums import AccessType
from onyx.db.enums import AccountType
from onyx.db.enums import ConnectorCredentialPairStatus
from onyx.db.enums import ProcessingMode
from onyx.db.enums import UserFileStatus
from onyx.db.models import ChatMessage
from onyx.db.models import ChatSessionSharedStatus
from onyx.db.models import Connector
from onyx.db.models import ConnectorCredentialPair
from onyx.db.models import Credential
from onyx.db.models import Document
from onyx.db.models import DocumentByConnectorCredentialPair
from onyx.db.models import FileContent
from onyx.db.models import FileRecord
from onyx.db.models import Persona
from onyx.db.models import Persona__User
from onyx.db.models import Persona__UserFile
from onyx.db.models import Project__UserFile
from onyx.db.models import ToolCall
from onyx.db.models import User
from onyx.db.models import User__ExternalUserGroupId
from onyx.db.models import UserFile
from onyx.db.models import UserProject
from onyx.db.users import assign_user_to_default_groups__no_commit
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.file_processing.image_utils import store_image_and_create_section
from onyx.file_store.file_store import get_default_file_store
from onyx.file_store.models import ChatFileType
from onyx.kg.models import KGStage
from onyx.main import include_auth_router_with_prefix
from onyx.main import include_router_with_global_prefix_prepended
import onyx.db.user_file as user_file_module
import onyx.server.query_and_chat.chat_backend as chat_backend_module
from onyx.server.query_and_chat.chat_backend import router as chat_router
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable

# Obvious placeholder. Not a deployed credential.
_PLACEHOLDER_PASSWORD = "placeholder-password"

pytestmark = pytest.mark.skipif(
    FILE_STORE_BACKEND != "postgres" or AUTH_BACKEND != "postgres",
    reason=(
        "Run with FILE_STORE_BACKEND=postgres and AUTH_BACKEND=postgres so the "
        "test uses the shipped Postgres file store and Postgres auth backend."
    ),
)


def _build_app() -> FastAPI:
    # Same EE switch the API server runs at startup (main.py).
    set_is_ee_based_on_env_variable()
    application = FastAPI()
    register_onyx_exception_handlers(application)
    include_router_with_global_prefix_prepended(application, chat_router)
    include_auth_router_with_prefix(
        application,
        fastapi_users.get_auth_router(auth_backend),
        prefix="/auth",
    )
    include_auth_router_with_prefix(
        application,
        fastapi_users.get_register_router(UserRead, UserCreate),
        prefix="/auth",
    )
    return application


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    SqlEngine.init_engine(pool_size=10, max_overflow=5)
    with TestClient(_build_app(), raise_server_exceptions=False) as test_client:
        yield test_client


def _create_basic_user(email_prefix: str, *, with_basic_access: bool) -> User:
    password_helper = PasswordHelper()
    user = User(
        id=uuid4(),
        email=f"{email_prefix}_{uuid4().hex[:8]}@example.com",
        hashed_password=password_helper.hash(_PLACEHOLDER_PASSWORD),
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role=UserRole.BASIC,
        account_type=AccountType.STANDARD,
    )
    with get_session_with_current_tenant() as db_session:
        db_session.add(user)
        db_session.flush()
        if with_basic_access:
            assign_user_to_default_groups__no_commit(db_session, user, is_admin=False)
        db_session.commit()
        db_session.refresh(user)
        db_session.expunge(user)
    return user


def _login(client: TestClient, user: User) -> TestClient:
    response = client.post(
        "/auth/login",
        data={"username": user.email, "password": _PLACEHOLDER_PASSWORD},
    )
    assert response.status_code == 204, response.text
    return client


@pytest.fixture(scope="module")
def owner(client: TestClient) -> User:
    user = _create_basic_user("file_owner", with_basic_access=True)
    _login(client, user)
    return user


@pytest.fixture(scope="module")
def attacker(client: TestClient) -> User:
    user = _create_basic_user("file_attacker", with_basic_access=True)
    _login(client, user)
    return user


@pytest.fixture(scope="module")
def stranger(client: TestClient) -> User:
    user = _create_basic_user("file_stranger", with_basic_access=True)
    _login(client, user)
    return user


@pytest.fixture(scope="module")
def no_basic_user(client: TestClient) -> User:
    user = _create_basic_user("file_no_basic", with_basic_access=False)
    _login(client, user)
    return user


def _as(client: TestClient, user: User) -> None:
    _login(client, user)


def _save(content: bytes, origin: FileOrigin, file_id: str | None = None) -> str:
    store = get_default_file_store()
    return store.save_file(
        content=BytesIO(content),
        display_name="stored.bin",
        file_origin=origin,
        file_type="application/octet-stream",
        file_id=file_id,
    )


def _add_user_file(owner: User, content: bytes, origin: FileOrigin = FileOrigin.USER_FILE) -> UserFile:
    file_id = _save(content, origin)
    user_file = UserFile(
        id=uuid4(),
        user_id=owner.id,
        file_id=file_id,
        name="notes.txt",
        file_type="text/plain",
        status=UserFileStatus.COMPLETED,
    )
    with get_session_with_current_tenant() as db_session:
        db_session.add(user_file)
        db_session.commit()
        db_session.refresh(user_file)
        db_session.expunge(user_file)
    return user_file


def _download(client: TestClient, file_id: str, headers: dict[str, str] | None = None) -> Any:
    return client.get(f"/chat/file/{file_id}", headers=headers)


def _assert_bytes(response: Any, content: bytes) -> None:
    assert response.status_code == 200, response.text
    assert response.content == content


def _assert_hidden(response: Any, content: bytes) -> None:
    assert response.status_code == 404, response.text
    assert content not in response.content


def _store_counts() -> dict[str, int]:
    with get_session_with_current_tenant() as db_session:
        large_objects = db_session.execute(text("SELECT count(*) FROM pg_largeobject_metadata")).scalar_one()
        return {
            "file_record": db_session.scalar(select(func.count()).select_from(FileRecord)) or 0,
            "file_content": db_session.scalar(select(func.count()).select_from(FileContent)) or 0,
            "user_file": db_session.scalar(select(func.count()).select_from(UserFile)) or 0,
            "large_objects": int(large_objects),
        }


def _assert_no_file_side_effects(before: dict[str, int]) -> None:
    assert _store_counts() == before


def test_anonymous_and_no_basic_access_cannot_download(client: TestClient, owner: User, no_basic_user: User) -> None:
    user_file = _add_user_file(owner, b"owner-secret-basic-gate")
    client.cookies.clear()
    anonymous = _download(client, user_file.file_id)
    assert anonymous.status_code == 403
    assert b"owner-secret-basic-gate" not in anonymous.content

    _as(client, no_basic_user)
    denied = _download(client, user_file.file_id)
    assert denied.status_code == 403
    assert b"owner-secret-basic-gate" not in denied.content


def test_owner_can_download_user_file_by_both_ids(client: TestClient, owner: User) -> None:
    content = b"owner-user-file-bytes"
    user_file = _add_user_file(owner, content)
    _as(client, owner)
    _assert_bytes(_download(client, user_file.file_id), content)
    _assert_bytes(_download(client, str(user_file.id)), content)


def test_other_basic_user_cannot_download_user_file_by_either_id(client: TestClient, owner: User, attacker: User) -> None:
    content = b"victim-user-file-secret"
    user_file = _add_user_file(owner, content)
    _as(client, attacker)
    before = _store_counts()
    by_store_id = _download(client, user_file.file_id)
    _assert_hidden(by_store_id, content)
    _assert_no_file_side_effects(before)

    before = _store_counts()
    by_user_file_id = _download(client, str(user_file.id))
    _assert_hidden(by_user_file_id, content)
    _assert_no_file_side_effects(before)

    etag = f'"{user_file.file_id}"'
    cached = _download(client, user_file.file_id, headers={"if-none-match": etag})
    _assert_hidden(cached, content)


def test_user_file_id_resolution_returns_the_authorized_bytes_only(
    client: TestClient, owner: User, attacker: User, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If user_file.file_id changes after the first resolution, do not stream the new target.

    The real resolver runs. After it returns, the row is pointed at a different
    file and the request session is expired. A second lookup in the same request
    observes the new file id; the bytes of that file must not be returned.
    """
    allowed = b"allowed-file-bytes-for-owner"
    forbidden = b"forbidden-file-bytes-swapped-in"
    user_file = _add_user_file(owner, allowed)
    forbidden_id = _save(forbidden, FileOrigin.CHAT_UPLOAD)
    watched = str(user_file.id)
    state = {"swapped": False}
    real_lookup = user_file_module.get_file_id_by_user_file_id

    def _lookup_then_retarget(user_file_id: str, db_session: Session) -> str | None:
        resolved = real_lookup(user_file_id, db_session)
        if not state["swapped"] and str(user_file_id) == watched:
            state["swapped"] = True
            with get_session_with_current_tenant() as other:
                row = other.get(UserFile, user_file.id)
                assert row is not None
                row.file_id = forbidden_id
                other.commit()
            db_session.expire_all()
        return resolved

    monkeypatch.setattr(chat_backend_module, "get_file_id_by_user_file_id", _lookup_then_retarget)
    monkeypatch.setattr(user_file_module, "get_file_id_by_user_file_id", _lookup_then_retarget)

    _as(client, owner)
    response = _download(client, watched)

    assert state["swapped"], "the request never resolved the user-file id"
    assert forbidden not in response.content
    if response.status_code == 200:
        assert response.content == allowed
    else:
        assert response.status_code == 404

    _as(client, attacker)
    _assert_hidden(_download(client, watched), allowed)
    _assert_hidden(_download(client, watched), forbidden)


def test_project_and_persona_sharing(client: TestClient, owner: User, attacker: User, stranger: User) -> None:
    project_bytes = b"project-shared-secret"
    project_file = _add_user_file(owner, project_bytes)
    with get_session_with_current_tenant() as db_session:
        project = UserProject(user_id=attacker.id, name="shared-project", instructions="")
        db_session.add(project)
        db_session.flush()
        db_session.add(Project__UserFile(project_id=project.id, user_file_id=project_file.id))
        db_session.commit()

    _as(client, attacker)
    _assert_bytes(_download(client, str(project_file.id)), project_bytes)
    _as(client, stranger)
    _assert_hidden(_download(client, str(project_file.id)), project_bytes)


def test_persona_user_file_follows_persona_access(client: TestClient, owner: User, attacker: User, stranger: User) -> None:
    persona_bytes = b"persona-shared-secret"
    persona_file = _add_user_file(owner, persona_bytes)
    with get_session_with_current_tenant() as db_session:
        persona = Persona(
            name=f"private-{uuid4().hex[:8]}",
            description="private",
            is_public=False,
            is_listed=False,
            user_id=owner.id,
        )
        db_session.add(persona)
        db_session.flush()
        db_session.add(Persona__UserFile(persona_id=persona.id, user_file_id=persona_file.id))
        db_session.add(Persona__User(persona_id=persona.id, user_id=attacker.id))
        db_session.commit()

    _as(client, attacker)
    _assert_bytes(_download(client, str(persona_file.id)), persona_bytes)
    _as(client, stranger)
    _assert_hidden(_download(client, str(persona_file.id)), persona_bytes)


def test_revoked_persona_user_loses_user_file_download(client: TestClient, owner: User, attacker: User) -> None:
    persona_bytes = b"persona-revoked-secret"
    persona_file = _add_user_file(owner, persona_bytes)
    with get_session_with_current_tenant() as db_session:
        persona = Persona(
            name=f"revoked-{uuid4().hex[:8]}",
            description="revoked",
            is_public=False,
            is_listed=False,
            user_id=owner.id,
        )
        db_session.add(persona)
        db_session.flush()
        db_session.add(Persona__UserFile(persona_id=persona.id, user_file_id=persona_file.id))
        db_session.add(Persona__User(persona_id=persona.id, user_id=attacker.id))
        db_session.commit()
        persona_id = persona.id

    with get_session_with_current_tenant() as db_session:
        db_session.execute(
            Persona__User.__table__.delete().where(
                Persona__User.persona_id == persona_id,
                Persona__User.user_id == attacker.id,
            )
        )
        db_session.commit()

    _as(client, attacker)
    _assert_hidden(_download(client, str(persona_file.id)), persona_bytes)
    _as(client, owner)
    _assert_bytes(_download(client, str(persona_file.id)), persona_bytes)


def test_persona_profile_image_follows_persona_access(client: TestClient, owner: User, attacker: User, stranger: User) -> None:
    image = b"persona-profile-image"
    file_id = _save(image, FileOrigin.CHAT_UPLOAD)
    with get_session_with_current_tenant() as db_session:
        persona = Persona(
            name=f"avatar-{uuid4().hex[:8]}",
            description="avatar",
            is_public=False,
            is_listed=False,
            user_id=owner.id,
            uploaded_image_id=file_id,
        )
        db_session.add(persona)
        db_session.flush()
        db_session.add(Persona__User(persona_id=persona.id, user_id=attacker.id))
        db_session.commit()

    _as(client, attacker)
    _assert_bytes(_download(client, file_id), image)
    _as(client, stranger)
    _assert_hidden(_download(client, file_id), image)


def test_revoked_persona_user_loses_profile_image_download(client: TestClient, owner: User, attacker: User) -> None:
    image = b"revoked-persona-profile-image"
    file_id = _save(image, FileOrigin.CHAT_UPLOAD)
    with get_session_with_current_tenant() as db_session:
        persona = Persona(
            name=f"avatar-revoked-{uuid4().hex[:8]}",
            description="avatar",
            is_public=False,
            is_listed=False,
            user_id=owner.id,
            uploaded_image_id=file_id,
        )
        db_session.add(persona)
        db_session.flush()
        db_session.add(Persona__User(persona_id=persona.id, user_id=attacker.id))
        db_session.commit()
        persona_id = persona.id

    with get_session_with_current_tenant() as db_session:
        db_session.execute(Persona__User.__table__.delete().where(Persona__User.persona_id == persona_id))
        db_session.commit()
    _as(client, attacker)
    _assert_hidden(_download(client, file_id), image)


def _chat_with_file(
    owner: User,
    content: bytes,
    *,
    shared: bool,
    origin: FileOrigin,
) -> tuple[str, str]:
    file_id = _save(content, origin)
    with get_session_with_current_tenant() as db_session:
        persona = Persona(name=f"chat-{uuid4().hex[:8]}", description="chat")
        db_session.add(persona)
        db_session.flush()
        chat = create_chat_session(
            db_session=db_session,
            description="chat",
            user_id=owner.id,
            persona_id=persona.id,
        )
        chat.shared_status = ChatSessionSharedStatus.PUBLIC if shared else ChatSessionSharedStatus.PRIVATE
        db_session.add(
            ChatMessage(
                chat_session_id=chat.id,
                message="see attachment",
                token_count=1,
                message_type=MessageType.USER,
                files=[
                    {
                        "id": file_id,
                        "type": ChatFileType.PLAIN_TEXT.value,
                        "name": "notes.txt",
                    }
                ],
            )
        )
        db_session.commit()
    return file_id, str(chat.id)


def test_chat_attachment_and_generated_image_follow_session_access(client: TestClient, owner: User, attacker: User) -> None:
    private_bytes = b"private-chat-attachment"
    file_id, _session_id = _chat_with_file(owner, private_bytes, shared=False, origin=FileOrigin.CHAT_UPLOAD)
    _as(client, owner)
    _assert_bytes(_download(client, file_id), private_bytes)
    _as(client, attacker)
    before = _store_counts()
    _assert_hidden(_download(client, file_id), private_bytes)
    _assert_no_file_side_effects(before)


def test_public_chat_attachment_is_visible_to_other_basic_users(client: TestClient, owner: User, attacker: User) -> None:
    public_bytes = b"public-chat-attachment"
    public_id, _public_session = _chat_with_file(owner, public_bytes, shared=True, origin=FileOrigin.CHAT_UPLOAD)
    _as(client, attacker)
    _assert_bytes(_download(client, public_id), public_bytes)


def test_generated_image_follows_chat_session_access(client: TestClient, owner: User, attacker: User) -> None:
    generated = b"generated-image-secret"
    generated_id = _save(generated, FileOrigin.CHAT_IMAGE_GEN)
    with get_session_with_current_tenant() as db_session:
        persona = Persona(name=f"img-{uuid4().hex[:8]}", description="img")
        db_session.add(persona)
        db_session.flush()
        chat = create_chat_session(
            db_session=db_session,
            description="images",
            user_id=owner.id,
            persona_id=persona.id,
        )
        message = ChatMessage(
            chat_session_id=chat.id,
            message="image",
            token_count=1,
            message_type=MessageType.ASSISTANT,
        )
        db_session.add(message)
        db_session.flush()
        db_session.add(
            ToolCall(
                chat_session_id=chat.id,
                parent_chat_message_id=message.id,
                turn_number=0,
                tab_index=0,
                tool_id=0,
                tool_call_id=f"call_{uuid4().hex[:8]}",
                tool_call_arguments={"prompt": "draw"},
                tool_call_response=json.dumps([{"file_id": generated_id}]),
                tool_call_tokens=1,
                generated_images=[
                    {
                        "file_id": generated_id,
                        "url": f"/api/chat/file/{generated_id}",
                        "revised_prompt": "draw",
                    }
                ],
            )
        )
        db_session.commit()

    _as(client, owner)
    _assert_bytes(_download(client, generated_id), generated)
    _as(client, attacker)
    _assert_hidden(_download(client, generated_id), generated)


def _index_document(
    *,
    document_id: str,
    source: DocumentSource,
    access_type: AccessType,
    owner: User,
    external_emails: list[str] | None = None,
    external_groups: list[str] | None = None,
    is_public: bool = False,
    group_member: User | None = None,
) -> None:
    with get_session_with_current_tenant() as db_session:
        connector = Connector(
            name=f"connector-{uuid4().hex[:8]}",
            source=source,
            input_type=InputType.POLL,
            connector_specific_config={},
            refresh_freq=None,
        )
        credential = Credential(
            name="cred",
            source=source,
            credential_json={},
            user_id=owner.id,
            admin_public=False,
        )
        db_session.add_all([connector, credential])
        db_session.flush()
        cc_pair = ConnectorCredentialPair(
            name=f"cc-{uuid4().hex[:8]}",
            status=ConnectorCredentialPairStatus.ACTIVE,
            connector_id=connector.id,
            credential_id=credential.id,
            access_type=access_type,
            processing_mode=ProcessingMode.REGULAR,
            in_repeated_error_state=False,
            total_docs_indexed=1,
        )
        db_session.add(cc_pair)
        db_session.flush()
        db_session.add(
            Document(
                id=document_id,
                semantic_id=document_id,
                kg_stage=KGStage.NOT_STARTED,
                is_public=is_public,
                external_user_emails=external_emails,
                external_user_group_ids=external_groups,
            )
        )
        db_session.add(
            DocumentByConnectorCredentialPair(
                id=document_id,
                connector_id=connector.id,
                credential_id=credential.id,
                has_been_indexed=True,
            )
        )
        if group_member is not None and external_groups:
            for group_id in external_groups:
                db_session.add(
                    User__ExternalUserGroupId(
                        user_id=group_member.id,
                        external_user_group_id=group_id,
                        cc_pair_id=cc_pair.id,
                        stale=False,
                    )
                )
        db_session.commit()


def _store_connector_image(file_id: str, content: bytes) -> str:
    section, stored_id = store_image_and_create_section(
        image_data=content,
        file_id=file_id,
        display_name="photo.png",
        media_type="image/png",
        file_origin=FileOrigin.CONNECTOR,
    )
    assert stored_id == file_id
    assert section.image_file_id == file_id
    return file_id


def test_uuid_connector_image_follows_document_acl(client: TestClient, owner: User, attacker: User) -> None:
    """File-connector images are stored under the uploaded file's UUID.

    backend/onyx/connectors/file/connector.py uses that UUID as the image file id
    and FILE_CONNECTOR__{uuid} as the document id.
    """
    image_id = str(uuid4())
    secret = b"file-connector-private-image"
    _store_connector_image(image_id, secret)
    _index_document(
        document_id=f"FILE_CONNECTOR__{image_id}",
        source=DocumentSource.FILE,
        access_type=AccessType.PRIVATE,
        owner=owner,
    )
    _as(client, owner)
    _assert_bytes(_download(client, image_id), secret)
    _as(client, attacker)
    before = _store_counts()
    _assert_hidden(_download(client, image_id), secret)
    _assert_no_file_side_effects(before)


def test_public_and_acl_connector_images_follow_document_access(client: TestClient, owner: User, attacker: User) -> None:
    public_id = str(uuid4())
    public_secret = b"file-connector-public-image"
    _store_connector_image(public_id, public_secret)
    _index_document(
        document_id=f"FILE_CONNECTOR__{public_id}",
        source=DocumentSource.FILE,
        access_type=AccessType.PUBLIC,
        owner=owner,
    )
    _as(client, attacker)
    _assert_bytes(_download(client, public_id), public_secret)


def test_external_email_acl_limits_connector_image_download(
    client: TestClient, owner: User, attacker: User, stranger: User
) -> None:
    emailed_id = str(uuid4())
    emailed_secret = b"file-connector-email-acl-image"
    _store_connector_image(emailed_id, emailed_secret)
    _index_document(
        document_id=f"FILE_CONNECTOR__{emailed_id}",
        source=DocumentSource.FILE,
        access_type=AccessType.SYNC,
        owner=owner,
        external_emails=[attacker.email],
    )
    _as(client, attacker)
    _assert_bytes(_download(client, emailed_id), emailed_secret)
    _as(client, stranger)
    _assert_hidden(_download(client, emailed_id), emailed_secret)


def test_external_group_acl_limits_connector_image_download(
    client: TestClient, owner: User, attacker: User, stranger: User
) -> None:
    group_id = f"file_finance-{uuid4().hex[:8]}"
    grouped_image_id = str(uuid4())
    grouped_secret = b"file-connector-group-acl-image"
    _store_connector_image(grouped_image_id, grouped_secret)
    _index_document(
        document_id=f"FILE_CONNECTOR__{grouped_image_id}",
        source=DocumentSource.FILE,
        access_type=AccessType.SYNC,
        owner=owner,
        external_groups=[group_id],
        group_member=attacker,
    )
    _as(client, attacker)
    _assert_bytes(_download(client, grouped_image_id), grouped_secret)
    _as(client, stranger)
    _assert_hidden(_download(client, grouped_image_id), grouped_secret)


def test_document_is_public_allows_connector_image_download(client: TestClient, owner: User, stranger: User) -> None:
    doc_public_id = str(uuid4())
    doc_public_secret = b"document-is-public-image"
    _store_connector_image(doc_public_id, doc_public_secret)
    _index_document(
        document_id=f"FILE_CONNECTOR__{doc_public_id}",
        source=DocumentSource.FILE,
        access_type=AccessType.SYNC,
        owner=owner,
        is_public=True,
    )
    _as(client, stranger)
    _assert_bytes(_download(client, doc_public_id), doc_public_secret)


def test_non_uuid_connector_image_ids_follow_document_acl(client: TestClient, owner: User) -> None:
    """Blob ids and embedded image ids are not UUIDs.

    The blob id is "{bucket_type}_{bucket}_{key with '/' replaced by '_'}".
    The document id is "{bucket_type}:{bucket}:{key}" (blob/connector.py).
    Embedded file-connector images use "{parent_file_uuid}_img_{n}" and the
    parent document id "FILE_CONNECTOR__{parent_file_uuid}".

    The caller named on the document ACL receives the bytes. Anyone else does not.
    """
    bucket_type = BlobType.S3
    bucket_name = "private-bucket"
    object_key = f"folder/secret-{uuid4().hex}.png"
    blob_file_id = f"{bucket_type}_{bucket_name}_{object_key.replace('/', '_')}"
    blob_document_id = f"{bucket_type}:{bucket_name}:{object_key}"
    blob_secret = b"blob-connector-image-not-a-uuid"
    _store_connector_image(blob_file_id, blob_secret)
    _index_document(
        document_id=blob_document_id,
        source=DocumentSource.S3,
        access_type=AccessType.SYNC,
        owner=owner,
        external_emails=[owner.email],
    )

    parent_id = str(uuid4())
    embedded_id = f"{parent_id}_img_0"
    embedded_secret = b"embedded-image-id-not-a-uuid"
    _store_connector_image(embedded_id, embedded_secret)
    _index_document(
        document_id=f"FILE_CONNECTOR__{parent_id}",
        source=DocumentSource.FILE,
        access_type=AccessType.SYNC,
        owner=owner,
        external_emails=[owner.email],
    )

    _as(client, owner)
    _assert_bytes(_download(client, blob_file_id), blob_secret)
    _assert_bytes(_download(client, embedded_id), embedded_secret)


def test_non_uuid_connector_image_ids_are_hidden_from_other_users(client: TestClient, owner: User, attacker: User) -> None:
    bucket_type = BlobType.S3
    bucket_name = "private-bucket"
    object_key = f"folder/secret-{uuid4().hex}.png"
    blob_file_id = f"{bucket_type}_{bucket_name}_{object_key.replace('/', '_')}"
    blob_secret = b"blob-connector-image-hidden-from-others"
    _store_connector_image(blob_file_id, blob_secret)
    _index_document(
        document_id=f"{bucket_type}:{bucket_name}:{object_key}",
        source=DocumentSource.S3,
        access_type=AccessType.SYNC,
        owner=owner,
        external_emails=[owner.email],
    )
    parent_id = str(uuid4())
    embedded_id = f"{parent_id}_img_0"
    embedded_secret = b"embedded-image-hidden-from-others"
    _store_connector_image(embedded_id, embedded_secret)
    _index_document(
        document_id=f"FILE_CONNECTOR__{parent_id}",
        source=DocumentSource.FILE,
        access_type=AccessType.SYNC,
        owner=owner,
        external_emails=[owner.email],
    )
    _as(client, attacker)
    _assert_hidden(_download(client, blob_file_id), blob_secret)
    _assert_hidden(_download(client, embedded_id), embedded_secret)


@pytest.mark.parametrize(
    ("origin", "content"),
    [
        (FileOrigin.CONNECTOR, b"orphan-connector-image"),
        (FileOrigin.PLAINTEXT_CACHE, b"plaintext-cache-secret"),
        (FileOrigin.INDEXING_CHECKPOINT, b"checkpoint-secret"),
        (FileOrigin.GENERATED_REPORT, b"report-secret"),
        (FileOrigin.QUERY_HISTORY_CSV, b"query-history-secret"),
        (FileOrigin.SANDBOX_SNAPSHOT, b"sandbox-snapshot-secret"),
        (FileOrigin.OTHER, b"other-origin-secret"),
    ],
)
def test_unlinked_file_store_records_are_hidden(
    client: TestClient,
    owner: User,
    attacker: User,
    origin: FileOrigin,
    content: bytes,
) -> None:
    file_id = _save(content, origin)
    for user in (owner, attacker):
        _as(client, user)
        before = _store_counts()
        _assert_hidden(_download(client, file_id), content)
        _assert_no_file_side_effects(before)
