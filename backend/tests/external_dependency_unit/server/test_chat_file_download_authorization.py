"""Authorization spec for GET /chat/file/{file_id}.

The public URL is /api/chat/file/{file_id}. Nginx strips the /api prefix and
the backend route is /chat/file/{file_id:path}.

These tests exercise that route, the BASIC_ACCESS gate, the user-file id
lookup, and the file store read. They fail while a signed-in user can read a
file they are not allowed to read.

Expected access, by the object that is actually streamed:

- User-file bytes: the file's owner, or a user who can access a persona the
  file is attached to (persona owner, a user linked on persona__user, or any
  basic user when the persona is public and listed).
- Chat upload and chat image bytes: the chat session owner, or any basic user
  when that session is shared publicly and a message in it references the file.
- Connector bytes: only when the same caller can already access them through
  one of the rules above. Document-index access (Vespa/OpenSearch
  image_file_name) is not covered here.
- Indexing checkpoints, indexing batches, sandbox snapshots, plaintext caches,
  query-history exports, generated reports, and connector metadata: never
  returned by this route, including to an admin or to the owner of a user-file
  row whose file_id points at that object.

A denial must not delete the stored bytes. The bytes returned for an allowed
id must be the bytes of that id, not the bytes of a different id the caller
can also access.
"""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.auth.users import current_user
from onyx.configs.constants import FileOrigin
from onyx.configs.constants import MessageType
from onyx.db.engine.sql_engine import get_sqlalchemy_engine
from onyx.db.enums import AccessType
from onyx.db.enums import AccountType
from onyx.db.enums import ChatSessionSharedStatus
from onyx.db.enums import ConnectorCredentialPairStatus
from onyx.db.enums import UserFileStatus
from onyx.db.models import ChatMessage
from onyx.db.models import ChatSession
from onyx.db.models import Connector
from onyx.db.models import ConnectorCredentialPair
from onyx.db.models import Credential
from onyx.db.models import Document
from onyx.db.models import DocumentByConnectorCredentialPair
from onyx.db.models import Persona
from onyx.db.models import Persona__User
from onyx.db.models import Persona__UserFile
from onyx.db.models import User
from onyx.db.models import UserFile
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.file_store.file_store import get_default_file_store
from onyx.file_store.utils import plaintext_file_name_for_id
from onyx.kg.models import KGStage
from onyx.server.query_and_chat.chat_backend import router as chat_router
from shared_configs.contextvars import get_current_tenant_id

_DENIED = frozenset({401, 403, 404})


def _user(role: UserRole, permissions: list[str]) -> User:
    return User(
        id=uuid4(),
        email=f"chat-file-{uuid4().hex[:10]}@example.com",
        hashed_password="not-a-secret",
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role=role,
        account_type=AccountType.STANDARD,
        effective_permissions=permissions,
    )


def _save(file_id: str, content: bytes, origin: FileOrigin, file_type: str) -> None:
    get_default_file_store().save_file(
        content=BytesIO(content),
        display_name=file_id,
        file_origin=origin,
        file_type=file_type,
        file_id=file_id,
    )


def _build_app() -> TestClient:
    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(chat_router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def client(db_session: Session) -> Iterator[TestClient]:
    # db_session initializes the engine. The route opens its own session.
    del db_session
    test_client = _build_app()
    yield test_client
    test_client.app.dependency_overrides.clear()


def _as(client: TestClient, user: User | None) -> None:
    if user is None:
        client.app.dependency_overrides.pop(current_user, None)
        return

    client.app.dependency_overrides[current_user] = lambda: user


def _get(client: TestClient, user: User | None, file_id: str):
    _as(client, user)
    return client.get(f"/chat/file/{file_id}")


def _assert_denied(response, secret: bytes) -> None:
    assert response.status_code in _DENIED, response.status_code
    assert secret not in response.content


def _assert_exact(response, secret: bytes) -> None:
    assert response.status_code == 200, response.status_code
    assert response.content == secret


def _assert_not_served(response, secret: bytes) -> None:
    """Non-UUID ids are rejected by the user-file lookup before the read.

    A later change that catches that error and then streams the file must
    still fail this assertion.
    """

    assert response.status_code != 200, response.status_code
    assert secret not in response.content


class TestChatFileDownloadAuthorization:
    @pytest.fixture(autouse=True)
    def _records(self, db_session: Session) -> Iterator[None]:
        self.db = db_session
        self.owner = _user(UserRole.BASIC, ["basic"])
        self.other = _user(UserRole.BASIC, ["basic"])
        self.admin = _user(UserRole.ADMIN, ["admin"])
        self.no_basic = _user(UserRole.BASIC, [])
        for user in (self.owner, self.other, self.admin, self.no_basic):
            # One row at a time. SQLAlchemy 2.0.15 insertmanyvalues cannot
            # match UUID sentinels returned by this psycopg2 build.
            self.db.add(user)
            self.db.flush()

        self.owner_bytes = b"owner-user-file-bytes"
        self.other_bytes = b"other-user-private-bytes"
        self.chat_bytes = b"private-chat-upload-bytes"
        self.image_bytes = b"private-chat-image-bytes"
        self.public_chat_bytes = b"public-chat-upload-bytes"
        self.connector_bytes = b"connector-image-bytes"
        self.report_bytes = b"generated-report-bytes"
        self.metadata_bytes = b"connector-metadata-bytes"
        self.checkpoint_bytes = b"indexing-checkpoint-bytes"
        self.batch_bytes = b'{"documents":[{"text":"restricted confluence page"}]}'
        self.snapshot_bytes = b"sandbox-snapshot-bytes"
        self.plaintext_bytes = b"plaintext-cache-bytes"
        self.history_bytes = b"query-history-bytes"
        self.attachment_bytes = b"confluence-attachment-bytes"
        self.persona_bytes = b"private-persona-file-bytes"
        self.public_persona_bytes = b"public-persona-file-bytes"

        self.owner_blob = str(uuid4())
        self.other_blob = str(uuid4())
        self.chat_blob = str(uuid4())
        self.image_blob = str(uuid4())
        self.public_chat_blob = str(uuid4())
        self.connector_blob = str(uuid4())
        self.report_blob = str(uuid4())
        self.metadata_blob = str(uuid4())
        self.checkpoint_id = f"checkpoint_{uuid4().hex}.json"
        self.batch_id = f"iab/{uuid4().int % 1000}/{uuid4().int % 1000}/0.json"
        self.snapshot_id = (
            f"sandbox-snapshots/{get_current_tenant_id()}/{uuid4()}/{uuid4()}.tar.gz"
        )
        self.history_id = f"query-history-{uuid4()}.csv"
        self.attachment_id = "att999001"
        self.persona_blob = str(uuid4())
        self.public_persona_blob = str(uuid4())

        _save(self.owner_blob, self.owner_bytes, FileOrigin.USER_FILE, "text/plain")
        _save(self.other_blob, self.other_bytes, FileOrigin.USER_FILE, "text/plain")
        _save(self.chat_blob, self.chat_bytes, FileOrigin.CHAT_UPLOAD, "text/plain")
        _save(self.image_blob, self.image_bytes, FileOrigin.CHAT_IMAGE_GEN, "image/png")
        _save(
            self.public_chat_blob,
            self.public_chat_bytes,
            FileOrigin.CHAT_UPLOAD,
            "text/plain",
        )
        _save(
            self.connector_blob,
            self.connector_bytes,
            FileOrigin.CONNECTOR,
            "image/png",
        )
        _save(
            self.report_blob,
            self.report_bytes,
            FileOrigin.GENERATED_REPORT,
            "text/csv",
        )
        _save(
            self.metadata_blob,
            self.metadata_bytes,
            FileOrigin.CONNECTOR_METADATA,
            "application/json",
        )
        _save(
            self.checkpoint_id,
            self.checkpoint_bytes,
            FileOrigin.INDEXING_CHECKPOINT,
            "application/json",
        )
        _save(self.batch_id, self.batch_bytes, FileOrigin.OTHER, "application/json")
        _save(
            self.snapshot_id,
            self.snapshot_bytes,
            FileOrigin.SANDBOX_SNAPSHOT,
            "application/gzip",
        )
        _save(
            self.history_id,
            self.history_bytes,
            FileOrigin.QUERY_HISTORY_CSV,
            "text/csv",
        )
        _save(
            self.attachment_id,
            self.attachment_bytes,
            FileOrigin.CONNECTOR,
            "image/png",
        )
        _save(
            self.persona_blob,
            self.persona_bytes,
            FileOrigin.USER_FILE,
            "text/plain",
        )
        _save(
            self.public_persona_blob,
            self.public_persona_bytes,
            FileOrigin.USER_FILE,
            "text/plain",
        )

        self.owner_file = UserFile(
            id=uuid4(),
            user_id=self.owner.id,
            file_id=self.owner_blob,
            name="owner.txt",
            file_type="text/plain",
            status=UserFileStatus.COMPLETED,
        )
        self.other_file = UserFile(
            id=uuid4(),
            user_id=self.other.id,
            file_id=self.other_blob,
            name="other.txt",
            file_type="text/plain",
            status=UserFileStatus.COMPLETED,
        )
        self.persona_file = UserFile(
            id=uuid4(),
            user_id=self.owner.id,
            file_id=self.persona_blob,
            name="persona.txt",
            file_type="text/plain",
            status=UserFileStatus.COMPLETED,
        )
        self.public_persona_file = UserFile(
            id=uuid4(),
            user_id=self.owner.id,
            file_id=self.public_persona_blob,
            name="public-persona.txt",
            file_type="text/plain",
            status=UserFileStatus.COMPLETED,
        )
        # Owner row whose blob id is an indexing checkpoint. Authorizing the
        # row must not stream the checkpoint.
        self.checkpoint_pointer = UserFile(
            id=uuid4(),
            user_id=self.owner.id,
            file_id=self.checkpoint_id,
            name="checkpoint-pointer.txt",
            file_type="application/json",
            status=UserFileStatus.COMPLETED,
        )
        for user_file in (
            self.owner_file,
            self.other_file,
            self.persona_file,
            self.public_persona_file,
            self.checkpoint_pointer,
        ):
            self.db.add(user_file)
            self.db.flush()

        private_persona = Persona(
            name=f"private-{uuid4().hex[:8]}",
            description="private",
            user_id=self.owner.id,
            is_public=False,
            is_listed=True,
            deleted=False,
        )
        public_persona = Persona(
            name=f"public-{uuid4().hex[:8]}",
            description="public",
            user_id=self.owner.id,
            is_public=True,
            is_listed=True,
            deleted=False,
        )
        self.db.add_all([private_persona, public_persona])
        self.db.flush()
        self.db.add_all(
            [
                Persona__UserFile(
                    persona_id=private_persona.id,
                    user_file_id=self.persona_file.id,
                ),
                Persona__UserFile(
                    persona_id=public_persona.id,
                    user_file_id=self.public_persona_file.id,
                ),
                Persona__User(persona_id=private_persona.id, user_id=self.owner.id),
            ]
        )

        private_chat = ChatSession(
            id=uuid4(),
            user_id=self.owner.id,
            shared_status=ChatSessionSharedStatus.PRIVATE,
            description="private",
        )
        public_chat = ChatSession(
            id=uuid4(),
            user_id=self.owner.id,
            shared_status=ChatSessionSharedStatus.PUBLIC,
            description="public",
        )
        self.db.add(private_chat)
        self.db.flush()
        self.db.add(public_chat)
        self.db.flush()
        self.db.add_all(
            [
                ChatMessage(
                    chat_session_id=private_chat.id,
                    message="see attachment",
                    token_count=1,
                    message_type=MessageType.USER,
                    files=[
                        {
                            "id": self.chat_blob,
                            "type": "plain_text",
                            "name": "note.txt",
                        },
                        {
                            "id": self.image_blob,
                            "type": "image",
                            "name": "pic.png",
                        },
                    ],
                ),
                ChatMessage(
                    chat_session_id=public_chat.id,
                    message="shared attachment",
                    token_count=1,
                    message_type=MessageType.USER,
                    files=[
                        {
                            "id": self.public_chat_blob,
                            "type": "plain_text",
                            "name": "shared.txt",
                        }
                    ],
                ),
            ]
        )

        # Restricted connector document. The image id is not stored on the
        # document row; the index link is out of scope for this test.
        connector = Connector(
            name=f"conf-{uuid4().hex[:8]}",
            source="confluence",
            input_type="poll",
            connector_specific_config={},
            refresh_freq=60 * 60,
            prune_freq=60 * 60,
        )
        credential = Credential(
            name="conf",
            source="confluence",
            user_id=self.owner.id,
            admin_public=False,
        )
        self.db.add_all([connector, credential])
        self.db.flush()
        self.db.add(
            ConnectorCredentialPair(
                name="conf pair",
                status=ConnectorCredentialPairStatus.ACTIVE,
                connector_id=connector.id,
                credential_id=credential.id,
                access_type=AccessType.SYNC,
            )
        )
        document = Document(
            id=f"conf-{uuid4().hex}",
            semantic_id="Restricted page",
            boost=0,
            hidden=False,
            external_user_emails=[self.owner.email],
            is_public=False,
            kg_stage=KGStage.SKIPPED,
        )
        self.db.add(document)
        self.db.flush()
        self.db.add(
            DocumentByConnectorCredentialPair(
                id=document.id,
                connector_id=connector.id,
                credential_id=credential.id,
                has_been_indexed=True,
            )
        )
        self.db.commit()

        self.plaintext_id = plaintext_file_name_for_id(str(self.other_file.id))
        _save(
            self.plaintext_id,
            self.plaintext_bytes,
            FileOrigin.PLAINTEXT_CACHE,
            "text/plain",
        )
        yield

    def test_unauthenticated_request_is_rejected(self, client: TestClient) -> None:
        response = _get(client, None, str(self.other_file.id))
        _assert_denied(response, self.other_bytes)

    def test_user_without_basic_access_is_rejected(self, client: TestClient) -> None:
        response = _get(client, self.no_basic, str(self.other_file.id))
        _assert_denied(response, self.other_bytes)

    def test_owner_reads_user_file_by_user_file_id_and_blob_id(
        self, client: TestClient
    ) -> None:
        _assert_exact(
            _get(client, self.owner, str(self.owner_file.id)), self.owner_bytes
        )
        _assert_exact(_get(client, self.owner, self.owner_blob), self.owner_bytes)

    def test_other_user_cannot_read_user_file_by_user_file_id(
        self, client: TestClient
    ) -> None:
        response = _get(client, self.other, str(self.owner_file.id))
        _assert_denied(response, self.owner_bytes)
        _assert_exact(
            _get(client, self.owner, str(self.owner_file.id)), self.owner_bytes
        )

    def test_other_user_cannot_read_user_file_by_blob_id(
        self, client: TestClient
    ) -> None:
        response = _get(client, self.other, self.owner_blob)
        _assert_denied(response, self.owner_bytes)
        _assert_exact(_get(client, self.owner, self.owner_blob), self.owner_bytes)

    def test_admin_cannot_read_another_users_file(self, client: TestClient) -> None:
        response = _get(client, self.admin, str(self.owner_file.id))
        _assert_denied(response, self.owner_bytes)
        response = _get(client, self.admin, self.owner_blob)
        _assert_denied(response, self.owner_bytes)

    def test_allowed_file_is_not_substituted_for_another_file(
        self, client: TestClient
    ) -> None:
        # The caller can read owner_blob. Requesting other_blob must not
        # return other_blob, and must not return owner_blob either.
        response = _get(client, self.owner, self.other_blob)
        _assert_denied(response, self.other_bytes)
        assert self.owner_bytes not in response.content

    def test_chat_owner_reads_chat_upload_and_generated_image(
        self, client: TestClient
    ) -> None:
        _assert_exact(_get(client, self.owner, self.chat_blob), self.chat_bytes)
        _assert_exact(_get(client, self.owner, self.image_blob), self.image_bytes)

    def test_other_user_cannot_read_private_chat_files(
        self, client: TestClient
    ) -> None:
        for file_id, secret in (
            (self.chat_blob, self.chat_bytes),
            (self.image_blob, self.image_bytes),
        ):
            response = _get(client, self.other, file_id)
            _assert_denied(response, secret)
        _assert_exact(_get(client, self.owner, self.chat_blob), self.chat_bytes)
        _assert_exact(_get(client, self.owner, self.image_blob), self.image_bytes)

    def test_public_chat_file_is_readable_by_another_basic_user(
        self, client: TestClient
    ) -> None:
        _assert_exact(
            _get(client, self.other, self.public_chat_blob), self.public_chat_bytes
        )

    def test_private_persona_file_is_not_readable_by_unrelated_user(
        self, client: TestClient
    ) -> None:
        response = _get(client, self.other, str(self.persona_file.id))
        _assert_denied(response, self.persona_bytes)
        response = _get(client, self.other, self.persona_blob)
        _assert_denied(response, self.persona_bytes)
        _assert_exact(
            _get(client, self.owner, str(self.persona_file.id)), self.persona_bytes
        )

    def test_public_persona_file_is_readable_by_another_basic_user(
        self, client: TestClient
    ) -> None:
        _assert_exact(
            _get(client, self.other, str(self.public_persona_file.id)),
            self.public_persona_bytes,
        )
        _assert_exact(
            _get(client, self.other, self.public_persona_blob),
            self.public_persona_bytes,
        )

    def test_connector_image_is_not_readable_without_a_chat_or_user_file_grant(
        self, client: TestClient
    ) -> None:
        # owner.email is on the source document, but that link is not enough:
        # the image id is not attached to a chat or user file the caller can
        # access. Both the document owner and a stranger are denied.
        for user in (self.owner, self.other, self.admin):
            response = _get(client, user, self.connector_blob)
            _assert_denied(response, self.connector_bytes)

    def test_internal_uuid_files_are_never_served(self, client: TestClient) -> None:
        for file_id, secret in (
            (self.report_blob, self.report_bytes),
            (self.metadata_blob, self.metadata_bytes),
        ):
            for user in (self.owner, self.other, self.admin):
                response = _get(client, user, file_id)
                _assert_denied(response, secret)
            stored = get_default_file_store().read_file(file_id).read()
            assert stored == secret

    def test_user_file_pointing_at_checkpoint_does_not_stream_checkpoint(
        self, client: TestClient
    ) -> None:
        response = _get(client, self.owner, str(self.checkpoint_pointer.id))
        _assert_denied(response, self.checkpoint_bytes)
        stored = get_default_file_store().read_file(self.checkpoint_id).read()
        assert stored == self.checkpoint_bytes

    def test_predictable_internal_ids_do_not_return_bytes(
        self, client: TestClient
    ) -> None:
        for file_id, secret in (
            (self.checkpoint_id, self.checkpoint_bytes),
            (self.batch_id, self.batch_bytes),
            (self.snapshot_id, self.snapshot_bytes),
            (self.plaintext_id, self.plaintext_bytes),
            (self.history_id, self.history_bytes),
            (self.attachment_id, self.attachment_bytes),
        ):
            for user in (self.owner, self.other, self.admin):
                response = _get(client, user, file_id)
                _assert_not_served(response, secret)

    def test_relookup_of_user_file_cannot_switch_the_blob(
        self, client: TestClient
    ) -> None:
        """If authorization re-reads user_file.file_id before the read, the
        bytes streamed must still be the blob that was authorized.

        The row is updated to the other user's blob immediately after the
        first user_file SELECT in the request.
        """

        engine = get_sqlalchemy_engine()
        target_id = str(self.owner_file.id)
        switched = {"done": False}

        def _swap(*args: object) -> None:
            statement = args[2]
            parameters = args[3]
            if switched["done"] or not isinstance(statement, str):
                return
            if "user_file" not in statement.lower():
                return
            if "select" not in statement.lower():
                return
            raw_params = parameters if isinstance(parameters, dict) else {}
            if target_id not in {str(value) for value in raw_params.values()}:
                return
            switched["done"] = True
            with engine.begin() as write_conn:
                write_conn.exec_driver_sql(
                    "UPDATE user_file SET file_id = %(file_id)s WHERE id = %(id)s",
                    {"file_id": self.other_blob, "id": target_id},
                )

        event.listen(engine, "after_cursor_execute", _swap)
        try:
            response = _get(client, self.owner, target_id)
        finally:
            event.remove(engine, "after_cursor_execute", _swap)
            self.db.rollback()
            self.owner_file.file_id = self.owner_blob
            self.db.add(self.owner_file)
            self.db.commit()

        assert self.other_bytes not in response.content
        if response.status_code == 200:
            assert response.content == self.owner_bytes
        else:
            _assert_denied(response, self.other_bytes)
