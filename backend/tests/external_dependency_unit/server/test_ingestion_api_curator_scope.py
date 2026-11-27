"""Curators must not read or write connectors outside the groups they curate.

These tests exercise the ingestion API the product mounts at ``/onyx-api``.
They fail while a curator or global curator can list, overwrite, or delete
documents on another group's connector or on the organization default
connector. They pass once those calls are refused before any document,
connector link, or index record changes.

The API process enables Enterprise Edition on startup
(``set_is_ee_based_on_env_variable`` in ``onyx/main.py``). These tests do the
same so document access recorded in the index includes user groups.

``ENABLE_OPENSEARCH_INDEXING_FOR_ONYX`` defaults to true and adds a second
index client. These tests turn that product flag off so deletion finishes on
the Vespa index and in Postgres. Ingestion was also observed with the flag
left on: the Vespa write is committed before the OpenSearch client errors.

Run against a disposable database (this resets that database's public schema)
with Redis and the embedding tokenizer available::

    POSTGRES_DB=onyx_ingestion_scope_test pytest \\
        tests/external_dependency_unit/server/test_ingestion_api_curator_scope.py -q
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.configs.constants import DEFAULT_CC_PAIR_ID
from onyx.configs.constants import DocumentSource
from onyx.connectors.models import InputType
from onyx.db.connector import create_initial_default_connector
from onyx.db.connector_credential_pair import associate_default_cc_pair
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.engine.sql_engine import SqlEngine
from onyx.db.enums import AccessType
from onyx.db.enums import AccountType
from onyx.db.enums import ConnectorCredentialPairStatus
from onyx.db.enums import EmbeddingPrecision
from onyx.db.enums import IndexModelStatus
from onyx.db.enums import ProcessingMode
from onyx.db.enums import SwitchoverType
from onyx.db.models import Base
from onyx.db.models import CloudEmbeddingProvider
from onyx.db.models import Connector
from onyx.db.models import ConnectorCredentialPair
from onyx.db.models import Credential
from onyx.db.models import Document
from onyx.db.models import DocumentByConnectorCredentialPair
from onyx.db.models import SearchSettings
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.db.models import UserGroup
from onyx.db.models import UserGroup__ConnectorCredentialPair
from onyx.db.pat import create_pat
from onyx.document_index.vespa_constants import CONTENT
from onyx.document_index.vespa_constants import CONTENT_SUMMARY
from onyx.document_index.vespa_constants import DOCUMENT_ID
from onyx.kg.models import KGStage
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.server.onyx_api.ingestion import router as onyx_api_router
from onyx.utils.variable_functionality import fetch_versioned_implementation
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
from shared_configs.enums import EmbeddingProvider
from shared_configs.model_server_models import Embedding

pytestmark = pytest.mark.usefixtures("world")

DENIED = frozenset({400, 401, 403, 404})
VICTIM_GROUP = "victim-private-group"
OWN_GROUP = "curator-own-group"
EMBEDDING_DIM = 8

_INDEX_LOCK = threading.Lock()
_INDEX_RECORDS: list[dict[str, Any]] = []


class _VespaHandler(BaseHTTPRequestHandler):
    """Records the document fields the real Vespa indexing client posts.

    ``_index_vespa_chunk`` posts ``{"fields": {...}}`` and treats 404 on GET
    as "chunk does not exist". This server speaks that contract and keeps the
    posted fields so a test can see the text and ACL that would be searchable.
    """

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(404)
        self.end_headers()

    def do_DELETE(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode())
        except json.JSONDecodeError:
            payload = {"raw": raw.decode(errors="replace")}
        fields = payload.get("fields", payload)
        with _INDEX_LOCK:
            _INDEX_RECORDS.append(fields if isinstance(fields, dict) else {"raw": fields})
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def _fake_encode(
    self: EmbeddingModel,
    texts: list[str],
    text_type: Any,
    large_chunks_present: bool = False,
    **kwargs: Any,
) -> list[Embedding]:
    """Stand in for the external embedding model server.

    The chunk text is already fixed by the product chunker before this call.
    Vespa stores that text, not these vectors.
    """
    del self, text_type, large_chunks_present, kwargs
    return [[0.0] * EMBEDDING_DIM for _ in texts]


@dataclass
class World:
    curator_token: str
    outsider_token: str
    global_curator_token: str
    basic_token: str
    admin_token: str
    own_cc_pair_id: int
    victim_cc_pair_id: int
    default_cc_pair_id: int
    victim_doc_id: str
    victim_title: str
    victim_link: str
    reuse_doc_id: str
    delete_doc_id: str


def _bootstrap_schema() -> None:
    cycle = {"document", "hierarchy_node", "mcp_connection_config", "mcp_server"}
    for table in Base.metadata.tables.values():
        for fk in table.foreign_key_constraints:
            referred = {element.column.table.name for element in fk.elements}
            if table.name in cycle or referred & cycle:
                fk.use_alter = True
    engine = SqlEngine.get_engine()
    for table in Base.metadata.sorted_tables:
        for _ in range(4):
            try:
                with engine.begin() as conn:
                    table.create(conn, checkfirst=True)
                break
            except ProgrammingError as exc:
                # Two models reuse one index or unique-constraint name, so
                # CREATE TABLE aborts. Drop that name from this table and
                # retry. The delete path needs the knowledge-graph tables.
                message = str(getattr(exc, "orig", exc))
                removed = False
                for index in list(table.indexes):
                    if index.name and index.name in message:
                        table.indexes.remove(index)
                        removed = True
                for constraint in list(table.constraints):
                    name = getattr(constraint, "name", None)
                    if name and name in message:
                        table.constraints.discard(constraint)
                        removed = True
                if not removed:
                    break


def _reset_rows() -> None:
    engine = SqlEngine.get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE "
                "document_by_connector_credential_pair, "
                "user_group__connector_credential_pair, "
                "document, "
                "connector_credential_pair, "
                "connector, "
                "credential, "
                "personal_access_token, "
                "user__user_group, "
                "permission_grant, "
                "user_group, "
                "search_settings, "
                "embedding_provider, "
                '"user" '
                "RESTART IDENTITY CASCADE"
            )
        )
        conn.execute(
            text("ALTER SEQUENCE connector_credential_pair_id_seq RESTART WITH 1")
        )
        # Alembic adds document.is_public as nullable (61ff3651add4). The ORM
        # insert used by the ingestion pipeline writes NULL when external_access
        # is unset, which the shipped column accepts.
        conn.execute(text("ALTER TABLE document ALTER COLUMN is_public DROP NOT NULL"))


def _add_group(
    db_session: Session,
    name: str,
    curator: User | None,
    members: list[User],
) -> UserGroup:
    group = UserGroup(name=name, is_up_to_date=True, is_up_for_deletion=False)
    db_session.add(group)
    db_session.flush()
    curator_ids = {curator.id} if curator is not None else set()
    for member in members:
        db_session.add(
            User__UserGroup(
                user_group_id=group.id,
                user_id=member.id,
                is_curator=member.id in curator_ids,
            )
        )
    return group


def _add_user(db_session: Session, role: UserRole) -> tuple[User, str]:
    from fastapi_users.password import PasswordHelper

    user = User(
        id=uuid4(),
        email=f"{role.value}-{uuid4().hex[:8]}@example.com",
        hashed_password=PasswordHelper().hash("placeholder"),
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role=role,
        account_type=AccountType.STANDARD,
    )
    db_session.add(user)
    db_session.flush()
    _pat, raw_token = create_pat(
        db_session=db_session,
        user_id=user.id,
        name=f"{role.value}-token",
        expiration_days=1,
    )
    return user, raw_token


def _add_private_connector(
    db_session: Session,
    name: str,
    group: UserGroup,
    creator: User,
) -> ConnectorCredentialPair:
    connector = Connector(
        name=name,
        source=DocumentSource.FILE,
        input_type=InputType.LOAD_STATE,
        connector_specific_config={"file_locations": [], "file_names": []},
        refresh_freq=None,
        prune_freq=None,
    )
    credential = Credential(
        name=name,
        source=DocumentSource.FILE,
        credential_json=None,
        user_id=creator.id,
        admin_public=True,
        curator_public=False,
    )
    db_session.add_all([connector, credential])
    db_session.flush()
    cc_pair = ConnectorCredentialPair(
        name=name,
        status=ConnectorCredentialPairStatus.ACTIVE,
        connector_id=connector.id,
        credential_id=credential.id,
        access_type=AccessType.PRIVATE,
        creator_id=creator.id,
        processing_mode=ProcessingMode.REGULAR,
    )
    db_session.add(cc_pair)
    db_session.flush()
    db_session.add(
        UserGroup__ConnectorCredentialPair(
            user_group_id=group.id,
            cc_pair_id=cc_pair.id,
            is_current=True,
        )
    )
    return cc_pair


def _add_document(
    db_session: Session,
    cc_pair: ConnectorCredentialPair,
    document_id: str,
    title: str,
    link: str,
) -> None:
    db_session.add(
        Document(
            id=document_id,
            from_ingestion_api=True,
            semantic_id=title,
            link=link,
            hidden=False,
            boost=0,
            chunk_count=1,
            is_public=False,
            kg_stage=KGStage.NOT_STARTED,
        )
    )
    db_session.add(
        DocumentByConnectorCredentialPair(
            id=document_id,
            connector_id=cc_pair.connector_id,
            credential_id=cc_pair.credential_id,
            has_been_indexed=True,
        )
    )


def _build_world(db_session: Session) -> World:
    db_session.add(
        CloudEmbeddingProvider(
            provider_type=EmbeddingProvider.OPENAI,
            api_key=None,
        )
    )
    db_session.add(
        SearchSettings(
            model_name="text-embedding-3-small",
            model_dim=EMBEDDING_DIM,
            normalize=True,
            status=IndexModelStatus.PRESENT,
            index_name="danswer_chunk",
            provider_type=EmbeddingProvider.OPENAI,
            switchover_type=SwitchoverType.REINDEX,
            embedding_precision=EmbeddingPrecision.FLOAT,
            multipass_indexing=False,
            enable_contextual_rag=False,
            multilingual_expansion=[],
        )
    )
    db_session.flush()

    admin, admin_token = _add_user(db_session, UserRole.ADMIN)
    curator, curator_token = _add_user(db_session, UserRole.CURATOR)
    outsider, outsider_token = _add_user(db_session, UserRole.CURATOR)
    global_curator, global_token = _add_user(db_session, UserRole.GLOBAL_CURATOR)
    _basic, basic_token = _add_user(db_session, UserRole.BASIC)

    # The default connector is the first credential pair, so its id matches
    # DEFAULT_CC_PAIR_ID. Requests that omit cc_pair_id select this pair.
    db_session.add(
        Credential(
            id=0,
            name="default-credential",
            source=DocumentSource.INGESTION_API,
            credential_json=None,
            admin_public=True,
            curator_public=False,
        )
    )
    db_session.commit()
    create_initial_default_connector(db_session)
    associate_default_cc_pair(db_session)
    default_pair = (
        db_session.query(ConnectorCredentialPair)
        .filter(ConnectorCredentialPair.name == "DefaultCCPair")
        .one()
    )
    assert default_pair.id == DEFAULT_CC_PAIR_ID

    own_group = _add_group(db_session, OWN_GROUP, curator, [curator])
    victim_group = _add_group(
        db_session,
        VICTIM_GROUP,
        None,
        [outsider],
    )
    # The outsider is a curator of their own group, and only a member of the
    # victim group. Editable connector access requires the curator flag.
    other_group = _add_group(
        db_session, "outsider-curated-group", outsider, [outsider]
    )
    _add_private_connector(db_session, "outsider-connector", other_group, outsider)

    own_pair = _add_private_connector(
        db_session, "curator-connector", own_group, curator
    )
    victim_pair = _add_private_connector(
        db_session, "victim-connector", victim_group, admin
    )

    victim_doc_id = "victimonlydoc"
    _add_document(
        db_session,
        victim_pair,
        victim_doc_id,
        title="Victim Only Quarterly Plan",
        link="https://example.internal/victim-only-plan",
    )
    reuse_doc_id = "victimreusedoc"
    _add_document(
        db_session,
        victim_pair,
        reuse_doc_id,
        title="Victim Reused Identifier",
        link="https://example.internal/victim-reused",
    )
    delete_doc_id = "victimdeletedoc"
    _add_document(
        db_session,
        victim_pair,
        delete_doc_id,
        title="Victim Delete Target",
        link="https://example.internal/victim-delete",
    )
    db_session.commit()

    return World(
        curator_token=curator_token,
        outsider_token=outsider_token,
        global_curator_token=global_token,
        basic_token=basic_token,
        admin_token=admin_token,
        own_cc_pair_id=own_pair.id,
        victim_cc_pair_id=victim_pair.id,
        default_cc_pair_id=default_pair.id,
        victim_doc_id=victim_doc_id,
        victim_title="Victim Only Quarterly Plan",
        victim_link="https://example.internal/victim-only-plan",
        reuse_doc_id=reuse_doc_id,
        delete_doc_id=delete_doc_id,
    )


@pytest.fixture(scope="module")
def vespa_server() -> Iterator[None]:
    from onyx.configs.app_configs import VESPA_HOST
    from onyx.configs.app_configs import VESPA_PORT

    server = ThreadingHTTPServer((VESPA_HOST, int(VESPA_PORT)), _VespaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield
    server.shutdown()


@pytest.fixture(scope="module")
def world(vespa_server: None) -> Generator[World, None, None]:
    del vespa_server
    set_is_ee_based_on_env_variable()
    fetch_versioned_implementation.cache_clear()
    # get_all_document_indices appends an OpenSearch client when this flag is
    # true. Nothing listens on that port here, and delete_ingestion_doc does
    # not catch the connection error, so the Postgres delete never runs.
    # Turning the flag off is the product's own switch. The missing scope
    # check is unchanged.
    import onyx.configs.app_configs as app_configs
    import onyx.document_index.factory as document_index_factory

    app_configs.ENABLE_OPENSEARCH_INDEXING_FOR_ONYX = False
    document_index_factory.ENABLE_OPENSEARCH_INDEXING_FOR_ONYX = False
    # The external embedding API is not part of authorization. Chunk text is
    # chosen before encode() and is what the Vespa client posts.
    EmbeddingModel.encode = _fake_encode  # type: ignore[method-assign]

    SqlEngine.init_engine(pool_size=4, max_overflow=2)
    _bootstrap_schema()
    _reset_rows()
    with get_session_with_current_tenant() as db_session:
        built = _build_world(db_session)
    yield built


@pytest.fixture(scope="module")
def client(world: World) -> Iterator[TestClient]:
    del world
    app = FastAPI()
    app.include_router(onyx_api_router)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _clear_index() -> Iterator[None]:
    with _INDEX_LOCK:
        _INDEX_RECORDS.clear()
    yield


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _payload(
    document_id: str,
    text: str,
    title: str,
    cc_pair_id: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "document": {
            "id": document_id,
            "sections": [{"text": text, "link": f"https://example.internal/{document_id}"}],
            "source": DocumentSource.FILE.value,
            "semantic_identifier": title,
            "metadata": {},
        }
    }
    if cc_pair_id is not None:
        body["cc_pair_id"] = cc_pair_id
    return body


def _document_state(document_id: str) -> dict[str, Any] | None:
    with get_session_with_current_tenant() as db_session:
        document = db_session.get(Document, document_id)
        if document is None:
            return None
        pairs = (
            db_session.query(DocumentByConnectorCredentialPair)
            .filter(DocumentByConnectorCredentialPair.id == document_id)
            .all()
        )
        return {
            "semantic_id": document.semantic_id,
            "link": document.link,
            "from_ingestion_api": document.from_ingestion_api,
            "pairs": sorted(
                (pair.connector_id, pair.credential_id, pair.has_been_indexed)
                for pair in pairs
            ),
        }


def _index_hits(needle: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    with _INDEX_LOCK:
        records = list(_INDEX_RECORDS)
    for fields in records:
        blob = json.dumps(fields)
        if needle in blob:
            hits.append(
                {
                    "document_id": fields.get(DOCUMENT_ID),
                    "content": fields.get(CONTENT),
                    "content_summary": fields.get(CONTENT_SUMMARY),
                    "access_control_list": fields.get("access_control_list"),
                }
            )
    return hits


def _assert_refused(
    response: Any,
    document_id: str,
    before: dict[str, Any] | None,
    planted_text: str,
) -> None:
    problems: list[str] = []
    if response.status_code not in DENIED:
        problems.append(f"status={response.status_code} body={response.text[:800]}")
    after = _document_state(document_id)
    if after != before:
        problems.append(f"document changed from {before} to {after}")
    hits = _index_hits(planted_text)
    if hits:
        problems.append(f"index received planted text: {hits}")
    assert not problems, "; ".join(problems)


def test_basic_user_cannot_call_ingestion_api(client: TestClient, world: World) -> None:
    headers = _headers(world.basic_token)
    listed = client.get(
        f"/onyx-api/connector-docs/{world.victim_cc_pair_id}", headers=headers
    )
    assert listed.status_code in DENIED
    assert world.victim_title not in listed.text

    ingested = client.post(
        "/onyx-api/ingestion",
        headers=headers,
        json=_payload(
            world.victim_doc_id,
            "basic-user-should-not-write",
            "Basic User Title",
            world.victim_cc_pair_id,
        ),
    )
    assert ingested.status_code in DENIED
    state = _document_state(world.victim_doc_id)
    assert state is not None
    assert state["semantic_id"] == world.victim_title


def test_curator_can_list_and_ingest_own_connector(
    client: TestClient, world: World
) -> None:
    """The allowed path must keep working, and it must reach the index."""
    headers = _headers(world.curator_token)
    listed = client.get(
        f"/onyx-api/connector-docs/{world.own_cc_pair_id}", headers=headers
    )
    assert listed.status_code == 200, listed.text

    document_id = "curatorowndoc"
    planted = "curator-own-body-visible"
    response = client.post(
        "/onyx-api/ingestion",
        headers=headers,
        json=_payload(document_id, planted, "Curator Own Note", world.own_cc_pair_id),
    )
    assert response.status_code == 200, response.text
    state = _document_state(document_id)
    assert state is not None
    assert state["semantic_id"] == "Curator Own Note"
    assert state["pairs"], "own document was not attached to the curator's connector"
    hits = _index_hits(planted)
    assert hits, "own ingestion did not reach the document index"
    assert any(planted in (hit["content"] or "") or planted in (hit["content_summary"] or "") for hit in hits)


def test_admin_can_list_another_groups_connector(
    client: TestClient, world: World
) -> None:
    response = client.get(
        f"/onyx-api/connector-docs/{world.victim_cc_pair_id}",
        headers=_headers(world.admin_token),
    )
    assert response.status_code == 200, response.text
    assert world.victim_title in response.text
    assert world.victim_link in response.text


def test_curator_cannot_list_another_groups_connector_documents(
    client: TestClient, world: World
) -> None:
    response = client.get(
        f"/onyx-api/connector-docs/{world.victim_cc_pair_id}",
        headers=_headers(world.curator_token),
    )
    assert response.status_code in DENIED, response.text
    assert world.victim_doc_id not in response.text
    assert world.victim_title not in response.text
    assert world.victim_link not in response.text


def test_global_curator_outside_the_group_cannot_list_connector_documents(
    client: TestClient, world: World
) -> None:
    response = client.get(
        f"/onyx-api/connector-docs/{world.victim_cc_pair_id}",
        headers=_headers(world.global_curator_token),
    )
    assert response.status_code in DENIED, response.text
    assert world.victim_title not in response.text
    assert world.victim_link not in response.text


def test_member_who_is_not_a_curator_cannot_list_connector_documents(
    client: TestClient, world: World
) -> None:
    response = client.get(
        f"/onyx-api/connector-docs/{world.victim_cc_pair_id}",
        headers=_headers(world.outsider_token),
    )
    assert response.status_code in DENIED, response.text
    assert world.victim_title not in response.text


def test_curator_cannot_list_ingestion_documents_on_other_connectors(
    client: TestClient, world: World
) -> None:
    headers = _headers(world.curator_token)
    own_id = "curatorlistedoc"
    own = client.post(
        "/onyx-api/ingestion",
        headers=headers,
        json=_payload(own_id, "curator-list-own-body", "Curator Listed Note", world.own_cc_pair_id),
    )
    assert own.status_code == 200, own.text

    listed = client.get("/onyx-api/ingestion", headers=headers)
    assert listed.status_code == 200, listed.text
    body = listed.text
    assert own_id in body
    assert world.victim_doc_id not in body
    assert world.victim_title not in body
    assert world.victim_link not in body
    assert world.reuse_doc_id not in body
    assert world.delete_doc_id not in body


def test_curator_cannot_ingest_into_another_groups_connector(
    client: TestClient, world: World
) -> None:
    planted = "attacker-planted-body-on-victim-connector"
    before = _document_state(world.victim_doc_id)
    response = client.post(
        "/onyx-api/ingestion",
        headers=_headers(world.curator_token),
        json=_payload(
            world.victim_doc_id,
            planted,
            "Attacker Replacement Title",
            world.victim_cc_pair_id,
        ),
    )
    _assert_refused(response, world.victim_doc_id, before, planted)
    # A successful write would attach this text to the victim group's ACL.
    for hit in _index_hits(planted):
        assert f"group:{VICTIM_GROUP}" not in json.dumps(hit["access_control_list"])


def test_global_curator_cannot_ingest_into_another_groups_connector(
    client: TestClient, world: World
) -> None:
    planted = "global-curator-planted-body"
    document_id = "globalcuratorplant"
    before = _document_state(document_id)
    response = client.post(
        "/onyx-api/ingestion",
        headers=_headers(world.global_curator_token),
        json=_payload(
            document_id,
            planted,
            "Global Curator Planted Title",
            world.victim_cc_pair_id,
        ),
    )
    _assert_refused(response, document_id, before, planted)


def test_curator_cannot_ingest_into_the_default_connector(
    client: TestClient, world: World
) -> None:
    planted = "attacker-default-connector-body"
    document_id = "defaultconnectorplant"
    before = _document_state(document_id)
    # Omitting cc_pair_id selects the organization default connector.
    response = client.post(
        "/onyx-api/ingestion",
        headers=_headers(world.curator_token),
        json=_payload(document_id, planted, "Default Connector Planted Title", None),
    )
    _assert_refused(response, document_id, before, planted)
    assert world.default_cc_pair_id == DEFAULT_CC_PAIR_ID


def test_admin_can_ingest_into_the_default_connector(
    client: TestClient, world: World
) -> None:
    document_id = "admindefaultdoc"
    planted = "admin-default-connector-body"
    response = client.post(
        "/onyx-api/ingestion",
        headers=_headers(world.admin_token),
        json=_payload(document_id, planted, "Admin Default Note", None),
    )
    assert response.status_code == 200, response.text
    state = _document_state(document_id)
    assert state is not None
    assert _index_hits(planted)


def test_curator_cannot_overwrite_a_document_id_from_another_connector(
    client: TestClient, world: World
) -> None:
    """Authorizing the caller's connector must not write a document id owned elsewhere."""
    planted = "attacker-reused-foreign-document-id"
    before = _document_state(world.reuse_doc_id)
    response = client.post(
        "/onyx-api/ingestion",
        headers=_headers(world.curator_token),
        json=_payload(
            world.reuse_doc_id,
            planted,
            "Reused Id Replacement Title",
            world.own_cc_pair_id,
        ),
    )
    _assert_refused(response, world.reuse_doc_id, before, planted)


def test_member_who_is_not_a_curator_cannot_ingest(
    client: TestClient, world: World
) -> None:
    planted = "member-not-curator-planted-body"
    before = _document_state(world.victim_doc_id)
    response = client.post(
        "/onyx-api/ingestion",
        headers=_headers(world.outsider_token),
        json=_payload(
            world.victim_doc_id,
            planted,
            "Member Replacement Title",
            world.victim_cc_pair_id,
        ),
    )
    _assert_refused(response, world.victim_doc_id, before, planted)


def test_curator_cannot_delete_another_groups_ingestion_document(
    client: TestClient, world: World
) -> None:
    before = _document_state(world.delete_doc_id)
    assert before is not None
    response = client.delete(
        f"/onyx-api/ingestion/{world.delete_doc_id}",
        headers=_headers(world.curator_token),
    )
    problems: list[str] = []
    if response.status_code not in DENIED:
        problems.append(f"status={response.status_code} body={response.text[:800]}")
    after = _document_state(world.delete_doc_id)
    if after != before:
        problems.append(f"document changed from {before} to {after}")
    assert not problems, "; ".join(problems)


def test_curator_can_delete_own_ingestion_document(
    client: TestClient, world: World
) -> None:
    headers = _headers(world.curator_token)
    document_id = "curatordeletowndoc"
    created = client.post(
        "/onyx-api/ingestion",
        headers=headers,
        json=_payload(
            document_id, "curator-delete-own-body", "Curator Delete Own", world.own_cc_pair_id
        ),
    )
    assert created.status_code == 200, created.text
    deleted = client.delete(f"/onyx-api/ingestion/{document_id}", headers=headers)
    assert deleted.status_code < 300, deleted.text
    assert _document_state(document_id) is None
