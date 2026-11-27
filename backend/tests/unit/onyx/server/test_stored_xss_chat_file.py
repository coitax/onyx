"""Stored XSS via the content type a client supplies for a chat file.

These tests exercise the real upload and download handlers. They fail while
`/chat/file/{id}` renders a caller-supplied HTML or SVG body inline, and they
pass once that response is no longer an active document (safe media type,
attachment disposition, or a script-blocking content security policy) or the
upload is refused without leaving a fetchable copy.
"""

from __future__ import annotations

import datetime
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy.sql.elements import ClauseElement

from onyx.auth.schemas import UserRole
from onyx.auth.users import current_user
from onyx.configs.constants import FileOrigin
from onyx.db.engine.sql_engine import get_session
from onyx.db.models import UserFile
from onyx.key_value_store.interface import KvKeyNotFoundError
from onyx.server.documents.connector import router as connector_router
from onyx.server.features.persona.api import admin_router as admin_persona_router
from onyx.file_processing import extract_file_text
from onyx.server.features.projects import projects_file_utils
from onyx.server.features.projects.api import router as projects_router
from onyx.server.query_and_chat import chat_backend
from onyx.server.query_and_chat.chat_backend import router as chat_router
from onyx.server.settings import store as settings_store

HTML_PAYLOAD = (
    b"<!doctype html><html><body><p>notes</p>"
    b"<script>window.__onyx_xss=1</script></body></html>"
)
SVG_PAYLOAD = (
    b'<?xml version="1.0"?>'
    b'<svg xmlns="http://www.w3.org/2000/svg">'
    b"<script>window.__onyx_xss=1</script></svg>"
)

# Media types that execute markup when a browser navigates to them directly.
_ACTIVE_MEDIA_TYPES = frozenset(
    {
        "text/html",
        "application/xhtml+xml",
        "image/svg+xml",
        "text/xml",
        "application/xml",
    }
)


class _CharTokenizer:
    """Counts characters. The upload gate only needs a small count for these payloads."""

    def encode(self, string: str) -> list[int]:
        return [ord(char) for char in string]

    def tokenize(self, string: str) -> list[str]:
        return list(string)

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


@dataclass
class _StoredFile:
    file_id: str
    data: bytes
    file_type: str


class _MemoryFileStore:
    """Persists bytes and file_type verbatim, matching the file stores.

    PostgresBackedFileStore.save_file and S3BackedFileStore.save_file both
    store the caller-supplied file_type string and the raw bytes, and
    read_file / read_file_record return them unchanged.
    """

    def __init__(self) -> None:
        self.files: dict[str, _StoredFile] = {}

    def save_file(
        self,
        content: Any,
        display_name: str | None,
        file_origin: FileOrigin,
        file_type: str,
        file_metadata: dict[str, Any] | None = None,
        file_id: str | None = None,
        db_session: Any = None,
    ) -> str:
        del display_name, file_origin, file_metadata, db_session
        if file_id is None:
            file_id = str(uuid.uuid4())
        if hasattr(content, "read"):
            raw = content.read()
        else:
            raw = content
        if isinstance(raw, str):
            data = raw.encode("utf-8")
        else:
            data = bytes(raw)
        self.files[file_id] = _StoredFile(
            file_id=file_id, data=data, file_type=file_type
        )
        return file_id

    def read_file(
        self,
        file_id: str,
        mode: str | None = None,
        use_tempfile: bool = False,
        db_session: Any = None,
    ) -> BytesIO:
        del mode, use_tempfile, db_session
        return BytesIO(self.files[file_id].data)

    def read_file_record(self, file_id: str, db_session: Any = None) -> Any:
        del db_session
        stored = self.files[file_id]

        class _Record:
            file_type: str

        record = _Record()
        record.file_type = stored.file_type
        return record


def _clause_matches(row: object, clause: ClauseElement) -> bool:
    clauses = getattr(clause, "clauses", None)
    left = getattr(clause, "left", None)
    if clauses is not None and left is None:
        return all(_clause_matches(row, child) for child in clauses)
    if left is None:
        return False
    key = getattr(left, "key", None)
    if key is None:
        return False
    right = getattr(clause, "right", None)
    expected = getattr(right, "value", None)
    if expected is None:
        expected = getattr(right, "effective_value", None)
    actual = getattr(row, key)
    return actual == expected or str(actual) == str(expected)


class _MemoryQuery:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def filter(self, *criteria: ClauseElement) -> _MemoryQuery:
        rows = self._rows
        for criterion in criteria:
            rows = [row for row in rows if _clause_matches(row, criterion)]
        return _MemoryQuery(rows)

    def first(self) -> object | None:
        if not self._rows:
            return None
        return self._rows[0]


class _MemorySession:
    """Holds UserFile rows added by create_user_files for the real lookup."""

    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def flush(self) -> None:
        now = datetime.datetime.now(datetime.timezone.utc)
        for obj in self.added:
            if isinstance(obj, UserFile) and obj.created_at is None:
                obj.created_at = now

    def commit(self) -> None:
        self.flush()

    def rollback(self) -> None:
        return None

    def query(self, model: type) -> _MemoryQuery:
        return _MemoryQuery([row for row in self.added if isinstance(row, model)])

    def scalar(self, statement: object = None) -> None:
        del statement
        return None


class _Actor:
    def __init__(
        self, user_id: UUID, role: UserRole, permissions: list[str]
    ) -> None:
        self.id = user_id
        self.role = role
        self.effective_permissions = permissions


class _MissingKv:
    def load(self, key: str) -> None:
        raise KvKeyNotFoundError(key)


class _BrokenCache:
    def get(self, key: str) -> None:
        del key
        raise RuntimeError("cache unavailable")

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        del key, value, ex
        raise RuntimeError("cache unavailable")


@dataclass
class _Harness:
    client: TestClient
    store: _MemoryFileStore
    actor: dict[str, _Actor]


def _make_actor(role: UserRole, permissions: list[str]) -> _Actor:
    return _Actor(user_id=uuid.uuid4(), role=role, permissions=permissions)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Harness]:
    store = _MemoryFileStore()
    session = _MemorySession()
    actor: dict[str, _Actor] = {
        "user": _make_actor(UserRole.BASIC, ["basic"]),
    }

    def _store() -> _MemoryFileStore:
        return store

    monkeypatch.setattr(
        "onyx.server.documents.connector.get_default_file_store", _store
    )
    monkeypatch.setattr(
        "onyx.server.features.persona.api.get_default_file_store", _store
    )
    monkeypatch.setattr(chat_backend, "get_default_file_store", _store)
    monkeypatch.setattr(
        projects_file_utils, "fetch_default_llm_model", lambda db_session: None
    )
    monkeypatch.setattr(
        projects_file_utils,
        "get_tokenizer",
        lambda model_name, provider_type: _CharTokenizer(),
    )
    monkeypatch.setattr(settings_store, "get_kv_store", lambda: _MissingKv())
    monkeypatch.setattr(settings_store, "get_cache_backend", lambda: _BrokenCache())
    # Default installs have no Unstructured key. The real lookup falls through
    # to Postgres after a cache miss; with the key absent it returns None and
    # extract_file_text uses its built-in text path.
    monkeypatch.setattr(extract_file_text, "get_unstructured_api_key", lambda: None)
    monkeypatch.setattr(
        "onyx.background.celery.versioned_apps.client.app.send_task",
        lambda *args, **kwargs: type("Task", (), {"id": "unused"})(),
    )

    app = FastAPI()
    app.include_router(projects_router)
    app.include_router(chat_router)
    app.include_router(admin_persona_router)
    app.include_router(connector_router)

    def _user() -> _Actor:
        return actor["user"]

    def _session() -> Iterator[_MemorySession]:
        yield session

    app.dependency_overrides[current_user] = _user
    app.dependency_overrides[get_session] = _session

    with TestClient(app) as client:
        yield _Harness(client=client, store=store, actor=actor)


def _csp_blocks_scripts(csp: str) -> bool:
    directives: dict[str, str] = {}
    for part in csp.split(";"):
        piece = part.strip().lower()
        if not piece:
            continue
        bits = piece.split()
        directives[bits[0]] = " ".join(bits[1:])
    if "sandbox" in directives and "allow-scripts" not in directives["sandbox"]:
        return True
    script_src = directives.get("script-src", directives.get("default-src"))
    if script_src is None:
        return False
    tokens = script_src.split()
    return "'none'" in tokens and "'unsafe-inline'" not in tokens


def _is_active_document(response: Response, payload: bytes) -> bool:
    if response.status_code != 200 or payload not in response.content:
        return False
    media_type = (
        response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    disposition = response.headers.get("content-disposition", "").lower()
    if "attachment" in disposition:
        return False
    if _csp_blocks_scripts(response.headers.get("content-security-policy", "")):
        return False
    return media_type in _ACTIVE_MEDIA_TYPES


def _collected_ids(upload: Response, store: _MemoryFileStore, payload: bytes) -> set[str]:
    found: set[str] = set()
    if upload.status_code == 200:
        body = upload.json()
        for user_file in body.get("user_files", []):
            found.add(str(user_file["id"]))
            found.add(str(user_file["file_id"]))
        for file_path in body.get("file_paths", []):
            found.add(str(file_path))
        file_id = body.get("file_id")
        if isinstance(file_id, str):
            found.add(file_id)
    for stored in store.files.values():
        if payload in stored.data:
            found.add(stored.file_id)
    return found


def _assert_not_served_active(
    harness: _Harness, upload: Response, payload: bytes
) -> None:
    targets = _collected_ids(upload, harness.store, payload)
    if upload.status_code < 400 and not targets:
        pytest.fail(
            f"upload returned {upload.status_code} without a file id: {upload.text}"
        )
    if upload.status_code >= 400 and not any(
        payload in stored.data for stored in harness.store.files.values()
    ):
        return
    for file_id in targets:
        downloaded = harness.client.get(f"/chat/file/{file_id}")
        if _is_active_document(downloaded, payload):
            stored_types = sorted({item.file_type for item in harness.store.files.values()})
            pytest.fail(
                "chat file download still renders caller-controlled markup inline: "
                f"id={file_id} status={downloaded.status_code} "
                f"content-type={downloaded.headers.get('content-type')!r} "
                f"content-disposition={downloaded.headers.get('content-disposition')!r} "
                f"content-security-policy={downloaded.headers.get('content-security-policy')!r} "
                f"x-content-type-options={downloaded.headers.get('x-content-type-options')!r} "
                f"stored_types={stored_types} body={downloaded.content[:180]!r}"
            )


def test_basic_user_text_html_upload_is_not_active_for_an_admin(
    harness: _Harness,
) -> None:
    attacker = harness.actor["user"]
    assert attacker.role == UserRole.BASIC
    upload = harness.client.post(
        "/user/projects/file/upload",
        files={"files": ("notes.txt", HTML_PAYLOAD, "text/html")},
    )
    harness.actor["user"] = _make_actor(UserRole.ADMIN, ["admin"])
    _assert_not_served_active(harness, upload, HTML_PAYLOAD)


def test_text_html_charset_parameter_is_not_active_for_another_user(
    harness: _Harness,
) -> None:
    upload = harness.client.post(
        "/user/projects/file/upload",
        files={"files": ("notes.txt", HTML_PAYLOAD, "text/html; charset=utf-8")},
    )
    harness.actor["user"] = _make_actor(UserRole.BASIC, ["basic"])
    _assert_not_served_active(harness, upload, HTML_PAYLOAD)


def test_html_filename_with_script_is_not_active_for_another_user(
    harness: _Harness,
) -> None:
    upload = harness.client.post(
        "/user/projects/file/upload",
        files={"files": ("notes.html", HTML_PAYLOAD, "text/html")},
    )
    harness.actor["user"] = _make_actor(UserRole.ADMIN, ["admin"])
    _assert_not_served_active(harness, upload, HTML_PAYLOAD)


def test_persona_image_upload_html_is_not_active_for_another_user(
    harness: _Harness,
) -> None:
    upload = harness.client.post(
        "/admin/persona/upload-image",
        files={"file": ("avatar.html", HTML_PAYLOAD, "text/html")},
    )
    harness.actor["user"] = _make_actor(UserRole.ADMIN, ["admin"])
    _assert_not_served_active(harness, upload, HTML_PAYLOAD)


def test_persona_svg_upload_is_not_active_for_another_user(
    harness: _Harness,
) -> None:
    upload = harness.client.post(
        "/admin/persona/upload-image",
        files={"file": ("avatar.svg", SVG_PAYLOAD, "image/svg+xml")},
    )
    harness.actor["user"] = _make_actor(UserRole.ADMIN, ["admin"])
    _assert_not_served_active(harness, upload, SVG_PAYLOAD)


def test_curator_connector_upload_html_is_not_active_for_an_admin(
    harness: _Harness,
) -> None:
    harness.actor["user"] = _make_actor(UserRole.CURATOR, ["basic"])
    upload = harness.client.post(
        "/manage/admin/connector/file/upload",
        files={"files": ("notes.txt", HTML_PAYLOAD, "text/html")},
    )
    harness.actor["user"] = _make_actor(UserRole.ADMIN, ["admin"])
    _assert_not_served_active(harness, upload, HTML_PAYLOAD)
