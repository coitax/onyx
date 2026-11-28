"""Security tests for OAuth config administration.

These tests are expected to fail while a curator can retarget another
principal's OAuth config, or while changing an OAuth URL keeps the stored
client secret and other users' refresh tokens. They pass once both are closed.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import AsyncIterator
from collections.abc import Iterator
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from queue import Queue
from typing import Any
from typing import cast
from unittest.mock import Mock
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from ee.onyx.db.user_group import insert_user_group
from ee.onyx.db.user_group import update_user_curator_relationship
from ee.onyx.server.user_group.models import SetCuratorRequest
from ee.onyx.server.user_group.models import UserGroupCreate
from onyx.auth.schemas import AuthBackend
from onyx.auth.schemas import UserRole
from onyx.auth.users import get_jwt_strategy
from onyx.cache.interface import CacheBackendType
from onyx.chat.emitter import Emitter
from onyx.configs.app_configs import AUTH_BACKEND
from onyx.configs.app_configs import CACHE_BACKEND
from onyx.configs.app_configs import DISABLE_VECTOR_DB
from onyx.configs.constants import FASTAPI_USERS_AUTH_COOKIE_NAME
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.engine.sql_engine import SqlEngine
from onyx.db.enums import EmbeddingPrecision
from onyx.db.enums import SwitchoverType
from onyx.db.models import IndexModelStatus
from onyx.db.models import OAuthConfig
from onyx.db.models import Persona
from onyx.db.models import SearchSettings
from onyx.db.models import Tool
from onyx.db.models import User
from onyx.db.oauth_config import get_oauth_config
from onyx.db.oauth_config import get_user_oauth_token
from onyx.db.users import assign_user_to_default_groups__no_commit
from onyx.llm.interfaces import LLM
from onyx.main import get_application
from onyx.tools.tool_constructor import construct_tools
from tests.external_dependency_unit.conftest import create_test_user


# Obvious placeholders. They are not credentials for any real service.
CLIENT_ID = "placeholderoauthclientid"
CLIENT_SECRET = "placeholderoauthclientsecret"
REFRESH_TOKEN = "placeholderuserrefreshtoken"
ACCESS_TOKEN = "placeholderuseraccesstoken"
AUTH_CODE = "placeholderauthorizationcode"

_OPENAPI_SCHEMA: dict[str, Any] = {
    "openapi": "3.0.0",
    "info": {"title": "Test API", "version": "1.0.0"},
    "servers": [{"url": "https://api.example.com"}],
    "paths": {
        "/test": {
            "get": {
                "operationId": "test_operation",
                "summary": "Test operation",
                "responses": {"200": {"description": "Success"}},
            }
        }
    },
}


class _TokenServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _TokenHandler)
        self.bodies: list[bytes] = []

    def url(self, path: str) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}{path}"


class _TokenHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        cast(_TokenServer, self.server).bodies.append(body)
        payload = json.dumps(
            {
                "access_token": ACCESS_TOKEN,
                "refresh_token": REFRESH_TOKEN,
                "expires_in": 1,
                "token_type": "Bearer",
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ARG002
        return


def _start_token_server() -> tuple[_TokenServer, threading.Thread]:
    server = _TokenServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_token_server(server: _TokenServer, thread: threading.Thread) -> None:
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def _contains(bodies: list[bytes], marker: str) -> bool:
    raw = marker.encode()
    return any(raw in body for body in bodies)


def _cookies_for(user: User) -> dict[str, str]:
    token = asyncio.run(get_jwt_strategy().write_token(user))
    return {FASTAPI_USERS_AUTH_COOKIE_NAME: token}


def _plain_secret(config: OAuthConfig) -> str:
    assert config.client_secret is not None
    return config.client_secret.get_value(apply_mask=False)


def _plain_refresh(session: Session, config_id: int, user_id: UUID) -> str | None:
    token = get_user_oauth_token(config_id, user_id, session)
    if token is None or token.token_data is None:
        return None
    value = token.token_data.get_value(apply_mask=False).get("refresh_token")
    if not isinstance(value, str):
        return None
    return value


def _reload(session: Session, config_id: int) -> OAuthConfig:
    session.expire_all()
    config = get_oauth_config(config_id, session)
    assert config is not None
    return config


@asynccontextmanager
async def _test_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # The routes under test use the real session dependency. Skip the full
    # process startup, which opens a pool of connections and blocks this test.
    SqlEngine.init_engine(pool_size=10, max_overflow=20)
    yield


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    assert DISABLE_VECTOR_DB, "DISABLE_VECTOR_DB must be set before importing the app"
    assert AUTH_BACKEND == AuthBackend.JWT, "AUTH_BACKEND must be jwt"
    assert CACHE_BACKEND == CacheBackendType.POSTGRES, "CACHE_BACKEND must be postgres"
    app: FastAPI = get_application(lifespan_override=_test_lifespan)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module", autouse=True)
def _search_settings(client: TestClient) -> None:
    del client
    with get_session_with_current_tenant() as session:
        existing = session.scalar(
            select(SearchSettings).where(
                SearchSettings.status == IndexModelStatus.PRESENT
            )
        )
        if existing is None:
            session.add(
                SearchSettings(
                    model_name="oauth-config-security-test",
                    model_dim=768,
                    normalize=True,
                    status=IndexModelStatus.PRESENT,
                    index_name="oauth_config_security_test",
                    embedding_precision=EmbeddingPrecision.FLOAT,
                    switchover_type=SwitchoverType.REINDEX,
                    multipass_indexing=False,
                    enable_contextual_rag=False,
                    multilingual_expansion=[],
                )
            )
            session.commit()


def _admin(session: Session) -> User:
    user = create_test_user(session, "oauth_admin", role=UserRole.ADMIN)
    assign_user_to_default_groups__no_commit(session, user, is_admin=True)
    session.commit()
    session.refresh(user)
    return user


def _basic_member(session: Session, prefix: str, role: UserRole) -> User:
    user = create_test_user(session, prefix, role=role)
    assign_user_to_default_groups__no_commit(session, user, is_admin=False)
    session.commit()
    session.refresh(user)
    return user


def _group_curator(session: Session, admin: User) -> User:
    user = create_test_user(session, "oauth_group_curator", role=UserRole.BASIC)
    group = insert_user_group(
        session,
        UserGroupCreate(
            name=f"oauth-curator-group-{uuid.uuid4().hex[:8]}",
            user_ids=[user.id],
            cc_pair_ids=[],
        ),
    )
    update_user_curator_relationship(
        session,
        group.id,
        SetCuratorRequest(user_id=user.id, is_curator=True),
        admin,
    )
    session.refresh(user)
    assert user.role == UserRole.CURATOR
    assert "admin" not in (user.effective_permissions or [])
    return user


def _create_config(
    client: TestClient,
    cookies: dict[str, str],
    *,
    token_url: str,
    authorization_url: str,
) -> int:
    response = client.post(
        "/admin/oauth-config/create",
        json={
            "name": f"oauth-config-{uuid.uuid4().hex}",
            "authorization_url": authorization_url,
            "token_url": token_url,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "scopes": ["read"],
        },
        cookies=cookies,
    )
    assert response.status_code == 200, response.status_code
    payload = response.json()
    assert "client_secret" not in payload
    assert "client_id" not in payload
    assert payload["has_client_credentials"] is True
    assert payload["token_url"] == token_url
    return int(payload["id"])


def _authorize(
    client: TestClient,
    cookies: dict[str, str],
    config_id: int,
) -> None:
    initiated = client.post(
        "/oauth-config/initiate",
        json={"oauth_config_id": config_id, "return_path": "/chat"},
        cookies=cookies,
    )
    assert initiated.status_code == 200, initiated.status_code
    state = initiated.json()["state"]
    callback = client.post(
        "/oauth-config/callback",
        params={"code": AUTH_CODE, "state": state},
        cookies=cookies,
    )
    assert callback.status_code == 200, callback.status_code
    assert callback.json().get("error") is None


def _run_action(session: Session, owner: User, victim: User, config_id: int) -> None:
    tool = Tool(
        name=f"oauth-tool-{uuid.uuid4().hex[:8]}",
        description="OAuth action",
        openapi_schema=_OPENAPI_SCHEMA,
        oauth_config_id=config_id,
        user_id=owner.id,
        enabled=True,
    )
    session.add(tool)
    session.commit()
    session.refresh(tool)
    persona = Persona(
        name=f"oauth-persona-{uuid.uuid4().hex[:8]}",
        description="OAuth action persona",
        tools=[tool],
        is_public=True,
        deleted=False,
    )
    session.add(persona)
    session.commit()
    session.refresh(persona)
    session.refresh(victim)
    construct_tools(
        persona=persona,
        db_session=session,
        emitter=Emitter(merged_queue=Queue()),
        user=victim,
        llm=cast(LLM, Mock()),
    )


def _assert_not_sent(attacker: _TokenServer) -> None:
    if _contains(attacker.bodies, CLIENT_SECRET):
        raise AssertionError(
            "stored client secret was posted to the caller-supplied token URL"
        )
    if _contains(attacker.bodies, REFRESH_TOKEN):
        raise AssertionError(
            "stored user refresh token was posted to the caller-supplied token URL"
        )


@pytest.mark.parametrize("kind", ["group_curator", "global_curator"])
def test_non_admin_cannot_retarget_oauth_config_or_exfiltrate_tokens(
    client: TestClient,
    db_session: Session,
    kind: str,
) -> None:
    legitimate, legitimate_thread = _start_token_server()
    attacker, attacker_thread = _start_token_server()
    try:
        admin = _admin(db_session)
        victim = _basic_member(db_session, "oauth_victim", UserRole.BASIC)
        if kind == "group_curator":
            actor = _group_curator(db_session, admin)
        else:
            actor = _basic_member(db_session, "oauth_global_curator", UserRole.BASIC)
            actor.role = UserRole.GLOBAL_CURATOR
            db_session.commit()
            db_session.refresh(actor)
            assert actor.role == UserRole.GLOBAL_CURATOR

        admin_cookies = _cookies_for(admin)
        actor_cookies = _cookies_for(actor)
        victim_cookies = _cookies_for(victim)
        original_authorization_url = "http://127.0.0.1:9/authorize"
        kept = _create_config(
            client,
            admin_cookies,
            token_url=legitimate.url("/kept"),
            authorization_url=original_authorization_url,
        )
        target = _create_config(
            client,
            admin_cookies,
            token_url=legitimate.url("/target"),
            authorization_url=original_authorization_url,
        )
        _authorize(client, victim_cookies, target)
        db_session.expire_all()
        assert _plain_refresh(db_session, target, victim.id) == REFRESH_TOKEN
        assert _contains(legitimate.bodies, CLIENT_SECRET)

        listed = client.get("/admin/oauth-config", cookies=actor_cookies)
        if listed.status_code == 200:
            for item in listed.json():
                assert "client_secret" not in item
                assert "client_id" not in item

        attacker_token_url = attacker.url("/token")
        updated = client.put(
            f"/admin/oauth-config/{target}",
            json={"token_url": attacker_token_url},
            cookies=actor_cookies,
        )
        _run_action(db_session, admin, victim, target)
        _authorize(client, actor_cookies, target)

        _assert_not_sent(attacker)
        assert updated.status_code >= 400

        kept_config = _reload(db_session, kept)
        target_config = _reload(db_session, target)
        assert kept_config.token_url == legitimate.url("/kept")
        assert target_config.token_url == legitimate.url("/target")
        assert target_config.authorization_url == original_authorization_url
        assert _plain_secret(kept_config) == CLIENT_SECRET
        assert _plain_secret(target_config) == CLIENT_SECRET
        assert _plain_refresh(db_session, target, victim.id) == REFRESH_TOKEN
    finally:
        _stop_token_server(legitimate, legitimate_thread)
        _stop_token_server(attacker, attacker_thread)


@pytest.mark.parametrize("kind", ["group_curator", "global_curator"])
def test_non_admin_cannot_delete_another_oauth_config(
    client: TestClient,
    db_session: Session,
    kind: str,
) -> None:
    legitimate, legitimate_thread = _start_token_server()
    try:
        admin = _admin(db_session)
        victim = _basic_member(db_session, "oauth_delete_victim", UserRole.BASIC)
        if kind == "group_curator":
            actor = _group_curator(db_session, admin)
        else:
            actor = _basic_member(
                db_session, "oauth_delete_global_curator", UserRole.BASIC
            )
            actor.role = UserRole.GLOBAL_CURATOR
            db_session.commit()
            db_session.refresh(actor)

        config_id = _create_config(
            client,
            _cookies_for(admin),
            token_url=legitimate.url("/delete-target"),
            authorization_url="http://127.0.0.1:9/authorize",
        )
        _authorize(client, _cookies_for(victim), config_id)

        other_id = _create_config(
            client,
            _cookies_for(admin),
            token_url=legitimate.url("/delete-other"),
            authorization_url="http://127.0.0.1:9/authorize",
        )
        deleted = client.delete(
            f"/admin/oauth-config/{config_id}",
            cookies=_cookies_for(actor),
        )
        assert deleted.status_code >= 400
        assert _reload(db_session, config_id).token_url.endswith("/delete-target")
        assert _reload(db_session, other_id).token_url.endswith("/delete-other")
        assert _plain_secret(_reload(db_session, config_id)) == CLIENT_SECRET
        assert _plain_refresh(db_session, config_id, victim.id) == REFRESH_TOKEN
    finally:
        _stop_token_server(legitimate, legitimate_thread)


def test_basic_user_cannot_update_or_delete_oauth_config(
    client: TestClient,
    db_session: Session,
) -> None:
    admin = _admin(db_session)
    basic = _basic_member(db_session, "oauth_basic", UserRole.BASIC)
    config_id = _create_config(
        client,
        _cookies_for(admin),
        token_url="http://127.0.0.1:9/basic-token",
        authorization_url="http://127.0.0.1:9/authorize",
    )
    cookies = _cookies_for(basic)
    updated = client.put(
        f"/admin/oauth-config/{config_id}",
        json={"token_url": "http://127.0.0.1:9/replaced"},
        cookies=cookies,
    )
    deleted = client.delete(f"/admin/oauth-config/{config_id}", cookies=cookies)
    assert updated.status_code >= 400
    assert deleted.status_code >= 400
    config = _reload(db_session, config_id)
    assert config.token_url == "http://127.0.0.1:9/basic-token"
    assert _plain_secret(config) == CLIENT_SECRET


def test_token_url_change_does_not_send_stored_secret_or_refresh_token(
    client: TestClient,
    db_session: Session,
) -> None:
    legitimate, legitimate_thread = _start_token_server()
    attacker, attacker_thread = _start_token_server()
    try:
        admin = _admin(db_session)
        victim = _basic_member(db_session, "oauth_admin_victim", UserRole.BASIC)
        original_token_url = legitimate.url("/admin-target")
        original_authorization_url = "http://127.0.0.1:9/authorize"
        config_id = _create_config(
            client,
            _cookies_for(admin),
            token_url=original_token_url,
            authorization_url=original_authorization_url,
        )
        _authorize(client, _cookies_for(victim), config_id)
        attacker_token_url = attacker.url("/token")
        client.put(
            f"/admin/oauth-config/{config_id}",
            json={"token_url": attacker_token_url},
            cookies=_cookies_for(admin),
        )
        _run_action(db_session, admin, victim, config_id)
        _authorize(client, _cookies_for(admin), config_id)

        _assert_not_sent(attacker)
        config = _reload(db_session, config_id)
        if config.token_url == attacker_token_url:
            if _plain_secret(config) == CLIENT_SECRET:
                raise AssertionError(
                    "token URL changed while the stored client secret was kept"
                )
            if _plain_refresh(db_session, config_id, victim.id) == REFRESH_TOKEN:
                raise AssertionError(
                    "token URL changed while the stored refresh token was kept"
                )
        else:
            assert config.token_url == original_token_url
            assert config.authorization_url == original_authorization_url
            assert _plain_secret(config) == CLIENT_SECRET
    finally:
        _stop_token_server(legitimate, legitimate_thread)
        _stop_token_server(attacker, attacker_thread)


def test_authorization_url_change_drops_stored_secret_and_refresh_token(
    client: TestClient,
    db_session: Session,
) -> None:
    legitimate, legitimate_thread = _start_token_server()
    try:
        admin = _admin(db_session)
        victim = _basic_member(db_session, "oauth_auth_url_victim", UserRole.BASIC)
        original_token_url = legitimate.url("/auth-url-target")
        original_authorization_url = "http://127.0.0.1:9/authorize"
        attacker_authorization_url = "http://127.0.0.1:9/attacker-authorize"
        config_id = _create_config(
            client,
            _cookies_for(admin),
            token_url=original_token_url,
            authorization_url=original_authorization_url,
        )
        _authorize(client, _cookies_for(victim), config_id)
        recorded_before = len(legitimate.bodies)
        client.put(
            f"/admin/oauth-config/{config_id}",
            json={"authorization_url": attacker_authorization_url},
            cookies=_cookies_for(admin),
        )
        _run_action(db_session, admin, victim, config_id)
        config = _reload(db_session, config_id)
        new_bodies = legitimate.bodies[recorded_before:]
        if config.authorization_url == attacker_authorization_url:
            if _plain_secret(config) == CLIENT_SECRET:
                raise AssertionError(
                    "authorization URL changed while the stored client secret was kept"
                )
            if _plain_refresh(db_session, config_id, victim.id) == REFRESH_TOKEN:
                raise AssertionError(
                    "authorization URL changed while the stored refresh token was kept"
                )
            if _contains(new_bodies, REFRESH_TOKEN) or _contains(
                new_bodies, CLIENT_SECRET
            ):
                raise AssertionError(
                    "stored OAuth credentials were posted after the authorization URL changed"
                )
        else:
            assert config.authorization_url == original_authorization_url
            assert config.token_url == original_token_url
            assert _plain_secret(config) == CLIENT_SECRET
    finally:
        _stop_token_server(legitimate, legitimate_thread)
