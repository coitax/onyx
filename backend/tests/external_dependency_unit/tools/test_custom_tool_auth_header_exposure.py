"""Basic users must not receive custom OpenAPI action header values.

Admins store authentication headers on a custom tool. Listing and detail
endpoints, and persona payloads that embed the same tool snapshot, are
available to any signed-in user with basic access. These tests fail while
those responses still contain the configured header values.
"""

from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_users.password import PasswordHelper
from sqlalchemy import text
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.configs.app_configs import POSTGRES_DB
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.engine.sql_engine import SqlEngine
from onyx.db.enums import AccountType
from onyx.db.enums import GrantSource
from onyx.db.enums import Permission
from onyx.db.models import Base
from onyx.db.models import PermissionGrant
from onyx.db.models import Persona
from onyx.db.models import Tool
from onyx.db.models import User
from onyx.db.models import UserGroup
from onyx.db.pat import create_pat
from onyx.db.permissions import recompute_user_permissions__no_commit
from onyx.db.users import assign_user_to_default_groups__no_commit
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.server.features.persona.api import admin_agents_router
from onyx.server.features.persona.api import admin_router as admin_persona_router
from onyx.server.features.persona.api import agents_router
from onyx.server.features.persona.api import basic_router as persona_router
from onyx.server.features.tool.api import admin_router as admin_tool_router
from onyx.server.features.tool.api import router as tool_router

# Obvious placeholders. These are not credentials.
HEADER_VALUE = "placeholder-custom-header-value"
UPDATED_HEADER_VALUE = "placeholder-updated-header-value"
DISABLED_HEADER_VALUE = "placeholder-disabled-tool-header-value"

ENABLED_TOOL_NAME = "exposure-check-action"
DISABLED_TOOL_NAME = "exposure-check-disabled-action"
UPDATED_TOOL_NAME = "exposure-check-updated-action"
PUBLIC_AGENT_NAME = "exposure-check-agent"

OPENAPI_DEFINITION: dict[str, Any] = {
    "openapi": "3.0.0",
    "info": {
        "title": "Exposure Check",
        "description": "Schema used to store a custom header on an action",
        "version": "1.0.0",
    },
    "servers": [{"url": "https://example.invalid"}],
    "paths": {
        "/ping": {
            "get": {
                "operationId": "pingExposureCheck",
                "summary": "Ping",
                "responses": {"200": {"description": "ok"}},
            }
        }
    },
}

_SCHEMA_TABLES = [
    "user",
    "oauth_account",
    "oauth_config",
    "llm_provider",
    "model_configuration",
    "mcp_connection_config",
    "mcp_server",
    "user_group",
    "user__user_group",
    "permission_grant",
    "tool",
    "personal_access_token",
    "document",
    "document_set",
    "hierarchy_node",
    "user_file",
    "persona_label",
    "persona",
    "persona__tool",
    "persona__user",
    "persona__user_group",
    "persona__document_set",
    "persona__hierarchy_node",
    "persona__document",
    "persona__user_file",
    "persona__persona_label",
]


def _table_closure(seeds: list[str]) -> list[Any]:
    seen: set[str] = set()
    stack = list(seeds)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        table = Base.metadata.tables[name]
        for foreign_key in table.foreign_keys:
            stack.append(foreign_key.column.table.name)
    return [Base.metadata.tables[name] for name in seen]


def _reset_schema() -> None:
    if POSTGRES_DB != "onyx_header_test":
        raise RuntimeError(
            "Refusing to reset a database other than onyx_header_test. "
            "Set POSTGRES_DB=onyx_header_test before starting pytest."
        )
    SqlEngine.init_engine(pool_size=5, max_overflow=2)
    engine = SqlEngine.get_engine()
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    Base.metadata.create_all(engine, tables=_table_closure(_SCHEMA_TABLES))


def _create_user(session: Session, role: UserRole) -> User:
    password_helper = PasswordHelper()
    user = User(
        id=uuid4(),
        email=f"{role.value}-{uuid4().hex[:8]}@example.com",
        hashed_password=password_helper.hash(password_helper.generate()),
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role=role,
        account_type=AccountType.STANDARD,
    )
    session.add(user)
    session.flush()
    return user


def _bearer(raw_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {raw_token}"}


def _tool_body(name: str, header_value: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": "Custom action with a header configured by an admin",
        "definition": OPENAPI_DEFINITION,
        "custom_headers": [
            {"key": "Authorization", "value": header_value},
            {"key": "X-Placeholder-Header", "value": header_value},
        ],
        "passthrough_auth": False,
    }


def _stored_header_values(tool_id: int) -> list[str]:
    with get_session_with_current_tenant() as session:
        tool = session.get(Tool, tool_id)
        assert tool is not None
        session.refresh(tool)
        headers = tool.custom_headers or []
        return [str(item["value"]) for item in headers]


def _assert_listed_without_secrets(
    response_text: str,
    status_code: int,
    expected_names: list[str],
    secrets: list[str],
) -> None:
    assert status_code == 200, response_text
    for name in expected_names:
        assert name in response_text
    for secret in secrets:
        assert secret not in response_text


class _World:
    def __init__(self, client: TestClient) -> None:
        self.client = client
        self.admin_headers: dict[str, str] = {}
        self.basic_headers: dict[str, str] = {}
        self.curator_headers: dict[str, str] = {}
        self.no_access_headers: dict[str, str] = {}
        self.enabled_tool_id: int = 0
        self.disabled_tool_id: int = 0
        self.persona_id: int = 0


@pytest.fixture(scope="module")
def world() -> Iterator[_World]:
    _reset_schema()
    with get_session_with_current_tenant() as session:
        group = UserGroup(
            name="Basic",
            is_up_to_date=True,
            is_up_for_deletion=False,
            is_default=True,
        )
        session.add(group)
        session.flush()
        session.add(
            PermissionGrant(
                group_id=group.id,
                permission=Permission.BASIC_ACCESS,
                grant_source=GrantSource.SYSTEM,
            )
        )
        session.flush()

        admin = _create_user(session, UserRole.ADMIN)
        basic = _create_user(session, UserRole.BASIC)
        curator = _create_user(session, UserRole.CURATOR)
        no_access = _create_user(session, UserRole.BASIC)
        assign_user_to_default_groups__no_commit(session, basic, is_admin=False)
        assign_user_to_default_groups__no_commit(session, curator, is_admin=False)
        # Recompute the user who was never added to a group so the column
        # stays empty rather than relying on the insert default alone.
        recompute_user_permissions__no_commit(no_access.id, session)
        session.commit()
        session.refresh(basic)
        session.refresh(curator)
        session.refresh(no_access)
        assert basic.effective_permissions == ["basic"]
        assert curator.effective_permissions == ["basic"]
        assert no_access.effective_permissions == []

        _, admin_token = create_pat(session, admin.id, "admin", None)
        _, basic_token = create_pat(session, basic.id, "basic", None)
        _, curator_token = create_pat(session, curator.id, "curator", None)
        _, no_access_token = create_pat(session, no_access.id, "none", None)
        admin_id = admin.id

    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(tool_router)
    app.include_router(admin_tool_router)
    app.include_router(persona_router)
    app.include_router(agents_router)
    app.include_router(admin_persona_router)
    app.include_router(admin_agents_router)

    with TestClient(app) as client:
        holder = _World(client)
        holder.admin_headers = _bearer(admin_token)
        holder.basic_headers = _bearer(basic_token)
        holder.curator_headers = _bearer(curator_token)
        holder.no_access_headers = _bearer(no_access_token)

        created = client.post(
            "/admin/tool/custom",
            json=_tool_body(ENABLED_TOOL_NAME, HEADER_VALUE),
            headers=holder.admin_headers,
        )
        assert created.status_code == 200, created.text
        holder.enabled_tool_id = int(created.json()["id"])
        assert HEADER_VALUE in _stored_header_values(holder.enabled_tool_id)

        disabled = client.post(
            "/admin/tool/custom",
            json=_tool_body(DISABLED_TOOL_NAME, DISABLED_HEADER_VALUE),
            headers=holder.admin_headers,
        )
        assert disabled.status_code == 200, disabled.text
        holder.disabled_tool_id = int(disabled.json()["id"])
        status_response = client.patch(
            "/admin/tool/status",
            json={"tool_ids": [holder.disabled_tool_id], "enabled": False},
            headers=holder.admin_headers,
        )
        assert status_response.status_code == 200, status_response.text
        assert DISABLED_HEADER_VALUE in _stored_header_values(holder.disabled_tool_id)

        with get_session_with_current_tenant() as session:
            tool = session.get(Tool, holder.enabled_tool_id)
            assert tool is not None
            persona = Persona(
                name=PUBLIC_AGENT_NAME,
                description="Public agent that exposes its tools",
                is_public=True,
                is_listed=True,
                deleted=False,
                builtin_persona=False,
                user_id=admin_id,
            )
            persona.tools.append(tool)
            session.add(persona)
            session.commit()
            session.refresh(persona)
            holder.persona_id = persona.id

        yield holder


def test_basic_and_curator_listing_omits_header_values(world: _World) -> None:
    secrets = [HEADER_VALUE, DISABLED_HEADER_VALUE]
    for headers in (world.basic_headers, world.curator_headers):
        listed = world.client.get("/tool", headers=headers)
        _assert_listed_without_secrets(
            listed.text,
            listed.status_code,
            [ENABLED_TOOL_NAME],
            secrets,
        )
        assert DISABLED_TOOL_NAME not in listed.text

        openapi = world.client.get("/tool/openapi", headers=headers)
        _assert_listed_without_secrets(
            openapi.text,
            openapi.status_code,
            [ENABLED_TOOL_NAME, DISABLED_TOOL_NAME],
            secrets,
        )

        detail = world.client.get(f"/tool/{world.enabled_tool_id}", headers=headers)
        _assert_listed_without_secrets(
            detail.text,
            detail.status_code,
            [ENABLED_TOOL_NAME],
            [HEADER_VALUE],
        )

        disabled_detail = world.client.get(
            f"/tool/{world.disabled_tool_id}", headers=headers
        )
        _assert_listed_without_secrets(
            disabled_detail.text,
            disabled_detail.status_code,
            [DISABLED_TOOL_NAME],
            [DISABLED_HEADER_VALUE],
        )


def test_updated_header_value_stays_hidden(world: _World) -> None:
    created = world.client.post(
        "/admin/tool/custom",
        json=_tool_body(UPDATED_TOOL_NAME, HEADER_VALUE),
        headers=world.admin_headers,
    )
    assert created.status_code == 200, created.text
    tool_id = int(created.json()["id"])

    updated = world.client.put(
        f"/admin/tool/custom/{tool_id}",
        json=_tool_body(UPDATED_TOOL_NAME, UPDATED_HEADER_VALUE),
        headers=world.admin_headers,
    )
    assert updated.status_code == 200, updated.text
    assert UPDATED_HEADER_VALUE in _stored_header_values(tool_id)

    for headers in (world.basic_headers, world.curator_headers):
        for path in (f"/tool/{tool_id}", "/tool", "/tool/openapi"):
            response = world.client.get(path, headers=headers)
            _assert_listed_without_secrets(
                response.text,
                response.status_code,
                [UPDATED_TOOL_NAME],
                [HEADER_VALUE, UPDATED_HEADER_VALUE],
            )


def test_public_agent_payloads_omit_header_values(world: _World) -> None:
    secrets = [HEADER_VALUE, DISABLED_HEADER_VALUE]
    for headers in (world.basic_headers, world.curator_headers):
        listed = world.client.get("/persona", headers=headers)
        _assert_listed_without_secrets(
            listed.text,
            listed.status_code,
            [PUBLIC_AGENT_NAME],
            secrets,
        )

        detail = world.client.get(f"/persona/{world.persona_id}", headers=headers)
        _assert_listed_without_secrets(
            detail.text,
            detail.status_code,
            [PUBLIC_AGENT_NAME, ENABLED_TOOL_NAME],
            secrets,
        )

        agents = world.client.get(
            "/agents",
            params={"page_num": 0, "page_size": 50},
            headers=headers,
        )
        _assert_listed_without_secrets(
            agents.text,
            agents.status_code,
            [PUBLIC_AGENT_NAME],
            secrets,
        )


def test_curator_admin_agent_lists_omit_header_values(world: _World) -> None:
    secrets = [HEADER_VALUE, DISABLED_HEADER_VALUE]
    for path in ("/admin/persona", "/admin/agents"):
        denied = world.client.get(path, headers=world.basic_headers)
        assert denied.status_code == 403, denied.text
        for secret in secrets:
            assert secret not in denied.text

    admin_listed = world.client.get("/admin/persona", headers=world.curator_headers)
    _assert_listed_without_secrets(
        admin_listed.text,
        admin_listed.status_code,
        [PUBLIC_AGENT_NAME, ENABLED_TOOL_NAME],
        secrets,
    )
    admin_agents = world.client.get(
        "/admin/agents",
        params={"page_num": 0, "page_size": 50},
        headers=world.curator_headers,
    )
    _assert_listed_without_secrets(
        admin_agents.text,
        admin_agents.status_code,
        [PUBLIC_AGENT_NAME, ENABLED_TOOL_NAME],
        secrets,
    )


def test_user_without_basic_access_does_not_receive_header_values(
    world: _World,
) -> None:
    secrets = [HEADER_VALUE, DISABLED_HEADER_VALUE]
    # Tool routes require basic access, so this user is rejected outright.
    for path in ("/tool", "/tool/openapi", f"/tool/{world.enabled_tool_id}"):
        response = world.client.get(path, headers=world.no_access_headers)
        assert response.status_code == 403, response.text
        for secret in secrets:
            assert secret not in response.text

    # Agent routes authenticate any signed-in user. They still must not
    # include custom header values.
    for path in ("/persona", f"/persona/{world.persona_id}", "/agents"):
        response = world.client.get(path, headers=world.no_access_headers)
        assert response.status_code == 200, response.text
        for secret in secrets:
            assert secret not in response.text


def test_anonymous_request_does_not_receive_header_values(world: _World) -> None:
    for path in (
        "/tool",
        "/tool/openapi",
        f"/tool/{world.enabled_tool_id}",
        "/persona",
        f"/persona/{world.persona_id}",
    ):
        response = world.client.get(path)
        assert response.status_code in (401, 403), response.text
        assert HEADER_VALUE not in response.text
        assert DISABLED_HEADER_VALUE not in response.text
