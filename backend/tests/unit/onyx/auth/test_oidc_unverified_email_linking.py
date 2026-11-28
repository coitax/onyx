"""OIDC login must not treat an unverified email as proof of account ownership.

The identity provider in these tests is a local stand-in. Token exchange and
the userinfo request go through ``httpx_oauth.clients.openid.OpenID`` (the
client ``onyx.main`` constructs for ``AUTH_TYPE=oidc``). Account lookup uses
``SQLAlchemyUserDatabase`` on SQLite, which applies the same case-insensitive
email match and oauth-account match as the PostgreSQL adapter. The session
cookie is issued by ``AuthenticationBackend.login`` and
``SingleTenantJWTStrategy``.

Not covered here:
- Google OAuth. That client calls Google's People API, not an email claim the
  user can edit.
- ``refresh_access_token`` in ``ee/onyx/server/enterprise_settings/api.py``.
  That endpoint associates whatever email the caller puts in the body; it is
  not the OIDC userinfo path.
- SAML callback in ``onyx/server/saml.py``. It trusts a signed SAML assertion
  rather than an OIDC ``email_verified`` claim.
"""

from __future__ import annotations

import ast
import base64
import json
import os
import threading
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import urlparse

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_users.authentication import AuthenticationBackend
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseOAuthAccountTableUUID
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from httpx_oauth.clients.openid import BASE_SCOPES
from httpx_oauth.clients.openid import OpenID
from sqlalchemy import DateTime
from sqlalchemy import Enum
from sqlalchemy import func
from sqlalchemy import JSON
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

os.environ["DISABLE_TELEMETRY"] = "true"
os.environ["DISPOSABLE_EMAIL_DOMAINS_URL"] = ""

from onyx.auth.schemas import UserRole  # noqa: E402
from onyx.auth.users import cookie_transport  # noqa: E402
from onyx.auth.users import create_onyx_oauth_router  # noqa: E402
from onyx.auth.users import get_user_manager  # noqa: E402
from onyx.auth.users import SingleTenantJWTStrategy  # noqa: E402
from onyx.auth.users import UserManager  # noqa: E402
from onyx.configs.constants import AuthType  # noqa: E402
from onyx.configs.constants import FASTAPI_USERS_AUTH_COOKIE_NAME  # noqa: E402
from onyx.db.enums import AccountType  # noqa: E402
from onyx.error_handling.exceptions import register_onyx_exception_handlers  # noqa: E402
from onyx.key_value_store.interface import KeyValueStore  # noqa: E402
from onyx.key_value_store.interface import KvKeyNotFoundError  # noqa: E402

# Placeholder signing key for the app session cookie. Not a deployed secret.
TEST_AUTH_SECRET = "test-auth-secret-placeholder-value"
ATTACKER_SUB = "attacker-subject"
ATTACKER_EMAIL = "attacker@evil.example"
VICTIM_EMAIL = "admin@victim-company.com"
_MISSING = object()


class _AuthBase(DeclarativeBase):
    pass


class OAuthAccountRow(SQLAlchemyBaseOAuthAccountTableUUID, _AuthBase):
    user: Mapped[UserRow] = relationship(back_populates="oauth_accounts")


class UserRow(SQLAlchemyBaseUserTableUUID, _AuthBase):
    oauth_accounts: Mapped[list[OAuthAccountRow]] = relationship(
        back_populates="user", lazy="selectin"
    )
    account_type: Mapped[AccountType] = mapped_column(
        Enum(AccountType, native_enum=False, length=32),
        nullable=False,
        default=AccountType.STANDARD,
    )
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, native_enum=False, length=32),
        nullable=False,
        default=UserRole.BASIC,
    )
    oidc_expiry: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    pinned_assistants: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)


@dataclass
class _IdP:
    """Local OpenID provider. Userinfo responses are returned in order."""

    profiles: list[dict[str, Any]]
    id_token_email: str
    id_token_email_verified: bool
    server: ThreadingHTTPServer | None = None
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _private_key: rsa.RSAPrivateKey = field(init=False, repr=False)
    _kid: str = "test-idp-key"

    def __post_init__(self) -> None:
        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    @property
    def base_url(self) -> str:
        assert self.server is not None
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def discovery_url(self) -> str:
        return f"{self.base_url}/.well-known/openid-configuration"

    def start(self) -> None:
        idp = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/.well-known/openid-configuration":
                    self._json(idp._discovery())
                    return
                if path == "/jwks":
                    self._json(idp._jwks())
                    return
                if path == "/userinfo":
                    if self.headers.get("Authorization") != "Bearer attacker-access-token":
                        self.send_error(401)
                        return
                    self._json(idp._next_profile())
                    return
                self.send_error(404)

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                if path == "/token":
                    self._json(idp._token())
                    return
                self.send_error(404)

            def _json(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

    def _discovery(self) -> dict[str, Any]:
        return {
            "issuer": self.base_url,
            "authorization_endpoint": f"{self.base_url}/authorize",
            "token_endpoint": f"{self.base_url}/token",
            "userinfo_endpoint": f"{self.base_url}/userinfo",
            "jwks_uri": f"{self.base_url}/jwks",
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": ["client_secret_post"],
            "id_token_signing_alg_values_supported": ["RS256"],
        }

    def _jwks(self) -> dict[str, Any]:
        public_numbers = self._private_key.public_key().public_numbers()
        return {
            "keys": [
                {
                    "kty": "RSA",
                    "use": "sig",
                    "alg": "RS256",
                    "kid": self._kid,
                    "n": _b64url_int(public_numbers.n),
                    "e": _b64url_int(public_numbers.e),
                }
            ]
        }

    def _token(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        id_claims = {
            "iss": self.base_url,
            "sub": ATTACKER_SUB,
            "aud": "test-oauth-client",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=5)).timestamp()),
            "email": self.id_token_email,
            "email_verified": self.id_token_email_verified,
        }
        private_pem = self._private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        id_token = jwt.encode(
            id_claims, private_pem, algorithm="RS256", headers={"kid": self._kid}
        )
        return {
            "access_token": "attacker-access-token",
            "token_type": "Bearer",
            "refresh_token": "attacker-refresh-token",
            "expires_in": 3600,
            "id_token": id_token,
        }

    def _next_profile(self) -> dict[str, Any]:
        with self._lock:
            index = min(self.calls, len(self.profiles) - 1)
            self.calls += 1
            return self.profiles[index]


def _b64url_int(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _profile(
    email: str, email_verified: Any = _MISSING, sub: str = ATTACKER_SUB
) -> dict[str, Any]:
    payload: dict[str, Any] = {"sub": sub, "email": email}
    if email_verified is not _MISSING:
        payload["email_verified"] = email_verified
    return payload


class _MemoryKv(KeyValueStore):
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def store(self, key: str, val: Any, encrypt: bool = False) -> None:
        self._data[key] = val

    def load(self, key: str, refresh_cache: bool = False) -> Any:
        if key not in self._data:
            raise KvKeyNotFoundError()
        return self._data[key]

    def delete(self, key: str) -> None:
        self._data.pop(key, None)


class _DummySyncSession:
    def commit(self) -> None:
        return None


class _QueryResult:
    def scalars(self) -> _QueryResult:
        return self

    def all(self) -> list[Any]:
        return []


class _DummyAsyncSession:
    async def execute(self, *args: Any, **kwargs: Any) -> _QueryResult:
        return _QueryResult()

    def expire(self, obj: Any) -> None:
        return None


@asynccontextmanager
async def _dummy_async_session() -> Any:
    yield _DummyAsyncSession()


def _dummy_async_session_cm(tenant_id: str | None = None) -> Any:
    return _dummy_async_session()


@contextmanager
def _dummy_sync_session(*args: Any, **kwargs: Any) -> Iterator[_DummySyncSession]:
    yield _DummySyncSession()


async def _user_count() -> int:
    return 2


def _no_default_admins() -> list[str]:
    return []


def _no_groups(*args: Any, **kwargs: Any) -> None:
    return None


def _noop(*args: Any, **kwargs: Any) -> None:
    return None


@dataclass
class _World:
    sync_session_factory: sessionmaker[Session]
    db_path: str
    victim_id: uuid.UUID
    idp: _IdP

    def emails(self) -> dict[str, tuple[bool, list[str]]]:
        """Map lowercased email to (is_verified, oauth account ids)."""
        with self.sync_session_factory() as session:
            rows = session.scalars(
                select(UserRow).options(selectinload(UserRow.oauth_accounts))
            ).all()
            return {
                row.email.lower(): (
                    bool(row.is_verified),
                    [account.account_id for account in row.oauth_accounts],
                )
                for row in rows
            }

    def session_subject(self, response: Any) -> str | None:
        token = response.cookies.get(FASTAPI_USERS_AUTH_COOKIE_NAME)
        if not token:
            return None
        payload = jwt.decode(
            token,
            TEST_AUTH_SECRET,
            algorithms=["HS256"],
            audience=["fastapi-users:auth"],
        )
        subject = payload.get("sub")
        return str(subject) if subject is not None else None


def _production_oidc_router(idp: _IdP, *, pkce: bool) -> Any:
    """Evaluate the OIDC ``create_onyx_oauth_router`` call inside ``get_application``.

    Names are resolved from ``onyx.main``'s imports, so a replacement client
    class referenced by that call is what the test runs.
    """
    main_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "onyx", "main.py"
    )
    main_path = os.path.abspath(main_path)
    source = open(main_path, encoding="utf-8").read()
    tree = ast.parse(source)
    call = _find_oidc_router_call(tree)
    namespace = _namespace_for_call(tree, call, idp, pkce)
    expression = ast.Expression(call)
    ast.fix_missing_locations(expression)
    return eval(compile(expression, main_path, "eval"), namespace)  # noqa: S307


def _find_oidc_router_call(tree: ast.AST) -> ast.Call:
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "get_application":
            continue
        for stmt in node.body:
            found = _oidc_call_in_if(stmt)
            if found is not None:
                return found
    raise AssertionError("OIDC create_onyx_oauth_router call not found in onyx.main")


def _oidc_call_in_if(stmt: ast.stmt) -> ast.Call | None:
    if not isinstance(stmt, ast.If):
        return None
    if _is_oidc_test(stmt.test):
        for child in ast.walk(ast.Module(body=stmt.body, type_ignores=[])):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "create_onyx_oauth_router"
            ):
                return child
        return None
    for child in stmt.orelse:
        found = _oidc_call_in_if(child)
        if found is not None:
            return found
    return None


def _is_oidc_test(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "AUTH_TYPE"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and isinstance(test.comparators[0], ast.Attribute)
        and test.comparators[0].attr == "OIDC"
    )


def _namespace_for_call(
    tree: ast.Module, call: ast.Call, idp: _IdP, pkce: bool
) -> dict[str, Any]:
    needed = {node.id for node in ast.walk(call) if isinstance(node, ast.Name)}
    imported: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        wanted = [
            alias
            for alias in node.names
            if (alias.asname or alias.name) in needed
        ]
        if not wanted:
            continue
        module = __import__(node.module, fromlist=[alias.name for alias in wanted])
        for alias in wanted:
            imported[alias.asname or alias.name] = getattr(module, alias.name)

    scopes = list(imported.get("OIDC_SCOPE_OVERRIDE") or BASE_SCOPES)
    if "offline_access" not in scopes:
        scopes.append("offline_access")

    def get_strategy() -> SingleTenantJWTStrategy:
        return SingleTenantJWTStrategy(
            secret=TEST_AUTH_SECRET, lifetime_seconds=3600
        )

    namespace: dict[str, Any] = {
        **imported,
        "create_onyx_oauth_router": create_onyx_oauth_router,
        "OpenID": OpenID,
        "OAUTH_CLIENT_ID": "test-oauth-client",
        "OAUTH_CLIENT_SECRET": "test-oauth-secret",
        "OPENID_CONFIG_URL": idp.discovery_url,
        "USER_AUTH_SECRET": TEST_AUTH_SECRET,
        "WEB_DOMAIN": "http://localhost:3000",
        "OIDC_PKCE_ENABLED": pkce,
        "oidc_scopes": scopes,
        "auth_backend": AuthenticationBackend(
            name="jwt",
            transport=cookie_transport,
            get_strategy=get_strategy,
        ),
    }
    return namespace


def _associate_by_email_from_source() -> bool:
    main_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "onyx", "main.py"
        )
    )
    tree = ast.parse(open(main_path, encoding="utf-8").read())
    call = _find_oidc_router_call(tree)
    for keyword in call.keywords:
        if keyword.arg == "associate_by_email" and isinstance(
            keyword.value, ast.Constant
        ):
            return bool(keyword.value.value)
    return False


@pytest.fixture
def world(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[_World]:
    db_path = str(tmp_path / "users.sqlite")
    sync_engine = __import__("sqlalchemy").create_engine(f"sqlite:///{db_path}")
    _AuthBase.metadata.create_all(sync_engine)
    factory = sessionmaker(sync_engine, expire_on_commit=False)
    victim_id = uuid.uuid4()
    with factory() as session:
        session.add(
            UserRow(
                id=victim_id,
                email=VICTIM_EMAIL,
                hashed_password="not-a-password",
                is_active=True,
                is_superuser=False,
                is_verified=True,
                account_type=AccountType.STANDARD,
                role=UserRole.ADMIN,
            )
        )
        session.commit()

    idp = _IdP(
        profiles=[_profile(VICTIM_EMAIL, False)],
        id_token_email=ATTACKER_EMAIL,
        id_token_email_verified=True,
    )
    idp.start()

    import onyx.auth.users as users_module
    import onyx.auth.invited_users as invited_users_module

    def fake_get_user_by_email(email: str, db_session: Any) -> UserRow | None:
        del db_session
        with factory() as session:
            return session.scalar(
                select(UserRow).where(func.lower(UserRow.email) == func.lower(email))
            )

    monkeypatch.setattr(users_module, "AUTH_TYPE", AuthType.OIDC)
    monkeypatch.setattr(
        users_module, "get_async_session_context_manager", _dummy_async_session_cm
    )
    monkeypatch.setattr(users_module, "get_session_with_tenant", _dummy_sync_session)
    monkeypatch.setattr(
        users_module, "get_session_with_current_tenant", _dummy_sync_session
    )
    monkeypatch.setattr(users_module, "get_user_by_email", fake_get_user_by_email)
    monkeypatch.setattr(users_module, "get_user_count", _user_count)
    monkeypatch.setattr(
        users_module, "get_default_admin_user_emails", _no_default_admins
    )
    monkeypatch.setattr(
        users_module, "assign_user_to_default_groups__no_commit", _no_groups
    )
    monkeypatch.setattr(users_module, "optional_telemetry", _noop)
    monkeypatch.setattr(invited_users_module, "get_kv_store", lambda: _MemoryKv())

    try:
        yield _World(
            sync_session_factory=factory,
            db_path=db_path,
            victim_id=victim_id,
            idp=idp,
        )
    finally:
        idp.stop()
        sync_engine.dispose()


def _run_callback(world: _World, *, pkce: bool) -> Any:
    router = _production_oidc_router(world.idp, pkce=pkce)
    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(router, prefix="/auth/oidc")

    async def override_user_manager() -> Any:
        engine = create_async_engine(f"sqlite+aiosqlite:///{world.db_path}")
        maker = async_sessionmaker(engine, expire_on_commit=False)
        async with maker() as session:
            manager = UserManager(
                SQLAlchemyUserDatabase(session, UserRow, OAuthAccountRow)
            )
            yield manager
        await engine.dispose()

    app.dependency_overrides[get_user_manager] = override_user_manager
    client = TestClient(app, raise_server_exceptions=False)
    authorize = client.get("/auth/oidc/authorize")
    assert authorize.status_code == 200, authorize.text
    state = parse_qs(urlparse(authorize.json()["authorization_url"]).query)["state"][0]
    return client.get(
        "/auth/oidc/callback",
        params={"code": "test-authorization-code", "state": state},
        follow_redirects=False,
    )


def _assert_victim_not_taken_over(world: _World, response: Any) -> None:
    _verified, account_ids = world.emails()[VICTIM_EMAIL]
    subject = world.session_subject(response)
    assert ATTACKER_SUB not in account_ids, (
        f"status={response.status_code} subject={subject} victim={world.victim_id} "
        f"accounts={account_ids} location={response.headers.get('location')}"
    )
    assert subject != str(world.victim_id), response.text


@pytest.mark.parametrize("pkce", [False, True])
@pytest.mark.parametrize(
    "email_verified",
    [False, _MISSING, "false", "False", "true", 0, 1],
)
def test_unverified_userinfo_does_not_link_or_sign_in_as_victim(
    world: _World, pkce: bool, email_verified: Any
) -> None:
    world.idp.profiles = [_profile(VICTIM_EMAIL, email_verified)]
    world.idp.id_token_email = ATTACKER_EMAIL
    world.idp.id_token_email_verified = True

    response = _run_callback(world, pkce=pkce)

    _assert_victim_not_taken_over(world, response)


def test_case_insensitive_unverified_email_does_not_link_victim(world: _World) -> None:
    world.idp.profiles = [_profile("Admin@Victim-Company.com", False)]
    world.idp.id_token_email = ATTACKER_EMAIL
    world.idp.id_token_email_verified = True

    response = _run_callback(world, pkce=False)

    _assert_victim_not_taken_over(world, response)


def test_unverified_email_does_not_create_a_verified_account(world: _World) -> None:
    new_email = "new-admin@victim-company.com"
    world.idp.profiles = [_profile(new_email, False)]
    world.idp.id_token_email = ATTACKER_EMAIL
    world.idp.id_token_email_verified = True

    response = _run_callback(world, pkce=False)

    accounts = world.emails()
    created = accounts.get(new_email)
    if created is not None:
        verified, _account_ids = created
        assert verified is False, response.text
    assert world.session_subject(response) != str(world.victim_id)


def test_later_userinfo_response_cannot_link_a_different_email(world: _World) -> None:
    """A check against one userinfo document must not authorize a later one.

    The first document verifies the attacker's own email. Every later document
    names the victim and is not verified. Signing in as the attacker is fine.
    Signing in as the victim, or attaching the attacker's subject to the victim,
    is not.
    """
    with world.sync_session_factory() as session:
        session.add(
            UserRow(
                email=ATTACKER_EMAIL,
                hashed_password="not-a-password",
                is_active=True,
                is_superuser=False,
                is_verified=True,
                account_type=AccountType.STANDARD,
                role=UserRole.BASIC,
            )
        )
        session.commit()

    world.idp.profiles = [
        _profile(ATTACKER_EMAIL, True),
        _profile(VICTIM_EMAIL, False),
    ]
    world.idp.id_token_email = ATTACKER_EMAIL
    world.idp.id_token_email_verified = True
    world.idp.calls = 0

    response = _run_callback(world, pkce=False)

    _assert_victim_not_taken_over(world, response)


def test_verified_userinfo_can_link_existing_account(world: _World) -> None:
    if not _associate_by_email_from_source():
        pytest.skip("OIDC router does not associate existing accounts by email")

    world.idp.profiles = [_profile(VICTIM_EMAIL, True)]
    world.idp.id_token_email = VICTIM_EMAIL
    world.idp.id_token_email_verified = True

    response = _run_callback(world, pkce=False)

    assert response.status_code == 302, response.text
    assert world.session_subject(response) == str(world.victim_id)
    verified, account_ids = world.emails()[VICTIM_EMAIL]
    assert verified is True
    assert ATTACKER_SUB in account_ids
