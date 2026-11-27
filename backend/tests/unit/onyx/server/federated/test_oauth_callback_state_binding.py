"""Federated OAuth callback must bind the token to the signed-in caller.

The callback at ``POST /federated/callback`` loads the pending OAuth session
from the ``state`` query parameter and, while this bug is present, writes the
provider access token for ``oauth_session.user_id`` without comparing it to
the authenticated user.

These tests are the spec for that check. They fail until the callback:

* rejects the request when the session user is not the authenticated caller
* does not call the provider token endpoint or write a token row in that case
* on a matching session, stores the token for the authenticated caller even
  if the state is read again or the request carries a different user id

The tool OAuth callback in ``onyx.server.features.oauth_config.api`` already
compares ``session.user_id`` to the caller, and the connector OAuth callback
in ``onyx.server.documents.standard_oauth`` stores the credential for the
authenticated caller. Those paths do not share this bug and are not covered
here. Both federated state issuers (``GET /federated/{id}/authorize`` and
``GET /federated/oauth-status``) are covered because they share this callback.
"""

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs
from urllib.parse import urlparse
from uuid import UUID
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from onyx.auth.schemas import UserRole
from onyx.auth.users import current_user
from onyx.cache.interface import CacheBackend
from onyx.cache.interface import CacheLock
from onyx.configs.constants import FederatedConnectorSource
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import AccountType
from onyx.db.enums import Permission
from onyx.db.federated import get_federated_connector_oauth_token
from onyx.db.federated import update_federated_connector_oauth_token
from onyx.db.models import FederatedConnector
from onyx.db.models import FederatedConnectorOAuthToken
from onyx.db.models import User
from onyx.server.federated.api import router as federated_router
from onyx.tools.tool_implementations.search.search_tool import SearchTool

_SLACK_TOKEN_URL = "https://slack.com/api/oauth.v2.access"
_CLIENT_ID = "placeholder-slack-client-id"
_CLIENT_SECRET = "placeholder-slack-client-secret"
_AUTH_CODE = "placeholder-oauth-authorization-code"
_VICTIM_SLACK_TOKEN = "placeholder-victim-slack-user-token"
_OWNER_SLACK_TOKEN = "placeholder-owner-slack-user-token"
_ATTACKER_EXISTING_TOKEN = "placeholder-attacker-existing-slack-token"
_CONNECTOR_ID = 1


class _Rows:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalar_one_or_none(self) -> Any:
        if not self._rows:
            return None
        if len(self._rows) > 1:
            raise RuntimeError(f"expected at most one row, found {len(self._rows)}")
        return self._rows[0]

    def scalars(self) -> "_Rows":
        return self

    def all(self) -> list[Any]:
        return list(self._rows)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._rows)


def _where_pairs(stmt: Any) -> list[tuple[str, Any]]:
    clause = stmt.whereclause
    if clause is None:
        return []
    nodes = list(clause.clauses) if hasattr(clause, "clauses") else [clause]
    pairs: list[tuple[str, Any]] = []
    for node in nodes:
        name = node.left.key
        value = node.right.value
        pairs.append((name, value))
    return pairs


def _same(left: Any, right: Any) -> bool:
    return str(left) == str(right)


class _TokenSession:
    """In-memory stand-in for the session methods the federated token code uses.

    ``update_federated_connector_oauth_token`` selects by connector id and
    user id, then updates that row or inserts one. ``list`` and ``get`` select
    by those same columns. This session applies those filters to objects the
    real functions added, so a token written for one user is not visible as
    another user's token.
    """

    def __init__(self, connector: FederatedConnector) -> None:
        self.connector = connector
        self.tokens: list[FederatedConnectorOAuthToken] = []

    def _entity(self, stmt: Any) -> type[Any]:
        description = stmt.column_descriptions[0]
        entity = description.get("entity")
        if entity is None:
            raise RuntimeError(f"unsupported statement: {stmt}")
        return entity

    def _matching_tokens(self, stmt: Any) -> list[FederatedConnectorOAuthToken]:
        matched = list(self.tokens)
        for name, value in _where_pairs(stmt):
            matched = [
                token for token in matched if _same(getattr(token, name), value)
            ]
        return matched

    def execute(self, stmt: Any) -> _Rows:
        entity = self._entity(stmt)
        if entity is FederatedConnector:
            rows: list[Any] = [self.connector]
            for name, value in _where_pairs(stmt):
                rows = [row for row in rows if _same(getattr(row, name), value)]
            return _Rows(rows)
        if entity is FederatedConnectorOAuthToken:
            return _Rows(self._matching_tokens(stmt))
        raise RuntimeError(f"unsupported query entity: {entity}")

    def scalars(self, stmt: Any) -> _Rows:
        return self.execute(stmt)

    def add(self, obj: Any) -> None:
        if isinstance(obj, FederatedConnectorOAuthToken):
            obj.federated_connector = self.connector
            self.tokens.append(obj)
            return
        raise RuntimeError(f"unsupported insert: {type(obj)}")

    def delete(self, obj: Any) -> None:
        if obj in self.tokens:
            self.tokens.remove(obj)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _RebindingCache(CacheBackend):
    """Cache whose second read of an OAuth state returns a different user id.

    ``verify_oauth_state`` reads the key once and deletes it. A callback that
    reads the state again between the user check and the token write observes
    ``decoy_user_id`` instead of the user stored by ``generate_oauth_state``.
    """

    def __init__(self, decoy_user_id: str) -> None:
        self._store: dict[str, bytes] = {}
        self._reads: dict[str, int] = {}
        self.decoy_user_id = decoy_user_id

    def get(self, key: str) -> bytes | None:
        raw = self._store.get(key)
        if raw is None:
            return None
        read_count = self._reads.get(key, 0) + 1
        self._reads[key] = read_count
        if read_count == 1:
            payload = json.loads(raw)
            payload["user_id"] = self.decoy_user_id
            self._store[key] = json.dumps(payload).encode()
            return raw
        return self._store.get(key)

    def set(
        self,
        key: str,
        value: str | bytes | int | float,
        ex: int | None = None,
    ) -> None:
        del ex
        if isinstance(value, bytes):
            self._store[key] = value
        else:
            self._store[key] = str(value).encode()

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self._store

    def expire(self, key: str, seconds: int) -> None:
        del key, seconds

    def ttl(self, key: str) -> int:
        return -2 if key not in self._store else -1

    def lock(self, name: str, timeout: float | None = None) -> CacheLock:
        del name, timeout
        raise NotImplementedError

    def rpush(self, key: str, value: str | bytes) -> None:
        del key, value
        raise NotImplementedError

    def blpop(self, keys: list[str], timeout: int = 0) -> tuple[bytes, bytes] | None:
        del keys, timeout
        raise NotImplementedError


def _user(email_prefix: str) -> User:
    return User(
        id=uuid4(),
        email=f"{email_prefix}-{uuid4().hex[:8]}@example.com",
        hashed_password="placeholder-password-hash",
        is_active=True,
        is_verified=True,
        is_superuser=False,
        role=UserRole.BASIC,
        account_type=AccountType.STANDARD,
        effective_permissions=[Permission.BASIC_ACCESS.value],
    )


def _connector() -> FederatedConnector:
    connector = FederatedConnector(
        source=FederatedConnectorSource.FEDERATED_SLACK,
        credentials={
            "client_id": _CLIENT_ID,
            "client_secret": _CLIENT_SECRET,
        },
        config={"include_dm": True, "include_private_channels": True},
    )
    connector.id = _CONNECTOR_ID
    connector.oauth_tokens = []
    return connector


def _token_value(token: FederatedConnectorOAuthToken | None) -> str | None:
    if token is None or token.token is None:
        return None
    return token.token.get_value(apply_mask=False)


class _SlackTokenResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.status_code = 200
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _slack_token_payload(access_token: str) -> dict[str, Any]:
    # Shape consumed by SlackFederatedConnector.callback / _exchange_code_for_token.
    # authed_user.access_token is the user token of the person who approved
    # the Slack consent screen (Slack oauth.v2.access).
    return {
        "ok": True,
        "team": {"id": "TPLACEHOLDER", "name": "Placeholder Team"},
        "authed_user": {
            "id": "UPLACEHOLDER",
            "scope": "search:read,im:read,im:history",
            "access_token": access_token,
            "token_type": "user",
            "expires_in": 3600,
        },
    }


class _CallbackHarness:
    def __init__(self, decoy: User) -> None:
        self.connector = _connector()
        self.session = _TokenSession(self.connector)
        self.cache = _RebindingCache(str(decoy.id))
        self.slack_posts: list[tuple[str, str | None]] = []
        self._tokens_by_code: dict[str, str] = {}
        self._actor: dict[str, User] = {}
        self.app = FastAPI()
        self.app.include_router(federated_router)
        self.app.dependency_overrides[current_user] = self._current_user
        self.app.dependency_overrides[get_session] = self._db_session
        self.client = TestClient(self.app)

    def _current_user(self) -> User:
        return self._actor["user"]

    def _db_session(self) -> Iterator[_TokenSession]:
        yield self.session

    def _post_slack(self, url: str, data: dict[str, str] | None = None, **_: Any) -> _SlackTokenResponse:
        code = None if data is None else data.get("code")
        self.slack_posts.append((url, code))
        access_token = self._tokens_by_code.get(code or "", _VICTIM_SLACK_TOKEN)
        return _SlackTokenResponse(_slack_token_payload(access_token))

    def act_as(self, user: User) -> None:
        self._actor["user"] = user

    def state_from_authorize(self, user: User) -> str:
        self.act_as(user)
        response = self.client.get(f"/federated/{_CONNECTOR_ID}/authorize")
        assert response.status_code == 200, response.text
        query = parse_qs(urlparse(response.json()["authorize_url"]).query)
        return query["state"][0]

    def state_from_oauth_status(self, user: User) -> str:
        self.act_as(user)
        response = self.client.get("/federated/oauth-status")
        assert response.status_code == 200, response.text
        body = response.json()
        assert len(body) == 1
        authorize_url = body[0]["authorize_url"]
        assert authorize_url
        query = parse_qs(urlparse(authorize_url).query)
        return query["state"][0]

    def callback(
        self,
        user: User,
        state: str,
        code: str,
        access_token: str,
        query_user_id: str,
    ) -> Any:
        self._tokens_by_code[code] = access_token
        self.act_as(user)
        with patch(
            "onyx.federated_connectors.slack.federated_connector.requests.post",
            side_effect=self._post_slack,
        ):
            return self.client.post(
                "/federated/callback",
                params={
                    "code": code,
                    "state": state,
                    "user_id": query_user_id,
                },
            )

    def stored(self, user_id: UUID) -> FederatedConnectorOAuthToken | None:
        return get_federated_connector_oauth_token(
            self.session,  # type: ignore[arg-type]
            _CONNECTOR_ID,
            user_id,
        )

    def search_token_for(self, user: User) -> str | None:
        tool = SearchTool.__new__(SearchTool)
        tool.user = user
        tool.slack_context = None
        access_token, _bot_token, _entities = tool._prefetch_slack_data(
            self.session  # type: ignore[arg-type]
        )
        return access_token


def _run_slack_search(access_token: str) -> tuple[list[str], list[str]]:
    """Run query_slack through slack_sdk and capture the bearer token it sends."""
    from onyx.context.search.federated.slack_search import query_slack

    authorizations: list[str] = []

    def _http(self: Any, *, url: str, args: dict[str, Any]) -> dict[str, Any]:
        del self, url
        headers = args.get("headers") or {}
        authorizations.append(str(headers.get("Authorization", "")))
        body = json.dumps(
            {
                "ok": True,
                "messages": {
                    "matches": [
                        {
                            "text": "placeholder direct message",
                            "permalink": "https://placeholder.example/archives/CPLACEHOLDER/p1700000000000100",
                            "ts": "1700000000.000100",
                            "channel": {
                                "id": "CPLACEHOLDER",
                                "name": "placeholder-channel",
                            },
                            "username": "placeholder-user",
                            "score": 1.0,
                        }
                    ]
                },
            }
        )
        return {"status": 200, "headers": {}, "body": body}

    with patch(
        "slack_sdk.web.base_client.BaseClient._perform_urllib_http_request",
        _http,
    ):
        result = query_slack("placeholder query", access_token)

    texts = [message.text for message in result.messages]
    return authorizations, texts


@pytest.fixture()
def harness() -> Iterator[_CallbackHarness]:
    decoy = _user("decoy")
    built = _CallbackHarness(decoy)
    with patch(
        "onyx.federated_connectors.oauth_utils.get_cache_backend",
        return_value=built.cache,
    ):
        yield built


def test_authorize_callback_does_not_keep_victim_token_for_state_owner(
    harness: _CallbackHarness,
) -> None:
    attacker = _user("attacker")
    victim = _user("victim")
    update_federated_connector_oauth_token(
        harness.session,  # type: ignore[arg-type]
        federated_connector_id=_CONNECTOR_ID,
        user_id=attacker.id,
        token=_ATTACKER_EXISTING_TOKEN,
    )

    state = harness.state_from_authorize(attacker)
    response = harness.callback(
        user=victim,
        state=state,
        code=_AUTH_CODE,
        access_token=_VICTIM_SLACK_TOKEN,
        query_user_id=str(attacker.id),
    )

    assert _token_value(harness.stored(attacker.id)) == _ATTACKER_EXISTING_TOKEN
    assert harness.stored(victim.id) is None
    assert harness.stored(UUID(harness.cache.decoy_user_id)) is None
    assert harness.slack_posts == []
    assert response.status_code == 403
    assert harness.search_token_for(attacker) == _ATTACKER_EXISTING_TOKEN
    assert harness.search_token_for(victim) is None


def test_oauth_status_callback_does_not_keep_victim_token_for_state_owner(
    harness: _CallbackHarness,
) -> None:
    attacker = _user("attacker")
    victim = _user("victim")

    state = harness.state_from_oauth_status(attacker)
    response = harness.callback(
        user=victim,
        state=state,
        code=_AUTH_CODE,
        access_token=_VICTIM_SLACK_TOKEN,
        query_user_id=str(attacker.id),
    )

    assert harness.stored(attacker.id) is None
    assert harness.stored(victim.id) is None
    assert harness.stored(UUID(harness.cache.decoy_user_id)) is None
    assert harness.slack_posts == []
    assert response.status_code == 403
    assert harness.search_token_for(attacker) is None


def test_same_user_callback_stores_token_for_authenticated_caller(
    harness: _CallbackHarness,
) -> None:
    owner = _user("owner")
    state = harness.state_from_authorize(owner)
    response = harness.callback(
        user=owner,
        state=state,
        code=_AUTH_CODE,
        access_token=_OWNER_SLACK_TOKEN,
        query_user_id=harness.cache.decoy_user_id,
    )

    assert response.status_code == 200
    assert _token_value(harness.stored(owner.id)) == _OWNER_SLACK_TOKEN
    assert harness.stored(UUID(harness.cache.decoy_user_id)) is None
    assert harness.slack_posts == [(_SLACK_TOKEN_URL, _AUTH_CODE)]

    access_token = harness.search_token_for(owner)
    assert access_token == _OWNER_SLACK_TOKEN
    authorizations, texts = _run_slack_search(access_token)
    assert authorizations == [f"Bearer {_OWNER_SLACK_TOKEN}"]
    assert any("placeholder direct message" in text for text in texts)
