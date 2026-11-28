"""Credentialed CORS exposure when CORS_ALLOWED_ORIGIN is left unset.

These tests are the spec for the default API CORS policy. They fail while an
unset or empty CORS_ALLOWED_ORIGIN is treated as allow-all and the API server
installs Starlette's CORSMiddleware with allow_credentials. A sibling origin
can then read credentialed responses, including preflighted POST, PUT, PATCH,
and DELETE.

They pass once an origin the operator did not list no longer receives
Access-Control-Allow-Origin set to itself together with
Access-Control-Allow-Credentials. Same-origin UI traffic does not need that
pair. An explicit allowlist must still reflect only the listed origins, and a
configured value of "*" must be rejected while credentials are enabled.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import AsyncIterator
from collections.abc import Iterator
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import Response

# Sibling of a typical Onyx host. Not listed unless a test sets the allowlist.
ATTACKER_ORIGIN = "https://docs.company.com"
OTHER_ATTACKER_ORIGIN = "https://marketing.company.com"
TRUSTED_ORIGIN = "https://app.partner.example"
# Placeholder session cookie. Starlette's CORS middleware only checks that a
# Cookie header is present; the value is not a real credential.
SESSION_COOKIE = "fastapiusersauth=placeholder-session"

# Handlers that do not open Redis or Postgres. They still run inside the CORS
# middleware get_application installs for every route, including the chat and
# admin routes listed in PREFLIGHT_TARGETS.
SIMPLE_CALLS: list[tuple[str, str]] = [
    ("GET", "/health"),
    ("GET", "/version"),
    ("POST", "/health"),
    ("PUT", "/health"),
    ("PATCH", "/version"),
    ("DELETE", "/version"),
]
# Real state-changing routes. Preflight is answered by CORSMiddleware before
# the handler, so these do not need a live session store.
PREFLIGHT_TARGETS: list[tuple[str, str]] = [
    ("POST", "/manage/admin/deletion-attempt"),
    ("PUT", "/chat/update-chat-session-temperature"),
    ("PATCH", "/manage/set-user-role"),
    ("DELETE", "/chat/delete-all-chat-sessions"),
    ("PUT", "/admin/settings"),
    ("DELETE", "/manage/admin/delete-user"),
]


def _browser_exposes_credentialed_body(response: Response, origin: str) -> bool:
    """True when a browser would reveal this response to `origin` for a
    credentialed fetch. `Access-Control-Allow-Origin: *` does not count:
    the CORS middleware itself refuses to use `*` once a cookie is present.
    """

    allow_origin = response.headers.get("access-control-allow-origin")
    allow_credentials = response.headers.get("access-control-allow-credentials")
    return allow_origin == origin and allow_credentials == "true"


@asynccontextmanager
async def _noop_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield


def _disable_ee_import() -> None:
    """Keep this test on the community application factory.

    License enforcement defaults on, and importing onyx.main then loads the
    EE factory. That factory calls the community get_application, which is
    the function that installs CORS.
    """

    from onyx.utils import variable_functionality

    variable_functionality._LICENSE_ENFORCEMENT_ENABLED = False
    variable_functionality.global_version.unset_ee()
    variable_functionality.fetch_versioned_implementation.cache_clear()


def _apply_cors_env(value: str | None) -> list[str]:
    if value is None:
        os.environ.pop("CORS_ALLOWED_ORIGIN", None)
    else:
        os.environ["CORS_ALLOWED_ORIGIN"] = value

    import shared_configs.configs as configs

    importlib.reload(configs)

    _disable_ee_import()
    import onyx.main as main

    # get_application reads the name bound in onyx.main, not the reloaded module.
    main.CORS_ALLOWED_ORIGIN = configs.CORS_ALLOWED_ORIGIN
    return list(configs.CORS_ALLOWED_ORIGIN)


def _reset_prometheus_registry() -> None:
    """get_application registers process-global metrics. A second app in the
    same interpreter otherwise dies on duplicate timeseries, before CORS runs.
    """

    from prometheus_client import REGISTRY

    for collector in list(REGISTRY._collector_to_names):
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            continue


def _build_client(value: str | None) -> TestClient:
    _apply_cors_env(value)
    _reset_prometheus_registry()
    import onyx.main as main

    application = main.get_application(lifespan_override=_noop_lifespan)
    return TestClient(application)


@pytest.fixture()
def restore_cors_config() -> Iterator[None]:
    import shared_configs.configs as configs

    saved_env = os.environ.get("CORS_ALLOWED_ORIGIN")
    yield
    if saved_env is None:
        os.environ.pop("CORS_ALLOWED_ORIGIN", None)
    else:
        os.environ["CORS_ALLOWED_ORIGIN"] = saved_env
    importlib.reload(configs)
    try:
        import onyx.main as main
    except ModuleNotFoundError:
        return
    main.CORS_ALLOWED_ORIGIN = configs.CORS_ALLOWED_ORIGIN


@pytest.fixture(scope="module")
def default_client() -> Iterator[TestClient]:
    _disable_ee_import()
    saved_env = os.environ.get("CORS_ALLOWED_ORIGIN")
    os.environ.pop("CORS_ALLOWED_ORIGIN", None)
    client = _build_client(None)
    try:
        yield client
    finally:
        if saved_env is None:
            os.environ.pop("CORS_ALLOWED_ORIGIN", None)
        else:
            os.environ["CORS_ALLOWED_ORIGIN"] = saved_env
        _apply_cors_env(saved_env)


@pytest.fixture(scope="module")
def empty_client() -> Iterator[TestClient]:
    _disable_ee_import()
    saved_env = os.environ.get("CORS_ALLOWED_ORIGIN")
    client = _build_client("")
    try:
        yield client
    finally:
        if saved_env is None:
            os.environ.pop("CORS_ALLOWED_ORIGIN", None)
        else:
            os.environ["CORS_ALLOWED_ORIGIN"] = saved_env
        _apply_cors_env(saved_env)


@pytest.fixture(scope="module")
def allowlist_client() -> Iterator[TestClient]:
    _disable_ee_import()
    saved_env = os.environ.get("CORS_ALLOWED_ORIGIN")
    client = _build_client(TRUSTED_ORIGIN)
    try:
        yield client
    finally:
        if saved_env is None:
            os.environ.pop("CORS_ALLOWED_ORIGIN", None)
        else:
            os.environ["CORS_ALLOWED_ORIGIN"] = saved_env
        _apply_cors_env(saved_env)


def _credentialed_headers(origin: str) -> dict[str, str]:
    return {
        "Origin": origin,
        "Cookie": SESSION_COOKIE,
    }


class TestUnsetCorsDoesNotExposeCredentialedResponses:
    def test_unset_config_is_not_a_wildcard(self, restore_cors_config: None) -> None:
        origins = _apply_cors_env(None)
        assert origins != ["*"]
        assert "*" not in origins

    def test_empty_config_is_not_a_wildcard(self, restore_cors_config: None) -> None:
        origins = _apply_cors_env("")
        assert origins != ["*"]
        assert "*" not in origins

    def test_sensitive_routes_share_the_app(self, default_client: TestClient) -> None:
        paths = {getattr(route, "path", None) for route in default_client.app.routes}
        assert "/chat/get-user-chat-sessions" in paths
        assert "/chat/delete-all-chat-sessions" in paths
        assert "/manage/admin/deletion-attempt" in paths
        assert "/manage/admin/delete-user" in paths
        assert "/admin/settings" in paths

    @pytest.mark.parametrize(("method", "path"), SIMPLE_CALLS)
    def test_unset_cookie_request_is_not_readable(
        self,
        default_client: TestClient,
        method: str,
        path: str,
    ) -> None:
        response = default_client.request(
            method,
            path,
            headers=_credentialed_headers(ATTACKER_ORIGIN),
        )
        assert not _browser_exposes_credentialed_body(response, ATTACKER_ORIGIN)

    @pytest.mark.parametrize(("method", "path"), SIMPLE_CALLS)
    def test_empty_cookie_request_is_not_readable(
        self,
        empty_client: TestClient,
        method: str,
        path: str,
    ) -> None:
        response = empty_client.request(
            method,
            path,
            headers=_credentialed_headers(OTHER_ATTACKER_ORIGIN),
        )
        assert not _browser_exposes_credentialed_body(response, OTHER_ATTACKER_ORIGIN)

    @pytest.mark.parametrize(("requested_method", "path"), PREFLIGHT_TARGETS)
    def test_unset_preflight_is_not_credentialed(
        self,
        default_client: TestClient,
        requested_method: str,
        path: str,
    ) -> None:
        response = default_client.options(
            path,
            headers={
                "Origin": ATTACKER_ORIGIN,
                "Access-Control-Request-Method": requested_method,
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert not _browser_exposes_credentialed_body(response, ATTACKER_ORIGIN)

    @pytest.mark.parametrize(("requested_method", "path"), PREFLIGHT_TARGETS)
    def test_empty_preflight_is_not_credentialed(
        self,
        empty_client: TestClient,
        requested_method: str,
        path: str,
    ) -> None:
        response = empty_client.options(
            path,
            headers={
                "Origin": OTHER_ATTACKER_ORIGIN,
                "Access-Control-Request-Method": requested_method,
                "Access-Control-Request-Headers": "content-type, authorization",
            },
        )
        assert not _browser_exposes_credentialed_body(response, OTHER_ATTACKER_ORIGIN)

    def test_health_is_behind_the_same_policy(self, default_client: TestClient) -> None:
        """Public routes use the middleware installed for the whole API."""

        response = default_client.get(
            "/health",
            headers=_credentialed_headers(ATTACKER_ORIGIN),
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert not _browser_exposes_credentialed_body(response, ATTACKER_ORIGIN)


class TestExplicitCorsAllowlist:
    def test_listed_origin_is_credentialed(self, allowlist_client: TestClient) -> None:
        response = allowlist_client.get(
            "/health",
            headers=_credentialed_headers(TRUSTED_ORIGIN),
        )
        assert response.status_code == 200
        assert _browser_exposes_credentialed_body(response, TRUSTED_ORIGIN)

    def test_unlisted_origin_is_not_credentialed(
        self, allowlist_client: TestClient
    ) -> None:
        response = allowlist_client.get(
            "/version",
            headers=_credentialed_headers(ATTACKER_ORIGIN),
        )
        assert response.status_code == 200
        assert not _browser_exposes_credentialed_body(response, ATTACKER_ORIGIN)

    def test_unlisted_preflight_is_not_credentialed(
        self, allowlist_client: TestClient
    ) -> None:
        response = allowlist_client.options(
            "/manage/admin/delete-user",
            headers={
                "Origin": ATTACKER_ORIGIN,
                "Access-Control-Request-Method": "DELETE",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert not _browser_exposes_credentialed_body(response, ATTACKER_ORIGIN)

    def test_star_is_rejected_with_credentials(
        self, restore_cors_config: None
    ) -> None:
        with pytest.raises(ValueError):
            _apply_cors_env("*")
