import asyncio
import json
import threading
import time
from enum import Enum
from typing import Any
from typing import cast

import jwt
import requests
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from jwt import decode as jwt_decode
from jwt import InvalidTokenError
from jwt import PyJWTError
from jwt.algorithms import RSAAlgorithm

from onyx.auth.constants import API_KEY_PREFIX
from onyx.auth.constants import DEPRECATED_API_KEY_PREFIX
from onyx.auth.constants import PAT_PREFIX
from onyx.configs.app_configs import JWT_PUBLIC_KEY_URL
from onyx.utils.logger import setup_logger


logger = setup_logger()


_JWT_ALGORITHM = "RS256"
_NON_JWT_BEARER_PREFIXES = (API_KEY_PREFIX, DEPRECATED_API_KEY_PREFIX, PAT_PREFIX)
_PUBLIC_KEY_FETCH_TIMEOUT_SECONDS = 10.0
# Lower bound between any two fetches of the key URL, so token contents can
# never drive more than one outbound request per interval.
_PUBLIC_KEY_MIN_REFETCH_INTERVAL_SECONDS = 60.0
# Key material is refreshed on this schedule regardless of incoming tokens so
# that rotated keys (including PEM keys, which carry no kid) are picked up.
_PUBLIC_KEY_CACHE_TTL_SECONDS = 3600.0


class PublicKeyFormat(Enum):
    JWKS = "jwks"
    PEM = "pem"


PublicKeyPayload = tuple[str | dict[str, Any], PublicKeyFormat]


def _download_public_key_payload() -> PublicKeyPayload | None:
    """Fetch the raw JWT verification material. Blocking; never call from the event loop."""
    if JWT_PUBLIC_KEY_URL is None:
        logger.error("JWT_PUBLIC_KEY_URL is not set")
        return None

    try:
        response = requests.get(
            JWT_PUBLIC_KEY_URL, timeout=_PUBLIC_KEY_FETCH_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error(f"Failed to fetch JWT public key: {str(exc)}")
        return None
    content_type = response.headers.get("Content-Type", "").lower()
    raw_body = response.text
    body_lstripped = raw_body.lstrip()

    if "application/json" in content_type or body_lstripped.startswith("{"):
        try:
            data = response.json()
        except ValueError:
            logger.error("JWT public key URL returned invalid JSON")
            return None

        if isinstance(data, dict) and "keys" in data:
            return data, PublicKeyFormat.JWKS

        logger.error(
            "JWT public key URL returned JSON but no JWKS 'keys' field was found"
        )
        return None

    body = raw_body.strip()
    if not body:
        logger.error("JWT public key URL returned an empty response")
        return None

    return body, PublicKeyFormat.PEM


class _PublicKeyPayloadCache:
    """Process-wide cache of the key material served at JWT_PUBLIC_KEY_URL.

    Fetches are serialized and spaced at least
    _PUBLIC_KEY_MIN_REFETCH_INTERVAL_SECONDS apart, whether they are caused by
    a cold cache, a failed previous fetch, expiry, or an unknown key id.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (payload, monotonic time of last fetch attempt), swapped atomically
        self._state: tuple[PublicKeyPayload | None, float | None] = (None, None)

    def latest(self) -> PublicKeyPayload | None:
        """Return the last successfully fetched payload, even if stale."""
        return self._state[0]

    def fresh(self) -> PublicKeyPayload | None:
        """Return the cached payload if it has not expired, without blocking."""
        payload, fetched_at = self._state
        if payload is None or fetched_at is None:
            return None
        if time.monotonic() - fetched_at >= _PUBLIC_KEY_CACHE_TTL_SECONDS:
            return None
        return payload

    def refetch_allowed(self) -> bool:
        _, fetched_at = self._state
        return (
            fetched_at is None
            or time.monotonic() - fetched_at >= _PUBLIC_KEY_MIN_REFETCH_INTERVAL_SECONDS
        )

    def __call__(
        self, refresh_for_unknown_key: bool = False
    ) -> PublicKeyPayload | None:
        """Return the key payload, fetching it if allowed. Blocking."""
        with self._lock:
            payload, fetched_at = self._state
            if fetched_at is not None:
                age = time.monotonic() - fetched_at
                if (
                    payload is not None
                    and age < _PUBLIC_KEY_CACHE_TTL_SECONDS
                    and not refresh_for_unknown_key
                ):
                    return payload
                if age < _PUBLIC_KEY_MIN_REFETCH_INTERVAL_SECONDS:
                    return payload

            fetched = _download_public_key_payload()
            # On failure keep serving the previous key material, if any.
            new_payload = fetched if fetched is not None else payload
            self._state = (new_payload, time.monotonic())
            return new_payload

    def cache_clear(self) -> None:
        with self._lock:
            self._state = (None, None)


_fetch_public_key_payload = _PublicKeyPayloadCache()


async def _get_public_key_payload(
    refresh_for_unknown_key: bool = False,
) -> PublicKeyPayload | None:
    if not refresh_for_unknown_key:
        if (payload := _fetch_public_key_payload.fresh()) is not None:
            return payload
    if not _fetch_public_key_payload.refetch_allowed():
        return _fetch_public_key_payload.latest()

    return await asyncio.to_thread(_fetch_public_key_payload, refresh_for_unknown_key)


def _parse_jwt_header(token: str) -> dict[str, Any] | None:
    """Return the unverified header if the token is structurally an RS256 JWT."""
    if token.startswith(_NON_JWT_BEARER_PREFIXES):
        return None

    segments = token.split(".")
    if len(segments) != 3 or not all(segments):
        logger.debug("Bearer token is not a JWT; skipping JWT verification")
        return None

    try:
        header = jwt.get_unverified_header(token)
    except PyJWTError as e:
        logger.debug(f"Unable to parse JWT header: {str(e)}")
        return None

    if header.get("alg") != _JWT_ALGORITHM:
        logger.debug(f"Unsupported JWT algorithm: {header.get('alg')}")
        return None

    return header


def _jwks_keys(payload: PublicKeyPayload) -> list[dict[str, Any]]:
    key_material, _ = payload
    if not isinstance(key_material, dict):
        return []
    keys = key_material.get("keys", [])
    return keys if isinstance(keys, list) else []


def _match_jwk(
    header: dict[str, Any], keys: list[dict[str, Any]]
) -> dict[str, Any] | None:
    kid = header.get("kid")
    thumbprint = header.get("x5t")

    candidates: list[dict[str, Any]] = []
    if kid:
        candidates = [k for k in keys if k.get("kid") == kid]
    if not candidates and thumbprint:
        candidates = [k for k in keys if k.get("x5t") == thumbprint]
    if not candidates:
        return None

    if len(candidates) > 1:
        logger.warning(
            "Multiple JWKs matched token header kid=%s; selecting the first occurrence",
            kid,
        )
    return candidates[0]


async def _resolve_public_key(header: dict[str, Any]) -> RSAPublicKey | str | None:
    """Return the concrete public key used to verify a token with this header."""
    payload = await _get_public_key_payload()
    if payload is None:
        logger.error("Failed to retrieve public key payload")
        return None

    key_material, key_format = payload
    if key_format is PublicKeyFormat.PEM:
        return cast(str, key_material)

    keys = _jwks_keys(payload)
    jwk = _match_jwk(header, keys)
    if jwk is None and (header.get("kid") or header.get("x5t")):
        # The token names a key we don't have; the IdP may have rotated keys.
        refreshed = await _get_public_key_payload(refresh_for_unknown_key=True)
        if refreshed is not None and refreshed[1] is PublicKeyFormat.JWKS:
            keys = _jwks_keys(refreshed)
            jwk = _match_jwk(header, keys)

    if jwk is None and len(keys) == 1:
        jwk = keys[0]

    if jwk is None:
        if not keys:
            logger.error("JWKS payload did not contain any keys")
        else:
            logger.warning(
                "No matching JWK found for token header (kid=%s, x5t=%s)",
                header.get("kid"),
                header.get("x5t"),
            )
        return None

    try:
        return cast(RSAPublicKey, RSAAlgorithm.from_jwk(json.dumps(jwk)))
    except ValueError as e:
        logger.error(f"Failed to construct RSA key from JWK: {str(e)}")
        return None


async def verify_jwt_token(token: str) -> dict[str, Any] | None:
    header = _parse_jwt_header(token)
    if header is None:
        return None

    public_key = await _resolve_public_key(header)
    if public_key is None:
        logger.error("Unable to resolve a public key for JWT verification")
        return None

    try:
        return jwt_decode(
            token,
            public_key,
            algorithms=[_JWT_ALGORITHM],
            options={"verify_aud": False},
        )
    except InvalidTokenError as e:
        logger.error(f"Invalid JWT token: {str(e)}")
        return None
    except PyJWTError as e:
        logger.error(f"JWT decoding error: {str(e)}")
        return None
