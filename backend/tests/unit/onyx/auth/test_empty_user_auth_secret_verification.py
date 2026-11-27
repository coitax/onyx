"""Forged email-verification tokens must not verify an account when the
signing secret is the Helm default.

Helm leaves ``auth.userauth.enabled`` false, so the API process never
receives ``USER_AUTH_SECRET``. ``app_configs`` then uses the empty default
and only logs a warning. Verification tokens are HS256 JWTs over the user
id, email, and the public verify audience. These tests sign that token the
way an outsider can and require that ``UserManager.verify`` and
``POST /auth/verify`` leave the account unverified.

The tests fail while that acceptance still happens. A fix passes them by
refusing this configuration, or by rejecting the forged token before any
write that sets ``is_verified``.

Not covered here, and why:
- Password reset uses the same secret, but the token also carries a
  fingerprint of the stored password hash. Registration does not return
  that hash, so an outsider cannot complete reset with this secret alone.
- OAuth/OIDC state is signed with the same secret, but the callback still
  exchanges an identity-provider authorization code before it logs anyone in.
- The JWT session backend already refuses to start when the secret is empty.
  Helm's default session backend is Redis, whose session tokens are not
  signed with this secret.
- Anonymous-user cookies signed with this secret are only decoded when
  multi-tenant mode is on. Helm does not turn that on.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import traceback
import uuid
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any


_BACKEND_ROOT = Path(__file__).resolve().parents[4]
_RESULT_REJECTED = "RESULT=rejected"
_RESULT_VERIFIED = "RESULT=verified"
_RESULT_STARTUP_REFUSED = "RESULT=startup_refused"
_RESULT_ERROR = "RESULT=error"


def _attacker_known_default_secret() -> str:
    """Value used when Helm does not inject USER_AUTH_SECRET.

    This is the same lookup as ``onyx.configs.app_configs``. The variable is
    absent in this process; the default is the absence of a secret, not a
    credential stored in the repo.
    """
    if "USER_AUTH_SECRET" in os.environ:
        raise RuntimeError("USER_AUTH_SECRET must be unset to reproduce Helm")
    return os.environ.get("USER_AUTH_SECRET", "")


def _is_secret_refusal(exc: BaseException) -> bool:
    text = " ".join(part for part in (str(exc), str(exc.__cause__ or ""), type(exc).__name__) if part).lower()
    return "user_auth_secret" in text or "user auth secret" in text


class _RegisteredUser:
    """In-memory stand-in for the row registration would have inserted.

    Token validation never reads the database row except for id, email, and
    is_verified. Those fields are the ones ``BaseUserManager.verify`` uses.
    """

    def __init__(self, email: str) -> None:
        from onyx.auth.schemas import UserRole
        from onyx.db.enums import AccountType

        self.id = uuid.uuid4()
        self.email = email
        self.is_active = True
        self.is_superuser = False
        self.is_verified = False
        self.role = UserRole.BASIC
        self.account_type = AccountType.STANDARD
        self.oidc_expiry = None
        self.hashed_password = "placeholder-password-hash"


class _MemoryUserDB:
    def __init__(self, user: _RegisteredUser) -> None:
        self.user = user
        self.updates: list[dict[str, Any]] = []

    async def get_by_email(self, email: str) -> _RegisteredUser | None:
        if email == self.user.email:
            return self.user
        return None

    async def get(self, user_id: uuid.UUID) -> _RegisteredUser | None:
        if user_id == self.user.id:
            return self.user
        return None

    async def update(self, user: _RegisteredUser, update_dict: dict[str, Any]) -> _RegisteredUser:
        self.updates.append(dict(update_dict))
        for key, value in update_dict.items():
            setattr(user, key, value)
        return user


def _marked_verified(user: _RegisteredUser, store: _MemoryUserDB) -> bool:
    if user.is_verified:
        return True
    return any(update.get("is_verified") is True for update in store.updates)


def _forge_verification_token(user_id: str, email: str, audience: str, lifetime: int) -> str:
    """Sign the verification claims with the known default secret.

    Mirrors ``BaseUserManager.request_verify``: subject, email, verify
    audience, and an expiry. The signature is HS256, which is what
    ``fastapi_users.jwt.generate_jwt`` asks PyJWT to use.
    """
    import jwt

    payload = {
        "sub": user_id,
        "email": email,
        "aud": audience,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=lifetime),
    }
    return jwt.encode(payload, _attacker_known_default_secret(), algorithm="HS256")


def _bind_library_lookup(manager: Any) -> None:
    """Use fastapi-users' own email lookup instead of the database override.

    ``UserManager.get_by_email`` only selects the tenant and loads the row.
    ``BaseUserManager.verify`` is what checks the token. The library lookup
    is the same check: load by email, or raise if the row is missing.
    """
    from fastapi_users import BaseUserManager

    async def _get_by_email(user_email: str) -> Any:
        return await BaseUserManager.get_by_email(manager, user_email)

    manager.get_by_email = _get_by_email


async def _direct_verify_accepts_forged_token() -> bool:
    from onyx.auth.users import UserManager

    user = _RegisteredUser("direct-target@example.com")
    store = _MemoryUserDB(user)
    manager = UserManager(store)
    _bind_library_lookup(manager)

    token = _forge_verification_token(
        str(user.id),
        user.email,
        manager.verification_token_audience,
        manager.verification_token_lifetime_seconds,
    )
    try:
        await manager.verify(token)
    except Exception:
        pass
    accepted = _marked_verified(user, store)
    if accepted:
        print(f"direct_access_granted={await _access_granted_after_verification(user)}")
    return accepted


async def _http_verify_accepts_forged_token() -> bool:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from onyx.auth.schemas import UserCreate
    from onyx.auth.schemas import UserRead
    from onyx.auth.users import fastapi_users
    from onyx.auth.users import get_user_manager
    from onyx.auth.users import UserManager

    user = _RegisteredUser("http-target@example.com")
    store = _MemoryUserDB(user)
    manager = UserManager(store)
    _bind_library_lookup(manager)

    async def _create(user_create: Any, safe: bool = False, request: Any = None) -> Any:
        del user_create, safe, request
        return user

    manager.create = _create

    async def _yield_manager() -> Any:
        yield manager

    app = FastAPI()
    app.include_router(
        fastapi_users.get_register_router(UserRead, UserCreate),
        prefix="/auth",
    )
    app.include_router(
        fastapi_users.get_verify_router(UserRead),
        prefix="/auth",
    )
    app.dependency_overrides[get_user_manager] = _yield_manager

    with TestClient(app) as client:
        registered = client.post(
            "/auth/register",
            json={
                "email": user.email,
                "password": uuid.uuid4().hex,
            },
        )
        if registered.status_code != 201:
            raise RuntimeError(f"registration did not return the new user: {registered.status_code} {registered.text}")
        body = registered.json()
        if body.get("id") != str(user.id) or body.get("email") != user.email:
            raise RuntimeError(f"registration response hid id or email: {body}")
        if body.get("is_verified") is not False:
            raise RuntimeError("registration created an already-verified user")

        token = _forge_verification_token(
            body["id"],
            body["email"],
            manager.verification_token_audience,
            manager.verification_token_lifetime_seconds,
        )
        client.post("/auth/verify", json={"token": token})

    accepted = _marked_verified(user, store)
    if accepted:
        print(f"http_access_granted={await _access_granted_after_verification(user)}")
    return accepted


async def _access_granted_after_verification(user: _RegisteredUser) -> bool:
    """The gate ``current_user`` uses once ``is_verified`` is true."""
    from onyx.auth.users import double_check_user
    from onyx.db.users import is_limited_user

    allowed = await double_check_user(user)
    return allowed.email == user.email and not is_limited_user(allowed)


async def _run_attack() -> int:
    try:
        from onyx.auth.users import UserManager
    except Exception as exc:
        if _is_secret_refusal(exc):
            print(_RESULT_STARTUP_REFUSED)
            print(f"refusal={type(exc).__name__}")
            return 0
        traceback.print_exc()
        print(_RESULT_ERROR)
        return 2

    server_secret_empty = not UserManager.verification_token_secret
    print(f"server_secret_empty={server_secret_empty}")

    try:
        direct_accepted = await _direct_verify_accepts_forged_token()
        http_accepted = await _http_verify_accepts_forged_token()
    except Exception as exc:
        if _is_secret_refusal(exc):
            print(_RESULT_STARTUP_REFUSED)
            print(f"refusal={type(exc).__name__}")
            return 0
        traceback.print_exc()
        print(_RESULT_ERROR)
        return 2

    print(f"direct_accepted={direct_accepted}")
    print(f"http_accepted={http_accepted}")
    if direct_accepted or http_accepted:
        print(_RESULT_VERIFIED)
        return 1
    print(_RESULT_REJECTED)
    return 0


def _child_main() -> int:
    try:
        return asyncio.run(_run_attack())
    except Exception as exc:
        if _is_secret_refusal(exc):
            print(_RESULT_STARTUP_REFUSED)
            print(f"refusal={type(exc).__name__}")
            return 0
        traceback.print_exc()
        print(_RESULT_ERROR)
        return 2


def test_helm_default_empty_secret_cannot_verify_unowned_email() -> None:
    """An outsider who can register must not be able to verify that account.

    The child process unsets ``USER_AUTH_SECRET``, which is the Helm default,
    then signs a verification token and submits it to the product verifier.
    """
    env = os.environ.copy()
    env.pop("USER_AUTH_SECRET", None)
    env["AUTH_TYPE"] = "basic"
    env["AUTH_BACKEND"] = "redis"
    env["REQUIRE_EMAIL_VERIFICATION"] = "true"
    env["PYTHONPATH"] = str(_BACKEND_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--run-attack"],
        cwd=str(_BACKEND_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, output
    assert _RESULT_VERIFIED not in completed.stdout, output
    assert _RESULT_REJECTED in completed.stdout or _RESULT_STARTUP_REFUSED in completed.stdout, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-attack", action="store_true")
    args = parser.parse_args()
    if args.run_attack:
        sys.exit(_child_main())
