import secrets
import threading

from onyx.configs.app_configs import USER_AUTH_SECRET
from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.user_auth_secret import get_or_create_generated_user_auth_secret
from onyx.utils.logger import setup_logger
from shared_configs.configs import MULTI_TENANT
from shared_configs.configs import POSTGRES_DEFAULT_SCHEMA

logger = setup_logger()

_generated_secret_lock = threading.Lock()
_generated_secret: str | None = None


def get_user_auth_secret() -> str:
    """Key for HMAC-signed auth tokens: email verification, password reset,
    OAuth state and anonymous-user cookies.

    Never returns a blank key, since anyone can sign a valid HS256 token with
    one. When USER_AUTH_SECRET is not configured on a single-tenant deployment,
    a random secret is generated once and stored in Postgres so every process
    and replica signs and verifies with the same value.
    """
    if USER_AUTH_SECRET.strip():
        return USER_AUTH_SECRET

    if MULTI_TENANT:
        raise RuntimeError("USER_AUTH_SECRET must be set for multi-tenant deployments")

    global _generated_secret
    with _generated_secret_lock:
        if _generated_secret is None:
            try:
                with get_session_with_tenant(
                    tenant_id=POSTGRES_DEFAULT_SCHEMA
                ) as db_session:
                    _generated_secret = get_or_create_generated_user_auth_secret(
                        db_session, secrets.token_urlsafe(48)
                    )
            except Exception as e:
                raise RuntimeError(
                    "USER_AUTH_SECRET is not set and the generated fallback "
                    "secret could not be loaded from Postgres"
                ) from e
            logger.notice(
                "USER_AUTH_SECRET is not set; using the generated secret stored in Postgres"
            )
        return _generated_secret
