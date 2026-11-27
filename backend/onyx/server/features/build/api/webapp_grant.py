"""Per-viewer grants for loading a Craft webapp's sub-resources.

Webapp documents are served with a CSP sandbox, which gives them an opaque
origin. Browsers send no Onyx cookies on requests made by such a document, so
its asset URLs carry a grant instead. A grant is minted when the document
request itself was authorized, and records which session it is for, which
viewer it was authorized as, and which tenant that happened in.
"""

import re
import secrets
from uuid import UUID

from pydantic import BaseModel

from onyx.redis.redis_pool import get_shared_redis_client
from shared_configs.contextvars import get_current_tenant_id

WEBAPP_GRANT_TTL_SECONDS = 12 * 60 * 60

_GRANT_KEY_PREFIX = "craft_webapp_grant:"
_GRANT_TOKEN_BYTES = 32
_GRANT_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}")


class WebappGrant(BaseModel):
    tenant_id: str
    session_id: UUID
    viewer_id: UUID | None


def mint_webapp_grant(session_id: UUID, viewer_id: UUID | None) -> str:
    token = secrets.token_urlsafe(_GRANT_TOKEN_BYTES)
    grant = WebappGrant(
        tenant_id=get_current_tenant_id(),
        session_id=session_id,
        viewer_id=viewer_id,
    )
    get_shared_redis_client().set(
        f"{_GRANT_KEY_PREFIX}{token}",
        grant.model_dump_json(),
        ex=WEBAPP_GRANT_TTL_SECONDS,
    )
    return token


def load_webapp_grant(token: str) -> WebappGrant | None:
    if not _GRANT_TOKEN_PATTERN.fullmatch(token):
        return None
    raw = get_shared_redis_client().get(f"{_GRANT_KEY_PREFIX}{token}")
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return WebappGrant.model_validate_json(str(raw))
