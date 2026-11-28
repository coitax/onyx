from datetime import datetime
from datetime import timezone

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Response
from fastapi import UploadFile
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from ee.onyx.db.oauth_account import refresh_linked_oauth_account
from ee.onyx.db.scim import ScimDAL
from ee.onyx.server.enterprise_settings.models import AnalyticsScriptUpload
from ee.onyx.server.enterprise_settings.models import EnterpriseSettings
from ee.onyx.server.enterprise_settings.store import get_logo_filename
from ee.onyx.server.enterprise_settings.store import get_logotype_filename
from ee.onyx.server.enterprise_settings.store import load_analytics_script
from ee.onyx.server.enterprise_settings.store import load_settings
from ee.onyx.server.enterprise_settings.store import store_analytics_script
from ee.onyx.server.enterprise_settings.store import store_settings
from ee.onyx.server.enterprise_settings.store import upload_logo
from ee.onyx.server.scim.auth import generate_scim_token
from ee.onyx.server.scim.models import ScimTokenCreate
from ee.onyx.server.scim.models import ScimTokenCreatedResponse
from ee.onyx.server.scim.models import ScimTokenResponse
from onyx.auth.permissions import require_permission
from onyx.auth.users import current_user_with_expired_token
from onyx.configs.app_configs import TRACK_EXTERNAL_IDP_EXPIRY
from onyx.db.engine.async_sql_engine import get_async_session
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.models import User
from onyx.file_store.file_store import get_default_file_store
from onyx.server.utils import BasicAuthenticationError
from onyx.utils.logger import setup_logger
from shared_configs.configs import MULTI_TENANT
from shared_configs.configs import POSTGRES_DEFAULT_SCHEMA
from shared_configs.contextvars import get_current_tenant_id

admin_router = APIRouter(prefix="/admin/enterprise-settings")
basic_router = APIRouter(prefix="/enterprise-settings")

logger = setup_logger()

_CUSTOM_REFRESH_OAUTH_NAME = "custom"


class RefreshTokenData(BaseModel):
    access_token: str
    refresh_token: str
    session: dict = Field(..., description="Contains session information")
    userinfo: dict = Field(..., description="Contains user information")

    @model_validator(mode="after")
    def _validate_session_and_userinfo(self) -> "RefreshTokenData":
        exp = self.session.get("exp")
        if isinstance(exp, bool) or not isinstance(exp, (int, float)):
            raise ValueError("'exp' must be set to a number in the session dictionary")
        # OAuthAccount.expires_at is a 32-bit integer column.
        if not 0 <= exp / 1000 < 2**31:
            raise ValueError("'exp' is not a valid timestamp")
        if not isinstance(self.userinfo.get("userId"), str) or not isinstance(
            self.userinfo.get("email"), str
        ):
            raise ValueError(
                "'userId' and 'email' must be set to strings in the userinfo dictionary"
            )
        return self

    @property
    def expiry(self) -> datetime:
        return datetime.fromtimestamp(self.session["exp"] / 1000, tz=timezone.utc)


@basic_router.post("/refresh-token")
async def refresh_access_token(
    refresh_token: RefreshTokenData,
    user: User = Depends(current_user_with_expired_token),
    db_session: AsyncSession = Depends(get_async_session),
) -> None:
    # The body is client-supplied and is not verified with any IdP, so it may
    # only refresh an OAuth account that is already linked to the caller.
    user_id = user.id

    if refresh_token.userinfo["email"].lower() != user.email.lower():
        raise BasicAuthenticationError(
            detail="Refresh token does not belong to the current user",
        )

    new_expiry = refresh_token.expiry
    refreshed = await refresh_linked_oauth_account(
        db_session=db_session,
        user_id=user_id,
        oauth_name=_CUSTOM_REFRESH_OAUTH_NAME,
        account_id=refresh_token.userinfo["userId"],
        access_token=refresh_token.access_token,
        refresh_token=refresh_token.refresh_token,
        expires_at=int(new_expiry.timestamp()),
        # Matches oauth_callback: clear a stale expiry when tracking is off.
        oidc_expiry=new_expiry if TRACK_EXTERNAL_IDP_EXPIRY else None,
    )
    if not refreshed:
        logger.warning(
            f"Rejected token refresh for user {user_id}: no linked account matches"
        )
        raise BasicAuthenticationError(
            detail="Refresh token does not match an account linked to the current user",
        )

    logger.info(f"Successfully refreshed tokens for user {user_id}")


@admin_router.put("")
def admin_ee_put_settings(
    settings: EnterpriseSettings,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
) -> None:
    store_settings(settings)


@basic_router.get("")
def ee_fetch_settings() -> EnterpriseSettings:
    if MULTI_TENANT:
        tenant_id = get_current_tenant_id()
        if not tenant_id or tenant_id == POSTGRES_DEFAULT_SCHEMA:
            raise BasicAuthenticationError(detail="User must authenticate")

    return load_settings()


@admin_router.put("/logo")
def put_logo(
    file: UploadFile,
    is_logotype: bool = False,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
) -> None:
    upload_logo(file=file, is_logotype=is_logotype)


def fetch_logo_helper(db_session: Session) -> Response:  # noqa: ARG001
    try:
        file_store = get_default_file_store()
        onyx_file = file_store.get_file_with_mime_type(get_logo_filename())
        if not onyx_file:
            raise ValueError("get_onyx_file returned None!")
    except Exception:
        logger.exception("Faield to fetch logo file")
        raise HTTPException(
            status_code=404,
            detail="No logo file found",
        )
    else:
        return Response(
            content=onyx_file.data,
            media_type=onyx_file.mime_type,
            headers={"Cache-Control": "no-cache"},
        )


def fetch_logotype_helper(db_session: Session) -> Response:  # noqa: ARG001
    try:
        file_store = get_default_file_store()
        onyx_file = file_store.get_file_with_mime_type(get_logotype_filename())
        if not onyx_file:
            raise ValueError("get_onyx_file returned None!")
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="No logotype file found",
        )
    else:
        return Response(content=onyx_file.data, media_type=onyx_file.mime_type)


@basic_router.get("/logotype")
def fetch_logotype(db_session: Session = Depends(get_session)) -> Response:
    return fetch_logotype_helper(db_session)


@basic_router.get("/logo")
def fetch_logo(
    is_logotype: bool = False, db_session: Session = Depends(get_session)
) -> Response:
    if is_logotype:
        return fetch_logotype_helper(db_session)

    return fetch_logo_helper(db_session)


@admin_router.put("/custom-analytics-script")
def upload_custom_analytics_script(
    script_upload: AnalyticsScriptUpload,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
) -> None:
    try:
        store_analytics_script(script_upload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@basic_router.get("/custom-analytics-script")
def fetch_custom_analytics_script() -> str | None:
    return load_analytics_script()


# ---------------------------------------------------------------------------
# SCIM token management
# ---------------------------------------------------------------------------


def _get_scim_dal(db_session: Session = Depends(get_session)) -> ScimDAL:
    return ScimDAL(db_session)


@admin_router.get("/scim/token")
def get_active_scim_token(
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    dal: ScimDAL = Depends(_get_scim_dal),
) -> ScimTokenResponse:
    """Return the currently active SCIM token's metadata, or 404 if none."""
    token = dal.get_active_token()
    if not token:
        raise HTTPException(status_code=404, detail="No active SCIM token")

    # Derive the IdP domain from the first synced user as a heuristic.
    idp_domain: str | None = None
    mappings, _total = dal.list_user_mappings(start_index=1, count=1)
    if mappings:
        user = dal.get_user(mappings[0].user_id)
        if user and "@" in user.email:
            idp_domain = user.email.rsplit("@", 1)[1]

    return ScimTokenResponse(
        id=token.id,
        name=token.name,
        token_display=token.token_display,
        is_active=token.is_active,
        created_at=token.created_at,
        last_used_at=token.last_used_at,
        idp_domain=idp_domain,
    )


@admin_router.post("/scim/token", status_code=201)
def create_scim_token(
    body: ScimTokenCreate,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    dal: ScimDAL = Depends(_get_scim_dal),
) -> ScimTokenCreatedResponse:
    """Create a new SCIM bearer token.

    Only one token is active at a time — creating a new token automatically
    revokes all previous tokens. The raw token value is returned exactly once
    in the response; it cannot be retrieved again.
    """
    raw_token, hashed_token, token_display = generate_scim_token()
    token = dal.create_token(
        name=body.name,
        hashed_token=hashed_token,
        token_display=token_display,
        created_by_id=user.id,
    )
    dal.commit()

    return ScimTokenCreatedResponse(
        id=token.id,
        name=token.name,
        token_display=token.token_display,
        is_active=token.is_active,
        created_at=token.created_at,
        last_used_at=token.last_used_at,
        raw_token=raw_token,
    )
