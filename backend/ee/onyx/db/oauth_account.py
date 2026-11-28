from datetime import datetime
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from onyx.db.models import OAuthAccount
from onyx.db.models import User


async def refresh_linked_oauth_account(
    db_session: AsyncSession,
    user_id: UUID,
    oauth_name: str,
    account_id: str,
    access_token: str,
    refresh_token: str,
    expires_at: int,
    oidc_expiry: datetime | None,
) -> bool:
    """Update tokens on an OAuth account already linked to `user_id`.

    The ownership filter and the write are a single statement, so an account
    linked to any other user can never be selected. Returns False, having
    written nothing, when `user_id` has no matching linked account.
    """
    updated_account_ids = (
        await db_session.execute(
            update(OAuthAccount)
            .where(
                OAuthAccount.user_id == user_id,  # type: ignore[arg-type]
                OAuthAccount.oauth_name == oauth_name,  # type: ignore[arg-type]
                OAuthAccount.account_id == account_id,  # type: ignore[arg-type]
            )
            .values(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=expires_at,
            )
            .returning(OAuthAccount.id)  # type: ignore[call-overload]
        )
    ).all()
    if not updated_account_ids:
        return False

    await db_session.execute(
        update(User)
        .where(User.id == user_id)  # type: ignore[arg-type]
        .values(oidc_expiry=oidc_expiry)
    )
    await db_session.commit()
    return True
