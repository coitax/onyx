from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from onyx.configs.constants import KV_GENERATED_USER_AUTH_SECRET_KEY
from onyx.db.models import KVStore

_SECRET_FIELD = "secret"


def get_or_create_generated_user_auth_secret(
    db_session: Session, candidate_secret: str
) -> str:
    """Persist `candidate_secret` unless a generated secret already exists, then
    return whichever value is stored.

    The insert is a no-op on conflict, so concurrent first callers across
    processes and replicas all end up with the first committed value.
    """
    db_session.execute(
        insert(KVStore)
        .values(
            key=KV_GENERATED_USER_AUTH_SECRET_KEY,
            encrypted_value={_SECRET_FIELD: candidate_secret},
        )
        .on_conflict_do_nothing(index_elements=[KVStore.key])
    )
    db_session.commit()

    stored = db_session.scalar(
        select(KVStore.encrypted_value).where(
            KVStore.key == KV_GENERATED_USER_AUTH_SECRET_KEY
        )
    )
    value = stored.get_value(apply_mask=False) if stored is not None else None
    secret = value.get(_SECRET_FIELD) if isinstance(value, dict) else None
    if not isinstance(secret, str) or not secret.strip():
        raise RuntimeError(
            f"Stored '{KV_GENERATED_USER_AUTH_SECRET_KEY}' entry is missing or empty"
        )
    return secret
