"""add user_id to oauth_config

Records the user who created an OAuth config. Existing configs keep a NULL
owner, which means only admins can modify them.

Revision ID: 28470c33d7cf
Revises: 503883791c39
Create Date: 2026-11-28 04:08:58.095558

"""

from alembic import op
import fastapi_users_db_sqlalchemy
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "28470c33d7cf"
down_revision = "503883791c39"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "oauth_config",
        sa.Column(
            "user_id",
            fastapi_users_db_sqlalchemy.generics.GUID(),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "oauth_config_user_id_fkey",
        "oauth_config",
        "user",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("oauth_config_user_id_fkey", "oauth_config", type_="foreignkey")
    op.drop_column("oauth_config", "user_id")
