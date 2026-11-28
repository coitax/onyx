"""resync user files linked to non owner personas

Personas no longer grant access to user files owned by someone other than the
persona's owner. Flag every user file that is linked to such a persona so the
persona sync task rewrites its access and persona ids in the document index.

Revision ID: 68945eb13f59
Revises: 503883791c39
Create Date: 2026-11-28 04:09:56.207484

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "68945eb13f59"
down_revision = "503883791c39"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            UPDATE user_file
            SET needs_persona_sync = TRUE
            WHERE id IN (
                SELECT persona__user_file.user_file_id
                FROM persona__user_file
                JOIN persona ON persona.id = persona__user_file.persona_id
                JOIN user_file AS linked_file
                    ON linked_file.id = persona__user_file.user_file_id
                WHERE persona.user_id IS DISTINCT FROM linked_file.user_id
            )
            """
        )
    )


def downgrade() -> None:
    pass
