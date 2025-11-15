"""add_is_active_column

Revision ID: c8dd9ef3ac05
Revises: 001
Create Date: 2025-11-15 14:04:44.484727

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8dd9ef3ac05"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add is_active column to memory_states
    op.add_column(
        "memory_states", sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true")
    )
    op.create_index("ix_memory_states_is_active", "memory_states", ["is_active"])

    # Add is_active column to topic_latest_states
    op.add_column(
        "topic_latest_states",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
    )


def downgrade() -> None:
    # Remove from topic_latest_states
    op.drop_column("topic_latest_states", "is_active")

    # Remove from memory_states
    op.drop_index("ix_memory_states_is_active", "memory_states")
    op.drop_column("memory_states", "is_active")
