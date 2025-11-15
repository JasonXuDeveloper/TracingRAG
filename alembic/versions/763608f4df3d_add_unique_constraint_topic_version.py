"""add_unique_constraint_topic_version

Revision ID: 763608f4df3d
Revises: c8dd9ef3ac05
Create Date: 2025-11-15 15:31:45.448010

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "763608f4df3d"
down_revision: str | None = "c8dd9ef3ac05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add unique constraint on (topic, version) to prevent concurrent creation of duplicate versions
    op.create_unique_constraint(
        "uq_memory_states_topic_version", "memory_states", ["topic", "version"]
    )


def downgrade() -> None:
    # Drop unique constraint
    op.drop_constraint("uq_memory_states_topic_version", "memory_states")
