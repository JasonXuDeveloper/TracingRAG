"""Initial schema - memory_states, traces, and topic_latest_states tables

Revision ID: 001
Revises:
Create Date: 2025-11-12 11:36:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create memory_states table
    op.create_table(
        'memory_states',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('topic', sa.String(500), nullable=False, index=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False, default=1),
        sa.Column('timestamp', sa.DateTime(), nullable=False, index=True),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column('parent_state_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=False, default=dict),
        sa.Column('tags', postgresql.ARRAY(sa.String()), nullable=False, default=list),
        sa.Column('confidence', sa.Float(), nullable=False, default=1.0),
        sa.Column('source', sa.String(255), nullable=True),
        sa.Column('created_by', sa.String(255), nullable=True),
        sa.Column('access_count', sa.Integer(), nullable=False, default=0),
        sa.Column('last_accessed', sa.DateTime(), nullable=False),
        sa.Column('importance_score', sa.Float(), nullable=False, default=0.5),
        sa.Column('storage_tier', sa.String(50), nullable=False, default='active'),
        sa.Column('consolidated_from', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('is_consolidated', sa.Boolean(), nullable=False, default=False),
        sa.Column('consolidation_level', sa.Integer(), nullable=False, default=0),
        sa.Column('diff_from_parent', sa.Text(), nullable=True),
        sa.Column('is_delta', sa.Boolean(), nullable=False, default=False),
        sa.Column('entity_type', sa.String(100), nullable=True, index=True),
        sa.Column('entity_schema', postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(['parent_state_id'], ['memory_states.id']),
    )

    # Create indexes for memory_states
    op.create_index('ix_memory_states_topic_version', 'memory_states', ['topic', 'version'])
    op.create_index('ix_memory_states_entity_type', 'memory_states', ['entity_type'])
    op.create_index('ix_memory_states_storage_tier', 'memory_states', ['storage_tier'])
    op.create_index('ix_memory_states_is_consolidated', 'memory_states', ['is_consolidated'])
    op.create_index('ix_memory_states_tags', 'memory_states', ['tags'], postgresql_using='gin')
    op.create_index('ix_memory_states_metadata', 'memory_states', ['metadata'], postgresql_using='gin')

    # Create traces table
    op.create_table(
        'traces',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('topic', sa.String(500), nullable=False, unique=True, index=True),
        sa.Column('state_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False, default=list),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(), nullable=False, default=dict),
        sa.Column('tags', postgresql.ARRAY(sa.String()), nullable=False, default=list),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
    )

    # Create indexes for traces
    op.create_index('ix_traces_is_active', 'traces', ['is_active'])
    op.create_index('ix_traces_tags', 'traces', ['tags'], postgresql_using='gin')

    # Create topic_latest_states table (materialized view pattern)
    op.create_table(
        'topic_latest_states',
        sa.Column('topic', sa.String(500), primary_key=True, index=True),
        sa.Column('latest_state_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['latest_state_id'], ['memory_states.id']),
    )

    # Create trigger function to auto-update topic_latest_states
    op.execute("""
        CREATE OR REPLACE FUNCTION update_topic_latest_state()
        RETURNS TRIGGER AS $$
        BEGIN
            INSERT INTO topic_latest_states (topic, latest_state_id, updated_at)
            VALUES (NEW.topic, NEW.id, NEW.timestamp)
            ON CONFLICT (topic) DO UPDATE
            SET latest_state_id = EXCLUDED.latest_state_id,
                updated_at = EXCLUDED.updated_at
            WHERE EXCLUDED.updated_at > topic_latest_states.updated_at;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Create trigger to call the function after insert on memory_states
    op.execute("""
        CREATE TRIGGER trigger_update_topic_latest_state
        AFTER INSERT ON memory_states
        FOR EACH ROW
        EXECUTE FUNCTION update_topic_latest_state();
    """)


def downgrade() -> None:
    # Drop trigger and function
    op.execute('DROP TRIGGER IF EXISTS trigger_update_topic_latest_state ON memory_states')
    op.execute('DROP FUNCTION IF EXISTS update_topic_latest_state()')

    # Drop tables in reverse order
    op.drop_table('topic_latest_states')
    op.drop_table('traces')
    op.drop_table('memory_states')
