"""Allow ordered immutable Generation Manifest shards."""

from __future__ import annotations

from alembic import op

revision = "20260713_0002"
down_revision = "20260712_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE whetstone_experiment_operation_manifests "
        "ADD COLUMN IF NOT EXISTS accepted_generation_ordinal INTEGER"
    )
    op.execute(
        "WITH ranked AS (SELECT ctid, row_number() OVER (PARTITION BY "
        "experiment_name ORDER BY accepted_at, operation_key) AS ordinal "
        "FROM whetstone_experiment_operation_manifests WHERE "
        "workflow_role='generation') UPDATE "
        "whetstone_experiment_operation_manifests AS relationship SET "
        "accepted_generation_ordinal=ranked.ordinal FROM ranked WHERE "
        "relationship.ctid=ranked.ctid"
    )
    op.execute("DROP INDEX IF EXISTS uq_whetstone_one_generation_manifest")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS "
        "uq_whetstone_generation_manifest_ordinal ON "
        "whetstone_experiment_operation_manifests "
        "(experiment_name, accepted_generation_ordinal) WHERE "
        "workflow_role='generation'"
    )
    op.execute(
        "ALTER TABLE whetstone_experiment_acceptance_evaluations "
        "ADD COLUMN IF NOT EXISTS generation_relationships JSONB NOT NULL "
        "DEFAULT '[]'::jsonb"
    )
    op.execute(
        "ALTER TABLE whetstone_experiment_acceptance_evaluations "
        "ADD COLUMN IF NOT EXISTS generation_relationships_digest TEXT NOT "
        "NULL DEFAULT ''"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_whetstone_generation_manifest_ordinal")
    op.execute(
        "CREATE UNIQUE INDEX uq_whetstone_one_generation_manifest ON "
        "whetstone_experiment_operation_manifests (experiment_name) WHERE "
        "workflow_role='generation'"
    )
    op.execute(
        "ALTER TABLE whetstone_experiment_acceptance_evaluations DROP COLUMN "
        "IF EXISTS generation_relationships_digest"
    )
    op.execute(
        "ALTER TABLE whetstone_experiment_acceptance_evaluations DROP COLUMN "
        "IF EXISTS generation_relationships"
    )
    op.execute(
        "ALTER TABLE whetstone_experiment_operation_manifests DROP COLUMN IF "
        "EXISTS accepted_generation_ordinal"
    )
