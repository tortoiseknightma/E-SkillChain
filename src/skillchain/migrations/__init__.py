"""Explicit schema migrations; migrations never mutate source artifacts."""

from skillchain.migrations.query_v1_to_v2 import (
    QueryMigrationContext,
    migrate_query_v1_to_v2,
)

__all__ = ["QueryMigrationContext", "migrate_query_v1_to_v2"]
