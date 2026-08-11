"""Audited Query v1 -> v2 migration.

V1 no longer contains enough information to recover a safe group or a
Bank-independent capability.  Callers must therefore join the legacy query to
its plan, asset/group catalog, taxonomy and label sidecars and provide that
context explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from skillchain.schemas import (
    BoundaryStrategy,
    Intent,
    LabelDecision,
    Query,
    QueryV1,
)


@dataclass(frozen=True)
class QueryMigrationContext:
    taxonomy_version: str
    task_spec_version: str
    asset_id: str
    leakage_group_id: str
    template_family: str
    generator_batch_id: str
    boundary_strategy: BoundaryStrategy | None
    boundary_group_id: str | None
    canonical_capability: str | None
    acceptable_capabilities: tuple[str, ...]
    requires_card: bool | None
    source_artifact_sha256: str
    annotator_id: str
    reason: str
    decided_at: datetime | None = None
    canonical_intent: Intent | None = None


def migrate_query_v1_to_v2(
    legacy: QueryV1,
    context: QueryMigrationContext,
) -> Query:
    """Return a new v2 Query; the v1 skill slug is provenance only."""

    canonical_intent = context.canonical_intent or legacy.gt_intent
    decision = LabelDecision(
        decision_type="migrated_unverified",
        annotator_kind="migration",
        annotator_id=context.annotator_id,
        canonical_intent=canonical_intent,
        canonical_capability=context.canonical_capability,
        acceptable_capabilities=list(context.acceptable_capabilities),
        reason=context.reason,
        legacy_skill_slug=legacy.gt_skill,
        source_artifact_sha256=context.source_artifact_sha256,
        decided_at=context.decided_at,
    )
    turns = legacy.turns or [{"role": "user", "content": legacy.text}]
    return Query(
        schema_version=2,
        taxonomy_version=context.taxonomy_version,
        task_spec_version=context.task_spec_version,
        query_id=legacy.query_id,
        asset_id=context.asset_id,
        image_path=legacy.image_path,
        leakage_group_id=context.leakage_group_id,
        boundary_group_id=context.boundary_group_id,
        template_family=context.template_family,
        generator_batch_id=context.generator_batch_id,
        text=legacy.text,
        turns=turns,
        canonical_intent=canonical_intent,
        canonical_capability=context.canonical_capability,
        acceptable_capabilities=list(context.acceptable_capabilities),
        is_boundary=legacy.is_boundary,
        boundary_strategy=context.boundary_strategy,
        requires_card=context.requires_card,
        episode="t0",
        split=legacy.split,
        # A legacy status is not sufficient evidence under the v2 label
        # contract.  Re-review must explicitly promote this migrated record.
        label_status="auto",
        label_provenance=[decision],
        synth_provider=legacy.synth_provider,
        synth_model=legacy.synth_model,
        synthesis_batch_id=legacy.synthesis_batch_id,
        synthesis_prompt_id=legacy.synthesis_prompt_id,
        seed_set_sha256=legacy.seed_set_sha256,
    )
