"""Focused contract tests for Query schema v2 and its explicit v1 migration."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from skillchain.migrations import QueryMigrationContext, migrate_query_v1_to_v2
from skillchain.schemas import Query, QueryV1


def _v2_payload() -> dict:
    capability = "product.exact_match"
    return {
        "schema_version": 2,
        "taxonomy_version": "ecommerce-test-v1",
        "task_spec_version": "task-spec-test-v1",
        "query_id": "query-001",
        "asset_id": "asset-001",
        "image_path": "query_images/exact_match/muge-001.jpg",
        "leakage_group_id": "leakage-001",
        "boundary_group_id": None,
        "template_family": "exact-match-single-turn",
        "generator_batch_id": "batch-001",
        "text": "请帮我找到图片中的同款商品。",
        "turns": [{"role": "user", "content": "请帮我找到图片中的同款商品。"}],
        "canonical_intent": "exact_match",
        "canonical_capability": capability,
        "acceptable_capabilities": [capability],
        "is_boundary": False,
        "boundary_strategy": None,
        "requires_card": True,
        "episode": "t0",
        "split": "dev_mini",
        "label_status": "cross_agreed",
        "label_provenance": [
            {
                "decision_type": "cross_review",
                "annotator_kind": "llm_pair",
                "annotator_id": "review-pair-test-v1",
                "canonical_intent": "exact_match",
                "canonical_capability": capability,
                "acceptable_capabilities": [capability],
                "reason": "两位独立审核者确认测试标签。",
                "legacy_skill_slug": None,
                "source_artifact_sha256": "a" * 64,
                "decided_at": "2026-07-19T12:00:00Z",
            }
        ],
        "synth_provider": None,
        "synth_model": None,
        "synthesis_batch_id": None,
        "synthesis_prompt_id": None,
        "seed_set_sha256": None,
    }


def _legacy_query(*, gt_skill: str | None = "bank-v1-exact-match") -> QueryV1:
    return QueryV1(
        schema_version=1,
        query_id="legacy-001",
        image_path="query_images/exact_match/muge-001.jpg",
        text="请帮我找到图片中的同款商品。",
        gt_intent="exact_match",
        gt_skill=gt_skill,
        split="opt_pool",
    )


def _migration_context() -> QueryMigrationContext:
    return QueryMigrationContext(
        taxonomy_version="ecommerce-test-v1",
        task_spec_version="task-spec-test-v1",
        asset_id="asset-001",
        leakage_group_id="leakage-001",
        template_family="exact-match-single-turn",
        generator_batch_id="legacy-batch-001",
        boundary_strategy=None,
        boundary_group_id=None,
        canonical_capability="product.exact_match",
        acceptable_capabilities=("product.exact_match",),
        requires_card=True,
        source_artifact_sha256="b" * 64,
        annotator_id="query-v1-to-v2-test",
        reason="显式测试迁移；legacy skill slug 仅供审计。",
        decided_at=datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize("schema_version", [None, 1, 3, "2"])
def test_query_v2_rejects_missing_or_incorrect_schema_version(schema_version):
    payload = _v2_payload()
    if schema_version is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = schema_version

    with pytest.raises(ValidationError):
        Query.model_validate(payload)


@pytest.mark.parametrize(
    ("legacy_field", "legacy_value"),
    [
        ("gt_skill", "bank-v1-exact-match"),
        ("gt_intent", "exact_match"),
    ],
)
def test_query_v2_rejects_extra_legacy_fields(legacy_field, legacy_value):
    payload = _v2_payload()
    payload[legacy_field] = legacy_value

    with pytest.raises(ValidationError, match=legacy_field):
        Query.model_validate(payload)


def test_v1_to_v2_migration_requires_explicit_context():
    legacy = _legacy_query()

    with pytest.raises(TypeError):
        migrate_query_v1_to_v2(legacy)  # type: ignore[call-arg]


def test_query_v1_payload_is_not_implicitly_accepted_as_v2():
    legacy_payload = _legacy_query().model_dump(mode="json")

    with pytest.raises(ValidationError):
        Query.model_validate(legacy_payload)


def test_v1_gt_skill_is_audited_but_never_becomes_canonical_capability():
    legacy_slug = "bank-v1-exact-match"
    migrated = migrate_query_v1_to_v2(
        _legacy_query(gt_skill=legacy_slug),
        _migration_context(),
    )

    assert migrated.canonical_capability == "product.exact_match"
    assert migrated.canonical_capability != legacy_slug
    assert migrated.label_provenance[-1].legacy_skill_slug == legacy_slug
    assert "gt_skill" not in migrated.model_dump(mode="json")


def test_migration_downgrades_legacy_review_status_until_v2_revalidation():
    legacy = _legacy_query().model_copy(update={"label_status": "cross_agreed"})

    migrated = migrate_query_v1_to_v2(legacy, _migration_context())

    assert migrated.label_status == "auto"
    assert migrated.label_provenance[-1].decision_type == "migrated_unverified"


def test_query_v2_json_round_trip_uses_only_canonical_label_names():
    original = Query.model_validate(_v2_payload())

    restored = Query.model_validate_json(original.model_dump_json())
    serialized = restored.model_dump(mode="json")

    assert restored == original
    assert serialized["schema_version"] == 2
    assert serialized["canonical_intent"] == "exact_match"
    assert serialized["canonical_capability"] == "product.exact_match"
    assert "gt_intent" not in serialized
    assert "gt_skill" not in serialized
