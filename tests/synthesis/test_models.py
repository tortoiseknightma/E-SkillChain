from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from skillchain.schemas import ConversationTurn, Query
from skillchain.synthesis.models import (
    ActivePlanPointer,
    BatchDraftManifest,
    BatchManifest,
    CorpusPlan,
    GeneratedTrajectory,
    PlannedQuery,
    PROVISIONAL_LEAKAGE_POLICY_VERSION,
)


BASE_QUERY = {
    "schema_version": 2,
    "taxonomy_version": "ecommerce-coarse-v1",
    "task_spec_version": "unfrozen-v0",
    "query_id": "dm-001",
    "asset_id": "asset.dm-001",
    "image_path": "query_images/exact_match/muge-1.jpg",
    "leakage_group_id": "leakage.dm-001",
    "template_family": "dev-mini-001/exact_match/v1",
    "generator_batch_id": "dev-mini-001",
    "text": "机械占位-01",
    "canonical_intent": "exact_match",
    "canonical_capability": "product.exact_match",
    "acceptable_capabilities": ["product.exact_match"],
    "requires_card": True,
    "split": "dev_mini",
    "label_provenance": [
        {
            "decision_type": "constructed",
            "annotator_kind": "planner",
            "annotator_id": "test-planner",
            "canonical_intent": "exact_match",
            "canonical_capability": "product.exact_match",
            "acceptable_capabilities": ["product.exact_match"],
        }
    ],
}

PROVENANCE = {
    "synth_provider": "codex",
    "synth_model": "5.6 Sol Ultra",
    "synthesis_batch_id": "dev-mini-001-r1",
    "synthesis_prompt_id": "dm-001",
    "seed_set_sha256": "a" * 64,
}


def test_v2_query_without_synthesis_fields_stays_valid():
    query = Query(
        **BASE_QUERY,
        turns=[{"role": "user", "content": "机械占位-01"}],
    )

    assert len(query.turns) == 1
    assert query.synth_provider is None
    assert query.seed_set_sha256 is None


@pytest.mark.parametrize(
    "turns",
    [
        [{"role": "user", "content": "机械占位-01"}],
        [
            {"role": "user", "content": "机械占位-00"},
            {"role": "assistant", "content": "机械应答-00"},
            {"role": "user", "content": "机械占位-01"},
        ],
    ],
)
def test_synthesized_query_accepts_only_canonical_trajectory_shapes(turns):
    query = Query(**BASE_QUERY, turns=turns, **PROVENANCE)

    assert query.turns[-1] == ConversationTurn(role="user", content=query.text)


@pytest.mark.parametrize(
    "turns",
    [
        [
            {"role": "user", "content": "机械占位-00"},
            {"role": "user", "content": "机械占位-01"},
        ],
        [{"role": "assistant", "content": "机械应答-00"}],
        [
            {"role": "user", "content": "机械占位-00"},
            {"role": "assistant", "content": "机械应答-00"},
        ],
    ],
)
def test_synthesized_query_rejects_noncanonical_turn_order(turns):
    with pytest.raises(
        ValidationError, match="\\[user\\].*\\[user, assistant, user\\]"
    ):
        Query(**BASE_QUERY, turns=turns, **PROVENANCE)


def test_synthesized_query_rejects_text_that_differs_from_final_user_turn():
    with pytest.raises(ValidationError, match="最后一个 user"):
        Query(
            **{**BASE_QUERY, "text": "机械占位-不一致"},
            turns=[{"role": "user", "content": "机械占位-01"}],
            **PROVENANCE,
        )


@pytest.mark.parametrize(
    "extra",
    [
        {
            "turns": [{"role": "user", "content": "机械占位-01"}],
            "synth_provider": "codex",
        },
        {
            "turns": [{"role": "user", "content": "机械占位-01"}],
            **{
                key: value
                for key, value in PROVENANCE.items()
                if key != "seed_set_sha256"
            },
        },
    ],
)
def test_synthesis_provenance_is_all_or_none(extra):
    with pytest.raises(ValidationError, match="合成溯源字段必须同时提供"):
        Query(**BASE_QUERY, **extra)


def test_conversation_turn_rejects_blank_content():
    with pytest.raises(ValidationError, match="content"):
        ConversationTurn(role="user", content="   ")


def test_generated_trajectory_uses_same_canonical_role_shapes():
    trajectory = GeneratedTrajectory(
        plan_id="dm-001",
        turns=[
            {"role": "user", "content": "机械占位-00"},
            {"role": "assistant", "content": "机械应答-00"},
            {"role": "user", "content": "机械占位-01"},
        ],
    )

    assert trajectory.turns[-1].content == "机械占位-01"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"is_boundary": False, "boundary_strategy": "natural_ambiguity"},
        {"is_boundary": True},
        {
            "is_boundary": True,
            "boundary_strategy": "cross_intent_triplet",
            "boundary_group_id": None,
        },
        {
            "is_boundary": True,
            "boundary_strategy": "natural_ambiguity",
            "boundary_group_id": "group-1",
        },
    ],
)
def test_planned_query_rejects_inconsistent_boundary_fields(kwargs):
    values = {
        "plan_id": "dm-001",
        "batch_id": "dev-mini-001",
        "position": 1,
        "taxonomy_version": "ecommerce-coarse-v1",
        "task_spec_version": "unfrozen-v0",
        "asset_id": "asset.dm-001",
        "image_path": "query_images/exact_match/muge-1.jpg",
        "leakage_group_id": "leakage.dm-001",
        "template_family": "dev-mini-001/exact_match/v1",
        "generator_batch_id": "dev-mini-001",
        "canonical_intent": "exact_match",
        "canonical_capability": "product.exact_match",
        "acceptable_capabilities": ["product.exact_match"],
        "requires_card": True,
        "is_boundary": False,
        "provisional_split": "dev_mini",
    }
    values.update(kwargs)

    with pytest.raises(ValidationError, match="boundary"):
        PlannedQuery(**values)


def _planned_query(index: int, split: str = "dev_mini") -> PlannedQuery:
    return PlannedQuery(
        plan_id=f"dm-{index:03d}",
        batch_id=f"dev-mini-{((index - 1) // 25) + 1:03d}",
        position=((index - 1) % 25) + 1,
        taxonomy_version="ecommerce-coarse-v1",
        task_spec_version="unfrozen-v0",
        asset_id=f"asset.dm-{index:03d}",
        image_path=f"query_images/exact_match/image-{index:03d}.jpg",
        leakage_group_id=f"leakage.dm-{index:03d}",
        template_family=(f"dev-mini-{((index - 1) // 25) + 1:03d}/exact_match/v1"),
        generator_batch_id=f"dev-mini-{((index - 1) // 25) + 1:03d}",
        canonical_intent="exact_match",
        canonical_capability="product.exact_match",
        acceptable_capabilities=["product.exact_match"],
        requires_card=True,
        is_boundary=False,
        provisional_split=split,
    )


def test_corpus_plan_enforces_scope_size_and_provisional_splits():
    dev_plan = CorpusPlan(
        scope="dev_mini",
        seed=20260711,
        queries=[_planned_query(index) for index in range(1, 201)],
    )
    assert len(dev_plan.queries) == 200

    with pytest.raises(ValidationError, match="dev_mini"):
        CorpusPlan(
            scope="dev_mini",
            seed=20260711,
            queries=[_planned_query(index) for index in range(1, 200)],
        )


@pytest.mark.parametrize("fault", ["duplicate_plan_id", "duplicate_position"])
def test_corpus_plan_rejects_ambiguous_batch_identity(fault):
    queries = [_planned_query(index) for index in range(1, 201)]
    if fault == "duplicate_plan_id":
        queries[1] = queries[1].model_copy(update={"plan_id": queries[0].plan_id})
    else:
        queries[1] = queries[1].model_copy(update={"position": queries[0].position})

    with pytest.raises(ValidationError, match="plan_id|position"):
        CorpusPlan(scope="dev_mini", seed=20260711, queries=queries)


def test_manifest_hashes_require_64_lowercase_hex_characters():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError, match="plan_sha256"):
        BatchManifest(
            base_batch_id="dev-mini-001",
            revision=1,
            data_origin="synthetic_derived",
            provider="codex",
            model_display_name="5.6 Sol Ultra",
            model_claim_source="user_confirmation",
            generated_at=now,
            staged_at=now,
            count=25,
            plan_sha256="not-a-sha",
            seed_set_sha256="a" * 64,
            results_sha256="b" * 64,
        )


def _batch_draft_manifest_payload() -> dict:
    return {
        "schema_version": 2,
        "base_batch_id": "dev-mini-001",
        "data_origin": "synthetic_derived",
        "provider": "codex",
        "model_display_name": "5.6 Sol Ultra",
        "model_claim_source": "user_confirmation",
        "generated_at": datetime.now(timezone.utc),
        "plan_sha256": "a" * 64,
        "asset_catalog_sha256": None,
        "leakage_policy_version": PROVISIONAL_LEAKAGE_POLICY_VERSION,
        "seed_set_sha256": "b" * 64,
        "draft_sha256": "c" * 64,
        "generation_input_sha256": "d" * 64,
    }


@pytest.mark.parametrize(
    "field",
    [
        "plan_sha256",
        "asset_catalog_sha256",
        "leakage_policy_version",
        "seed_set_sha256",
        "draft_sha256",
        "generation_input_sha256",
        "data_origin",
    ],
)
def test_batch_draft_manifest_v2_requires_every_generation_binding(field):
    payload = _batch_draft_manifest_payload()
    payload.pop(field)

    with pytest.raises(ValidationError, match=field):
        BatchDraftManifest.model_validate(payload)


def test_batch_draft_manifest_v2_supports_explicit_provisional_binding():
    manifest = BatchDraftManifest.model_validate(_batch_draft_manifest_payload())

    assert manifest.schema_version == 2
    assert manifest.asset_catalog_sha256 is None


def test_batch_draft_manifest_v2_rejects_legacy_or_incoherent_catalog_binding():
    legacy = _batch_draft_manifest_payload()
    legacy["schema_version"] = 1
    with pytest.raises(ValidationError, match="schema_version"):
        BatchDraftManifest.model_validate(legacy)

    incoherent = _batch_draft_manifest_payload()
    incoherent["asset_catalog_sha256"] = "e" * 64
    with pytest.raises(ValidationError, match="provisional"):
        BatchDraftManifest.model_validate(incoherent)


def test_active_plan_pointer_forbids_unknown_fields():
    with pytest.raises(ValidationError, match="extra"):
        ActivePlanPointer(
            scope="dev_mini",
            plan_path="plans/dev_mini.json",
            plan_sha256="a" * 64,
            unexpected=True,
        )
