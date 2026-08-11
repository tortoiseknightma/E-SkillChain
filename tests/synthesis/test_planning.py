import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from skillchain.synthesis.batches import corpus_status, show_next
from skillchain.synthesis.planning import (
    CORE_BOUNDARY_COUNTS,
    CORE_INTENT_COUNTS,
    activate_full_plan,
    build_dev_mini_plan,
    extend_core_plan,
    extend_full_plan,
    write_core_plan,
    write_dev_mini_plan,
    write_full_plan,
)


EXPECTED_INTENT_COUNTS = {
    "exact_match": 35,
    "multi_product": 35,
    "divergent_rec": 35,
    "encyclopedia": 35,
    "utility": 60,
}

EXPECTED_CAPABILITY_COUNTS = {
    "product.exact_match": 35,
    "product.multi_search": 35,
    "product.style_recommendation": 35,
    "knowledge.visual_encyclopedia": 35,
    "utility.document_reading": 30,
    "utility.recipe_guidance": 30,
}


def test_dev_mini_plan_has_exact_counts_and_boundary_groups(fake_image_root):
    plan = build_dev_mini_plan(
        fake_image_root, seed=20260711, allow_provisional_asset_groups=True
    )

    assert plan.scope == "dev_mini"
    assert len(plan.queries) == 200
    assert Counter(item.gt_intent for item in plan.queries) == EXPECTED_INTENT_COUNTS
    assert Counter(item.canonical_capability for item in plan.queries) == (
        EXPECTED_CAPABILITY_COUNTS
    )
    assert Counter(item.batch_id for item in plan.queries) == {
        f"dev-mini-{index:03d}": 25 for index in range(1, 9)
    }
    assert Counter(item.boundary_strategy for item in plan.queries) == {
        None: 160,
        "cross_intent_triplet": 24,
        "natural_ambiguity": 16,
    }

    triplets = defaultdict(list)
    for item in plan.queries:
        if item.boundary_strategy == "cross_intent_triplet":
            triplets[item.boundary_group_id].append(item)
    assert len(triplets) == 8
    assert all(len(items) == 3 for items in triplets.values())
    assert all(
        len({item.image_path for item in items}) == 1 for items in triplets.values()
    )
    assert all(
        {item.gt_intent for item in items}
        == {"exact_match", "divergent_rec", "encyclopedia"}
        for items in triplets.values()
    )


def test_dev_mini_plan_is_deterministic_and_caps_image_intent_variants(fake_image_root):
    first = build_dev_mini_plan(
        fake_image_root, seed=20260711, allow_provisional_asset_groups=True
    )
    second = build_dev_mini_plan(
        fake_image_root, seed=20260711, allow_provisional_asset_groups=True
    )

    assert first == second
    assert [item.plan_id for item in first.queries] == [
        f"dm-{index:03d}" for index in range(1, 201)
    ]
    assert all(
        [
            item.position
            for item in first.queries
            if item.batch_id == f"dev-mini-{batch:03d}"
        ]
        == list(range(1, 26))
        for batch in range(1, 9)
    )
    pair_counts = Counter((item.image_path, item.gt_intent) for item in first.queries)
    assert max(pair_counts.values()) <= 3


def test_dev_mini_plan_rejects_missing_or_too_small_pool(fake_image_root):
    for path in list((fake_image_root / "utility").glob("*.jpg"))[5:]:
        path.unlink()

    with pytest.raises(ValueError, match="utility.*图片不足"):
        build_dev_mini_plan(
            fake_image_root, seed=20260711, allow_provisional_asset_groups=True
        )


def test_write_dev_mini_plan_records_hash_and_refuses_different_existing_bytes(
    fake_image_root, tmp_path
):
    plan = build_dev_mini_plan(
        fake_image_root, seed=20260711, allow_provisional_asset_groups=True
    )
    output = tmp_path / "plans" / "dev_mini.json"

    plan_path, manifest_path = write_dev_mini_plan(plan, output)

    plan_bytes = plan_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "count": 200,
        "asset_catalog_sha256": None,
        "capability_assignments_sha256": None,
        "leakage_policy_version": "relative-path-plus-plan-groups-v1",
        "parent_plan_sha256": None,
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "query_schema_version": 2,
        "schema_version": 2,
        "scope": "dev_mini",
    }
    assert write_dev_mini_plan(plan, output) == (plan_path, manifest_path)

    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        write_dev_mini_plan(plan, output)


def test_full_plan_preserves_dev_mini_and_adds_4300_slots(fake_image_root):
    dev_plan = build_dev_mini_plan(
        fake_image_root, seed=20260711, allow_provisional_asset_groups=True
    )
    image_pools = {
        intent: [
            Path(f"query_images/{intent}/full-{index:04d}.jpg")
            for index in range(1, 401)
        ]
        for intent in (
            "exact_match",
            "multi_product",
            "divergent_rec",
            "encyclopedia",
            "utility",
        )
    }

    full_plan = extend_full_plan(
        dev_plan,
        image_pools,
        seed=20260711,
        allow_provisional_asset_groups=True,
    )

    assert full_plan.queries[:200] == dev_plan.queries
    assert len(full_plan.queries) == 4500
    assert Counter(item.gt_intent for item in full_plan.queries) == {
        "exact_match": 1125,
        "multi_product": 675,
        "divergent_rec": 900,
        "encyclopedia": 900,
        "utility": 900,
    }
    assert sum(item.is_boundary for item in full_plan.queries) == 788
    assert len({item.batch_id for item in full_plan.queries}) == 180
    assert full_plan.queries[200].batch_id == "full-009"
    pair_counts = Counter(
        (item.image_path, item.gt_intent) for item in full_plan.queries
    )
    assert max(pair_counts.values()) <= 3
    appended = full_plan.queries[200:]
    assert Counter(item.generator_batch_id for item in appended) == {
        f"full-{index:03d}": 25 for index in range(9, 181)
    }
    assert not (
        {item.leakage_group_id for item in dev_plan.queries}
        & {item.leakage_group_id for item in appended}
    )
    batches_by_leakage_group = defaultdict(set)
    batches_by_template_family = defaultdict(set)
    for item in appended:
        batches_by_leakage_group[item.leakage_group_id].add(item.generator_batch_id)
        batches_by_template_family[item.template_family].add(item.generator_batch_id)
    assert all(len(batch_ids) == 1 for batch_ids in batches_by_leakage_group.values())
    assert all(len(batch_ids) == 2 for batch_ids in batches_by_template_family.values())
    assert all(
        not family.startswith(tuple(batches))
        for family, batches in batches_by_template_family.items()
    )


def test_core_plan_preserves_dev_mini_and_adds_1300_slots(fake_image_root):
    dev_plan = build_dev_mini_plan(
        fake_image_root, seed=20260711, allow_provisional_asset_groups=True
    )
    image_pools = {
        intent: [
            Path(f"query_images/{intent}/core-{index:04d}.jpg")
            for index in range(1, 401)
        ]
        for intent in EXPECTED_INTENT_COUNTS
    }

    core_plan = extend_core_plan(
        dev_plan,
        image_pools,
        seed=20260711,
        allow_provisional_asset_groups=True,
    )

    assert core_plan.scope == "core"
    assert core_plan.queries[:200] == dev_plan.queries
    assert len(core_plan.queries) == 1500
    assert Counter(item.gt_intent for item in core_plan.queries) == CORE_INTENT_COUNTS
    assert sum(item.is_boundary for item in core_plan.queries) == sum(
        CORE_BOUNDARY_COUNTS.values()
    )
    assert len({item.batch_id for item in core_plan.queries}) == 60
    assert core_plan.queries[200].batch_id == "core-009"
    assert Counter(item.generator_batch_id for item in core_plan.queries[200:]) == {
        f"core-{index:03d}": 25 for index in range(9, 61)
    }
    pair_counts = Counter(
        (item.image_path, item.gt_intent) for item in core_plan.queries
    )
    assert max(pair_counts.values()) <= 3
    assert not (
        {item.leakage_group_id for item in dev_plan.queries}
        & {item.leakage_group_id for item in core_plan.queries[200:]}
    )


def test_write_core_plan_records_parent_hash(fake_image_root, tmp_path):
    dev_plan = build_dev_mini_plan(
        fake_image_root, seed=20260711, allow_provisional_asset_groups=True
    )
    dev_path = tmp_path / "dev_mini.json"
    _, dev_manifest_path = write_dev_mini_plan(dev_plan, dev_path)
    parent_hash = json.loads(dev_manifest_path.read_text("utf-8"))["plan_sha256"]
    pools = {
        intent: [
            Path(f"query_images/{intent}/core-{index:04d}.jpg")
            for index in range(400)
        ]
        for intent in EXPECTED_INTENT_COUNTS
    }
    core_plan = extend_core_plan(
        dev_plan,
        pools,
        seed=20260711,
        allow_provisional_asset_groups=True,
    )

    _, manifest_path = write_core_plan(
        core_plan,
        tmp_path / "core.json",
        parent_plan_sha256=parent_hash,
    )

    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["scope"] == "core"
    assert manifest["count"] == 1500
    assert manifest["parent_plan_sha256"] == parent_hash


def test_write_full_plan_records_parent_hash(fake_image_root, tmp_path):
    dev_plan = build_dev_mini_plan(
        fake_image_root, seed=20260711, allow_provisional_asset_groups=True
    )
    dev_path = tmp_path / "dev_mini.json"
    _, dev_manifest_path = write_dev_mini_plan(dev_plan, dev_path)
    parent_hash = json.loads(dev_manifest_path.read_text("utf-8"))["plan_sha256"]
    pools = {
        intent: [
            Path(f"query_images/{intent}/full-{index:04d}.jpg") for index in range(400)
        ]
        for intent in EXPECTED_INTENT_COUNTS
    }
    full_plan = extend_full_plan(
        dev_plan,
        pools,
        seed=20260711,
        allow_provisional_asset_groups=True,
    )

    _, manifest_path = write_full_plan(
        full_plan,
        tmp_path / "full.json",
        parent_plan_sha256=parent_hash,
    )

    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["scope"] == "full"
    assert manifest["count"] == 4500
    assert manifest["parent_plan_sha256"] == parent_hash


def test_full_activation_resumes_at_batch_nine(full_activation_fixture):
    active = activate_full_plan(
        full_activation_fixture.queries_root,
        full_activation_fixture.full_plan_path,
        allow_provisional_asset_groups=True,
    )

    status = corpus_status(full_activation_fixture.queries_root)
    assert active.scope == "full"
    assert status["next_batch_id"] == "full-009"
    assert status["active_plan_sha256"] == active.plan_sha256
    assert (
        show_next(full_activation_fixture.queries_root)["plan_sha256"]
        == active.plan_sha256
    )


@pytest.mark.parametrize(
    "fault",
    ["too_few_accepted", "parent_hash_mismatch", "pending_staging"],
)
def test_full_activation_gate_is_failure_atomic(full_activation_fixture, fault):
    before = full_activation_fixture.active_path.read_bytes()
    full_activation_fixture.inject(fault)

    with pytest.raises(ValueError, match="前 8 批|parent|staging"):
        activate_full_plan(
            full_activation_fixture.queries_root,
            full_activation_fixture.full_plan_path,
            allow_provisional_asset_groups=True,
        )

    assert full_activation_fixture.active_path.read_bytes() == before


def test_full_activation_idempotent_retry_rechecks_accepted_gate(
    full_activation_fixture,
):
    activate_full_plan(
        full_activation_fixture.queries_root,
        full_activation_fixture.full_plan_path,
        allow_provisional_asset_groups=True,
    )
    active_before = full_activation_fixture.active_path.read_bytes()
    full_activation_fixture.inject("too_few_accepted")

    with pytest.raises(ValueError, match="前 8 批"):
        activate_full_plan(
            full_activation_fixture.queries_root,
            full_activation_fixture.full_plan_path,
            allow_provisional_asset_groups=True,
        )

    assert full_activation_fixture.active_path.read_bytes() == active_before


def test_full_activation_retry_does_not_depend_on_parent_plan_path(
    full_activation_fixture,
):
    activate_full_plan(
        full_activation_fixture.queries_root,
        full_activation_fixture.full_plan_path,
        allow_provisional_asset_groups=True,
    )
    dev_path = full_activation_fixture.queries_root / "plans/dev_mini.json"
    dev_path.unlink()
    dev_path.with_name("dev_mini.manifest.json").unlink()

    retried = activate_full_plan(
        full_activation_fixture.queries_root,
        full_activation_fixture.full_plan_path,
        allow_provisional_asset_groups=True,
    )

    assert retried.scope == "full"
