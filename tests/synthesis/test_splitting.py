import hashlib
import json
import os
import stat
from collections import Counter
from pathlib import Path

import pytest

from skillchain.schemas import Query
from skillchain.synthesis.planning import (
    build_dev_mini_plan,
    extend_core_plan,
    write_core_plan,
    write_dev_mini_plan,
)
from skillchain.synthesis.splitting import (
    CORE_SPLIT_SPEC,
    FULL_SPLIT_SPEC,
    FrozenSplitError,
    freeze_test_split,
    load_labeled_queries_for_split,
    stratified_split,
    verify_frozen_split,
)
from conftest import make_query_v2
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)

MECHANICAL_FULL_SPLIT_SPEC = FULL_SPLIT_SPEC.model_copy(
    update={"profile": "mechanical-full-test"}
)


def test_full_split_sizes_and_minimum_test_intents(full_candidate_fixture):
    candidates, locked_dev_ids = full_candidate_fixture

    assigned = stratified_split(
        candidates,
        locked_dev_query_ids=locked_dev_ids,
        seed=20260711,
    )

    counts = Counter(query.split for query in assigned)
    assert counts == {
        "dev_mini": 200,
        "opt_pool": 2800,
        "val": 500,
        "test_frozen": 1000,
    }
    test_counts = Counter(
        query.gt_intent for query in assigned if query.split == "test_frozen"
    )
    assert min(test_counts.values()) >= 150
    assigned_by_id = {query.query_id: query for query in assigned}
    assert all(
        assigned_by_id[query_id].split == "dev_mini" for query_id in locked_dev_ids
    )


def test_core_split_binds_to_a_core_plan(fake_image_root, tmp_path):
    dev_plan = build_dev_mini_plan(
        fake_image_root, seed=20260711, allow_provisional_asset_groups=True
    )
    pools = {
        intent: [
            Path(f"query_images/{intent}/core-{index:04d}.jpg")
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
    core_plan = extend_core_plan(
        dev_plan,
        pools,
        seed=20260711,
        allow_provisional_asset_groups=True,
    )
    dev_path = tmp_path / "plans" / "core-dev-prefix.json"
    _, dev_manifest_path = write_dev_mini_plan(dev_plan, dev_path)
    parent_sha256 = json.loads(dev_manifest_path.read_text("utf-8"))["plan_sha256"]
    core_path, _ = write_core_plan(
        core_plan,
        tmp_path / "plans" / "core.json",
        parent_plan_sha256=parent_sha256,
    )
    candidates = [
        make_query_v2(
            query_id=item.plan_id,
            image_path=item.image_path,
            text=f"mechanical core query {item.plan_id}",
            intent=item.canonical_intent,
            capability=item.canonical_capability,
            requires_card=item.requires_card,
            split=item.provisional_split,
            generator_batch_id=item.generator_batch_id,
            asset_id=item.asset_id,
            leakage_group_id=item.leakage_group_id,
            template_family=item.template_family,
            is_boundary=item.is_boundary,
            boundary_strategy=item.boundary_strategy,
            boundary_group_id=item.boundary_group_id,
        )
        for item in core_plan.queries
    ]
    assigned = stratified_split(
        candidates,
        locked_dev_query_ids={item.plan_id for item in core_plan.queries[:200]},
        seed=20260711,
        split_spec=CORE_SPLIT_SPEC,
    )

    root = tmp_path / "queries"
    freeze_test_split(
        root,
        assigned,
        seed=20260711,
        full_plan_path=core_path,
        allow_provisional_asset_groups=True,
        split_spec=CORE_SPLIT_SPEC,
    )
    manifest = verify_frozen_split(
        root,
        full_plan_path=core_path,
        allow_provisional_asset_groups=True,
    )

    assert manifest is not None
    assert manifest.profile == "core"
    assert manifest.split_sizes == CORE_SPLIT_SPEC.sizes
    assert manifest.count == 300


@pytest.mark.parametrize(
    "fault",
    [
        "missing_query_id",
        "accepted_source_hash_mismatch",
        "unresolved_arbitration",
        "arbitration_label_mismatch",
    ],
)
def test_split_full_requires_complete_reviewed_input(labeled_split_fixture, fault):
    labeled_split_fixture.inject(fault)

    with pytest.raises(ValueError, match="标签|仲裁|哈希"):
        load_labeled_queries_for_split(
            labeled_split_fixture.queries_root,
            labeled_split_fixture.full_plan_path,
            allow_provisional_asset_groups=True,
        )


def test_load_labeled_queries_returns_locked_dev_ids(labeled_split_fixture):
    with pytest.raises(ValueError, match="provisional"):
        load_labeled_queries_for_split(
            labeled_split_fixture.queries_root,
            labeled_split_fixture.full_plan_path,
        )

    candidates, locked_ids = load_labeled_queries_for_split(
        labeled_split_fixture.queries_root,
        labeled_split_fixture.full_plan_path,
        allow_provisional_asset_groups=True,
    )

    assert len(candidates) == 4500
    assert len(locked_ids) == 200
    assert sum(query.label_status == "cross_agreed" for query in candidates) == 4275
    assert sum(query.label_status == "arbitrated" for query in candidates) == 225


def test_split_rejects_derived_queries_not_rebuildable_from_accepted_batches(
    labeled_split_fixture,
):
    derived = labeled_split_fixture.queries_root / "queries.jsonl"
    rows = derived.read_text(encoding="utf-8").splitlines()
    first = Query.model_validate_json(rows[0])
    payload = first.model_dump(mode="json")
    payload["text"] = "tampered derived query"
    payload["turns"][-1]["content"] = "tampered derived query"
    rows[0] = Query.model_validate(payload).model_dump_json()
    derived.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="queries.jsonl|accepted ledger"):
        load_labeled_queries_for_split(
            labeled_split_fixture.queries_root,
            labeled_split_fixture.full_plan_path,
            allow_provisional_asset_groups=True,
        )


def test_split_rejects_labeled_nonlabel_field_tampering(labeled_split_fixture):
    labeled_path = (
        labeled_split_fixture.queries_root / "labels" / "labeled_queries.jsonl"
    )
    rows = labeled_path.read_text(encoding="utf-8").splitlines()
    first = Query.model_validate_json(rows[0])
    payload = first.model_dump(mode="json")
    payload["text"] = "tampered labeled text"
    payload["turns"][-1]["content"] = "tampered labeled text"
    rows[0] = Query.model_validate(payload).model_dump_json()
    labeled_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="非标签字段|accepted"):
        load_labeled_queries_for_split(
            labeled_split_fixture.queries_root,
            labeled_split_fixture.full_plan_path,
            allow_provisional_asset_groups=True,
        )


def test_split_rejects_coherent_spot_check_queue_removal(labeled_split_fixture):
    labels = labeled_split_fixture.queries_root / "labels"
    queue_path = labels / "arbitration_queue.jsonl"
    arbitration_path = labels / "arbitrations.jsonl"
    labeled_path = labels / "labeled_queries.jsonl"
    manifest_path = labels / "manifest.json"

    queue_rows = [
        json.loads(line) for line in queue_path.read_text("utf-8").splitlines()
    ]
    removed_id = queue_rows[0]["query_id"]
    queue_bytes = canonical_jsonl_bytes(queue_rows[1:])
    queue_path.write_bytes(queue_bytes)

    arbitration_rows = [
        json.loads(line)
        for line in arbitration_path.read_text("utf-8").splitlines()
        if json.loads(line)["query_id"] != removed_id
    ]
    arbitration_path.write_bytes(canonical_jsonl_bytes(arbitration_rows))

    labeled = [
        Query.model_validate_json(line)
        for line in labeled_path.read_text("utf-8").splitlines()
    ]
    target_index = next(
        index for index, query in enumerate(labeled) if query.query_id == removed_id
    )
    payload = labeled[target_index].model_dump(mode="json")
    payload["label_status"] = "cross_agreed"
    payload["label_provenance"] = [
        decision
        for decision in payload["label_provenance"]
        if decision["decision_type"] != "arbitration"
    ]
    labeled[target_index] = Query.model_validate(payload)
    labeled_path.write_bytes(canonical_jsonl_bytes(labeled))

    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["arbitration_queue_sha256"] = sha256_bytes(queue_bytes)
    manifest["spot_check_count"] -= 1
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError, match="确定性重建|计数"):
        load_labeled_queries_for_split(
            labeled_split_fixture.queries_root,
            labeled_split_fixture.full_plan_path,
            allow_provisional_asset_groups=True,
        )


def test_freeze_is_idempotent_but_rejects_different_content_and_tampering(
    full_candidate_fixture, tmp_path
):
    candidates, locked_ids = full_candidate_fixture
    assigned = stratified_split(
        candidates,
        locked_dev_query_ids=locked_ids,
        seed=20260711,
    )
    root = tmp_path / "queries"

    with pytest.raises(ValueError, match="full_plan_path|plan hash"):
        freeze_test_split(
            tmp_path / "formal-rejects-self-reported-hash",
            assigned,
            seed=20260711,
            full_plan_sha256="f" * 64,
            allow_provisional_asset_groups=True,
        )

    paths = freeze_test_split(
        root,
        assigned,
        seed=20260711,
        full_plan_sha256="f" * 64,
        allow_provisional_asset_groups=True,
        split_spec=MECHANICAL_FULL_SPLIT_SPEC,
    )

    manifest = verify_frozen_split(
        root,
        expected_full_plan_sha256="f" * 64,
        allow_provisional_asset_groups=True,
    )
    assert manifest is not None
    assert manifest.count == 1000
    assert manifest.intent_counts == {
        intent: count
        for intent, count in sorted(
            Counter(
                query.gt_intent for query in assigned if query.split == "test_frozen"
            ).items()
        )
    }
    assert not (paths[0].stat().st_mode & stat.S_IWRITE)
    assignment_path = root / "split_assignment.jsonl"
    assignment_bytes = assignment_path.read_bytes()
    assert not (assignment_path.stat().st_mode & stat.S_IWRITE)
    assert manifest.assignment_sha256 == hashlib.sha256(assignment_bytes).hexdigest()
    assert (
        freeze_test_split(
            root,
            assigned,
            seed=20260711,
            full_plan_sha256="f" * 64,
            allow_provisional_asset_groups=True,
            split_spec=MECHANICAL_FULL_SPLIT_SPEC,
        )
        == paths
    )

    changed = list(assigned)
    test_index = next(
        index for index, query in enumerate(changed) if query.split == "test_frozen"
    )
    changed_payload = changed[test_index].model_dump(mode="json")
    changed_payload["text"] = "机械占位-已改变"
    changed_payload["turns"][-1]["content"] = "机械占位-已改变"
    changed[test_index] = Query.model_validate(changed_payload)
    with pytest.raises(FrozenSplitError, match="冻结|不同"):
        freeze_test_split(
            root,
            changed,
            seed=20260711,
            full_plan_sha256="f" * 64,
            allow_provisional_asset_groups=True,
            split_spec=MECHANICAL_FULL_SPLIT_SPEC,
        )

    os.chmod(assignment_path, stat.S_IWRITE | stat.S_IREAD)
    assignment_path.write_bytes(assignment_bytes + b" ")
    with pytest.raises(FrozenSplitError, match="assignment.*哈希"):
        verify_frozen_split(root, allow_provisional_asset_groups=True)
    assignment_path.write_bytes(assignment_bytes)
    os.chmod(assignment_path, stat.S_IREAD)

    os.chmod(paths[0], stat.S_IWRITE | stat.S_IREAD)
    paths[0].write_bytes(paths[0].read_bytes() + b" ")
    with pytest.raises(FrozenSplitError, match="哈希"):
        verify_frozen_split(root, allow_provisional_asset_groups=True)
