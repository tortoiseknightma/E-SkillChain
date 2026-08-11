"""Focused r2 planner tests that do not need private image bytes."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pytest

from skillchain.data.asset_catalog import AssetCatalog, AssetResolution
from skillchain.schemas import DatasetAsset, Intent
from skillchain.synthesis.models import CorpusPlan, PlannedQuery
from skillchain.synthesis.planning import (
    PHASE3_TASK_SPEC_SHA256,
    PHASE3_TASK_SPEC_VERSION,
    CapabilityAssignment,
    R2DevPrefixRebind,
    _r2_solve_joint_batch_layout,
    audit_r2_core_in_memory_plan,
    build_r2_core_in_memory_plan,
    build_r2_split_atomic_batch_schedule,
    rebind_r2_dev_prefix,
)
from skillchain.synthesis.store import canonical_jsonl_bytes
from skillchain.task_spec import load_mvp_task_specification_v1
from skillchain.taxonomy import DEFAULT_TAXONOMY_SHA256, TAXONOMY_VERSION


@dataclass(frozen=True)
class _Manifest:
    catalog_sha256: str
    leakage_policy_version: str = "dataset-asset-components-v1"


@dataclass(frozen=True)
class _Record:
    index: int
    asset_id: str
    sha256: str
    component_id: str
    old_path: str
    new_path: str
    capability: str
    intent: Intent
    is_boundary: bool = False
    boundary_strategy: str | None = None
    boundary_group_id: str | None = None


class _Catalog:
    """A verified-in-spirit catalog with independently controlled identities."""

    def __init__(self, records: list[_Record], *, target: bool, digest: str) -> None:
        self.manifest = _Manifest(digest)
        self._by_path: dict[str, AssetResolution] = {}
        self._by_asset_id: dict[str, AssetResolution] = {}
        for record in records:
            path = record.new_path if target else record.old_path
            asset = DatasetAsset(
                schema_version=2,
                asset_id=record.asset_id,
                source_dataset="r2-fixture",
                source_revision="fixture-v1",
                source_record_id=f"record-{record.index:03d}",
                transform_policy_version="fixture-copy-v1",
                local_path=path,
                sha256=record.sha256,
                phash=record.sha256[:16],
                near_duplicate_cluster_id=record.component_id,
                product_id=None,
                license_id="test-fixture-only",
                cloud_upload_allowed=False,
                public_demo_allowed=False,
            )
            resolution = AssetResolution(asset=asset, leakage_group_id=record.component_id)
            self._by_path[path] = resolution
            self._by_asset_id[record.asset_id] = resolution

    def require_verified_files(self) -> None:
        return None

    def verify_asset_ids(self, asset_ids) -> None:
        return None

    def resolve_asset_id(self, asset_id: str) -> AssetResolution:
        return self._by_asset_id[asset_id]

    def resolve_path(self, image_path: str | Path) -> AssetResolution:
        return self._by_path[Path(image_path).as_posix()]

    def verify_reference(
        self,
        asset_id: str,
        image_path: str,
        leakage_group_id: str,
    ) -> AssetResolution:
        by_id = self.resolve_asset_id(asset_id)
        by_path = self.resolve_path(image_path)
        if by_id != by_path or by_id.leakage_group_id != leakage_group_id:
            raise ValueError("fixture catalog reference mismatch")
        return by_id

    def as_loaded(self) -> AssetCatalog:
        return cast(AssetCatalog, self)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _records(
    *,
    document_count: int = 30,
    mutation: Literal[
        "none", "sha", "component", "unchanged_path", "non_document_path"
    ] = "none",
) -> list[_Record]:
    if document_count not in {29, 30, 31}:
        raise ValueError("fixture only supports the document-count boundary cases")

    records: list[_Record] = []
    document_index = 0
    regular_exact_count = 27 + (30 - document_count)

    def append(
        *,
        component_key: str,
        capability: str,
        intent: Intent,
        is_document: bool = False,
        shared_asset_path: bool = False,
        is_boundary: bool = False,
        boundary_strategy: str | None = None,
        boundary_group_id: str | None = None,
    ) -> None:
        nonlocal document_index
        index = len(records) + 1
        asset_key = component_key
        if is_document:
            document_index += 1
            old_path = f"query_images/utility/wikimedia-{document_index:03d}.jpg"
            new_path = f"query_images/utility/document-r2-{document_index:03d}.jpg"
        elif shared_asset_path:
            old_path = f"query_images/cross_intent/{asset_key}.jpg"
            new_path = old_path
        else:
            old_path = f"query_images/{intent}/{asset_key}.jpg"
            new_path = old_path
        records.append(
            _Record(
                index=index,
                asset_id=f"asset.fixture.{_digest(f'asset-{asset_key}')}",
                sha256=_digest(f"bytes-{asset_key}"),
                component_id=f"component.fixture.{_digest(f'component-{component_key}')}",
                old_path=old_path,
                new_path=new_path,
                capability=capability,
                intent=intent,
                is_boundary=is_boundary,
                boundary_strategy=boundary_strategy,
                boundary_group_id=boundary_group_id,
            )
        )

    documents = [
        {
            "component_key": f"dev-document-{index:03d}",
            "capability": "utility.document_reading",
            "intent": "utility",
            "is_document": True,
            "is_boundary": index <= 3,
            "boundary_strategy": "natural_ambiguity" if index <= 3 else None,
        }
        for index in range(1, document_count + 1)
    ]
    normal: list[dict[str, object]] = []
    for capability, intent, count, boundary_count, label in (
        ("product.exact_match", "exact_match", regular_exact_count, 3, "exact"),
        ("product.multi_search", "multi_product", 35, 2, "multi"),
        ("product.style_recommendation", "divergent_rec", 27, 2, "style"),
        ("knowledge.visual_encyclopedia", "encyclopedia", 27, 3, "ency"),
        ("utility.recipe_guidance", "utility", 30, 3, "recipe"),
    ):
        normal.extend(
            {
                "component_key": f"dev-{label}-{index:03d}",
                "capability": capability,
                "intent": intent,
                "is_boundary": index <= boundary_count,
                "boundary_strategy": "natural_ambiguity"
                if index <= boundary_count
                else None,
            }
            for index in range(1, count + 1)
        )
    abo_groups = [
        [
            {
                "component_key": f"dev-abo-{index:03d}",
                "capability": capability,
                "intent": intent,
                "shared_asset_path": True,
                "is_boundary": True,
                "boundary_strategy": "cross_intent_triplet",
                "boundary_group_id": f"legacy-triplet-{index:03d}",
            }
            for capability, intent in (
                ("product.exact_match", "exact_match"),
                ("product.style_recommendation", "divergent_rec"),
                ("knowledge.visual_encyclopedia", "encyclopedia"),
            )
        ]
        for index in range(1, 9)
    ]

    document_slots_by_count = {
        29: (4, 4, 4, 4, 4, 4, 3, 2),
        30: (4, 4, 4, 4, 4, 4, 3, 3),
        31: (4, 4, 4, 4, 4, 4, 3, 4),
    }
    document_offset = 0
    normal_offset = 0
    for batch_index, document_slots in enumerate(
        document_slots_by_count[document_count],
        start=1,
    ):
        for payload in documents[document_offset : document_offset + document_slots]:
            append(**payload)  # type: ignore[arg-type]
        document_offset += document_slots
        for payload in abo_groups[batch_index - 1]:
            append(**payload)  # type: ignore[arg-type]
        normal_slots = 25 - document_slots - len(abo_groups[batch_index - 1])
        for payload in normal[normal_offset : normal_offset + normal_slots]:
            append(**payload)  # type: ignore[arg-type]
        normal_offset += normal_slots
    if document_offset != document_count or normal_offset != len(normal):
        raise AssertionError("fixture records do not fill the eight dev batches")

    mutated_document = False
    mutated_non_document = False
    output: list[_Record] = []
    for record in records:
        is_document = record.capability == "utility.document_reading"
        sha256 = record.sha256
        component_id = record.component_id
        new_path = record.new_path
        if is_document and not mutated_document:
            if mutation == "sha":
                sha256 = _digest("drifted-bytes")
            elif mutation == "component":
                component_id = f"component.fixture.{_digest('drifted-component')}"
            elif mutation == "unchanged_path":
                new_path = record.old_path
            mutated_document = mutation in {"sha", "component", "unchanged_path"}
        if (
            not is_document
            and record.boundary_group_id is None
            and not mutated_non_document
            and mutation == "non_document_path"
        ):
            new_path = f"query_images/{record.intent}/drifted-{record.index:03d}.jpg"
            mutated_non_document = True
        output.append(
            _Record(
                index=record.index,
                asset_id=record.asset_id,
                sha256=sha256,
                component_id=component_id,
                old_path=record.old_path,
                new_path=new_path,
                capability=record.capability,
                intent=record.intent,
                is_boundary=record.is_boundary,
                boundary_strategy=record.boundary_strategy,
                boundary_group_id=record.boundary_group_id,
            )
        )
    return output


def _assignments(
    records: list[_Record],
    *,
    target: bool,
    capability_drift: bool = False,
) -> list[CapabilityAssignment]:
    task_specification = load_mvp_task_specification_v1()
    assignments: list[CapabilityAssignment] = []
    for record in records:
        capability = record.capability
        if capability_drift and record.index == 1:
            capability = "utility.recipe_guidance"
        task = task_specification.capabilities_by_id[capability]
        path = record.new_path if target else record.old_path
        assignment_key = _digest(f"{record.asset_id}\n{path}\n{record.intent}")
        assignments.append(
            CapabilityAssignment(
                assignment_id=f"assignment.fixture.{assignment_key}",
                asset_id=record.asset_id,
                image_path=path,
                canonical_intent=record.intent,
                canonical_capability=capability,
                acceptable_capabilities=(capability,),
                requires_card=(
                    task.output_contract.card_requirement == "required"
                ),
                allowed_tools=task.allowed_tools,
                taxonomy_version=TAXONOMY_VERSION,
                taxonomy_sha256=DEFAULT_TAXONOMY_SHA256,
                task_spec_version=PHASE3_TASK_SPEC_VERSION,
                task_spec_sha256=PHASE3_TASK_SPEC_SHA256,
                source_ref="fixture:public-input-review-v1",
                source_artifact_sha256=("f" if target else "e") * 64,
                rationale="Mechanical public-input capability fixture.",
            )
        )
    return assignments


def _assignments_sha(assignments: list[CapabilityAssignment]) -> str:
    return hashlib.sha256(canonical_jsonl_bytes(assignments)).hexdigest()


def _parent_core_plan(
    records: list[_Record],
    assignments: list[CapabilityAssignment],
    *,
    catalog_sha256: str,
    assignments_sha256: str,
) -> CorpusPlan:
    by_key = {
        (assignment.asset_id, assignment.image_path, assignment.canonical_intent): assignment
        for assignment in assignments
    }
    prefix: list[PlannedQuery] = []
    for record in records:
        assignment = by_key[(record.asset_id, record.old_path, record.intent)]
        batch_number = (record.index - 1) // 25 + 1
        prefix.append(
            PlannedQuery(
                plan_id=f"dm-{record.index:03d}",
                batch_id=f"dev-mini-{batch_number:03d}",
                position=(record.index - 1) % 25 + 1,
                taxonomy_version=assignment.taxonomy_version,
                task_spec_version=assignment.task_spec_version,
                asset_id=record.asset_id,
                image_path=record.old_path,
                leakage_group_id=record.component_id,
                template_family=f"fixture-dev-{batch_number:03d}",
                generator_batch_id=f"dev-mini-{batch_number:03d}",
                canonical_intent=assignment.canonical_intent,
                canonical_capability=assignment.canonical_capability,
                acceptable_capabilities=list(assignment.acceptable_capabilities),
                requires_card=assignment.requires_card,
                capability_assignment_id=assignment.assignment_id,
                capability_assignment_source_sha256=(
                    assignment.source_artifact_sha256
                ),
                is_boundary=record.is_boundary,
                boundary_strategy=record.boundary_strategy,
                boundary_group_id=record.boundary_group_id,
                provisional_split="dev_mini",
            )
        )
    tail: list[PlannedQuery] = []
    source = prefix[-1]
    for index in range(201, 1501):
        batch_number = (index - 1) // 25 + 1
        payload = source.model_dump(mode="python")
        payload.update(
            {
                "plan_id": f"core-{index:04d}",
                "batch_id": f"core-{batch_number:03d}",
                "position": (index - 1) % 25 + 1,
                "template_family": f"fixture-core-{batch_number:03d}",
                "generator_batch_id": f"core-{batch_number:03d}",
                "provisional_split": "opt_pool",
            }
        )
        tail.append(PlannedQuery.model_validate(payload, strict=True))
    return CorpusPlan(
        scope="core",
        seed=20260804,
        asset_catalog_sha256=catalog_sha256,
        capability_assignments_sha256=assignments_sha256,
        leakage_policy_version="dataset-asset-components-v1",
        queries=[*prefix, *tail],
    )


def _plan_sha256(plan: CorpusPlan) -> str:
    content = json.dumps(
        plan.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"{content}\n".encode("utf-8")).hexdigest()


def _rebind_fixture(
    *,
    mutation: Literal[
        "none",
        "sha",
        "component",
        "unchanged_path",
        "non_document_path",
        "capability",
    ] = "none",
    document_count: int = 30,
) -> R2DevPrefixRebind:
    record_mutation: Literal[
        "none", "sha", "component", "unchanged_path", "non_document_path"
    ] = (
        "none" if mutation == "capability" else mutation
    )
    parent_records = _records(document_count=document_count)
    target_records = _records(
        document_count=document_count,
        mutation=record_mutation,
    )
    parent_assignments = _assignments(parent_records, target=False)
    target_assignments = _assignments(
        target_records,
        target=True,
        capability_drift=mutation == "capability",
    )
    parent_catalog_sha256 = "a" * 64
    target_catalog_sha256 = "b" * 64
    parent_assignments_sha256 = _assignments_sha(parent_assignments)
    target_assignments_sha256 = _assignments_sha(target_assignments)
    parent_plan = _parent_core_plan(
        parent_records,
        parent_assignments,
        catalog_sha256=parent_catalog_sha256,
        assignments_sha256=parent_assignments_sha256,
    )
    return rebind_r2_dev_prefix(
        parent_plan,
        parent_core_plan_sha256=_plan_sha256(parent_plan),
        parent_catalog=_Catalog(
            parent_records,
            target=False,
            digest=parent_catalog_sha256,
        ).as_loaded(),
        expected_parent_catalog_sha256=parent_catalog_sha256,
        parent_capability_assignments=parent_assignments,
        expected_parent_capability_assignments_sha256=parent_assignments_sha256,
        target_catalog=_Catalog(
            target_records,
            target=True,
            digest=target_catalog_sha256,
        ).as_loaded(),
        expected_target_catalog_sha256=target_catalog_sha256,
        target_capability_assignments=target_assignments,
        expected_target_capability_assignments_sha256=target_assignments_sha256,
    )


def _tail_records() -> list[_Record]:
    records: list[_Record] = []
    next_index = 201

    def append(
        *,
        component_key: str,
        capability: str,
        intent: Intent,
        shared_asset_path: bool = False,
    ) -> None:
        nonlocal next_index
        path = (
            f"query_images/cross_intent/{component_key}.jpg"
            if shared_asset_path
            else f"query_images/{intent}/{component_key}.jpg"
        )
        records.append(
            _Record(
                index=next_index,
                asset_id=f"asset.fixture.{_digest(f'asset-{component_key}')}",
                sha256=_digest(f"bytes-{component_key}"),
                component_id=f"component.fixture.{_digest(f'component-{component_key}')}",
                old_path=path,
                new_path=path,
                capability=capability,
                intent=intent,
            )
        )
        next_index += 1

    for capability, intent, count, label in (
        ("product.exact_match", "exact_match", 136, "tail-exact"),
        ("product.multi_search", "multi_product", 90, "tail-multi"),
        ("product.style_recommendation", "divergent_rec", 102, "tail-style"),
        ("knowledge.visual_encyclopedia", "encyclopedia", 95, "tail-ency"),
        ("utility.document_reading", "utility", 65, "tail-document"),
        ("utility.recipe_guidance", "utility", 170, "tail-recipe"),
    ):
        for index in range(1, count + 1):
            append(
                component_key=f"{label}-{index:03d}",
                capability=capability,
                intent=intent,
            )
    for index in range(1, 21):
        component_key = f"tail-abo-{index:03d}"
        for capability, intent in (
            ("product.exact_match", "exact_match"),
            ("product.style_recommendation", "divergent_rec"),
            ("knowledge.visual_encyclopedia", "encyclopedia"),
        ):
            append(
                component_key=component_key,
                capability=capability,
                intent=intent,
                shared_asset_path=True,
            )
    for index in range(1, 11):
        component_key = f"tail-food-{index:03d}"
        for capability, intent in (
            ("knowledge.visual_encyclopedia", "encyclopedia"),
            ("utility.recipe_guidance", "utility"),
        ):
            append(
                component_key=component_key,
                capability=capability,
                intent=intent,
                shared_asset_path=True,
            )
    return records


def _full_r2_fixture(*, reverse_target_assignments: bool = False):
    parent_records = _records()
    target_records = [*parent_records, *_tail_records()]
    parent_assignments = _assignments(parent_records, target=False)
    target_assignments = _assignments(target_records, target=True)
    if reverse_target_assignments:
        target_assignments = list(reversed(target_assignments))
    parent_catalog_sha256 = "a" * 64
    target_catalog_sha256 = "b" * 64
    parent_assignments_sha256 = _assignments_sha(parent_assignments)
    target_assignments_sha256 = _assignments_sha(target_assignments)
    parent_plan = _parent_core_plan(
        parent_records,
        parent_assignments,
        catalog_sha256=parent_catalog_sha256,
        assignments_sha256=parent_assignments_sha256,
    )
    parent_catalog = _Catalog(
        parent_records,
        target=False,
        digest=parent_catalog_sha256,
    ).as_loaded()
    target_catalog = _Catalog(
        target_records,
        target=True,
        digest=target_catalog_sha256,
    ).as_loaded()
    rebound = rebind_r2_dev_prefix(
        parent_plan,
        parent_core_plan_sha256=_plan_sha256(parent_plan),
        parent_catalog=parent_catalog,
        expected_parent_catalog_sha256=parent_catalog_sha256,
        parent_capability_assignments=parent_assignments,
        expected_parent_capability_assignments_sha256=parent_assignments_sha256,
        target_catalog=target_catalog,
        expected_target_catalog_sha256=target_catalog_sha256,
        target_capability_assignments=target_assignments,
        expected_target_capability_assignments_sha256=target_assignments_sha256,
    )
    return build_r2_core_in_memory_plan(
        rebound,
        target_catalog=target_catalog,
        target_capability_assignments=target_assignments,
        expected_target_catalog_sha256=target_catalog_sha256,
        expected_target_capability_assignments_sha256=target_assignments_sha256,
    )


def test_r2_split_atomic_schedule_is_feasible_after_document_recipe_correction() -> None:
    batch_splits, schedule = build_r2_split_atomic_batch_schedule()

    assert Counter(batch_splits) == {
        "opt_pool": 32,
        "val": 8,
        "test_frozen": 12,
    }
    assert all(sum(batch.values()) == 25 for batch in schedule)
    assert all(set(batch) == {
        "product.exact_match",
        "product.multi_search",
        "product.style_recommendation",
        "knowledge.visual_encyclopedia",
        "utility.document_reading",
        "utility.recipe_guidance",
    } for batch in schedule)
    assert all(
        1 <= batch["utility.document_reading"] <= 2 for batch in schedule
    )


def test_r2_joint_layout_uses_only_the_proven_minimum_counterfactual_stacks() -> None:
    solution = _r2_solve_joint_batch_layout(seed=20260804)

    assert solution == _r2_solve_joint_batch_layout(seed=20260804)
    assert solution.stacked_counterfactual_batch_count == 2
    assert sum(value > 0 for value in solution.abo_by_batch) == 18
    assert max(solution.abo_by_batch) == 2
    assert sum(value > 0 for value in solution.food_by_batch) == 10
    assert max(solution.food_by_batch) == 1
    assert all(
        solution.batch_final_splits[index] == "opt_pool"
        for index, value in enumerate(solution.abo_by_batch)
        if value == 2
    )
    for split, expected_abo, expected_food in (
        ("opt_pool", 12, 6),
        ("val", 3, 2),
        ("test_frozen", 5, 2),
    ):
        indices = [
            index
            for index, batch_split in enumerate(solution.batch_final_splits)
            if batch_split == split
        ]
        assert sum(solution.abo_by_batch[index] for index in indices) == expected_abo
        assert sum(solution.food_by_batch[index] for index in indices) == expected_food


def test_r2_layout_is_stable_when_capability_assignment_input_order_changes() -> None:
    canonical = _full_r2_fixture()
    reordered = _full_r2_fixture(reverse_target_assignments=True)

    assert canonical.plan.queries == reordered.plan.queries
    assert canonical.final_split_by_plan_id == reordered.final_split_by_plan_id
    assert canonical.reuse_variant_by_plan_id == reordered.reuse_variant_by_plan_id
    assert canonical.reuse_reason_by_plan_id == reordered.reuse_reason_by_plan_id
    assert canonical.audit == reordered.audit


def test_r2_builder_returns_a_fully_audited_in_memory_layout() -> None:
    started = time.perf_counter()
    result = _full_r2_fixture()
    elapsed = time.perf_counter() - started
    audit = audit_r2_core_in_memory_plan(result)

    assert len(result.plan.queries) == 1500
    assert audit.unique_component_count == 867
    assert audit.max_component_reuse == 3
    assert audit.tail_multi_component_count == 90
    assert audit.counterfactual_components_by_split == {
        "opt_pool": 18,
        "val": 5,
        "test_frozen": 7,
    }
    assert audit.capacity_required_triples_by_capability == {
        "product.exact_match": 48,
        "product.multi_search": 14,
        "product.style_recommendation": 41,
        "knowledge.visual_encyclopedia": 45,
        "utility.document_reading": 0,
        "utility.recipe_guidance": 0,
    }
    batch_splits: dict[str, set[str]] = {}
    for item in result.plan.queries[200:]:
        batch_splits.setdefault(item.generator_batch_id, set()).add(
            result.final_split_by_plan_id[item.plan_id]
        )
    assert len(batch_splits) == 52
    assert all(len(splits) == 1 for splits in batch_splits.values())
    assert elapsed < 30


def test_r2_rebinds_exactly_the_30_carried_document_paths() -> None:
    rebound = _rebind_fixture()
    parent_records = _records()
    parent_assignments = _assignments(parent_records, target=False)
    parent = _parent_core_plan(
        parent_records,
        parent_assignments,
        catalog_sha256="a" * 64,
        assignments_sha256=_assignments_sha(parent_assignments),
    )

    assert rebound.dev_prefix.scope == "dev_mini"
    assert len(rebound.dev_prefix.queries) == 200
    assert rebound.dev_prefix.asset_catalog_sha256 == "b" * 64
    assert rebound.dev_prefix.capability_assignments_sha256 == _assignments_sha(
        _assignments(_records(), target=True)
    )
    entries = rebound.document_path_rebind_manifest.entries
    assert len(entries) == 30
    document_indices = [
        index
        for index, item in enumerate(parent.queries[:200], start=1)
        if item.canonical_capability == "utility.document_reading"
    ]
    assert [entry.prefix_index for entry in entries] == document_indices
    assert all(entry.old_image_path != entry.new_image_path for entry in entries)
    assert all(entry.old_assignment_id != entry.new_assignment_id for entry in entries)
    assert Counter(
        item.generator_batch_id
        for item in rebound.dev_prefix.queries
        if item.canonical_capability == "utility.document_reading"
    ) == {
        "dev-mini-001": 4,
        "dev-mini-002": 4,
        "dev-mini-003": 4,
        "dev-mini-004": 4,
        "dev-mini-005": 4,
        "dev-mini-006": 4,
        "dev-mini-007": 3,
        "dev-mini-008": 3,
    }

    preserved_fields = (
        "plan_id",
        "batch_id",
        "position",
        "asset_id",
        "leakage_group_id",
        "template_family",
        "generator_batch_id",
        "taxonomy_version",
        "task_spec_version",
        "canonical_intent",
        "canonical_capability",
        "acceptable_capabilities",
        "requires_card",
        "is_boundary",
        "boundary_strategy",
        "boundary_group_id",
        "provisional_split",
    )
    for original, rebound_item in zip(
        parent.queries[:200],
        rebound.dev_prefix.queries,
        strict=True,
    ):
        assert all(
            getattr(original, field) == getattr(rebound_item, field)
            for field in preserved_fields
        )
        if original.canonical_capability == "utility.document_reading":
            assert original.image_path != rebound_item.image_path
            assert rebound_item.image_path.startswith("query_images/utility/document-r2-")
        else:
            assert original.image_path == rebound_item.image_path
        assert (
            original.capability_assignment_source_sha256
            != rebound_item.capability_assignment_source_sha256
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("sha", "SHA-256 drift"),
        ("component", "leakage component drift"),
        ("unchanged_path", "path did not rebind"),
        ("non_document_path", "non-Document path changed"),
        ("capability", "semantics drifted"),
    ],
)
def test_r2_rebind_fails_closed_on_identity_or_semantic_drift(
    mutation: Literal[
        "sha", "component", "unchanged_path", "non_document_path", "capability"
    ],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _rebind_fixture(mutation=mutation)


@pytest.mark.parametrize("document_count", [29, 31])
def test_r2_rebind_requires_exactly_30_document_path_changes(
    document_count: int,
) -> None:
    with pytest.raises(ValueError, match="exactly 30"):
        _rebind_fixture(document_count=document_count)
