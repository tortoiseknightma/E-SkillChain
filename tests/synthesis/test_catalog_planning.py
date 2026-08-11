"""Fail-closed planner tests for the independently loaded asset catalog."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from skillchain.data.asset_catalog import AssetCatalog, AssetResolution
from skillchain.schemas import DatasetAsset
from skillchain.synthesis.planning import (
    CapabilityAssignment,
    DEV_MINI_CAPABILITY_COUNTS,
    INTENT_ORDER,
    PHASE3_TASK_SPEC_SHA256,
    PHASE3_TASK_SPEC_VERSION,
    activate_dev_plan,
    audit_plan_catalog_binding,
    build_dev_mini_plan,
    extend_full_plan,
    load_capability_assignments,
    write_dev_mini_plan,
)
from skillchain.synthesis.store import canonical_jsonl_bytes
from skillchain.task_spec import (
    load_mvp_task_specification_v1,
)
from skillchain.taxonomy import (
    DEFAULT_TAXONOMY_SHA256,
    TAXONOMY_VERSION,
    capabilities_for_intent,
)


@dataclass(frozen=True)
class _FakeCatalogManifest:
    catalog_sha256: str
    leakage_policy_version: str = "dataset-asset-components-v1"


class _FakeLoadedCatalog:
    """Small duck-typed catalog that exercises only the planner boundary."""

    def __init__(
        self,
        paths: list[str],
        *,
        catalog_sha256: str = "a" * 64,
        components: dict[str, str] | None = None,
    ) -> None:
        components = components or {}
        self.manifest = _FakeCatalogManifest(catalog_sha256=catalog_sha256)
        self._by_path: dict[str, AssetResolution] = {}
        self._by_asset_id: dict[str, AssetResolution] = {}
        for raw_path in paths:
            path = Path(raw_path).as_posix()
            digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
            component = components.get(path, f"component.test.{digest}")
            resolution = AssetResolution(
                asset=DatasetAsset(
                    schema_version=2,
                    asset_id=f"asset.test.{digest}",
                    source_dataset="mechanical-test",
                    source_revision="fixture-v1",
                    source_record_id=f"record-{digest}",
                    transform_policy_version="mechanical-copy-v1",
                    local_path=path,
                    sha256=digest,
                    phash=digest[:16],
                    near_duplicate_cluster_id=component,
                    product_id=None,
                    license_id="test-fixture-only",
                    cloud_upload_allowed=False,
                    public_demo_allowed=False,
                ),
                leakage_group_id=component,
            )
            self._by_path[path] = resolution
            self._by_asset_id[resolution.asset.asset_id] = resolution

    def resolve_path(self, local_path: str | Path) -> AssetResolution:
        path = Path(local_path).as_posix()
        try:
            return self._by_path[path]
        except KeyError:
            raise ValueError(f"asset not cataloged: {path}") from None

    def require_verified_files(self) -> None:
        return None

    def verify_asset_ids(self, asset_ids) -> None:
        return None

    def resolve_asset_id(self, asset_id: str) -> AssetResolution:
        try:
            return self._by_asset_id[asset_id]
        except KeyError:
            raise ValueError(f"asset not cataloged: {asset_id}") from None

    def component_for_asset(self, asset_id: str) -> str:
        return self.resolve_asset_id(asset_id).leakage_group_id

    def verify_reference(
        self,
        asset_id: str,
        image_path: str,
        leakage_group_id: str,
    ) -> AssetResolution:
        resolution = self.resolve_path(image_path)
        if (
            resolution.asset.asset_id != asset_id
            or resolution.leakage_group_id != leakage_group_id
        ):
            raise ValueError(
                "asset catalog reference mismatch: "
                f"{asset_id}, {image_path}, {leakage_group_id}"
            )
        return resolution

    def without_path(self, image_path: str) -> AssetCatalog:
        clone = object.__new__(_FakeLoadedCatalog)
        clone.manifest = self.manifest
        clone._by_path = dict(self._by_path)
        removed = clone._by_path.pop(Path(image_path).as_posix())
        clone._by_asset_id = dict(self._by_asset_id)
        clone._by_asset_id.pop(removed.asset.asset_id)
        return cast(AssetCatalog, clone)

    def as_loaded(self) -> AssetCatalog:
        return cast(AssetCatalog, self)


def _dev_paths(image_root: Path) -> list[str]:
    return [
        (Path("query_images") / path.relative_to(image_root)).as_posix()
        for intent in INTENT_ORDER
        for path in sorted((image_root / intent).glob("*.jpg"))
    ]


def _full_pools() -> dict[str, list[Path]]:
    return {
        intent: [
            Path(f"query_images/{intent}/full-{index:04d}.jpg")
            for index in range(1, 401)
        ]
        for intent in INTENT_ORDER
    }


def _catalog_for(
    image_root: Path,
    full_pools: dict[str, list[Path]] | None = None,
    *,
    catalog_sha256: str = "a" * 64,
    components: dict[str, str] | None = None,
) -> AssetCatalog:
    paths = _dev_paths(image_root)
    if full_pools is not None:
        paths.extend(path.as_posix() for pool in full_pools.values() for path in pool)
    return _FakeLoadedCatalog(
        paths,
        catalog_sha256=catalog_sha256,
        components=components,
    ).as_loaded()


def _capability_assignments(catalog: AssetCatalog) -> list[CapabilityAssignment]:
    loaded = cast(_FakeLoadedCatalog, catalog)
    task_specification = load_mvp_task_specification_v1()
    assignments: list[CapabilityAssignment] = []
    utility_index = 0
    for path, resolution in sorted(loaded._by_path.items()):
        source_intent = Path(path).parts[-2]
        intents = (
            ("exact_match", "divergent_rec", "encyclopedia")
            if source_intent == "exact_match"
            else (source_intent,)
        )
        for intent in intents:
            capabilities = capabilities_for_intent(intent)
            if intent == "utility":
                capability = capabilities[utility_index % 2]
                utility_index += 1
            else:
                assert len(capabilities) == 1
                capability = capabilities[0]
            task = task_specification.capabilities_by_id[capability.capability_id]
            assignment_digest = hashlib.sha256(
                f"{resolution.asset.asset_id}\n{path}\n{intent}".encode()
            ).hexdigest()
            assignments.append(
                CapabilityAssignment(
                    assignment_id=f"assignment.{assignment_digest}",
                    asset_id=resolution.asset.asset_id,
                    image_path=path,
                    canonical_intent=intent,
                    canonical_capability=capability.capability_id,
                    acceptable_capabilities=(capability.capability_id,),
                    requires_card=capability.requires_card,
                    allowed_tools=task.allowed_tools,
                    taxonomy_version=TAXONOMY_VERSION,
                    taxonomy_sha256=DEFAULT_TAXONOMY_SHA256,
                    task_spec_version=PHASE3_TASK_SPEC_VERSION,
                    task_spec_sha256=PHASE3_TASK_SPEC_SHA256,
                    source_ref="fixture:public-asset-capability-review-v1",
                    source_artifact_sha256="e" * 64,
                    rationale="Mechanical public-input asset subtype fixture.",
                )
            )
    return assignments


def _assignments_sha(assignments: list[CapabilityAssignment]) -> str:
    return hashlib.sha256(canonical_jsonl_bytes(assignments)).hexdigest()


def test_formal_planning_requires_catalog_and_provisional_fallback_is_explicit(
    fake_image_root: Path,
) -> None:
    with pytest.raises(ValueError, match="catalog|资产目录"):
        build_dev_mini_plan(fake_image_root, seed=20260711)

    catalog = _catalog_for(fake_image_root)
    with pytest.raises(ValueError, match="capability assignment"):
        build_dev_mini_plan(
            fake_image_root,
            seed=20260711,
            asset_catalog=catalog,
        )

    provisional_dev = build_dev_mini_plan(
        fake_image_root,
        seed=20260711,
        allow_provisional_asset_groups=True,
    )
    pools = _full_pools()
    with pytest.raises(ValueError, match="catalog|资产目录"):
        extend_full_plan(provisional_dev, pools, seed=20260711)

    provisional_full = extend_full_plan(
        provisional_dev,
        pools,
        seed=20260711,
        allow_provisional_asset_groups=True,
    )
    assert len(provisional_full.queries) == 4500


def test_dev_plan_and_manifest_bind_catalog_and_policy(
    fake_image_root: Path, tmp_path: Path
) -> None:
    catalog = _catalog_for(fake_image_root)
    assignments = _capability_assignments(catalog)

    plan = build_dev_mini_plan(
        fake_image_root,
        seed=20260711,
        asset_catalog=catalog,
        capability_assignments=assignments,
        expected_capability_assignments_sha256=_assignments_sha(assignments),
    )
    audit_plan_catalog_binding(plan, catalog)

    assert Counter(item.canonical_capability for item in plan.queries) == (
        DEV_MINI_CAPABILITY_COUNTS
    )
    assert {item.task_spec_version for item in plan.queries} == {
        PHASE3_TASK_SPEC_VERSION
    }
    assert PHASE3_TASK_SPEC_SHA256 == (
        "f8d5596de5d0ba98235f82c7c176a5b774b33d7bdd7e84fb00a07b5b0b7a7f0d"
    )
    assert {
        assignment.allowed_tools
        for assignment in assignments
        if assignment.canonical_capability == "product.multi_search"
    } == {("multi_product_search",)}
    assert plan.asset_catalog_sha256 == catalog.manifest.catalog_sha256
    assert plan.leakage_policy_version == catalog.manifest.leakage_policy_version
    plan_path, manifest_path = write_dev_mini_plan(plan, tmp_path / "dev-mini.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["asset_catalog_sha256"] == catalog.manifest.catalog_sha256
    assert manifest["leakage_policy_version"] == catalog.manifest.leakage_policy_version
    with pytest.raises(ValueError, match="catalog"):
        activate_dev_plan(plan_path, tmp_path / "queries-missing-catalog")
    pointer = activate_dev_plan(
        plan_path,
        tmp_path / "queries",
        asset_catalog=catalog,
    )
    assert pointer.asset_catalog_sha256 == catalog.manifest.catalog_sha256
    assert pointer.leakage_policy_version == catalog.manifest.leakage_policy_version


def test_formal_planner_rejects_missing_and_invalid_capability_assignments(
    fake_image_root: Path,
) -> None:
    catalog = _catalog_for(fake_image_root)
    assignments = _capability_assignments(catalog)
    baseline = build_dev_mini_plan(
        fake_image_root,
        seed=20260711,
        asset_catalog=catalog,
        capability_assignments=assignments,
        expected_capability_assignments_sha256=_assignments_sha(assignments),
    )
    selected = baseline.queries[0]
    missing = [
        assignment
        for assignment in assignments
        if (
            assignment.asset_id,
            assignment.image_path,
            assignment.canonical_intent,
        )
        != (selected.asset_id, selected.image_path, selected.canonical_intent)
    ]
    with pytest.raises(ValueError, match="missing capability assignment"):
        build_dev_mini_plan(
            fake_image_root,
            seed=20260711,
            asset_catalog=catalog,
            capability_assignments=missing,
            expected_capability_assignments_sha256=_assignments_sha(missing),
        )

    first = assignments[0]
    bad_records = (
        first.model_copy(
            update={
                "acceptable_capabilities": (
                    first.canonical_capability,
                    "utility.recipe_guidance",
                )
            }
        ),
        first.model_copy(update={"requires_card": not first.requires_card}),
        first.model_copy(update={"allowed_tools": ("document_ocr",)}),
        first.model_copy(update={"rationale": "selected from a judge result"}),
        first.model_copy(update={"taxonomy_sha256": "0" * 64}),
    )
    for bad_record in bad_records:
        polluted = [bad_record, *assignments[1:]]
        with pytest.raises(ValueError, match="capability|requires_card|allowed_tools"):
            build_dev_mini_plan(
                fake_image_root,
                seed=20260711,
                asset_catalog=catalog,
                capability_assignments=polluted,
                expected_capability_assignments_sha256=_assignments_sha(polluted),
            )


def test_sparse_triplet_assignments_preserve_exact_only_candidates(
    fake_image_root: Path,
) -> None:
    pool_sizes = {
        "exact_match": 35,
        "multi_product": 35,
        "divergent_rec": 27,
        "encyclopedia": 27,
        "utility": 60,
    }
    for intent, size in pool_sizes.items():
        for path in sorted((fake_image_root / intent).glob("*.jpg"))[size:]:
            path.unlink()

    catalog = _catalog_for(fake_image_root)
    assignments = _capability_assignments(catalog)
    exact_paths = sorted(
        assignment.image_path
        for assignment in assignments
        if assignment.canonical_intent == "exact_match"
    )
    triplet_paths = set(exact_paths[:8])
    assignments = [
        assignment
        for assignment in assignments
        if not (
            Path(assignment.image_path).parts[-2] == "exact_match"
            and assignment.canonical_intent in {"divergent_rec", "encyclopedia"}
            and assignment.image_path not in triplet_paths
        )
    ]

    assert len(assignments) == 200
    plan = build_dev_mini_plan(
        fake_image_root,
        seed=20260711,
        asset_catalog=catalog,
        capability_assignments=assignments,
        expected_capability_assignments_sha256=_assignments_sha(assignments),
    )

    selected_exact_paths = {
        item.image_path
        for item in plan.queries
        if item.canonical_capability == "product.exact_match"
    }
    selected_triplet_paths = {
        item.image_path
        for item in plan.queries
        if item.boundary_strategy == "cross_intent_triplet"
    }
    assert selected_exact_paths == set(exact_paths)
    assert selected_triplet_paths == triplet_paths
    assert len({item.image_path for item in plan.queries}) == 184


def test_capability_assignment_loader_requires_canonical_complete_jsonl(
    fake_image_root: Path, tmp_path: Path
) -> None:
    assignments = _capability_assignments(_catalog_for(fake_image_root))
    path = tmp_path / "capability-assignments.jsonl"
    path.write_bytes(canonical_jsonl_bytes(assignments))

    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_capability_assignments(path, expected_sha256=expected_sha256) == tuple(
        assignments
    )

    with pytest.raises(ValueError, match="external expected digest"):
        load_capability_assignments(path, expected_sha256="0" * 64)

    path.write_text(
        json.dumps(assignments[0].model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical|JSONL|JSON"):
        load_capability_assignments(path, expected_sha256=expected_sha256)


def test_catalog_lookup_is_fail_closed_for_a_selected_missing_path(
    fake_image_root: Path,
) -> None:
    complete = _catalog_for(fake_image_root)
    assignments = _capability_assignments(complete)
    baseline = build_dev_mini_plan(
        fake_image_root,
        seed=20260711,
        asset_catalog=complete,
        capability_assignments=assignments,
        expected_capability_assignments_sha256=_assignments_sha(assignments),
    )
    missing = cast(_FakeLoadedCatalog, complete).without_path(
        baseline.queries[0].image_path
    )

    with pytest.raises(ValueError, match="catalog|not cataloged|未编目"):
        build_dev_mini_plan(
            fake_image_root,
            seed=20260711,
            asset_catalog=missing,
            capability_assignments=assignments,
            expected_capability_assignments_sha256=_assignments_sha(assignments),
        )


def test_catalog_a_plan_rejects_catalog_b_and_tampered_references(
    fake_image_root: Path,
) -> None:
    catalog_a = _catalog_for(fake_image_root, catalog_sha256="a" * 64)
    catalog_b = _catalog_for(fake_image_root, catalog_sha256="b" * 64)
    assignments = _capability_assignments(catalog_a)
    plan = build_dev_mini_plan(
        fake_image_root,
        seed=20260711,
        asset_catalog=catalog_a,
        capability_assignments=assignments,
        expected_capability_assignments_sha256=_assignments_sha(assignments),
    )

    with pytest.raises(ValueError, match="catalog|hash|绑定"):
        audit_plan_catalog_binding(plan, catalog_b)

    original = plan.queries[0]
    different_image_path = next(
        item.image_path
        for item in plan.queries
        if item.image_path != original.image_path
    )
    for patch in (
        {"asset_id": "asset.test.tampered"},
        {"leakage_group_id": "component.test.tampered"},
        {"image_path": different_image_path},
    ):
        tampered_query = original.model_copy(update=patch)
        tampered = plan.model_copy(
            update={"queries": [tampered_query, *plan.queries[1:]]}
        )
        with pytest.raises(ValueError, match="catalog|reference|引用|not cataloged"):
            audit_plan_catalog_binding(tampered, catalog_a)


def test_full_planner_keeps_every_catalog_component_in_one_generator_batch(
    fake_image_root: Path,
) -> None:
    pools = _full_pools()
    # Exact Match needs almost every one of these 390 components.  Ten
    # components deliberately have two distinct paths, so a path-based planner
    # would select component siblings independently and scatter them over batches.
    shared: dict[str, str] = {}
    exact_paths = pools["exact_match"]
    for pair_index in range(10):
        component = f"component.test.shared-{pair_index + 1:03d}"
        for path in exact_paths[pair_index * 2 : pair_index * 2 + 2]:
            shared[path.as_posix()] = component
    catalog = _catalog_for(fake_image_root, pools, components=shared)
    assignments = _capability_assignments(catalog)
    dev_plan = build_dev_mini_plan(
        fake_image_root,
        seed=20260711,
        asset_catalog=catalog,
        capability_assignments=assignments,
        expected_capability_assignments_sha256=_assignments_sha(assignments),
    )

    full_plan = extend_full_plan(
        dev_plan,
        pools,
        seed=20260711,
        asset_catalog=catalog,
        capability_assignments=assignments,
        expected_capability_assignments_sha256=_assignments_sha(assignments),
    )
    audit_plan_catalog_binding(full_plan, catalog)

    batches_by_component: dict[str, set[str]] = defaultdict(set)
    selected_shared_components: set[str] = set()
    for query in full_plan.queries:
        component = catalog.resolve_path(query.image_path).leakage_group_id
        batches_by_component[component].add(query.generator_batch_id)
        if component.startswith("component.test.shared-"):
            selected_shared_components.add(component)

    assert selected_shared_components
    assert all(len(batch_ids) == 1 for batch_ids in batches_by_component.values())
    assert full_plan.asset_catalog_sha256 == catalog.manifest.catalog_sha256
    assert full_plan.leakage_policy_version == catalog.manifest.leakage_policy_version
    assert {
        item.canonical_capability
        for item in full_plan.queries
        if item.canonical_intent == "utility"
    } == {"utility.document_reading", "utility.recipe_guidance"}
