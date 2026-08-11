"""Fail-closed catalog binding tests for full split loading and freezing."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillchain.schemas import LabelDecision, Query
from skillchain.synthesis.splitting import (
    FrozenSplitError,
    SplitSpec,
    freeze_test_split,
    load_labeled_queries_for_split,
    verify_frozen_split,
)
from skillchain.synthesis.planning import PHASE3_TASK_SPEC_VERSION
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from skillchain.taxonomy import (
    TAXONOMY_VERSION,
    capability_for_intent,
    requires_card_for_intent,
)


CATALOG_SHA = "c" * 64
FULL_PLAN_SHA = "f" * 64
LEAKAGE_POLICY = "dataset-asset-components-v1"


@dataclass
class _FakeCatalog:
    references: dict[str, tuple[str, str]]
    catalog_sha256: str = CATALOG_SHA
    leakage_policy_version: str = LEAKAGE_POLICY
    rejected_asset_id: str | None = None

    def __post_init__(self):
        self.manifest = SimpleNamespace(
            catalog_sha256=self.catalog_sha256,
            leakage_policy_version=self.leakage_policy_version,
        )

    def verify_reference(
        self,
        asset_id: str,
        image_path: str | Path,
        leakage_group_id: str | None = None,
    ):
        if asset_id == self.rejected_asset_id:
            raise ValueError("mechanically rejected catalog reference")
        try:
            expected_path, expected_group = self.references[asset_id]
        except KeyError:
            raise ValueError(f"unknown asset_id: {asset_id}") from None
        if Path(image_path).as_posix() != expected_path:
            raise ValueError("image_path mismatch")
        if leakage_group_id is not None and leakage_group_id != expected_group:
            raise ValueError("leakage_group_id mismatch")
        return SimpleNamespace(
            asset_id=asset_id,
            local_path=expected_path,
            leakage_group_id=expected_group,
        )

    def require_verified_files(self) -> None:
        return None

    def verify_asset_ids(self, asset_ids) -> None:
        return None


def _bind_labeled_fixture_to_catalog(fixture) -> tuple[_FakeCatalog, str]:
    """Upgrade the mechanical full plan binding without changing query triples."""

    plan_path = fixture.full_plan_path
    manifest_path = plan_path.with_name(f"{plan_path.stem}.manifest.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["asset_catalog_sha256"] = CATALOG_SHA
    plan["capability_assignments_sha256"] = "a" * 64
    plan["leakage_policy_version"] = LEAKAGE_POLICY
    for item in plan["queries"]:
        item["capability_assignment_id"] = f"fixture.{item['plan_id']}"
        item["capability_assignment_source_sha256"] = "b" * 64
    plan_bytes = canonical_json_bytes(plan)
    plan_sha = sha256_bytes(plan_bytes)
    plan_path.write_bytes(plan_bytes)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["plan_sha256"] = plan_sha
    manifest["asset_catalog_sha256"] = CATALOG_SHA
    manifest["capability_assignments_sha256"] = "a" * 64
    manifest["leakage_policy_version"] = LEAKAGE_POLICY
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    labels_manifest_path = fixture.queries_root / "labels" / "manifest.json"
    labels_manifest = json.loads(labels_manifest_path.read_text(encoding="utf-8"))
    labels_manifest["plan_sha256"] = plan_sha
    labels_manifest["asset_catalog_sha256"] = CATALOG_SHA
    labels_manifest["leakage_policy_version"] = LEAKAGE_POLICY
    labels_manifest_path.write_bytes(canonical_json_bytes(labels_manifest))

    ledger_path = fixture.queries_root / "accepted-ledger.jsonl"
    ledger = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    for entry in ledger:
        if not entry["base_batch_id"].startswith("dev-mini-"):
            entry["plan_sha256"] = plan_sha
        batch_manifest_path = (
            fixture.queries_root / "accepted" / entry["batch_id"] / "manifest.json"
        )
        batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
        batch_manifest["plan_sha256"] = entry["plan_sha256"]
        if not entry["base_batch_id"].startswith("dev-mini-"):
            batch_manifest["asset_catalog_sha256"] = CATALOG_SHA
            batch_manifest["leakage_policy_version"] = LEAKAGE_POLICY
        batch_manifest_path.write_bytes(canonical_json_bytes(batch_manifest))
    ledger_path.write_bytes(canonical_jsonl_bytes(ledger))

    references = {
        item["asset_id"]: (item["image_path"], item["leakage_group_id"])
        for item in plan["queries"]
    }
    return _FakeCatalog(references), plan_sha


def test_catalog_bound_full_plan_rejects_missing_and_wrong_catalog(
    labeled_split_fixture,
):
    catalog, _ = _bind_labeled_fixture_to_catalog(labeled_split_fixture)

    with pytest.raises(ValueError, match="缺少.*asset_catalog|缺少匹配"):
        load_labeled_queries_for_split(
            labeled_split_fixture.queries_root,
            labeled_split_fixture.full_plan_path,
        )

    wrong_catalog = _FakeCatalog(
        catalog.references,
        catalog_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="catalog hash|hash.*plan"):
        load_labeled_queries_for_split(
            labeled_split_fixture.queries_root,
            labeled_split_fixture.full_plan_path,
            asset_catalog=wrong_catalog,
        )

    candidates, locked_ids = load_labeled_queries_for_split(
        labeled_split_fixture.queries_root,
        labeled_split_fixture.full_plan_path,
        asset_catalog=catalog,
    )
    assert len(candidates) == 4500
    assert len(locked_ids) == 200


def test_catalog_bound_full_plan_checks_each_query_reference(labeled_split_fixture):
    catalog, _ = _bind_labeled_fixture_to_catalog(labeled_split_fixture)
    rejected = next(iter(catalog.references))
    broken = _FakeCatalog(
        catalog.references,
        rejected_asset_id=rejected,
    )

    with pytest.raises(ValueError, match="catalog 引用不一致"):
        load_labeled_queries_for_split(
            labeled_split_fixture.queries_root,
            labeled_split_fixture.full_plan_path,
            asset_catalog=broken,
        )


def _query(index: int, split: str) -> Query:
    intent = "exact_match"
    capability = capability_for_intent(intent)
    text = f"mechanical catalog split query {index}"
    decision = LabelDecision(
        decision_type="cross_review",
        annotator_kind="llm_pair",
        annotator_id="mechanical-review",
        canonical_intent=intent,
        canonical_capability=capability,
        acceptable_capabilities=[capability],
        reason="mechanical catalog binding test",
    )
    return Query(
        schema_version=2,
        taxonomy_version=TAXONOMY_VERSION,
        task_spec_version=PHASE3_TASK_SPEC_VERSION,
        query_id=f"q-{index}",
        asset_id=f"asset-{index}",
        image_path=f"images/{index}.png",
        leakage_group_id=f"component-{index}",
        boundary_group_id=None,
        template_family=f"template-{index}",
        generator_batch_id=f"batch-{index}",
        text=text,
        turns=[{"role": "user", "content": text}],
        canonical_intent=intent,
        canonical_capability=capability,
        acceptable_capabilities=[capability],
        is_boundary=False,
        boundary_strategy=None,
        requires_card=requires_card_for_intent(intent),
        split=split,
        label_status="cross_agreed",
        label_provenance=[decision],
    )


def _tiny_assignment() -> tuple[list[Query], SplitSpec, _FakeCatalog]:
    queries = [
        _query(1, "dev_mini"),
        _query(2, "opt_pool"),
        _query(3, "val"),
        _query(4, "test_frozen"),
    ]
    spec = SplitSpec(
        profile="catalog-binding-test",
        sizes={"dev_mini": 1, "opt_pool": 1, "val": 1, "test_frozen": 1},
        min_test_per_intent=0,
    )
    references = {
        query.asset_id: (query.image_path, query.leakage_group_id) for query in queries
    }
    return queries, spec, _FakeCatalog(references)


def test_freeze_binds_and_rechecks_catalog_and_full_plan(tmp_path: Path):
    assigned, spec, catalog = _tiny_assignment()
    root = tmp_path / "queries"

    freeze_test_split(
        root,
        assigned,
        seed=20260711,
        full_plan_sha256=FULL_PLAN_SHA,
        asset_catalog=catalog,
        split_spec=spec,
    )
    manifest = verify_frozen_split(
        root,
        asset_catalog=catalog,
        expected_full_plan_sha256=FULL_PLAN_SHA,
    )

    assert manifest is not None
    assert manifest.schema_version == 3
    assert manifest.full_plan_sha256 == FULL_PLAN_SHA
    assert manifest.asset_catalog_sha256 == CATALOG_SHA
    assert manifest.leakage_policy_version == LEAKAGE_POLICY

    with pytest.raises(FrozenSplitError, match="asset catalog"):
        verify_frozen_split(root)
    with pytest.raises(FrozenSplitError, match="full plan hash"):
        verify_frozen_split(
            root,
            asset_catalog=catalog,
            expected_full_plan_sha256="e" * 64,
        )
    with pytest.raises(FrozenSplitError, match="catalog hash"):
        verify_frozen_split(
            root,
            asset_catalog=_FakeCatalog(catalog.references, catalog_sha256="d" * 64),
        )


def test_freeze_and_verify_fail_closed_on_catalog_reference_tampering(
    tmp_path: Path,
):
    assigned, spec, catalog = _tiny_assignment()
    tampered = list(assigned)
    payload = tampered[1].model_dump(mode="json")
    payload["leakage_group_id"] = "component-tampered"
    tampered[1] = Query.model_validate(payload)

    with pytest.raises(ValueError, match="catalog 引用不一致"):
        freeze_test_split(
            tmp_path / "pre-freeze",
            tampered,
            seed=20260711,
            full_plan_sha256=FULL_PLAN_SHA,
            asset_catalog=catalog,
            split_spec=spec,
        )

    root = tmp_path / "post-freeze"
    freeze_test_split(
        root,
        assigned,
        seed=20260711,
        full_plan_sha256=FULL_PLAN_SHA,
        asset_catalog=catalog,
        split_spec=spec,
    )
    assignment_path = root / "split_assignment.jsonl"
    manifest_path = root / "test_frozen.manifest.json"
    os.chmod(assignment_path, stat.S_IWRITE | stat.S_IREAD)
    os.chmod(manifest_path, stat.S_IWRITE | stat.S_IREAD)
    assignment_bytes = canonical_jsonl_bytes(tampered)
    assignment_path.write_bytes(assignment_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assignment_sha256"] = hashlib.sha256(assignment_bytes).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    os.chmod(assignment_path, stat.S_IREAD)
    os.chmod(manifest_path, stat.S_IREAD)

    with pytest.raises(FrozenSplitError, match="catalog 引用不一致"):
        verify_frozen_split(root, asset_catalog=catalog)


def test_freeze_requires_catalog_unless_provisional_mode_is_explicit(tmp_path: Path):
    assigned, spec, _ = _tiny_assignment()
    with pytest.raises(ValueError, match="必须绑定 asset catalog"):
        freeze_test_split(
            tmp_path / "closed",
            assigned,
            seed=20260711,
            full_plan_sha256=FULL_PLAN_SHA,
            split_spec=spec,
        )

    root = tmp_path / "provisional"
    freeze_test_split(
        root,
        assigned,
        seed=20260711,
        full_plan_sha256=FULL_PLAN_SHA,
        allow_provisional_asset_groups=True,
        split_spec=spec,
    )
    with pytest.raises(FrozenSplitError, match="provisional"):
        verify_frozen_split(root)
    manifest = verify_frozen_split(root, allow_provisional_asset_groups=True)
    assert manifest is not None
    assert manifest.asset_catalog_sha256 is None
    assert manifest.leakage_policy_version == "relative-path-plus-plan-groups-v1"
