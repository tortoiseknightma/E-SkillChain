from __future__ import annotations

from collections import Counter
from pathlib import Path
from random import Random

from PIL import Image
import pytest

from skillchain.data.asset_catalog import (
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)
from skillchain.data.portfolio_core_assets import (
    CORE_R2_CAPABILITY_ASSET_FLOORS,
    CandidateCapabilityBinding,
    PortfolioCoreAssetCandidate,
    PortfolioCoreAssetError,
    PortfolioCoreReserveActivationEntry,
    PortfolioCoreReserveActivationManifest,
    PortfolioCoreSourcePolicy,
    PortfolioCoreSourcePolicyDocument,
    build_portfolio_core_capability_assignments,
    build_portfolio_core_v9_multi_capability_assignments,
    build_portfolio_core_v9_multi_closure_catalog_receipt,
    build_portfolio_core_v9_multi_closure_inventory,
    build_portfolio_core_v8_capability_assignments,
    build_portfolio_core_v8_capability_binding_plan,
    default_source_policy_bytes,
    load_candidate_inventory,
    load_portfolio_core_capability_binding_plan_manifest,
    load_portfolio_core_v9_multi_closure_inventory_manifest,
    load_source_policy,
    materialize_portfolio_core_assets,
    preflight_portfolio_core_assets,
)
from skillchain.data.asset_catalog_cli import load_drafts
from skillchain.synthesis.models import CorpusPlan, PlannedQuery
from skillchain.synthesis.planning import (
    load_capability_assignments,
    write_core_plan,
)
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


def test_r2_capability_floors_are_unique_candidate_capacity_not_query_quota() -> None:
    """Keep r2 preflight aligned with the controlled-reuse planning contract."""

    assert CORE_R2_CAPABILITY_ASSET_FLOORS == {
        "product.exact_match": 340,
        "product.multi_search": 105,
        "product.style_recommendation": 150,
        "knowledge.visual_encyclopedia": 152,
        "utility.document_reading": 90,
        "utility.recipe_guidance": 210,
    }


def _write_image(path: Path, seed: int) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(
        "RGB",
        (16, 16),
        color=(seed % 251, (seed * 7) % 251, (seed * 19) % 251),
    )
    for index in range(16):
        image.putpixel(
            (index, (seed + index * 3) % 16),
            ((seed * 11) % 251, index * 13, (seed * 17) % 251),
        )
    image.save(path, format="PNG")
    return path.read_bytes()


def _candidate(
    *,
    candidate_id: str,
    source_id: str,
    source_local_path: str,
    destination_path: str,
    content: bytes,
    pool: str,
    intent: str,
    capability: str,
    selection: str = "include",
    capability_bindings: tuple[CandidateCapabilityBinding, ...] | None = None,
) -> PortfolioCoreAssetCandidate:
    from skillchain.data.asset_catalog import DatasetAssetDraft

    return PortfolioCoreAssetCandidate(
        candidate_id=candidate_id,
        source_id=source_id,
        selection=selection,
        pool=pool,
        source_local_path=source_local_path,
        destination_path=destination_path,
        expected_bytes=len(content),
        expected_sha256=sha256_bytes(content),
        draft=DatasetAssetDraft(
            source_dataset=source_id,
            source_revision="fixture-v1",
            source_record_id=f"record:{candidate_id}",
            transform_policy_version="fixture-copy-v1",
            local_path=source_local_path,
            product_id=(None if pool == "utility" else f"{source_id}:{candidate_id}"),
            license_id="LicenseRef-Fixture",
            cloud_upload_allowed=False,
            public_demo_allowed=False,
        ),
        capability_bindings=(
            (
                CandidateCapabilityBinding(
                    canonical_intent=intent,
                    canonical_capability=capability,
                ),
            )
            if capability_bindings is None
            else capability_bindings
        ),
    )


def _write_inventory(path: Path, candidates: list[PortfolioCoreAssetCandidate]) -> None:
    path.write_bytes(canonical_jsonl_bytes(candidates))


def _write_policy(
    path: Path,
    *,
    deepfashion: str = "ready",
    products: str = "ready",
) -> None:
    policy = PortfolioCoreSourcePolicyDocument(
        sources=(
            PortfolioCoreSourcePolicy(
                source_id="deepfashion",
                availability=deepfashion,
                allowed_pools=(),
                note="Fixture DeepFashion policy.",
            ),
            PortfolioCoreSourcePolicy(
                source_id="polyvore",
                availability="ready",
                allowed_pools=("divergent_rec",),
                note="Fixture Polyvore policy.",
            ),
            PortfolioCoreSourcePolicy(
                source_id="products_10k",
                availability=products,
                allowed_pools=("exact_match",),
                note="Fixture Products-10K policy.",
            ),
            PortfolioCoreSourcePolicy(
                source_id="recipe1m_plus",
                availability="ready",
                allowed_pools=("utility",),
                note="Fixture Recipe1M policy.",
            ),
        )
    )
    path.write_bytes(canonical_json_bytes(policy))


def _prepared_inputs(tmp_path: Path) -> dict[str, object]:
    source_root = tmp_path / "sources"
    recipe = _write_image(source_root / "recipe1m_plus" / "recipe.png", 1)
    polyvore = _write_image(source_root / "polyvore" / "look.png", 2)
    product = _write_image(source_root / "products_10k" / "item.png", 3)
    reserve = _write_image(source_root / "deepfashion" / "future.png", 4)
    candidates = [
        _candidate(
            candidate_id="fixture.recipe.001",
            source_id="recipe1m_plus",
            source_local_path="recipe.png",
            destination_path="query_images/utility/recipe-001.png",
            content=recipe,
            pool="utility",
            intent="utility",
            capability="utility.recipe_guidance",
        ),
        _candidate(
            candidate_id="fixture.polyvore.001",
            source_id="polyvore",
            source_local_path="look.png",
            destination_path="query_images/divergent_rec/look-001.png",
            content=polyvore,
            pool="divergent_rec",
            intent="divergent_rec",
            capability="product.style_recommendation",
        ),
        _candidate(
            candidate_id="fixture.products.001",
            source_id="products_10k",
            source_local_path="item.png",
            destination_path="query_images/exact_match/item-001.png",
            content=product,
            pool="exact_match",
            intent="exact_match",
            capability="product.exact_match",
        ),
        _candidate(
            candidate_id="fixture.deepfashion.reserve",
            source_id="deepfashion",
            source_local_path="future.png",
            destination_path="query_images/divergent_rec/future-001.png",
            content=reserve,
            pool="divergent_rec",
            intent="divergent_rec",
            capability="product.style_recommendation",
            selection="reserve",
        ),
    ]
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(inventory, candidates)
    policy = tmp_path / "policy.json"
    _write_policy(policy, deepfashion="pending_extraction")
    return {
        "candidates": candidates,
        "inventory": inventory,
        "output_root": tmp_path / "clean",
        "policy": policy,
        "source_roots": {
            "deepfashion": source_root / "deepfashion",
            "polyvore": source_root / "polyvore",
            "recipe1m_plus": source_root / "recipe1m_plus",
            "products_10k": source_root / "products_10k",
        },
    }


def test_default_policy_explicitly_marks_pending_and_locked_sources(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / "default-policy.json"
    policy_path.write_bytes(default_source_policy_bytes())
    policy, _ = load_source_policy(policy_path)
    assert policy.policy_version == "portfolio-core-source-pool-policy-v3"
    by_source = {item.source_id: item for item in policy.sources}
    assert by_source["abo"].allowed_pools == ("exact_match",)
    assert by_source["rpc"].allowed_pools == ("multi_product",)
    assert by_source["fashioniq"].allowed_pools == ("divergent_rec",)
    assert by_source["inaturalist"].allowed_pools == ("encyclopedia",)
    assert by_source["isia_food500"].allowed_pools == ("utility",)
    assert by_source["cord"].allowed_pools == ("utility",)
    assert by_source["sroie"].allowed_pools == ("utility",)
    assert by_source["wikimedia_commons_documents"].allowed_pools == ("utility",)
    assert by_source["deepfashion"].allowed_pools == ()
    assert by_source["sku_110k"].availability == "restricted"
    assert by_source["mep3m"].availability == "restricted"
    deep_root = tmp_path / "deepfashion"
    products_root = tmp_path / "products"
    deep = _write_image(deep_root / "look.png", 11)
    product = _write_image(products_root / "item.png", 12)
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(
        inventory,
        [
            _candidate(
                candidate_id="fixture.deepfashion.001",
                source_id="deepfashion",
                source_local_path="look.png",
                destination_path="query_images/divergent_rec/look.png",
                content=deep,
                pool="divergent_rec",
                intent="divergent_rec",
                capability="product.style_recommendation",
            ),
            _candidate(
                candidate_id="fixture.products.001",
                source_id="products_10k",
                source_local_path="item.png",
                destination_path="query_images/exact_match/item.png",
                content=product,
                pool="exact_match",
                intent="exact_match",
                capability="product.exact_match",
            ),
        ],
    )

    result = preflight_portfolio_core_assets(
        inventory_path=inventory,
        source_policy_path=policy_path,
        source_roots={"deepfashion": deep_root, "products_10k": products_root},
    )

    assert not result.ready
    assert result.blocked_candidates == {
        "deepfashion": ("pending_extraction",),
        "products_10k": ("locked_archive",),
    }
    assert result.source_statuses["u_need"] == "permanently_unavailable"
    with pytest.raises(PortfolioCoreAssetError, match="pending_extraction"):
        materialize_portfolio_core_assets(
            inventory_path=inventory,
            source_policy_path=policy_path,
            source_roots={"deepfashion": deep_root, "products_10k": products_root},
            output_root=tmp_path / "clean",
        )
    assert not (tmp_path / "clean").exists()


def test_materialization_checkpoints_resume_and_publish_drafts(tmp_path: Path) -> None:
    arguments = _prepared_inputs(tmp_path)

    preflight = preflight_portfolio_core_assets(
        inventory_path=arguments["inventory"],
        source_policy_path=arguments["policy"],
        source_roots=arguments["source_roots"],
    )
    assert preflight.source_ready
    assert not preflight.capacity_ready
    assert not preflight.ready
    assert preflight.selected_candidate_count == 3
    assert preflight.reserve_candidate_count == 1
    assert preflight.selected_pool_counts == {
        "divergent_rec": 1,
        "exact_match": 1,
        "utility": 1,
    }
    assert preflight.capacity_shortfalls["exact_match"] == 156
    with pytest.raises(PortfolioCoreAssetError, match="capacity floor"):
        materialize_portfolio_core_assets(
            inventory_path=arguments["inventory"],
            source_policy_path=arguments["policy"],
            source_roots=arguments["source_roots"],
            output_root=arguments["output_root"],
        )
    assert not arguments["output_root"].exists()

    first = materialize_portfolio_core_assets(
        inventory_path=arguments["inventory"],
        source_policy_path=arguments["policy"],
        source_roots=arguments["source_roots"],
        output_root=arguments["output_root"],
        max_items=1,
        allow_incomplete_capacity=True,
    )
    assert not first.complete
    assert first.copied_this_call == 1
    assert first.checkpointed_candidate_count == 1
    assert not (arguments["output_root"] / "selection-manifest.json").exists()

    resumed = materialize_portfolio_core_assets(
        inventory_path=arguments["inventory"],
        source_policy_path=arguments["policy"],
        source_roots=arguments["source_roots"],
        output_root=arguments["output_root"],
        allow_incomplete_capacity=True,
    )
    assert resumed.complete
    assert resumed.copied_this_call == 2
    assert resumed.checkpointed_candidate_count == 3
    assert resumed.dataset_assets_path is not None
    drafts = load_drafts(resumed.dataset_assets_path)
    assert {draft.local_path for draft in drafts} == {
        "query_images/divergent_rec/look-001.png",
        "query_images/exact_match/item-001.png",
        "query_images/utility/recipe-001.png",
    }
    assert all(draft.local_path.startswith("query_images/") for draft in drafts)

    repeated = materialize_portfolio_core_assets(
        inventory_path=arguments["inventory"],
        source_policy_path=arguments["policy"],
        source_roots=arguments["source_roots"],
        output_root=arguments["output_root"],
        allow_incomplete_capacity=True,
    )
    assert repeated.complete
    assert repeated.copied_this_call == 0
    assert repeated.checkpointed_candidate_count == 3


def test_materialization_blocks_a_second_writer_until_lock_is_explicitly_resolved(
    tmp_path: Path,
) -> None:
    arguments = _prepared_inputs(tmp_path)
    first = materialize_portfolio_core_assets(
        inventory_path=arguments["inventory"],
        source_policy_path=arguments["policy"],
        source_roots=arguments["source_roots"],
        output_root=arguments["output_root"],
        max_items=1,
        allow_incomplete_capacity=True,
    )
    assert first.checkpointed_candidate_count == 1
    lock_path = arguments["output_root"] / "portfolio-core-materialize.writer.lock"
    assert not lock_path.exists()

    retained_lock = canonical_json_bytes(
        {"owner_pid": 999_999, "owner_token": "simulated-active-worker"}
    )
    lock_path.write_bytes(retained_lock)
    with pytest.raises(PortfolioCoreAssetError, match="writer lock already exists"):
        materialize_portfolio_core_assets(
            inventory_path=arguments["inventory"],
            source_policy_path=arguments["policy"],
            source_roots=arguments["source_roots"],
            output_root=arguments["output_root"],
            max_items=1,
            allow_incomplete_capacity=True,
        )

    assert lock_path.read_bytes() == retained_lock
    assert len(list((arguments["output_root"] / "checkpoints").glob("*.json"))) == 1


def test_reserve_activation_manifest_is_bound_and_fail_closed(tmp_path: Path) -> None:
    arguments = _prepared_inputs(tmp_path)
    candidates = arguments["candidates"]
    source_roots = arguments["source_roots"]
    assert isinstance(candidates, list)
    assert isinstance(source_roots, dict)
    recipe_root = source_roots["recipe1m_plus"]
    assert isinstance(recipe_root, Path)
    reserve_content = _write_image(recipe_root / "reserve.png", 25)
    activated_reserve = _candidate(
        candidate_id="fixture.recipe.reserve",
        source_id="recipe1m_plus",
        source_local_path="reserve.png",
        destination_path="query_images/utility/recipe-reserve.png",
        content=reserve_content,
        pool="utility",
        intent="utility",
        capability="utility.recipe_guidance",
        selection="reserve",
    )
    inventory = arguments["inventory"]
    assert isinstance(inventory, Path)
    _write_inventory(inventory, [*candidates, activated_reserve])
    _loaded, inventory_sha256 = load_candidate_inventory(inventory)

    activation = PortfolioCoreReserveActivationManifest(
        parent_inventory_sha256=inventory_sha256,
        parent_selection_manifest_sha256="a" * 64,
        parent_catalog_sha256="b" * 64,
        baseline_component_count=0,
        required_component_floor=1,
        closure_margin=0,
        predicted_component_count=1,
        reason="Fixture reserve activation.",
        selections=(
            PortfolioCoreReserveActivationEntry(
                candidate_id=activated_reserve.candidate_id,
                source_id=activated_reserve.source_id,
                destination_path=activated_reserve.destination_path,
                expected_sha256=activated_reserve.expected_sha256,
            ),
        ),
    )
    activation_path = tmp_path / "reserve-activation.json"
    activation_path.write_bytes(canonical_json_bytes(activation))

    preflight = preflight_portfolio_core_assets(
        inventory_path=inventory,
        source_policy_path=arguments["policy"],
        source_roots=source_roots,
        reserve_activation_manifest_path=activation_path,
    )
    assert preflight.source_ready
    assert preflight.selected_candidate_count == 4
    assert preflight.reserve_candidate_count == 1
    assert preflight.activated_reserve_candidate_count == 1
    assert preflight.reserve_activation_manifest_sha256 == sha256_bytes(
        activation_path.read_bytes()
    )

    output_root = tmp_path / "activated-clean"
    result = materialize_portfolio_core_assets(
        inventory_path=inventory,
        source_policy_path=arguments["policy"],
        source_roots=source_roots,
        output_root=output_root,
        allow_incomplete_capacity=True,
        reserve_activation_manifest_path=activation_path,
    )
    assert result.complete
    assert result.dataset_assets_path is not None
    drafts = load_drafts(result.dataset_assets_path)
    assert {draft.local_path for draft in drafts} == {
        "query_images/divergent_rec/look-001.png",
        "query_images/exact_match/item-001.png",
        "query_images/utility/recipe-001.png",
        "query_images/utility/recipe-reserve.png",
    }

    tampered = activation.model_copy(
        update={
            "selections": (
                activation.selections[0].model_copy(
                    update={"expected_sha256": "0" * 64}
                ),
            )
        }
    )
    activation_path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(PortfolioCoreAssetError, match="binding drifted"):
        preflight_portfolio_core_assets(
            inventory_path=inventory,
            source_policy_path=arguments["policy"],
            source_roots=source_roots,
            reserve_activation_manifest_path=activation_path,
        )

    omitted = activation.model_dump(mode="json")
    omitted["selections"] = []
    activation_path.write_bytes(canonical_json_bytes(omitted))
    with pytest.raises(PortfolioCoreAssetError, match="violates schema"):
        preflight_portfolio_core_assets(
            inventory_path=inventory,
            source_policy_path=arguments["policy"],
            source_roots=source_roots,
            reserve_activation_manifest_path=activation_path,
        )


def test_checkpoint_recovers_after_asset_write_before_checkpoint(
    tmp_path: Path,
) -> None:
    arguments = _prepared_inputs(tmp_path)
    first = materialize_portfolio_core_assets(
        inventory_path=arguments["inventory"],
        source_policy_path=arguments["policy"],
        source_roots=arguments["source_roots"],
        output_root=arguments["output_root"],
        max_items=1,
        allow_incomplete_capacity=True,
    )
    assert first.checkpointed_candidate_count == 1
    checkpoint = next((arguments["output_root"] / "checkpoints").glob("*.json"))
    checkpoint.unlink()

    resumed = materialize_portfolio_core_assets(
        inventory_path=arguments["inventory"],
        source_policy_path=arguments["policy"],
        source_roots=arguments["source_roots"],
        output_root=arguments["output_root"],
        max_items=1,
        allow_incomplete_capacity=True,
    )
    assert resumed.checkpointed_candidate_count == 1
    assert resumed.copied_this_call == 1
    assert len(list((arguments["output_root"] / "checkpoints").glob("*.json"))) == 1


def test_builds_catalog_bound_capability_assignments(tmp_path: Path) -> None:
    arguments = _prepared_inputs(tmp_path)
    result = materialize_portfolio_core_assets(
        inventory_path=arguments["inventory"],
        source_policy_path=arguments["policy"],
        source_roots=arguments["source_roots"],
        output_root=arguments["output_root"],
        allow_incomplete_capacity=True,
    )
    assert result.selection_manifest_path is not None
    assert result.dataset_assets_path is not None
    drafts = load_drafts(result.dataset_assets_path)
    assets = tuple(
        inventory_dataset_asset(draft, arguments["output_root"]) for draft in drafts
    )
    catalog_root = tmp_path / "catalog"
    publish_asset_catalog(
        assets,
        catalog_root,
        arguments["output_root"],
        coverage_roots=("query_images",),
    )
    assignment_path = tmp_path / "assignments.jsonl"
    assignment = build_portfolio_core_capability_assignments(
        selection_manifest_path=result.selection_manifest_path,
        asset_catalog_dir=catalog_root,
        asset_root=arguments["output_root"],
        output_path=assignment_path,
    )
    loaded = load_capability_assignments(
        assignment_path,
        expected_sha256=assignment.output_sha256,
    )
    assert assignment.assignment_count == 3
    assert {item.canonical_capability for item in loaded} == {
        "product.exact_match",
        "product.style_recommendation",
        "utility.recipe_guidance",
    }
    assert {item.source_artifact_sha256 for item in loaded} == {
        assignment.selection_manifest_sha256
    }


def _write_binding_image(path: Path, seed: int) -> bytes:
    """Create deliberately non-near-duplicate fixture images."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = Random(seed)
    image = Image.new("RGB", (32, 32))
    image.putdata(
        [
            (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for _ in range(32 * 32)
        ]
    )
    image.save(path, format="PNG")
    return path.read_bytes()


def _prepared_v8_binding_inputs(tmp_path: Path) -> dict[str, Path]:
    """Build a tiny catalog plus a structurally valid r1-sized core plan.

    The fixture has the real v7 baseline shape relevant to this closure: eight
    ABO triplet components in the dev prefix, six in the r1 tail, fourteen
    exact-only ABO tail components, and ten recipe-only ISIA tail components.
    """

    source_root = tmp_path / "sources"
    abo_root = source_root / "abo"
    isia_root = source_root / "isia_food500"
    triplet_bindings = (
        CandidateCapabilityBinding(
            canonical_intent="exact_match",
            canonical_capability="product.exact_match",
        ),
        CandidateCapabilityBinding(
            canonical_intent="divergent_rec",
            canonical_capability="product.style_recommendation",
        ),
        CandidateCapabilityBinding(
            canonical_intent="encyclopedia",
            canonical_capability="knowledge.visual_encyclopedia",
        ),
    )
    candidates: list[PortfolioCoreAssetCandidate] = []
    dev_triplet_paths: list[str] = []
    tail_paths: list[str] = []

    for index in range(1, 15):
        source_local_path = f"triplet-{index:03d}.png"
        destination_path = f"query_images/exact_match/abo-triplet-{index:03d}.png"
        content = _write_binding_image(abo_root / source_local_path, index)
        candidates.append(
            _candidate(
                candidate_id=f"fixture.abo.triplet.{index:03d}",
                source_id="abo",
                source_local_path=source_local_path,
                destination_path=destination_path,
                content=content,
                pool="exact_match",
                intent="exact_match",
                capability="product.exact_match",
                capability_bindings=triplet_bindings,
            )
        )
        if index <= 8:
            dev_triplet_paths.append(destination_path)
        else:
            tail_paths.append(destination_path)

    for index in range(1, 15):
        source_local_path = f"exact-{index:03d}.png"
        destination_path = f"query_images/exact_match/abo-exact-{index:03d}.png"
        content = _write_binding_image(abo_root / source_local_path, 100 + index)
        candidates.append(
            _candidate(
                candidate_id=f"fixture.abo.exact.{index:03d}",
                source_id="abo",
                source_local_path=source_local_path,
                destination_path=destination_path,
                content=content,
                pool="exact_match",
                intent="exact_match",
                capability="product.exact_match",
            )
        )
        tail_paths.append(destination_path)

    for index in range(1, 11):
        source_local_path = f"food-{index:03d}.png"
        destination_path = f"query_images/utility/isia-food-{index:03d}.png"
        content = _write_binding_image(isia_root / source_local_path, 200 + index)
        candidates.append(
            _candidate(
                candidate_id=f"fixture.isia.food.{index:03d}",
                source_id="isia_food500",
                source_local_path=source_local_path,
                destination_path=destination_path,
                content=content,
                pool="utility",
                intent="utility",
                capability="utility.recipe_guidance",
            )
        )
        tail_paths.append(destination_path)

    inventory_path = tmp_path / "inventory.jsonl"
    _write_inventory(inventory_path, candidates)
    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(
        canonical_json_bytes(
            PortfolioCoreSourcePolicyDocument(
                sources=(
                    PortfolioCoreSourcePolicy(
                        source_id="abo",
                        availability="ready",
                        allowed_pools=("exact_match",),
                        note="Fixture ABO source.",
                    ),
                    PortfolioCoreSourcePolicy(
                        source_id="isia_food500",
                        availability="ready",
                        allowed_pools=("utility",),
                        note="Fixture ISIA source.",
                    ),
                )
            )
        )
    )
    asset_root = tmp_path / "clean"
    materialized = materialize_portfolio_core_assets(
        inventory_path=inventory_path,
        source_policy_path=policy_path,
        source_roots={"abo": abo_root, "isia_food500": isia_root},
        output_root=asset_root,
        allow_incomplete_capacity=True,
    )
    assert materialized.selection_manifest_path is not None
    assert materialized.dataset_assets_path is not None
    assets = tuple(
        inventory_dataset_asset(draft, asset_root)
        for draft in load_drafts(materialized.dataset_assets_path)
    )
    catalog_root = tmp_path / "catalog"
    publish_asset_catalog(
        assets,
        catalog_root,
        asset_root,
        coverage_roots=("query_images",),
    )
    catalog = load_asset_catalog(catalog_root, asset_root, verify_files=True)
    assert len(catalog.components) == len(candidates)

    base_assignment_path = tmp_path / "assignments-v7.jsonl"
    build_portfolio_core_capability_assignments(
        selection_manifest_path=materialized.selection_manifest_path,
        asset_catalog_dir=catalog_root,
        asset_root=asset_root,
        output_path=base_assignment_path,
    )
    candidates_by_path = {
        candidate.destination_path: candidate for candidate in candidates
    }
    planned_queries: list[PlannedQuery] = []
    for index in range(1_500):
        if index < 200:
            image_path = dev_triplet_paths[index % len(dev_triplet_paths)]
            split = "dev_mini"
        else:
            image_path = tail_paths[(index - 200) % len(tail_paths)]
            split = "opt_pool"
        candidate = candidates_by_path[image_path]
        resolution = catalog.resolve_path(image_path)
        if candidate.source_id == "abo":
            canonical_intent = "exact_match"
            canonical_capability = "product.exact_match"
        else:
            canonical_intent = "utility"
            canonical_capability = "utility.recipe_guidance"
        batch_number = index // 25 + 1
        batch_id = f"fixture-batch-{batch_number:03d}"
        planned_queries.append(
            PlannedQuery(
                plan_id=f"fixture-r1-{index + 1:04d}",
                batch_id=batch_id,
                position=index % 25 + 1,
                taxonomy_version="fixture-taxonomy-v1",
                task_spec_version="fixture-task-spec-v1",
                asset_id=resolution.asset.asset_id,
                image_path=image_path,
                leakage_group_id=resolution.leakage_group_id,
                template_family=f"fixture-template-{batch_number:03d}",
                generator_batch_id=batch_id,
                canonical_intent=canonical_intent,
                canonical_capability=canonical_capability,
                acceptable_capabilities=[canonical_capability],
                requires_card=False,
                is_boundary=False,
                provisional_split=split,
            )
        )
    r1_plan_path = tmp_path / "core.json"
    write_core_plan(
        CorpusPlan(scope="core", seed=17, queries=planned_queries),
        r1_plan_path,
        parent_plan_sha256="a" * 64,
    )
    return {
        "asset_root": asset_root,
        "base_assignments": base_assignment_path,
        "catalog_root": catalog_root,
        "r1_plan": r1_plan_path,
        "selection_manifest": materialized.selection_manifest_path,
    }


def test_v8_capability_binding_plan_is_create_only_and_fail_closed(
    tmp_path: Path,
) -> None:
    inputs = _prepared_v8_binding_inputs(tmp_path)
    base_assignment_path = inputs["base_assignments"]
    base_assignment_bytes = base_assignment_path.read_bytes()
    base_assignment_sha256 = sha256_bytes(base_assignment_bytes)
    binding_plan_path = tmp_path / "v8-binding-plan.json"

    with pytest.raises(ValueError, match="external expected digest"):
        build_portfolio_core_v8_capability_binding_plan(
            selection_manifest_path=inputs["selection_manifest"],
            asset_catalog_dir=inputs["catalog_root"],
            asset_root=inputs["asset_root"],
            parent_assignments_path=base_assignment_path,
            parent_assignment_sha256="0" * 64,
            parent_r1_core_plan_path=inputs["r1_plan"],
            output_path=tmp_path / "wrong-parent-hash.json",
        )

    plan_result = build_portfolio_core_v8_capability_binding_plan(
        selection_manifest_path=inputs["selection_manifest"],
        asset_catalog_dir=inputs["catalog_root"],
        asset_root=inputs["asset_root"],
        parent_assignments_path=base_assignment_path,
        parent_assignment_sha256=base_assignment_sha256,
        parent_r1_core_plan_path=inputs["r1_plan"],
        output_path=binding_plan_path,
    )
    assert len(plan_result.abo_candidate_ids) == 14
    assert len(plan_result.isia_food_candidate_ids) == 10
    binding_plan, binding_plan_sha256 = (
        load_portfolio_core_capability_binding_plan_manifest(binding_plan_path)
    )
    assert binding_plan.parent_assignment_sha256 == base_assignment_sha256
    assert binding_plan.baseline_global_abo_triplet_component_count == 14
    assert binding_plan.baseline_dev_abo_triplet_component_count == 8
    assert binding_plan.baseline_non_dev_abo_triplet_component_count == 6
    assert binding_plan.target_non_dev_abo_triplet_component_count == 20
    assert binding_plan.baseline_global_isia_food_pair_component_count == 0
    assert binding_plan.target_non_dev_isia_food_pair_component_count == 10
    assert Counter(entry.binding_profile for entry in binding_plan.selections) == {
        "abo_product_identity_style_knowledge_triplet": 14,
        "isia_food_recipe_knowledge_pair": 10,
    }

    repeated_plan = build_portfolio_core_v8_capability_binding_plan(
        selection_manifest_path=inputs["selection_manifest"],
        asset_catalog_dir=inputs["catalog_root"],
        asset_root=inputs["asset_root"],
        parent_assignments_path=base_assignment_path,
        parent_assignment_sha256=base_assignment_sha256,
        parent_r1_core_plan_path=inputs["r1_plan"],
        output_path=binding_plan_path,
    )
    assert repeated_plan.output_sha256 == plan_result.output_sha256

    assignment_path = tmp_path / "assignments-v8.jsonl"
    v8_assignments = build_portfolio_core_v8_capability_assignments(
        selection_manifest_path=inputs["selection_manifest"],
        asset_catalog_dir=inputs["catalog_root"],
        asset_root=inputs["asset_root"],
        parent_assignments_path=base_assignment_path,
        parent_r1_core_plan_path=inputs["r1_plan"],
        binding_plan_manifest_path=binding_plan_path,
        output_path=assignment_path,
    )
    assert base_assignment_path.read_bytes() == base_assignment_bytes
    assert v8_assignments.assignment_count == 104
    assert v8_assignments.binding_plan_manifest_sha256 == binding_plan_sha256
    loaded_assignments = load_capability_assignments(
        assignment_path,
        expected_sha256=v8_assignments.output_sha256,
    )
    added_assignments = [
        assignment
        for assignment in loaded_assignments
        if assignment.source_artifact_sha256 == binding_plan_sha256
    ]
    assert len(added_assignments) == 38
    assert Counter(item.canonical_capability for item in added_assignments) == {
        "knowledge.visual_encyclopedia": 24,
        "product.style_recommendation": 14,
    }

    omitted_payload = binding_plan.model_dump(mode="json")
    omitted_payload["selections"] = []
    omitted_path = tmp_path / "omitted-v8-binding-plan.json"
    omitted_path.write_bytes(canonical_json_bytes(omitted_payload))
    with pytest.raises(PortfolioCoreAssetError, match="violates schema"):
        load_portfolio_core_capability_binding_plan_manifest(omitted_path)

    tampered_entries = list(binding_plan.selections)
    tampered_entries[0] = tampered_entries[0].model_copy(
        update={"expected_sha256": "0" * 64}
    )
    tampered_plan = binding_plan.model_copy(
        update={"selections": tuple(tampered_entries)}
    )
    tampered_path = tmp_path / "tampered-v8-binding-plan.json"
    tampered_path.write_bytes(canonical_json_bytes(tampered_plan))
    loaded_tampered, _ = load_portfolio_core_capability_binding_plan_manifest(
        tampered_path
    )
    assert loaded_tampered != binding_plan
    with pytest.raises(FileExistsError, match="conflicts"):
        build_portfolio_core_v8_capability_assignments(
            selection_manifest_path=inputs["selection_manifest"],
            asset_catalog_dir=inputs["catalog_root"],
            asset_root=inputs["asset_root"],
            parent_assignments_path=base_assignment_path,
            parent_r1_core_plan_path=inputs["r1_plan"],
            binding_plan_manifest_path=tampered_path,
            output_path=tmp_path / "tampered-assignments-v8.jsonl",
        )


def _prepared_v9_multi_closure_inputs(tmp_path: Path) -> dict[str, object]:
    """Build the exact 105+4 -> 125+4 RPC shape used by the v9 closure."""

    parent_rpc_root = tmp_path / "parent-rpc"
    staging_rpc_root = tmp_path / "rpc-staging"
    cord_root = tmp_path / "cord"
    parent_candidates: list[PortfolioCoreAssetCandidate] = []
    staging_candidates: list[PortfolioCoreAssetCandidate] = []
    for index in range(1, 130):
        source_local_path = f"query_images/multi_product/rpc-{index:04d}.png"
        destination_path = source_local_path
        content = _write_binding_image(
            staging_rpc_root / source_local_path,
            10_000 + index,
        )
        if index <= 109:
            parent_content_path = parent_rpc_root / source_local_path
            parent_content_path.parent.mkdir(parents=True, exist_ok=True)
            parent_content_path.write_bytes(content)
            parent_candidates.append(
                _candidate(
                    candidate_id=f"fixture.rpc.{index:04d}",
                    source_id="rpc",
                    source_local_path=source_local_path,
                    destination_path=destination_path,
                    content=content,
                    pool="multi_product",
                    intent="multi_product",
                    capability="product.multi_search",
                    selection="include" if index <= 105 else "reserve",
                )
            )
        staging_candidates.append(
            _candidate(
                candidate_id=f"fixture.rpc.{index:04d}",
                source_id="rpc",
                source_local_path=source_local_path,
                destination_path=destination_path,
                content=content,
                pool="multi_product",
                intent="multi_product",
                capability="product.multi_search",
                selection="include" if index <= 125 else "reserve",
            )
        )

    cord_source_local_path = "query_images/utility/cord-reserve-0001.png"
    cord_content = _write_binding_image(cord_root / cord_source_local_path, 20_000)
    cord_reserve = _candidate(
        candidate_id="fixture.cord.reserve.0001",
        source_id="cord",
        source_local_path=cord_source_local_path,
        destination_path=cord_source_local_path,
        content=cord_content,
        pool="utility",
        intent="utility",
        capability="utility.document_reading",
        selection="reserve",
    )
    parent_candidates.append(cord_reserve)

    parent_inventory = tmp_path / "parent-inventory.jsonl"
    _write_inventory(parent_inventory, parent_candidates)
    _, parent_inventory_sha256 = load_candidate_inventory(parent_inventory)
    policy = tmp_path / "policy.json"
    policy.write_bytes(
        canonical_json_bytes(
            PortfolioCoreSourcePolicyDocument(
                sources=(
                    PortfolioCoreSourcePolicy(
                        source_id="cord",
                        availability="ready",
                        allowed_pools=("utility",),
                        note="Fixture CORD source.",
                    ),
                    PortfolioCoreSourcePolicy(
                        source_id="rpc",
                        availability="ready",
                        allowed_pools=("multi_product",),
                        note="Fixture RPC source.",
                    ),
                )
            )
        )
    )
    parent_activation = PortfolioCoreReserveActivationManifest(
        parent_inventory_sha256=parent_inventory_sha256,
        parent_selection_manifest_sha256="a" * 64,
        parent_catalog_sha256="b" * 64,
        baseline_component_count=1,
        required_component_floor=1,
        closure_margin=0,
        predicted_component_count=2,
        reason="Fixture reserve activation retained across the v9 RPC closure.",
        selections=(
            PortfolioCoreReserveActivationEntry(
                candidate_id=cord_reserve.candidate_id,
                source_id="cord",
                destination_path=cord_reserve.destination_path,
                expected_sha256=cord_reserve.expected_sha256,
            ),
        ),
    )
    parent_activation_path = tmp_path / "parent-activation.json"
    parent_activation_path.write_bytes(canonical_json_bytes(parent_activation))

    parent_root = tmp_path / "parent-clean"
    parent_materialized = materialize_portfolio_core_assets(
        inventory_path=parent_inventory,
        source_policy_path=policy,
        source_roots={
            "cord": cord_root,
            "rpc": parent_rpc_root,
        },
        output_root=parent_root,
        allow_incomplete_capacity=True,
        reserve_activation_manifest_path=parent_activation_path,
    )
    assert parent_materialized.selection_manifest_path is not None
    assert parent_materialized.dataset_assets_path is not None
    parent_catalog_root = tmp_path / "parent-catalog"
    publish_asset_catalog(
        tuple(
            inventory_dataset_asset(draft, parent_root)
            for draft in load_drafts(parent_materialized.dataset_assets_path)
        ),
        parent_catalog_root,
        parent_root,
        coverage_roots=("query_images",),
    )
    parent_v8_assignments = tmp_path / "parent-v8-assignments.jsonl"
    build_portfolio_core_capability_assignments(
        selection_manifest_path=parent_materialized.selection_manifest_path,
        asset_catalog_dir=parent_catalog_root,
        asset_root=parent_root,
        output_path=parent_v8_assignments,
    )
    base_assignments = load_capability_assignments(
        parent_v8_assignments,
        expected_sha256=sha256_bytes(parent_v8_assignments.read_bytes()),
    )
    # The closure builder must preserve a large immutable parent assignment
    # artifact byte-for-byte.  These rows deliberately use synthetic public
    # paths because this test targets the hash/inheritance boundary; the
    # receipt independently validates every actual v9 catalog selection row.
    template = base_assignments[0]
    inherited_assignments = tuple(
        template.model_copy(
            update={
                "assignment_id": f"fixture.parent-v8-{index:04d}",
                "asset_id": f"fixture.parent-asset-{index:04d}",
                "image_path": f"query_images/multi_product/inherited-{index:04d}.png",
                "source_ref": "fixture:immutable-parent-v8",
                "source_artifact_sha256": "a" * 64,
                "rationale": "Fixture immutable parent assignment retention.",
            }
        )
        for index in range(1, 1_101)
    )
    parent_v8_assignments.write_bytes(canonical_jsonl_bytes(inherited_assignments))
    parent_v8_assignment_sha256 = sha256_bytes(parent_v8_assignments.read_bytes())
    assert (
        len(
            load_capability_assignments(
                parent_v8_assignments,
                expected_sha256=parent_v8_assignment_sha256,
            )
        )
        == 1_100
    )

    staging_candidates_path = staging_rpc_root / "candidates.jsonl"
    _write_inventory(staging_candidates_path, staging_candidates)
    staging_adapter_manifest = staging_rpc_root / "adapter-run.json"
    staging_adapter_manifest.write_bytes(
        canonical_json_bytes(
            {
                "archives": [
                    {
                        "bytes": 1,
                        "logical_path": "rpc/kaggle-v5/archive.zip",
                        "sha256": "c" * 64,
                    }
                ],
                "cross_intent_count": 0,
                "dev_mini_selection_sha256": "d" * 64,
                "formal_eligible": False,
                "formal_status": "non_formal",
                "include_count": 125,
                "policy_version": "portfolio-core-selected-source-adapters-v1",
                "raw_mutation_performed": False,
                "reserve_count": 4,
                "schema_version": 1,
                "source_id": "rpc",
                "source_lock_sha256": "e" * 64,
                "source_policy_sha256": sha256_bytes(policy.read_bytes()),
                "source_revision": "fixture-rpc-v1",
                "track": "portfolio",
            }
        )
    )
    return {
        "parent_activation": parent_activation_path,
        "parent_catalog": parent_catalog_root,
        "parent_inventory": parent_inventory,
        "parent_inventory_sha256": parent_inventory_sha256,
        "parent_root": parent_root,
        "parent_selection": parent_materialized.selection_manifest_path,
        "parent_v8_assignments": parent_v8_assignments,
        "parent_v8_assignment_sha256": parent_v8_assignment_sha256,
        "policy": policy,
        "staging_adapter_manifest": staging_adapter_manifest,
        "staging_candidates": staging_candidates_path,
        "staging_root": staging_rpc_root,
    }


def test_v9_rpc_multi_closure_merges_post_catalog_and_inherits_v8(
    tmp_path: Path,
) -> None:
    inputs = _prepared_v9_multi_closure_inputs(tmp_path)
    parent_v8_assignments = inputs["parent_v8_assignments"]
    assert isinstance(parent_v8_assignments, Path)
    parent_v8_bytes = parent_v8_assignments.read_bytes()
    parent_inventory = inputs["parent_inventory"]
    parent_inventory_sha256 = inputs["parent_inventory_sha256"]
    parent_activation = inputs["parent_activation"]
    policy = inputs["policy"]
    staging_candidates = inputs["staging_candidates"]
    staging_adapter_manifest = inputs["staging_adapter_manifest"]
    parent_root = inputs["parent_root"]
    parent_selection = inputs["parent_selection"]
    parent_catalog = inputs["parent_catalog"]
    for value in (
        parent_inventory,
        parent_activation,
        policy,
        staging_candidates,
        staging_adapter_manifest,
        parent_root,
        parent_selection,
        parent_catalog,
    ):
        assert isinstance(value, Path)
    assert isinstance(parent_inventory_sha256, str)
    parent_v8_assignment_sha256 = inputs["parent_v8_assignment_sha256"]
    assert isinstance(parent_v8_assignment_sha256, str)

    closure = build_portfolio_core_v9_multi_closure_inventory(
        parent_inventory_path=parent_inventory,
        parent_inventory_sha256=parent_inventory_sha256,
        parent_reserve_activation_manifest_path=parent_activation,
        parent_v8_assignments_path=parent_v8_assignments,
        parent_v8_assignment_sha256=parent_v8_assignment_sha256,
        v9_source_policy_path=policy,
        rpc_staging_candidates_path=staging_candidates,
        rpc_staging_adapter_manifest_path=staging_adapter_manifest,
        output_inventory_path=tmp_path / "v9-inventory.jsonl",
        output_reserve_activation_manifest_path=tmp_path / "v9-activation.json",
        output_manifest_path=tmp_path / "v9-closure.json",
    )
    assert len(closure.selection_delta_candidate_ids) == 20
    manifest, manifest_sha256 = load_portfolio_core_v9_multi_closure_inventory_manifest(
        closure.manifest_path
    )
    assert manifest_sha256 == closure.manifest_sha256
    assert manifest.parent_v8_assignment_count == 1_100
    assert len(manifest.promoted_reserve_candidate_ids) == 4
    assert len(manifest.fresh_candidate_ids) == 16
    assert len(manifest.remaining_reserve_candidate_ids) == 4

    v9_root = tmp_path / "v9-clean"
    v9_materialized = materialize_portfolio_core_assets(
        inventory_path=closure.inventory_path,
        source_policy_path=policy,
        source_roots={
            "cord": parent_root,
            "rpc": inputs["staging_root"],
        },
        output_root=v9_root,
        allow_incomplete_capacity=True,
        reserve_activation_manifest_path=closure.reserve_activation_manifest_path,
    )
    assert v9_materialized.selection_manifest_path is not None
    assert v9_materialized.dataset_assets_path is not None
    v9_catalog = tmp_path / "v9-catalog"
    publish_asset_catalog(
        tuple(
            inventory_dataset_asset(draft, v9_root)
            for draft in load_drafts(v9_materialized.dataset_assets_path)
        ),
        v9_catalog,
        v9_root,
        coverage_roots=("query_images",),
    )

    parent_catalog_model = load_asset_catalog(
        parent_catalog,
        parent_root,
        verify_files=True,
    )
    rpc_path = "query_images/multi_product/rpc-0001.png"
    rpc_resolution = parent_catalog_model.resolve_path(rpc_path)
    planned_queries = [
        PlannedQuery(
            plan_id=f"fixture-r1-{index + 1:04d}",
            batch_id=f"fixture-batch-{index // 25 + 1:03d}",
            position=index % 25 + 1,
            taxonomy_version="fixture-taxonomy-v1",
            task_spec_version="fixture-task-spec-v1",
            asset_id=rpc_resolution.asset.asset_id,
            image_path=rpc_path,
            leakage_group_id=rpc_resolution.leakage_group_id,
            template_family=f"fixture-template-{index // 25 + 1:03d}",
            generator_batch_id=f"fixture-batch-{index // 25 + 1:03d}",
            canonical_intent="multi_product",
            canonical_capability="product.multi_search",
            acceptable_capabilities=["product.multi_search"],
            requires_card=False,
            is_boundary=False,
            provisional_split="dev_mini" if index < 200 else "opt_pool",
        )
        for index in range(1_500)
    ]
    r1_plan = tmp_path / "r1-core.json"
    write_core_plan(
        CorpusPlan(scope="core", seed=23, queries=planned_queries),
        r1_plan,
        parent_plan_sha256="f" * 64,
    )

    receipt = build_portfolio_core_v9_multi_closure_catalog_receipt(
        closure_inventory_manifest_path=closure.manifest_path,
        parent_selection_manifest_path=parent_selection,
        parent_asset_catalog_dir=parent_catalog,
        parent_asset_root=parent_root,
        parent_v8_assignments_path=parent_v8_assignments,
        parent_r1_core_plan_path=r1_plan,
        v9_selection_manifest_path=v9_materialized.selection_manifest_path,
        v9_asset_catalog_dir=v9_catalog,
        v9_asset_root=v9_root,
        output_path=tmp_path / "v9-receipt.json",
    )
    assert receipt.component_count_delta >= 20
    assert len(receipt.new_nondev_multi_component_ids) == 20
    assignment_result = build_portfolio_core_v9_multi_capability_assignments(
        closure_catalog_receipt_path=receipt.output_path,
        selection_manifest_path=v9_materialized.selection_manifest_path,
        asset_catalog_dir=v9_catalog,
        asset_root=v9_root,
        parent_v8_assignments_path=parent_v8_assignments,
        output_path=tmp_path / "v9-assignments.jsonl",
    )
    assignments = load_capability_assignments(
        assignment_result.output_path,
        expected_sha256=assignment_result.output_sha256,
    )
    assert assignment_result.assignment_count == len(assignments) == 1_120
    assert parent_v8_assignments.read_bytes() == parent_v8_bytes
    parent_assignments = load_capability_assignments(
        parent_v8_assignments,
        expected_sha256=parent_v8_assignment_sha256,
    )
    assignment_by_id = {
        assignment.assignment_id: assignment for assignment in assignments
    }
    assert all(
        assignment_by_id[assignment.assignment_id] == assignment
        for assignment in parent_assignments
    )
    added = [
        assignment
        for assignment in assignments
        if assignment.source_artifact_sha256 == receipt.output_sha256
    ]
    assert len(added) == 20
    assert {assignment.canonical_capability for assignment in added} == {
        "product.multi_search"
    }

    with pytest.raises(
        PortfolioCoreAssetError, match="parent inventory SHA-256 drifted"
    ):
        build_portfolio_core_v9_multi_closure_inventory(
            parent_inventory_path=parent_inventory,
            parent_inventory_sha256="0" * 64,
            parent_reserve_activation_manifest_path=parent_activation,
            parent_v8_assignments_path=parent_v8_assignments,
            parent_v8_assignment_sha256=parent_v8_assignment_sha256,
            v9_source_policy_path=policy,
            rpc_staging_candidates_path=staging_candidates,
            rpc_staging_adapter_manifest_path=staging_adapter_manifest,
            output_inventory_path=tmp_path / "wrong-v9-inventory.jsonl",
            output_reserve_activation_manifest_path=tmp_path
            / "wrong-v9-activation.json",
            output_manifest_path=tmp_path / "wrong-v9-closure.json",
        )

    staged_rows, _ = load_candidate_inventory(staging_candidates)
    tampered_staging_root = tmp_path / "tampered-rpc-staging"
    tampered_staging_root.mkdir()
    tampered_rows = list(staged_rows)
    tampered_rows[0] = tampered_rows[0].model_copy(update={"selection": "reserve"})
    _write_inventory(tampered_staging_root / "candidates.jsonl", tampered_rows)
    (tampered_staging_root / "adapter-run.json").write_bytes(
        staging_adapter_manifest.read_bytes()
    )
    with pytest.raises(PortfolioCoreAssetError, match="selection states drifted"):
        build_portfolio_core_v9_multi_closure_inventory(
            parent_inventory_path=parent_inventory,
            parent_inventory_sha256=parent_inventory_sha256,
            parent_reserve_activation_manifest_path=parent_activation,
            parent_v8_assignments_path=parent_v8_assignments,
            parent_v8_assignment_sha256=parent_v8_assignment_sha256,
            v9_source_policy_path=policy,
            rpc_staging_candidates_path=tampered_staging_root / "candidates.jsonl",
            rpc_staging_adapter_manifest_path=tampered_staging_root
            / "adapter-run.json",
            output_inventory_path=tmp_path / "tampered-v9-inventory.jsonl",
            output_reserve_activation_manifest_path=(
                tmp_path / "tampered-v9-activation.json"
            ),
            output_manifest_path=tmp_path / "tampered-v9-closure.json",
        )

    tampered_manifest = manifest.model_copy(
        update={"merged_inventory_sha256": "0" * 64}
    )
    tampered_manifest_path = tmp_path / "tampered-v9-closure.json"
    tampered_manifest_path.write_bytes(canonical_json_bytes(tampered_manifest))
    with pytest.raises(PortfolioCoreAssetError, match="does not bind the v9 closure"):
        build_portfolio_core_v9_multi_closure_catalog_receipt(
            closure_inventory_manifest_path=tampered_manifest_path,
            parent_selection_manifest_path=parent_selection,
            parent_asset_catalog_dir=parent_catalog,
            parent_asset_root=parent_root,
            parent_v8_assignments_path=parent_v8_assignments,
            parent_r1_core_plan_path=r1_plan,
            v9_selection_manifest_path=v9_materialized.selection_manifest_path,
            v9_asset_catalog_dir=v9_catalog,
            v9_asset_root=v9_root,
            output_path=tmp_path / "tampered-v9-receipt.json",
        )


def test_inventory_rejects_noncanonical_or_duplicate_destinations(
    tmp_path: Path,
) -> None:
    content = _write_image(tmp_path / "source" / "one.png", 22)
    first = _candidate(
        candidate_id="fixture.one.001",
        source_id="recipe1m_plus",
        source_local_path="one.png",
        destination_path="query_images/utility/one.png",
        content=content,
        pool="utility",
        intent="utility",
        capability="utility.recipe_guidance",
    )
    duplicate = first.model_copy(update={"candidate_id": "fixture.two.001"})
    inventory = tmp_path / "duplicate.jsonl"
    _write_inventory(inventory, [first, duplicate])
    with pytest.raises(PortfolioCoreAssetError, match="duplicate destination_path"):
        load_candidate_inventory(inventory)


def test_source_policy_cannot_broaden_registered_roles(tmp_path: Path) -> None:
    policy = PortfolioCoreSourcePolicyDocument(
        sources=(
            PortfolioCoreSourcePolicy(
                source_id="products_10k",
                availability="ready",
                allowed_pools=("exact_match", "multi_product"),
                note="Invalid broadening fixture.",
            ),
        )
    )
    path = tmp_path / "too-broad-policy.json"
    path.write_bytes(canonical_json_bytes(policy))
    with pytest.raises(PortfolioCoreAssetError, match="broadens"):
        load_source_policy(path)


def test_candidate_source_id_must_match_draft_source_dataset(tmp_path: Path) -> None:
    content = _write_image(tmp_path / "source" / "one.png", 23)
    candidate = _candidate(
        candidate_id="fixture.recipe.002",
        source_id="recipe1m_plus",
        source_local_path="one.png",
        destination_path="query_images/utility/one.png",
        content=content,
        pool="utility",
        intent="utility",
        capability="utility.recipe_guidance",
    )
    payload = candidate.model_dump(mode="json")
    payload["source_id"] = "abo"
    with pytest.raises(ValueError, match="draft.source_dataset must equal source_id"):
        PortfolioCoreAssetCandidate.model_validate(payload, strict=True)


def test_inventory_rejects_isia_document_reading_binding(tmp_path: Path) -> None:
    content = _write_image(tmp_path / "source" / "food.png", 24)
    candidate = _candidate(
        candidate_id="fixture.isia.001",
        source_id="isia_food500",
        source_local_path="food.png",
        destination_path="query_images/utility/food.png",
        content=content,
        pool="utility",
        intent="utility",
        capability="utility.document_reading",
    )
    inventory = tmp_path / "invalid-isia-binding.jsonl"
    _write_inventory(inventory, [candidate])
    with pytest.raises(PortfolioCoreAssetError, match="source/capability role"):
        load_candidate_inventory(inventory)


def test_materialization_rejects_source_output_containment(tmp_path: Path) -> None:
    arguments = _prepared_inputs(tmp_path)
    source_roots = arguments["source_roots"]
    recipe_root = source_roots["recipe1m_plus"]
    outputs = (
        recipe_root / "nested-output",
        recipe_root.parent,
    )
    for output_root in outputs:
        with pytest.raises(
            PortfolioCoreAssetError,
            match="output root must be disjoint from every source root",
        ):
            materialize_portfolio_core_assets(
                inventory_path=arguments["inventory"],
                source_policy_path=arguments["policy"],
                source_roots=source_roots,
                output_root=output_root,
                allow_incomplete_capacity=True,
            )
    assert not (recipe_root / "nested-output").exists()
    assert not (recipe_root.parent / "portfolio-core-asset-run.json").exists()
