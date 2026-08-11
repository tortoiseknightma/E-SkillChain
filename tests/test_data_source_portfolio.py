import json
from pathlib import Path

from skillchain.synthesis.planning import DEV_MINI_CAPABILITY_COUNTS


ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_PATH = ROOT / "specs/data_sources/ecommerce-mvp-source-portfolio-v1.json"
TAXONOMY_PATH = ROOT / "specs/taxonomy/ecommerce-mvp-taxonomy-v0.json"
TASK_SPEC_PATH = ROOT / "specs/task_specs/ecommerce-task-spec-v1.json"
TRAJECTORY_POLICY_PATH = (
    ROOT / "specs/data_sources/dialogue-trajectory-source-policy-v1.json"
)


def _load_portfolio() -> dict:
    return json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))


def test_mvp_source_portfolio_has_the_frozen_200_query_allocation() -> None:
    portfolio = _load_portfolio()
    mvp = portfolio["profiles"]["mvp"]
    allocations = {
        item["capability"]: item["count"] for item in mvp["capability_allocations"]
    }

    assert mvp["query_count"] == 200
    assert sum(allocations.values()) == 200
    assert allocations == {
        "exact_match": 35,
        "multi_product": 35,
        "style_recommendation": 35,
        "visual_encyclopedia": 35,
        "document_reading": 30,
        "recipe_guidance": 30,
    }
    assert {
        item["capability"]: item["count"] for item in mvp["capability_allocations"]
    } == {
        "exact_match": DEV_MINI_CAPABILITY_COUNTS["product.exact_match"],
        "multi_product": DEV_MINI_CAPABILITY_COUNTS["product.multi_search"],
        "style_recommendation": DEV_MINI_CAPABILITY_COUNTS[
            "product.style_recommendation"
        ],
        "visual_encyclopedia": DEV_MINI_CAPABILITY_COUNTS[
            "knowledge.visual_encyclopedia"
        ],
        "document_reading": DEV_MINI_CAPABILITY_COUNTS["utility.document_reading"],
        "recipe_guidance": DEV_MINI_CAPABILITY_COUNTS["utility.recipe_guidance"],
    }
    assert mvp["boundary_target_approx"] == 40
    assert mvp["human_authored_or_rewritten_min"] == 40


def test_primary_sources_match_the_required_supervision_signals() -> None:
    portfolio = _load_portfolio()
    allocations = {
        item["capability"]: item
        for item in portfolio["profiles"]["mvp"]["capability_allocations"]
    }

    assert allocations["exact_match"]["primary_sources"] == ["abo"]
    assert allocations["multi_product"]["primary_sources"] == ["rpc"]
    assert allocations["style_recommendation"]["primary_sources"] == ["fashioniq"]
    assert allocations["document_reading"]["primary_sources"] == ["wildreceipt"]


def test_language_sources_and_challenge_sources_cannot_supply_capability_gold() -> None:
    portfolio = _load_portfolio()
    sources = {item["source_id"]: item for item in portfolio["sources"]}

    assert sources["jddc_2_0"]["disposition"] == "permanently_abandoned"
    assert "mvp_input" in sources["jddc_2_0"]["must_not_supply"]
    assert "capability_gold" in sources["durecdial_2_0"]["must_not_supply"]
    assert "capability_gold" in sources["crosswoz"]["must_not_supply"]
    assert sources["codex_mock_trajectories"]["required_origin"] == "synthetic_derived"
    assert "real_user_log" in sources["codex_mock_trajectories"]["must_not_supply"]
    assert "exact_match_positive" in sources["muge"]["must_not_supply"]
    assert (
        "multi_product_sku_retrieval_primary_metric"
        in sources["coco_open_images"]["must_not_supply"]
    )
    assert (
        "field_level_ocr_kie_gold" in sources["wikimedia_documents"]["must_not_supply"]
    )
    assert "recipe_evidence" in sources["isia_food500"]["must_not_supply"]


def test_mock_trajectory_policy_retires_unavailable_sources_and_defers_full_design() -> None:
    policy = json.loads(TRAJECTORY_POLICY_PATH.read_text(encoding="utf-8"))

    assert policy["decision"]["jddc_2_0"]["disposition"] == "permanently_abandoned"
    assert policy["decision"]["u_need"]["disposition"] == "permanently_abandoned"
    assert policy["mvp"]["trajectory_mode"] == "mock"
    assert policy["mvp"]["composition"]["required_data_origin"] == "synthetic_derived"
    assert {item["source_id"] for item in policy["mvp"]["source_patterns"]} == {
        "durecdial_2_0",
        "crosswoz",
        "muge",
    }
    assert policy["full"]["implementation_plan_status"] == "deferred"
    assert {item["source_id"] for item in policy["full"]["conditional_candidates"]} == {
        "simmc_2_1",
        "csds",
    }
    assert (
        policy["full"]["owner_reported_pending_dataset_label"]["status"]
        == "identity_and_terms_unverified"
    )


def test_mock_trajectory_policy_binds_the_frozen_paper_aligned_intents() -> None:
    policy = json.loads(TRAJECTORY_POLICY_PATH.read_text(encoding="utf-8"))
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    task_spec = json.loads(TASK_SPEC_PATH.read_text(encoding="utf-8"))
    composition = policy["mvp"]["composition"]

    assert tuple(composition["canonical_intents"]) == tuple(
        item["intent_id"] for item in taxonomy["intents"]
    )
    assert composition["taxonomy_version"] == taxonomy["taxonomy_version"]
    assert composition["taxonomy_sha256"] == taxonomy["taxonomy_sha256"]
    assert composition["task_spec_version"] == task_spec["task_spec_version"]
    assert composition["task_spec_sha256"] == task_spec["task_spec_sha256"]
    assert task_spec["taxonomy_version"] == taxonomy["taxonomy_version"]
    assert task_spec["taxonomy_sha256"] == taxonomy["taxonomy_sha256"]


def test_deferred_sources_are_not_promoted_to_mvp_primary() -> None:
    portfolio = _load_portfolio()
    sources = {item["source_id"]: item for item in portfolio["sources"]}
    primary_sources = {
        source
        for allocation in portfolio["profiles"]["mvp"]["capability_allocations"]
        for source in allocation["primary_sources"]
    }

    for source_id in ("mep3m", "products_10k", "polyvore", "recipe1m_plus"):
        assert sources[source_id]["tier"] == "core"
        assert source_id not in primary_sources
    assert sources["sku_110k"]["tier"] == "core_challenge"
    assert "sku_110k" not in primary_sources
    assert sources["u_need"]["disposition"] == "permanently_abandoned"
    assert "full_input" in sources["u_need"]["must_not_supply"]


def test_plans_and_status_docs_link_to_the_source_portfolio() -> None:
    for relative_path in (
        "README.md",
        "docs/plans/2026-07-09-skillchain-reproduction.md",
        "docs/plans/2026-07-20-p0-p1-closure.md",
        "docs/go-no-go/2026-07-20-core-no-go.md",
        "docs/superpowers/plans/2026-07-11-phase2-tool-layer.md",
        "docs/superpowers/plans/2026-07-11-product-search-tools.md",
        "docs/superpowers/specs/2026-07-11-phase2-tool-layer-design.md",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "ecommerce-mvp-source-portfolio-v1.json" in text
