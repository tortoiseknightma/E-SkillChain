"""Focused tests for the offline-only Hybrid Router."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillchain.evolution.hybrid_router import (
    AUTHORIZATION_STATUS,
    CAPABILITY_ORDER,
    HybridRouteExample,
    HybridRoutePrediction,
    HybridRouterError,
    decide_hybrid_route,
    evaluate_hybrid_predictions,
    fit_hybrid_router,
    load_hybrid_router_bundle,
    local_takeover,
    make_hybrid_route_example,
    project_user_turns,
    publish_hybrid_router_bundle,
    run_grouped_oof,
    select_threshold_policy,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


_TEXT_BY_CAPABILITY = {
    "product.exact_match": "find the exact identical product and sku",
    "product.multi_search": "list every multiple product visible separately",
    "product.style_recommendation": "recommend a matching style outfit alternative",
    "knowledge.visual_encyclopedia": "explain this object as a visual encyclopedia",
    "utility.document_reading": "read the receipt document and extract its fields",
    "utility.recipe_guidance": "give cooking recipe ingredients and steps",
}


def _examples(*, baseline_correct: bool = False) -> tuple[HybridRouteExample, ...]:
    rows: list[HybridRouteExample] = []
    for fold_index in range(4):
        for capability_index, capability in enumerate(CAPABILITY_ORDER):
            baseline = (
                capability
                if baseline_correct
                else CAPABILITY_ORDER[(capability_index + 1) % len(CAPABILITY_ORDER)]
            )
            rows.append(
                HybridRouteExample(
                    query_id=f"q-{fold_index}-{capability_index}",
                    user_text=f"{_TEXT_BY_CAPABILITY[capability]} example {fold_index}",
                    canonical_capability=capability,
                    fold_id=f"fold-{fold_index:02d}",
                    baseline_capability=baseline,
                    baseline_observed=True,
                )
            )
    return tuple(rows)


def _prediction(
    *,
    query_id: str,
    expected: str,
    baseline: str | None,
    local: str,
    top1: float = 0.8,
) -> HybridRoutePrediction:
    probabilities = [0.04] * len(CAPABILITY_ORDER)
    probabilities[CAPABILITY_ORDER.index(local)] = top1
    # The default values sum to one. This helper is intentionally only used
    # with the default confidence; low-margin behavior has an explicit row.
    ranked = sorted(probabilities, reverse=True)
    return HybridRoutePrediction(
        query_id=query_id,
        fold_id="fold-00",
        expected_capability=expected,
        baseline_capability=baseline,
        baseline_observed=True,
        local_capability=local,
        probabilities=tuple(probabilities),
        top1_probability=ranked[0],
        top2_margin=ranked[0] - ranked[1],
    )


def test_user_projection_ignores_assistant_and_non_turn_query_fields() -> None:
    turns = [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "SECRET_LABEL product.multi_search"},
        {"role": "user", "content": "follow up"},
    ]
    assert project_user_turns(turns) == "first request\nfollow up"

    left = make_hybrid_route_example(
        {
            "query_id": "q-1",
            "turns": turns,
            "canonical_capability": "product.exact_match",
            "image_path": "one.jpg",
            "split": "opt_pool",
            "leakage_group_id": "component-one",
        },
        fold_id="fold-00",
    )
    right = make_hybrid_route_example(
        {
            "query_id": "q-1",
            "turns": [
                turns[0],
                {"role": "assistant", "content": "different secret"},
                turns[2],
            ],
            "canonical_capability": "product.exact_match",
            "image_path": "different.jpg",
            "split": "test_frozen",
            "leakage_group_id": "different-component",
        },
        fold_id="fold-00",
    )
    assert left == right
    assert left.user_text == "first request\nfollow up"


def test_four_fold_oof_is_order_invariant_and_uses_frozen_probability_order() -> None:
    examples = _examples()
    forward = run_grouped_oof(examples)
    reverse = run_grouped_oof(tuple(reversed(examples)))

    assert forward == reverse
    assert forward.fold_ids == ("fold-00", "fold-01", "fold-02", "fold-03")
    assert len(forward.predictions) == len(examples)
    assert len({item.query_id for item in forward.predictions}) == len(examples)
    assert forward.authorization_status == AUTHORIZATION_STATUS
    assert forward.metrics.coverage == 1.0
    for row in forward.predictions:
        assert len(row.probabilities) == len(CAPABILITY_ORDER)
        assert sum(row.probabilities) == pytest.approx(1.0)
        assert row.local_capability == CAPABILITY_ORDER[
            max(
                range(len(CAPABILITY_ORDER)),
                key=lambda index: (row.probabilities[index], -index),
            )
        ]


def test_unresolved_fallback_is_wrong_and_metrics_include_transitions() -> None:
    predictions = tuple(
        _prediction(
            query_id=f"q-{index}",
            expected=capability,
            baseline=None,
            local=capability,
        )
        for index, capability in enumerate(CAPABILITY_ORDER)
    )
    fallback = evaluate_hybrid_predictions(
        predictions, probability_threshold=1.0, margin_threshold=1.0
    )
    assert fallback.coverage == 0.0
    assert fallback.query_accuracy == 0.0
    assert fallback.micro_f1 == 0.0
    assert fallback.macro_f1 == 0.0

    local = evaluate_hybrid_predictions(predictions)
    assert local.query_accuracy == 1.0
    assert local.micro_f1 == 1.0
    assert local.macro_f1 == 1.0
    assert set(local.recall_by_capability) == set(CAPABILITY_ORDER)
    assert local.corrected == len(CAPABILITY_ORDER)
    assert local.broken == 0
    assert local.utility == len(CAPABILITY_ORDER)


def test_threshold_uses_probability_and_margin_and_selector_can_fail_closed() -> None:
    low_margin_probabilities = (0.40, 0.39, 0.0525, 0.0525, 0.0525, 0.0525)
    low_margin = HybridRoutePrediction(
        query_id="q-low-margin",
        fold_id="fold-00",
        expected_capability=CAPABILITY_ORDER[0],
        baseline_capability=CAPABILITY_ORDER[1],
        baseline_observed=True,
        local_capability=CAPABILITY_ORDER[0],
        probabilities=low_margin_probabilities,
        top1_probability=0.40,
        top2_margin=0.01,
    )
    assert local_takeover(
        low_margin, probability_threshold=0.30, margin_threshold=0.01
    )
    assert not local_takeover(
        low_margin, probability_threshold=0.30, margin_threshold=0.02
    )
    assert decide_hybrid_route(
        low_margin, probability_threshold=0.30, margin_threshold=0.02
    ).outcome == "fallback_required"
    local_decision = decide_hybrid_route(
        low_margin, probability_threshold=0.30, margin_threshold=0.01
    )
    assert local_decision.outcome == "local_route"
    assert local_decision.selected_capability == CAPABILITY_ORDER[0]
    assert not local_takeover(
        low_margin, probability_threshold=0.41, margin_threshold=0.0
    )

    improved = tuple(
        _prediction(
            query_id=f"q-{index}",
            expected=capability,
            baseline=CAPABILITY_ORDER[(index + 1) % len(CAPABILITY_ORDER)],
            local=capability,
        )
        for index, capability in enumerate(CAPABILITY_ORDER)
    )
    selected = select_threshold_policy(
        improved,
        probability_thresholds=(0.0, 0.8),
        margin_thresholds=(0.0, 0.76),
    )
    assert selected.outcome == "candidate"
    assert selected.policy is not None
    assert selected.policy.authorization_status == AUTHORIZATION_STATUS
    assert selected.policy.metrics.coverage >= 0.70
    assert selected.policy.metrics.utility > 0

    unavailable = tuple(
        row.model_copy(update={"baseline_observed": False}) for row in improved
    )
    with pytest.raises(HybridRouterError, match="observed paired fallback"):
        select_threshold_policy(
            unavailable,
            probability_thresholds=(0.0,),
            margin_thresholds=(0.0,),
        )

    perfect_fallback = tuple(
        row.model_copy(update={"baseline_capability": row.expected_capability})
        for row in improved
    )
    rejected = select_threshold_policy(
        perfect_fallback,
        probability_thresholds=(0.0,),
        margin_thresholds=(0.0,),
    )
    assert rejected.outcome == "no_candidate"
    assert rejected.policy is None


def test_bundle_round_trip_is_create_only_and_rejects_hash_tamper(tmp_path: Path) -> None:
    examples = _examples()
    oof = run_grouped_oof(examples)
    pipeline = fit_hybrid_router(examples)
    destination = tmp_path / "router-bundle"
    manifest = publish_hybrid_router_bundle(
        destination,
        pipeline=pipeline,
        oof_result=oof,
        source_queries_sha256="a" * 64,
        fold_plan_sha256="b" * 64,
    )
    assert manifest.authorization_status == AUTHORIZATION_STATUS
    assert {item.name for item in destination.iterdir()} == {
        "router.joblib",
        "model-manifest.json",
        "oof-rows.jsonl",
        "cv-report.json",
    }
    manifest_file_sha256 = sha256_bytes(
        (destination / "model-manifest.json").read_bytes()
    )

    loaded = load_hybrid_router_bundle(
        destination,
        expected_manifest_file_sha256=manifest_file_sha256,
        expected_source_queries_sha256="a" * 64,
        expected_fold_plan_sha256="b" * 64,
    )
    assert loaded.predictions == oof.predictions
    assert loaded.metrics == oof.metrics
    assert loaded.manifest == manifest
    assert loaded.pipeline.predict_proba([examples[0].user_text]) == pytest.approx(
        pipeline.predict_proba([examples[0].user_text])
    )
    with pytest.raises(FileExistsError):
        publish_hybrid_router_bundle(
            destination,
            pipeline=pipeline,
            oof_result=oof,
            source_queries_sha256="a" * 64,
            fold_plan_sha256="b" * 64,
        )

    with (destination / "router.joblib").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(HybridRouterError, match="content hash mismatch"):
        load_hybrid_router_bundle(
            destination,
            expected_manifest_file_sha256=manifest_file_sha256,
        )

    manifest_bytes = (destination / "model-manifest.json").read_bytes()
    with pytest.raises(HybridRouterError, match="manifest file hash differs"):
        load_hybrid_router_bundle(
            destination,
            expected_manifest_file_sha256="0" * 64,
        )
    assert (destination / "model-manifest.json").read_bytes() == manifest_bytes


def test_bundle_rejects_dependency_version_and_pipeline_contract_drift(
    tmp_path: Path,
) -> None:
    examples = _examples()
    oof = run_grouped_oof(examples)
    pipeline = fit_hybrid_router(examples)

    drifted_pipeline = fit_hybrid_router(examples)
    drifted_pipeline.named_steps["tfidf"].lowercase = False
    with pytest.raises(HybridRouterError, match="vectorizer parameter drifted"):
        publish_hybrid_router_bundle(
            tmp_path / "bad-pipeline",
            pipeline=drifted_pipeline,
            oof_result=oof,
            source_queries_sha256="a" * 64,
            fold_plan_sha256="b" * 64,
        )

    drifted_classifier = fit_hybrid_router(examples)
    drifted_classifier.named_steps["classifier"].fit_intercept = False
    with pytest.raises(HybridRouterError, match="classifier parameter drifted"):
        publish_hybrid_router_bundle(
            tmp_path / "bad-classifier",
            pipeline=drifted_classifier,
            oof_result=oof,
            source_queries_sha256="a" * 64,
            fold_plan_sha256="b" * 64,
        )

    destination = tmp_path / "version-drift"
    publish_hybrid_router_bundle(
        destination,
        pipeline=pipeline,
        oof_result=oof,
        source_queries_sha256="a" * 64,
        fold_plan_sha256="b" * 64,
    )
    manifest_path = destination / "model-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["sklearn_version"] = "0.0.invalid"
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    payload["manifest_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    manifest_path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(HybridRouterError, match="dependency version drifted"):
        load_hybrid_router_bundle(
            destination,
            expected_manifest_file_sha256=sha256_bytes(manifest_path.read_bytes()),
        )


def test_oof_rejects_duplicate_ids_and_missing_fold_or_capability() -> None:
    examples = _examples()
    with pytest.raises(HybridRouterError, match="query IDs must be unique"):
        run_grouped_oof((*examples, examples[0]))
    with pytest.raises(HybridRouterError, match="exactly the frozen folds"):
        run_grouped_oof(tuple(item for item in examples if item.fold_id != "fold-03"))
    without_recipe = tuple(
        item
        for item in examples
        if item.canonical_capability != "utility.recipe_guidance"
    )
    with pytest.raises(HybridRouterError, match="cover the frozen six"):
        run_grouped_oof(without_recipe)
