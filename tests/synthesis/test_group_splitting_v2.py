import random
from collections import Counter

import pytest

from skillchain.schemas import LabelDecision, Query
from skillchain.synthesis.splitting import (
    GROUP_FIELDS,
    ExactSizeInfeasibleError,
    LockedGroupConflictError,
    SplitSpec,
    assert_no_group_leakage,
    build_atomic_groups,
    group_leakage_report,
    stratified_split,
)
from skillchain.synthesis.planning import PHASE3_TASK_SPEC_VERSION
from skillchain.taxonomy import (
    TAXONOMY_VERSION,
    capabilities_for_intent,
)


def _query(
    index: int,
    *,
    split: str = "opt_pool",
    intent: str = "exact_match",
    leakage_group_id: str | None = None,
    boundary_group_id: str | None = None,
    template_family: str | None = None,
    generator_batch_id: str | None = None,
    capability: str | None = None,
    acceptable_capabilities: list[str] | None = None,
) -> Query:
    text = f"mechanical query {index:03d}"
    definitions = capabilities_for_intent(intent)
    capability = capability or definitions[0].capability_id
    acceptable_capabilities = acceptable_capabilities or [capability]
    requires_card = next(
        item.requires_card for item in definitions if item.capability_id == capability
    )
    is_boundary = boundary_group_id is not None
    decision = LabelDecision(
        decision_type="cross_review",
        annotator_kind="llm_pair",
        annotator_id="mechanical-reviewers",
        canonical_intent=intent,
        canonical_capability=capability,
        acceptable_capabilities=acceptable_capabilities,
        reason="mechanical test fixture",
    )
    return Query(
        schema_version=2,
        taxonomy_version=TAXONOMY_VERSION,
        task_spec_version=PHASE3_TASK_SPEC_VERSION,
        query_id=f"q-{index:03d}",
        asset_id=f"asset-{index:03d}",
        image_path=f"query_images/mechanical-{index:03d}.jpg",
        leakage_group_id=leakage_group_id or f"leakage-{index:03d}",
        boundary_group_id=boundary_group_id,
        template_family=template_family or f"template-{index:03d}",
        generator_batch_id=generator_batch_id or f"generator-{index:03d}",
        text=text,
        turns=[{"role": "user", "content": text}],
        canonical_intent=intent,
        canonical_capability=capability,
        acceptable_capabilities=acceptable_capabilities,
        is_boundary=is_boundary,
        boundary_strategy="cross_intent_triplet" if is_boundary else None,
        requires_card=requires_card,
        split=split,
        label_status="cross_agreed",
        label_provenance=[decision],
    )


def _spec(*, dev: int, opt: int, val: int, test: int) -> SplitSpec:
    return SplitSpec(
        profile="mechanical-test",
        sizes={
            "dev_mini": dev,
            "opt_pool": opt,
            "val": val,
            "test_frozen": test,
        },
        min_test_per_intent=0,
    )


def _split_by_query_id(queries: list[Query]) -> dict[str, str]:
    return {query.query_id: query.split for query in queries}


def test_atomic_groups_follow_transitive_closure_across_group_fields():
    first = _query(1, leakage_group_id="asset-family")
    bridge = _query(
        2,
        leakage_group_id="asset-family",
        template_family="template-bridge",
    )
    last = _query(3, template_family="template-bridge")
    isolated = _query(4)

    groups = build_atomic_groups([last, isolated, bridge, first])
    member_sets = {
        frozenset(query.query_id for query in group.queries) for group in groups
    }

    assert member_sets == {
        frozenset({first.query_id, bridge.query_id, last.query_id}),
        frozenset({isolated.query_id}),
    }


def test_equal_raw_values_in_different_group_fields_are_namespaced():
    same_token = "same-raw-token"
    queries = [
        _query(1, leakage_group_id=same_token),
        _query(2, template_family=same_token),
        _query(3, generator_batch_id=same_token),
        _query(4, boundary_group_id=same_token),
    ]

    groups = build_atomic_groups(queries)

    assert len(groups) == len(queries)
    assert all(group.size == 1 for group in groups)


def test_group_split_is_deterministic_under_candidate_reordering():
    split_spec = _spec(dev=2, opt=2, val=2, test=2)
    candidates = [
        _query(index, split="dev_mini" if index <= 2 else "opt_pool")
        for index in range(1, 9)
    ]
    locked = {"q-001", "q-002"}

    expected = stratified_split(
        candidates,
        locked_dev_query_ids=locked,
        seed=20260711,
        split_spec=split_spec,
    )
    reordered = list(candidates)
    random.Random(9173).shuffle(reordered)
    actual = stratified_split(
        reordered,
        locked_dev_query_ids=locked,
        seed=20260711,
        split_spec=split_spec,
    )

    assert _split_by_query_id(actual) == _split_by_query_id(expected)


def test_split_accepts_multiple_capabilities_under_one_intent():
    utility_capabilities = [
        item.capability_id for item in capabilities_for_intent("utility")
    ]
    candidates = [
        _query(
            1,
            split="dev_mini",
            intent="utility",
            capability="utility.document_reading",
            acceptable_capabilities=utility_capabilities,
        ),
        _query(
            2,
            intent="utility",
            capability="utility.recipe_guidance",
        ),
        _query(3),
        _query(4),
    ]

    assigned = stratified_split(
        candidates,
        locked_dev_query_ids={"q-001"},
        seed=20260711,
        split_spec=_spec(dev=1, opt=1, val=1, test=1),
    )

    assert {
        query.canonical_capability
        for query in assigned
        if query.canonical_intent == "utility"
    } == {"utility.document_reading", "utility.recipe_guidance"}
    assert (
        next(
            query for query in assigned if query.query_id == "q-001"
        ).acceptable_capabilities
        == utility_capabilities
    )


@pytest.mark.parametrize(
    "patch",
    [
        {
            "canonical_capability": "utility.recipe_guidance",
            "acceptable_capabilities": ["utility.recipe_guidance"],
        },
        {"requires_card": False},
        {
            "acceptable_capabilities": [
                "product.exact_match",
                "utility.document_reading",
            ]
        },
    ],
)
def test_split_rejects_wrong_capability_membership_or_card(patch):
    candidates = [
        _query(1, split="dev_mini").model_copy(update=patch),
        _query(2),
        _query(3),
        _query(4),
    ]

    with pytest.raises(ValueError, match="capability binding|requires_card"):
        stratified_split(
            candidates,
            locked_dev_query_ids={"q-001"},
            seed=20260711,
            split_spec=_spec(dev=1, opt=1, val=1, test=1),
        )


def test_locked_dev_cannot_share_an_atomic_group_with_unlocked_query():
    split_spec = _spec(dev=1, opt=1, val=1, test=1)
    candidates = [
        _query(1, split="dev_mini", leakage_group_id="mixed-group"),
        _query(2, leakage_group_id="mixed-group"),
        _query(3),
        _query(4),
    ]

    with pytest.raises(LockedGroupConflictError, match="locked dev|原子 group"):
        stratified_split(
            candidates,
            locked_dev_query_ids={"q-001"},
            seed=20260711,
            split_spec=split_spec,
        )


def test_exact_capacity_infeasibility_fails_without_splitting_groups():
    split_spec = _spec(dev=1, opt=1, val=1, test=2)
    candidates = [
        _query(1, split="dev_mini"),
        _query(2, leakage_group_id="pair-a"),
        _query(3, leakage_group_id="pair-a"),
        _query(4, leakage_group_id="pair-b"),
        _query(5, leakage_group_id="pair-b"),
    ]

    with pytest.raises(ExactSizeInfeasibleError, match="精确 split sizes"):
        stratified_split(
            candidates,
            locked_dev_query_ids={"q-001"},
            seed=20260711,
            split_spec=split_spec,
        )


def test_successful_assignment_has_exact_sizes_and_zero_group_leakage():
    split_spec = _spec(dev=2, opt=3, val=2, test=3)
    candidates = [
        _query(1, split="dev_mini", leakage_group_id="locked-pair"),
        _query(2, split="dev_mini", leakage_group_id="locked-pair"),
        _query(3, leakage_group_id="asset-pair"),
        _query(4, leakage_group_id="asset-pair"),
        _query(5, template_family="template-pair"),
        _query(6, template_family="template-pair"),
        _query(7, boundary_group_id="boundary-pair"),
        _query(8, boundary_group_id="boundary-pair"),
        _query(9),
        _query(10),
    ]

    assigned = stratified_split(
        candidates,
        locked_dev_query_ids={"q-001", "q-002"},
        seed=20260711,
        split_spec=split_spec,
    )

    assert Counter(query.split for query in assigned) == Counter(split_spec.sizes)
    assert all(
        len({query.split for query in group.queries}) == 1
        for group in build_atomic_groups(assigned)
    )
    report = group_leakage_report(assigned)
    assert report["group_fields"] == list(GROUP_FIELDS)
    assert report["violation_count"] == 0
    assert report["violations"] == {}
    assert_no_group_leakage(assigned)
