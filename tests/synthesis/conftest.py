import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from skillchain.schemas import LabelDecision, Query
from skillchain.synthesis.batches import compute_generation_input_sha256
from skillchain.synthesis.models import CorpusPlan, PlannedQuery
from skillchain.synthesis.planning import (
    PHASE3_TASK_SPEC_VERSION,
    activate_dev_plan,
    build_dev_mini_plan,
    extend_full_plan,
    write_dev_mini_plan,
    write_full_plan,
)
from skillchain.synthesis.seeds import accept_seed_batch, stage_seed_batch
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


def make_query_v2(
    *,
    query_id: str,
    image_path: str,
    text: str,
    intent: str,
    split: str,
    generator_batch_id: str,
    label_status: str = "auto",
    is_boundary: bool = False,
    boundary_strategy: str | None = None,
    boundary_group_id: str | None = None,
    asset_id: str | None = None,
    leakage_group_id: str | None = None,
    template_family: str | None = None,
    capability: str | None = None,
    requires_card: bool | None = None,
) -> Query:
    """Build a complete schema-v2 query for synthesis integration tests."""

    capability = capability or capability_for_intent(intent)
    requires_card = (
        requires_card_for_intent(intent) if requires_card is None else requires_card
    )
    decision_type = "cross_review" if label_status == "cross_agreed" else "constructed"
    annotator_kind = "llm_pair" if label_status == "cross_agreed" else "planner"
    return Query(
        schema_version=2,
        taxonomy_version=TAXONOMY_VERSION,
        task_spec_version=PHASE3_TASK_SPEC_VERSION,
        query_id=query_id,
        asset_id=asset_id or f"asset.{query_id}",
        image_path=image_path,
        leakage_group_id=leakage_group_id or f"leakage.{query_id}",
        boundary_group_id=boundary_group_id,
        template_family=template_family or f"{generator_batch_id}/{intent}/v1",
        generator_batch_id=generator_batch_id,
        text=text,
        turns=[{"role": "user", "content": text}],
        canonical_intent=intent,
        canonical_capability=capability,
        acceptable_capabilities=[capability],
        is_boundary=is_boundary,
        boundary_strategy=boundary_strategy,
        requires_card=requires_card,
        split=split,
        label_status=label_status,
        label_provenance=[
            LabelDecision(
                decision_type=decision_type,
                annotator_kind=annotator_kind,
                annotator_id="mechanical-test-fixture",
                canonical_intent=intent,
                canonical_capability=capability,
                acceptable_capabilities=[capability],
            )
        ],
    )


@pytest.fixture
def fake_image_root(tmp_path: Path) -> Path:
    for intent in (
        "exact_match",
        "multi_product",
        "divergent_rec",
        "encyclopedia",
        "utility",
    ):
        directory = tmp_path / intent
        directory.mkdir()
        for index in range(1, 61):
            (directory / f"image-{index:03d}.jpg").write_bytes(b"mechanical-image")
    return tmp_path


@pytest.fixture
def mechanical_seed_draft() -> dict:
    intents = (
        "exact_match",
        "multi_product",
        "divergent_rec",
        "encyclopedia",
        "utility",
    )
    return {
        "provider": "codex",
        "model_display_name": "5.6 Sol Ultra",
        "model_claim_source": "user_confirmation",
        "generated_at": "2026-07-11T00:00:00Z",
        "examples": {
            intent: [f"机械占位-{intent}-{index}" for index in range(1, 4)]
            for intent in intents
        },
    }


@dataclass
class CorpusFixture:
    root: Path
    plan_path: Path
    draft_path: Path
    draft_manifest_path: Path
    batch_items: list[PlannedQuery]

    @property
    def stage_kwargs(self) -> dict:
        return {
            "draft_path": self.draft_path,
            "draft_manifest_path": self.draft_manifest_path,
            "plan_path": self.plan_path,
            "queries_root": self.root,
            "base_batch_id": "dev-mini-001",
            "allow_provisional_asset_groups": True,
        }

    def draft_rows(self) -> list[dict]:
        return [
            {
                "plan_id": item.plan_id,
                "turns": [{"role": "user", "content": f"机械占位-{item.plan_id}"}],
            }
            for item in self.batch_items
        ]

    def write_draft(
        self,
        rows: list[dict] | None = None,
        manifest_overrides: dict | None = None,
    ) -> None:
        draft_bytes = canonical_jsonl_bytes(rows or self.draft_rows())
        self.draft_path.write_bytes(draft_bytes)
        plan = CorpusPlan.model_validate_json(self.plan_path.read_bytes())
        plan_items = [item for item in plan.queries if item.batch_id == "dev-mini-001"]
        plan_manifest_path = self.plan_path.with_name(
            f"{self.plan_path.stem}.manifest.json"
        )
        plan_manifest = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
        seed_manifest = json.loads(
            (self.root / "seeds/accepted/manifest.json").read_text(encoding="utf-8")
        )
        draft_sha256 = sha256_bytes(draft_bytes)
        manifest = {
            "schema_version": 2,
            "base_batch_id": "dev-mini-001",
            "data_origin": "synthetic_derived",
            "provider": "codex",
            "model_display_name": "5.6 Sol Ultra",
            "model_claim_source": "user_confirmation",
            "generated_at": "2026-07-11T00:00:00Z",
            "plan_sha256": plan_manifest["plan_sha256"],
            "asset_catalog_sha256": plan_manifest["asset_catalog_sha256"],
            "leakage_policy_version": plan_manifest["leakage_policy_version"],
            "seed_set_sha256": seed_manifest["seed_set_sha256"],
            "draft_sha256": draft_sha256,
            "generation_input_sha256": compute_generation_input_sha256(
                base_batch_id="dev-mini-001",
                plan_items=plan_items,
                plan_sha256=plan_manifest["plan_sha256"],
                asset_catalog_sha256=plan_manifest["asset_catalog_sha256"],
                leakage_policy_version=plan_manifest["leakage_policy_version"],
                seed_set_sha256=seed_manifest["seed_set_sha256"],
            ),
        }
        manifest.update(manifest_overrides or {})
        self.draft_manifest_path.write_bytes(canonical_json_bytes(manifest))

    def stage_valid_batch(self) -> Path:
        from skillchain.synthesis.batches import stage_generated_batch

        self.write_draft()
        return stage_generated_batch(**self.stage_kwargs)


@pytest.fixture
def corpus_fixture(
    tmp_path: Path, mechanical_seed_draft: dict, fake_image_root: Path
) -> CorpusFixture:
    root = tmp_path / "queries"
    plan = build_dev_mini_plan(
        fake_image_root,
        seed=20260711,
        allow_provisional_asset_groups=True,
    )
    batch_items = [item for item in plan.queries if item.batch_id == "dev-mini-001"]
    plan_path = tmp_path / "plans" / "dev_mini.json"
    write_dev_mini_plan(plan, plan_path)

    seed_input = tmp_path / "seed-input.json"
    seed_input.write_text(
        json.dumps(mechanical_seed_draft, ensure_ascii=False), encoding="utf-8"
    )
    stage_seed_batch(seed_input, root, "seed-r1")
    accept_seed_batch(root, "seed-r1", confirmation="ACCEPT")

    return CorpusFixture(
        root=root,
        plan_path=plan_path,
        draft_path=tmp_path / "draft.jsonl",
        draft_manifest_path=tmp_path / "draft.manifest.json",
        batch_items=batch_items,
    )


@dataclass
class FullActivationFixture:
    queries_root: Path
    full_plan_path: Path
    active_path: Path

    def inject(self, fault: str) -> None:
        ledger_path = self.queries_root / "accepted-ledger.jsonl"
        rows = [
            json.loads(line) for line in ledger_path.read_text("utf-8").splitlines()
        ]
        if fault == "too_few_accepted":
            rows.pop()
        elif fault == "parent_hash_mismatch":
            rows[0]["plan_sha256"] = "f" * 64
        elif fault == "pending_staging":
            (self.queries_root / "staging/dev-mini-009-r1").mkdir(parents=True)
            return
        else:  # pragma: no cover - fixture misuse guard
            raise AssertionError(f"unknown fault: {fault}")
        ledger_path.write_bytes(canonical_jsonl_bytes(rows))


@pytest.fixture
def full_activation_fixture(
    fake_image_root: Path, tmp_path: Path
) -> FullActivationFixture:
    root = tmp_path / "queries"
    dev_plan = build_dev_mini_plan(
        fake_image_root,
        seed=20260711,
        allow_provisional_asset_groups=True,
    )
    dev_path = root / "plans/dev_mini.json"
    write_dev_mini_plan(dev_plan, dev_path)
    active = activate_dev_plan(
        dev_path,
        root,
        allow_provisional_asset_groups=True,
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
    full_path = root / "plans/full.json"
    write_full_plan(
        full_plan,
        full_path,
        parent_plan_sha256=active.plan_sha256,
    )

    entries = []
    for index in range(1, 9):
        base_batch_id = f"dev-mini-{index:03d}"
        batch_id = f"{base_batch_id}-r1"
        (root / "accepted" / batch_id).mkdir(parents=True)
        entries.append(
            {
                "batch_id": batch_id,
                "base_batch_id": base_batch_id,
                "revision": 1,
                "count": 25,
                "results_sha256": f"{index:064x}",
                "plan_sha256": active.plan_sha256,
                "seed_set_sha256": "a" * 64,
                "accepted_at": "2026-07-11T00:00:00Z",
            }
        )
    (root / "accepted-ledger.jsonl").write_bytes(canonical_jsonl_bytes(entries))
    return FullActivationFixture(
        queries_root=root,
        full_plan_path=full_path,
        active_path=root / "plans/active.json",
    )


@pytest.fixture
def full_candidate_fixture():
    totals = {
        "exact_match": (1125, 50, 197, 9),
        "multi_product": (675, 30, 118, 5),
        "divergent_rec": (900, 40, 158, 7),
        "encyclopedia": (900, 40, 158, 7),
        "utility": (900, 40, 157, 7),
    }
    queries = []
    locked_dev_ids = set()
    sequence = 0
    locked_sequence = 0
    non_dev_sequence = 0
    for intent, (
        total,
        dev_total,
        boundary_total,
        dev_boundary_total,
    ) in totals.items():
        for offset in range(total):
            sequence += 1
            is_locked_dev = offset < dev_total
            if is_locked_dev:
                is_boundary = offset < dev_boundary_total
            else:
                non_dev_offset = offset - dev_total
                is_boundary = non_dev_offset < boundary_total - dev_boundary_total
            query_id = f"full-{sequence:04d}"
            if is_locked_dev:
                locked_sequence += 1
                generator_batch_id = (
                    f"fixture-dev-{((locked_sequence - 1) // 25) + 1:03d}"
                )
            else:
                non_dev_sequence += 1
                generator_batch_id = (
                    f"fixture-full-{((non_dev_sequence - 1) // 25) + 1:03d}"
                )
            queries.append(
                make_query_v2(
                    query_id=query_id,
                    image_path=f"query_images/{intent}/image-{sequence:04d}.jpg",
                    text=f"机械占位-{sequence:04d}",
                    intent=intent,
                    is_boundary=is_boundary,
                    boundary_strategy=("natural_ambiguity" if is_boundary else None),
                    split="dev_mini" if is_locked_dev else "opt_pool",
                    generator_batch_id=generator_batch_id,
                )
            )
            if is_locked_dev:
                locked_dev_ids.add(query_id)
    assert len(queries) == 4500
    assert sum(query.is_boundary for query in queries) == 788
    assert len(locked_dev_ids) == 200
    return queries, locked_dev_ids


@dataclass
class LabeledSplitFixture:
    queries_root: Path
    full_plan_path: Path

    def inject(self, fault: str) -> None:
        labels = self.queries_root / "labels"
        manifest_path = labels / "manifest.json"
        if fault == "missing_query_id":
            rows = (labels / "labeled_queries.jsonl").read_text("utf-8").splitlines()
            (labels / "labeled_queries.jsonl").write_text(
                "\n".join(rows[:-1]) + "\n",
                encoding="utf-8",
            )
            return
        manifest = json.loads(manifest_path.read_text("utf-8"))
        if fault == "accepted_source_hash_mismatch":
            manifest["accepted_source_sha256"] = "f" * 64
        elif fault in {"unresolved_arbitration", "arbitration_label_mismatch"}:
            first = Query.model_validate_json(
                (labels / "labeled_queries.jsonl").read_text("utf-8").splitlines()[0]
            )
            queue = canonical_jsonl_bytes(
                [
                    {
                        "query_id": first.query_id,
                        "image_path": first.image_path,
                        "text": first.text,
                        "constructed_intent": first.gt_intent,
                        "qwen_intent": first.gt_intent,
                        "qwen_reason": "机械理由",
                        "qwen_review_error": None,
                        "deepseek_intent": first.gt_intent,
                        "deepseek_reason": "机械理由",
                        "deepseek_review_error": None,
                        "spot_check": True,
                    }
                ]
            )
            (labels / "arbitration_queue.jsonl").write_bytes(queue)
            manifest["arbitration_queue_sha256"] = sha256_bytes(queue)
            if fault == "arbitration_label_mismatch":
                (labels / "arbitrations.jsonl").write_bytes(
                    canonical_jsonl_bytes(
                        [
                            {
                                "query_id": first.query_id,
                                "intent": "multi_product",
                                "reason": "机械裁决",
                                "decided_at": "2026-07-11T00:00:00Z",
                            }
                        ]
                    )
                )
        else:  # pragma: no cover - fixture misuse guard
            raise AssertionError(f"unknown fault: {fault}")
        manifest_path.write_bytes(canonical_json_bytes(manifest))


@pytest.fixture
def labeled_split_fixture(fake_image_root: Path, tmp_path: Path) -> LabeledSplitFixture:
    root = tmp_path / "queries"
    dev_plan = build_dev_mini_plan(
        fake_image_root,
        seed=20260711,
        allow_provisional_asset_groups=True,
    )
    dev_path = root / "plans/dev_mini.json"
    _, dev_manifest_path = write_dev_mini_plan(dev_plan, dev_path)
    dev_sha = json.loads(dev_manifest_path.read_text("utf-8"))["plan_sha256"]
    pools = {
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
        pools,
        seed=20260711,
        allow_provisional_asset_groups=True,
    )
    full_path = root / "plans/full.json"
    _, full_manifest_path = write_full_plan(
        full_plan,
        full_path,
        parent_plan_sha256=dev_sha,
    )
    full_sha = json.loads(full_manifest_path.read_text("utf-8"))["plan_sha256"]

    queries = [
        make_query_v2(
            query_id=item.plan_id,
            image_path=item.image_path,
            text=f"机械占位-{item.plan_id}",
            intent=item.canonical_intent,
            is_boundary=item.is_boundary,
            boundary_strategy=item.boundary_strategy,
            boundary_group_id=item.boundary_group_id,
            split=item.provisional_split,
            generator_batch_id=item.generator_batch_id,
            label_status="cross_agreed",
            asset_id=item.asset_id,
            leakage_group_id=item.leakage_group_id,
            template_family=item.template_family,
            capability=item.canonical_capability,
            requires_card=item.requires_card,
        )
        for item in full_plan.queries
    ]
    source_bytes = canonical_jsonl_bytes(queries)
    root.mkdir(parents=True, exist_ok=True)
    (root / "queries.jsonl").write_bytes(source_bytes)

    ordered_batches = list(dict.fromkeys(item.batch_id for item in full_plan.queries))
    ledger = []
    accepted_root = root / "accepted"
    accepted_root.mkdir(parents=True)
    for index, base_batch_id in enumerate(ordered_batches, start=1):
        batch_id = f"{base_batch_id}-r1"
        batch_queries = [
            query for query in queries if query.generator_batch_id == base_batch_id
        ]
        assert len(batch_queries) == 25
        results_bytes = canonical_jsonl_bytes(batch_queries)
        results_sha256 = sha256_bytes(results_bytes)
        plan_sha256 = dev_sha if index <= 8 else full_sha
        batch_dir = accepted_root / batch_id
        batch_dir.mkdir()
        (batch_dir / "results.jsonl").write_bytes(results_bytes)
        (batch_dir / "manifest.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": 2,
                    "query_schema_version": 2,
                    "plan_schema_version": 2,
                    "base_batch_id": base_batch_id,
                    "revision": 1,
                    "data_origin": "synthetic_derived",
                    "provider": "codex",
                    "model_display_name": "5.6 Sol Ultra",
                    "model_claim_source": "user_confirmation",
                    "generated_at": "2026-07-11T00:00:00Z",
                    "staged_at": "2026-07-11T00:00:00Z",
                    "count": 25,
                    "plan_sha256": plan_sha256,
                    "asset_catalog_sha256": None,
                    "leakage_policy_version": ("relative-path-plus-plan-groups-v1"),
                    "seed_set_sha256": "a" * 64,
                    "results_sha256": results_sha256,
                }
            )
        )
        ledger.append(
            {
                "batch_id": batch_id,
                "base_batch_id": base_batch_id,
                "revision": 1,
                "count": 25,
                "results_sha256": results_sha256,
                "plan_sha256": plan_sha256,
                "seed_set_sha256": "a" * 64,
                "accepted_at": "2026-07-11T00:00:00Z",
            }
        )
    (root / "accepted-ledger.jsonl").write_bytes(canonical_jsonl_bytes(ledger))

    labels = root / "labels"
    labels.mkdir()
    review_rows = [
        {
            "query_id": query.query_id,
            "constructed_intent": query.canonical_intent,
            "qwen_intent": query.canonical_intent,
            "qwen_reason": "机械一致",
            "qwen_raw": ["机械一致"],
            "qwen_review_error": None,
            "deepseek_intent": query.canonical_intent,
            "deepseek_reason": "机械一致",
            "deepseek_raw": ["机械一致"],
            "deepseek_review_error": None,
            "label_status": "cross_agreed",
            "needs_arbitration": False,
        }
        for query in queries
    ]
    reviews = canonical_jsonl_bytes(review_rows)
    spot_count = 225
    spot_ids = {
        query.query_id
        for query in sorted(
            queries,
            key=lambda item: (
                hashlib.sha256(item.query_id.encode("utf-8")).hexdigest(),
                item.query_id,
            ),
        )[:spot_count]
    }
    queue_rows = []
    arbitration_rows = []
    labeled_queries = []
    for query, review in zip(queries, review_rows, strict=True):
        if query.query_id not in spot_ids:
            labeled_queries.append(query)
            continue
        queue_rows.append(
            {
                "query_id": query.query_id,
                "image_path": query.image_path,
                "text": query.text,
                "constructed_intent": query.canonical_intent,
                "constructed_capability": query.canonical_capability,
                "constructed_acceptable_capabilities": query.acceptable_capabilities,
                "constructed_requires_card": query.requires_card,
                "qwen_intent": review["qwen_intent"],
                "qwen_reason": review["qwen_reason"],
                "qwen_review_error": None,
                "deepseek_intent": review["deepseek_intent"],
                "deepseek_reason": review["deepseek_reason"],
                "deepseek_review_error": None,
                "spot_check": True,
            }
        )
        arbitration_rows.append(
            {
                "query_id": query.query_id,
                "intent": query.canonical_intent,
                "canonical_capability": query.canonical_capability,
                "acceptable_capabilities": query.acceptable_capabilities,
                "requires_card": query.requires_card,
                "reason": "机械抽检确认",
                "decided_at": "2026-07-11T00:00:00Z",
            }
        )
        payload = query.model_dump(mode="json")
        payload["label_status"] = "arbitrated"
        payload["label_provenance"].append(
            {
                "decision_type": "arbitration",
                "annotator_kind": "human",
                "annotator_id": "mechanical-test-fixture",
                "canonical_intent": query.canonical_intent,
                "canonical_capability": query.canonical_capability,
                "acceptable_capabilities": query.acceptable_capabilities,
                "reason": "机械抽检确认",
                "decided_at": "2026-07-11T00:00:00Z",
            }
        )
        labeled_queries.append(Query.model_validate(payload))
    queue = canonical_jsonl_bytes(queue_rows)
    arbitrations = canonical_jsonl_bytes(arbitration_rows)
    (labels / "reviews.jsonl").write_bytes(reviews)
    (labels / "arbitration_queue.jsonl").write_bytes(queue)
    (labels / "arbitrations.jsonl").write_bytes(arbitrations)
    (labels / "labeled_queries.jsonl").write_bytes(
        canonical_jsonl_bytes(labeled_queries)
    )
    (labels / "manifest.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 2,
                "query_schema_version": 2,
                "taxonomy_version": TAXONOMY_VERSION,
                "generated_at": "2026-07-11T00:00:00Z",
                "accepted_source": str((root / "queries.jsonl").resolve()),
                "accepted_source_sha256": sha256_bytes(source_bytes),
                "plan_sha256": full_sha,
                "asset_catalog_sha256": None,
                "leakage_policy_version": "relative-path-plus-plan-groups-v1",
                "count": 4500,
                "review_results_sha256": sha256_bytes(reviews),
                "arbitration_queue_sha256": sha256_bytes(queue),
                "disagreement_count": 0,
                "review_error_count": 0,
                "spot_check_count": spot_count,
            }
        )
    )
    return LabeledSplitFixture(queries_root=root, full_plan_path=full_path)
