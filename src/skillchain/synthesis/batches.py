"""正式话术草稿的确定性校验、Query 物化与 staging。"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from skillchain.data import publish_staged_directory_and_file
from skillchain.data.asset_catalog import AssetCatalog
from skillchain.schemas import LabelDecision, Query
from skillchain.synthesis.models import (
    BatchDraftManifest,
    BatchManifest,
    CorpusPlan,
    GeneratedTrajectory,
    PlanManifest,
    PlannedQuery,
)
from skillchain.synthesis.seeds import (
    list_staged_seed_manifests,
    load_accepted_seed_manifest,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    atomic_replace_file,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    new_staging_directory,
    next_revision,
    sha256_bytes,
)

_PUNCTUATION = "，。！？；：、,.!?;:‘’“”\"'（）()【】[]{}<>《》-—_"
_SAFE_BATCH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION_ID = re.compile(r"^(?P<base>[A-Za-z0-9][A-Za-z0-9._-]*)-r(?P<revision>\d+)$")


class AcceptedLedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    base_batch_id: str
    revision: int
    count: int
    results_sha256: str
    plan_sha256: str
    seed_set_sha256: str
    accepted_at: datetime


def _next_unaccepted_plan_batch(
    plan: CorpusPlan,
    entries: Sequence[AcceptedLedgerEntry],
) -> str | None:
    """Return the next batch only when the ledger is an exact plan prefix."""

    ordered_batches = tuple(dict.fromkeys(item.batch_id for item in plan.queries))
    accepted_batches = tuple(entry.base_batch_id for entry in entries)
    expected_prefix = ordered_batches[: len(accepted_batches)]
    if accepted_batches != expected_prefix:
        raise ValueError(
            "accepted ledger is not an exact active-plan batch prefix: "
            f"expected {list(expected_prefix)}, got {list(accepted_batches)}"
        )
    if len(accepted_batches) == len(ordered_batches):
        return None
    return ordered_batches[len(accepted_batches)]


def _require_next_plan_batch(
    plan: CorpusPlan,
    entries: Sequence[AcceptedLedgerEntry],
    requested_base_batch_id: str,
) -> None:
    expected = _next_unaccepted_plan_batch(plan, entries)
    if expected is None:
        raise ValueError("all active-plan batches are already accepted")
    if requested_base_batch_id != expected:
        raise ValueError(
            "batch must follow active plan order: "
            f"expected {expected}, got {requested_base_batch_id}"
        )


def normalized_text(value: str) -> str:
    return "".join(value.casefold().split()).translate(
        str.maketrans("", "", _PUNCTUATION)
    )


def compute_generation_input_sha256(
    *,
    base_batch_id: str,
    plan_items: Sequence[PlannedQuery],
    plan_sha256: str,
    asset_catalog_sha256: str | None,
    leakage_policy_version: str,
    seed_set_sha256: str,
) -> str:
    """Canonical digest of every authoritative pre-generation batch input."""

    base_batch_id = _validate_batch_id(base_batch_id)
    ordered_items = sorted(plan_items, key=lambda item: item.position)
    if len(ordered_items) != 25:
        raise ValueError(
            "generation input 必须包含 base batch 的完整 25 个 planned items"
        )
    if any(item.batch_id != base_batch_id for item in ordered_items):
        raise ValueError("generation input planned items 不属于指定 base_batch_id")
    if [item.position for item in ordered_items] != list(range(1, 26)):
        raise ValueError("generation input planned items 必须恰好覆盖 position 1..25")
    if len({item.plan_id for item in ordered_items}) != 25:
        raise ValueError("generation input planned items 的 plan_id 必须唯一")
    payload = {
        "schema_version": 1,
        "base_batch_id": base_batch_id,
        "planned_items": [item.model_dump(mode="json") for item in ordered_items],
        "plan_sha256": plan_sha256,
        "asset_catalog_sha256": asset_catalog_sha256,
        "leakage_policy_version": leakage_policy_version,
        "seed_set_sha256": seed_set_sha256,
    }
    return sha256_bytes(canonical_json_bytes(payload))


def stage_generated_batch(
    *,
    draft_path: str | Path,
    draft_manifest_path: str | Path,
    plan_path: str | Path,
    queries_root: str | Path,
    base_batch_id: str,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
) -> Path:
    """验证 25 条模型草稿，回填权威计划字段，并发布一个 staging revision。"""

    base_batch_id = _validate_batch_id(base_batch_id)
    queries_root = Path(queries_root)
    plan, plan_manifest = _load_plan(Path(plan_path))
    _require_formal_or_explicit_provisional(
        plan,
        plan_manifest,
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
    )
    _require_next_plan_batch(
        plan,
        _load_ledger(queries_root / "accepted-ledger.jsonl"),
        base_batch_id,
    )
    plan_items = sorted(
        (item for item in plan.queries if item.batch_id == base_batch_id),
        key=lambda item: item.position,
    )
    if len(plan_items) != 25:
        raise ValueError(f"{base_batch_id} 在计划中必须恰好有 25 条")

    seed_manifest = load_accepted_seed_manifest(queries_root)
    draft_manifest = _load_draft_manifest(Path(draft_manifest_path))
    if draft_manifest.base_batch_id != base_batch_id:
        raise ValueError("draft manifest 的 base_batch_id 与请求不一致")
    draft_bytes, drafts = _load_drafts(Path(draft_path))
    draft_sha256 = sha256_bytes(draft_bytes)
    if draft_manifest.plan_sha256 != plan_manifest.plan_sha256:
        raise ValueError("draft manifest 的 plan_sha256 与当前 plan 不一致")
    if draft_manifest.asset_catalog_sha256 != plan_manifest.asset_catalog_sha256:
        raise ValueError("draft manifest 的 asset_catalog_sha256 与当前 plan 不一致")
    if draft_manifest.leakage_policy_version != plan_manifest.leakage_policy_version:
        raise ValueError("draft manifest 的 leakage_policy_version 与当前 plan 不一致")
    if draft_manifest.seed_set_sha256 != seed_manifest.seed_set_sha256:
        raise ValueError(
            "draft manifest 的 seed_set_sha256 与当前 accepted seed 不一致"
        )
    if draft_manifest.draft_sha256 != draft_sha256:
        raise ValueError("draft manifest 的 draft_sha256 与 draft JSONL 不一致")
    generation_input_sha256 = compute_generation_input_sha256(
        base_batch_id=base_batch_id,
        plan_items=plan_items,
        plan_sha256=plan_manifest.plan_sha256,
        asset_catalog_sha256=plan_manifest.asset_catalog_sha256,
        leakage_policy_version=plan_manifest.leakage_policy_version,
        seed_set_sha256=seed_manifest.seed_set_sha256,
    )
    if draft_manifest.generation_input_sha256 != generation_input_sha256:
        raise ValueError(
            "draft manifest 的 generation_input_sha256 与当前生成输入不一致"
        )
    expected_ids = [item.plan_id for item in plan_items]
    drafts_by_id: dict[str, GeneratedTrajectory] = {}
    for draft in drafts:
        if draft.plan_id in drafts_by_id:
            raise ValueError(f"结果包含重复 plan_id: {draft.plan_id}")
        drafts_by_id[draft.plan_id] = draft
    unknown = sorted(set(drafts_by_id) - set(expected_ids))
    missing = sorted(set(expected_ids) - set(drafts_by_id))
    if unknown:
        raise ValueError(f"结果包含未知 plan_id: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"结果缺少 plan_id: {', '.join(missing)}")

    revision = next_revision(queries_root, base_batch_id)
    batch_id = f"{base_batch_id}-r{revision}"
    queries: list[Query] = []
    for item in plan_items:
        draft = drafts_by_id[item.plan_id]
        text = draft.turns[-1].content
        queries.append(
            Query(
                schema_version=2,
                taxonomy_version=item.taxonomy_version,
                task_spec_version=item.task_spec_version,
                query_id=item.plan_id,
                asset_id=item.asset_id,
                image_path=item.image_path,
                leakage_group_id=item.leakage_group_id,
                boundary_group_id=item.boundary_group_id,
                template_family=item.template_family,
                generator_batch_id=item.generator_batch_id,
                text=text,
                turns=draft.turns,
                canonical_intent=item.canonical_intent,
                canonical_capability=item.canonical_capability,
                acceptable_capabilities=item.acceptable_capabilities,
                is_boundary=item.is_boundary,
                boundary_strategy=item.boundary_strategy,
                requires_card=item.requires_card,
                episode="t0",
                split=item.provisional_split,
                label_status="auto",
                label_provenance=[
                    LabelDecision(
                        decision_type="constructed",
                        annotator_kind="planner",
                        annotator_id="corpus-plan-v2",
                        canonical_intent=item.canonical_intent,
                        canonical_capability=item.canonical_capability,
                        acceptable_capabilities=item.acceptable_capabilities,
                        reason="label copied from the frozen schema-v2 corpus plan",
                        source_artifact_sha256=plan_manifest.plan_sha256,
                    )
                ],
                synth_provider="codex",
                synth_model="5.6 Sol Ultra",
                synthesis_batch_id=batch_id,
                synthesis_prompt_id=item.plan_id,
                seed_set_sha256=seed_manifest.seed_set_sha256,
            )
        )

    normalized = [normalized_text(query.text) for query in queries]
    if any(not value for value in normalized):
        raise ValueError("规范化后存在空话术")
    duplicate_count = len(normalized) - len(set(normalized))
    if duplicate_count:
        raise ValueError(f"批次内存在 {duplicate_count} 条规范化重复话术")
    accepted_texts = _load_accepted_normalized_texts(queries_root / "queries.jsonl")
    overlap = sorted(set(normalized) & accepted_texts)
    if overlap:
        raise ValueError(f"accepted queries 存在重复文本，共 {len(overlap)} 条")

    results_bytes = canonical_jsonl_bytes(queries)
    results_sha256 = sha256_bytes(results_bytes)
    staged_at = datetime.now(timezone.utc)
    batch_manifest = BatchManifest(
        base_batch_id=base_batch_id,
        revision=revision,
        data_origin=draft_manifest.data_origin,
        provider=draft_manifest.provider,
        model_display_name=draft_manifest.model_display_name,
        model_claim_source=draft_manifest.model_claim_source,
        generated_at=draft_manifest.generated_at,
        staged_at=staged_at,
        count=25,
        plan_sha256=plan_manifest.plan_sha256,
        asset_catalog_sha256=plan_manifest.asset_catalog_sha256,
        leakage_policy_version=plan_manifest.leakage_policy_version,
        seed_set_sha256=seed_manifest.seed_set_sha256,
        results_sha256=results_sha256,
    )
    intent_counts = Counter(query.canonical_intent for query in queries)
    quality_report = {
        "schema_version": 1,
        "query_schema_version": 2,
        "batch_id": batch_id,
        "data_origin": draft_manifest.data_origin,
        "count": len(queries),
        "intent_counts": {key: intent_counts[key] for key in sorted(intent_counts)},
        "boundary_count": sum(query.is_boundary for query in queries),
        "single_turn_count": sum(len(query.turns) == 1 for query in queries),
        "two_turn_count": sum(len(query.turns) == 3 for query in queries),
        "duplicate_count": duplicate_count,
        "plan_sha256": plan_manifest.plan_sha256,
        "asset_catalog_sha256": plan_manifest.asset_catalog_sha256,
        "leakage_policy_version": plan_manifest.leakage_policy_version,
        "seed_set_sha256": seed_manifest.seed_set_sha256,
        "results_sha256": results_sha256,
    }

    destination = queries_root / "staging" / batch_id
    staging = new_staging_directory(destination)
    try:
        atomic_create_file(staging / "results.jsonl", results_bytes)
        atomic_create_file(
            staging / "manifest.json", canonical_json_bytes(batch_manifest)
        )
        atomic_create_file(
            staging / "quality_report.json", canonical_json_bytes(quality_report)
        )
        return atomic_publish_new_directory(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def accept_generated_batch(
    queries_root: str | Path,
    batch_id: str,
    *,
    confirmation: str,
    plan_path: str | Path,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
) -> Path:
    """在重新校验全部内容后，联合发布 accepted 目录与 ledger。"""

    if confirmation != "ACCEPT":
        raise ValueError("接受批次必须提供字面确认 ACCEPT")
    batch_id = _validate_revision_id(batch_id)
    queries_root = Path(queries_root)
    staged = queries_root / "staging" / batch_id
    accepted = queries_root / "accepted" / batch_id
    ledger_path = queries_root / "accepted-ledger.jsonl"
    entries = _load_ledger(ledger_path)
    plan, _ = _load_plan(Path(plan_path))
    next_base_batch_id = _next_unaccepted_plan_batch(plan, entries)

    if accepted.is_dir() and not staged.exists():
        matching = [entry for entry in entries if entry.batch_id == batch_id]
        if len(matching) != 1:
            raise ValueError("accepted 目录与 ledger 不一致")
        _verify_batch_directory(
            accepted,
            Path(plan_path),
            queries_root,
            expected_entry=matching[0],
            asset_catalog=asset_catalog,
            allow_provisional_asset_groups=allow_provisional_asset_groups,
        )
        rebuild_accepted_queries(queries_root)
        return accepted
    if not staged.is_dir():
        raise FileNotFoundError(f"batch staging 不存在: {staged}")

    manifest, _ = _verify_batch_directory(
        staged,
        Path(plan_path),
        queries_root,
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
    )
    if next_base_batch_id is None:
        raise ValueError("all active-plan batches are already accepted")
    if manifest.base_batch_id != next_base_batch_id:
        raise ValueError(
            "batch must follow active plan order: "
            f"expected {next_base_batch_id}, got {manifest.base_batch_id}"
        )
    if any(entry.base_batch_id == manifest.base_batch_id for entry in entries):
        raise ValueError(f"ledger 已存在 base_batch_id: {manifest.base_batch_id}")
    entry = AcceptedLedgerEntry(
        batch_id=batch_id,
        base_batch_id=manifest.base_batch_id,
        revision=manifest.revision,
        count=manifest.count,
        results_sha256=manifest.results_sha256,
        plan_sha256=manifest.plan_sha256,
        seed_set_sha256=manifest.seed_set_sha256,
        accepted_at=datetime.now(timezone.utc),
    )
    new_ledger_bytes = canonical_jsonl_bytes([*entries, entry])

    accepted.parent.mkdir(parents=True, exist_ok=True)
    queries_root.mkdir(parents=True, exist_ok=True)
    ledger_staging: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=queries_root,
            prefix=".accepted-ledger.",
            suffix=".staging",
            delete=False,
        ) as handle:
            ledger_staging = Path(handle.name)
            handle.write(new_ledger_bytes)
            handle.flush()
        publish_staged_directory_and_file(
            staged,
            accepted,
            ledger_staging,
            ledger_path,
        )
    finally:
        if ledger_staging is not None:
            ledger_staging.unlink(missing_ok=True)
    rebuild_accepted_queries(queries_root)
    return accepted


def reject_generated_batch(
    queries_root: str | Path, batch_id: str, *, reason: str
) -> Path:
    batch_id = _validate_revision_id(batch_id)
    reason = reason.strip()
    if not reason:
        raise ValueError("拒绝理由不得为空")
    queries_root = Path(queries_root)
    staged = queries_root / "staging" / batch_id
    if not staged.is_dir():
        raise FileNotFoundError(f"batch staging 不存在: {staged}")
    destination = queries_root / "rejected" / batch_id
    reason_path = staged / "reason.json"
    reason_bytes = canonical_json_bytes({"reason": reason})
    try:
        atomic_create_file(reason_path, reason_bytes)
    except FileExistsError:
        if not reason_path.is_file() or reason_path.read_bytes() != reason_bytes:
            raise
    return atomic_publish_new_directory(staged, destination)


def rebuild_accepted_queries(queries_root: str | Path) -> Path:
    """从不可变 accepted 目录重建派生 queries.jsonl。"""

    queries_root = Path(queries_root)
    entries = _load_ledger(queries_root / "accepted-ledger.jsonl")
    chunks: list[bytes] = []
    for entry in entries:
        directory = queries_root / "accepted" / entry.batch_id
        manifest = _load_batch_manifest(directory / "manifest.json")
        results = (directory / "results.jsonl").read_bytes()
        if sha256_bytes(results) != manifest.results_sha256:
            raise ValueError(f"accepted results 哈希不一致: {entry.batch_id}")
        if (
            manifest.results_sha256 != entry.results_sha256
            or manifest.plan_sha256 != entry.plan_sha256
            or manifest.seed_set_sha256 != entry.seed_set_sha256
        ):
            raise ValueError(f"accepted manifest 与 ledger 不一致: {entry.batch_id}")
        queries = _parse_query_jsonl(
            results, f"accepted/{entry.batch_id}/results.jsonl"
        )
        if len(queries) != entry.count:
            raise ValueError(f"accepted count 与 ledger 不一致: {entry.batch_id}")
        chunks.append(canonical_jsonl_bytes(queries))
    destination = queries_root / "queries.jsonl"
    expected = b"".join(chunks)
    if destination.exists() and destination.read_bytes() == expected:
        return destination
    return atomic_replace_file(destination, expected)


def verify_accepted_corpus(queries_root: str | Path) -> list[Query]:
    """只读验证 accepted 语料的权威账本、批次目录与派生查询视图。"""

    queries_root = Path(queries_root)
    ledger_path = queries_root / "accepted-ledger.jsonl"
    ledger_bytes = _read_required_regular_file(ledger_path, "accepted ledger")
    entries = _load_ledger(ledger_path)
    if ledger_bytes != canonical_jsonl_bytes(entries):
        raise ValueError("accepted ledger 不是 canonical JSONL")

    for entry in entries:
        _validate_ledger_identity(entry)

    accepted_root = queries_root / "accepted"
    if accepted_root.is_symlink() or not accepted_root.is_dir():
        raise ValueError(f"accepted 目录不存在或不是普通目录: {accepted_root}")
    accepted_children = list(accepted_root.iterdir())
    invalid_children = sorted(
        child.name
        for child in accepted_children
        if child.is_symlink() or not child.is_dir()
    )
    if invalid_children:
        raise ValueError(
            "accepted 目录只能包含普通批次目录: " + ", ".join(invalid_children)
        )
    expected_directories = {entry.batch_id for entry in entries}
    actual_directories = {child.name for child in accepted_children}
    if actual_directories != expected_directories:
        missing = sorted(expected_directories - actual_directories)
        unexpected = sorted(actual_directories - expected_directories)
        details = []
        if missing:
            details.append("缺少 " + ", ".join(missing))
        if unexpected:
            details.append("未入账 " + ", ".join(unexpected))
        raise ValueError("accepted 目录集合与 ledger 不一致: " + "; ".join(details))

    all_queries: list[Query] = []
    canonical_chunks: list[bytes] = []
    for entry in entries:
        directory = accepted_root / entry.batch_id
        manifest_path = directory / "manifest.json"
        manifest_bytes = _read_required_regular_file(
            manifest_path, f"accepted/{entry.batch_id}/manifest.json"
        )
        manifest = _parse_batch_manifest_bytes(manifest_bytes, manifest_path)
        if manifest_bytes != canonical_json_bytes(manifest):
            raise ValueError(
                f"accepted batch manifest 不是 canonical JSON: {entry.batch_id}"
            )
        _verify_manifest_ledger_binding(manifest, entry)

        results_path = directory / "results.jsonl"
        results_bytes = _read_required_regular_file(
            results_path, f"accepted/{entry.batch_id}/results.jsonl"
        )
        queries = _parse_query_jsonl(
            results_bytes, f"accepted/{entry.batch_id}/results.jsonl"
        )
        canonical_results = canonical_jsonl_bytes(queries)
        if results_bytes != canonical_results:
            raise ValueError(
                f"accepted batch results 不是 canonical JSONL: {entry.batch_id}"
            )
        actual_results_sha256 = sha256_bytes(results_bytes)
        if (
            actual_results_sha256 != manifest.results_sha256
            or actual_results_sha256 != entry.results_sha256
        ):
            raise ValueError(f"accepted results SHA256 不一致: {entry.batch_id}")
        if len(queries) != manifest.count or len(queries) != entry.count:
            raise ValueError(f"accepted results count 不一致: {entry.batch_id}")

        canonical_chunks.append(canonical_results)
        all_queries.extend(queries)

    derived_path = queries_root / "queries.jsonl"
    derived_bytes = _read_required_regular_file(derived_path, "derived queries.jsonl")
    expected_derived = b"".join(canonical_chunks)
    if derived_bytes != expected_derived:
        raise ValueError(
            "queries.jsonl 不等于按 accepted ledger 顺序拼接的 canonical results"
        )
    return all_queries


def corpus_status(
    queries_root: str | Path, *, plan_path: str | Path | None = None
) -> dict:
    queries_root = Path(queries_root)
    entries = _load_ledger(queries_root / "accepted-ledger.jsonl")
    staged_seed_manifests = list_staged_seed_manifests(queries_root)
    accepted_seed_path = queries_root / "seeds" / "accepted" / "seed_examples.json"
    accepted_seed_exists = accepted_seed_path.is_file()
    if accepted_seed_exists:
        load_accepted_seed_manifest(queries_root)
    next_batch_id: str | None = None
    revision: int | None = None
    plan_sha256: str | None = None
    if plan_path is None:
        try:
            from skillchain.synthesis.planning import read_active_plan

            plan_path = read_active_plan(queries_root)[0]
        except FileNotFoundError:
            plan_path = None
    if plan_path is not None:
        plan, manifest = _load_plan(Path(plan_path))
        plan_sha256 = manifest.plan_sha256
        next_batch_id = _next_unaccepted_plan_batch(plan, entries)
        if next_batch_id is not None:
            try:
                revision = next_revision(queries_root, next_batch_id)
            except ValueError:
                staging = sorted((queries_root / "staging").glob(f"{next_batch_id}-r*"))
                if staging:
                    revision = _load_batch_manifest(
                        staging[0] / "manifest.json"
                    ).revision
                else:
                    raise
    return {
        "accepted_batches": len(entries),
        "accepted_queries": sum(entry.count for entry in entries),
        "next_batch_id": next_batch_id,
        "next_revision": revision,
        "seed_status": (
            "accepted"
            if accepted_seed_exists
            else ("staging" if staged_seed_manifests else "missing")
        ),
        "staged_seed_batch_ids": [
            manifest.seed_batch_id for manifest in staged_seed_manifests
        ],
        "active_plan_sha256": plan_sha256,
    }


def show_next(queries_root: str | Path) -> dict:
    from skillchain.synthesis.planning import read_active_plan

    queries_root = Path(queries_root)
    _, plan, manifest, _ = read_active_plan(queries_root)
    status = corpus_status(queries_root)
    batch_id = status["next_batch_id"]
    if batch_id is None:
        return {
            "batch_id": None,
            "revision": None,
            "plan_sha256": manifest.plan_sha256,
            "items": [],
        }
    items = sorted(
        (item for item in plan.queries if item.batch_id == batch_id),
        key=lambda item: item.position,
    )
    return {
        "batch_id": batch_id,
        "revision": status["next_revision"],
        "plan_sha256": manifest.plan_sha256,
        "items": [item.model_dump(mode="json") for item in items],
    }


def _verify_batch_directory(
    directory: Path,
    plan_path: Path,
    queries_root: Path,
    *,
    expected_entry: AcceptedLedgerEntry | None = None,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
) -> tuple[BatchManifest, list[Query]]:
    manifest = _load_batch_manifest(directory / "manifest.json")
    expected_name = f"{manifest.base_batch_id}-r{manifest.revision}"
    if directory.name != expected_name:
        raise ValueError("batch 目录名与 manifest revision 不一致")
    results = (directory / "results.jsonl").read_bytes()
    if sha256_bytes(results) != manifest.results_sha256:
        raise ValueError("results 哈希与 batch manifest 不一致")
    plan, plan_manifest = _load_plan(plan_path)
    _require_formal_or_explicit_provisional(
        plan,
        plan_manifest,
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
    )
    if (
        plan_manifest.plan_sha256 != manifest.plan_sha256
        or plan_manifest.asset_catalog_sha256 != manifest.asset_catalog_sha256
        or plan_manifest.leakage_policy_version != manifest.leakage_policy_version
    ):
        raise ValueError("plan/catalog binding 与 batch manifest 不一致")
    seed_manifest = load_accepted_seed_manifest(queries_root)
    if seed_manifest.seed_set_sha256 != manifest.seed_set_sha256:
        raise ValueError("seed_set_sha256 哈希与 batch manifest 不一致")
    queries = _parse_query_jsonl(results, directory.name)
    if len(queries) != manifest.count:
        raise ValueError("results count 与 batch manifest 不一致")
    if asset_catalog is not None:
        asset_catalog.verify_asset_ids(query.asset_id for query in queries)
    plan_items = {
        item.plan_id: item
        for item in plan.queries
        if item.batch_id == manifest.base_batch_id
    }
    if set(plan_items) != {query.query_id for query in queries}:
        raise ValueError("results query_id 与当前 plan 不一致")
    for query in queries:
        item = plan_items[query.query_id]
        if (
            query.schema_version != 2
            or query.taxonomy_version != item.taxonomy_version
            or query.task_spec_version != item.task_spec_version
            or query.asset_id != item.asset_id
            or query.image_path != item.image_path
            or query.leakage_group_id != item.leakage_group_id
            or query.boundary_group_id != item.boundary_group_id
            or query.template_family != item.template_family
            or query.generator_batch_id != item.generator_batch_id
            or query.canonical_intent != item.canonical_intent
            or query.canonical_capability != item.canonical_capability
            or query.acceptable_capabilities != item.acceptable_capabilities
            or query.is_boundary != item.is_boundary
            or query.boundary_strategy != item.boundary_strategy
            or query.requires_card != item.requires_card
            or query.split != item.provisional_split
            or query.synthesis_batch_id != directory.name
            or query.synthesis_prompt_id != item.plan_id
            or query.seed_set_sha256 != manifest.seed_set_sha256
        ):
            raise ValueError(f"results 与权威 plan/seed 不一致: {query.query_id}")
    if expected_entry is not None and (
        expected_entry.results_sha256 != manifest.results_sha256
        or expected_entry.plan_sha256 != manifest.plan_sha256
        or expected_entry.seed_set_sha256 != manifest.seed_set_sha256
    ):
        raise ValueError("accepted batch 与 ledger 哈希不一致")
    return manifest, queries


def _require_formal_or_explicit_provisional(
    plan: CorpusPlan,
    plan_manifest: PlanManifest,
    *,
    asset_catalog: AssetCatalog | None,
    allow_provisional_asset_groups: bool,
) -> None:
    if plan_manifest.asset_catalog_sha256 is None:
        if asset_catalog is not None:
            raise ValueError("provisional plan 不能事后绑定 asset catalog")
        if not allow_provisional_asset_groups:
            raise ValueError(
                "provisional plan batch operations require explicit "
                "allow_provisional_asset_groups=True"
            )
        return
    if asset_catalog is None:
        raise ValueError("catalog-bound batch operation requires the asset catalog")
    if (
        asset_catalog.manifest.catalog_sha256 != plan_manifest.asset_catalog_sha256
        or asset_catalog.manifest.leakage_policy_version
        != plan_manifest.leakage_policy_version
    ):
        raise ValueError("asset catalog 与 batch plan binding 不一致")
    for item in plan.queries:
        asset_catalog.verify_reference(
            item.asset_id,
            item.image_path,
            leakage_group_id=item.leakage_group_id,
        )


def _load_batch_manifest(path: Path) -> BatchManifest:
    try:
        return BatchManifest.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        raise FileNotFoundError(f"batch manifest 不存在: {path}") from None
    except ValidationError as exc:
        raise ValueError(f"batch manifest 校验失败: {exc}") from exc


def _parse_batch_manifest_bytes(content: bytes, path: Path) -> BatchManifest:
    try:
        return BatchManifest.model_validate_json(content)
    except ValidationError as exc:
        raise ValueError(f"batch manifest 校验失败: {path}: {exc}") from exc


def _read_required_regular_file(path: Path, artifact: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{artifact} 不存在或不是普通文件: {path}")
    return path.read_bytes()


def _validate_ledger_identity(entry: AcceptedLedgerEntry) -> None:
    batch_id = _validate_revision_id(entry.batch_id)
    match = _REVISION_ID.fullmatch(batch_id)
    assert match is not None  # _validate_revision_id 已完成同一正则校验。
    expected_base_batch_id = match.group("base")
    expected_revision = int(match.group("revision"))
    if (
        entry.base_batch_id != expected_base_batch_id
        or entry.revision != expected_revision
    ):
        raise ValueError(f"accepted ledger 批次身份字段不一致: {entry.batch_id}")


def _verify_manifest_ledger_binding(
    manifest: BatchManifest, entry: AcceptedLedgerEntry
) -> None:
    expected_batch_id = f"{manifest.base_batch_id}-r{manifest.revision}"
    if expected_batch_id != entry.batch_id:
        raise ValueError(
            f"accepted manifest 批次身份与 ledger 不一致: {entry.batch_id}"
        )
    if (
        manifest.base_batch_id != entry.base_batch_id
        or manifest.revision != entry.revision
        or manifest.count != entry.count
        or manifest.results_sha256 != entry.results_sha256
        or manifest.plan_sha256 != entry.plan_sha256
        or manifest.seed_set_sha256 != entry.seed_set_sha256
    ):
        raise ValueError(f"accepted manifest 字段与 ledger 不一致: {entry.batch_id}")


def _load_ledger(path: Path) -> list[AcceptedLedgerEntry]:
    if not path.exists():
        return []
    entries: list[AcceptedLedgerEntry] = []
    seen_batches: set[str] = set()
    seen_bases: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            entry = AcceptedLedgerEntry.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(f"ledger 第 {line_number} 行校验失败: {exc}") from exc
        if entry.batch_id in seen_batches or entry.base_batch_id in seen_bases:
            raise ValueError("ledger 包含重复 batch_id 或 base_batch_id")
        seen_batches.add(entry.batch_id)
        seen_bases.add(entry.base_batch_id)
        entries.append(entry)
    return entries


def _parse_query_jsonl(content: bytes, source: str) -> list[Query]:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source} 不是 UTF-8") from exc
    queries: list[Query] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            queries.append(Query.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{source} 第 {line_number} 行校验失败: {exc}") from exc
    return queries


def _validate_revision_id(value: str) -> str:
    value = value.strip()
    if not _REVISION_ID.fullmatch(value):
        raise ValueError("batch_id 必须采用 <base>-r<revision> 格式")
    return value


def _load_plan(path: Path) -> tuple[CorpusPlan, PlanManifest]:
    manifest_path = path.with_name(f"{path.stem}.manifest.json")
    try:
        plan_bytes = path.read_bytes()
        manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = PlanManifest.model_validate(manifest_raw)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"plan 或 manifest 不存在: {exc.filename}") from None
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"plan manifest 校验失败: {exc}") from exc
    actual_hash = hashlib.sha256(plan_bytes).hexdigest()
    if actual_hash != manifest.plan_sha256:
        raise ValueError("plan 哈希与 manifest 不一致")
    try:
        plan = CorpusPlan.model_validate_json(plan_bytes)
    except ValidationError as exc:
        raise ValueError(f"plan 内容校验失败: {exc}") from exc
    if (
        plan.scope != manifest.scope
        or len(plan.queries) != manifest.count
        or plan.asset_catalog_sha256 != manifest.asset_catalog_sha256
        or plan.leakage_policy_version != manifest.leakage_policy_version
    ):
        raise ValueError("plan 内容与 manifest scope/count/catalog binding 不一致")
    return plan, manifest


def _load_draft_manifest(path: Path) -> BatchDraftManifest:
    try:
        content = path.read_bytes()
        manifest = BatchDraftManifest.model_validate_json(content)
    except FileNotFoundError:
        raise FileNotFoundError(f"draft manifest 不存在: {path}") from None
    except ValidationError as exc:
        raise ValueError(f"draft manifest 校验失败: {exc}") from exc
    if content != canonical_json_bytes(manifest):
        raise ValueError("draft manifest 不是 canonical JSON")
    return manifest


def _load_drafts(path: Path) -> tuple[bytes, list[GeneratedTrajectory]]:
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(f"draft JSONL 不存在: {path}") from None
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"draft JSONL 不是有效 UTF-8: {exc}") from exc
    if len(lines) != 25:
        raise ValueError(f"生成草稿必须恰好 25 条，实际 {len(lines)} 条")
    drafts: list[GeneratedTrajectory] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"draft JSONL 第 {line_number} 行为空")
        try:
            drafts.append(GeneratedTrajectory.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"draft JSONL 第 {line_number} 行校验失败: {exc}") from exc
    return content, drafts


def _load_accepted_normalized_texts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    values: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            query = Query.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(
                f"accepted queries 第 {line_number} 行校验失败: {exc}"
            ) from exc
        values.add(normalized_text(query.text))
    return values


def _validate_batch_id(value: str) -> str:
    value = value.strip()
    if not _SAFE_BATCH_ID.fullmatch(value):
        raise ValueError("base_batch_id 只能包含字母、数字、点、下划线和连字符")
    return value
