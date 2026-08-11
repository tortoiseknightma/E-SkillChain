"""Phase 3 双盲意图复核、确定性抽检与人工仲裁派生产物。"""

from __future__ import annotations

import hashlib
import math
import shutil
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain import config
from skillchain.data.asset_catalog import AssetCatalog
from skillchain.llm import LLMResponse, chat
from skillchain.schemas import Intent, LabelDecision, Query
from skillchain.synthesis.models import PROVISIONAL_LEAKAGE_POLICY_VERSION
from skillchain.synthesis.planning import validate_capability_binding
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    atomic_replace_file,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    new_staging_directory,
    sha256_bytes,
)
from skillchain.task_spec import load_mvp_task_specification_v1
from skillchain.taxonomy import (
    capabilities_for_intent,
    load_default_taxonomy_registry,
)

Reviewer = Literal["qwen", "deepseek"]
ReviewChat = Callable[..., str | LLMResponse]
_INTENT_ADAPTER = TypeAdapter(Intent)
_MAX_PARSE_RETRIES = 2


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _nonblank(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} 不得为空")
    return value


class BlindReviewInput(StrictModel):
    """消息构造器唯一允许接收的查询字段。"""

    query_id: str
    text: str

    @field_validator("query_id", "text")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)


class ReviewRequest(StrictModel):
    reviewer: Reviewer
    messages: list[dict[str, str]]
    images: list[str]


class ReviewDecision(StrictModel):
    intent: Intent
    reason: str

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _nonblank(value, "reason")


class ReviewerOutcome(StrictModel):
    intent: Intent | None
    reason: str | None
    raw: list[str]
    review_error: str | None = None


class CrossReviewResult(StrictModel):
    query_id: str
    constructed_intent: Intent
    qwen_intent: Intent | None
    qwen_reason: str | None
    qwen_raw: list[str]
    qwen_review_error: str | None
    deepseek_intent: Intent | None
    deepseek_reason: str | None
    deepseek_raw: list[str]
    deepseek_review_error: str | None
    label_status: Literal["auto", "cross_agreed"]
    needs_arbitration: bool


class ArbitrationItem(StrictModel):
    query_id: str
    image_path: str
    text: str
    constructed_intent: Intent
    constructed_capability: str | None
    constructed_acceptable_capabilities: list[str]
    constructed_requires_card: bool | None
    qwen_intent: Intent | None
    qwen_reason: str | None
    qwen_review_error: str | None
    deepseek_intent: Intent | None
    deepseek_reason: str | None
    deepseek_review_error: str | None
    spot_check: bool


class ArbitrationRecord(StrictModel):
    query_id: str
    intent: Intent
    canonical_capability: str
    acceptable_capabilities: list[str]
    requires_card: bool
    reason: str
    decided_at: datetime

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _nonblank(value, "reason")


class LabelsManifest(StrictModel):
    schema_version: Literal[2] = 2
    query_schema_version: Literal[2] = 2
    taxonomy_version: str
    generated_at: datetime
    accepted_source: str
    accepted_source_sha256: str
    plan_sha256: str | None
    asset_catalog_sha256: str | None
    leakage_policy_version: str
    count: int
    review_results_sha256: str
    arbitration_queue_sha256: str
    disagreement_count: int
    review_error_count: int
    spot_check_count: int

    @field_validator(
        "accepted_source_sha256",
        "review_results_sha256",
        "arbitration_queue_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("哈希必须是 64 位小写十六进制")
        return value

    @field_validator("plan_sha256", "asset_catalog_sha256")
    @classmethod
    def validate_optional_sha256(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError("可选哈希必须是 64 位小写十六进制")
        return value

    @model_validator(mode="after")
    def validate_binding(self):
        provisional = self.leakage_policy_version == PROVISIONAL_LEAKAGE_POLICY_VERSION
        if provisional and self.asset_catalog_sha256 is not None:
            raise ValueError("provisional labels manifest 不得绑定 asset catalog")
        if not provisional and (
            self.plan_sha256 is None or self.asset_catalog_sha256 is None
        ):
            raise ValueError("正式 labels manifest 必须绑定 plan 与 asset catalog")
        return self


def build_review_request(
    blind: BlindReviewInput,
    *,
    reviewer: Reviewer,
    qwen_image_path: str | None = None,
) -> ReviewRequest:
    """只从盲审 DTO 构造消息；图片路径仅存在于 transport 字段。"""

    taxonomy = (
        "意图定义：exact_match=找同款或确认同一商品；"
        "multi_product=识别、拆解或比较图片中的多商品；"
        "divergent_rec=围绕风格、替代品或搭配进行发散推荐；"
        "encyclopedia=对可见对象做视觉百科式识别或知识解释；"
        "utility=执行 OCR、提取、换算、步骤指导等工具型任务。"
        "只返回 JSON 对象，且只能包含 intent 和 reason。intent 必须是 "
        "exact_match、multi_product、divergent_rec、encyclopedia、utility 之一。"
    )
    if reviewer == "qwen":
        if qwen_image_path is None or not qwen_image_path.strip():
            raise ValueError("Qwen 视觉复核必须提供图片传输路径")
        instruction = "独立判断给定图片与用户话术所表达的主要任务意图。"
        images = [qwen_image_path]
    elif reviewer == "deepseek":
        if qwen_image_path is not None:
            raise ValueError("DeepSeek 文本复核不得接收图片路径")
        instruction = "独立判断用户话术自身所表达的主要任务意图。"
        images = []
    else:  # pragma: no cover - Literal 的运行时防御
        raise ValueError(f"未知 reviewer: {reviewer}")

    messages = [
        {"role": "system", "content": f"{instruction}{taxonomy}"},
        {
            "role": "user",
            "content": f"query_id: {blind.query_id}\ntext: {blind.text}",
        },
    ]
    return ReviewRequest(reviewer=reviewer, messages=messages, images=images)


def compare_reviews(
    query: Query,
    *,
    qwen_intent: Intent | None,
    deepseek_intent: Intent | None,
    qwen_raw: str | list[str],
    deepseek_raw: str | list[str],
    qwen_reason: str | None = None,
    deepseek_reason: str | None = None,
    qwen_review_error: str | None = None,
    deepseek_review_error: str | None = None,
) -> CrossReviewResult:
    """比较构造标签和两份互不共享上下文的复核结果。"""

    qwen_raw_values = [qwen_raw] if isinstance(qwen_raw, str) else list(qwen_raw)
    deepseek_raw_values = (
        [deepseek_raw] if isinstance(deepseek_raw, str) else list(deepseek_raw)
    )
    if qwen_reason is None:
        qwen_reason = _reason_from_last_valid(qwen_raw_values)
    if deepseek_reason is None:
        deepseek_reason = _reason_from_last_valid(deepseek_raw_values)

    agreed = (
        qwen_review_error is None
        and deepseek_review_error is None
        and qwen_intent is not None
        and qwen_intent == deepseek_intent == query.canonical_intent
    )
    return CrossReviewResult(
        query_id=query.query_id,
        constructed_intent=query.canonical_intent,
        qwen_intent=qwen_intent,
        qwen_reason=qwen_reason,
        qwen_raw=qwen_raw_values,
        qwen_review_error=qwen_review_error,
        deepseek_intent=deepseek_intent,
        deepseek_reason=deepseek_reason,
        deepseek_raw=deepseek_raw_values,
        deepseek_review_error=deepseek_review_error,
        label_status="cross_agreed" if agreed else "auto",
        needs_arbitration=not agreed,
    )


def run_cross_review(
    accepted_source: str | Path,
    *,
    output_dir: str | Path,
    chat_fn: ReviewChat | None = None,
    image_root: str | Path | None = None,
    plan_path: str | Path | None = None,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
) -> Path:
    """复核 accepted 查询并原子发布 labels 派生产物。"""

    accepted_source = Path(accepted_source)
    output_dir = Path(output_dir)
    try:
        source_bytes = accepted_source.read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(f"accepted source 不存在: {accepted_source}") from None
    queries = _parse_queries(source_bytes, accepted_source)
    plan_sha256, catalog_sha256, leakage_policy_version = _review_binding(
        queries,
        plan_path=plan_path,
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
    )
    effective_chat = chat_fn or chat
    resolved_image_root = Path(image_root or config.DATA_DIR / "clean").resolve()

    reviews: list[CrossReviewResult] = []
    labeled: list[Query] = []
    for query in queries:
        if asset_catalog is not None:
            asset_catalog.verify_reference(
                query.asset_id,
                query.image_path,
                leakage_group_id=query.leakage_group_id,
            )
            asset_catalog.verify_asset_ids([query.asset_id])
        blind = BlindReviewInput(query_id=query.query_id, text=query.text)
        qwen_request = build_review_request(
            blind,
            reviewer="qwen",
            qwen_image_path=_resolve_image_transport_path(
                query.image_path,
                resolved_image_root,
            ),
        )
        deepseek_request = build_review_request(blind, reviewer="deepseek")
        qwen = _run_reviewer(qwen_request, effective_chat)
        deepseek = _run_reviewer(deepseek_request, effective_chat)
        review = compare_reviews(
            query,
            qwen_intent=qwen.intent,
            deepseek_intent=deepseek.intent,
            qwen_raw=qwen.raw,
            deepseek_raw=deepseek.raw,
            qwen_reason=qwen.reason,
            deepseek_reason=deepseek.reason,
            qwen_review_error=qwen.review_error,
            deepseek_review_error=deepseek.review_error,
        )
        reviews.append(review)
        label_provenance = list(query.label_provenance)
        if review.label_status == "cross_agreed":
            label_provenance.append(
                LabelDecision(
                    decision_type="cross_review",
                    annotator_kind="llm_pair",
                    annotator_id="qwen+deepseek",
                    canonical_intent=query.canonical_intent,
                    canonical_capability=query.canonical_capability,
                    acceptable_capabilities=query.acceptable_capabilities,
                    reason="both blinded reviewers confirmed the constructed intent",
                )
            )
        labeled.append(
            Query.model_validate(
                {
                    **query.model_dump(mode="json"),
                    "label_status": review.label_status,
                    "label_provenance": label_provenance,
                }
            )
        )

    queue, disagreement_count, review_error_count, spot_count = (
        _build_expected_arbitration_queue(queries, reviews)
    )

    review_bytes = canonical_jsonl_bytes(reviews)
    queue_bytes = canonical_jsonl_bytes(queue)
    labeled_bytes = canonical_jsonl_bytes(labeled)
    manifest = LabelsManifest(
        generated_at=datetime.now(timezone.utc),
        taxonomy_version=_single_taxonomy_version(queries),
        accepted_source=str(accepted_source.resolve()),
        accepted_source_sha256=sha256_bytes(source_bytes),
        plan_sha256=plan_sha256,
        asset_catalog_sha256=catalog_sha256,
        leakage_policy_version=leakage_policy_version,
        count=len(queries),
        review_results_sha256=sha256_bytes(review_bytes),
        arbitration_queue_sha256=sha256_bytes(queue_bytes),
        disagreement_count=disagreement_count,
        review_error_count=review_error_count,
        spot_check_count=spot_count,
    )

    staging = new_staging_directory(output_dir)
    try:
        atomic_create_file(staging / "manifest.json", canonical_json_bytes(manifest))
        atomic_create_file(staging / "reviews.jsonl", review_bytes)
        atomic_create_file(staging / "arbitration_queue.jsonl", queue_bytes)
        atomic_create_file(staging / "arbitrations.jsonl", b"")
        atomic_create_file(staging / "labeled_queries.jsonl", labeled_bytes)
        if accepted_source.read_bytes() != source_bytes:
            raise ValueError("accepted source 在审核期间发生变化，拒绝发布 labels")
        return atomic_publish_new_directory(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def pending_arbitrations(output_dir: str | Path) -> list[ArbitrationItem]:
    output_dir = Path(output_dir)
    _verify_immutable_review_files(output_dir)
    queue = _read_models(
        output_dir / "arbitration_queue.jsonl", ArbitrationItem, "arbitration queue"
    )
    decisions = _read_models(
        output_dir / "arbitrations.jsonl", ArbitrationRecord, "arbitrations"
    )
    decision_ids = [record.query_id for record in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("arbitrations 包含重复 query_id")
    queue_ids = {item.query_id for item in queue}
    unknown = sorted(set(decision_ids) - queue_ids)
    if unknown:
        raise ValueError(f"arbitrations 包含未入队 query_id: {', '.join(unknown)}")
    resolved = set(decision_ids)
    return [item for item in queue if item.query_id not in resolved]


def apply_arbitration(
    output_dir: str | Path,
    *,
    query_id: str,
    intent: Intent | str,
    canonical_capability: str | None = None,
    acceptable_capabilities: Sequence[str] | None = None,
    reason: str,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
) -> Path:
    """追加人工裁决并只更新 derived labeled_queries。"""

    output_dir = Path(output_dir)
    query_id = _nonblank(query_id, "query_id")
    reason = _nonblank(reason, "reason")
    try:
        validated_intent = _INTENT_ADAPTER.validate_python(intent)
    except ValidationError as exc:
        raise ValueError(f"仲裁 intent 非法: {intent}") from exc
    manifest = _verify_immutable_review_files(output_dir)
    _require_labels_catalog_binding(
        manifest,
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
    )
    pending = {item.query_id: item for item in pending_arbitrations(output_dir)}
    if query_id not in pending:
        raise ValueError(f"query_id 不在待仲裁队列: {query_id}")
    pending_item = pending[query_id]

    labeled_path = output_dir / "labeled_queries.jsonl"
    arbitrations_path = output_dir / "arbitrations.jsonl"
    labeled_before = labeled_path.read_bytes()
    arbitrations_before = arbitrations_path.read_bytes()
    queries = _parse_queries(labeled_before, labeled_path)
    matches = [
        index for index, query in enumerate(queries) if query.query_id == query_id
    ]
    if len(matches) != 1:
        raise ValueError(f"labeled_queries 无法唯一定位 query_id: {query_id}")
    index = matches[0]
    existing = queries[index]
    if (
        existing.image_path != pending_item.image_path
        or existing.text != pending_item.text
    ):
        raise ValueError("待仲裁查询与冻结 arbitration queue 不一致")
    if asset_catalog is not None:
        asset_catalog.verify_reference(
            existing.asset_id,
            existing.image_path,
            leakage_group_id=existing.leakage_group_id,
        )
        asset_catalog.verify_asset_ids([existing.asset_id])
    (
        canonical_capability,
        resolved_acceptable_capabilities,
        requires_card,
    ) = _resolve_arbitration_capability(
        existing,
        intent=validated_intent,
        canonical_capability=canonical_capability,
        acceptable_capabilities=acceptable_capabilities,
    )
    decided_at = datetime.now(timezone.utc)
    decision = LabelDecision(
        decision_type="arbitration",
        annotator_kind="human",
        annotator_id="interactive-user",
        canonical_intent=validated_intent,
        canonical_capability=canonical_capability,
        acceptable_capabilities=resolved_acceptable_capabilities,
        reason=reason,
        decided_at=decided_at,
    )
    queries[index] = Query.model_validate(
        {
            **existing.model_dump(mode="json"),
            "canonical_intent": validated_intent,
            "canonical_capability": canonical_capability,
            "acceptable_capabilities": resolved_acceptable_capabilities,
            "requires_card": requires_card,
            "label_status": "arbitrated",
            "label_provenance": [*existing.label_provenance, decision],
        }
    )
    decision = ArbitrationRecord(
        query_id=query_id,
        intent=validated_intent,
        canonical_capability=canonical_capability,
        acceptable_capabilities=resolved_acceptable_capabilities,
        requires_card=requires_card,
        reason=reason,
        decided_at=decided_at,
    )
    labeled_after = canonical_jsonl_bytes(queries)
    if arbitrations_before and not arbitrations_before.endswith(b"\n"):
        raise ValueError("arbitrations.jsonl 末尾缺少换行，拒绝破坏旧记录")
    arbitrations_after = arbitrations_before + canonical_json_bytes(decision)

    atomic_replace_file(labeled_path, labeled_after)
    try:
        atomic_replace_file(arbitrations_path, arbitrations_after)
    except BaseException:
        atomic_replace_file(labeled_path, labeled_before)
        atomic_replace_file(arbitrations_path, arbitrations_before)
        raise
    return labeled_path


def _run_reviewer(request: ReviewRequest, chat_fn: ReviewChat) -> ReviewerOutcome:
    raw_responses: list[str] = []
    last_error: str | None = None
    for _ in range(_MAX_PARSE_RETRIES + 1):
        if request.reviewer == "qwen":
            response = chat_fn(
                config.LABEL_VISION_SYNTH_PROVIDER,
                request.messages,
                model=config.LABEL_VISION_SYNTH_MODEL,
                images=request.images,
                temperature=0.0,
                json_mode=True,
            )
        else:
            response = chat_fn(
                config.LABEL_TEXT_REVIEW_PROVIDER,
                request.messages,
                model=config.LABEL_TEXT_REVIEW_MODEL,
                temperature=0.0,
                json_mode=True,
                thinking=False,
            )
        raw = response.text if isinstance(response, LLMResponse) else response
        raw_responses.append(raw)
        try:
            decision = ReviewDecision.model_validate_json(raw)
            return ReviewerOutcome(
                intent=decision.intent,
                reason=decision.reason,
                raw=raw_responses,
            )
        except (ValidationError, ValueError, TypeError) as exc:
            last_error = str(exc)
    return ReviewerOutcome(
        intent=None,
        reason=None,
        raw=raw_responses,
        review_error=(
            f"{request.reviewer} 严格 JSON 解析失败，"
            f"共 {len(raw_responses)} 次响应: {last_error}"
        ),
    )


def _reason_from_last_valid(raw_values: list[str]) -> str | None:
    for raw in reversed(raw_values):
        try:
            return ReviewDecision.model_validate_json(raw).reason
        except (ValidationError, ValueError, TypeError):
            continue
    return None


def _resolve_image_transport_path(image_path: str, image_root: Path) -> str:
    candidate = Path(image_path)
    if not candidate.is_absolute():
        candidate = image_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(image_root)
    except ValueError:
        raise ValueError(f"image_path 越界，不在 image_root 内: {image_path}") from None
    return str(resolved)


def _to_arbitration_item(
    query: Query,
    review: CrossReviewResult,
    *,
    spot_check: bool,
) -> ArbitrationItem:
    return ArbitrationItem(
        query_id=query.query_id,
        image_path=query.image_path,
        text=query.text,
        constructed_intent=query.canonical_intent,
        constructed_capability=query.canonical_capability,
        constructed_acceptable_capabilities=query.acceptable_capabilities,
        constructed_requires_card=query.requires_card,
        qwen_intent=review.qwen_intent,
        qwen_reason=review.qwen_reason,
        qwen_review_error=review.qwen_review_error,
        deepseek_intent=review.deepseek_intent,
        deepseek_reason=review.deepseek_reason,
        deepseek_review_error=review.deepseek_review_error,
        spot_check=spot_check,
    )


def _resolve_arbitration_capability(
    existing: Query,
    *,
    intent: Intent,
    canonical_capability: str | None,
    acceptable_capabilities: Sequence[str] | None,
) -> tuple[str, list[str], bool]:
    """Resolve an arbitration label without an intent-only primary fallback."""

    taxonomy = load_default_taxonomy_registry()
    task_specification = load_mvp_task_specification_v1()
    available = capabilities_for_intent(intent)
    if canonical_capability is None:
        if len(available) != 1:
            choices = ", ".join(item.capability_id for item in available)
            raise ValueError(
                f"intent {intent} has multiple capabilities; "
                f"canonical_capability is required ({choices})"
            )
        canonical_capability = available[0].capability_id
    canonical_capability = _nonblank(canonical_capability, "canonical_capability")

    if acceptable_capabilities is None:
        if (
            existing.canonical_intent == intent
            and existing.canonical_capability == canonical_capability
            and existing.acceptable_capabilities
        ):
            resolved_acceptable = list(existing.acceptable_capabilities)
        else:
            resolved_acceptable = [canonical_capability]
    else:
        resolved_acceptable = [
            _nonblank(value, "acceptable_capabilities")
            for value in acceptable_capabilities
        ]
        if resolved_acceptable != sorted(resolved_acceptable):
            raise ValueError("acceptable_capabilities must be sorted")

    definition = taxonomy.capabilities_by_id.get(canonical_capability)
    if definition is None:
        raise ValueError(f"unknown canonical_capability: {canonical_capability}")
    requires_card = definition.requires_card
    validate_capability_binding(
        taxonomy_version=existing.taxonomy_version,
        task_spec_version=existing.task_spec_version,
        canonical_intent=intent,
        canonical_capability=canonical_capability,
        acceptable_capabilities=resolved_acceptable,
        requires_card=requires_card,
        taxonomy=taxonomy,
        task_specification=task_specification,
    )
    return canonical_capability, resolved_acceptable, requires_card


def _build_expected_arbitration_queue(
    queries: Sequence[Query],
    reviews: Sequence[CrossReviewResult],
) -> tuple[list[ArbitrationItem], int, int, int]:
    query_ids = [query.query_id for query in queries]
    review_ids = [review.query_id for review in reviews]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("review source 包含重复 query_id")
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("reviews 包含重复 query_id")
    if review_ids != query_ids:
        raise ValueError("reviews 必须与 review source 逐条同序对应")

    query_by_id = {query.query_id: query for query in queries}
    for review in reviews:
        query = query_by_id[review.query_id]
        agreed = (
            review.qwen_review_error is None
            and review.deepseek_review_error is None
            and review.qwen_intent is not None
            and review.qwen_intent
            == review.deepseek_intent
            == review.constructed_intent
            == query.canonical_intent
        )
        expected_status = "cross_agreed" if agreed else "auto"
        if (
            review.constructed_intent != query.canonical_intent
            or review.label_status != expected_status
            or review.needs_arbitration == agreed
        ):
            raise ValueError(
                f"review 状态无法由 source 与 reviewer 结果重建: {review.query_id}"
            )

    agreed_reviews = [review for review in reviews if not review.needs_arbitration]
    spot_count = math.ceil(len(agreed_reviews) * 0.05) if agreed_reviews else 0
    spot_ids = {
        review.query_id
        for review in sorted(
            agreed_reviews,
            key=lambda item: (
                hashlib.sha256(item.query_id.encode("utf-8")).hexdigest(),
                item.query_id,
            ),
        )[:spot_count]
    }
    queue = [
        _to_arbitration_item(
            query_by_id[review.query_id],
            review,
            spot_check=review.query_id in spot_ids,
        )
        for review in reviews
        if review.needs_arbitration or review.query_id in spot_ids
    ]
    disagreement_count = sum(review.needs_arbitration for review in reviews)
    review_error_count = sum(
        bool(review.qwen_review_error or review.deepseek_review_error)
        for review in reviews
    )
    return queue, disagreement_count, review_error_count, spot_count


def validate_review_queue_semantics(
    queries: Sequence[Query],
    reviews: Sequence[CrossReviewResult],
    queue: Sequence[ArbitrationItem],
    manifest: LabelsManifest,
) -> None:
    """Rebuild the mandatory arbitration queue and reject semantic drift."""

    expected, disagreement_count, review_error_count, spot_count = (
        _build_expected_arbitration_queue(queries, reviews)
    )
    if list(queue) != expected:
        raise ValueError("标签仲裁队列无法由 reviews 与 accepted source 确定性重建")
    expected_counts = (disagreement_count, review_error_count, spot_count)
    manifest_counts = (
        manifest.disagreement_count,
        manifest.review_error_count,
        manifest.spot_check_count,
    )
    if manifest_counts != expected_counts:
        raise ValueError("labels manifest 的审核/仲裁计数与 reviews 不一致")
    if manifest.count != len(queries):
        raise ValueError("labels manifest count 与 accepted source 不一致")
    if manifest.taxonomy_version != _single_taxonomy_version(list(queries)):
        raise ValueError("labels manifest taxonomy_version 与 accepted source 不一致")


def _parse_queries(content: bytes, source: Path) -> list[Query]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source} 不是 UTF-8: {exc}") from exc
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"{source} 不得为空")
    queries: list[Query] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{source} 第 {line_number} 行为空")
        try:
            queries.append(Query.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{source} 第 {line_number} 行校验失败: {exc}") from exc
    ids = [query.query_id for query in queries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source} 包含重复 query_id")
    return queries


def _single_taxonomy_version(queries: list[Query]) -> str:
    versions = {query.taxonomy_version for query in queries}
    if len(versions) != 1:
        raise ValueError("accepted queries 必须使用同一个 taxonomy_version")
    return next(iter(versions))


def _read_models(path: Path, model_type, label: str) -> list:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} 不存在: {path}") from None
    values = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{label} 第 {line_number} 行为空")
        try:
            values.append(model_type.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{label} 第 {line_number} 行校验失败: {exc}") from exc
    return values


def _verify_immutable_review_files(output_dir: Path) -> LabelsManifest:
    try:
        manifest_bytes = (output_dir / "manifest.json").read_bytes()
        manifest = LabelsManifest.model_validate_json(manifest_bytes)
        review_bytes = (output_dir / "reviews.jsonl").read_bytes()
        queue_bytes = (output_dir / "arbitration_queue.jsonl").read_bytes()
        accepted_source = Path(manifest.accepted_source)
        accepted_bytes = accepted_source.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"labels 产物不完整: {exc.filename}") from None
    except ValidationError as exc:
        raise ValueError(f"labels manifest 校验失败: {exc}") from exc
    if manifest_bytes != canonical_json_bytes(manifest):
        raise ValueError("labels manifest 不是 canonical JSON")
    if sha256_bytes(review_bytes) != manifest.review_results_sha256:
        raise ValueError("reviews.jsonl 哈希与 labels manifest 不一致")
    if sha256_bytes(queue_bytes) != manifest.arbitration_queue_sha256:
        raise ValueError("arbitration_queue.jsonl 哈希与 labels manifest 不一致")
    if sha256_bytes(accepted_bytes) != manifest.accepted_source_sha256:
        raise ValueError("accepted source 哈希与 labels manifest 不一致")
    queries = _parse_queries(accepted_bytes, accepted_source)
    reviews = _parse_models_bytes(review_bytes, CrossReviewResult, "reviews.jsonl")
    queue = _parse_models_bytes(queue_bytes, ArbitrationItem, "arbitration_queue.jsonl")
    if review_bytes != canonical_jsonl_bytes(reviews):
        raise ValueError("reviews.jsonl 不是 canonical JSONL")
    if queue_bytes != canonical_jsonl_bytes(queue):
        raise ValueError("arbitration_queue.jsonl 不是 canonical JSONL")
    validate_review_queue_semantics(queries, reviews, queue, manifest)
    return manifest


def _parse_models_bytes(content: bytes, model_type, label: str) -> list:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} 不是 UTF-8: {exc}") from exc
    if not text:
        return []
    parsed = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{label} 第 {line_number} 行为空")
        try:
            parsed.append(model_type.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{label} 第 {line_number} 行校验失败: {exc}") from exc
    return parsed


def _review_binding(
    queries: list[Query],
    *,
    plan_path: str | Path | None,
    asset_catalog: AssetCatalog | None,
    allow_provisional_asset_groups: bool,
) -> tuple[str | None, str | None, str]:
    if plan_path is None:
        if asset_catalog is not None or not allow_provisional_asset_groups:
            raise ValueError(
                "正式 review 必须提供 full plan 与 asset catalog；"
                "测试需显式 allow_provisional_asset_groups=True"
            )
        return None, None, PROVISIONAL_LEAKAGE_POLICY_VERSION

    from skillchain.synthesis.planning import audit_plan_catalog_binding, load_plan

    plan, manifest = load_plan(plan_path)
    if plan.scope != "full":
        raise ValueError("正式 review 必须绑定 full plan")
    if plan.asset_catalog_sha256 is None:
        if asset_catalog is not None or not allow_provisional_asset_groups:
            raise ValueError("provisional full plan review 必须显式允许")
        catalog_sha256 = None
    else:
        if asset_catalog is None:
            raise ValueError("catalog-bound review 缺少 asset catalog")
        asset_catalog.require_verified_files()
        audit_plan_catalog_binding(plan, asset_catalog)
        catalog_sha256 = asset_catalog.catalog_sha256
    _verify_queries_against_plan(queries, plan)
    return manifest.plan_sha256, catalog_sha256, plan.leakage_policy_version


def _verify_queries_against_plan(queries: list[Query], plan) -> None:
    by_id = {query.query_id: query for query in queries}
    plan_by_id = {item.plan_id: item for item in plan.queries}
    if len(by_id) != len(queries) or set(by_id) != set(plan_by_id):
        raise ValueError("review queries 与 full plan 的 query_id 不完整")
    for query_id, query in by_id.items():
        item = plan_by_id[query_id]
        static_pairs = (
            (query.taxonomy_version, item.taxonomy_version),
            (query.task_spec_version, item.task_spec_version),
            (query.asset_id, item.asset_id),
            (query.image_path, item.image_path),
            (query.leakage_group_id, item.leakage_group_id),
            (query.boundary_group_id, item.boundary_group_id),
            (query.template_family, item.template_family),
            (query.generator_batch_id, item.generator_batch_id),
            (query.canonical_intent, item.canonical_intent),
            (query.is_boundary, item.is_boundary),
            (query.boundary_strategy, item.boundary_strategy),
        )
        if any(actual != expected for actual, expected in static_pairs):
            raise ValueError(f"review query 与 full plan 静态字段不一致: {query_id}")


def _require_labels_catalog_binding(
    manifest: LabelsManifest,
    *,
    asset_catalog: AssetCatalog | None,
    allow_provisional_asset_groups: bool,
) -> None:
    if manifest.asset_catalog_sha256 is None:
        if asset_catalog is not None or not allow_provisional_asset_groups:
            raise ValueError("provisional labels 仲裁必须显式允许")
        return
    if asset_catalog is None:
        raise ValueError("正式 labels 仲裁缺少 asset catalog")
    asset_catalog.require_verified_files()
    if (
        asset_catalog.catalog_sha256 != manifest.asset_catalog_sha256
        or asset_catalog.leakage_policy_version != manifest.leakage_policy_version
    ):
        raise ValueError("labels manifest 与 asset catalog 绑定不一致")
