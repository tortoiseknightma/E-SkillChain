"""Public-safe scorer evidence captured from validated Portfolio tool results.

This module intentionally does not import the Assistant runner or receipt types.
The runner may therefore capture a typed call immediately after registry
validation, while the raw DTO is still available, and bind the completed calls
to the Assistant result and receipt only after those artifacts exist.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.evaluation.packets import AssistantRunConfig
from skillchain.schemas import Query
from skillchain.tools.contracts import (
    JSONValue,
    ProductSearchTrace,
    StyleHit,
    validate_json_value,
)
from skillchain.tools.document_ocr import DocumentOCRResult
from skillchain.tools.kb_lookup import KBHit
from skillchain.tools.multi_product import MultiProductResult
from skillchain.tools.object_detect import ObjectDetectionResult
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
StyleSubmode = Literal["same_category_alternative", "cross_category_coordination"]
StyleSupportStatus = Literal["candidates", "no_result", "unsupported"]
ScorerPayloadKindV2 = Literal[
    "product_candidates_v1",
    "multi_mapping_v1",
    "knowledge_sources_v1",
    "ocr_lines_v1",
    "detections_v1",
    "style_candidates_v2",
]

GCS_SCORER_EVIDENCE_V2_POLICY_VERSION = "portfolio-gcs-scorer-evidence-v2"
GCS_SCORER_EVIDENCE_V2_SCHEMA_VERSION = 2

_PRODUCT_EVIDENCE_RE = re.compile(r"^tool-call-(\d+)-evidence-(\d+)$")
_PRODUCT_ID_RE = re.compile(r"^tool-call-(\d+)-product-(\d+)$")
_SOURCE_RE = re.compile(r"^tool-call-(\d+)-source-(\d+)$")
_LINE_RE = re.compile(r"^tool-call-(\d+)-line-(\d+)$")
_STYLE_EVIDENCE_RE = re.compile(r"^tool-call-(\d+)-style-evidence-(\d+)-(\d+)$")
_ITEM_REF_RE = re.compile(r"^item-(\d{3})$")
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "asset_id",
        "image_path",
        "local_path",
        "source_dataset",
        "source_record_id",
        "registry_sha256",
        "registry_runtime_sha256",
        "runtime_binding_sha256",
        "query_asset_id",
        "query_image_sha256",
        "query_vector_sha256",
        "artifact_binding",
    }
)
_ASSET_TOOLS = frozenset(
    {
        "image_product_search",
        "object_detect",
        "document_ocr",
        "multi_product_search",
    }
)
_SAME_STYLE_SOURCES = frozenset(
    {
        "fashioniq_relative_caption_graph",
        "local_image_feature_cosine",
        "verified_embedding_cosine",
    }
)
_STYLE_PROVENANCE = frozenset(
    {
        "fashioniq_relative_caption",
        "local_image_feature",
        "verified_embedding",
        "portfolio_curated_coordination_rule",
    }
)


class PublicScorerEvidenceIntegrityError(ValueError):
    """The v2 scorer evidence cannot safely enter a formal denominator."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _nonblank(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-blank and trimmed")
    return value


def _jsonable(value: object) -> JSONValue:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON-serializable: {type(value)!r}")


def _hash_json(value: object) -> str:
    payload = _jsonable(value)
    validate_json_value(payload)
    return sha256_bytes(canonical_json_bytes(payload))


def _self_hash(model: BaseModel, field_name: str) -> str:
    return _hash_json(model.model_dump(mode="json", exclude={field_name}))


def _strict_revalidate_dump(value: object, model_type: type[BaseModel]) -> BaseModel:
    if type(value) is model_type:
        return value
    return model_type.model_validate_json(
        canonical_json_bytes(_jsonable(value)), strict=True
    )


def _coerce_tuple(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


def _reject_private_values(
    value: JSONValue, *, path: str, reject_identity_strings: bool = True
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"{path} contains forbidden private key {key}")
            _reject_private_values(
                item,
                path=f"{path}.{key}",
                reject_identity_strings=reject_identity_strings,
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_private_values(
                item,
                path=f"{path}[{index}]",
                reject_identity_strings=reject_identity_strings,
            )
        return
    if reject_identity_strings and isinstance(value, str):
        lowered = value.casefold()
        if (
            lowered.startswith("asset.v")
            or re.match(r"^[a-z]:[\\/]", value, re.IGNORECASE)
            or lowered.startswith("query_images/")
            or "../" in lowered
        ):
            raise ValueError(f"{path} contains a private identity or local path")


class ExpectedMultiItemV2(_StrictFrozenModel):
    item_ref: str
    label: str

    @field_validator("item_ref", "label")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        value = _nonblank(value, info.field_name)
        if info.field_name == "item_ref" and _ITEM_REF_RE.fullmatch(value) is None:
            raise ValueError("expected multi item_ref is invalid")
        return value


class ScorerProductCandidateV2(_StrictFrozenModel):
    candidate_ordinal: int = Field(ge=1, le=12)
    evidence_reference: str
    product_id: str
    title: str
    eligible: bool
    public_attributes: tuple[tuple[str, str], ...] = ()

    @field_validator("public_attributes", mode="before")
    @classmethod
    def coerce_attributes(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(tuple(item) for item in value)
        return value

    @field_validator("evidence_reference", "product_id", "title")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        evidence = _PRODUCT_EVIDENCE_RE.fullmatch(self.evidence_reference)
        product = _PRODUCT_ID_RE.fullmatch(self.product_id)
        if evidence is None or product is None or evidence.groups() != product.groups():
            raise ValueError("product candidate handles are inconsistent")
        if int(evidence.group(2)) != self.candidate_ordinal:
            raise ValueError("product candidate ordinal differs from handles")
        if self.public_attributes != tuple(sorted(self.public_attributes)) or len(
            {key for key, _value in self.public_attributes}
        ) != len(self.public_attributes):
            raise ValueError("public product attributes must be sorted and unique")
        for key, value in self.public_attributes:
            _nonblank(key, "public attribute key")
            _nonblank(value, "public attribute value")
            if key in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError("public product attributes contain a private key")
        return self


class ScorerProductPayloadV2(_StrictFrozenModel):
    candidates: tuple[ScorerProductCandidateV2, ...] = ()

    @field_validator("candidates", mode="before")
    @classmethod
    def coerce_candidates(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_candidates(self) -> Self:
        if tuple(item.candidate_ordinal for item in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("product candidate ordinals must be contiguous")
        return self


class ScorerMultiItemV2(_StrictFrozenModel):
    item_ref: str
    label: str
    status: Literal["matched", "unresolved"]
    candidate_ordinal: int | None = Field(default=None, ge=1, le=12)

    @field_validator("item_ref", "label")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        value = _nonblank(value, info.field_name)
        if info.field_name == "item_ref" and _ITEM_REF_RE.fullmatch(value) is None:
            raise ValueError("multi item_ref is invalid")
        return value

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if (self.status == "matched") != (self.candidate_ordinal is not None):
            raise ValueError("multi item status and candidate must be paired")
        return self


class ScorerMultiPayloadV2(_StrictFrozenModel):
    items: tuple[ScorerMultiItemV2, ...]
    candidates: tuple[ScorerProductCandidateV2, ...] = ()

    @field_validator("items", "candidates", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_mapping(self) -> Self:
        if tuple(item.item_ref for item in self.items) != tuple(
            f"item-{index:03d}" for index in range(1, len(self.items) + 1)
        ):
            raise ValueError("multi item refs must be contiguous")
        if tuple(item.candidate_ordinal for item in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("multi candidate ordinals must be contiguous")
        known = {item.candidate_ordinal for item in self.candidates}
        referenced = {
            item.candidate_ordinal
            for item in self.items
            if item.candidate_ordinal is not None
        }
        if any(not item.eligible for item in self.candidates) or known != referenced:
            raise ValueError("multi candidates must exactly cover matched items")
        return self


class ScorerKnowledgeSourceV2(_StrictFrozenModel):
    evidence_reference: str
    title: str
    text: str
    uri: str | None = None

    @field_validator("evidence_reference", "title", "text")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("evidence_reference")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        if _SOURCE_RE.fullmatch(value) is None:
            raise ValueError("knowledge source handle is invalid")
        return value

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = _nonblank(value, "uri")
        if not value.startswith("https://") or "@" in value.split("/", 3)[2]:
            raise ValueError("knowledge URI must be credential-free HTTPS")
        return value


class ScorerKnowledgePayloadV2(_StrictFrozenModel):
    sources: tuple[ScorerKnowledgeSourceV2, ...] = ()

    @field_validator("sources", mode="before")
    @classmethod
    def coerce_sources(cls, value: object) -> object:
        return _coerce_tuple(value)


class ScorerOCRLineV2(_StrictFrozenModel):
    line_reference: str
    text: str
    content_trust: Literal["untrusted_document_text"]
    truncated: bool = False
    fields: tuple[tuple[str, str], ...] = ()

    @field_validator("fields", mode="before")
    @classmethod
    def coerce_fields(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(tuple(item) for item in value)
        return value

    @field_validator("line_reference", "text")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_line(self) -> Self:
        if _LINE_RE.fullmatch(self.line_reference) is None:
            raise ValueError("OCR line handle is invalid")
        if self.fields != tuple(sorted(self.fields)) or len(
            {key for key, _value in self.fields}
        ) != len(self.fields):
            raise ValueError("OCR fields must be sorted and unique")
        if any(key in _FORBIDDEN_PUBLIC_KEYS for key, _value in self.fields):
            raise ValueError("OCR fields contain a private key")
        return self


class ScorerOCRPayloadV2(_StrictFrozenModel):
    lines: tuple[ScorerOCRLineV2, ...] = ()

    @field_validator("lines", mode="before")
    @classmethod
    def coerce_lines(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_lines(self) -> Self:
        if len({item.line_reference for item in self.lines}) != len(self.lines):
            raise ValueError("OCR line handles must be unique")
        return self


class ScorerDetectionV2(_StrictFrozenModel):
    item_ref: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("item_ref", "label")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        value = _nonblank(value, info.field_name)
        if info.field_name == "item_ref" and _ITEM_REF_RE.fullmatch(value) is None:
            raise ValueError("detection item_ref is invalid")
        return value


class ScorerDetectionPayloadV2(_StrictFrozenModel):
    detections: tuple[ScorerDetectionV2, ...] = ()

    @field_validator("detections", mode="before")
    @classmethod
    def coerce_detections(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_detections(self) -> Self:
        if tuple(item.item_ref for item in self.detections) != tuple(
            f"item-{index:03d}" for index in range(1, len(self.detections) + 1)
        ):
            raise ValueError("detection item refs must be contiguous")
        return self


class ScorerStyleFacetV2(_StrictFrozenModel):
    evidence_reference: str
    facet: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: Literal[
        "fashioniq_relative_caption",
        "local_image_feature",
        "verified_embedding",
        "portfolio_curated_coordination_rule",
    ]

    @field_validator("evidence_reference", "facet", "value")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("evidence_reference")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        if _STYLE_EVIDENCE_RE.fullmatch(value) is None:
            raise ValueError("Style evidence handle is invalid")
        return value


class ScorerStyleCandidateV2(_StrictFrozenModel):
    candidate_ordinal: int = Field(ge=1, le=12)
    evidence_reference: str
    product_id: str
    title: str
    category: str
    style_submode: StyleSubmode
    similarity_source: (
        Literal[
            "fashioniq_relative_caption_graph",
            "local_image_feature_cosine",
            "verified_embedding_cosine",
        ]
        | None
    ) = None
    style_evidence: tuple[ScorerStyleFacetV2, ...]

    @field_validator("style_evidence", mode="before")
    @classmethod
    def coerce_evidence(cls, value: object) -> object:
        return _coerce_tuple(value)

    @field_validator("evidence_reference", "product_id", "title", "category")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        evidence = _PRODUCT_EVIDENCE_RE.fullmatch(self.evidence_reference)
        product = _PRODUCT_ID_RE.fullmatch(self.product_id)
        if evidence is None or product is None or evidence.groups() != product.groups():
            raise ValueError("Style product handles are inconsistent")
        call_index, ordinal = (int(item) for item in evidence.groups())
        if ordinal != self.candidate_ordinal:
            raise ValueError("Style candidate ordinal differs from handles")
        if not self.style_evidence:
            raise ValueError("Style candidates require public facet evidence")
        expected = tuple(range(1, len(self.style_evidence) + 1))
        observed: list[int] = []
        for item in self.style_evidence:
            match = _STYLE_EVIDENCE_RE.fullmatch(item.evidence_reference)
            assert match is not None
            if (int(match.group(1)), int(match.group(2))) != (
                call_index,
                ordinal,
            ):
                raise ValueError("Style facet handle differs from its candidate")
            observed.append(int(match.group(3)))
        if tuple(observed) != expected:
            raise ValueError("Style facet handles must be contiguous")
        if self.style_submode == "cross_category_coordination":
            if self.similarity_source is not None or any(
                item.provenance != "portfolio_curated_coordination_rule"
                for item in self.style_evidence
            ):
                raise ValueError("cross-category Style evidence must be curated")
        elif self.similarity_source not in _SAME_STYLE_SOURCES or any(
            item.provenance == "portfolio_curated_coordination_rule"
            for item in self.style_evidence
        ):
            raise ValueError("same-category Style evidence source is invalid")
        return self


class ScorerStylePayloadV2(_StrictFrozenModel):
    support_status: StyleSupportStatus
    style_submode: StyleSubmode
    anchor_category: str | None = None
    requested_target_families: tuple[str, ...] = ()
    allowed_target_families: tuple[str, ...] = ()
    candidates: tuple[ScorerStyleCandidateV2, ...] = ()

    @field_validator(
        "requested_target_families",
        "allowed_target_families",
        "candidates",
        mode="before",
    )
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _coerce_tuple(value)

    @field_validator("anchor_category")
    @classmethod
    def validate_anchor(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value, "anchor_category")

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        for label, values in (
            ("requested_target_families", self.requested_target_families),
            ("allowed_target_families", self.allowed_target_families),
        ):
            if len(values) != len(set(values)) or any(
                not item or item != item.strip() for item in values
            ):
                raise ValueError(f"{label} must be unique, ordered, trimmed text")
        if (self.support_status == "candidates") != bool(self.candidates):
            raise ValueError("Style support status and candidates disagree")
        if any(item.style_submode != self.style_submode for item in self.candidates):
            raise ValueError("Style payload and candidate submodes disagree")
        if tuple(item.candidate_ordinal for item in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("Style candidate ordinals must be contiguous")
        return self


_PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    "product_candidates_v1": ScorerProductPayloadV2,
    "multi_mapping_v1": ScorerMultiPayloadV2,
    "knowledge_sources_v1": ScorerKnowledgePayloadV2,
    "ocr_lines_v1": ScorerOCRPayloadV2,
    "detections_v1": ScorerDetectionPayloadV2,
    "style_candidates_v2": ScorerStylePayloadV2,
}


class PublicScorerCallEvidenceV2(_StrictFrozenModel):
    call_index: int = Field(ge=1)
    tool_name: str
    arguments_sha256: Sha256
    result_sha256: Sha256
    argument_projection: dict[str, JSONValue]
    payload_kind: ScorerPayloadKindV2
    payload: dict[str, JSONValue]
    payload_sha256: Sha256

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        return _nonblank(value, "tool_name")

    @model_validator(mode="after")
    def validate_call(self) -> Self:
        validate_json_value(self.argument_projection)
        validate_json_value(self.payload)
        _reject_private_values(
            self.argument_projection,
            path="arguments",
            reject_identity_strings=False,
        )
        _reject_private_values(self.payload, path="payload")
        keys = set(self.argument_projection)
        if keys not in (
            {"asset_handle"},
            {"asset_handle", "query"},
            {"query"},
            {"entity"},
            {"dish"},
        ):
            raise ValueError("scorer argument projection is not public-safe")
        if self.argument_projection.get("asset_handle", "query_asset") != (
            "query_asset"
        ) or any(
            not isinstance(item, str) or not item.strip()
            for key, item in self.argument_projection.items()
            if key != "asset_handle"
        ):
            raise ValueError("scorer argument projection is invalid")
        _PAYLOAD_MODELS[self.payload_kind].model_validate(self.payload, strict=True)
        if self.payload_sha256 != _hash_json(self.payload):
            raise ValueError("scorer payload hash mismatch")
        return self


class PublicScorerEvidenceV2(_StrictFrozenModel):
    schema_version: Literal[2] = GCS_SCORER_EVIDENCE_V2_SCHEMA_VERSION
    policy_version: Literal["portfolio-gcs-scorer-evidence-v2"] = (
        GCS_SCORER_EVIDENCE_V2_POLICY_VERSION
    )
    matrix_run_id: str
    instance_id: str
    request_sha256: Sha256
    query_id: str
    config: AssistantRunConfig
    query_artifact_sha256: Sha256
    assistant_result_sha256: Sha256
    assistant_receipt_sha256: Sha256
    expected_multi_items: tuple[ExpectedMultiItemV2, ...] | None = None
    calls: tuple[PublicScorerCallEvidenceV2, ...]
    evidence_sha256: Sha256

    @field_validator("expected_multi_items", "calls", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return None if value is None else _coerce_tuple(value)

    @field_validator("matrix_run_id", "instance_id", "query_id")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        indices = tuple(item.call_index for item in self.calls)
        if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ValueError("scorer call indices must be sorted and unique")
        if self.expected_multi_items is not None:
            refs = tuple(item.item_ref for item in self.expected_multi_items)
            if not refs or refs != tuple(
                f"item-{index:03d}" for index in range(1, len(refs) + 1)
            ):
                raise ValueError(
                    "expected multi item refs must be non-empty and contiguous"
                )
        if self.evidence_sha256 != _self_hash(self, "evidence_sha256"):
            raise ValueError("public scorer evidence self hash mismatch")
        return self


class StyleQueryClassificationV2(_StrictFrozenModel):
    style_submode: StyleSubmode
    requested_target_families: tuple[str, ...] = ()
    allowed_target_families: tuple[str, ...] = ()


def classify_style_query_v2(query_text: str) -> StyleQueryClassificationV2:
    """Use the runtime's frozen matcher as the sole Style mode/family classifier."""

    query_text = _nonblank(query_text, "query_text")
    # Lazy import avoids pulling the Portfolio runtime into non-Style evidence paths.
    from skillchain.tools.portfolio_runtime import (
        _is_style_coordination_query,
        _requested_style_coordination_categories,
        _style_coordination_allowed_categories,
    )

    if not _is_style_coordination_query(query_text):
        return StyleQueryClassificationV2(style_submode="same_category_alternative")
    return StyleQueryClassificationV2(
        style_submode="cross_category_coordination",
        requested_target_families=(
            _requested_style_coordination_categories(query_text)
        ),
        allowed_target_families=_style_coordination_allowed_categories(query_text),
    )


def _public_product_title(product: object, *, candidate_ordinal: int) -> str:
    title = _nonblank(str(getattr(product, "title")), "product title")
    product_id = str(getattr(product, "product_id"))
    source = str(getattr(product, "source"))
    generated_prefixes = (
        f"{source.strip().upper()} item ",
        f"{source.strip().upper()} catalog class ",
    )
    if any(title.casefold().startswith(item.casefold()) for item in generated_prefixes):
        return f"商品候选 {candidate_ordinal}"
    fragments = (product_id, product_id.rsplit(":", 1)[-1])
    if any(
        len(fragment) >= 6 and fragment.casefold() in title.casefold()
        for fragment in fragments
    ):
        return f"商品候选 {candidate_ordinal}"
    return title


def _product_candidate(
    hit: object, *, call_index: int, candidate_ordinal: int
) -> dict[str, JSONValue]:
    product = getattr(hit, "product")
    category = _nonblank(str(getattr(product, "category_l1")), "category")
    return {
        "candidate_ordinal": candidate_ordinal,
        "evidence_reference": (f"tool-call-{call_index}-evidence-{candidate_ordinal}"),
        "product_id": f"tool-call-{call_index}-product-{candidate_ordinal}",
        "title": _public_product_title(product, candidate_ordinal=candidate_ordinal),
        "eligible": True,
        "public_attributes": [["category", category]],
    }


def _style_payload(
    query: Query, trace: ProductSearchTrace, *, call_index: int
) -> dict[str, JSONValue]:
    classification = classify_style_query_v2(query.text)
    candidates: list[JSONValue] = []
    anchors: list[str] = []
    for candidate_ordinal, hit in enumerate(trace.hits[:5], 1):
        if not isinstance(hit, StyleHit):
            raise PublicScorerEvidenceIntegrityError(
                "Style trace contains a non-Style hit"
            )
        anchors.append(hit.anchor_category_l1)
        facets: list[JSONValue] = []
        for evidence_ordinal, evidence in enumerate(hit.facet_evidence[:4], 1):
            facets.append(
                {
                    "evidence_reference": (
                        f"tool-call-{call_index}-style-evidence-"
                        f"{candidate_ordinal}-{evidence_ordinal}"
                    ),
                    "facet": evidence.facet,
                    "value": evidence.value,
                    "confidence": float(evidence.confidence),
                    "provenance": evidence.provenance,
                }
            )
        product = hit.product
        candidates.append(
            {
                "candidate_ordinal": candidate_ordinal,
                "evidence_reference": (
                    f"tool-call-{call_index}-evidence-{candidate_ordinal}"
                ),
                "product_id": f"tool-call-{call_index}-product-{candidate_ordinal}",
                "title": _public_product_title(
                    product, candidate_ordinal=candidate_ordinal
                ),
                "category": product.category_l1,
                "style_submode": hit.style_submode,
                "similarity_source": (
                    None
                    if hit.style_submode == "cross_category_coordination"
                    else hit.similarity_source
                ),
                "style_evidence": facets,
            }
        )
    anchor_category = anchors[0] if anchors and len(set(anchors)) == 1 else None
    status: StyleSupportStatus = (
        "candidates"
        if candidates
        else "unsupported"
        if trace.unsupported_reason is not None
        else "no_result"
    )
    return {
        "support_status": status,
        "style_submode": trace.style_submode or classification.style_submode,
        "anchor_category": anchor_category,
        "requested_target_families": list(classification.requested_target_families),
        "allowed_target_families": list(classification.allowed_target_families),
        "candidates": candidates,
    }


def _multi_payload(
    result: MultiProductResult, *, call_index: int
) -> dict[str, JSONValue]:
    items: list[JSONValue] = []
    candidates: list[dict[str, JSONValue]] = []
    candidate_by_private_product: dict[str, int] = {}
    for item_ordinal, item in enumerate(result.objects[:12], 1):
        item_ref = f"item-{item_ordinal:03d}"
        label = item.label_zh or item.label
        candidate_ordinal: int | None = None
        if item.hits:
            hit = item.hits[0]
            private_key = hit.product.product_id
            candidate_ordinal = candidate_by_private_product.get(private_key)
            if candidate_ordinal is None:
                candidate_ordinal = len(candidates) + 1
                candidate_by_private_product[private_key] = candidate_ordinal
                candidates.append(
                    _product_candidate(
                        hit,
                        call_index=call_index,
                        candidate_ordinal=candidate_ordinal,
                    )
                )
        items.append(
            {
                "item_ref": item_ref,
                "label": label,
                "status": "matched" if candidate_ordinal is not None else "unresolved",
                "candidate_ordinal": candidate_ordinal,
            }
        )
    return {"items": items, "candidates": candidates}


def project_document_ocr_lines_v2(
    raw_validated_output: object, *, call_index: int
) -> dict[str, JSONValue]:
    """Return the single canonical OCR line/handle projection used everywhere."""

    if call_index < 1:
        raise ValueError("call_index must be positive")
    try:
        result = (
            raw_validated_output
            if type(raw_validated_output) is DocumentOCRResult
            else _strict_revalidate_dump(raw_validated_output, DocumentOCRResult)
        )
    except (TypeError, ValidationError, ValueError) as error:
        raise PublicScorerEvidenceIntegrityError(
            "OCR line projection received the wrong validated DTO"
        ) from error
    fields_by_line: dict[str, dict[str, str]] = {
        item.line_id: {} for item in result.lines
    }
    for field in result.fields:
        for line_id in field.evidence_line_ids:
            fields_by_line[line_id].setdefault(field.field_name, field.value)
    lines: list[JSONValue] = []
    for line_ordinal, line in enumerate(result.lines, 1):
        lines.append(
            {
                "line_reference": f"tool-call-{call_index}-line-{line_ordinal}",
                "text": line.text,
                "content_trust": "untrusted_document_text",
                "truncated": bool(result.truncated),
                "fields": [
                    [key, value]
                    for key, value in sorted(fields_by_line[line.line_id].items())
                ],
            }
        )
    return {"lines": lines}


def project_validated_tool_result_v2(
    *,
    query: Query,
    tool_name: str,
    raw_validated_output: object,
    call_index: int,
) -> tuple[ScorerPayloadKindV2, dict[str, JSONValue]]:
    """Project one registry-validated raw DTO into a public scorer payload."""

    if type(query) is not Query:
        raise TypeError("v2 scorer projection requires exactly a Query")
    if call_index < 1:
        raise ValueError("call_index must be positive")
    kind: ScorerPayloadKindV2
    payload: dict[str, JSONValue]
    if tool_name in {"image_product_search", "text_product_search"}:
        try:
            output = (
                raw_validated_output
                if type(raw_validated_output) is ProductSearchTrace
                else _strict_revalidate_dump(raw_validated_output, ProductSearchTrace)
            )
        except (TypeError, ValidationError, ValueError) as error:
            raise PublicScorerEvidenceIntegrityError(
                "product scorer projection received the wrong validated DTO"
            ) from error
        kind = "product_candidates_v1"
        payload = {
            "candidates": [
                _product_candidate(
                    hit, call_index=call_index, candidate_ordinal=ordinal
                )
                for ordinal, hit in enumerate(output.hits[:5], 1)
            ]
        }
    elif tool_name == "style_similar_search":
        try:
            output = (
                raw_validated_output
                if type(raw_validated_output) is ProductSearchTrace
                else _strict_revalidate_dump(raw_validated_output, ProductSearchTrace)
            )
        except (TypeError, ValidationError, ValueError) as error:
            raise PublicScorerEvidenceIntegrityError(
                "Style scorer projection received the wrong validated DTO"
            ) from error
        kind = "style_candidates_v2"
        payload = _style_payload(query, output, call_index=call_index)
    elif tool_name == "multi_product_search":
        try:
            output = (
                raw_validated_output
                if type(raw_validated_output) is MultiProductResult
                else _strict_revalidate_dump(raw_validated_output, MultiProductResult)
            )
        except (TypeError, ValidationError, ValueError) as error:
            raise PublicScorerEvidenceIntegrityError(
                "multi scorer projection received the wrong validated DTO"
            ) from error
        kind = "multi_mapping_v1"
        payload = _multi_payload(output, call_index=call_index)
    elif tool_name in {"encyclopedia_lookup", "recipe_lookup"}:
        try:
            if not isinstance(raw_validated_output, list):
                raise TypeError
            output = [
                item if type(item) is KBHit else _strict_revalidate_dump(item, KBHit)
                for item in raw_validated_output
            ]
        except (TypeError, ValidationError, ValueError) as error:
            raise PublicScorerEvidenceIntegrityError(
                "knowledge scorer projection received the wrong validated DTO"
            ) from error
        kind = "knowledge_sources_v1"
        payload = {
            "sources": [
                {
                    "evidence_reference": f"tool-call-{call_index}-source-{ordinal}",
                    "title": item.title,
                    "text": item.text,
                    "uri": item.citation.source_uri,
                }
                for ordinal, item in enumerate(output[:3], 1)
            ]
        }
    elif tool_name == "object_detect":
        try:
            output = (
                raw_validated_output
                if type(raw_validated_output) is ObjectDetectionResult
                else _strict_revalidate_dump(
                    raw_validated_output, ObjectDetectionResult
                )
            )
        except (TypeError, ValidationError, ValueError) as error:
            raise PublicScorerEvidenceIntegrityError(
                "detection scorer projection received the wrong validated DTO"
            ) from error
        kind = "detections_v1"
        payload = {
            "detections": [
                {
                    "item_ref": f"item-{ordinal:03d}",
                    "label": item.label_zh or item.label,
                    "confidence": float(item.confidence),
                }
                for ordinal, item in enumerate(output.detections[:20], 1)
            ]
        }
    elif tool_name == "document_ocr":
        kind = "ocr_lines_v1"
        payload = project_document_ocr_lines_v2(
            raw_validated_output, call_index=call_index
        )
    else:
        raise PublicScorerEvidenceIntegrityError(
            f"unsupported scorer tool: {tool_name}"
        )
    validate_json_value(payload)
    _reject_private_values(payload, path="payload")
    _PAYLOAD_MODELS[kind].model_validate(payload, strict=True)
    return kind, payload


def _argument_projection(
    query: Query, tool_name: str, bound_arguments: Mapping[str, JSONValue]
) -> dict[str, JSONValue]:
    actual = dict(bound_arguments)
    if tool_name in _ASSET_TOOLS:
        expected: dict[str, JSONValue] = {"asset_id": query.asset_id}
        projection: dict[str, JSONValue] = {"asset_handle": "query_asset"}
    elif tool_name == "style_similar_search":
        expected = {"asset_id": query.asset_id, "query": query.text}
        projection = {"asset_handle": "query_asset", "query": query.text}
    elif tool_name == "text_product_search":
        expected = {"query": query.text}
        projection = {"query": query.text}
    elif tool_name in {"encyclopedia_lookup", "recipe_lookup"}:
        key = "entity" if tool_name == "encyclopedia_lookup" else "dish"
        if set(actual) != {key} or not isinstance(actual[key], str):
            raise PublicScorerEvidenceIntegrityError(
                f"{tool_name} arguments are not exactly bound"
            )
        value = _nonblank(actual[key], key)
        expected = {key: value}
        projection = {key: value}
    else:
        raise PublicScorerEvidenceIntegrityError(
            f"unsupported scorer tool: {tool_name}"
        )
    if actual != expected:
        raise PublicScorerEvidenceIntegrityError(
            f"{tool_name} arguments differ from the authoritative query"
        )
    return projection


def build_public_scorer_call_v2(
    *,
    query: Query,
    call_index: int,
    tool_name: str,
    bound_arguments: Mapping[str, JSONValue],
    arguments_sha256: str,
    raw_validated_output: object,
    result_sha256: str,
) -> PublicScorerCallEvidenceV2:
    """Capture one validated tool result without retaining private raw fields."""

    try:
        if type(query) is not Query:
            raise TypeError("v2 scorer capture requires exactly a Query")
        arguments = dict(bound_arguments)
        if arguments_sha256 != _hash_json(arguments):
            raise PublicScorerEvidenceIntegrityError("tool arguments hash mismatch")
        if result_sha256 != _hash_json(raw_validated_output):
            raise PublicScorerEvidenceIntegrityError(
                "validated tool result hash mismatch"
            )
        projection = _argument_projection(query, tool_name, arguments)
        payload_kind, payload = project_validated_tool_result_v2(
            query=query,
            tool_name=tool_name,
            raw_validated_output=raw_validated_output,
            call_index=call_index,
        )
        return PublicScorerCallEvidenceV2.model_validate(
            {
                "call_index": call_index,
                "tool_name": tool_name,
                "arguments_sha256": arguments_sha256,
                "result_sha256": result_sha256,
                "argument_projection": projection,
                "payload_kind": payload_kind,
                "payload": payload,
                "payload_sha256": _hash_json(payload),
            },
            strict=True,
        )
    except PublicScorerEvidenceIntegrityError:
        raise
    except (TypeError, ValueError, ValidationError) as error:
        raise PublicScorerEvidenceIntegrityError(
            f"invalid v2 public scorer call: {error}"
        ) from error


def expected_multi_items_from_call_v2(
    call: PublicScorerCallEvidenceV2,
) -> tuple[ExpectedMultiItemV2, ...] | None:
    """Return a non-empty frozen object set; None when none can be derived."""

    if call.payload_kind != "multi_mapping_v1":
        return None
    payload = ScorerMultiPayloadV2.model_validate(call.payload, strict=True)
    if not payload.items:
        return None
    return tuple(
        ExpectedMultiItemV2(item_ref=item.item_ref, label=item.label)
        for item in payload.items
    )


def make_public_scorer_evidence_v2(
    *,
    matrix_run_id: str,
    instance_id: str,
    request_sha256: str,
    query_id: str,
    config: AssistantRunConfig,
    query_artifact_sha256: str,
    assistant_result_sha256: str,
    assistant_receipt_sha256: str,
    calls: Sequence[PublicScorerCallEvidenceV2],
    expected_multi_items: Sequence[ExpectedMultiItemV2] | None = None,
) -> PublicScorerEvidenceV2:
    """Finalize one self-authenticating v2 scorer sidecar."""

    unsigned = {
        "schema_version": GCS_SCORER_EVIDENCE_V2_SCHEMA_VERSION,
        "policy_version": GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
        "matrix_run_id": matrix_run_id,
        "instance_id": instance_id,
        "request_sha256": request_sha256,
        "query_id": query_id,
        "config": config,
        "query_artifact_sha256": query_artifact_sha256,
        "assistant_result_sha256": assistant_result_sha256,
        "assistant_receipt_sha256": assistant_receipt_sha256,
        "expected_multi_items": (
            None if expected_multi_items is None else tuple(expected_multi_items)
        ),
        "calls": tuple(calls),
    }
    return PublicScorerEvidenceV2.model_validate(
        {**unsigned, "evidence_sha256": _hash_json(unsigned)}, strict=True
    )


def require_public_scorer_evidence_v2(value: object) -> PublicScorerEvidenceV2:
    """Strictly reparse evidence and expose integrity failures as fatal errors."""

    try:
        if isinstance(value, PublicScorerEvidenceV2):
            raw = canonical_json_bytes(value.model_dump(mode="json"))
        elif isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
        elif isinstance(value, Mapping):
            raw = canonical_json_bytes(_jsonable(value))
        else:
            raise TypeError("v2 scorer evidence has an unsupported representation")
        return PublicScorerEvidenceV2.model_validate_json(raw, strict=True)
    except (TypeError, ValueError, ValidationError) as error:
        raise PublicScorerEvidenceIntegrityError(
            f"invalid v2 public scorer evidence: {error}"
        ) from error


__all__ = [
    "ExpectedMultiItemV2",
    "GCS_SCORER_EVIDENCE_V2_POLICY_VERSION",
    "GCS_SCORER_EVIDENCE_V2_SCHEMA_VERSION",
    "PublicScorerCallEvidenceV2",
    "PublicScorerEvidenceIntegrityError",
    "PublicScorerEvidenceV2",
    "ScorerDetectionPayloadV2",
    "ScorerKnowledgePayloadV2",
    "ScorerMultiPayloadV2",
    "ScorerOCRPayloadV2",
    "ScorerProductPayloadV2",
    "ScorerStyleCandidateV2",
    "ScorerStyleFacetV2",
    "ScorerStylePayloadV2",
    "StyleQueryClassificationV2",
    "build_public_scorer_call_v2",
    "classify_style_query_v2",
    "expected_multi_items_from_call_v2",
    "make_public_scorer_evidence_v2",
    "project_document_ocr_lines_v2",
    "project_validated_tool_result_v2",
    "require_public_scorer_evidence_v2",
]
