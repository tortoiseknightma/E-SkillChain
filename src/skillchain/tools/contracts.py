"""Shared validated values returned by Phase 2 tools."""

import math
from typing import Annotated, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.schemas import Product

type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def validate_json_value(value: object) -> JSONValue:
    """Return a value after verifying it can be encoded as strict JSON."""
    return _validate_json_value(value, set())


def _validate_json_value(value: object, active_container_ids: set[int]) -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, (list, dict)):
        container_id = id(value)
        if container_id in active_container_ids:
            raise ValueError("JSON containers must not contain cycles")
        active_container_ids.add(container_id)
        try:
            if isinstance(value, list):
                for item in value:
                    _validate_json_value(item, active_container_ids)
            else:
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise TypeError("JSON object keys must be strings")
                    _validate_json_value(item, active_container_ids)
        finally:
            active_container_ids.remove(container_id)
        return cast(JSONValue, value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


class RetrievalArtifactBinding(BaseModel):
    """Identity inherited by every product retrieval result."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    mode: Literal["verified", "provisional"]
    index_integrity_sha256: Sha256 | None = None
    eligibility_sha256: Sha256 | None = None
    query_artifact_sha256: Sha256 | None = None
    gallery_artifact_sha256: Sha256 | None = None
    asset_catalog_sha256: Sha256 | None = None
    products_parquet_sha256: Sha256 | None = None
    leakage_policy_version: str | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        bound_values = (
            self.index_integrity_sha256,
            self.eligibility_sha256,
            self.query_artifact_sha256,
            self.gallery_artifact_sha256,
            self.asset_catalog_sha256,
            self.products_parquet_sha256,
            self.leakage_policy_version,
        )
        if self.mode == "verified":
            if any(value is None for value in bound_values):
                raise ValueError(
                    "verified retrieval binding 必须包含完整 artifact 身份"
                )
            if (
                not self.leakage_policy_version
                or not self.leakage_policy_version.strip()
            ):
                raise ValueError("leakage_policy_version 不得为空")
        elif any(value is not None for value in bound_values):
            raise ValueError("provisional retrieval binding 不得伪装正式 artifact 身份")
        return self


class ProductHit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    rank: int = Field(ge=1)
    score: FiniteFloat
    product: Product
    artifact_binding: RetrievalArtifactBinding


class StyleFacetEvidence(BaseModel):
    """One inspectable style fact supporting a returned candidate.

    The evidence value is intentionally semantic (for example a FashionIQ
    relative caption) rather than an opaque score.  Runner projections replace
    the private source identity with a per-call public evidence handle.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    facet: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=512)
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    provenance: Literal[
        "fashioniq_relative_caption",
        "local_image_feature",
        "verified_embedding",
        "portfolio_curated_coordination_rule",
    ]
    source_record_id: str = Field(min_length=1, max_length=256)

    @field_validator("facet", "value", "source_record_id")
    @classmethod
    def validate_trimmed_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("style facet evidence text must be trimmed")
        return value


class StyleHit(ProductHit):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    mmr_score: FiniteFloat
    anchor_category_l1: str = Field(min_length=1)
    style_submode: (
        Literal["same_category_alternative", "cross_category_coordination"] | None
    ) = None
    similarity_source: (
        Literal[
            "fashioniq_relative_caption_graph",
            "local_image_feature_cosine",
            "verified_embedding_cosine",
            "verified_coordination_graph",
        ]
        | None
    ) = None
    facet_evidence: tuple[StyleFacetEvidence, ...] = ()

    @field_validator("facet_evidence", mode="before")
    @classmethod
    def canonicalize_facet_evidence(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_style_evidence(self) -> Self:
        extended = (
            self.style_submode,
            self.similarity_source,
            bool(self.facet_evidence),
        )
        if any(value not in (None, False) for value in extended) and (
            self.style_submode is None
            or self.similarity_source is None
            or not self.facet_evidence
        ):
            raise ValueError(
                "extended style hits require submode, source, and facet evidence"
            )
        if self.style_submode == "same_category_alternative":
            if self.product.category_l1 != self.anchor_category_l1:
                raise ValueError(
                    "same-category style hits must preserve the anchor category"
                )
            if self.similarity_source == "verified_coordination_graph" or any(
                evidence.provenance == "portfolio_curated_coordination_rule"
                for evidence in self.facet_evidence
            ):
                raise ValueError(
                    "same-category style hits cannot use coordination evidence"
                )
        if self.style_submode == "cross_category_coordination":
            if self.product.category_l1 == self.anchor_category_l1:
                raise ValueError(
                    "cross-category style hits must change the product category"
                )
            if self.similarity_source != "verified_coordination_graph" or any(
                evidence.provenance != "portfolio_curated_coordination_rule"
                for evidence in self.facet_evidence
            ):
                raise ValueError(
                    "cross-category style hits require curated coordination evidence"
                )
        return self


class ProductSearchTrace(BaseModel):
    """Formal product-call evidence retained even when retrieval has no hits."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    tool_name: Literal[
        "image_product_search", "text_product_search", "style_similar_search"
    ] = "image_product_search"
    input_kind: Literal["image", "text"] = "image"
    query_input_sha256: Sha256 | None = None
    query_image_sha256: Sha256 | None = None
    query_asset_id: str | None = None
    query_text: str | None = None
    query_vector_sha256: Sha256
    artifact_binding: RetrievalArtifactBinding
    hits: tuple[ProductHit | StyleHit, ...]
    style_submode: (
        Literal["same_category_alternative", "cross_category_coordination"] | None
    ) = None
    unsupported_reason: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="before")
    @classmethod
    def map_legacy_image_hash(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        mapped = dict(value)
        if mapped.get("input_kind", "image") == "image":
            digest = mapped.get("query_input_sha256") or mapped.get(
                "query_image_sha256"
            )
            if digest is not None:
                mapped.setdefault("query_input_sha256", digest)
                mapped.setdefault("query_image_sha256", digest)
        return mapped

    @field_validator("hits", mode="before")
    @classmethod
    def canonicalize_hits(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_input_identity(self) -> Self:
        if self.input_kind == "image":
            if self.query_text is not None:
                raise ValueError("image trace must not contain query_text")
            if self.query_input_sha256 != self.query_image_sha256:
                raise ValueError("image trace hashes must match")
        elif self.query_asset_id is not None or not self.query_text:
            raise ValueError("text trace requires only query_text")
        elif self.query_input_sha256 is None or self.query_image_sha256 is not None:
            raise ValueError("text trace requires only query_input_sha256")
        if self.tool_name == "text_product_search" and self.input_kind != "text":
            raise ValueError("text search trace requires text input")
        if self.tool_name != "text_product_search" and self.input_kind != "image":
            raise ValueError("image/style trace requires image input")
        if self.tool_name != "style_similar_search" and any(
            isinstance(hit, StyleHit) for hit in self.hits
        ):
            raise ValueError("style hits are only valid for style search")
        if self.tool_name != "style_similar_search" and (
            self.style_submode is not None or self.unsupported_reason is not None
        ):
            raise ValueError("style trace metadata is only valid for style search")
        if self.unsupported_reason is not None:
            if self.unsupported_reason != self.unsupported_reason.strip():
                raise ValueError("unsupported style reason must be trimmed")
            if self.style_submode is None or self.hits:
                raise ValueError(
                    "unsupported style traces require a submode and zero hits"
                )
        if self.style_submode is not None and any(
            not isinstance(hit, StyleHit) or hit.style_submode != self.style_submode
            for hit in self.hits
        ):
            raise ValueError("style trace and hit submodes must agree")
        expected_ranks = tuple(range(1, len(self.hits) + 1))
        if tuple(hit.rank for hit in self.hits) != expected_ranks:
            raise ValueError("product trace hit ranks must be contiguous from one")
        if any(hit.artifact_binding != self.artifact_binding for hit in self.hits):
            raise ValueError(
                "every product hit must inherit the trace artifact binding"
            )
        return self


class Detection(BaseModel):
    """Legacy compact DTO; formal detection uses ObjectDetectionResult."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    label: str = Field(min_length=1)
    label_zh: str = Field(min_length=1)
    bbox: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        if self.label != self.label.strip() or self.label_zh != self.label_zh.strip():
            raise ValueError("detection labels must not contain surrounding whitespace")
        return self

    @model_validator(mode="after")
    def validate_bbox_order(self) -> Self:
        x1, y1, x2, y2 = self.bbox
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must be non-negative and have positive area")
        return self


__all__ = [
    "Detection",
    "JSONValue",
    "ProductHit",
    "ProductSearchTrace",
    "RetrievalArtifactBinding",
    "StyleFacetEvidence",
    "StyleHit",
    "validate_json_value",
]
